# This file is part of lsst-images.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# Use of this source code is governed by a 3-clause BSD-style
# license that can be found in the LICENSE file.
"""Turn `pydantic` serialization models into composition diagrams."""

from __future__ import annotations

__all__ = (
    "DEFAULT_LEAF_TYPES",
    "Attribute",
    "Graph",
    "Node",
    "Policy",
    "Reference",
    "build_graph",
    "build_instance_graph",
    "graph_from_file",
    "make_policy",
    "render",
)

import dataclasses
import re
import types
from collections.abc import Iterable, Mapping, Sequence, Set
from typing import TypeAliasType, TypeGuard, Union, get_args, get_origin

import pydantic

from .serialization import open as open_archive

#: Serialization-plumbing helper models collapsed to leaves by default; these
#: carry array/table payloads rather than meaningful model composition.
DEFAULT_LEAF_TYPES: frozenset[str] = frozenset(
    {
        "ArrayReferenceModel",
        "ArrayReferenceQuantityModel",
        "InlineArrayModel",
        "InlineArrayQuantityModel",
        "QuantityModel",
        "TimeModel",
        "TableModel",
        "TableColumnModel",
        # Per-backend array-payload pointers seen when diagramming a file.
        "JsonRef",
        "PointerModel",
        "NdfPointerModel",
    }
)

_SEQUENCE_ORIGINS: frozenset[object] = frozenset({list, set, frozenset, tuple, Sequence, Set})
_MAPPING_ORIGINS: frozenset[object] = frozenset({dict, Mapping})
_UNION_ORIGINS: frozenset[object] = frozenset({Union, types.UnionType})

# Compact, format-safe cardinality markers appended to a field name in edge
# labels: ``*`` a list, ``+`` a mapping, ``?`` optional. (Brackets and braces
# are avoided because they are structural in dot and mermaid.)
_CARDINALITY_MARKER: dict[str, str] = {"one": "", "optional": "?", "list": "*", "dict": "+"}


@dataclasses.dataclass
class Attribute:
    """A scalar field rendered as an attribute inside a model's node."""

    name: str
    type_str: str


@dataclasses.dataclass
class Reference:
    """A model-valued field rendered as a composition edge to other nodes."""

    name: str
    targets: list[str]
    cardinality: str
    has_other: bool = False


@dataclasses.dataclass
class Node:
    """A model class in the diagram."""

    key: str
    label: str
    attributes: list[Attribute]
    references: list[Reference]
    collapsed: bool = False


@dataclasses.dataclass
class Graph:
    """A composition graph: a root model and all models reachable from it."""

    root: str
    nodes: dict[str, Node]


@dataclasses.dataclass(frozen=True)
class Policy:
    """Controls how much of each model the walk records.

    Models whose type name (the unparameterized class name) is in ``leaves``
    are rendered as leaf nodes without expanding their fields.  When
    ``include_attributes`` is false, scalar (non-model) fields are dropped, so
    the diagram shows only model composition and all-scalar models such as
    ``ObservationInfo`` become leaves.
    """

    leaves: frozenset[str] = DEFAULT_LEAF_TYPES
    include_attributes: bool = True
    hide_fields: frozenset[str] = frozenset()
    hide_types: frozenset[str] = frozenset()
    public_names: bool = False


def make_policy(
    expand_leaves: bool = False,
    expand: Iterable[str] = (),
    collapse: Iterable[str] = (),
    include_attributes: bool = True,
    hide_fields: Iterable[str] = (),
    hide_types: Iterable[str] = (),
    public_names: bool = False,
) -> Policy:
    """Build a `Policy` from the default leaf set and CLI-style overrides.

    Parameters
    ----------
    expand_leaves
        Start from an empty leaf set instead of `DEFAULT_LEAF_TYPES`.
    expand
        Type names to remove from the leaf set (force expansion).
    collapse
        Type names to add to the leaf set (force collapse).
    include_attributes
        Record scalar (non-model) fields as node attributes.  When false the
        diagram shows only model composition.
    hide_fields
        Field names to drop entirely, along with any sub-tree reachable only
        through them.
    hide_types
        Type names (public or serialization) to drop entirely, removing every
        edge that points at them.
    public_names
        Label nodes with the public in-memory class name (e.g.
        ``SkyProjection``) instead of the serialization model name, where one
        is available.
    """
    leaves = set() if expand_leaves else set(DEFAULT_LEAF_TYPES)
    leaves -= set(expand)
    leaves |= set(collapse)
    return Policy(
        leaves=frozenset(leaves),
        include_attributes=include_attributes,
        hide_fields=frozenset(hide_fields),
        hide_types=frozenset(hide_types),
        public_names=public_names,
    )


def _key(cls: type) -> str:
    """Return a stable unique key for ``cls``."""
    return f"{cls.__module__}.{cls.__qualname__}"


def _label(cls: type) -> str:
    """Return the display label for ``cls`` (with any generic parameters)."""
    return cls.__name__


def _type_name(cls: type) -> str:
    """Return the unparameterized class name used for policy matching."""
    metadata = getattr(cls, "__pydantic_generic_metadata__", None)
    if metadata and metadata.get("origin") is not None:
        return metadata["origin"].__name__
    return cls.__name__


def _public_name(cls: type) -> str | None:
    """Return the public in-memory class name for ``cls``, if any.

    Serialization models expose a ``PUBLIC_TYPE`` ClassVar naming the class
    their data deserializes to.  Builtins (e.g. ``dict``) are ignored so the
    model name is kept instead.
    """
    public = getattr(cls, "PUBLIC_TYPE", None)
    if isinstance(public, type) and public.__module__ != "builtins":
        return public.__name__
    return None


def _node_label(cls: type, policy: Policy) -> str:
    """Return the node label, preferring the public class name if requested."""
    if policy.public_names and (public := _public_name(cls)) is not None:
        return public
    return _label(cls)


def _type_names(cls: type) -> set[str]:
    """Return the names that match a leaf/collapse directive for ``cls``.

    Both the serialization model name and the public class name match, so
    ``--collapse Image`` works whether or not public names are displayed.
    """
    names = {_type_name(cls)}
    if (public := _public_name(cls)) is not None:
        names.add(public)
    return names


def _is_hidden_type(cls: type, policy: Policy) -> bool:
    """Return whether ``cls`` should be dropped from the diagram entirely."""
    return bool(_type_names(cls) & policy.hide_types)


def _resolve(annotation: object) -> object:
    """Unwrap ``TypeAliasType`` and ``Annotated`` layers to the real type.

    PEP 695 aliases (e.g. ``FieldSerializationModel``) and ``Annotated``
    wrappers (discriminated unions, validators) hide the underlying union or
    model from naive introspection.
    """
    while True:
        if isinstance(annotation, TypeAliasType):
            annotation = annotation.__value__
        elif hasattr(annotation, "__metadata__"):
            annotation = getattr(annotation, "__origin__")
        else:
            return annotation


def _strip_annotated(annotation: object) -> object:
    """Unwrap ``Annotated`` layers while preserving type aliases."""
    while hasattr(annotation, "__metadata__"):
        annotation = getattr(annotation, "__origin__")
    return annotation


def _is_model(annotation: object) -> TypeGuard[type[pydantic.BaseModel]]:
    """Return whether ``annotation`` is a `pydantic.BaseModel` subclass."""
    return isinstance(annotation, type) and issubclass(annotation, pydantic.BaseModel)


def _type_str(annotation: object) -> str:
    """Return a readable, module-stripped string for a scalar annotation."""
    # Use an alias's own name rather than expanding it: aliases such as the
    # JSON-like ``MetadataValue`` are recursive and would not terminate.
    if isinstance(annotation, TypeAliasType):
        return annotation.__name__
    if hasattr(annotation, "__metadata__"):
        return _type_str(getattr(annotation, "__origin__"))
    if annotation is type(None):
        return "None"
    if annotation is Ellipsis:
        return "..."
    origin = get_origin(annotation)
    if origin in _UNION_ORIGINS:
        return " | ".join(_type_str(arg) for arg in get_args(annotation))
    if origin is not None:
        args = get_args(annotation)
        origin_name = _type_str(origin)
        if args:
            return f"{origin_name}[{', '.join(_type_str(arg) for arg in args)}]"
        return origin_name
    name = getattr(annotation, "__name__", None)
    if isinstance(name, str):
        return name
    return str(annotation).replace("typing.", "")


def _models_in(
    args: tuple[object, ...], seen_aliases: frozenset[int] = frozenset()
) -> tuple[list[type], bool]:
    """Split union members into model types and a "something else" flag.

    Nested unions and containers are flattened; `None` is ignored (the
    optionality is captured by the caller); anything else sets the "other"
    flag.
    """
    models: list[type] = []
    has_other = False
    for arg in args:
        arg = _strip_annotated(arg)
        if isinstance(arg, TypeAliasType):
            alias_id = id(arg)
            if alias_id in seen_aliases:
                has_other = True
                continue
            sub_models, sub_other = _models_in((arg.__value__,), seen_aliases | {alias_id})
            models.extend(sub_models)
            has_other = has_other or sub_other
            continue
        if arg is type(None):
            continue
        if _is_model(arg):
            models.append(arg)
        elif get_origin(arg) in _UNION_ORIGINS:
            sub_models, sub_other = _models_in(get_args(arg), seen_aliases)
            models.extend(sub_models)
            has_other = has_other or sub_other
        elif get_origin(arg) in _SEQUENCE_ORIGINS:
            sub_models, sub_other = _models_in(get_args(arg), seen_aliases)
            models.extend(sub_models)
            has_other = has_other or sub_other
        elif get_origin(arg) in _MAPPING_ORIGINS:
            mapping_args = get_args(arg)
            value = (mapping_args[1],) if len(mapping_args) == 2 else ()
            sub_models, sub_other = _models_in(value, seen_aliases)
            models.extend(sub_models)
            has_other = has_other or sub_other
        else:
            has_other = True
    # Preserve order while removing duplicate model types.
    return list(dict.fromkeys(models)), has_other


def _classify(annotation: object) -> tuple[list[type], str, bool] | None:
    """Classify a field annotation as a model reference or a scalar.

    Returns ``(model_types, cardinality, has_other)`` for a field that
    references one or more models, or `None` for a pure scalar field.
    """
    annotation = _resolve(annotation)
    origin = get_origin(annotation)
    if origin in _UNION_ORIGINS:
        args = get_args(annotation)
        models, has_other = _models_in(args)
        if not models:
            return None
        cardinality = "optional" if type(None) in args else "one"
        return models, cardinality, has_other
    if origin in _SEQUENCE_ORIGINS:
        models, has_other = _models_in(get_args(annotation))
        if not models:
            return None
        return models, "list", has_other
    if origin in _MAPPING_ORIGINS:
        args = get_args(annotation)
        value = (args[1],) if len(args) == 2 else ()
        models, has_other = _models_in(value)
        if not models:
            return None
        return models, "dict", has_other
    if _is_model(annotation):
        return [annotation], "one", False
    return None


def build_graph(model_cls: type[pydantic.BaseModel], policy: Policy | None = None) -> Graph:
    """Walk ``model_cls`` and its sub-models into a composition `Graph`."""
    if policy is None:
        policy = Policy()
    nodes: dict[str, Node] = {}
    root = _walk(model_cls, nodes, policy)
    return Graph(root=root, nodes=nodes)


def _walk(cls: type[pydantic.BaseModel], nodes: dict[str, Node], policy: Policy) -> str:
    """Add ``cls`` (and its sub-models) to ``nodes``; return its key."""
    key = _key(cls)
    if key in nodes:
        return key
    node = Node(key=key, label=_node_label(cls, policy), attributes=[], references=[])
    nodes[key] = node
    if _type_names(cls) & policy.leaves:
        node.collapsed = True
        return key
    for name, field in cls.model_fields.items():
        if name in policy.hide_fields:
            continue
        annotation = field.annotation
        classified = _classify(annotation)
        if classified is None:
            if policy.include_attributes:
                node.attributes.append(Attribute(name=name, type_str=_type_str(annotation)))
            continue
        model_types, cardinality, has_other = classified
        model_types = [model for model in model_types if not _is_hidden_type(model, policy)]
        if not model_types and not has_other:
            continue
        targets = [_walk(model, nodes, policy) for model in model_types]
        node.references.append(
            Reference(name=name, targets=targets, cardinality=cardinality, has_other=has_other)
        )
    return key


def build_instance_graph(instance: pydantic.BaseModel, policy: Policy | None = None) -> Graph:
    """Walk a model *instance* into a composition `Graph`.

    Unlike `build_graph`, which works from field annotations, this follows the
    actual values, so unions collapse to the concrete member present and
    lists/dicts expand only the element types that actually occur.
    """
    if policy is None:
        policy = Policy()
    nodes: dict[str, Node] = {}
    root = _walk_instance(instance, nodes, policy, visited=set())
    return Graph(root=root, nodes=nodes)


def graph_from_file(path: str, policy: Policy | None = None) -> Graph:
    """Build an instance `Graph` from a serialized ``lsst.images`` file.

    Reads only the on-disk reference tree (pointers, not pixel data), so this
    is cheap even for large images.
    """
    with open_archive(path) as reader:
        return build_instance_graph(reader.get_tree(), policy)


def _walk_instance(
    instance: pydantic.BaseModel, nodes: dict[str, Node], policy: Policy, visited: set[int]
) -> str:
    """Add ``instance`` (and the models it holds) to ``nodes``; return key."""
    cls = type(instance)
    key = _key(cls)
    if id(instance) in visited:
        return key
    visited.add(id(instance))
    if (node := nodes.get(key)) is None:
        node = Node(key=key, label=_node_label(cls, policy), attributes=[], references=[])
        nodes[key] = node
    if _type_names(cls) & policy.leaves:
        node.collapsed = True
        return key
    for name in type(instance).model_fields:
        if name in policy.hide_fields:
            continue
        value = getattr(instance, name)
        _add_instance_field(node, name, value, nodes, policy, visited)
    return key


def _merge_reference(node: Node, reference: Reference) -> None:
    """Merge a newly observed instance reference into ``node``."""
    node.attributes = [attribute for attribute in node.attributes if attribute.name != reference.name]
    for existing in node.references:
        if existing.name == reference.name and existing.cardinality == reference.cardinality:
            for target in reference.targets:
                if target not in existing.targets:
                    existing.targets.append(target)
            existing.has_other = existing.has_other or reference.has_other
            return
    node.references.append(reference)


def _merge_attribute(node: Node, attribute: Attribute) -> None:
    """Merge a newly observed scalar instance attribute into ``node``."""
    if any(reference.name == attribute.name for reference in node.references):
        return
    for existing in node.attributes:
        if existing.name == attribute.name:
            if attribute.type_str not in existing.type_str.split(" | "):
                existing.type_str += f" | {attribute.type_str}"
            return
    node.attributes.append(attribute)


def _add_instance_field(
    node: Node,
    name: str,
    value: object,
    nodes: dict[str, Node],
    policy: Policy,
    visited: set[int],
) -> None:
    """Classify a field *value* as a concrete-model reference or scalar.

    Instance mode reports only what the file actually holds: an empty container
    yields no edge, not the declared element types.
    """
    if isinstance(value, pydantic.BaseModel):
        if _is_hidden_type(type(value), policy):
            return
        target = _walk_instance(value, nodes, policy, visited)
        _merge_reference(node, Reference(name=name, targets=[target], cardinality="one"))
        return
    if isinstance(value, Mapping):
        models = [v for v in value.values() if isinstance(v, pydantic.BaseModel)]
        cardinality = "dict"
    elif isinstance(value, (list, tuple, set, frozenset)):
        models = [v for v in value if isinstance(v, pydantic.BaseModel)]
        cardinality = "list"
    else:
        models = []
        cardinality = ""
    models = [model for model in models if not _is_hidden_type(type(model), policy)]
    if models:
        targets: list[str] = []
        for model in models:
            target = _walk_instance(model, nodes, policy, visited)
            if target not in targets:
                targets.append(target)
        _merge_reference(node, Reference(name=name, targets=targets, cardinality=cardinality))
    elif not policy.include_attributes:
        return
    elif value is None:
        _merge_attribute(node, Attribute(name=name, type_str="None"))
    else:
        _merge_attribute(node, Attribute(name=name, type_str=type(value).__name__))


def render(graph: Graph, fmt: str) -> str:
    """Render ``graph`` as ``fmt`` text (``dot``, ``mermaid`` or ``tree``)."""
    emitters = {"dot": _to_dot, "mermaid": _to_mermaid, "tree": _to_tree}
    try:
        emitter = emitters[fmt]
    except KeyError:
        choices = ", ".join(sorted(emitters))
        raise ValueError(f"Unknown diagram format {fmt!r}; choose from {choices}.") from None
    return emitter(graph)


def _edge_label(reference: Reference) -> str:
    """Return the field-name edge label with cardinality/other markers."""
    label = reference.name + _CARDINALITY_MARKER[reference.cardinality]
    if reference.has_other:
        label += " (+other)"
    return label


def _node_ids(graph: Graph) -> dict[str, str]:
    """Map node keys to dot/mermaid-safe identifiers (alphanumeric + ``_``)."""
    ids: dict[str, str] = {}
    used: set[str] = set()
    for key, node in graph.nodes.items():
        base = re.sub(r"\W", "_", node.label) or "node"
        if base[0].isdigit():
            base = "_" + base
        candidate = base
        suffix = 1
        while candidate in used:
            suffix += 1
            candidate = f"{base}_{suffix}"
        used.add(candidate)
        ids[key] = candidate
    return ids


def _dot_escape(text: str) -> str:
    """Escape characters that are structural in a dot record label."""
    return "".join("\\" + ch if ch in '\\{}|<>"' else ch for ch in text)


def _to_dot(graph: Graph) -> str:
    ids = _node_ids(graph)
    lines = [
        f'digraph "{_dot_escape(graph.nodes[graph.root].label)}" {{',
        "  rankdir=LR;",
        "  node [shape=record];",
    ]
    for key, node in graph.nodes.items():
        attrs = "".join(f"{_dot_escape(a.name)} : {_dot_escape(a.type_str)}\\l" for a in node.attributes)
        body = "{" + _dot_escape(node.label) + ("|" + attrs if attrs else "") + "}"
        lines.append(f'  {ids[key]} [label="{body}"];')
    for key, node in graph.nodes.items():
        for reference in node.references:
            label = _dot_escape(_edge_label(reference))
            for target in reference.targets:
                lines.append(f'  {ids[key]} -> {ids[target]} [label="{label}"];')
    lines.append("}")
    return "\n".join(lines) + "\n"


def _mermaid_escape(text: str) -> str:
    """Make ``text`` safe for mermaid labels and class-member lines.

    Brackets and braces are structural in mermaid; square brackets become the
    ``~`` generic markers and curly braces become parentheses so generic and
    container types survive without breaking the parser.
    """
    return text.replace('"', "'").replace("[", "~").replace("]", "~").replace("{", "(").replace("}", ")")


def _to_mermaid(graph: Graph) -> str:
    ids = _node_ids(graph)
    lines = ["classDiagram"]
    for key, node in graph.nodes.items():
        lines.append(f'  class {ids[key]}["{_mermaid_escape(node.label)}"] {{')
        for attribute in node.attributes:
            lines.append(f"    +{_mermaid_escape(attribute.type_str)} {_mermaid_escape(attribute.name)}")
        lines.append("  }")
    for key, node in graph.nodes.items():
        for reference in node.references:
            label = _mermaid_escape(_edge_label(reference))
            for target in reference.targets:
                lines.append(f"  {ids[key]} --> {ids[target]} : {label}")
    return "\n".join(lines) + "\n"


def _to_tree(graph: Graph) -> str:
    lines = [graph.nodes[graph.root].label]
    _tree_children(graph, graph.root, prefix="", path=frozenset({graph.root}), lines=lines)
    return "\n".join(lines) + "\n"


def _tree_children(graph: Graph, key: str, prefix: str, path: frozenset[str], lines: list[str]) -> None:
    """Append the tree(1)-style child lines of ``key`` to ``lines``."""
    node = graph.nodes[key]
    items: list[tuple[str, object]] = [("attribute", a) for a in node.attributes]
    items += [("reference", r) for r in node.references]
    for index, (kind, obj) in enumerate(items):
        last = index == len(items) - 1
        branch = "└── " if last else "├── "
        child_prefix = prefix + ("    " if last else "│   ")
        if kind == "attribute":
            assert isinstance(obj, Attribute)
            lines.append(f"{prefix}{branch}{obj.name}: {obj.type_str}")
            continue
        assert isinstance(obj, Reference)
        marker = _CARDINALITY_MARKER[obj.cardinality]
        if len(obj.targets) == 1 and not obj.has_other:
            _tree_single(graph, obj, marker, prefix, branch, child_prefix, path, lines)
        else:
            _tree_union(graph, obj, marker, prefix, branch, child_prefix, path, lines)


def _tree_single(
    graph: Graph,
    reference: Reference,
    marker: str,
    prefix: str,
    branch: str,
    child_prefix: str,
    path: frozenset[str],
    lines: list[str],
) -> None:
    target = reference.targets[0]
    label = graph.nodes[target].label
    if target in path:
        lines.append(f"{prefix}{branch}{reference.name}{marker}: {label} (↻)")
        return
    lines.append(f"{prefix}{branch}{reference.name}{marker}: {label}")
    _tree_children(graph, target, child_prefix, path | {target}, lines)


def _tree_union(
    graph: Graph,
    reference: Reference,
    marker: str,
    prefix: str,
    branch: str,
    child_prefix: str,
    path: frozenset[str],
    lines: list[str],
) -> None:
    lines.append(f"{prefix}{branch}{reference.name}{marker} (one of):")
    total = len(reference.targets) + (1 if reference.has_other else 0)
    for index, target in enumerate(reference.targets):
        member_last = index == total - 1
        member_branch = "└── " if member_last else "├── "
        member_prefix = child_prefix + ("    " if member_last else "│   ")
        label = graph.nodes[target].label
        if target in path:
            lines.append(f"{child_prefix}{member_branch}{label} (↻)")
        else:
            lines.append(f"{child_prefix}{member_branch}{label}")
            _tree_children(graph, target, member_prefix, path | {target}, lines)
    if reference.has_other:
        lines.append(f"{child_prefix}└── …(other)")
