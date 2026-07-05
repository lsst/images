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
from __future__ import annotations

import importlib.metadata
import os.path
from pathlib import Path
from typing import Annotated, Any
from unittest import mock

import pydantic
import pytest
from click.testing import CliRunner

from lsst.images import VisitImageSerializationModel
from lsst.images.cells import CellCoaddSerializationModel
from lsst.images.cli import main
from lsst.images.diagram import (
    DEFAULT_LEAF_TYPES,
    Policy,
    build_graph,
    build_instance_graph,
    graph_from_file,
    make_policy,
    render,
)
from lsst.images.serialization import JsonRef

TESTDIR = os.path.abspath(os.path.dirname(__file__))


class Child(pydantic.BaseModel):
    """A leaf model with a single scalar field."""

    x: int


class Other(pydantic.BaseModel):
    """A second leaf model, used for union members."""

    y: int


class Parent(pydantic.BaseModel):
    """A model with a scalar field and a nested model field."""

    name: str
    child: Child


class Containers(pydantic.BaseModel):
    """A model exercising optional, list, dict, and union fields."""

    maybe: Child | None
    many: list[Child]
    mapping: dict[str, Child]
    either: Child | Other
    open_union: Child | Any


class OptionalContainers(pydantic.BaseModel):
    """A model with optional containers of nested models."""

    maybe_many: list[Child] | None
    maybe_mapping: dict[str, Child] | None


class TreeNode(pydantic.BaseModel):
    """A self-referential model, used to test the cycle guard."""

    children: list[TreeNode] = []


type AliasUnion = Child | Other


class AliasHolder(pydantic.BaseModel):
    """A model whose field is a PEP 695 type alias union."""

    thing: AliasUnion


class Scalars(pydantic.BaseModel):
    """A model with only scalar fields, used to test type strings."""

    mapping: dict[str, int]
    sequence: list[str]
    optional: int | None
    plain: str
    annotated: Annotated[int, "metadata-with-unstable-repr"]
    annotated_optional: Annotated[float, object()] | None


class Leaf(pydantic.BaseModel):
    """The deepest model in the collapse-policy fixture."""

    v: int


class Middle(pydantic.BaseModel):
    """The middle model in the collapse-policy fixture."""

    leaf: Leaf


class Top(pydantic.BaseModel):
    """The root model in the collapse-policy fixture."""

    middle: Middle


class Dog(pydantic.BaseModel):
    """A concrete union member for the instance-mode fixture."""

    bark: str


class Cat(pydantic.BaseModel):
    """A second concrete union member for the instance-mode fixture."""

    meow: str


class Owner(pydantic.BaseModel):
    """A model with union, list-of-union, and optional fields."""

    pet: Dog | Cat
    pets: list[Dog | Cat]
    missing: Dog | None


class Kennel(pydantic.BaseModel):
    """A model with a dict of union models, for empty-container fallback."""

    occupants: dict[str, Dog | Cat]


class RepeatedWrapper(pydantic.BaseModel):
    """A repeated model whose nested union can vary by instance."""

    thing: Dog | Cat


class WrapperCollection(pydantic.BaseModel):
    """A model that contains multiple instances of the same wrapper type."""

    wrappers: list[RepeatedWrapper]


def _invoke(runner: CliRunner, *args: str):
    """Return the Click result of invoking the diagram subcommand with args."""
    return runner.invoke(main, ["diagram", *args])


def test_scalar_and_nested_model() -> None:
    """Test that a scalar field is an attribute and a nested model is an
    edge.
    """
    graph = build_graph(Parent)

    root = graph.nodes[graph.root]
    assert root.label == "Parent"

    attrs = {a.name: a.type_str for a in root.attributes}
    assert "name" in attrs
    assert "str" in attrs["name"]
    assert "child" not in attrs

    refs = {r.name: r for r in root.references}
    assert "child" in refs
    assert refs["child"].cardinality == "one"
    (child_key,) = refs["child"].targets
    assert graph.nodes[child_key].label == "Child"


def test_container_and_union_cardinalities() -> None:
    """Test that optional, list, dict, and union fields map to the correct
    cardinalities.
    """
    graph = build_graph(Containers)
    refs = {r.name: r for r in graph.nodes[graph.root].references}

    assert not graph.nodes[graph.root].attributes

    assert refs["maybe"].cardinality == "optional"
    assert {graph.nodes[t].label for t in refs["maybe"].targets} == {"Child"}

    assert refs["many"].cardinality == "list"
    assert {graph.nodes[t].label for t in refs["many"].targets} == {"Child"}

    assert refs["mapping"].cardinality == "dict"
    assert {graph.nodes[t].label for t in refs["mapping"].targets} == {"Child"}

    assert {graph.nodes[t].label for t in refs["either"].targets} == {"Child", "Other"}
    assert not refs["either"].has_other

    assert {graph.nodes[t].label for t in refs["open_union"].targets} == {"Child"}
    assert refs["open_union"].has_other


def test_optional_containers_are_model_references() -> None:
    """Test that optional list and dict container fields produce edges, not
    scalar attributes.
    """
    graph = build_graph(OptionalContainers)
    root = graph.nodes[graph.root]
    refs = {r.name: r for r in root.references}
    attrs = {a.name for a in root.attributes}

    assert "maybe_many" in refs
    assert {graph.nodes[t].label for t in refs["maybe_many"].targets} == {"Child"}
    assert "maybe_many" not in attrs

    assert "maybe_mapping" in refs
    assert {graph.nodes[t].label for t in refs["maybe_mapping"].targets} == {"Child"}
    assert "maybe_mapping" not in attrs


def test_scalar_type_strings_are_readable() -> None:
    """Test that scalar field type strings are human-readable and Annotated
    metadata is stripped.
    """
    graph = build_graph(Scalars)
    attrs = {a.name: a.type_str for a in graph.nodes[graph.root].attributes}
    assert attrs["mapping"] == "dict[str, int]"
    assert attrs["sequence"] == "list[str]"
    assert attrs["optional"] == "int | None"
    assert attrs["plain"] == "str"
    assert attrs["annotated"] == "int"
    assert attrs["annotated_optional"] == "float | None"


def test_type_alias_union_is_resolved() -> None:
    """Test that a type alias union expands to its member models."""
    graph = build_graph(AliasHolder)
    (ref,) = graph.nodes[graph.root].references
    assert ref.name == "thing"
    assert {graph.nodes[t].label for t in ref.targets} == {"Child", "Other"}


def test_self_referential_model_does_not_loop() -> None:
    """Test that a self-referential model produces a single self-loop node, not
    infinite recursion.
    """
    graph = build_graph(TreeNode)
    assert len(graph.nodes) == 1
    (ref,) = graph.nodes[graph.root].references
    assert ref.targets == [graph.root]
    assert ref.cardinality == "list"


def test_visit_image_real_model() -> None:
    """Test that build_graph handles the real VisitImageSerializationModel
    correctly.
    """
    graph = build_graph(VisitImageSerializationModel[JsonRef])
    root = graph.nodes[graph.root]
    refs = {r.name: r for r in root.references}

    for expected in ("image", "mask", "variance", "sky_projection", "psf"):
        assert expected in refs

    psf_labels = {graph.nodes[t].label for t in refs["psf"].targets}
    assert len(psf_labels) >= 3
    assert refs["psf"].has_other

    assert "photometric_scaling" in refs
    field_labels = {graph.nodes[t].label for t in refs["photometric_scaling"].targets}
    assert "SumFieldSerializationModel" in field_labels


def test_cell_coadd_real_model() -> None:
    """Test that build_graph handles the real CellCoaddSerializationModel
    correctly.
    """
    graph = build_graph(CellCoaddSerializationModel[JsonRef])
    refs = {r.name: r for r in graph.nodes[graph.root].references}

    assert refs["noise_realizations"].cardinality == "list"
    assert refs["mask_fractions"].cardinality == "dict"


def test_collapsed_type_is_a_leaf() -> None:
    """Test that a type listed as a leaf in the policy is present but not
    expanded.
    """
    graph = build_graph(Top, policy=Policy(leaves=frozenset({"Middle"})))
    labels = {n.label for n in graph.nodes.values()}
    assert labels == {"Top", "Middle"}
    middle = next(n for n in graph.nodes.values() if n.label == "Middle")
    assert middle.collapsed
    assert not middle.references
    assert not middle.attributes


def test_make_policy_expand_and_collapse() -> None:
    """Test that make_policy correctly populates expand and collapse
    entries.
    """
    policy = make_policy(expand=["ArrayReferenceModel"], collapse=["Middle"])
    assert "ArrayReferenceModel" not in policy.leaves
    assert "Middle" in policy.leaves
    assert "TableModel" in policy.leaves


def test_make_policy_expand_leaves_clears_defaults() -> None:
    """Test that make_policy with expand_leaves=True produces an empty leaves
    set.
    """
    assert make_policy(expand_leaves=True).leaves == frozenset()


def test_include_attributes_false_elides_scalars() -> None:
    """Test that include_attributes=False drops scalar fields but keeps model
    edges.
    """
    graph = build_graph(Parent, policy=make_policy(include_attributes=False))
    root = graph.nodes[graph.root]
    assert not root.attributes
    assert [r.name for r in root.references] == ["child"]


def test_all_scalar_model_becomes_leaf_without_attributes() -> None:
    """Test that a model with only scalar fields has no attributes or
    references when include_attributes=False.
    """
    graph = build_graph(Scalars, policy=make_policy(include_attributes=False))
    root = graph.nodes[graph.root]
    assert not root.attributes
    assert not root.references


def test_instance_attributes_can_be_elided() -> None:
    """Test that include_attributes=False removes None attributes from
    instance graphs.
    """
    owner = Owner(pet=Dog(bark="woof"), pets=[Dog(bark="a")], missing=None)
    graph = build_instance_graph(owner, policy=make_policy(include_attributes=False))
    assert not graph.nodes[graph.root].attributes


def test_public_names_relabel_real_model() -> None:
    """Test that public_names=True replaces serialization model names with
    their PUBLIC_TYPE labels.
    """
    graph = build_graph(
        VisitImageSerializationModel[JsonRef],
        policy=make_policy(public_names=True),
    )
    labels = {n.label for n in graph.nodes.values()}
    assert "VisitImage" in labels
    assert "SkyProjection" in labels
    assert "VisitImageSerializationModel[JsonRef]" not in labels
    assert "ApertureCorrectionMapSerializationModel" in labels


def test_public_names_collapse_matches_either_name() -> None:
    """Test that a collapse rule matches a model regardless of whether public
    or serialization name is used.
    """
    graph = build_graph(
        VisitImageSerializationModel[JsonRef],
        policy=make_policy(public_names=True, collapse=["Image"]),
    )
    image = next(n for n in graph.nodes.values() if n.label == "Image")
    assert image.collapsed


def test_hide_fields_drops_named_fields() -> None:
    """Test that hide_fields removes the named edge and any type only
    reachable through it.
    """
    graph = build_graph(Parent, policy=make_policy(hide_fields=["child"]))
    assert not graph.nodes[graph.root].references
    assert "Child" not in {n.label for n in graph.nodes.values()}


def test_hide_type_removes_nodes_and_edges() -> None:
    """Test that hide_types removes the named type node and any edge pointing
    to it.
    """
    graph = build_graph(Parent, policy=make_policy(hide_types=["Child"]))
    assert "Child" not in {n.label for n in graph.nodes.values()}
    assert not graph.nodes[graph.root].references


def test_hide_type_matches_public_name() -> None:
    """Test that hide_types matches a node by its public name when
    public_names=True.
    """
    model = VisitImageSerializationModel[JsonRef]
    graph = build_graph(model, policy=make_policy(public_names=True, hide_types=["Image"]))
    assert "Image" not in {n.label for n in graph.nodes.values()}


def test_default_collapses_asdf_helpers_in_real_model() -> None:
    """Test that DEFAULT_LEAF_TYPES are collapsed by default and expanding them
    reveals more nodes.
    """
    assert "ArrayReferenceModel" in DEFAULT_LEAF_TYPES
    model = VisitImageSerializationModel[JsonRef]
    default_graph = build_graph(model)
    array_node = next(n for n in default_graph.nodes.values() if n.label == "ArrayReferenceModel")
    assert array_node.collapsed
    expanded_graph = build_graph(model, policy=make_policy(expand_leaves=True))
    expanded_array = next(n for n in expanded_graph.nodes.values() if n.label == "ArrayReferenceModel")
    assert not expanded_array.collapsed
    assert len(expanded_graph.nodes) > len(default_graph.nodes)


def test_union_collapses_to_concrete_member() -> None:
    """Test that instance mode shows only the concrete union member present in
    each field.
    """
    owner = Owner(pet=Dog(bark="woof"), pets=[Dog(bark="a"), Cat(meow="m")], missing=None)
    graph = build_instance_graph(owner)
    refs = {r.name: r for r in graph.nodes[graph.root].references}

    assert {graph.nodes[t].label for t in refs["pet"].targets} == {"Dog"}
    assert refs["pet"].cardinality == "one"

    assert {graph.nodes[t].label for t in refs["pets"].targets} == {"Dog", "Cat"}
    assert refs["pets"].cardinality == "list"

    assert "missing" not in refs
    attrs = {a.name: a.type_str for a in graph.nodes[graph.root].attributes}
    assert attrs["missing"] == "None"


def test_populated_container_shows_only_present_types() -> None:
    """Test that instance mode shows only the concrete types present in a
    populated container.
    """
    graph = build_instance_graph(Kennel(occupants={"x": Dog(bark="woof")}))
    ref = {r.name: r for r in graph.nodes[graph.root].references}["occupants"]
    assert {graph.nodes[t].label for t in ref.targets} == {"Dog"}


def test_repeated_instances_merge_nested_concrete_types() -> None:
    """Test that instance mode merges concrete types across multiple instances
    of the same wrapper.
    """
    root = WrapperCollection(
        wrappers=[
            RepeatedWrapper(thing=Dog(bark="woof")),
            RepeatedWrapper(thing=Cat(meow="m")),
        ]
    )
    graph = build_instance_graph(root)
    wrapper = next(n for n in graph.nodes.values() if n.label == "RepeatedWrapper")
    refs = {r.name: r for r in wrapper.references}

    assert "thing" in refs
    assert {graph.nodes[t].label for t in refs["thing"].targets} == {"Dog", "Cat"}


def test_hide_type_instance_mode() -> None:
    """Test that hide_types removes the named type from an instance graph."""
    graph = build_instance_graph(
        Kennel(occupants={"x": Dog(bark="woof")}),
        policy=make_policy(hide_types=["Dog"], include_attributes=False),
    )
    assert not graph.nodes[graph.root].references
    assert "Dog" not in {n.label for n in graph.nodes.values()}


def test_empty_container_yields_no_edge() -> None:
    """Test that instance mode does not expand an empty container into its
    declared element types.
    """
    graph = build_instance_graph(Kennel(occupants={}), policy=make_policy(include_attributes=False))
    assert not graph.nodes[graph.root].references
    assert not graph.nodes[graph.root].attributes


def test_graph_from_file_collapses_psf() -> None:
    """Test that graph_from_file collapses a union PSF field to the concrete
    type stored in the file.
    """
    path = os.path.join(TESTDIR, "data", "schema_v1", "visit_image.json")
    graph = graph_from_file(path)
    assert graph.nodes[graph.root].label.startswith("VisitImageSerializationModel")
    refs = {r.name: r for r in graph.nodes[graph.root].references}
    psf_labels = {graph.nodes[t].label for t in refs["psf"].targets}
    assert psf_labels == {"GaussianPSFSerializationModel"}
    assert not refs["psf"].has_other


def test_dot_format() -> None:
    """Test that the dot emitter produces digraph output containing expected
    labels.
    """
    graph = build_graph(Parent)
    dot = render(graph, "dot")
    assert dot.startswith("digraph")
    assert "Parent" in dot
    assert "Child" in dot
    assert "name" in dot
    assert "->" in dot
    assert "child" in dot


def test_mermaid_format() -> None:
    """Test that the mermaid emitter produces a classDiagram block with
    expected labels and arrows.
    """
    graph = build_graph(Parent)
    mermaid = render(graph, "mermaid")
    assert mermaid.lstrip().startswith("classDiagram")
    assert "Parent" in mermaid
    assert "Child" in mermaid
    assert "name" in mermaid
    assert "-->" in mermaid
    assert "child" in mermaid


def test_tree_format() -> None:
    """Test that the tree emitter produces a tree-style text representation
    with branch glyphs.
    """
    graph = build_graph(Parent)
    tree = render(graph, "tree")
    lines = tree.splitlines()
    assert lines[0] == "Parent"
    assert any("name" in line for line in lines)
    assert any("child" in line and "Child" in line for line in lines)
    assert any(line.startswith(("├──", "└──")) for line in lines)


def test_mermaid_real_model_is_bracket_safe() -> None:
    """Test that the mermaid emitter escapes square brackets that would break
    the parser.
    """
    mermaid = render(build_graph(VisitImageSerializationModel[JsonRef]), "mermaid")
    stripped = mermaid.replace('["', "").replace('"]', "")
    assert "[" not in stripped
    assert "]" not in stripped
    assert "VisitImageSerializationModel~JsonRef~" in mermaid


def test_unknown_format_raises() -> None:
    """Test that render raises ValueError for an unrecognised format string."""
    graph = build_graph(Parent)
    with pytest.raises(ValueError):
        render(graph, "svg")


def test_render_is_deterministic() -> None:
    """Test that repeated render calls for the same graph and format produce
    identical output.
    """
    for fmt in ("dot", "mermaid", "tree"):
        assert render(build_graph(Parent), fmt) == render(build_graph(Parent), fmt)


def test_diagram_registered_in_group() -> None:
    """Test that the diagram subcommand is listed in the top-level help
    output.
    """
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert "diagram" in result.output


def test_diagram_list() -> None:
    """Test that --list outputs known schema names."""
    runner = CliRunner()
    result = _invoke(runner, "--list")
    assert result.exit_code == 0, result.output
    assert "visit_image" in result.output
    assert "cell_coadd" in result.output


def test_diagram_list_includes_entry_point_schemas() -> None:
    """Test that --list shows third-party schemas registered via entry
    points.
    """
    runner = CliRunner()
    fake = importlib.metadata.EntryPoint(
        name="third_party_schema", value="some.module:Model", group="lsst.images.schemas"
    )
    with mock.patch(
        "lsst.images.cli._diagram.importlib.metadata.entry_points", return_value=[fake]
    ) as entry_points:
        result = _invoke(runner, "--list")
    assert result.exit_code == 0, result.output
    entry_points.assert_called_once_with(group="lsst.images.schemas")
    assert "third_party_schema" in result.output


def test_diagram_model_default_mermaid() -> None:
    """Test that the diagram subcommand renders mermaid output with public
    class names by default.
    """
    runner = CliRunner()
    result = _invoke(runner, "visit-image")
    assert result.exit_code == 0, result.output
    assert "classDiagram" in result.output
    assert "VisitImage" in result.output
    assert "VisitImageSerializationModel" not in result.output


def test_diagram_serialization_names_flag() -> None:
    """Test that --serialization-names switches output to serialization model
    names.
    """
    runner = CliRunner()
    result = _invoke(runner, "visit-image", "--serialization-names")
    assert result.exit_code == 0, result.output
    assert "VisitImageSerializationModel" in result.output


def test_diagram_model_tree_and_dot_formats() -> None:
    """Test that --format tree and --format dot produce valid output for the
    cell-coadd model.
    """
    runner = CliRunner()
    tree = _invoke(runner, "cell-coadd", "--format", "tree")
    assert tree.exit_code == 0, tree.output
    assert "CellCoadd" in tree.output
    assert any(line.startswith(("├──", "└──")) for line in tree.output.splitlines())
    dot = _invoke(runner, "cell-coadd", "--format", "dot")
    assert dot.exit_code == 0, dot.output
    assert dot.output.lstrip().startswith("digraph")


def test_diagram_unknown_model_errors() -> None:
    """Test that an unrecognised model name causes a non-zero exit code."""
    runner = CliRunner()
    result = _invoke(runner, "not-a-model")
    assert result.exit_code != 0
    assert "Unknown model" in result.output


def test_diagram_model_and_file_are_mutually_exclusive() -> None:
    """Test that passing both a model name and --from-file produces a non-zero
    exit code.
    """
    runner = CliRunner()
    fixture = os.path.join(TESTDIR, "data", "schema_v1", "visit_image.json")
    assert _invoke(runner, "visit-image", "--from-file", fixture).exit_code != 0
    assert _invoke(runner).exit_code != 0


def test_diagram_from_file_collapses_psf() -> None:
    """Test that --from-file collapses the PSF to the concrete type stored in
    the JSON file.
    """
    runner = CliRunner()
    fixture = os.path.join(TESTDIR, "data", "schema_v1", "visit_image.json")
    result = _invoke(runner, "--from-file", fixture, "--format", "tree")
    assert result.exit_code == 0, result.output
    assert "GaussianPointSpreadFunction" in result.output


def test_diagram_output_to_file(tmp_path: Path) -> None:
    """Test that -o writes the diagram to the specified file."""
    runner = CliRunner()
    out = str(tmp_path / "diagram.mmd")
    result = _invoke(runner, "visit-image", "-o", out)
    assert result.exit_code == 0, result.output
    with open(out) as f:
        assert "classDiagram" in f.read()


def test_diagram_scalars_hidden_by_default() -> None:
    """Test that scalar fields are hidden by default but all-scalar leaf models
    still appear.
    """
    runner = CliRunner()
    result = _invoke(runner, "visit-image", "--format", "tree")
    assert result.exit_code == 0, result.output
    assert "schema_version" not in result.output
    assert "can_see_sky" not in result.output
    assert "ObservationInfo" in result.output
    assert "ObservationSummaryStats" in result.output


def test_diagram_attributes_flag_shows_scalars() -> None:
    """Test that --attributes causes scalar fields to appear in the tree
    output.
    """
    runner = CliRunner()
    result = _invoke(runner, "visit-image", "--format", "tree", "--attributes")
    assert result.exit_code == 0, result.output
    assert "schema_version" in result.output


def test_diagram_hide_field_clips_edges() -> None:
    """Test that --hide-field removes the named edge and any type only
    reachable through it.
    """
    runner = CliRunner()
    full = _invoke(runner, "cell-coadd", "--format", "tree").output
    assert "ArrayReferenceQuantityModel" in full
    assert "data (one of)" in full

    clipped = _invoke(runner, "cell-coadd", "--format", "tree", "--hide-field", "data").output
    assert "data (one of)" not in clipped
    assert "ArrayReferenceQuantityModel" not in clipped


def test_diagram_hide_type_removes_type() -> None:
    """Test that --hide-type removes the named type from the tree output."""
    runner = CliRunner()
    full = _invoke(runner, "cell-coadd", "--format", "tree").output
    assert "TableModel" in full
    clipped = _invoke(runner, "cell-coadd", "--format", "tree", "--hide-type", "TableModel").output
    assert "TableModel" not in clipped


def test_diagram_expand_leaves_changes_output() -> None:
    """Test that --expand-leaves produces more output lines than the default
    collapsed view.
    """
    runner = CliRunner()
    default = _invoke(runner, "visit-image", "--format", "tree", "--attributes").output
    expanded = _invoke(runner, "visit-image", "--format", "tree", "--attributes", "--expand-leaves").output
    assert len(expanded.splitlines()) > len(default.splitlines())
