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

__all__ = ("diagram",)

import importlib.metadata

import click

from ..diagram import build_graph, graph_from_file, make_policy, render
from ..serialization._asdf_utils import ArrayReferenceModel
from ..serialization._io import (
    _BUILTIN_SCHEMA_PROVIDERS,
    _REGISTRY,
    _SCHEMA_ENTRY_POINT_GROUP,
    class_for_schema,
    parameterize_tree,
)


def _available_schemas() -> list[str]:
    """Return the sorted schema names that can be diagrammed.

    Importing this module imports ``lsst.images`` first, so every
    unconditionally-imported model is already registered in ``_REGISTRY``.  The
    lazily-loaded built-in providers and the third-party
    ``lsst.images.schemas`` entry points are added by name without importing
    them, so a schema appears here even though `class_for_schema` only loads
    its provider on demand.
    """
    entry_point_names = {
        entry_point.name for entry_point in importlib.metadata.entry_points(group=_SCHEMA_ENTRY_POINT_GROUP)
    }
    return sorted(set(_REGISTRY) | set(_BUILTIN_SCHEMA_PROVIDERS) | entry_point_names)


@click.command(name="diagram")
@click.argument("model", required=False)
@click.option(
    "--from-file",
    "from_file",
    type=click.Path(exists=True, dir_okay=False),
    help="Diagram the concrete structure of a serialized file instead of a model.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["mermaid", "dot", "tree"]),
    default="mermaid",
    show_default=True,
    help="Output format.",
)
@click.option("--expand", multiple=True, metavar="TYPE", help="Force a model type to expand (repeatable).")
@click.option(
    "--collapse", multiple=True, metavar="TYPE", help="Force a model type to collapse to a leaf (repeatable)."
)
@click.option(
    "--expand-leaves",
    is_flag=True,
    help="Expand the serialization-helper leaves that are collapsed by default.",
)
@click.option(
    "--attributes",
    is_flag=True,
    help="Show scalar fields too; by default only model composition is shown.",
)
@click.option(
    "--hide-field",
    "hide_fields",
    multiple=True,
    metavar="NAME",
    help="Drop fields with this name and any sub-tree reached only through them (repeatable).",
)
@click.option(
    "--hide-type",
    "hide_types",
    multiple=True,
    metavar="TYPE",
    help="Drop a type entirely, removing every edge that points at it (repeatable).",
)
@click.option(
    "--serialization-names",
    is_flag=True,
    help="Label nodes with serialization-model class names instead of public class names.",
)
@click.option(
    "-o", "--output", type=click.Path(dir_okay=False), default=None, help="Write to a file instead of stdout."
)
@click.option("--list", "list_models", is_flag=True, help="List available schema names and exit.")
def diagram(
    model: str | None,
    from_file: str | None,
    fmt: str,
    expand: tuple[str, ...],
    collapse: tuple[str, ...],
    expand_leaves: bool,
    attributes: bool,
    hide_fields: tuple[str, ...],
    hide_types: tuple[str, ...],
    serialization_names: bool,
    output: str | None,
    list_models: bool,
) -> None:
    """Generate a composition diagram of an lsst.images model.

    Pass a schema name (e.g. ``visit-image`` or ``cell-coadd``) to diagram the
    abstract model, or ``--from-file`` to diagram the concrete structure of a
    serialized file, which collapses unions such as the PSF to the type
    actually stored.  Use ``--list`` to see the available schema names.
    """
    if list_models:
        for name in _available_schemas():
            click.echo(name)
        return
    if bool(model) == bool(from_file):
        raise click.UsageError("Provide exactly one of MODEL or --from-file.")

    policy = make_policy(
        expand_leaves=expand_leaves,
        expand=expand,
        collapse=collapse,
        include_attributes=attributes,
        hide_fields=hide_fields,
        hide_types=hide_types,
        public_names=not serialization_names,
    )
    if from_file is not None:
        graph = graph_from_file(from_file, policy=policy)
    else:
        assert model is not None
        tree_cls = class_for_schema(model.replace("-", "_"))
        if tree_cls is None:
            available = ", ".join(_available_schemas())
            raise click.ClickException(f"Unknown model {model!r}; choose from: {available}.")
        graph = build_graph(parameterize_tree(tree_cls, ArrayReferenceModel), policy=policy)

    text = render(graph, fmt)
    if output is not None:
        with open(output, "w") as stream:
            stream.write(text)
    else:
        click.echo(text, nl=False)
