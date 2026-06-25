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
import tempfile
import unittest
from typing import Annotated, Any
from unittest import mock

import pydantic
from click.testing import CliRunner

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


class BuildGraphTestCase(unittest.TestCase):
    """The introspection walk turns a model class into a node/edge graph."""

    def test_scalar_and_nested_model(self) -> None:
        graph = build_graph(Parent)

        root = graph.nodes[graph.root]
        self.assertEqual(root.label, "Parent")

        # The scalar field is an attribute on the node, not an edge.
        attrs = {a.name: a.type_str for a in root.attributes}
        self.assertIn("name", attrs)
        self.assertIn("str", attrs["name"])
        self.assertNotIn("child", attrs)

        # The model-valued field is a composition reference to the child node.
        refs = {r.name: r for r in root.references}
        self.assertIn("child", refs)
        self.assertEqual(refs["child"].cardinality, "one")
        (child_key,) = refs["child"].targets
        self.assertEqual(graph.nodes[child_key].label, "Child")

    def test_container_and_union_cardinalities(self) -> None:
        graph = build_graph(Containers)
        refs = {r.name: r for r in graph.nodes[graph.root].references}

        # No container field is mistaken for a scalar attribute.
        self.assertFalse(graph.nodes[graph.root].attributes)

        self.assertEqual(refs["maybe"].cardinality, "optional")
        self.assertEqual({graph.nodes[t].label for t in refs["maybe"].targets}, {"Child"})

        self.assertEqual(refs["many"].cardinality, "list")
        self.assertEqual({graph.nodes[t].label for t in refs["many"].targets}, {"Child"})

        self.assertEqual(refs["mapping"].cardinality, "dict")
        self.assertEqual({graph.nodes[t].label for t in refs["mapping"].targets}, {"Child"})

        # A union of two models points at both, with no "other" marker.
        self.assertEqual({graph.nodes[t].label for t in refs["either"].targets}, {"Child", "Other"})
        self.assertFalse(refs["either"].has_other)

        # A union mixing a model with Any keeps the edge and flags "other".
        self.assertEqual({graph.nodes[t].label for t in refs["open_union"].targets}, {"Child"})
        self.assertTrue(refs["open_union"].has_other)

    def test_optional_containers_are_model_references(self) -> None:
        graph = build_graph(OptionalContainers)
        root = graph.nodes[graph.root]
        refs = {r.name: r for r in root.references}
        attrs = {a.name for a in root.attributes}

        self.assertIn("maybe_many", refs)
        self.assertEqual({graph.nodes[t].label for t in refs["maybe_many"].targets}, {"Child"})
        self.assertNotIn("maybe_many", attrs)

        self.assertIn("maybe_mapping", refs)
        self.assertEqual({graph.nodes[t].label for t in refs["maybe_mapping"].targets}, {"Child"})
        self.assertNotIn("maybe_mapping", attrs)

    def test_scalar_type_strings_are_readable(self) -> None:
        graph = build_graph(Scalars)
        attrs = {a.name: a.type_str for a in graph.nodes[graph.root].attributes}
        self.assertEqual(attrs["mapping"], "dict[str, int]")
        self.assertEqual(attrs["sequence"], "list[str]")
        self.assertEqual(attrs["optional"], "int | None")
        self.assertEqual(attrs["plain"], "str")
        # Annotated metadata (which can have an unstable repr) is stripped.
        self.assertEqual(attrs["annotated"], "int")
        self.assertEqual(attrs["annotated_optional"], "float | None")

    def test_type_alias_union_is_resolved(self) -> None:
        # A PEP 695 ``type X = A | B`` alias expands to its member models.
        graph = build_graph(AliasHolder)
        (ref,) = graph.nodes[graph.root].references
        self.assertEqual(ref.name, "thing")
        self.assertEqual({graph.nodes[t].label for t in ref.targets}, {"Child", "Other"})

    def test_self_referential_model_does_not_loop(self) -> None:
        graph = build_graph(TreeNode)
        # A single node that references itself, rather than infinite recursion.
        self.assertEqual(len(graph.nodes), 1)
        (ref,) = graph.nodes[graph.root].references
        self.assertEqual(ref.targets, [graph.root])
        self.assertEqual(ref.cardinality, "list")


class RealModelTestCase(unittest.TestCase):
    """The walk handles the real serialization models."""

    def test_visit_image(self) -> None:
        from lsst.images import VisitImageSerializationModel
        from lsst.images.serialization._asdf_utils import ArrayReferenceModel

        graph = build_graph(VisitImageSerializationModel[ArrayReferenceModel])
        root = graph.nodes[graph.root]
        refs = {r.name: r for r in root.references}

        for expected in ("image", "mask", "variance", "sky_projection", "psf"):
            self.assertIn(expected, refs)

        # The PSF field is the Piff | PSFEx | Gaussian | Any union.
        psf_labels = {graph.nodes[t].label for t in refs["psf"].targets}
        self.assertGreaterEqual(len(psf_labels), 3)
        self.assertTrue(refs["psf"].has_other)

        # photometric_scaling is a TypeAliasType over a discriminated union of
        # field models; it must expand to those members, not show as a scalar.
        self.assertIn("photometric_scaling", refs)
        field_labels = {graph.nodes[t].label for t in refs["photometric_scaling"].targets}
        self.assertIn("SumFieldSerializationModel", field_labels)

    def test_cell_coadd(self) -> None:
        from lsst.images.cells import CellCoaddSerializationModel
        from lsst.images.serialization._asdf_utils import ArrayReferenceModel

        graph = build_graph(CellCoaddSerializationModel[ArrayReferenceModel])
        refs = {r.name: r for r in graph.nodes[graph.root].references}

        self.assertEqual(refs["noise_realizations"].cardinality, "list")
        self.assertEqual(refs["mask_fractions"].cardinality, "dict")


class PolicyTestCase(unittest.TestCase):
    """The collapse policy controls how far the walk recurses."""

    def test_collapsed_type_is_a_leaf(self) -> None:
        graph = build_graph(Top, policy=Policy(leaves=frozenset({"Middle"})))
        # Middle is present (so its edge has a target) but not expanded.
        labels = {n.label for n in graph.nodes.values()}
        self.assertEqual(labels, {"Top", "Middle"})
        middle = next(n for n in graph.nodes.values() if n.label == "Middle")
        self.assertTrue(middle.collapsed)
        self.assertFalse(middle.references)
        self.assertFalse(middle.attributes)

    def test_make_policy_expand_and_collapse(self) -> None:
        policy = make_policy(expand=["ArrayReferenceModel"], collapse=["Middle"])
        self.assertNotIn("ArrayReferenceModel", policy.leaves)
        self.assertIn("Middle", policy.leaves)
        # Other defaults remain collapsed.
        self.assertIn("TableModel", policy.leaves)

    def test_make_policy_expand_leaves_clears_defaults(self) -> None:
        self.assertEqual(make_policy(expand_leaves=True).leaves, frozenset())

    def test_include_attributes_false_elides_scalars(self) -> None:
        # The scalar field is dropped, but the model edge is kept.
        graph = build_graph(Parent, policy=make_policy(include_attributes=False))
        root = graph.nodes[graph.root]
        self.assertFalse(root.attributes)
        self.assertEqual([r.name for r in root.references], ["child"])

    def test_all_scalar_model_becomes_leaf_without_attributes(self) -> None:
        graph = build_graph(Scalars, policy=make_policy(include_attributes=False))
        root = graph.nodes[graph.root]
        self.assertFalse(root.attributes)
        self.assertFalse(root.references)

    def test_instance_attributes_can_be_elided(self) -> None:
        owner = Owner(pet=Dog(bark="woof"), pets=[Dog(bark="a")], missing=None)
        graph = build_instance_graph(owner, policy=make_policy(include_attributes=False))
        # The absent optional "missing" is not shown as a None attribute.
        self.assertFalse(graph.nodes[graph.root].attributes)

    def test_public_names_relabel_real_model(self) -> None:
        from lsst.images import VisitImageSerializationModel
        from lsst.images.serialization._asdf_utils import ArrayReferenceModel

        graph = build_graph(
            VisitImageSerializationModel[ArrayReferenceModel], policy=make_policy(public_names=True)
        )
        labels = {n.label for n in graph.nodes.values()}
        self.assertIn("VisitImage", labels)
        self.assertIn("SkyProjection", labels)
        self.assertNotIn("VisitImageSerializationModel[ArrayReferenceModel]", labels)
        # PUBLIC_TYPE is the builtin dict here, so the model name is kept.
        self.assertIn("ApertureCorrectionMapSerializationModel", labels)

    def test_public_names_collapse_matches_either_name(self) -> None:
        from lsst.images import VisitImageSerializationModel
        from lsst.images.serialization._asdf_utils import ArrayReferenceModel

        graph = build_graph(
            VisitImageSerializationModel[ArrayReferenceModel],
            policy=make_policy(public_names=True, collapse=["Image"]),
        )
        image = next(n for n in graph.nodes.values() if n.label == "Image")
        self.assertTrue(image.collapsed)

    def test_hide_fields_drops_named_fields(self) -> None:
        graph = build_graph(Parent, policy=make_policy(hide_fields=["child"]))
        # The named field is gone, and the sub-tree reached only via it too.
        self.assertFalse(graph.nodes[graph.root].references)
        self.assertNotIn("Child", {n.label for n in graph.nodes.values()})

    def test_hide_type_removes_nodes_and_edges(self) -> None:
        graph = build_graph(Parent, policy=make_policy(hide_types=["Child"]))
        self.assertNotIn("Child", {n.label for n in graph.nodes.values()})
        self.assertFalse(graph.nodes[graph.root].references)

    def test_hide_type_matches_public_name(self) -> None:
        from lsst.images import VisitImageSerializationModel
        from lsst.images.serialization._asdf_utils import ArrayReferenceModel

        model = VisitImageSerializationModel[ArrayReferenceModel]
        # "Image" is the public name of ImageSerializationModel.
        graph = build_graph(model, policy=make_policy(public_names=True, hide_types=["Image"]))
        self.assertNotIn("Image", {n.label for n in graph.nodes.values()})

    def test_default_collapses_asdf_helpers_in_real_model(self) -> None:
        from lsst.images import VisitImageSerializationModel
        from lsst.images.serialization._asdf_utils import ArrayReferenceModel

        self.assertIn("ArrayReferenceModel", DEFAULT_LEAF_TYPES)
        model = VisitImageSerializationModel[ArrayReferenceModel]

        default_graph = build_graph(model)
        array_node = next(n for n in default_graph.nodes.values() if n.label == "ArrayReferenceModel")
        self.assertTrue(array_node.collapsed)

        expanded_graph = build_graph(model, policy=make_policy(expand_leaves=True))
        expanded_array = next(n for n in expanded_graph.nodes.values() if n.label == "ArrayReferenceModel")
        self.assertFalse(expanded_array.collapsed)
        # Expanding the leaves reveals strictly more of the model.
        self.assertGreater(len(expanded_graph.nodes), len(default_graph.nodes))


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


class InstanceGraphTestCase(unittest.TestCase):
    """Instance mode collapses unions to the concrete values present."""

    def test_union_collapses_to_concrete_member(self) -> None:
        owner = Owner(pet=Dog(bark="woof"), pets=[Dog(bark="a"), Cat(meow="m")], missing=None)
        graph = build_instance_graph(owner)
        refs = {r.name: r for r in graph.nodes[graph.root].references}

        # A single-valued union shows only the concrete type present.
        self.assertEqual({graph.nodes[t].label for t in refs["pet"].targets}, {"Dog"})
        self.assertEqual(refs["pet"].cardinality, "one")

        # A list shows every distinct concrete element type present.
        self.assertEqual({graph.nodes[t].label for t in refs["pets"].targets}, {"Dog", "Cat"})
        self.assertEqual(refs["pets"].cardinality, "list")

        # An absent optional is just a None attribute, not an edge.
        self.assertNotIn("missing", refs)
        attrs = {a.name: a.type_str for a in graph.nodes[graph.root].attributes}
        self.assertEqual(attrs["missing"], "None")

    def test_populated_container_shows_only_present_types(self) -> None:
        graph = build_instance_graph(Kennel(occupants={"x": Dog(bark="woof")}))
        ref = {r.name: r for r in graph.nodes[graph.root].references}["occupants"]
        self.assertEqual({graph.nodes[t].label for t in ref.targets}, {"Dog"})

    def test_repeated_instances_merge_nested_concrete_types(self) -> None:
        root = WrapperCollection(
            wrappers=[
                RepeatedWrapper(thing=Dog(bark="woof")),
                RepeatedWrapper(thing=Cat(meow="m")),
            ]
        )
        graph = build_instance_graph(root)
        wrapper = next(n for n in graph.nodes.values() if n.label == "RepeatedWrapper")
        refs = {r.name: r for r in wrapper.references}

        self.assertIn("thing", refs)
        self.assertEqual({graph.nodes[t].label for t in refs["thing"].targets}, {"Dog", "Cat"})

    def test_hide_type_instance_mode(self) -> None:
        graph = build_instance_graph(
            Kennel(occupants={"x": Dog(bark="woof")}),
            policy=make_policy(hide_types=["Dog"], include_attributes=False),
        )
        self.assertFalse(graph.nodes[graph.root].references)
        self.assertNotIn("Dog", {n.label for n in graph.nodes.values()})

    def test_empty_container_yields_no_edge(self) -> None:
        # Instance mode reports only what the file holds; an empty container is
        # not expanded into its declared element types.
        graph = build_instance_graph(Kennel(occupants={}), policy=make_policy(include_attributes=False))
        self.assertFalse(graph.nodes[graph.root].references)
        self.assertFalse(graph.nodes[graph.root].attributes)

    def test_graph_from_file_collapses_psf(self) -> None:
        path = os.path.join(TESTDIR, "data", "schema_v1", "visit_image.json")
        graph = graph_from_file(path)
        self.assertTrue(graph.nodes[graph.root].label.startswith("VisitImageSerializationModel"))
        refs = {r.name: r for r in graph.nodes[graph.root].references}
        # The fixture's PSF is a single concrete type, not the abstract union.
        psf_labels = {graph.nodes[t].label for t in refs["psf"].targets}
        self.assertEqual(psf_labels, {"GaussianPSFSerializationModel"})
        self.assertFalse(refs["psf"].has_other)


class EmitterTestCase(unittest.TestCase):
    """The three output formats render the graph faithfully."""

    def setUp(self) -> None:
        self.graph = build_graph(Parent)

    def test_dot(self) -> None:
        dot = render(self.graph, "dot")
        self.assertTrue(dot.startswith("digraph"))
        self.assertIn("Parent", dot)
        self.assertIn("Child", dot)
        self.assertIn("name", dot)  # scalar attribute
        self.assertIn("->", dot)  # composition edge
        self.assertIn("child", dot)  # edge label (field name)

    def test_mermaid(self) -> None:
        mermaid = render(self.graph, "mermaid")
        self.assertTrue(mermaid.lstrip().startswith("classDiagram"))
        self.assertIn("Parent", mermaid)
        self.assertIn("Child", mermaid)
        self.assertIn("name", mermaid)
        self.assertIn("-->", mermaid)
        self.assertIn("child", mermaid)

    def test_tree(self) -> None:
        tree = render(self.graph, "tree")
        lines = tree.splitlines()
        self.assertEqual(lines[0], "Parent")
        # The scalar attribute and the nested model both appear as children.
        self.assertTrue(any("name" in line for line in lines))
        self.assertTrue(any("child" in line and "Child" in line for line in lines))
        # tree(1)-style branch glyphs.
        self.assertTrue(any(line.startswith(("├──", "└──")) for line in lines))

    def test_mermaid_real_model_is_bracket_safe(self) -> None:
        # Square brackets (from generic parameters / typing) break mermaid's
        # class-diagram parser, so they must not survive into the output.
        from lsst.images import VisitImageSerializationModel
        from lsst.images.serialization._asdf_utils import ArrayReferenceModel

        mermaid = render(build_graph(VisitImageSerializationModel[ArrayReferenceModel]), "mermaid")
        # The only legitimate brackets are the ``class X["label"]`` declaration
        # tokens; none may leak into label text, member lines, or edge labels.
        stripped = mermaid.replace('["', "").replace('"]', "")
        self.assertNotIn("[", stripped)
        self.assertNotIn("]", stripped)
        # The generic parameter is still shown, in mermaid's ~ generic style.
        self.assertIn("VisitImageSerializationModel~ArrayReferenceModel~", mermaid)

    def test_unknown_format_raises(self) -> None:
        with self.assertRaises(ValueError):
            render(self.graph, "svg")

    def test_deterministic(self) -> None:
        for fmt in ("dot", "mermaid", "tree"):
            self.assertEqual(render(build_graph(Parent), fmt), render(build_graph(Parent), fmt))


class DiagramCliTestCase(unittest.TestCase):
    """The ``lsst-images-admin diagram`` subcommand."""

    def setUp(self) -> None:
        self.runner = CliRunner()
        self.fixture = os.path.join(TESTDIR, "data", "schema_v1", "visit_image.json")

    def invoke(self, *args: str):
        return self.runner.invoke(main, ["diagram", *args])

    def test_registered_in_group(self) -> None:
        result = self.runner.invoke(main, ["--help"])
        self.assertIn("diagram", result.output)

    def test_list(self) -> None:
        result = self.invoke("--list")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("visit_image", result.output)
        self.assertIn("cell_coadd", result.output)

    def test_list_includes_entry_point_schemas(self) -> None:
        # A third-party schema known only through an entry point is listed by
        # name without its provider module being imported.
        fake = importlib.metadata.EntryPoint(
            name="third_party_schema", value="some.module:Model", group="lsst.images.schemas"
        )
        with mock.patch(
            "lsst.images.cli._diagram.importlib.metadata.entry_points", return_value=[fake]
        ) as entry_points:
            result = self.invoke("--list")
        self.assertEqual(result.exit_code, 0, result.output)
        entry_points.assert_called_once_with(group="lsst.images.schemas")
        self.assertIn("third_party_schema", result.output)

    def test_model_default_mermaid(self) -> None:
        result = self.invoke("visit-image")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("classDiagram", result.output)
        # Public class names are used by default.
        self.assertIn("VisitImage", result.output)
        self.assertNotIn("VisitImageSerializationModel", result.output)

    def test_serialization_names_flag(self) -> None:
        result = self.invoke("visit-image", "--serialization-names")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("VisitImageSerializationModel", result.output)

    def test_model_tree_and_dot_formats(self) -> None:
        tree = self.invoke("cell-coadd", "--format", "tree")
        self.assertEqual(tree.exit_code, 0, tree.output)
        self.assertIn("CellCoadd", tree.output)
        self.assertTrue(any(line.startswith(("├──", "└──")) for line in tree.output.splitlines()))

        dot = self.invoke("cell-coadd", "--format", "dot")
        self.assertEqual(dot.exit_code, 0, dot.output)
        self.assertTrue(dot.output.lstrip().startswith("digraph"))

    def test_unknown_model_errors(self) -> None:
        result = self.invoke("not-a-model")
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Unknown model", result.output)

    def test_model_and_file_are_mutually_exclusive(self) -> None:
        self.assertNotEqual(self.invoke("visit-image", "--from-file", self.fixture).exit_code, 0)
        self.assertNotEqual(self.invoke().exit_code, 0)

    def test_from_file_collapses_psf(self) -> None:
        result = self.invoke("--from-file", self.fixture, "--format", "tree")
        self.assertEqual(result.exit_code, 0, result.output)
        # Public name of the concrete PSF type stored in the file.
        self.assertIn("GaussianPointSpreadFunction", result.output)

    def test_output_to_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "diagram.mmd")
            result = self.invoke("visit-image", "-o", out)
            self.assertEqual(result.exit_code, 0, result.output)
            with open(out) as f:
                self.assertIn("classDiagram", f.read())

    def test_scalars_hidden_by_default(self) -> None:
        result = self.invoke("visit-image", "--format", "tree")
        self.assertEqual(result.exit_code, 0, result.output)
        # Scalar fields (incl. schema bookkeeping) are gone by default...
        self.assertNotIn("schema_version", result.output)
        self.assertNotIn("can_see_sky", result.output)
        # ...but all-scalar models still appear as leaf nodes.
        self.assertIn("ObservationInfo", result.output)
        self.assertIn("ObservationSummaryStats", result.output)

    def test_attributes_flag_shows_scalars(self) -> None:
        result = self.invoke("visit-image", "--format", "tree", "--attributes")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("schema_version", result.output)

    def test_hide_field_clips_edges(self) -> None:
        full = self.invoke("cell-coadd", "--format", "tree").output
        self.assertIn("ArrayReferenceQuantityModel", full)
        self.assertIn("data (one of)", full)

        clipped = self.invoke("cell-coadd", "--format", "tree", "--hide-field", "data").output
        # The named edges, and the type only reachable through them, are gone.
        self.assertNotIn("data (one of)", clipped)
        self.assertNotIn("ArrayReferenceQuantityModel", clipped)

    def test_hide_type_removes_type(self) -> None:
        full = self.invoke("cell-coadd", "--format", "tree").output
        self.assertIn("TableModel", full)
        clipped = self.invoke("cell-coadd", "--format", "tree", "--hide-type", "TableModel").output
        self.assertNotIn("TableModel", clipped)

    def test_expand_leaves_changes_output(self) -> None:
        default = self.invoke("visit-image", "--format", "tree", "--attributes").output
        expanded = self.invoke("visit-image", "--format", "tree", "--attributes", "--expand-leaves").output
        self.assertGreater(len(expanded.splitlines()), len(default.splitlines()))


if __name__ == "__main__":
    unittest.main()
