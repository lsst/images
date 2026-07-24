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

import os

import numpy as np
from rich.console import Console

from lsst.images._geom import Box
from lsst.images._image import Image
from lsst.images._mask import Mask, MaskPlane, MaskSchema
from lsst.images.describe import DescribableMixin, FieldRole, Report, ReportField, ReportTable
from lsst.images.serialization import read_archive


def test_report_model_defaults() -> None:
    """Report and its components have the expected fields and defaults."""
    field = ReportField(label="bbox", value="[y=0:4, x=0:4]")
    assert field.unit is None
    assert field.repr_value is None
    assert field.role is FieldRole.ARG
    assert field.positional is False

    table = ReportTable(title="Axes", columns=["Axis", "Label"], rows=[[1, "RA"]])
    assert table.role is FieldRole.DERIVED

    report = Report(type_name="Image")
    assert report.title is None
    assert report.summary is None
    assert report.fields == []
    assert report.tables == []
    assert report.children == {}


def test_to_repr_uses_arg_fields_only() -> None:
    """to_repr emits Type(label=repr_value) for ARG fields, skipping others."""
    report = Report(
        type_name="Image",
        fields=[
            ReportField(label="bbox", value="[y=0:4, x=0:4]", repr_value="Box(...)"),
            ReportField(label="array", value="<huge>", repr_value="...", role=FieldRole.ARG),
            ReportField(label="corner", value="10:04:21", role=FieldRole.DERIVED),
        ],
        tables=[ReportTable(title="T", columns=["a"], rows=[[1]])],
    )
    assert report.to_repr() == "Image(bbox=Box(...), array=...)"


def test_to_repr_defaults_repr_value_to_repr_of_value() -> None:
    """A field with no repr_value falls back to repr(value)."""
    report = Report(type_name="Interval", fields=[ReportField(label="start", value=3)])
    assert report.to_repr() == "Interval(start=3)"


def test_to_repr_supports_positional_fields() -> None:
    """Positional ARG fields emit their value without a label=."""
    report = Report(
        type_name="MaskSchema",
        fields=[
            ReportField(label="planes", value="[...]", repr_value="[...]", positional=True),
            ReportField(label="dtype", value="uint8", repr_value="dtype('uint8')"),
        ],
    )
    assert report.to_repr() == "MaskSchema([...], dtype=dtype('uint8'))"


def test_to_str_prefers_summary() -> None:
    """to_str returns the summary verbatim when present."""
    report = Report(type_name="Image", summary="Image([y=0:4, x=0:4], float32)")
    assert report.to_str() == "Image([y=0:4, x=0:4], float32)"


def test_to_str_without_summary_lists_fields() -> None:
    """to_str falls back to type name plus the first few ARG values."""
    report = Report(
        type_name="Interval",
        fields=[ReportField(label="start", value=0), ReportField(label="stop", value=4)],
    )
    assert report.to_str() == "Interval(0, 4)"


def test_rich_renders_fields_tables_and_children() -> None:
    """__rich__ output contains labels, table headers, and child keys."""
    report = Report(
        type_name="SkyProjection",
        title="ICRS coordinates",
        fields=[ReportField(label="Domain", value="SKY")],
        tables=[ReportTable(title="Axes", columns=["Axis", "Label"], rows=[[1, "RA"], [2, "Dec"]])],
        children={"pixel": Report(type_name="GeneralFrame", fields=[ReportField(label="unit", value="pix")])},
    )
    console = Console(record=True, width=100)
    console.print(report)
    text = console.export_text()
    assert "ICRS coordinates" in text
    assert "Domain" in text and "SKY" in text
    assert "Axis" in text and "Label" in text and "RA" in text
    assert "pixel" in text and "GeneralFrame" in text


def test_repr_html_produces_html() -> None:
    """_repr_html_ returns an HTML fragment mentioning the content."""
    report = Report(type_name="Interval", fields=[ReportField(label="start", value=3)])
    html = report._repr_html_()
    assert "<" in html and ">" in html
    assert "Interval" in html


def test_mixin_derives_dunders_from_describe() -> None:
    """DescribableMixin wires repr/str/html to _describe."""

    class Widget(DescribableMixin):
        def _describe(self, **kwargs: object) -> Report:
            return Report(
                type_name="Widget",
                summary="Widget(size=5)",
                fields=[ReportField(label="size", value=5)],
            )

    widget = Widget()
    assert repr(widget) == "Widget(size=5)"
    assert str(widget) == "Widget(size=5)"
    assert "Widget" in widget._repr_html_()
    assert isinstance(widget.describe(), Report)


def test_public_api_importable_from_package() -> None:
    """The describe public API is re-exported from lsst.images."""
    import lsst.images as images

    for name in ("Describable", "DescribableMixin", "FieldRole", "Report", "ReportField", "ReportTable"):
        assert hasattr(images, name), name


def test_visit_image_describe_nested() -> None:
    """A deserialized VisitImage produces a nested report with WCS corners."""
    path = os.path.join(os.path.dirname(__file__), "data", "schema_v1", "visit_image.json")
    visit_image = read_archive(path)
    report = visit_image.describe()
    assert report.type_name == "VisitImage"
    # Components appear as children.
    assert "image" in report.children
    assert "mask" in report.children
    assert "sky_projection" in report.children
    # The sky_projection child received the container bbox, so it has corners.
    sky = report.children["sky_projection"]
    assert any(t.title == "Corners" for t in sky.tables)
    # Rich and HTML renderers run without error.
    assert isinstance(report._repr_html_(), str)
    report.__rich__()


def test_rich_renders_bracketed_values_literally() -> None:
    """Bracketed strings in field values and table cells render verbatim."""
    report = Report(
        type_name="TestType",
        fields=[ReportField(label="region", value="[y=0:4, x=0:4]")],
        tables=[
            ReportTable(
                title="Cells",
                columns=["Value"],
                rows=[["[y=0:4, x=0:4]"], ["[/x=0:4]"]],
            )
        ],
    )
    console = Console(record=True, width=120)
    console.print(report)
    text = console.export_text()
    # Both bracket styles must appear verbatim in the exported text.
    assert "[y=0:4, x=0:4]" in text
    assert "[/x=0:4]" in text
    # _repr_html_ must not raise on either bracket style.
    html = report._repr_html_()
    assert "[y=0:4, x=0:4]" in html
    assert "[/x=0:4]" in html


def test_rich_renders_real_image_bbox_literally() -> None:
    """A real Image with a bracketed bbox renders the bbox verbatim."""
    img = Image(np.zeros((4, 4), dtype=np.float32), bbox=Box.factory[0:4, 0:4])
    report = img._describe()
    console = Console(record=True, width=120)
    console.print(report)
    text = console.export_text()
    assert "[y=0:4, x=0:4]" in text
    html = report._repr_html_()
    assert "[y=0:4, x=0:4]" in html


def test_composite_report_deduplicates_bbox_and_sky_projection() -> None:
    """Composite reports show bbox and sky_projection once, not per
    component.
    """
    path = os.path.join(os.path.dirname(__file__), "data", "schema_v1", "visit_image.json")
    report = read_archive(path).describe()

    # The composite carries exactly one top-level bbox field and one
    # top-level sky_projection child.
    assert sum(1 for f in report.fields if f.label == "bbox") == 1
    assert "sky_projection" in report.children

    # The image/mask/variance children no longer repeat the shared geometry.
    for name in ("image", "mask", "variance"):
        child = report.children[name]
        assert not any(f.label == "bbox" for f in child.fields), name
        assert "sky_projection" not in child.children, name


def test_repr_str_do_not_trigger_detail() -> None:
    """Repr and str never pass detail; a MaskSchema report from repr has no
    counts column.
    """
    schema = MaskSchema([MaskPlane("BAD", "bad")], dtype=np.uint8)
    mask = Mask(0, schema=schema, bbox=Box.factory[0:2, 0:2])
    # repr/str must be unaffected and cheap.
    assert repr(mask).startswith("Mask(")
    assert str(mask).startswith("Mask(")
