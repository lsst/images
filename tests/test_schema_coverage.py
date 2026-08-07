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
"""The fixture coverage reporter, exercised through purpose-built schemas.

The mechanics are pinned with doubles rather than the package's own schemas,
so a new model or a new fixture variant cannot turn these into failures.  The
checks against the committed tree assert structure that stays true as the tree
grows, never counts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pydantic
import pytest

from lsst.images.serialization import ArchiveTree, InputArchive
from lsst.images.tests import (
    canonical_fixture_text,
    format_coverage_report,
    schema_coverage,
)

FIXTURE_DIR = Path(__file__).parent / "data" / "schemas"


class _Leaf(ArchiveTree):
    """A stamped sub-schema a container can hold."""

    SCHEMA_NAME: ClassVar[str] = "coverage_leaf"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = dict

    tag: str = pydantic.Field(default="", description="A value to round-trip.")

    def deserialize(self, archive: InputArchive[Any], **kwargs: Any) -> Any:
        """Return nothing; these doubles are never deserialized."""
        raise NotImplementedError()


class _OtherLeaf(ArchiveTree):
    """A second stamped sub-schema, so a union has more than one candidate."""

    SCHEMA_NAME: ClassVar[str] = "coverage_other_leaf"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = dict

    other: int = pydantic.Field(default=0, description="A value to round-trip.")

    def deserialize(self, archive: InputArchive[Any], **kwargs: Any) -> Any:
        """Return nothing; these doubles are never deserialized."""
        raise NotImplementedError()


class _Inner(pydantic.BaseModel):
    """An unstamped nested model, whose paths belong to its container."""

    depth: int = pydantic.Field(default=0, description="A nested value.")
    note: str = pydantic.Field(default="", description="Another nested value.")


class _Container(ArchiveTree):
    """A composite schema holding leaves in several kinds of position."""

    SCHEMA_NAME: ClassVar[str] = "coverage_container"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = dict

    chosen: _Leaf | _OtherLeaf = pydantic.Field(
        default_factory=_Leaf, description="A union position with two candidates."
    )
    entries: list[_Leaf] = pydantic.Field(default_factory=list, description="A list position.")
    table: dict[str, _Leaf] = pydantic.Field(default_factory=dict, description="A mapping position.")
    inner: _Inner = pydantic.Field(default_factory=_Inner, description="An unstamped nested model.")

    def deserialize(self, archive: InputArchive[Any], **kwargs: Any) -> Any:
        """Return nothing; these doubles are never deserialized."""
        raise NotImplementedError()


_PACKAGE = _Container.__module__
_CONTAINER_KEY = ("coverage_container", "1.0.0")
_LEAF_KEY = ("coverage_leaf", "1.0.0")
_OTHER_KEY = ("coverage_other_leaf", "1.0.0")


def _write(directory: Path, tree: ArchiveTree, *, retired: bool = False) -> Path:
    """Write a tree as a canonical fixture and return its path.

    Parameters
    ----------
    directory
        Root of the fixture tree to write into.
    tree
        Instance to serialize.
    retired
        Whether to place the file in the schema's ``retired`` subdirectory.
    """
    cls = type(tree)
    parent = directory / cls.SCHEMA_NAME / ("retired" if retired else "")
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / f"{cls.SCHEMA_NAME}-{cls.SCHEMA_VERSION}.json"
    path.write_text(canonical_fixture_text(tree))
    return path


def test_container_fixture_credits_its_embedded_sub_schema(tmp_path: Path) -> None:
    """Verify a sub-schema is credited by the container that embeds it.

    This is the property that makes per-container attribution unnecessary: a
    leaf reached only from a container still has that container as a source.
    """
    _write(tmp_path, _Container(chosen=_Leaf(tag="x")))
    report = schema_coverage(tmp_path, package=_PACKAGE)
    assert report.schemas[_LEAF_KEY].sources == {"coverage_container-1.0.0.json"}
    assert ".tag" in report.schemas[_LEAF_KEY].expressed


def test_coverage_is_keyed_by_schema_never_by_container(tmp_path: Path) -> None:
    """Verify the report's keys are schemas, so a gap cannot name a container.

    A key that mentioned the containing fixture is what would let the report
    claim one container "fails to cover" a schema another one exercises.
    """
    _write(tmp_path, _Container(chosen=_Leaf()))
    report = schema_coverage(tmp_path, package=_PACKAGE)
    assert set(report.schemas) == {_CONTAINER_KEY, _LEAF_KEY, _OTHER_KEY}


def test_union_position_reports_candidates_and_what_was_reached(tmp_path: Path) -> None:
    """Verify a union position names every candidate and those reached."""
    _write(tmp_path, _Container(chosen=_Leaf()))
    report = schema_coverage(tmp_path, package=_PACKAGE)
    (position,) = [p for p in report.schemas[_CONTAINER_KEY].positions if p.path == ".chosen"]
    assert position.candidates == {"coverage_leaf", "coverage_other_leaf"}
    assert position.reached == {"coverage_leaf"}
    assert position.missing == {"coverage_other_leaf"}


def test_a_second_fixture_closes_a_union_gap(tmp_path: Path) -> None:
    """Verify reaching the other branch empties the position's missing set.

    A variant added to widen coverage should show up here as the gap closing,
    which is the whole point of reporting the position.
    """
    _write(tmp_path, _Container(chosen=_Leaf()))
    other = tmp_path / "coverage_container" / "coverage_container-1.0.0-other.json"
    other.write_text(canonical_fixture_text(_Container(chosen=_OtherLeaf(other=3))))
    report = schema_coverage(tmp_path, package=_PACKAGE)
    (position,) = [p for p in report.schemas[_CONTAINER_KEY].positions if p.path == ".chosen"]
    assert position.reached == {"coverage_leaf", "coverage_other_leaf"}
    assert not position.missing


def test_list_and_mapping_positions_normalize_away_data_keys(tmp_path: Path) -> None:
    """Verify list indices and mapping keys do not appear as schema paths.

    An index and a mapping key are chosen by whoever wrote the data, so a
    fixture with two list entries must not read as two distinct positions.
    """
    _write(
        tmp_path,
        _Container(entries=[_Leaf(tag="a"), _Leaf(tag="b")], table={"first": _Leaf(tag="c")}),
    )
    report = schema_coverage(tmp_path, package=_PACKAGE)
    paths = {p.path for p in report.schemas[_CONTAINER_KEY].positions}
    assert ".entries[]" in paths
    assert ".table" in paths
    assert not any(path.startswith(".entries[0") or path.startswith(".table.") for path in paths)


def test_absent_roots_collapse_an_unset_subtree(tmp_path: Path) -> None:
    """Verify an unset field reports one gap, not one per descendant.

    ``butler_info`` is `None` unless a butler wrote the file, and its
    ``exclude_if`` drops it from the dump entirely, taking its whole declared
    subtree with it.  Only the outermost path is actionable: populate that
    field and everything beneath it becomes reachable.
    """
    _write(tmp_path, _Container())
    coverage = schema_coverage(tmp_path, package=_PACKAGE).schemas[_CONTAINER_KEY]
    assert ".butler_info.dataset" in coverage.absent
    assert ".butler_info.dataset" not in coverage.absent_roots
    assert ".butler_info" in coverage.absent_roots


def test_retired_fixtures_do_not_credit_coverage(tmp_path: Path) -> None:
    """Verify a retired fixture contributes nothing.

    A retired fixture is expected not to validate, so counting it would credit
    coverage to a shape current code rejects.
    """
    _write(tmp_path, _Leaf(tag="x"), retired=True)
    coverage = schema_coverage(tmp_path, package=_PACKAGE).schemas[_LEAF_KEY]
    assert not coverage.sources
    assert ".tag" in coverage.absent


def test_schema_with_no_fixture_reports_everything_absent(tmp_path: Path) -> None:
    """Verify a schema nothing reaches is reported rather than omitted."""
    report = schema_coverage(tmp_path, package=_PACKAGE)
    coverage = report.schemas[_OTHER_KEY]
    assert not coverage.sources
    assert not coverage.expressed
    assert "no fixture reaches this schema" in format_coverage_report(report, schema=coverage.name)


def test_format_marks_gaps_and_full_positions_differently(tmp_path: Path) -> None:
    """Verify the text output distinguishes a reached position from a gap."""
    _write(tmp_path, _Container(chosen=_Leaf()))
    report = schema_coverage(tmp_path, package=_PACKAGE)
    text = format_coverage_report(report, schema="coverage_container")
    assert "gap    .chosen [coverage_leaf] of {coverage_leaf, coverage_other_leaf}" in text
    assert "holds" not in text.split(".chosen")[0].splitlines()[-1]


def test_committed_tree_credits_a_sub_schema_from_a_container() -> None:
    """Verify the real fixture tree exercises cross-container attribution.

    Asserted structurally rather than by count: some schema in the committed
    tree is credited by a fixture that is not its own, which is what the
    aggregation exists to capture.
    """
    report = schema_coverage(FIXTURE_DIR)
    borrowed = {
        coverage.name: sorted(coverage.sources)
        for coverage in report.schemas.values()
        if any(not source.startswith(f"{coverage.name}-") for source in coverage.sources)
    }
    assert borrowed, "no schema is credited by a container's fixture"


@pytest.mark.parametrize("schema", ["cell_coadd", "visit_image"])
def test_committed_composites_report_their_sub_schema_positions(schema: str) -> None:
    """Verify a real composite reports positions it actually holds."""
    report = schema_coverage(FIXTURE_DIR)
    (coverage,) = [c for c in report.schemas.values() if c.name == schema]
    assert coverage.positions
    assert any(position.reached for position in coverage.positions)
