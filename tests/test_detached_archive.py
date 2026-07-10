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
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from lsst.images import VisitImage, VisitImageSerializationModel
from lsst.images.serialization import (
    ArchiveAccessRequiredError,
    ArchiveReadError,
    ArchiveTree,
    DetachedArchive,
    InvalidComponentError,
    open_archive,
    read_archive,
    write_archive,
)

LOCAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "schema_v1")


@pytest.fixture
def component_probe_env(tmp_path: Path) -> Iterator[tuple[str, VisitImage, DetachedArchive]]:
    """Return (tmpdir, visit_image, archive) for component-probe tests."""
    visit_image = read_archive(os.path.join(LOCAL_DATA_DIR, "visit_image.json"))
    yield str(tmp_path), visit_image, DetachedArchive()


def test_exception_hierarchy() -> None:
    """Verify ArchiveAccessRequiredError is a RuntimeError but not an
    ArchiveReadError.
    """
    assert issubclass(ArchiveAccessRequiredError, RuntimeError)
    # This is a control-flow signal, not a corrupt-file diagnosis: it
    # must never be swallowed by 'except ArchiveReadError' handlers
    # (e.g. the deferred-PSF handling in VisitImage full reads).
    assert not issubclass(ArchiveAccessRequiredError, ArchiveReadError)


# TODO[DM-54965]: many of the tests below are fragile, because they pass in
# None on the assumption that the archive isn't going to check an argument's
# type for correctness before exhibiting the behavior the test is trying to
# verify.  Running MyPy seems like a good way to flag these.


def test_deserialize_pointer_raises() -> None:
    """Verify DetachedArchive.deserialize_pointer raises
    ArchiveAccessRequiredError.
    """
    with pytest.raises(ArchiveAccessRequiredError):
        DetachedArchive().deserialize_pointer(None, ArchiveTree, lambda model, archive: None)


def test_get_frame_set_raises() -> None:
    """Verify DetachedArchive.get_frame_set raises
    ArchiveAccessRequiredError.
    """
    with pytest.raises(ArchiveAccessRequiredError):
        DetachedArchive().get_frame_set(None)


def test_get_array_raises() -> None:
    """Verify DetachedArchive.get_array raises ArchiveAccessRequiredError."""
    with pytest.raises(ArchiveAccessRequiredError):
        DetachedArchive().get_array(None)


def test_get_table_raises() -> None:
    """Verify DetachedArchive.get_table raises ArchiveAccessRequiredError."""
    with pytest.raises(ArchiveAccessRequiredError):
        DetachedArchive().get_table(None)


def test_get_structured_array_raises() -> None:
    """Verify DetachedArchive.get_structured_array raises
    ArchiveAccessRequiredError.
    """
    with pytest.raises(ArchiveAccessRequiredError):
        DetachedArchive().get_structured_array(None)


def test_get_opaque_metadata_is_none() -> None:
    """Verify DetachedArchive.get_opaque_metadata returns None."""
    # A detached probe has no file to take opaque metadata from.
    assert DetachedArchive().get_opaque_metadata() is None


def _get_tree(visit_image: VisitImage, tmpdir: str, extension: str) -> VisitImageSerializationModel[Any]:
    """Write the fixture in the given format and return its tree.

    The tree is deliberately used after the reader is closed, mirroring
    how the formatter cache holds a tree with no open file.
    """
    path = os.path.join(tmpdir, "visit_image" + extension)
    write_archive(visit_image, path)
    with open_archive(path) as reader:
        return cast(VisitImageSerializationModel[Any], reader.get_tree())


@pytest.mark.parametrize(
    "component",
    [
        "sky_projection",
        "psf",
        "obs_info",
        "summary_stats",
        "detector",
        "aperture_corrections",
        "backgrounds",
        "band",
        "bbox",
    ],
)
def test_free_components(
    component: str, component_probe_env: tuple[str, VisitImage, DetachedArchive]
) -> None:
    """Verify each free component deserializes without file access via
    DetachedArchive.
    """
    tmpdir, visit_image, archive = component_probe_env
    tree = _get_tree(visit_image, tmpdir, ".fits")
    value = tree.deserialize_component(component, archive)
    assert value is not None


@pytest.mark.parametrize(
    "component",
    ["image", "mask", "variance"],
)
def test_pixel_components_need_file(
    component: str,
    component_probe_env: tuple[str, object, DetachedArchive],
) -> None:
    """Test that pixel components raise ArchiveAccessRequiredError via
    DetachedArchive.
    """
    tmpdir, visit_image, archive = component_probe_env
    tree = _get_tree(visit_image, tmpdir, ".fits")
    with pytest.raises(ArchiveAccessRequiredError):
        tree.deserialize_component(component, archive)


def test_json_pixel_components_need_file(
    component_probe_env: tuple[str, object, DetachedArchive],
) -> None:
    """Test that inline JSON arrays still require file access for pixel
    components.

    Inline arrays in a JSON tree still go through archive.get_array,
    so pixel components fall back to the file even for .json.
    """
    tmpdir, visit_image, archive = component_probe_env
    tree = _get_tree(visit_image, tmpdir, ".json")
    with pytest.raises(ArchiveAccessRequiredError):
        tree.deserialize_component("image", archive)


def test_invalid_component_propagates(
    component_probe_env: tuple[str, object, DetachedArchive],
) -> None:
    """Test that that deserializing an unknown component raises
    InvalidComponentError.
    """
    tmpdir, visit_image, archive = component_probe_env
    tree = _get_tree(visit_image, tmpdir, ".fits")
    with pytest.raises(InvalidComponentError):
        tree.deserialize_component("not_a_component", archive)
