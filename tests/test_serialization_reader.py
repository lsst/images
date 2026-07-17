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

import builtins
import contextlib
import os
from pathlib import Path

import pytest

from lsst.images import VisitImage
from lsst.images import fits as images_fits
from lsst.images import json as images_json
from lsst.images.fits import FitsInputArchive
from lsst.images.json import JsonInputArchive
from lsst.images.serialization import ArchiveTree, read_archive

try:
    import h5py  # noqa: F401

    from lsst.images import ndf as images_ndf
    from lsst.images.ndf import NdfInputArchive

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False

skip_no_h5py = pytest.mark.skipif(not HAVE_H5PY, reason="h5py is not installed")

LOCAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "schema_v1")


@pytest.fixture(scope="session")
def visit_image() -> VisitImage:
    """Return a VisitImage loaded once from the committed JSON fixture."""
    return read_archive(os.path.join(LOCAL_DATA_DIR, "visit_image.json"))  # type: ignore[return-value]


@contextlib.contextmanager
def count_opens(path: Path | str):
    """Return a one-element list that counts how many times ``path`` is
    opened.
    """
    count = [0]
    real_open = builtins.open

    def counting_open(file, *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)) and os.fspath(file) == os.fspath(path):
            count[0] += 1
        return real_open(file, *args, **kwargs)

    builtins.open = counting_open
    try:
        yield count
    finally:
        builtins.open = real_open


def test_fits_open_tree_yields_archive_tree_and_info(visit_image: VisitImage, tmp_path: Path) -> None:
    """Verify FitsInputArchive.open_tree yields a live archive/tree/info
    triple.
    """
    path = tmp_path / "v.fits"
    images_fits.write(visit_image, path)
    with FitsInputArchive.open_tree(path) as (archive, tree, info):
        assert isinstance(tree, ArchiveTree)
        assert info.schema_name == "visit_image"
        proj = tree.deserialize_component("sky_projection", archive)
        assert proj is not None


def test_fits_read_still_works(visit_image: VisitImage, tmp_path: Path) -> None:
    """Verify read_archive() returns the deserialized object directly
    from a FITS file.
    """
    path = tmp_path / "v.fits"
    images_fits.write(visit_image, path)
    result = read_archive(path)
    assert type(result).__name__ == "VisitImage"


@skip_no_h5py
def test_ndf_open_tree_yields_archive_tree_and_info(visit_image: VisitImage, tmp_path: Path) -> None:
    """Verify NdfInputArchive.open_tree yields a live archive/tree/info
    triple.
    """
    path = tmp_path / "v.sdf"
    images_ndf.write(visit_image, path)
    with NdfInputArchive.open_tree(path) as (archive, tree, info):
        assert isinstance(tree, ArchiveTree)
        assert info.schema_name == "visit_image"
        assert tree.deserialize_component("obs_info", archive) is not None


@skip_no_h5py
def test_ndf_read_still_works(visit_image: VisitImage, tmp_path: Path) -> None:
    """Verify read_archive() returns the deserialized object directly
    from an NDF file.
    """
    path = tmp_path / "v.sdf"
    images_ndf.write(visit_image, path)
    assert type(read_archive(path)).__name__ == "VisitImage"


def test_json_open_tree_yields_archive_tree_and_info(visit_image: VisitImage, tmp_path: Path) -> None:
    """Verify JsonInputArchive.open_tree yields a live archive/tree/info
    triple.
    """
    path = tmp_path / "v.json"
    images_json.write(visit_image, path)
    with JsonInputArchive.open_tree(path) as (archive, tree, info):
        assert isinstance(tree, ArchiveTree)
        assert info.schema_name == "visit_image"
        assert tree.deserialize_component("sky_projection", archive) is not None


def test_json_read_still_works(visit_image: VisitImage, tmp_path: Path) -> None:
    """Verify read_archive() returns the deserialized object directly
    from a JSON file.
    """
    path = tmp_path / "v.json"
    images_json.write(visit_image, path)
    assert type(read_archive(path)).__name__ == "VisitImage"


def _check_components_and_read(path: Path | str) -> None:
    """Assert that serialization.open_archive() exposes components and a
    full read on ``path``.
    """
    import lsst.images.serialization as ser

    with ser.open_archive(path) as reader:
        assert reader.get_component("sky_projection") is not None
        assert reader.get_component("obs_info") is not None
        full = reader.read()
        assert type(full).__name__ == "VisitImage"


def test_reader_api_components_and_read_fits(visit_image: VisitImage, tmp_path: Path) -> None:
    """Verify the Reader API exposes components and a full read for FITS."""
    path = tmp_path / "v.fits"
    images_fits.write(visit_image, path)
    _check_components_and_read(path)


def test_reader_api_components_and_read_json(visit_image: VisitImage, tmp_path: Path) -> None:
    """Verify the Reader API exposes components and a full read for JSON."""
    path = tmp_path / "v.json"
    images_json.write(visit_image, path)
    _check_components_and_read(path)


@skip_no_h5py
def test_reader_api_components_and_read_ndf(visit_image: VisitImage, tmp_path: Path) -> None:
    """Verify the Reader API exposes components and a full read for NDF."""
    path = tmp_path / "v.sdf"
    images_ndf.write(visit_image, path)
    _check_components_and_read(path)


def test_reader_api_info(visit_image: VisitImage, tmp_path: Path) -> None:
    """Verify the Reader API exposes correct schema info for a FITS file."""
    import lsst.images.serialization as ser

    path = tmp_path / "v.fits"
    images_fits.write(visit_image, path)
    with ser.open_archive(path) as reader:
        assert reader.info.schema_name == "visit_image"
        assert reader.info.schema_version == "1.0.0.dev0"
        assert isinstance(reader.metadata, dict)


def test_reader_api_cls_match(visit_image: VisitImage, tmp_path: Path) -> None:
    """Verify the Reader API accepts cls= when the type matches."""
    import lsst.images.serialization as ser
    from lsst.images import VisitImage

    path = tmp_path / "v.fits"
    images_fits.write(visit_image, path)
    with ser.open_archive(path, cls=VisitImage) as reader:
        assert isinstance(reader.read(), VisitImage)


def test_reader_api_cls_mismatch_raises(visit_image: VisitImage, tmp_path: Path) -> None:
    """Verify the Reader API raises TypeError when cls= does not match the
    schema.
    """
    import lsst.images.serialization as ser
    from lsst.images import Mask

    path = tmp_path / "v.fits"
    images_fits.write(visit_image, path)
    with pytest.raises(TypeError):
        with ser.open_archive(path, cls=Mask):
            pass


def test_reader_api_unknown_component(visit_image: VisitImage, tmp_path: Path) -> None:
    """Verify the Reader API raises InvalidComponentError for an unknown
    component name.
    """
    import lsst.images.serialization as ser
    from lsst.images.serialization import InvalidComponentError

    path = tmp_path / "v.fits"
    images_fits.write(visit_image, path)
    with ser.open_archive(path) as reader:
        with pytest.raises(InvalidComponentError):
            reader.get_component("does_not_exist")


def test_reader_api_use_after_close_raises(visit_image: VisitImage, tmp_path: Path) -> None:
    """Verify the Reader API raises RuntimeError when used after closing."""
    import lsst.images.serialization as ser

    path = tmp_path / "v.fits"
    images_fits.write(visit_image, path)
    with ser.open_archive(path) as reader:
        pass
    with pytest.raises(RuntimeError):
        reader.get_component("sky_projection")


def test_fits_open_reads_file_once(visit_image: VisitImage, tmp_path: Path) -> None:
    """Verify serialization.open_archive() opens a FITS file exactly once
    regardless of component reads.
    """
    import lsst.images.serialization as ser

    path = tmp_path / "v.fits"
    images_fits.write(visit_image, path)
    with count_opens(path) as count:
        with ser.open_archive(path) as reader:
            reader.get_component("sky_projection")
            reader.get_component("obs_info")
    assert count[0] == 1
