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

from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import astropy.io.fits
import astropy.table
import numpy as np
import pydantic

from lsst.images.fits import FitsInputArchive, FitsOutputArchive
from lsst.images.serialization import ArchiveTree, InputArchive


class _TinyTree(ArchiveTree):
    """Minimal concrete ArchiveTree for low-level archive writes."""

    SCHEMA_NAME: ClassVar[str] = "test_fits_output_archive"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = object

    def deserialize(
        self, archive: InputArchive[Any], **kwargs: Any
    ) -> _TinyTree:  # pragma: no cover - never invoked
        raise NotImplementedError()


class _PointerTarget(pydantic.BaseModel):
    """A trivial pointer-target model holding an array reference."""

    data: dict[str, Any] | None = None


def _write_archive(body: Callable[[FitsOutputArchive], None], tmp_path: Path) -> list[tuple[str, int | None]]:
    """Write an archive, applying ``body`` to it, and return the
    ``(EXTNAME, EXTVER)`` pairs of the resulting extension HDUs.
    """
    filename = tmp_path / "test.fits"
    with FitsOutputArchive.open(filename) as archive:
        body(archive)
        archive.add_tree(_TinyTree())
    with astropy.io.fits.open(filename) as hdu_list:
        return [
            (hdu.header["EXTNAME"], hdu.header.get("EXTVER"))
            for hdu in hdu_list[1:]
            if hdu.header.get("EXTNAME") not in ("JSON", "INDEX")
        ]


def test_repeated_direct_names_get_increasing_extver(tmp_path: Path) -> None:
    """Verify repeated direct names get increasing EXTVER disambiguation."""
    array = np.zeros((2, 2), dtype=np.float32)
    sources = []

    def body(archive: FitsOutputArchive) -> None:
        sources.append(archive.add_array(array, name="data").source)
        sources.append(archive.add_array(array, name="data").source)

    keys = _write_archive(body, tmp_path)
    assert sources == ["fits:DATA", "fits:DATA,2"]
    assert keys == [("DATA", None), ("DATA", 2)]


def test_direct_and_pointer_target_names_do_not_collide(tmp_path: Path) -> None:
    """Verify a direct name and a pointer target's nested name do not
    collide.
    """
    # A direct name and a pointer target's nested name (registered with
    # a leading slash because the pointer's nested archive is rooted at
    # "") already produce distinct EXTNAMEs, so neither needs EXTVER
    # disambiguation.
    array = np.zeros((2, 2), dtype=np.float32)
    sources = []

    def serializer(archive: FitsOutputArchive):
        ref = archive.add_array(array, name="data")
        sources.append(ref.source)
        return _PointerTarget(data=ref.model_dump())

    def body(archive: FitsOutputArchive):
        sources.append(archive.add_array(array, name="data").source)
        archive.serialize_pointer("psf", serializer, key="psf-key")  # type: ignore[arg-type]

    keys = _write_archive(body, tmp_path)
    assert sources == ["fits:DATA", "fits:/DATA"]
    assert keys == [("DATA", None), ("/DATA", None)]


def test_table_read_is_native_byte_order(tmp_path: Path) -> None:
    """Verify that tables read back from FITS (always big-endian on disk)
    are in native byte order, with scaled and logical columns intact.
    """
    table = astropy.table.Table(
        {
            "f": np.array([1.5, 2.5]),
            "i": np.array([1, 2], dtype=np.int32),
            # Stored as a signed column with TZERO; naive byte swapping of
            # the raw storage loses the offset.
            "u": np.array([1, 2**31 + 5], dtype=np.uint32),
            "b": np.array([True, False]),
            "v": np.arange(6, dtype=np.float32).reshape(2, 3),
        }
    )
    filename = tmp_path / "table.fits"
    with FitsOutputArchive.open(filename) as archive:
        model = archive.add_table(table, name="t")
        archive.add_tree(_TinyTree())
    with FitsInputArchive.open(filename) as archive:
        array = archive.get_structured_array(model)
        read_table = archive.get_table(model)
    assert array.dtype.isnative
    for name in table.colnames:
        assert read_table[name].dtype.isnative, name
        np.testing.assert_array_equal(array[name], table[name], err_msg=name)
        np.testing.assert_array_equal(read_table[name], table[name], err_msg=name)
    assert array["u"].dtype == np.uint32
    assert array["b"].dtype == np.bool_
