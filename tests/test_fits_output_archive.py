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
import numpy as np
import pydantic

from lsst.images.fits import FitsOutputArchive
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
