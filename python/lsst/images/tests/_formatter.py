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

__all__ = ("make_test_formatter",)

from typing import Any

from lsst.daf.butler import (
    DataCoordinate,
    DatasetRef,
    DatasetType,
    DimensionUniverse,
    FileDescriptor,
    FormatterV2,
    Location,
    StorageClass,
)

_UNIVERSE = DimensionUniverse()


def make_test_formatter[F: FormatterV2](
    formatter_cls: type[F],
    pytype: type,
    *,
    location: str = "/tmp/test.fits",
    parameters: dict[str, Any] | None = None,
    write_parameters: dict[str, Any] | None = None,
) -> F:
    """Construct a butler formatter wired to a minimal
    `~lsst.daf.butler.DatasetRef`.

    Intended for unit tests that exercise formatter logic without a full
    butler. ``pytype`` is wrapped in a fresh `~lsst.daf.butler.StorageClass`
    whose name is taken from the class.
    """
    storage_class = StorageClass(name=pytype.__name__, pytype=pytype)
    file_descriptor = FileDescriptor(Location(None, location), storage_class, parameters=parameters)
    dataset_type = DatasetType(
        pytype.__name__,
        dimensions=_UNIVERSE.empty,
        storageClass=storage_class,
    )
    ref = DatasetRef(dataset_type, DataCoordinate.make_empty(_UNIVERSE), run="test")
    return formatter_cls(
        file_descriptor=file_descriptor,
        ref=ref,
        write_parameters=write_parameters,
    )
