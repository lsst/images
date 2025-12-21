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

__all__ = (
    "ExtensionHDU",
    "FitsCompressionOptions",
    "FitsOpaqueMetadata",
    "InvalidFitsArchiveError",
)

import dataclasses

import astropy.io.fits

type ExtensionHDU = astropy.io.fits.ImageHDU | astropy.io.fits.CompImageHDU | astropy.io.fits.BinTableHDU


class InvalidFitsArchiveError(RuntimeError):
    """The error type raised when the content of a FITS file presumed to have
    been written by FitsOutputArchive is not self-consistent.
    """


@dataclasses.dataclass
class FitsCompressionOptions:
    """Configuration options for FITS compression."""

    # TODO: fill out this placeholder.  The (C++-implemented) class of the same
    # name in lsst.afw.fits might be a good model.


@dataclasses.dataclass
class FitsOpaqueMetadata:
    """Opaque metadata that may be carried around by a serializable type to
    propagate serialization options and opaque information without that type
    knowing how it was serialized.
    """

    headers: dict[str, astropy.io.fits.Header] = dataclasses.field(default_factory=dict)
    """FITS headers found (but not interpreted and stripped) when reading, to
    be propagated on write.

    Keys are EXTNAME values, or "" for the primary header.
    """

    compression: dict[str, FitsCompressionOptions] = dataclasses.field(default_factory=dict)
    """FITS compression options found on when reading, to be used again (by
    default) on write.
    """
