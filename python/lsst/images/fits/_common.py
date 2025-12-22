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
    "FitsCompressionAlgorithm",
    "FitsCompressionOptions",
    "FitsDitherAlgorithm",
    "FitsOpaqueMetadata",
    "FitsQuantizationOptions",
    "InvalidFitsArchiveError",
)

import dataclasses
import enum
from typing import ClassVar

import astropy.io.fits
import numpy as np

type ExtensionHDU = astropy.io.fits.ImageHDU | astropy.io.fits.CompImageHDU | astropy.io.fits.BinTableHDU


class InvalidFitsArchiveError(RuntimeError):
    """The error type raised when the content of a FITS file presumed to have
    been written by FitsOutputArchive is not self-consistent.
    """


class FitsCompressionAlgorithm(enum.StrEnum):
    """FITS compression algorithms supported by this package.

    See the FITS standard for definitions.
    """

    GZIP_1 = "GZIP_1"
    GZIP_2 = "GZIP_2"
    RICE_1 = "RICE_1"


class FitsDitherAlgorithm(enum.StrEnum):
    """FITS quantization dither algorithms supported by this package.

    See the FITS standard for definitions.
    """

    NO_DITHER = "NO_DITHER"
    SUBTRACTIVE_DITHER_1 = "SUBTRACTIVE_DITHER_1"
    SUBTRACTIVE_DITHER_2 = "SUBTRACTIVE_DITHER_2"

    def to_astropy_quantize_method(self) -> int:
        """Convert to the integer code used by Astropy."""
        match self:
            case self.NO_DITHER:
                return -1
            case self.SUBTRACTIVE_DITHER_1:
                return 1
            case self.SUBTRACTIVE_DITHER_2:
                return 2
        raise AssertionError("Invalid enum value.")


@dataclasses.dataclass(frozen=True)
class FitsQuantizationOptions:
    """Quantization options for FITS compression."""

    dither: FitsDitherAlgorithm
    """How to add random noise during quantization to reduce biases."""

    level: float
    """Quantization level.

    When positive, this is the fraction of the measured standard deviation that
    corresponds to an integer step.  When negative, it is ``-ZSCALE``, the
    scaling to apply directly to the original pixels before quantization.
    """

    seed: int
    """Random number seed to use for dithering.

    Values between 1 and 10000 (inclusive) are used directly.  ``0`` will
    generate a value from the current time, and ``-1`` will generate a value
    from the checksum of the image.
    """


@dataclasses.dataclass(frozen=True)
class FitsCompressionOptions:
    """Configuration options for FITS compression."""

    algorithm: FitsCompressionAlgorithm = FitsCompressionAlgorithm.GZIP_2
    """Compression algorithm to use."""

    tile_shape: tuple[int, ...] | None = None
    """Shape ``(..., y, x)`` of independently compressed tiles.

    The default of `None` compresses each row separately.
    """

    quantization: FitsQuantizationOptions | None = None
    """Quantization to apply before compression, if any."""

    DEFAULT: ClassVar[FitsCompressionOptions | None]
    """Default compression options (lossless ``GZIP_2``)."""

    LOSSY: ClassVar[FitsCompressionOptions]
    """Default lossy compression options."""

    def make_hdu(self, data: np.ndarray, name: str) -> astropy.io.fits.CompImageHDU:
        """Make an `astropy.io.fits.CompImageHDU` object from these options."""
        if self.quantization is not None:
            return astropy.io.fits.CompImageHDU(
                data,
                name=name,
                compression_type=self.algorithm.value,
                tile_shape=self.tile_shape,
                quantize_method=self.quantization.dither.to_astropy_quantize_method(),
                quantize_level=self.quantization.level,
                dither_seed=self.quantization.seed,
            )
        else:
            return astropy.io.fits.CompImageHDU(
                data,
                name=name,
                compression_type=self.algorithm.value,
                tile_shape=self.tile_shape,
                quantize_level=0.0,
            )


FitsCompressionOptions.DEFAULT = FitsCompressionOptions()
FitsCompressionOptions.LOSSY = FitsCompressionOptions(
    algorithm=FitsCompressionAlgorithm.RICE_1,
    quantization=FitsQuantizationOptions(
        dither=FitsDitherAlgorithm.SUBTRACTIVE_DITHER_2, level=16.0, seed=-1
    ),
)


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
