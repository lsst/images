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
    "PrecompressedImage",
    "add_offset_wcs",
    "strip_wcs_cards",
)

import dataclasses
import enum
import itertools
import string
from typing import ClassVar, Self, final

import astropy.io.fits
import numpy as np

from .._geom import Box
from ..serialization import OpaqueArchiveMetadata

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


_COMPRESSION_KEYS = frozenset(
    (
        "ZIMAGE",
        "ZCMPTYPE",
        "ZBITPIX",
        "ZNAXIS",
        "ZMASKCMP",
        "ZQUANTIZ",
        "ZDITHER0",
        "ZCHECKSUM",
        "ZDATASUM",
    )
)
_COMPRESSION_PREFIX_KEYS = ("ZNAXIS", "ZTILE", "ZNAME", "ZVAL")


@dataclasses.dataclass
class PrecompressedImage:
    """Already-compressed FITS HDUs that are attached to high-level objects
    via `FitsOpaqueMetadata`, allowing lossy-compressed pixel values to be
    round-tripped exactly.
    """

    header: astropy.io.fits.Header
    """Header for the HDU.

    This contains only FITS tile-compression keywords.
    """

    data: astropy.io.fits.FITS_rec
    """FITS binary table data that serves as the low-level representation of a
    tile-compressed image HDU.
    """

    @classmethod
    def from_bintable(cls, hdu: astropy.io.fits.BinTableHDU) -> Self:
        """Construct from a binary table HDU.

        Parameters
        ----------
        hdu
            Binary table HDU, typically read from a FITS file opened with
            ``disable_image_compression=True``.

        Returns
        -------
        PrecompressedImage
            A `PrecompressedImage` instance.
        """
        header = astropy.io.fits.Header(
            [
                card
                for card in hdu.header.cards
                if card.keyword in _COMPRESSION_KEYS
                or any(card.keyword.startswith(k) for k in _COMPRESSION_PREFIX_KEYS)
            ]
        )
        # This is an opportunity to fix CFITSIO's non-standard writing of the
        # old RICE_ONE value instead of RICE_1.
        if header["ZCMPTYPE"] == "RICE_ONE":
            header["ZCMPTYPE"] = "RICE_1"
        return cls(header=header, data=hdu.data)


@final
@dataclasses.dataclass
class FitsOpaqueMetadata(OpaqueArchiveMetadata):
    """Opaque metadata that may be carried around by a serializable type to
    propagate serialization options and opaque information without that type
    knowing how it was serialized.
    """

    headers: dict[str, astropy.io.fits.Header] = dataclasses.field(default_factory=dict)
    """FITS headers found (but not interpreted and stripped) when reading, to
    be propagated on write.

    Keys are EXTNAME values, or "" for the primary header.  Header information
    in opaque metadata is considered immutable, allowing it to be transferred
    by reference to copies and subsets of the object it is attached to.
    """

    precompressed: dict[str, PrecompressedImage] = dataclasses.field(default_factory=dict)
    """FITS tile-compressed HDUs that should be written out directly instead
    of the in-memory data provided.

    Keys are EXTNAME values.  Precompressed pixel values are never copied or
    transferred to subsets.
    """

    def maybe_use_precompressed(self, name: str) -> astropy.io.fits.BinTableHDU | None:
        """Look up the given EXTNAME to see if there is a tile compressed image
        HDU that should be used directly, instead of requantizing.

        Parameters
        ----------
        name
            EXTNAME (all caps).

        Returns
        -------
        `astropy.io.fits.BinTableHDU` | `None`
            An already-compressed HDU, in binary table form, or `None` if there
            is no precompressed HDU for this EXTNAME.
        """
        if (precompressed := self.precompressed.get(name)) is None:
            return None
        return astropy.io.fits.BinTableHDU(precompressed.data, header=precompressed.header.copy(), name=name)

    def copy(self) -> FitsOpaqueMetadata:
        # Docstring inherited.
        return FitsOpaqueMetadata(headers=self.headers)

    def subset(self, bbox: Box) -> FitsOpaqueMetadata:
        # Docstring inherited.
        return FitsOpaqueMetadata(headers=self.headers)


def add_offset_wcs(header: astropy.io.fits.Header, *, x: int, y: int, key: str = "A") -> None:
    """Add a trivial FITS WCS to a header that applies the appropriate offset
    to map FITS array coordinates to a logical pixel grid.

    Parameters
    ----------
    header
        Header to update in-place.
    x
        Logical coordinate of the first column.
    y
        Logical coordinate of the first row.
    key
        Single-character suffix for this WCS.
    """
    header.set(f"CTYPE1{key}", "LINEAR")
    header.set(f"CTYPE2{key}", "LINEAR")
    header.set(f"CRPIX1{key}", 1.0)
    header.set(f"CRPIX2{key}", 1.0)
    header.set(f"CRVAL1{key}", float(x))
    header.set(f"CRVAL2{key}", float(y))
    header.set(f"CUNIT1{key}", "PIXEL")
    header.set(f"CUNIT2{key}", "PIXEL")


_WCS_VECTOR_KEYS = ("CUNIT", "CRPIX", "CRPIX", "CRVAL", "CRDELT", "CROTA", "CRDER", "CSYER")
_WCS_MATRIX_KEYS = ("CD{0}_{1}", "PC{0}_{1}")


def strip_wcs_cards(header: astropy.io.fits.Header) -> None:
    """Strip WCS cards from a FITS header.

    This does *not* attempt to cover all possible FITS WCS forms; it focuses on
    the ones we actually plan to write (simple undistorted ones + TAN-SIP).
    """
    wcsaxes = header.pop("WCSAXES", 2)
    for wcsname in [""] + list(string.ascii_uppercase):
        header.remove("RADESYS" + wcsname, ignore_missing=True)
        if "CTYPE1" + wcsname in header:
            ctype: str = ""  # just for linters that can't figure out that the loop always executes
            for n in range(wcsaxes):
                suffix = f"{n + 1}{wcsname}"
                ctype = header.pop("CTYPE" + suffix)
                for key in _WCS_VECTOR_KEYS:
                    header.remove(key + suffix, ignore_missing=True)
                for m in range(wcsaxes):
                    for tmpl in _WCS_MATRIX_KEYS:
                        header.remove(tmpl.format(m + 1, suffix), ignore_missing=True)
            if ctype.endswith("-SIP"):
                _strip_sip_poly(header, wcsname, "A")
                _strip_sip_poly(header, wcsname, "B")
                _strip_sip_poly(header, wcsname, "AP")
                _strip_sip_poly(header, wcsname, "BP")


def _strip_sip_poly(header: astropy.io.fits.Header, wcsname: str, which: str) -> None:
    order: int | None = header.pop(f"{which}_ORDER{wcsname}", None)
    if order is not None:
        for i, j in itertools.product(range(order + 1), range(order + 1)):
            header.remove(f"{which}_{i}_{j}{wcsname}", ignore_missing=True)
