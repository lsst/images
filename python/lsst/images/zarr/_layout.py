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

"""Per-archive-class layout rules for the zarr backend.

This module centralises the decisions that vary by image type:

- which OME axes apply (``ColorImage`` has no root multiscale)
- default chunk sizes (clamped to ``DEFAULT_CHUNK_AXIS_LIMIT`` per axis,
  image-aligned for `variance` / `mask` siblings)
- the affine residual validator that gates the OME
  ``coordinateTransformations`` block

Keeping these in one place lets the output archive populate the IR
generically.
"""

from __future__ import annotations

__all__ = (
    "AffineCheckResult",
    "affine_check",
    "axes_for_archive_class",
    "chunks_aligned_to",
    "chunks_for",
    "decorate_sub_archives",
    "default_shards",
    "deserialize_fits_opaque_metadata",
    "serialize_fits_opaque_metadata",
)

import math
from dataclasses import dataclass
from typing import Any

import astropy.io.fits
import numpy as np

from ..fits._common import ExtensionKey, FitsOpaqueMetadata
from ._common import DEFAULT_CHUNK_AXIS_LIMIT
from ._model import OmeMultiscale, ZarrArray, ZarrDocument


def axes_for_archive_class(name: str) -> tuple[str, ...]:
    """Return the OME axis tuple for a given archive class.

    Returns an empty tuple for ``ColorImage`` to signal that there is
    no OME multiscale at the root of that class — the per-channel
    sub-archives carry their own ``(y, x)`` multiscales.

    Parameters
    ----------
    name
        Archive class name (e.g. ``"ColorImage"``).
    """
    if name == "ColorImage":
        return ()
    return ("y", "x")


def chunks_for(
    shape: tuple[int, ...],
    override: tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Return the default chunk shape for a top-level array.

    This is the fallback used when neither an explicit override nor an
    ``add_array`` ``tile_shape`` hint applies; cell-aligned chunks for a
    `CellCoadd` arrive via that ``tile_shape`` hint instead.

    Parameters
    ----------
    shape
        The full array shape, used to clamp the default per-axis.
    override
        User-supplied chunk shape. If not ``None`` it is returned
        verbatim after a length check.
    """
    if override is not None:
        if len(override) != len(shape):
            raise ValueError(f"chunks override has rank {len(override)}, expected {len(shape)}.")
        return tuple(override)
    return tuple(min(DEFAULT_CHUNK_AXIS_LIMIT, dim) for dim in shape)


def chunks_aligned_to(
    *,
    image_chunks: tuple[int, ...],
    shape: tuple[int, ...],
) -> tuple[int, ...]:
    """Derive a sibling array's chunks from the ``image`` array's chunks.

    Used by `ZarrOutputArchive.add_array` for ``variance`` and
    ``mask`` siblings when the user has not provided an explicit
    override. The result is per-axis ``min(image_chunks[i],
    shape[i])`` so a sibling smaller than ``image`` is not
    over-chunked.

    Parameters
    ----------
    image_chunks
        Chunk shape of the ``image`` array to align the sibling to.
    shape
        The sibling array's shape; each axis caps the aligned chunk.
    """
    if len(image_chunks) != len(shape):
        raise ValueError(
            f"image_chunks rank {len(image_chunks)} does not match sibling shape rank {len(shape)}."
        )
    return tuple(min(c, dim) for c, dim in zip(image_chunks, shape, strict=True))


def default_shards(
    *,
    chunks: tuple[int, ...],
    shape: tuple[int, ...],
    dtype: np.dtype,
    target_bytes: int,
) -> tuple[int, ...] | None:
    """Derive a default shard shape from ``chunks``, ``shape``, and ``dtype``.

    Returns ``None`` when sharding would be a no-op: ``dtype.itemsize``
    is zero (object dtypes), the array is already a single chunk per
    axis, the chunk is already at least ``target_bytes`` big, or the
    byte budget rounds to ``k == 1`` chunks per growable axis.

    The rule grows only axes whose ``chunks[i] < shape[i]`` (the
    others already cover the full extent), uses one uniform multiplier
    ``k = round(ratio ** (1 / num_growable_axes))`` to stay close to
    the byte budget, and caps each axis at ``chunks[i] * ceil(shape[i]
    / chunks[i])`` so a small array does not get a shard larger than
    itself. Every shard axis is an integer multiple of the
    corresponding chunk axis, as required by zarr v3.

    Parameters
    ----------
    chunks
        Chunk shape, one int per axis.
    shape
        Array shape, one int per axis.
    dtype
        Array dtype; only ``itemsize`` is consulted.
    target_bytes
        Target uncompressed shard size. Typically
        `~lsst.images.zarr._common.DEFAULT_TARGET_SHARD_BYTES`.

    Raises
    ------
    ValueError
        If ``len(chunks) != len(shape)``.
    """
    if len(chunks) != len(shape):
        raise ValueError(f"chunks rank {len(chunks)} does not match shape rank {len(shape)}.")
    itemsize = dtype.itemsize
    if itemsize == 0:
        return None
    chunk_bytes = math.prod(chunks) * itemsize
    if chunk_bytes >= target_bytes:
        return None
    growable = [i for i in range(len(shape)) if chunks[i] < shape[i]]
    if not growable:
        return None
    ratio = target_bytes / chunk_bytes
    k = round(ratio ** (1.0 / len(growable)))
    if k <= 1:
        return None  # budget allows at most a 1x multiplier — no-op shard
    shard = list(chunks)
    for i in growable:
        n_chunks_axis = math.ceil(shape[i] / chunks[i])
        shard[i] = min(chunks[i] * k, chunks[i] * n_chunks_axis)
    return tuple(shard)


@dataclass
class AffineCheckResult:
    """Result of asking AST whether a simplified affine fits a full WCS.

    When ``dropped`` is False, ``coordinate_transformations`` is the
    OME-NGFF ``coordinateTransformations`` list to emit. When True,
    AST could not find a linear approximation that stays within the
    requested per-pixel tolerance over the whole image footprint, and
    the caller must omit the block (or emit a unit scale only).
    """

    dropped: bool
    coordinate_transformations: list[dict[str, Any]] | None


def affine_check(
    *,
    frame_set: Any,
    image_shape: tuple[int, int],
    max_residual_pixels: float = 1.0,
) -> AffineCheckResult:
    """Build an OME affine ``coordinateTransformations`` from ``frame_set``.

    Delegates to AST's ``linearapprox`` over the image footprint with
    a tolerance scaled to ``max_residual_pixels`` of pixel-equivalent
    error. AST returns the affine coefficients when the approximation
    fits and ``None`` otherwise.

    Parameters
    ----------
    frame_set
        AST FrameSet whose base→current mapping goes from pixel
        coordinates to sky.
    image_shape
        ``(h, w)`` of the image; used as the bounds of the box AST is
        asked to approximate over.
    max_residual_pixels
        Maximum permitted deviation, in pixels, of any point in the
        box from the linear prediction. AST is given the equivalent
        threshold in output (sky) units after multiplying by the local
        pixel scale.
    """
    h, w = image_shape
    mapping = frame_set.getMapping(frame_set.base, frame_set.current)

    # Local pixel scale near the image origin: convert the user-supplied
    # pixel tolerance into the output-coordinate units AST expects.
    sample = _frame_set_apply(frame_set, np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    origin = sample[0]
    pixel_scale_axis0 = float(np.linalg.norm(sample[1] - origin))
    pixel_scale_axis1 = float(np.linalg.norm(sample[2] - origin))
    pixel_scale = float(np.sqrt(pixel_scale_axis0 * pixel_scale_axis1))
    if pixel_scale <= 0.0:
        return AffineCheckResult(dropped=True, coordinate_transformations=None)

    tol_output = max_residual_pixels * pixel_scale
    try:
        coeffs = mapping.linearApprox(
            [0.0, 0.0],
            [float(max(h - 1, 0)), float(max(w - 1, 0))],
            tol_output,
        )
    except RuntimeError:
        # AST signals "no linear approximation fits within tol over the
        # footprint" by raising; ``_ast`` normalises both backends to that.
        return AffineCheckResult(dropped=True, coordinate_transformations=None)

    # ``linearApprox`` returns a (1 + Nout, Nin) array: row 0 holds the
    # per-output constant offsets and the remaining rows are the Jacobian
    # ``J[i][j] = ∂out_i/∂in_j``. For a 2-D pixel→sky mapping that's a
    # (3, 2) array.
    fit = np.asarray(coeffs, dtype=float)
    if fit.shape != (3, 2):
        raise ValueError(
            f"linearApprox returned shape {fit.shape}; expected (3, 2) for a 2-D pixel→sky mapping."
        )
    c0, c1 = (float(x) for x in fit[0])
    (j00, j01), (j10, j11) = (float(x) for x in fit[1]), (float(x) for x in fit[2])

    # Pixel scale per input axis: length of the corresponding Jacobian
    # column in output coordinates.
    scale_axis0 = float(np.hypot(j00, j10))
    scale_axis1 = float(np.hypot(j01, j11))

    # NGFF composes ``coordinateTransformations`` in list order: the
    # scale is applied first, then the affine. To avoid double-counting
    # the pixel-size factor, normalise each Jacobian column by its
    # length so the affine carries only the rotation / shear that the
    # scale does not capture. ``pixel_scale`` is the geometric mean of
    # the two column norms; if it were zero we'd already have returned
    # above, so dividing by ``scale_axis*`` is safe here.
    j00_n = j00 / scale_axis0
    j10_n = j10 / scale_axis0
    j01_n = j01 / scale_axis1
    j11_n = j11 / scale_axis1
    affine_matrix = [[j00_n, j01_n, c0], [j10_n, j11_n, c1], [0.0, 0.0, 1.0]]

    coordinate_transformations: list[dict[str, Any]] = [
        {"type": "scale", "scale": [scale_axis0, scale_axis1]},
        {"type": "affine", "affine": affine_matrix},
    ]
    return AffineCheckResult(
        dropped=False,
        coordinate_transformations=coordinate_transformations,
    )


def _frame_set_apply(frame_set: Any, pixels: Any) -> Any:
    """Apply ``frame_set``'s base->current mapping to a (N, 2) pixel array."""
    pixels = np.asarray(pixels, dtype=float)
    mapping = frame_set.getMapping(frame_set.base, frame_set.current)
    # astshim's pybind11 bindings require a C-contiguous array; ``pixels.T``
    # is an F-contiguous view, so copy it into C order before passing it in.
    out = mapping.applyForward(np.ascontiguousarray(pixels.T))
    return np.asarray(out).T


def decorate_sub_archives(document: ZarrDocument) -> None:
    """Decorate sub-archive groups with ``lsst.archive_class`` and OME attrs.

    A sub-archive is any group below the root that contains an
    ``image`` array. Decoration adds ``lsst.archive_class = "Image"``
    and an ``ome.multiscales`` block pointing at the sub-archive's
    ``image`` array. Recursive: nested sub-archives are decorated too.

    The root group is left alone — its ``lsst.archive_class`` is set
    by ``add_tree`` based on the in-memory object's type.

    Parameters
    ----------
    document
        IR document whose sub-archive groups are decorated in place.
    """
    if not isinstance(document, ZarrDocument):
        raise TypeError(type(document).__name__)
    _decorate_walk(document.root)


def _decorate_walk(group: Any) -> None:
    for sub in group.groups.values():
        if "image" in sub.arrays:
            sub.attributes.lsst.setdefault("archive_class", "Image")
            if "lsst_json" in sub.arrays:
                sub.attributes.lsst.setdefault("json", "lsst_json")
            if "multiscales" not in sub.attributes.ome:
                multiscale = OmeMultiscale(
                    name="image",
                    axes=("y", "x"),
                    dataset_path="image",
                )
                sub.attributes.ome["multiscales"] = [multiscale.dump()]
        _decorate_walk(sub)


def serialize_fits_opaque_metadata(document: ZarrDocument, opaque: FitsOpaqueMetadata) -> None:
    """Stage a `FitsOpaqueMetadata` object into the IR.

    Stores the primary-HDU header as a 2-D ``(N, 80)`` ``uint8`` array
    at ``/lsst/opaque_metadata/fits/primary`` — one row per FITS card,
    one column per character — and sets ``lsst.opaque_metadata_format
    = "fits"`` on the root group. The bytes are
    ``astropy.io.fits.Header.tostring()`` output verbatim (cards +
    ``END`` + padding to a 2880-byte block), so the round-trip is
    byte-exact and preserves comments, ``HISTORY``, ``COMMENT``,
    ``CONTINUE``, and ``HIERARCH`` cards. No-op if the metadata is
    empty or missing a primary header.

    Parameters
    ----------
    document
        IR document the header array is staged into.
    opaque
        FITS opaque metadata whose primary header is staged.
    """
    primary = opaque.headers.get(ExtensionKey())
    if primary is None or len(primary) == 0:
        return
    text = primary.tostring()
    if len(text) % 80 != 0:
        raise ValueError(
            f"Header.tostring() returned {len(text)} bytes; expected a "
            "multiple of 80 (one 80-char FITS card per row)."
        )
    n_cards = len(text) // 80
    cards = np.ascontiguousarray(np.frombuffer(text.encode("ascii"), dtype=np.uint8).reshape(n_cards, 80))
    parent = document.root.ensure_group("/lsst/opaque_metadata/fits")
    # Single chunk: the header is always read whole.
    ir_array = ZarrArray(data=cards, chunks=cards.shape)
    ir_array.attributes.extra["_ARRAY_DIMENSIONS"] = ["card", "char"]
    parent.arrays["primary"] = ir_array
    document.root.attributes.lsst["opaque_metadata_format"] = "fits"


def deserialize_fits_opaque_metadata(document: ZarrDocument) -> FitsOpaqueMetadata | None:
    """Reconstruct a `FitsOpaqueMetadata` from the IR, or return None.

    Returns ``None`` when the archive does not have a FITS opaque
    metadata block (the common case for archives that originated as
    native zarr). ``Header.fromstring`` parses cards up to the ``END``
    marker and drops the padding, so the recovered header carries
    only the real cards.

    Parameters
    ----------
    document
        IR document to recover the FITS opaque metadata from.
    """
    if document.root.attributes.lsst.get("opaque_metadata_format") != "fits":
        return None
    try:
        node = document.root.get("/lsst/opaque_metadata/fits/primary")
    except KeyError:
        return None
    if not isinstance(node, ZarrArray):
        return None
    text = bytes(node.read()).decode("ascii")
    header = astropy.io.fits.Header.fromstring(text)
    opaque = FitsOpaqueMetadata()
    opaque.headers[ExtensionKey()] = header
    return opaque
