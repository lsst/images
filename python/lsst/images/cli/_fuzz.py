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

__all__ = ("fuzz_masked_image", "shuffle_blocks")

from pathlib import Path
from typing import Any

import click
import numpy as np

from .._geom import YX
from ..fits import (
    FitsCompressionAlgorithm,
    FitsCompressionOptions,
    FitsDitherAlgorithm,
    FitsQuantizationOptions,
)
from ..serialization import ArchiveReadError, backend_for_path, read


def shuffle_blocks(
    image: np.ndarray,
    mask: np.ndarray,
    variance: np.ndarray,
    block_shape: YX[int],
    rng: np.random.Generator,
) -> None:
    """Shuffle image, mask, and variance pixels within each block in place.

    A single permutation is drawn for each block and applied to all three
    planes, so a pixel's values move together and stay mutually consistent.
    Blocks at the edges may be smaller than ``block_shape`` when its dimensions
    do not divide the image evenly.

    Parameters
    ----------
    image : `numpy.ndarray`
        The 2-d image plane, modified in place.
    mask : `numpy.ndarray`
        The mask plane with a leading ``(ny, nx)`` shape and any trailing axes
        (for example a per-pixel byte axis), modified in place.
    variance : `numpy.ndarray`
        The 2-d variance plane, modified in place.
    block_shape : `~lsst.images.YX`
        The ``(y, x)`` size of a single block.
    rng : `numpy.random.Generator`
        Random number generator used to permute pixels.
    """
    block_y, block_x = block_shape
    n_y, n_x = image.shape
    for y0 in range(0, n_y, block_y):
        y1 = min(y0 + block_y, n_y)
        for x0 in range(0, n_x, block_x):
            x1 = min(x0 + block_x, n_x)
            count = (y1 - y0) * (x1 - x0)
            permutation = rng.permutation(count)
            for plane in (image, mask, variance):
                block = plane[y0:y1, x0:x1]
                flat = block.reshape(count, *block.shape[2:])
                block[...] = flat[permutation].reshape(block.shape)


def _output_path(in_path: Path, suffix: str) -> Path:
    """Insert ``suffix`` before the file extension, preserving the recognised
    ``.fits.gz`` compound extension.
    """
    name = in_path.name
    if name.endswith(".fits.gz"):
        return in_path.with_name(name[: -len(".fits.gz")] + suffix + ".fits.gz")
    return in_path.with_name(in_path.stem + suffix + in_path.suffix)


def _block_shape(obj: Any, tile_override: tuple[int, int] | None) -> YX[int]:
    """Decide the shuffle block: the override, else the cell size when the
    object has a cell grid, else the full image.
    """
    if tile_override is not None:
        return YX(int(tile_override[0]), int(tile_override[1]))
    cell_shape = getattr(getattr(obj, "grid", None), "cell_shape", None)
    if cell_shape is not None:
        return YX(int(cell_shape.y), int(cell_shape.x))
    shape = obj.image.array.shape
    return YX(int(shape[0]), int(shape[1]))


def _compression_options(
    fits_tile: tuple[int, int] | None, quantize_level: float
) -> dict[str, FitsCompressionOptions]:
    """Build the per-plane compression profile keyed by logical options_name.

    ``image`` and ``variance`` (and noise realizations, which reuse the
    ``image`` profile) get lossy RICE with subtractive dithering; ``mask`` gets
    lossless GZIP.  Every other plane falls back to the lossless GZIP default.
    """
    lossy = FitsCompressionOptions(
        algorithm=FitsCompressionAlgorithm.RICE_1,
        tile_shape=fits_tile,
        quantization=FitsQuantizationOptions(
            dither=FitsDitherAlgorithm.SUBTRACTIVE_DITHER_2, level=quantize_level
        ),
    )
    gzip = FitsCompressionOptions(
        algorithm=FitsCompressionAlgorithm.GZIP_2, tile_shape=fits_tile, quantization=None
    )
    return {"image": lossy, "variance": lossy, "mask": gzip}


def _verify(in_path: Path, original: dict[str, np.ndarray], check: Any) -> None:
    """Confirm the shuffled planes actually changed in the re-read output."""
    for name in ("image", "variance"):
        orig = original[name]
        finite = np.isfinite(orig)
        if not np.any(finite) or np.ptp(orig[finite]) == 0:
            continue  # A constant plane cannot change under permutation.
        new = getattr(check, name).array
        changed = float(np.mean(new[finite] != orig[finite]))
        if changed < 0.5:
            raise click.ClickException(
                f"{name} plane barely changed ({changed:.1%}) after fuzzing {in_path}."
            )
    orig_mask = original["mask"]
    if np.ptp(orig_mask) != 0 and np.array_equal(check.mask.array, orig_mask):
        raise click.ClickException(f"Mask plane unchanged after fuzzing {in_path}.")


def _fuzz_file(
    in_path: Path,
    out_path: Path,
    *,
    seed: int,
    tile_override: tuple[int, int] | None,
    quantize_level: float,
    compression_seed: int,
) -> None:
    """Read one file, shuffle its proprietary planes, write it, and verify."""
    try:
        obj = read(str(in_path))
    except (ValueError, ArchiveReadError) as err:
        raise click.ClickException(f"Could not read {in_path}: {err}") from None

    block_shape = _block_shape(obj, tile_override)
    original = {
        "image": obj.image.array.copy(),
        "mask": obj.mask.array.copy(),
        "variance": obj.variance.array.copy(),
    }
    shuffle_blocks(
        obj.image.array, obj.mask.array, obj.variance.array, block_shape, np.random.default_rng(seed)
    )

    kwargs: dict[str, Any] = {}
    if backend_for_path(str(out_path)).name == "fits":
        # tile_shape stays None unless overridden, so each object's natural
        # tiling applies (cell for CellCoadd, astropy default otherwise); an
        # explicit override aligns the FITS tiles with the shuffle blocks.
        fits_tile = block_shape if tile_override is not None else None
        kwargs["compression_options"] = _compression_options(fits_tile, quantize_level)
        kwargs["compression_seed"] = compression_seed

    obj.write(str(out_path), **kwargs)
    _verify(in_path, original, read(str(out_path)))
    click.echo(f"Fuzzed {in_path} -> {out_path}")


@click.command(name="fuzz-masked-image")
@click.argument("files", nargs=-1, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--seed", type=int, default=1, show_default=True, help="Seed for pixel shuffling.")
@click.option(
    "--suffix",
    default=".fuzzed",
    show_default=True,
    help="Text inserted before the extension of each output file.",
)
@click.option(
    "--tile-shape",
    type=int,
    nargs=2,
    default=None,
    metavar="Y X",
    help="Override the shuffle block and FITS compression tile; default is the "
    "cell size if the object has a cell grid, else the full image.",
)
@click.option(
    "--quantize-level",
    type=float,
    default=16.0,
    show_default=True,
    help="RICE quantization level for the lossy image/variance planes.",
)
@click.option(
    "--compression-seed",
    type=int,
    default=1,
    show_default=True,
    help="FITS tile-compression dither seed (FITS output only).",
)
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing output files.")
def fuzz_masked_image(
    files: tuple[Path, ...],
    seed: int,
    suffix: str,
    tile_shape: tuple[int, int] | None,
    quantize_level: float,
    compression_seed: int,
    overwrite: bool,
) -> None:
    """Shuffle the proprietary pixels of MaskedImage files for public release.

    Each FILE is read in whatever format it is given, its image, mask, and
    variance planes are shuffled within tiles (one shared permutation per tile
    keeps the three planes mutually consistent), and the result is written
    beside the input with SUFFIX inserted before the extension.  The output
    format follows that extension.  Every other plane and all metadata are
    written back unchanged, and each output is re-read to confirm the
    proprietary planes really changed.
    """
    if not files:
        raise click.UsageError("No input files given.")
    failures = 0
    for in_path in files:
        out_path = _output_path(in_path, suffix)
        if out_path.exists() and not overwrite:
            click.echo(f"Skipping {in_path}: output {out_path} already exists.", err=True)
            continue
        try:
            _fuzz_file(
                in_path,
                out_path,
                seed=seed,
                tile_override=tile_shape if tile_shape else None,
                quantize_level=quantize_level,
                compression_seed=compression_seed,
            )
        except click.ClickException as err:
            click.echo(f"Error: {err.format_message()}", err=True)
            failures += 1
    if failures:
        raise click.ClickException(f"{failures} file(s) failed to fuzz.")
