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

__all__ = ("convert",)

import os
import tempfile
from typing import TYPE_CHECKING, Any

import astropy.io.fits
import click
from click.core import ParameterSource

from ..serialization import backend_for_path, write_archive

if TYPE_CHECKING:
    from .. import VisitImage
    from ..cells import CellCoadd

_LEGACY_TYPES = ("visit_image", "cell_coadd")


def detect_legacy_type(path: str) -> str | None:
    """Return ``"visit_image"`` or ``"cell_coadd"`` from a legacy FITS file's
    ``HIERARCH LSST BUTLER DATASETTYPE`` header, or `None` if it cannot be
    determined.

    A dataset type ending in ``visit_image`` (e.g. ``visit_image``,
    ``preliminary_visit_image``, difference images) is a `VisitImage`; one
    containing ``coadd`` is a `CellCoadd`.  astropy exposes the
    ``HIERARCH LSST BUTLER DATASETTYPE`` card as
    ``header["LSST BUTLER DATASETTYPE"]``.

    Parameters
    ----------
    path
        Path to the legacy FITS file to inspect.
    """
    dataset_type: str | None = None
    with astropy.io.fits.open(path) as hdul:
        for hdu in hdul:
            value = hdu.header.get("LSST BUTLER DATASETTYPE")
            if value:
                dataset_type = str(value)
                break
    if dataset_type is None:
        return None
    if dataset_type.endswith("visit_image"):
        return "visit_image"
    if dataset_type == "difference_image":
        return "difference_image"
    if "coadd" in dataset_type:
        return "cell_coadd"
    return None


def _load_skymap(skymap: str | None, butler: str | None, collection: str | None, skymap_name: str) -> Any:
    """Load a skymap object from a pickle path or a butler repository."""
    if skymap is not None:
        import pickle

        with open(skymap, "rb") as stream:
            return pickle.load(stream)
    if butler is not None:
        if collection is None:
            raise click.ClickException("--butler also requires --collection (the skymap's collection).")
        from lsst.daf.butler import Butler

        with Butler.from_config(butler) as repo:
            return repo.get("skyMap", skymap=skymap_name, collections=collection)
    raise click.ClickException("Converting a cell coadd requires --skymap (a pickled skymap) or --butler.")


def _read_legacy(
    input: str,
    legacy_type: str,
    skymap: str | None,
    butler: str | None,
    collection: str | None,
    preserve_quantization: bool = False,
) -> VisitImage | CellCoadd:
    """Read a legacy FITS file into the corresponding lsst.images object."""
    if legacy_type == "visit_image":
        from .. import VisitImage

        return VisitImage.read_legacy(input, preserve_quantization=preserve_quantization)
    if legacy_type == "difference_image":
        from .. import DifferenceImage

        return DifferenceImage.read_legacy(input, preserve_quantization=preserve_quantization)
    if legacy_type == "cell_coadd":
        from lsst.cell_coadds import MultipleCellCoadd

        from .. import get_legacy_deep_coadd_mask_planes
        from ..cells import CellCoadd

        legacy = MultipleCellCoadd.read_fits(input)
        sky = _load_skymap(skymap, butler, collection, legacy.identifiers.skymap)
        tract_info = sky[legacy.identifiers.tract]
        return CellCoadd.from_legacy_cell_coadd(
            legacy,
            tract_info=tract_info,
            plane_map=get_legacy_deep_coadd_mask_planes(),
        )
    raise click.ClickException(f"Conversion of {legacy_type!r} is not yet implemented.")


@click.command(name="convert")
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(dir_okay=False))
@click.option(
    "--type",
    "type_",
    type=click.Choice(_LEGACY_TYPES),
    default=None,
    help="Legacy input type; overrides auto-detection.",
)
@click.option(
    "--skymap",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Pickled skymap (required for cell coadds unless --butler is given).",
)
@click.option(
    "--butler",
    default=None,
    help="Butler repository to resolve the skymap (cell coadds only).",
)
@click.option(
    "--collection",
    default=None,
    help="Butler collection holding the skymap (required with --butler).",
)
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite OUTPUT if it exists.")
@click.option(
    "--preserve-quantization/--no-preserve-quantization",
    default=True,
    help=(
        "Preserve quantization-compressed pixel values so they can be written "
        "back out losslessly (visit images only).  On by default."
    ),
)
def convert(
    input: str,
    output: str,
    type_: str | None,
    skymap: str | None,
    butler: str | None,
    collection: str | None,
    overwrite: bool,
    preserve_quantization: bool,
) -> None:  # numpydoc ignore=PR01
    """Convert a legacy FITS file to a new lsst.images format.

    The output format is chosen from OUTPUT's extension.
    """
    try:
        backend = backend_for_path(output)
    except ValueError as err:
        raise click.ClickException(str(err)) from None

    legacy_type = type_ or detect_legacy_type(input)
    if legacy_type is None:
        raise click.ClickException(f"Could not determine the legacy type of {input!r}; pass --type.")

    # The flag is on by default, so only object when the user explicitly set
    # it for a type that does not support it (only visit images do).
    if legacy_type != "visit_image":
        source = click.get_current_context().get_parameter_source("preserve_quantization")
        if source == ParameterSource.COMMANDLINE:
            raise click.ClickException(
                f"--preserve-quantization is only valid for visit images, not {legacy_type!r}."
            )
        preserve_quantization = False

    output_abs = os.path.realpath(output)
    if os.path.realpath(input) == output_abs:
        raise click.ClickException("INPUT and OUTPUT must be different paths.")

    if os.path.exists(output_abs) and not overwrite:
        raise click.ClickException(f"{output!r} already exists; pass --overwrite to replace it.")

    try:
        obj = _read_legacy(input, legacy_type, skymap, butler, collection, preserve_quantization)
    except click.ClickException:
        raise
    except ImportError as err:
        raise click.ClickException(
            f"Reading a legacy {legacy_type} requires Rubin packages that are not installed: {err}"
        ) from None

    # Write to a temporary file in the output's directory and move it into
    # place only after a successful write, so a read or write failure never
    # destroys an existing OUTPUT (the backends refuse to overwrite, so the
    # temporary path must not already exist).
    output_dir = os.path.dirname(output_abs)
    with tempfile.TemporaryDirectory(dir=output_dir) as staging:
        staged = os.path.join(staging, os.path.basename(output_abs))
        write_archive(obj, staged)
        os.replace(staged, output_abs)
    click.echo(f"Wrote {output} ({backend.name}, {legacy_type}).")
