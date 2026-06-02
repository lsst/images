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

import astropy.io.fits
import click

from ..serialization import backend_for_path

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
    if "coadd" in dataset_type:
        return "cell_coadd"
    return None


def _read_legacy(
    input: str,
    legacy_type: str,
    skymap: str | None,
    butler: str | None,
    collection: str | None,
):
    """Read a legacy FITS file into the corresponding lsst.images object."""
    if legacy_type == "visit_image":
        from .. import VisitImage

        return VisitImage.read_legacy(input)
    # cell_coadd handled in a later task.
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
def convert(
    input: str,
    output: str,
    type_: str | None,
    skymap: str | None,
    butler: str | None,
    collection: str | None,
    overwrite: bool,
) -> None:
    """Convert a legacy FITS file to a new lsst.images format.

    The output format is chosen from OUTPUT's extension
    (.fits, .sdf/.ndf, .json).
    """
    try:
        backend = backend_for_path(output)
    except ValueError as err:
        raise click.ClickException(str(err)) from None

    legacy_type = type_ or detect_legacy_type(input)
    if legacy_type is None:
        raise click.ClickException(f"Could not determine the legacy type of {input!r}; pass --type.")

    if os.path.exists(output):
        if not overwrite:
            raise click.ClickException(f"{output!r} already exists; pass --overwrite to replace it.")
        os.remove(output)

    obj = _read_legacy(input, legacy_type, skymap, butler, collection)
    backend.write(obj, output)
    click.echo(f"Wrote {output} ({backend.name}, {legacy_type}).")
