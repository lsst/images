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

__all__ = ("reformat",)

import os
import tempfile

import click

from ..serialization import ArchiveReadError, backend_for_path, read, write


@click.command(name="reformat")
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(dir_okay=False))
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite OUTPUT if it exists.")
def reformat(input: str, output: str, overwrite: bool) -> None:
    """Rewrite an lsst.images file in a different container format.

    Reads INPUT and writes it back out to OUTPUT, choosing the format from
    OUTPUT's extension (.fits, .sdf/.ndf, .json).  This is the easy way to,
    for example, turn a FITS file into an NDF for testing.
    """
    try:
        backend = backend_for_path(output)
    except ValueError as err:
        raise click.ClickException(str(err)) from None

    output_abs = os.path.realpath(output)
    if os.path.realpath(input) == output_abs:
        raise click.ClickException("INPUT and OUTPUT must be different paths.")

    if os.path.exists(output_abs) and not overwrite:
        raise click.ClickException(f"{output!r} already exists; pass --overwrite to replace it.")

    try:
        obj = read(input)
    except (ValueError, ArchiveReadError) as err:
        raise click.ClickException(f"Could not read {input!r}: {err}") from None

    # Write to a temporary file in the output's directory and move it into
    # place only after a successful write, so a failure never destroys an
    # existing OUTPUT (the backends refuse to overwrite, so the temporary path
    # must not already exist).
    output_dir = os.path.dirname(output_abs)
    with tempfile.TemporaryDirectory(dir=output_dir) as staging:
        staged = os.path.join(staging, os.path.basename(output_abs))
        write(obj, staged)
        os.replace(staged, output_abs)
    click.echo(f"Wrote {output} ({backend.name}).")
