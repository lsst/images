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

__all__ = ("minify",)

import os

import click


@click.command(name="minify")
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(dir_okay=False))
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite OUTPUT if it exists.")
def minify(input: str, output: str, overwrite: bool) -> None:  # numpydoc ignore=PR01
    """Subset a real data file into a small test fixture."""
    from ..tests._minify_for_fixtures import minify as _minify

    if os.path.exists(output) and not overwrite:
        raise click.ClickException(f"{output!r} already exists; pass --overwrite to replace it.")
    _minify(input, output)
    click.echo(f"Wrote {output}.")
