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

__all__ = ("inspect",)

import click


@click.command(name="inspect")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
def inspect(file: str) -> None:
    """Print basic information about an lsst.images file."""
    raise click.ClickException("not yet implemented")
