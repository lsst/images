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

__all__ = ("describe",)

import click
from rich.console import Console

from ..describe import Describable
from ..serialization import ArchiveReadError, read_archive


@click.command(name="describe")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
def describe(file: str) -> None:  # numpydoc ignore=PR01
    """Deserialize an lsst.images file and print its data-model report.

    Unlike ``inspect`` (which reports only the file layout), this reads the
    full object and renders its `~lsst.images.Report` via the rich terminal
    renderer, including nested components and, where available, WCS corner
    sky coordinates.
    """
    try:
        obj = read_archive(file)
    except (ArchiveReadError, ValueError, TypeError) as err:
        raise click.ClickException(f"Could not read {file}: {err}") from None
    if not isinstance(obj, Describable):
        raise click.ClickException(f"{type(obj).__name__} does not support 'describe'.")
    Console().print(obj.describe())
