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

from ..serialization import backend_for_path


@click.command(name="inspect")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
def inspect(file: str) -> None:
    """Print basic information about an lsst.images file.

    Reports the schema URL and container format version without
    deserializing pixel data.
    """
    try:
        backend = backend_for_path(file)
    except ValueError as err:
        raise click.ClickException(str(err)) from None
    info = backend.input_archive.get_basic_info(file)
    fmt = "n/a" if info.format_version is None else str(info.format_version)
    click.echo(f"path:           {file}")
    click.echo(f"format:         {backend.name}")
    click.echo(f"schema name:    {info.schema_name}")
    click.echo(f"schema version: {info.schema_version}")
    click.echo(f"schema URL:     {info.schema_url}")
    click.echo(f"format version: {fmt}")
