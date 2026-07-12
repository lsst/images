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

__all__ = ("schemas",)

from pathlib import Path

import click

from ..frozen_schemas import check_frozen_schemas, write_frozen_schemas

_DIR_OPTION = click.option(
    "--dir",
    "directory",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("schemas"),
    show_default=True,
    help="Directory holding the frozen schema files.",
)


@click.group(name="schemas")
def schemas() -> None:
    """Manage the frozen JSON schema files committed to the repository."""


@schemas.command(name="write")
@_DIR_OPTION
def write(directory: Path) -> None:  # numpydoc ignore=PR01
    """Write the JSON schema file for every current schema.

    Overwrites a stale file for the same schema version (schemas evolve in
    place until the first data release) and never touches files for
    superseded versions.
    """
    changed = write_frozen_schemas(directory)
    for path in changed:
        click.echo(f"wrote {path}")
    if not changed:
        click.echo("all frozen schema files are already up to date")


@schemas.command(name="check")
@_DIR_OPTION
def check(directory: Path) -> None:  # numpydoc ignore=PR01
    """Exit nonzero if any frozen schema file is missing or stale."""
    problems = check_frozen_schemas(directory)
    for problem in problems:
        click.echo(problem)
    if problems:
        raise click.ClickException("run 'lsst-images-admin schemas write' and commit the result")
