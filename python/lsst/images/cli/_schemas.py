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

from ..serialization import check_frozen_schemas, write_frozen_schemas

_DIR_OPTION = click.option(
    "--dir",
    "directory",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("schemas"),
    show_default=True,
    help="Directory holding the frozen schema files.",
)

_PACKAGE_OPTION = click.option(
    "--package",
    default="lsst.images",
    show_default=True,
    help="Freeze only schemas whose model classes are defined in this package; "
    "external packages providing schemas through the 'lsst.images.schemas' "
    "entry point group can use this to freeze their own schemas for their "
    "own documentation site.",
)


@click.group(name="schemas")
def schemas() -> None:
    """Manage the frozen JSON schema files committed to the repository."""


@schemas.command(name="write")
@_DIR_OPTION
@_PACKAGE_OPTION
def write(directory: Path, package: str) -> None:  # numpydoc ignore=PR01
    """Write the JSON schema file for every current schema.

    Overwrites a stale file for the same schema version (schemas evolve in
    place until the first data release) and never touches files for
    superseded versions.
    """
    changed = write_frozen_schemas(directory, package)
    for path in changed:
        click.echo(f"wrote {path}")
    if not changed:
        click.echo("all frozen schema files are already up to date")


@schemas.command(name="check")
@_DIR_OPTION
@_PACKAGE_OPTION
def check(directory: Path, package: str) -> None:  # numpydoc ignore=PR01
    """Exit nonzero if any frozen schema file is missing or stale."""
    problems = check_frozen_schemas(directory, package)
    for problem in problems:
        click.echo(problem)
    if problems:
        raise click.ClickException("run 'lsst-images-admin schemas write' and commit the result")
