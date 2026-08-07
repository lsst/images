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

__all__ = ("fixtures",)

from pathlib import Path

import click
import pydantic

from ..serialization import ArchiveReadError
from ..tests import (
    SchemaFixtureError,
    check_schema_fixtures,
    freeze_schema_fixtures,
    refresh_schema_fixtures,
)

# read_fixture_tree (called internally by both refresh_schema_fixtures and
# freeze_schema_fixtures) can raise either of these for a fixture that no
# longer validates, in addition to SchemaFixtureError for a conflict these
# commands detect themselves; all three get the same clean-exception
# treatment rather than a raw traceback.
_FIXTURE_ERRORS = (SchemaFixtureError, ArchiveReadError, pydantic.ValidationError)

_DIR_OPTION = click.option(
    "--dir",
    "directory",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("tests/data/schemas"),
    show_default=True,
    help="Directory holding the fixture tree.",
)

_PACKAGE_OPTION = click.option(
    "--package",
    default="lsst.images",
    show_default=True,
    help="Act only on schemas whose model classes are defined in this package; "
    "external packages providing schemas through the 'lsst.images.schemas' "
    "entry point group can use this to manage their own fixtures.",
)


@click.group(name="fixtures")
def fixtures() -> None:
    """Manage the committed schema fixtures."""


@fixtures.command(name="check")
@_DIR_OPTION
@_PACKAGE_OPTION
@click.option(
    "--schema-dir",
    "schema_directory",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory holding the frozen schema documents. Fixtures and schema "
    "documents are checked for pairing in both directions.",
)
@click.option(
    "--exempt",
    multiple=True,
    help="Schema name allowed to have no fixture; repeatable.",
)
def check(
    directory: Path,
    package: str,
    schema_directory: Path,
    exempt: tuple[str, ...],
) -> None:
    """Exit nonzero if any fixture is missing, stale, or misplaced."""
    problems = check_schema_fixtures(
        directory, schema_directory=schema_directory, package=package, exempt=exempt
    )
    for problem in problems:
        click.echo(problem)
    if problems:
        raise click.ClickException("the fixture tree does not match the current models")


@fixtures.command(name="refresh")
@_DIR_OPTION
@_PACKAGE_OPTION
def refresh(directory: Path, package: str) -> None:
    """Rewrite every development fixture in canonical form.

    Seeds a missing fixture from the newest existing version of that schema,
    and regenerates the canonical twin of a development as_shipped fixture.
    Refuses to rewrite a fixture at a finalized version; bump SCHEMA_VERSION
    instead.
    """
    try:
        changed = refresh_schema_fixtures(directory, package=package)
    except _FIXTURE_ERRORS as exc:
        raise click.ClickException(str(exc)) from exc
    for path in changed:
        click.echo(f"wrote {path}")
    if not changed:
        click.echo("all fixtures are already up to date")


@fixtures.command(name="freeze")
@_DIR_OPTION
@_PACKAGE_OPTION
def freeze(directory: Path, package: str) -> None:
    """Move each ordinary development fixture to its finalized version.

    Run this after dropping the .devN suffix from SCHEMA_VERSION and running
    'lsst-images-admin schemas write'.  Files are written and deleted; staging
    that in version control is left to you.  Development ``as_shipped``
    fixtures are refused because their preserved bytes cannot be re-stamped.
    """
    try:
        frozen = freeze_schema_fixtures(directory, package=package)
    except _FIXTURE_ERRORS as exc:
        raise click.ClickException(str(exc)) from exc
    for written, removed in frozen:
        click.echo(f"wrote {written}")
        click.echo(f"removed {removed}")
    if not frozen:
        click.echo("nothing to freeze")
    else:
        click.echo("now run 'lsst-images-admin schemas write' if you have not already")
