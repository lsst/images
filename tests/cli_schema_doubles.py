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
"""A development schema that exists only to exercise the fixtures CLI.

``fixtures refresh`` acts only on schemas still in development, and every
schema is finalized during a release, so a test that named a real one would
stop exercising the path it claims.  This double is never finalized.

It is defined in the test tree, so ``available_schema_classes`` never returns
it for ``lsst.images``: it is never frozen, published, or part of shipped
data.  Tests reach it by passing this module as the CLI's ``--package``, which
selects this schema and no other.

This is a plain module rather than a test module so importing it registers the
schema deterministically, independent of pytest's collection order.
"""

from __future__ import annotations

__all__ = ("CliFixtureDouble",)

from typing import Any, ClassVar

import pydantic

from lsst.images.serialization import ArchiveTree, InputArchive


class CliFixtureDouble(ArchiveTree):
    """Development schema whose fixture the CLI tests write by hand."""

    SCHEMA_NAME: ClassVar[str] = "cli_fixture_double"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0.dev0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = dict

    value: str = pydantic.Field(description="Arbitrary content; required, so an empty tree fails to read.")

    def deserialize(self, archive: InputArchive[Any], **kwargs: Any) -> Any:
        return {"value": self.value}
