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
"""Purpose-built schemas that exist only to exercise the fixture machinery.

These are defined in the test tree, so ``available_schema_classes`` — which
filters by defining module — never sees them: they are never frozen, never
published, and never part of shipped data.  Their fixtures live under
``tests/data/fixture_doubles`` in the same layout as the real fixture tree, so
the same ladder runs over both.

This is a plain module rather than a test module so importing it registers the
schemas deterministically, independent of pytest's collection order.
"""

from __future__ import annotations

__all__ = (
    "ProjectionTestModel",
    "RetireTestModel",
)

from typing import Any, ClassVar

import pydantic

from lsst.images.serialization import ArchiveTree, InputArchive


class ProjectionTestModel(ArchiveTree):
    """Schema whose 1.1.0 adds an optional field over 1.0.0.

    The additive common case: the model absorbs a 1.0.0 tree unaided, so the
    projection oracle must pass on this pair with no escape hatch.  This is the
    oracle's positive control.
    """

    SCHEMA_NAME: ClassVar[str] = "projection_test"
    SCHEMA_VERSION: ClassVar[str] = "1.1.0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = dict

    kept: str = pydantic.Field(default="", description="Present at both versions.")
    added: int = pydantic.Field(default=0, description="Added at 1.1.0.")

    def deserialize(self, archive: InputArchive[Any], **kwargs: Any) -> Any:
        return {"kept": self.kept, "added": self.added}


class RetireTestModel(ArchiveTree):
    """Schema whose 2.0.0 shape a 1.0.0 tree cannot satisfy.

    ``required_at_v2`` has no default and no migration is registered, so the
    1.0.0 fixture can no longer be read.  That is what retirement means, and
    the fixture under ``retired/`` asserts it.
    """

    SCHEMA_NAME: ClassVar[str] = "retire_test"
    SCHEMA_VERSION: ClassVar[str] = "2.0.0"
    MIN_READ_VERSION: ClassVar[int] = 2
    PUBLIC_TYPE: ClassVar[type] = dict

    required_at_v2: str = pydantic.Field(description="Required from 2.0.0 on.")

    def deserialize(self, archive: InputArchive[Any], **kwargs: Any) -> Any:
        return {"required_at_v2": self.required_at_v2}
