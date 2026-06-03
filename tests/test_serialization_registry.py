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

import unittest

import lsst.images  # import for side-effect class registration
import lsst.images.cells  # noqa: F401  -- side-effect import: registers cell schemas
from lsst.images._image import ImageSerializationModel
from lsst.images._visit_image import VisitImageSerializationModel
from lsst.images.serialization import class_for_schema


class ClassForSchemaTestCase(unittest.TestCase):
    """class_for_schema returns None for unknown (name, version)."""

    def test_unknown_returns_none(self) -> None:
        self.assertIsNone(class_for_schema("does-not-exist", "1.0.0"))


class RegistrationTestCase(unittest.TestCase):
    """ArchiveTree subclasses register themselves under (name, version)."""

    def test_visit_image_registered(self) -> None:
        cls = class_for_schema("visit_image", "1.0.0")
        self.assertIs(cls, VisitImageSerializationModel)

    def test_nested_image_registered(self) -> None:
        # Nested types are registered too -- "register all" was the spec
        # decision so callers can read sub-models directly.
        cls = class_for_schema("image", "1.0.0")
        self.assertIs(cls, ImageSerializationModel)

    def test_duplicate_registration_raises(self) -> None:
        from typing import ClassVar

        from lsst.images.serialization._io import register_schema_class

        # Re-registering the same class is a no-op.
        register_schema_class(VisitImageSerializationModel)

        # Defining a subclass that redeclares SCHEMA_NAME / SCHEMA_VERSION
        # to the same values triggers __pydantic_init_subclass__, which
        # calls register_schema_class with a different class object under
        # an existing key.  That call raises RuntimeError.
        with self.assertRaises(RuntimeError):

            class _Imposter(VisitImageSerializationModel):  # type: ignore[misc]
                SCHEMA_NAME: ClassVar[str] = "visit_image"
                SCHEMA_VERSION: ClassVar[str] = "1.0.0"


class PublicTypeTestCase(unittest.TestCase):
    """The internal _public_type helper resolves deserialize's return."""

    def test_concrete_return_annotation(self) -> None:
        from lsst.images import VisitImage
        from lsst.images.serialization._io import _public_type

        cls = class_for_schema("visit_image", "1.0.0")
        assert cls is not None  # for type checkers
        self.assertIs(_public_type(cls), VisitImage)

    def test_parameterised_generic_unwrapped(self) -> None:
        # ProjectionSerializationModel.deserialize returns Projection[Any];
        # _public_type should unwrap to Projection.
        from lsst.images import Projection
        from lsst.images.serialization._io import _public_type

        cls = class_for_schema("projection", "1.0.0")
        assert cls is not None
        self.assertIs(_public_type(cls), Projection)

    def test_any_return_annotation(self) -> None:
        # The abstract base ArchiveTree returns Any; if a concrete subclass
        # ever does the same, _public_type must return None.
        from typing import Any, ClassVar

        from lsst.images.serialization import ArchiveTree
        from lsst.images.serialization._io import _REGISTRY, _public_type

        # Construct a stand-alone ArchiveTree subclass with an Any return.
        class _AnyTree(ArchiveTree):
            SCHEMA_NAME: ClassVar[str] = "_any_tree_test"
            SCHEMA_VERSION: ClassVar[str] = "1.0.0"
            MIN_READ_VERSION: ClassVar[int] = 1

            def deserialize(self, archive: Any, **kwargs: Any) -> Any:
                return None

        try:
            self.assertIsNone(_public_type(_AnyTree))
        finally:
            # Tidy up: pop the stand-alone class out of the registry so
            # it doesn't leak into other tests.
            _REGISTRY.pop(("_any_tree_test", "1.0.0"), None)


if __name__ == "__main__":
    unittest.main()
