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
from typing import Any, ClassVar

import lsst.images  # registers schema classes
import lsst.images.cells  # noqa: F401  -- side-effect import: registers cell schemas
from lsst.images import Projection, VisitImage
from lsst.images._image import ImageSerializationModel
from lsst.images._visit_image import VisitImageSerializationModel
from lsst.images.serialization import ArchiveTree, class_for_schema
from lsst.images.serialization._io import _REGISTRY, _public_type, register_schema_class


class ClassForSchemaTestCase(unittest.TestCase):
    """class_for_schema returns None for unknown schema names."""

    def test_unknown_returns_none(self) -> None:
        self.assertIsNone(class_for_schema("does-not-exist"))


class RegistrationTestCase(unittest.TestCase):
    """ArchiveTree subclasses register themselves under SCHEMA_NAME."""

    def test_visit_image_registered(self) -> None:
        cls = class_for_schema("visit_image")
        self.assertIs(cls, VisitImageSerializationModel)

    def test_nested_image_registered(self) -> None:
        # Nested types are registered too -- "register all" was the spec
        # decision so callers can read sub-models directly.
        cls = class_for_schema("image")
        self.assertIs(cls, ImageSerializationModel)

    def test_duplicate_registration_raises(self) -> None:
        # Re-registering the same class is a no-op.
        register_schema_class(VisitImageSerializationModel)

        # Defining a subclass that redeclares SCHEMA_NAME triggers
        # __pydantic_init_subclass__, which calls register_schema_class
        # with a different class object under an existing name.  That
        # call raises RuntimeError.
        with self.assertRaises(RuntimeError):

            class _Imposter(VisitImageSerializationModel):  # type: ignore[misc]
                SCHEMA_NAME: ClassVar[str] = "visit_image"
                SCHEMA_VERSION: ClassVar[str] = "1.0.0"


class PublicTypeTestCase(unittest.TestCase):
    """The internal _public_type helper resolves deserialize's return."""

    def test_concrete_return_annotation(self) -> None:
        cls = class_for_schema("visit_image")
        assert cls is not None  # for type checkers
        self.assertIs(_public_type(cls), VisitImage)

    def test_parameterised_generic_unwrapped(self) -> None:
        # ProjectionSerializationModel.deserialize returns Projection[Any];
        # _public_type should unwrap to Projection.
        cls = class_for_schema("projection")
        assert cls is not None
        self.assertIs(_public_type(cls), Projection)

    def test_any_return_annotation(self) -> None:
        # The abstract base ArchiveTree returns Any; if a concrete subclass
        # ever does the same, _public_type must return None.
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
            _REGISTRY.pop("_any_tree_test", None)


def _all_concrete_archive_tree_subclasses() -> list[type]:
    """Walk ArchiveTree's subclass tree and return all concrete subclasses
    that declare SCHEMA_NAME (i.e. are real schema-bearing leaves).
    """
    seen: list[type] = []
    stack: list[type] = list(ArchiveTree.__subclasses__())
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        if "SCHEMA_NAME" in cls.__dict__:
            seen.append(cls)
    return seen


class ClassInvariantsTestCase(unittest.TestCase):
    """Every ArchiveTree subclass with SCHEMA_NAME is registered and has a
    resolvable, concrete deserialize return annotation.
    """

    def test_every_subclass_registered(self) -> None:
        # Test-local classes from other test methods may linger in
        # __subclasses__ after their cleanup runs; restrict the check to
        # classes whose modules belong to the package itself.
        missing: list[str] = []
        for cls in _all_concrete_archive_tree_subclasses():
            if not cls.__module__.startswith("lsst.images"):
                continue
            registered = _REGISTRY.get(cls.SCHEMA_NAME)
            if registered is None or registered is not cls:
                missing.append(f"{cls.__qualname__} -> {cls.SCHEMA_NAME}")
        self.assertEqual(missing, [], f"Unregistered subclasses: {missing}")

    def test_every_registered_class_resolves_public_type(self) -> None:
        # Restrict to package-local classes; test-only ArchiveTree
        # subclasses (e.g. tests/test_schema_versioning.py's
        # _DummyArchiveTree) may register but intentionally have no
        # concrete return annotation.
        unresolved: list[str] = []
        for cls in _REGISTRY.values():
            if not cls.__module__.startswith("lsst.images"):
                continue
            if _public_type(cls) is None:
                unresolved.append(cls.__qualname__)
        self.assertEqual(unresolved, [], f"No concrete return annotation: {unresolved}")


if __name__ == "__main__":
    unittest.main()
