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
from unittest import mock

from lsst.images import Projection, VisitImage
from lsst.images._image import ImageSerializationModel
from lsst.images._visit_image import VisitImageSerializationModel
from lsst.images.serialization import ArchiveTree, _io, class_for_schema, public_type_for_schema
from lsst.images.serialization._io import _REGISTRY, register_schema_class


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

    def test_builtin_provider_loaded_on_miss(self) -> None:
        schema_names = ("cell_coadd", "cell_psf", "coadd_provenance")
        saved = {schema_name: _REGISTRY.pop(schema_name, None) for schema_name in schema_names}
        try:
            for schema_name in schema_names:
                with self.subTest(schema_name=schema_name):
                    cls = class_for_schema(schema_name)
                    self.assertIsNotNone(cls)
                    self.assertEqual(cls.SCHEMA_NAME, schema_name)
        finally:
            # Preserve any registrations that existed before this test, even
            # if an assertion above fails.
            _REGISTRY.update({schema_name: cls for schema_name, cls in saved.items() if cls is not None})

    def test_entry_point_provider_loaded_on_miss(self) -> None:
        class _EntryPointTree(ArchiveTree):
            SCHEMA_NAME: ClassVar[str] = "_entry_point_schema_test"
            SCHEMA_VERSION: ClassVar[str] = "1.0.0"
            MIN_READ_VERSION: ClassVar[int] = 1

            def deserialize(self, archive: Any, **kwargs: Any) -> VisitImage:
                raise AssertionError("not used")

        class _FakeEntryPoint:
            value = "tests.test_serialization_registry:_EntryPointTree"

            def load(self) -> type[ArchiveTree]:
                return _EntryPointTree

        _REGISTRY.pop("_entry_point_schema_test", None)
        try:
            with mock.patch.object(
                _io.importlib.metadata,
                "entry_points",
                return_value=[_FakeEntryPoint()],
            ) as entry_points:
                cls = class_for_schema("_entry_point_schema_test")
            entry_points.assert_called_once_with(
                group="lsst.images.schemas",
                name="_entry_point_schema_test",
            )
            self.assertIs(cls, _EntryPointTree)
        finally:
            _REGISTRY.pop("_entry_point_schema_test", None)


class PublicTypeTestCase(unittest.TestCase):
    """public_type_for_schema returns each tree's PUBLIC_TYPE ClassVar."""

    def test_concrete_type(self) -> None:
        self.assertIs(public_type_for_schema("visit_image"), VisitImage)

    def test_generic_in_memory_type(self) -> None:
        # ProjectionSerializationModel produces a Projection (its PUBLIC_TYPE
        # is the unparameterised class, not Projection[Any]).
        self.assertIs(public_type_for_schema("projection"), Projection)

    def test_unregistered_schema_returns_none(self) -> None:
        self.assertIsNone(public_type_for_schema("no-such-schema"))


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

    def test_every_registered_class_declares_public_type(self) -> None:
        # Restrict to package-local classes; test-only ArchiveTree
        # subclasses (e.g. tests/test_schema_versioning.py's
        # _DummyArchiveTree) may register but are not part of the package.
        unresolved: list[str] = []
        for cls in _REGISTRY.values():
            if not cls.__module__.startswith("lsst.images"):
                continue
            if not isinstance(getattr(cls, "PUBLIC_TYPE", None), type):
                unresolved.append(cls.__qualname__)
        self.assertEqual(unresolved, [], f"No PUBLIC_TYPE declared: {unresolved}")


if __name__ == "__main__":
    unittest.main()
