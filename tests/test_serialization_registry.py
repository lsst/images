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

from typing import Any, ClassVar
from unittest import mock

import pytest

from lsst.images import SkyProjection, VisitImage
from lsst.images._image import ImageSerializationModel
from lsst.images._visit_image import VisitImageSerializationModel
from lsst.images.serialization import ArchiveTree, _io, class_for_schema, public_type_for_schema
from lsst.images.serialization._io import _REGISTRY, register_schema_class


def test_unknown_returns_none() -> None:
    """Verify class_for_schema returns None for an unrecognised schema name."""
    assert class_for_schema("does-not-exist") is None


def test_visit_image_registered() -> None:
    """Verify visit_image is registered and maps to
    VisitImageSerializationModel.
    """
    cls = class_for_schema("visit_image")
    assert cls is VisitImageSerializationModel


def test_nested_image_registered() -> None:
    """Verify nested types like image are registered so sub-models can
    be read directly.
    """
    # Nested types are registered too -- "register all" was the spec
    # decision so callers can read sub-models directly.
    cls = class_for_schema("image")
    assert cls is ImageSerializationModel


def test_duplicate_registration_raises() -> None:
    """Verify re-registering a class is a no-op but a conflicting class
    raises RuntimeError.
    """
    # Re-registering the same class is a no-op.
    register_schema_class(VisitImageSerializationModel)

    # Defining a subclass that redeclares SCHEMA_NAME triggers
    # __pydantic_init_subclass__, which calls register_schema_class
    # with a different class object under an existing name.  That
    # call raises RuntimeError.
    with pytest.raises(RuntimeError):

        class _Imposter(VisitImageSerializationModel):  # type: ignore[misc]
            SCHEMA_NAME: ClassVar[str] = "visit_image"
            SCHEMA_VERSION: ClassVar[str] = "1.0.0"


def test_builtin_provider_loaded_on_miss() -> None:
    """Verify builtin schema providers are loaded lazily on a registry
    cache miss.
    """
    schema_names = ("cell_coadd", "cell_psf", "coadd_provenance")
    saved = {schema_name: _REGISTRY.pop(schema_name, None) for schema_name in schema_names}
    try:
        for schema_name in schema_names:
            cls = class_for_schema(schema_name)
            assert cls is not None, f"schema_name={schema_name!r}: expected a registered class"
            assert cls.SCHEMA_NAME == schema_name, f"schema_name={schema_name!r}: SCHEMA_NAME mismatch"
    finally:
        # Preserve any registrations that existed before this test, even
        # if an assertion above fails.
        _REGISTRY.update({schema_name: cls for schema_name, cls in saved.items() if cls is not None})


def test_entry_point_provider_loaded_on_miss() -> None:
    """Verify entry-point providers are loaded lazily on a registry
    cache miss.
    """

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
        assert cls is _EntryPointTree
    finally:
        _REGISTRY.pop("_entry_point_schema_test", None)


def test_concrete_type() -> None:
    """Verify public_type_for_schema returns VisitImage for the
    visit_image schema.
    """
    assert public_type_for_schema("visit_image") is VisitImage


def test_generic_in_memory_type() -> None:
    """Verify public_type_for_schema returns SkyProjection for the
    sky_projection schema.
    """
    # ProjectionSerializationModel produces a Projection (its PUBLIC_TYPE
    # is the unparameterised class, not Projection[Any]).
    assert public_type_for_schema("sky_projection") is SkyProjection


def test_unregistered_schema_returns_none() -> None:
    """Verify public_type_for_schema returns None for an unregistered
    schema name.
    """
    assert public_type_for_schema("no-such-schema") is None


def _all_concrete_archive_tree_subclasses() -> list[type]:
    """Return all concrete ArchiveTree subclasses that declare SCHEMA_NAME."""
    seen: list[type] = []
    stack: list[type] = list(ArchiveTree.__subclasses__())
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        if "SCHEMA_NAME" in cls.__dict__:
            seen.append(cls)
    return seen


def test_every_subclass_registered() -> None:
    """Verify every package-local ArchiveTree subclass with SCHEMA_NAME
    is registered.
    """
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
    assert missing == [], f"Unregistered subclasses: {missing}"


def test_every_registered_class_declares_public_type() -> None:
    """Verify every package-local registered ArchiveTree class declares
    PUBLIC_TYPE.
    """
    # Restrict to package-local classes; test-only ArchiveTree
    # subclasses (e.g. tests/test_schema_versioning.py's
    # _DummyArchiveTree) may register but are not part of the package.
    unresolved: list[str] = []
    for cls in _REGISTRY.values():
        if not cls.__module__.startswith("lsst.images"):
            continue
        if not isinstance(getattr(cls, "PUBLIC_TYPE", None), type):
            unresolved.append(cls.__qualname__)
    assert unresolved == [], f"No PUBLIC_TYPE declared: {unresolved}"
