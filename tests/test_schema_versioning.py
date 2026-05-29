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
from typing import ClassVar

import pydantic

from lsst.images.serialization import ArchiveReadError, ArchiveTree
from lsst.images.serialization._common import _check_compat, _check_format_version


class _DummyArchiveTree(ArchiveTree):
    """Minimal concrete ArchiveTree for testing the base-class machinery."""

    SCHEMA_NAME: ClassVar[str] = "dummy"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1

    def deserialize(self, archive, **kwargs):  # pragma: no cover - never invoked
        raise NotImplementedError()


class CheckCompatTestCase(unittest.TestCase):
    """Tests for the _check_compat and _check_format_version helpers."""

    def test_silent_when_min_read_satisfied(self):
        # min_read_version equals reader major: silent.
        _check_compat("foo", "1.0.0", 1, "1.0.0")

    def test_silent_when_on_disk_major_is_lower(self):
        # 1.0.0 file with min_read_version=1 read by 2.0.0 code: silent.
        _check_compat("foo", "1.0.0", 1, "2.0.0")

    def test_silent_when_on_disk_major_is_higher_but_min_read_low(self):
        # 2.0.0 file declares it is safe for major-1 readers: silent.
        _check_compat("foo", "2.0.0", 1, "1.0.0")

    def test_raises_when_min_read_exceeds_reader_major(self):
        with self.assertRaises(ArchiveReadError) as ctx:
            _check_compat("foo", "2.0.0", 2, "1.0.0")
        self.assertIn("foo", str(ctx.exception))
        self.assertIn(">= 2", str(ctx.exception))

    def test_format_version_silent_when_equal(self):
        _check_format_version("fits", 1, 1)

    def test_format_version_silent_when_on_disk_lower(self):
        _check_format_version("fits", 1, 2)

    def test_format_version_raises_when_on_disk_higher(self):
        with self.assertRaises(ArchiveReadError):
            _check_format_version("fits", 2, 1)


class ArchiveTreeVersionFieldsTestCase(unittest.TestCase):
    """Tests for the schema_version / min_read_version / schema_url fields."""

    def test_default_values_filled_from_classvars(self):
        instance = _DummyArchiveTree()
        self.assertEqual(instance.schema_version, "1.0.0")
        self.assertEqual(instance.min_read_version, 1)

    def test_schema_url_is_computed(self):
        instance = _DummyArchiveTree()
        self.assertEqual(instance.schema_url, "https://images.lsst.io/schemas/dummy-1.0.0")

    def test_schema_url_appears_in_dump(self):
        instance = _DummyArchiveTree()
        dumped = instance.model_dump()
        self.assertEqual(dumped["schema_url"], "https://images.lsst.io/schemas/dummy-1.0.0")
        self.assertEqual(dumped["schema_version"], "1.0.0")
        self.assertEqual(dumped["min_read_version"], 1)

    def test_schema_url_ignored_in_input(self):
        # Pydantic's default extra='ignore' drops it from inputs.
        instance = _DummyArchiveTree.model_validate(
            {"schema_url": "https://example.com/wrong", "schema_version": "1.0.0", "min_read_version": 1}
        )
        self.assertEqual(instance.schema_url, "https://images.lsst.io/schemas/dummy-1.0.0")

    def test_normalises_to_in_code_values(self):
        # An older file's values are normalised on load.
        instance = _DummyArchiveTree.model_validate({"schema_version": "0.9.0", "min_read_version": 1})
        self.assertEqual(instance.schema_version, "1.0.0")
        self.assertEqual(instance.min_read_version, 1)

    def test_absent_fields_default_to_legacy(self):
        instance = _DummyArchiveTree.model_validate({})
        self.assertEqual(instance.schema_version, "1.0.0")
        self.assertEqual(instance.min_read_version, 1)

    def test_min_read_version_too_high_rejected(self):
        # Pydantic mode='after' re-raises ArchiveReadError without
        # wrapping it in ValidationError.
        with self.assertRaises((ArchiveReadError, pydantic.ValidationError)):
            _DummyArchiveTree.model_validate({"schema_version": "2.0.0", "min_read_version": 2})


class JsonSchemaInjectionTestCase(unittest.TestCase):
    """ArchiveTree injects $id and title into each subclass's JSON Schema."""

    def test_image_schema_has_id_and_title(self):
        """Image's serialization-mode schema has ``$id`` / ``title`` set."""
        from lsst.images._image import ImageSerializationModel

        schema = ImageSerializationModel.model_json_schema(mode="serialization")
        self.assertEqual(schema["$id"], "https://images.lsst.io/schemas/image-1.0.0")
        self.assertEqual(schema["title"], "image")


class ArchiveTreeClassInvariantsTestCase(unittest.TestCase):
    """Concrete ArchiveTree subclasses must declare the version ClassVars."""

    @classmethod
    def _all_concrete_subclasses(cls):
        # Force-import every package module that defines a subclass.
        import lsst.images
        import lsst.images.cells
        import lsst.images.fields
        import lsst.images.psfs  # noqa: F401
        from lsst.images.serialization import ArchiveTree

        seen: set[type] = set()
        stack: list[type] = [ArchiveTree]
        while stack:
            kls = stack.pop()
            for sub in kls.__subclasses__():
                if sub in seen:
                    continue
                seen.add(sub)
                stack.append(sub)
                if not getattr(sub, "__abstractmethods__", None):
                    yield sub

    def test_constants_are_declared(self):
        """All three ClassVars are declared and well-formed everywhere."""
        found = list(self._all_concrete_subclasses())
        self.assertGreater(len(found), 0)
        for sub in found:
            with self.subTest(cls=sub.__name__):
                self.assertTrue(hasattr(sub, "SCHEMA_NAME"), f"{sub.__name__} lacks SCHEMA_NAME")
                self.assertTrue(hasattr(sub, "SCHEMA_VERSION"), f"{sub.__name__} lacks SCHEMA_VERSION")
                self.assertTrue(hasattr(sub, "MIN_READ_VERSION"), f"{sub.__name__} lacks MIN_READ_VERSION")
                self.assertIsInstance(sub.SCHEMA_NAME, str)
                self.assertGreater(len(sub.SCHEMA_NAME), 0)
                self.assertRegex(sub.SCHEMA_VERSION, r"^\d+\.\d+\.\d+$")
                self.assertIsInstance(sub.MIN_READ_VERSION, int)
                self.assertGreaterEqual(sub.MIN_READ_VERSION, 1)
                major = int(sub.SCHEMA_VERSION.split(".")[0])
                self.assertLessEqual(sub.MIN_READ_VERSION, major)

    def test_schema_names_unique(self):
        """All SCHEMA_NAME values across concrete subclasses are unique."""
        names: dict[str, type] = {}
        for sub in self._all_concrete_subclasses():
            # Skip our local _DummyArchiveTree (it lives in a test module).
            if sub.__module__.startswith("tests."):
                continue
            # Skip Pydantic generic parametrisations (e.g.
            # MaskedImageSerializationModel[TypeVar]); only the original
            # generic class counts. A parametrised form has a non-empty
            # __pydantic_generic_metadata__["args"].
            generic_meta = getattr(sub, "__pydantic_generic_metadata__", {})
            if generic_meta.get("args"):
                continue
            existing = names.get(sub.SCHEMA_NAME)
            if existing is not None:
                self.fail(
                    f"Duplicate SCHEMA_NAME {sub.SCHEMA_NAME!r}: "
                    f"{sub.__qualname__} vs {existing.__qualname__}"
                )
            names[sub.SCHEMA_NAME] = sub


if __name__ == "__main__":
    unittest.main()
