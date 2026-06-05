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
from pathlib import Path
from typing import ClassVar

import pydantic

from lsst.images.serialization import ArchiveReadError, ArchiveTree
from lsst.images.serialization._common import _check_compat, _check_format_version
from lsst.images.tests import check_archive_tree_class_invariants, iter_concrete_archive_tree_subclasses

SCHEMA_DIR = Path(__file__).parent / "data" / "schema_v1"


class _DummyArchiveTree(ArchiveTree):
    """Minimal concrete ArchiveTree for testing the base-class machinery."""

    SCHEMA_NAME: ClassVar[str] = "dummy"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = object

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
    """Concrete ArchiveTree subclasses must declare the version ClassVars.

    The reusable discovery and per-class check live in ``lsst.images.tests``
    (`iter_concrete_archive_tree_subclasses` and
    `check_archive_tree_class_invariants`) so the latter can be invoked
    manually on a single ``ArchiveTree`` independent of the metaprogramming.
    """

    def test_constants_are_declared(self):
        """All three ClassVars are declared and well-formed everywhere."""
        found = list(iter_concrete_archive_tree_subclasses())
        self.assertGreater(len(found), 0)
        for sub in found:
            with self.subTest(cls=sub.__name__):
                check_archive_tree_class_invariants(self, sub)

    def test_schema_names_unique(self):
        """All SCHEMA_NAME values across concrete subclasses are unique."""
        names: dict[str, type] = {}
        for sub in iter_concrete_archive_tree_subclasses():
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


class FixtureMutationTestCase(unittest.TestCase):
    """Mutate a fixture in-memory and verify the read behavior."""

    def setUp(self):
        self.fixture_path = SCHEMA_DIR / "image.json"
        self.assertTrue(self.fixture_path.exists())

    def test_min_read_too_high_raises(self):
        """Setting min_read_version above reader major rejects the read."""
        import json as json_module

        from lsst.images._image import ImageSerializationModel

        tree = json_module.loads(self.fixture_path.read_text())
        tree["min_read_version"] = 99
        with self.assertRaises((ArchiveReadError, pydantic.ValidationError)):
            ImageSerializationModel.model_validate(tree)

    def test_higher_major_with_low_min_read_succeeds(self):
        """A higher schema_version with low min_read_version reads silently."""
        import json as json_module

        from lsst.images._image import ImageSerializationModel

        tree = json_module.loads(self.fixture_path.read_text())
        tree["schema_version"] = "99.0.0"
        tree["min_read_version"] = 1
        # Asymmetric escape: a 99.0.0 file that declares it's safe for
        # major-1 readers reads silently.
        instance = ImageSerializationModel.model_validate(tree)
        # And gets normalised back to in-code values.
        self.assertEqual(instance.schema_version, "1.0.0")
        self.assertEqual(instance.min_read_version, 1)

    def test_absent_fields_default_to_legacy(self):
        """Stripping the version fields entirely reads with v1 defaults."""
        import json as json_module

        from lsst.images._image import ImageSerializationModel

        tree = json_module.loads(self.fixture_path.read_text())
        del tree["schema_version"]
        del tree["min_read_version"]
        del tree["schema_url"]
        instance = ImageSerializationModel.model_validate(tree)
        self.assertEqual(instance.schema_version, "1.0.0")
        self.assertEqual(instance.min_read_version, 1)


if __name__ == "__main__":
    unittest.main()
