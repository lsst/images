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

import json as json_module
import warnings
from pathlib import Path
from typing import Any, ClassVar

import pydantic
import pytest

from lsst.images import ImageSerializationModel
from lsst.images.json._output_archive import JsonOutputArchive
from lsst.images.serialization import (
    ArchiveReadError,
    ArchiveTree,
    DevelopmentSchemaWarning,
    InputArchive,
    is_development_version,
    warn_for_development_schemas,
)
from lsst.images.serialization._common import _check_compat, _check_format_version
from lsst.images.tests import check_archive_tree_class_invariants, iter_concrete_archive_tree_subclasses

SCHEMA_DIR = Path(__file__).parent / "data" / "schema_v1"


class _DummyArchiveTree(ArchiveTree):
    """Minimal concrete ArchiveTree for testing the base-class machinery."""

    SCHEMA_NAME: ClassVar[str] = "dummy"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = object

    def deserialize(
        self, archive: InputArchive[Any], **kwargs: Any
    ) -> Any:  # pragma: no cover - never invoked
        raise NotImplementedError()


class _DevTree(ArchiveTree):
    """Development-version tree double."""

    SCHEMA_NAME = "in_dev_helper_test_dev"
    SCHEMA_VERSION = "1.0.0.dev0"
    MIN_READ_VERSION = 1
    PUBLIC_TYPE = object

    def deserialize(self, archive, **kwargs):  # pragma: no cover - never invoked
        raise NotImplementedError()


class _FinalTree(ArchiveTree):
    """Finalized-version tree double."""

    SCHEMA_NAME = "in_dev_helper_test_final"
    SCHEMA_VERSION = "1.0.0"
    MIN_READ_VERSION = 1
    PUBLIC_TYPE = object

    def deserialize(self, archive, **kwargs):  # pragma: no cover - never invoked
        raise NotImplementedError()


def test_is_development_version() -> None:
    """Verify is_development_version detects development versions correctly."""
    assert is_development_version("1.0.0.dev0")
    assert is_development_version("2.3.4.dev12")
    assert not is_development_version("1.0.0")
    assert not is_development_version("2.3.4")


def test_warn_for_development_schemas_warns() -> None:
    """Verify warn_for_development_schemas warns for development schemas."""
    with pytest.warns(DevelopmentSchemaWarning, match="in_dev_helper_test_dev"):
        warn_for_development_schemas(_DevTree())


def test_warn_for_development_schemas_silent_when_finalized() -> None:
    """Verify warn_for_development_schemas is silent for finalized schemas."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_for_development_schemas(_FinalTree())


def test_silent_when_min_read_satisfied() -> None:
    """Verify no error when min_read_version equals reader major."""
    _check_compat("foo", "1.0.0", 1, "1.0.0")


def test_silent_when_on_disk_major_is_lower() -> None:
    """Verify no error when a 1.0.0 file is read by 2.0.0 code."""
    _check_compat("foo", "1.0.0", 1, "2.0.0")


def test_silent_when_on_disk_major_is_higher_but_min_read_low() -> None:
    """Verify no error when a 2.0.0 file is safe for major-1 readers."""
    _check_compat("foo", "2.0.0", 1, "1.0.0")


def test_raises_when_min_read_exceeds_reader_major() -> None:
    """Verify ArchiveReadError is raised when min_read_version exceeds
    major.
    """
    with pytest.raises(ArchiveReadError) as exc_info:
        _check_compat("foo", "2.0.0", 2, "1.0.0")
    assert "foo" in str(exc_info.value)
    assert ">= 2" in str(exc_info.value)


def test_format_version_silent_when_equal() -> None:
    """Verify no error when on-disk format version equals reader version."""
    _check_format_version("fits", 1, 1)


def test_format_version_silent_when_on_disk_lower() -> None:
    """Verify no error when on-disk format version is lower than reader's."""
    _check_format_version("fits", 1, 2)


def test_format_version_raises_when_on_disk_higher() -> None:
    """Verify ArchiveReadError when on-disk format version is too new."""
    with pytest.raises(ArchiveReadError):
        _check_format_version("fits", 2, 1)


def test_default_values_filled_from_classvars() -> None:
    """Verify default instance values are filled from class-var constants."""
    instance = _DummyArchiveTree()
    assert instance.schema_version == "1.0.0"
    assert instance.min_read_version == 1


def test_schema_url_is_computed() -> None:
    """Verify schema_url is computed from SCHEMA_NAME and SCHEMA_VERSION."""
    instance = _DummyArchiveTree()
    assert instance.schema_url == "https://images.lsst.io/schemas/dummy-1.0.0"


def test_schema_url_appears_in_dump() -> None:
    """Verify schema_url, schema_version, and min_read_version are in dumps."""
    instance = _DummyArchiveTree()
    dumped = instance.model_dump()
    assert dumped["schema_url"] == "https://images.lsst.io/schemas/dummy-1.0.0"
    assert dumped["schema_version"] == "1.0.0"
    assert dumped["min_read_version"] == 1


def test_schema_url_ignored_in_input() -> None:
    """Verify schema_url in input data is ignored; computed value is used."""
    # Pydantic's default extra='ignore' drops it from inputs.
    instance = _DummyArchiveTree.model_validate(
        {"schema_url": "https://example.com/wrong", "schema_version": "1.0.0", "min_read_version": 1}
    )
    assert instance.schema_url == "https://images.lsst.io/schemas/dummy-1.0.0"


def test_normalises_to_in_code_values() -> None:
    """Verify an older file's schema values are normalised to in-code
    values.
    """
    # An older file's values are normalised on load.
    instance = _DummyArchiveTree.model_validate({"schema_version": "0.9.0", "min_read_version": 1})
    assert instance.schema_version == "1.0.0"
    assert instance.min_read_version == 1


def test_absent_fields_default_to_legacy() -> None:
    """Verify absent version fields default to the legacy v1 values."""
    instance = _DummyArchiveTree.model_validate({})
    assert instance.schema_version == "1.0.0"
    assert instance.min_read_version == 1


def test_min_read_version_too_high_rejected() -> None:
    """Verify min_read_version higher than the reader major is rejected."""
    # Pydantic mode='after' re-raises ArchiveReadError without
    # wrapping it in ValidationError.
    with pytest.raises((ArchiveReadError, pydantic.ValidationError)):
        _DummyArchiveTree.model_validate({"schema_version": "2.0.0", "min_read_version": 2})


def test_image_schema_has_id_and_title() -> None:
    """Verify Image's serialization-mode schema has ``$id`` / ``title`` set."""
    schema = ImageSerializationModel.model_json_schema(mode="serialization")
    assert schema["$id"] == "https://images.lsst.io/schemas/image-1.0.0"
    assert schema["title"] == "image"


def test_constants_are_declared() -> None:
    """Verify all three ClassVars are declared and well-formed everywhere."""
    found = list(iter_concrete_archive_tree_subclasses())
    assert len(found) > 0
    for sub in found:
        check_archive_tree_class_invariants(sub)


def test_schema_names_unique() -> None:
    """Verify all SCHEMA_NAME values across concrete subclasses are unique."""
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
            pytest.fail(
                f"Duplicate SCHEMA_NAME {sub.SCHEMA_NAME!r}: {sub.__qualname__} vs {existing.__qualname__}"
            )
        names[sub.SCHEMA_NAME] = sub


@pytest.fixture
def fixture_path() -> Path:
    """Return the path to the committed image.json schema fixture."""
    path = SCHEMA_DIR / "image.json"
    assert path.exists()
    return path


def test_min_read_too_high_raises(fixture_path: Path) -> None:
    """Verify setting min_read_version above reader major rejects the read."""
    tree = json_module.loads(fixture_path.read_text())
    tree["min_read_version"] = 99
    with pytest.raises((ArchiveReadError, pydantic.ValidationError)):
        ImageSerializationModel.model_validate(tree)


def test_higher_major_with_low_min_read_succeeds(fixture_path: Path) -> None:
    """Verify a high schema_version with low min_read_version reads
    silently.
    """
    tree = json_module.loads(fixture_path.read_text())
    tree["schema_version"] = "99.0.0"
    tree["min_read_version"] = 1
    # Asymmetric escape: a 99.0.0 file that declares it's safe for
    # major-1 readers reads silently.
    instance = ImageSerializationModel.model_validate(tree)
    # And gets normalised back to in-code values.
    assert instance.schema_version == "1.0.0"
    assert instance.min_read_version == 1


def test_absent_fields_default_to_legacy_fixture(fixture_path: Path) -> None:
    """Verify stripping the version fields entirely reads with v1 defaults."""
    tree = json_module.loads(fixture_path.read_text())
    del tree["schema_version"]
    del tree["min_read_version"]
    del tree["schema_url"]
    instance = ImageSerializationModel.model_validate(tree)
    assert instance.schema_version == "1.0.0"
    assert instance.min_read_version == 1


class _WritableDevObject:
    _archive_default_name = None

    def serialize(self, archive):
        return _DevTree()


class _WritableFinalObject:
    _archive_default_name = None

    def serialize(self, archive):
        return _FinalTree()


def test_serialize_root_warns_for_development() -> None:
    """Verify serialize_root warns when the tree uses a development schema."""
    with pytest.warns(DevelopmentSchemaWarning, match="in_dev_helper_test_dev"):
        JsonOutputArchive().serialize_root(_WritableDevObject())


def test_serialize_root_silent_for_finalized() -> None:
    """Verify serialize_root does not warn for a finalized schema."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        JsonOutputArchive().serialize_root(_WritableFinalObject())
