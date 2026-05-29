# Schema Versioning Implementation Plan (DM-54557)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `schema_version` and `min_read_version` fields plus a computed `schema_url` to every concrete `ArchiveTree` subclass; add backend-layout (`FMTVER` / `FORMAT_VERSION`) stamps to FITS and NDF; ship reference v1 JSON fixtures and version-mismatch tests.

**Architecture:** A `mode="after"` Pydantic validator on `ArchiveTree` runs `_check_compat` (rejects when `min_read_version` exceeds the in-code reader major) and then normalises the fields to the in-code values. Each concrete subclass declares three `ClassVar`s (`SCHEMA_NAME`, `SCHEMA_VERSION`, `MIN_READ_VERSION`). The FITS primary header gains `DATAMODL` (root tree's `schema_url`) and `FMTVER` (int); the NDF top-level structure gains `.MORE.LSST.DATA_MODEL` (root tree's `schema_url`) and `.MORE.LSST.FORMAT_VERSION` (int). Container versions are integer-major-only; data-model versions are full `major.minor.patch`. v1 hard-fails on mismatch — deferred-fail substitution is design-only.

**Tech Stack:** Python 3.13+, Pydantic v2, pytest, astropy.io.fits, h5py, NDF/HDS.

**Project conventions worth knowing:**
- Run python with `.pyenv/bin/python` (project venv; system python lacks zarr).
- Run tests with `.pyenv/bin/python -m pytest tests/test_X.py -v`.
- Use top-of-file imports; resolve `pkg._submodule` to `pkg` for symbols re-exported from `__init__.py`.
- Spec file: `docs/superpowers/specs/2026-05-15-schema-versioning-design.md` — read it first if anything in this plan looks ambiguous.

---

## File Structure

**Files to create:**

| Path | Purpose |
|---|---|
| `tests/test_schema_versioning.py` | Class-invariants test, mutation tests on a fixture, computed-field tests. |
| `tests/test_schema_v1_fixtures.py` | Round-trip + presence-of-stamps for every fixture. |
| `tests/test_fits_format_version.py` | FITS `FMTVER` write/read/mismatch/absent. |
| `tests/test_ndf_format_version.py` | NDF `FORMAT_VERSION` write/read/mismatch/absent. |
| `tests/data/schema_v1/<name>.json` | Reference fixtures, one per concrete subclass. |
| `tests/data/schema_v1/README.md` | How fixtures are produced and regenerated. |
| `python/lsst/images/tests/_make_schema_fixtures.py` | Helper that builds synthetic v1 fixtures from in-memory factories. |
| `python/lsst/images/tests/_minify_for_fixtures.py` | Helper that minifies real legacy files into JSON fixtures. |
| `doc/changes/DM-54557.feature.md` | News fragment. |

**Files to modify:**

| Path | Change |
|---|---|
| `python/lsst/images/serialization/_common.py` | Add `ArchiveVersionError` (or reuse `ArchiveReadError`), `_check_compat`, `_check_format_version`, `_parse_major`; add three `ClassVar`s, `schema_version` and `min_read_version` fields, `schema_url` computed field, `_check_and_normalize_schema_version` validator on `ArchiveTree`. |
| `python/lsst/images/_image.py` | Add three `ClassVar`s to `ImageSerializationModel`. |
| `python/lsst/images/_mask.py` | Add three `ClassVar`s to `MaskSerializationModel`. |
| `python/lsst/images/_masked_image.py` | Add three `ClassVar`s to `MaskedImageSerializationModel`. |
| `python/lsst/images/_visit_image.py` | Override three `ClassVar`s on `VisitImageSerializationModel` (inherits `MaskedImageSerializationModel`). |
| `python/lsst/images/cells/_coadd.py` | Override three `ClassVar`s on `CellCoaddSerializationModel` (inherits `MaskedImageSerializationModel`). |
| `python/lsst/images/cells/_provenance.py` | Add three `ClassVar`s to `CoaddProvenanceSerializationModel`. |
| `python/lsst/images/cells/_psf.py` | Add three `ClassVar`s to `CellPointSpreadFunctionSerializationModel`. |
| `python/lsst/images/_color_image.py` | Add three `ClassVar`s to `ColorImageSerializationModel`. |
| `python/lsst/images/_backgrounds.py` | Add three `ClassVar`s to `BackgroundMapSerializationModel`. |
| `python/lsst/images/aperture_corrections.py` | Add three `ClassVar`s to `ApertureCorrectionMapSerializationModel`. |
| `python/lsst/images/cameras.py` | Add three `ClassVar`s to `DetectorSerializationModel`. |
| `python/lsst/images/psfs/_gaussian.py` | Add three `ClassVar`s to `GaussianPSFSerializationModel`. |
| `python/lsst/images/psfs/_legacy.py` | Add three `ClassVar`s to `PSFExSerializationModel`. |
| `python/lsst/images/psfs/_piff.py` | Add three `ClassVar`s to `PiffSerializationModel`. |
| `python/lsst/images/_transforms/_camera_frame_set.py` | Add three `ClassVar`s to `CameraFrameSetSerializationModel`. |
| `python/lsst/images/_transforms/_projection.py` | Add three `ClassVar`s to `ProjectionSerializationModel`. |
| `python/lsst/images/_transforms/_transform.py` | Add three `ClassVar`s to `TransformSerializationModel`. |
| `python/lsst/images/fields/_chebyshev.py` | Add three `ClassVar`s to `ChebyshevFieldSerializationModel`. |
| `python/lsst/images/fields/_spline.py` | Add three `ClassVar`s to `SplineFieldSerializationModel`. |
| `python/lsst/images/fields/_sum.py` | Add three `ClassVar`s to `SumFieldSerializationModel`. |
| `python/lsst/images/fields/_product.py` | Add three `ClassVar`s to `ProductFieldSerializationModel`. |
| `python/lsst/images/fits/_output_archive.py` | Add `_FITS_FORMAT_VERSION = 1`; in `__init__` after the existing INDXADDR/INDXSIZE/JSONADDR/JSONSIZE block, write `FMTVER`; in `add_tree(tree)` write `DATAMODL = tree.schema_url`. Replace TODO at line 128. |
| `python/lsst/images/fits/_input_archive.py` | Add `_FITS_FORMAT_VERSION = 1`; in `__init__` read+pop `FMTVER` (default 1) and run `_check_format_version`; pop `DATAMODL` (informational only). Replace TODO at line ~114. |
| `python/lsst/images/ndf/_output_archive.py` | Add `_NDF_FORMAT_VERSION = 1`; in `add_tree(tree)` write `<lsst_path>/DATA_MODEL = tree.schema_url` and `<lsst_path>/FORMAT_VERSION = _NDF_FORMAT_VERSION`. |
| `python/lsst/images/ndf/_input_archive.py` | Add `_NDF_FORMAT_VERSION = 1`; on open, read `<lsst_path>/FORMAT_VERSION` (default 1) and run `_check_format_version`. |

---

## Task Group A: Foundation in `serialization/_common.py`

These tasks set up the version-checking helpers and base-class machinery before any subclass touches them. Tests live in `tests/test_schema_versioning.py`.

### Task A1: Add `_parse_major` helper

**Files:**
- Modify: `python/lsst/images/serialization/_common.py`

- [ ] **Step 1: Append the helper near the bottom of `_common.py`**

```python
def _parse_major(version: str) -> int:
    """Return the integer major component of a major.minor.patch string.

    Raises
    ------
    ArchiveReadError
        If ``version`` is not a non-empty string of the form
        ``major.minor.patch`` with integer components.
    """
    if not isinstance(version, str) or not version:
        raise ArchiveReadError(f"Schema version {version!r} is not a non-empty string.")
    head = version.split(".", 1)[0]
    try:
        return int(head)
    except ValueError as exc:
        raise ArchiveReadError(f"Schema version {version!r} has non-integer major.") from exc
```

- [ ] **Step 2: No test yet — covered by `_check_compat` tests in A2**

### Task A2: Add `_check_compat` and `_check_format_version` helpers

**Files:**
- Modify: `python/lsst/images/serialization/_common.py`
- Test: `tests/test_schema_versioning.py`

- [ ] **Step 1: Write failing tests in `tests/test_schema_versioning.py`**

Create the file:

```python
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

import lsst.utils.tests
from lsst.images.serialization import ArchiveReadError
from lsst.images.serialization._common import _check_compat, _check_format_version


class CheckCompatTestCase(unittest.TestCase):
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


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.pyenv/bin/python -m pytest tests/test_schema_versioning.py -v`

Expected: FAIL — `ImportError: cannot import name '_check_compat'`.

- [ ] **Step 3: Implement helpers in `_common.py`**

Add after `_parse_major`:

```python
def _check_compat(
    name: str,
    on_disk_version: str,
    on_disk_min_read: int,
    in_code_version: str,
) -> None:
    """Raise `ArchiveReadError` if a tree written with the given
    schema_version/min_read_version cannot be read by the current code.

    See ``docs/superpowers/specs/2026-05-15-schema-versioning-design.md``
    §4.2 for the rule.
    """
    in_code_major = _parse_major(in_code_version)
    if on_disk_min_read > in_code_major:
        raise ArchiveReadError(
            f"{name}: tree requires reader major >= {on_disk_min_read}; "
            f"this release is {in_code_version}."
        )


def _check_format_version(name: str, on_disk: int, in_code: int) -> None:
    """Raise `ArchiveReadError` if a backend file's container layout
    version is newer than this release knows how to read.
    """
    if on_disk > in_code:
        raise ArchiveReadError(
            f"{name}: on-disk container format version {on_disk} is "
            f"newer than this release ({in_code}); cannot read."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.pyenv/bin/python -m pytest tests/test_schema_versioning.py -v`

Expected: PASS for all seven tests in `CheckCompatTestCase`.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/serialization/_common.py tests/test_schema_versioning.py
git commit -m "$(cat <<'EOF'
Add _check_compat and _check_format_version helpers (DM-54557)

The version-check primitives for the schema-versioning design.
_check_compat enforces the asymmetric min_read_version gate;
_check_format_version enforces the integer-major-only backend rule.

Generated with AI

Co-Authored-By: SLAC AI
EOF
)"
```

### Task A3: Add `schema_version`, `min_read_version`, `schema_url` to `ArchiveTree`

**Files:**
- Modify: `python/lsst/images/serialization/_common.py`
- Test: `tests/test_schema_versioning.py`

- [ ] **Step 1: Add the failing test class to `tests/test_schema_versioning.py`**

Append after `CheckCompatTestCase`:

```python
class _DummyArchiveTree(__import__("lsst.images.serialization", fromlist=["ArchiveTree"]).ArchiveTree):
    SCHEMA_NAME = "dummy"
    SCHEMA_VERSION = "1.0.0"
    MIN_READ_VERSION = 1

    def deserialize(self, archive, **kwargs):  # pragma: no cover - never invoked
        raise NotImplementedError()


class ArchiveTreeVersionFieldsTestCase(unittest.TestCase):
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
        instance = _DummyArchiveTree.model_validate(
            {"schema_version": "0.9.0", "min_read_version": 1}
        )
        self.assertEqual(instance.schema_version, "1.0.0")
        self.assertEqual(instance.min_read_version, 1)

    def test_absent_fields_default_to_legacy(self):
        instance = _DummyArchiveTree.model_validate({})
        self.assertEqual(instance.schema_version, "1.0.0")
        self.assertEqual(instance.min_read_version, 1)

    def test_min_read_version_too_high_rejected(self):
        with self.assertRaises(__import__("pydantic").ValidationError):
            _DummyArchiveTree.model_validate(
                {"schema_version": "2.0.0", "min_read_version": 2}
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.pyenv/bin/python -m pytest tests/test_schema_versioning.py::ArchiveTreeVersionFieldsTestCase -v`

Expected: FAIL — fields not yet declared on `ArchiveTree`.

- [ ] **Step 3: Modify `ArchiveTree` in `python/lsst/images/serialization/_common.py`**

Add imports near the top of the file (alongside the existing imports):

```python
from typing import ClassVar, Self
```

(Note: `ClassVar` and `Self` may already be imported; if so, just ensure they're present.)

Replace the existing `ArchiveTree` class definition with:

```python
class ArchiveTree(
    pydantic.BaseModel, ABC, ser_json_inf_nan="constants", ser_json_bytes="base64", val_json_bytes="base64"
):
    """An intermediate base class of `pydantic.BaseModel` that should be used
    for all objects that may be used as the top-level tree models written to
    archives.
    """

    SCHEMA_NAME: ClassVar[str]
    SCHEMA_VERSION: ClassVar[str]
    MIN_READ_VERSION: ClassVar[int]

    schema_version: str = pydantic.Field(
        default="1.0.0",
        description="Data-model schema version of this tree (major.minor.patch).",
    )
    min_read_version: int = pydantic.Field(
        default=1,
        description="Smallest reader major that can interpret this tree.",
    )
    metadata: dict[str, MetadataValue] = pydantic.Field(
        default_factory=dict, description="Additional unstructured metadata.", exclude_if=operator.not_
    )
    butler_info: ButlerInfo | None = pydantic.Field(
        default=None,
        description="Information about the butler dataset backed by this file.",
        exclude_if=is_none,
    )
    indirect: list[Any] = pydantic.Field(
        default_factory=list,
        description="Serialized nested objects that may be saved or read more than once.",
        exclude_if=operator.not_,
    )

    @pydantic.computed_field(description="Canonical schema URL for this tree.")  # type: ignore[prop-decorator]
    @property
    def schema_url(self) -> str:
        cls = type(self)
        return f"https://images.lsst.io/schemas/{cls.SCHEMA_NAME}-{cls.SCHEMA_VERSION}"

    @pydantic.model_validator(mode="after")
    def _check_and_normalize_schema_version(self) -> Self:
        cls = type(self)
        # ArchiveTree itself is abstract (deserialize is @abstractmethod).
        # Subclasses that haven't yet declared SCHEMA_NAME are skipped —
        # this matters during Task A3 / Group B's incremental rollout, and
        # remains a safe no-op afterwards (Task B0's class-invariants test
        # ensures every concrete subclass has the constants).
        if not hasattr(cls, "SCHEMA_NAME"):
            return self
        _check_compat(
            cls.SCHEMA_NAME,
            self.schema_version,
            self.min_read_version,
            cls.SCHEMA_VERSION,
        )
        if self.schema_version != cls.SCHEMA_VERSION:
            self.schema_version = cls.SCHEMA_VERSION
        if self.min_read_version != cls.MIN_READ_VERSION:
            self.min_read_version = cls.MIN_READ_VERSION
        return self

    @abstractmethod
    def deserialize(self, archive: InputArchive[Any], **kwargs: Any) -> Any:
        """Return the in-memory object that was serialized to this tree.

        Parameters
        ----------
        archive
            The input archive to read from.
        **kwargs
            Additional keyword arguments specific to this type.

        Raises
        ------
        ~lsst.images.serialization.InvalidParameterError
            Raised for unsupported ``**kwargs``.

        Notes
        -----
        Subclass implementations may take additional keyword-only arguments.
        Callers that invoke this method without knowing what those might be
        should catch `TypeError` and re-raise as
        `~lsst.images.serialization.InvalidParameterError` if they pass
        additional keyword arguments.
        """
        raise NotImplementedError()

    def deserialize_component(self, component: str, archive: InputArchive[Any], **kwargs: Any) -> Any:
        """Return a component in-memory object that was serialized to this
        tree.
        """
        try:
            component_model = getattr(self, component)
        except AttributeError:
            raise InvalidComponentError(
                f"Component {component!r} is not recognized by {type(self).__name__}."
            ) from None
        if component_model is None:
            return None
        if isinstance(component_model, ArchiveTree):
            return component_model.deserialize(archive, **kwargs)
        return component_model
```

(Preserve the existing docstring for `deserialize_component` — copy it verbatim from the current `_common.py`. The version above abbreviates it for plan readability.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.pyenv/bin/python -m pytest tests/test_schema_versioning.py::ArchiveTreeVersionFieldsTestCase -v`

Expected: PASS for all seven tests.

- [ ] **Step 5: Run the full existing test suite to confirm no regressions**

Run: `.pyenv/bin/python -m pytest tests/ -x --timeout=120`

Expected: PASS. The new validator silently skips subclasses that haven't yet declared `SCHEMA_NAME` (the `hasattr` guard), so existing tests exercising `ImageSerializationModel`, `MaskSerializationModel`, etc. before they get their `ClassVar`s in Group B continue to pass.

What may legitimately fail at this step: a test that compares a serialized JSON byte-for-byte and notices the new `schema_url`/`schema_version`/`min_read_version` keys at the root. Note these for Task B22 (the clean-up commit after Group B).

If unrelated tests fail (e.g. import errors in `_common.py`), STOP and fix `_common.py` before continuing.

- [ ] **Step 6: Commit**

```bash
git add python/lsst/images/serialization/_common.py tests/test_schema_versioning.py
git commit -m "$(cat <<'EOF'
Add schema_version/min_read_version/schema_url to ArchiveTree (DM-54557)

The version-check validator runs in mode='after' for performance —
pydantic-core handles field parsing in its compiled fast path, and our
Python-level callback only runs once per instance on already-parsed
values. Field defaults (schema_version='1.0.0', min_read_version=1)
provide the absence-policy values directly.

Concrete ArchiveTree subclasses must still declare SCHEMA_NAME,
SCHEMA_VERSION, MIN_READ_VERSION ClassVars; the base validator
silently skips when they're missing so the abstract base remains
usable. Subclass tests follow in subsequent tasks.

Generated with AI

Co-Authored-By: SLAC AI
EOF
)"
```

### Task A4: Inject `$id` and `title` into per-subclass JSON Schema

**Files:**
- Modify: `python/lsst/images/serialization/_common.py`
- Test: `tests/test_schema_versioning.py`

This implements §8 of the spec: a generated `model_json_schema()` for any
concrete `ArchiveTree` subclass should carry `$id` (= `schema_url`) and
`title` (= `SCHEMA_NAME`). We do this once on the base class via
`__pydantic_init_subclass__`, so no per-subclass `model_config`
boilerplate is required.

- [ ] **Step 1: Append a failing test to `tests/test_schema_versioning.py`**

```python
class JsonSchemaInjectionTestCase(unittest.TestCase):
    def test_image_schema_has_id_and_title(self):
        from lsst.images._image import ImageSerializationModel
        schema = ImageSerializationModel.model_json_schema(mode="serialization")
        self.assertEqual(schema["$id"], "https://images.lsst.io/schemas/image-1.0.0")
        self.assertEqual(schema["title"], "image")
```

- [ ] **Step 2: Run to confirm it fails**

Run: `.pyenv/bin/python -m pytest tests/test_schema_versioning.py::JsonSchemaInjectionTestCase -v`

Expected: FAIL — `KeyError: '$id'` (or the existing default title doesn't match).

- [ ] **Step 3: Add `__pydantic_init_subclass__` to `ArchiveTree`**

In `python/lsst/images/serialization/_common.py`, inside the `ArchiveTree`
class body (after the `_check_and_normalize_schema_version` validator),
add:

```python
    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        # Only concrete subclasses (those that actually declare the
        # ClassVars) get $id/title injected. Abstract intermediates
        # without a SCHEMA_NAME are skipped.
        name = cls.__dict__.get("SCHEMA_NAME")
        version = cls.__dict__.get("SCHEMA_VERSION")
        if name is None or version is None:
            return
        existing = dict(cls.model_config.get("json_schema_extra") or {})
        existing.setdefault("$id", f"https://images.lsst.io/schemas/{name}-{version}")
        existing.setdefault("title", name)
        # model_config is a TypedDict-ish dict; mutate in place.
        cls.model_config = {**cls.model_config, "json_schema_extra": existing}
```

- [ ] **Step 4: Run the new test (expected: PASS)**

Run: `.pyenv/bin/python -m pytest tests/test_schema_versioning.py::JsonSchemaInjectionTestCase -v`

Expected: PASS. (Task B1 is needed first if it isn't already done — `ImageSerializationModel` must have its `ClassVar`s set. If the test fails because `SCHEMA_NAME` isn't on `ImageSerializationModel` yet, run B1 first or skip this test until after B21.)

- [ ] **Step 5: Run the full versioning test file**

Run: `.pyenv/bin/python -m pytest tests/test_schema_versioning.py -v`

Expected: All previously-passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add python/lsst/images/serialization/_common.py tests/test_schema_versioning.py
git commit -m "$(cat <<'EOF'
Inject $id and title into per-subclass JSON Schema (DM-54557)

ArchiveTree.__pydantic_init_subclass__ populates each concrete
subclass's model_config['json_schema_extra'] with $id (=schema_url)
and title (=SCHEMA_NAME). This makes model_json_schema() output
self-identifying without per-subclass boilerplate. Implements §8 of
the schema-versioning design.

Generated with AI

Co-Authored-By: SLAC AI
EOF
)"
```

---

## Task Group B: Declare `ClassVar`s on every concrete `ArchiveTree` subclass

Each task adds three lines to one subclass. The `__pydantic_init_subclass__` work is deferred to Task A4 if needed; for now, declaring the `ClassVar`s is enough.

**Important:** `VisitImageSerializationModel` and `CellCoaddSerializationModel` *inherit* from `MaskedImageSerializationModel`. Each must explicitly redeclare its own `SCHEMA_NAME` so the URL reflects the actual subclass, not the parent. `SCHEMA_VERSION` and `MIN_READ_VERSION` may be inherited if identical, but redeclare them anyway for clarity.

### Task B0: Add the class-invariants test (will fail until B21 is done)

**Files:**
- Test: `tests/test_schema_versioning.py`

- [ ] **Step 1: Append the invariants test to `tests/test_schema_versioning.py`**

```python
class ArchiveTreeClassInvariantsTestCase(unittest.TestCase):
    """Every concrete ArchiveTree subclass must declare the version
    ClassVars and they must be well-formed."""

    @classmethod
    def _all_concrete_subclasses(cls):
        # Force-import every package module that defines a subclass.
        import lsst.images  # noqa: F401
        import lsst.images.cells  # noqa: F401
        import lsst.images.fields  # noqa: F401
        import lsst.images.psfs  # noqa: F401
        import lsst.images._transforms  # noqa: F401
        from lsst.images.serialization import ArchiveTree
        seen = set()
        stack = [ArchiveTree]
        while stack:
            kls = stack.pop()
            for sub in kls.__subclasses__():
                if sub in seen:
                    continue
                seen.add(sub)
                stack.append(sub)
                # Skip abstract subclasses (those that re-list themselves
                # as abstract). Heuristic: concrete = no abstract methods.
                if not getattr(sub, "__abstractmethods__", None):
                    yield sub

    def test_constants_are_declared(self):
        for sub in self._all_concrete_subclasses():
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
        names = {}
        for sub in self._all_concrete_subclasses():
            self.assertNotIn(sub.SCHEMA_NAME, names,
                             f"Duplicate SCHEMA_NAME {sub.SCHEMA_NAME!r}: "
                             f"{sub.__name__} vs {names.get(sub.SCHEMA_NAME)}")
            names[sub.SCHEMA_NAME] = sub.__name__
```

- [ ] **Step 2: Run the invariants test and confirm it fails**

Run: `.pyenv/bin/python -m pytest tests/test_schema_versioning.py::ArchiveTreeClassInvariantsTestCase -v`

Expected: FAIL — at least one subclass missing `SCHEMA_NAME`.

(Don't commit — leave the failing test in place; it acts as a reminder until Task B21 is done.)

### Task B1: `ImageSerializationModel`

**Files:**
- Modify: `python/lsst/images/_image.py`

- [ ] **Step 1: Add three `ClassVar` lines at the top of `ImageSerializationModel`'s body**

Locate the class (it starts at line ~499 with `class ImageSerializationModel[P: pydantic.BaseModel](ArchiveTree):`), then immediately after the class line and any docstring add:

```python
    SCHEMA_NAME: ClassVar[str] = "image"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
```

If `ClassVar` is not already imported in `_image.py`, add `from typing import ClassVar` at the top of the file (or extend an existing `from typing import ...` line).

- [ ] **Step 2: Run a sanity test**

Run: `.pyenv/bin/python -m pytest tests/test_image.py -v`

Expected: PASS.

- [ ] **Step 3: No commit yet — batch with the other subclasses (B21)**

### Task B2: `MaskSerializationModel`

**Files:**
- Modify: `python/lsst/images/_mask.py`

- [ ] **Step 1: Add three `ClassVar`s to the class body**

```python
    SCHEMA_NAME: ClassVar[str] = "mask"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
```

Add `from typing import ClassVar` if missing.

- [ ] **Step 2: Run a sanity test**

Run: `.pyenv/bin/python -m pytest tests/test_mask.py -v`

Expected: PASS.

### Task B3: `MaskedImageSerializationModel`

**Files:**
- Modify: `python/lsst/images/_masked_image.py`

- [ ] **Step 1: Add three `ClassVar`s**

```python
    SCHEMA_NAME: ClassVar[str] = "masked_image"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
```

Add `from typing import ClassVar` if missing.

- [ ] **Step 2: Run a sanity test**

Run: `.pyenv/bin/python -m pytest tests/test_masked_image.py -v`

Expected: PASS.

### Task B4: `VisitImageSerializationModel` (inherits MaskedImage)

**Files:**
- Modify: `python/lsst/images/_visit_image.py`

- [ ] **Step 1: Override three `ClassVar`s**

`VisitImageSerializationModel` inherits from `MaskedImageSerializationModel`, so it inherits `"masked_image"` by default. Override:

```python
    SCHEMA_NAME: ClassVar[str] = "visit_image"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
```

Add `from typing import ClassVar` if missing.

- [ ] **Step 2: Run sanity tests**

Run: `.pyenv/bin/python -m pytest tests/test_visit_image.py -v`

Expected: PASS.

### Task B5: `CellCoaddSerializationModel` (inherits MaskedImage)

**Files:**
- Modify: `python/lsst/images/cells/_coadd.py`

- [ ] **Step 1: Override three `ClassVar`s**

```python
    SCHEMA_NAME: ClassVar[str] = "cell_coadd"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
```

- [ ] **Step 2: Run sanity tests**

Run: `.pyenv/bin/python -m pytest tests/test_cell_coadd.py -v`

Expected: PASS.

### Task B6: `CoaddProvenanceSerializationModel`

**Files:**
- Modify: `python/lsst/images/cells/_provenance.py`

- [ ] **Step 1: Add three `ClassVar`s**

```python
    SCHEMA_NAME: ClassVar[str] = "coadd_provenance"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
```

- [ ] **Step 2: Run a sanity test (provenance is exercised by cell_coadd tests)**

Run: `.pyenv/bin/python -m pytest tests/test_cell_coadd.py -v`

Expected: PASS.

### Task B7: `CellPointSpreadFunctionSerializationModel`

**Files:**
- Modify: `python/lsst/images/cells/_psf.py`

- [ ] **Step 1: Add three `ClassVar`s**

```python
    SCHEMA_NAME: ClassVar[str] = "cell_psf"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
```

- [ ] **Step 2: Run a sanity test**

Run: `.pyenv/bin/python -m pytest tests/test_cell_coadd.py tests/test_psfs.py -v`

Expected: PASS.

### Task B8: `ColorImageSerializationModel`

**Files:**
- Modify: `python/lsst/images/_color_image.py`

- [ ] **Step 1: Add three `ClassVar`s**

```python
    SCHEMA_NAME: ClassVar[str] = "color_image"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
```

- [ ] **Step 2: Run a sanity test**

Run: `.pyenv/bin/python -m pytest tests/test_color_image.py -v`

Expected: PASS.

### Task B9: `BackgroundMapSerializationModel`

**Files:**
- Modify: `python/lsst/images/_backgrounds.py`

- [ ] **Step 1: Add three `ClassVar`s**

```python
    SCHEMA_NAME: ClassVar[str] = "background_map"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
```

- [ ] **Step 2: Run a sanity test if one exists**

Run: `.pyenv/bin/python -m pytest tests/ -k background -v`

Expected: PASS (or no tests collected — that's also fine).

### Task B10: `ApertureCorrectionMapSerializationModel`

**Files:**
- Modify: `python/lsst/images/aperture_corrections.py`

- [ ] **Step 1: Add three `ClassVar`s**

```python
    SCHEMA_NAME: ClassVar[str] = "aperture_correction_map"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
```

- [ ] **Step 2: Run a sanity test**

Run: `.pyenv/bin/python -m pytest tests/ -k aperture -v`

Expected: PASS.

### Task B11: `DetectorSerializationModel`

**Files:**
- Modify: `python/lsst/images/cameras.py`

- [ ] **Step 1: Add three `ClassVar`s**

```python
    SCHEMA_NAME: ClassVar[str] = "detector"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
```

- [ ] **Step 2: Run a sanity test**

Run: `.pyenv/bin/python -m pytest tests/test_cameras.py -v`

Expected: PASS.

### Task B12: `GaussianPSFSerializationModel`

**Files:**
- Modify: `python/lsst/images/psfs/_gaussian.py`

- [ ] **Step 1: Add three `ClassVar`s**

```python
    SCHEMA_NAME: ClassVar[str] = "gaussian_psf"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
```

- [ ] **Step 2: Run a sanity test**

Run: `.pyenv/bin/python -m pytest tests/test_psfs.py -v`

Expected: PASS.

### Task B13: `PSFExSerializationModel`

**Files:**
- Modify: `python/lsst/images/psfs/_legacy.py`

- [ ] **Step 1: Add three `ClassVar`s**

```python
    SCHEMA_NAME: ClassVar[str] = "psfex_psf"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
```

- [ ] **Step 2: Run a sanity test**

Run: `.pyenv/bin/python -m pytest tests/test_psfs.py tests/test_legacy.py -v`

Expected: PASS.

### Task B14: `PiffSerializationModel`

**Files:**
- Modify: `python/lsst/images/psfs/_piff.py`

- [ ] **Step 1: Add three `ClassVar`s**

```python
    SCHEMA_NAME: ClassVar[str] = "piff_psf"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
```

- [ ] **Step 2: Run a sanity test**

Run: `.pyenv/bin/python -m pytest tests/test_psfs.py -v`

Expected: PASS.

### Task B15: `CameraFrameSetSerializationModel`

**Files:**
- Modify: `python/lsst/images/_transforms/_camera_frame_set.py`

- [ ] **Step 1: Add three `ClassVar`s**

```python
    SCHEMA_NAME: ClassVar[str] = "camera_frame_set"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
```

- [ ] **Step 2: Run a sanity test**

Run: `.pyenv/bin/python -m pytest tests/test_transforms.py -v`

Expected: PASS.

### Task B16: `ProjectionSerializationModel`

**Files:**
- Modify: `python/lsst/images/_transforms/_projection.py`

- [ ] **Step 1: Add three `ClassVar`s**

```python
    SCHEMA_NAME: ClassVar[str] = "projection"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
```

- [ ] **Step 2: Run a sanity test**

Run: `.pyenv/bin/python -m pytest tests/test_transforms.py -v`

Expected: PASS.

### Task B17: `TransformSerializationModel`

**Files:**
- Modify: `python/lsst/images/_transforms/_transform.py`

- [ ] **Step 1: Add three `ClassVar`s**

```python
    SCHEMA_NAME: ClassVar[str] = "transform"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
```

- [ ] **Step 2: Run a sanity test**

Run: `.pyenv/bin/python -m pytest tests/test_transforms.py -v`

Expected: PASS.

### Task B18: `ChebyshevFieldSerializationModel`

**Files:**
- Modify: `python/lsst/images/fields/_chebyshev.py`

- [ ] **Step 1: Add three `ClassVar`s**

```python
    SCHEMA_NAME: ClassVar[str] = "chebyshev_field"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
```

- [ ] **Step 2: Run a sanity test**

Run: `.pyenv/bin/python -m pytest tests/test_fields.py -v`

Expected: PASS.

### Task B19: `SplineFieldSerializationModel`

**Files:**
- Modify: `python/lsst/images/fields/_spline.py`

- [ ] **Step 1: Add three `ClassVar`s**

```python
    SCHEMA_NAME: ClassVar[str] = "spline_field"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
```

- [ ] **Step 2: Run a sanity test**

Run: `.pyenv/bin/python -m pytest tests/test_fields.py -v`

Expected: PASS.

### Task B20: `SumFieldSerializationModel`

**Files:**
- Modify: `python/lsst/images/fields/_sum.py`

- [ ] **Step 1: Add three `ClassVar`s**

```python
    SCHEMA_NAME: ClassVar[str] = "sum_field"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
```

- [ ] **Step 2: Run a sanity test**

Run: `.pyenv/bin/python -m pytest tests/test_fields.py -v`

Expected: PASS.

### Task B21: `ProductFieldSerializationModel` and full-suite checkpoint

**Files:**
- Modify: `python/lsst/images/fields/_product.py`
- Test: `tests/test_schema_versioning.py`

- [ ] **Step 1: Add three `ClassVar`s**

```python
    SCHEMA_NAME: ClassVar[str] = "product_field"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
```

- [ ] **Step 2: Run the invariants test (now expected to PASS)**

Run: `.pyenv/bin/python -m pytest tests/test_schema_versioning.py::ArchiveTreeClassInvariantsTestCase -v`

Expected: PASS — every concrete subclass has the three constants set.

- [ ] **Step 3: Run the full test suite to confirm no regressions**

Run: `.pyenv/bin/python -m pytest tests/ -x --timeout=300`

Expected: PASS. If a JSON round-trip test fails because the serialized form now contains extra `schema_url` / `schema_version` / `min_read_version` keys, capture which tests and fix them in Task B22 (a single clean-up commit).

- [ ] **Step 4: Commit B1–B21 together**

```bash
git add python/lsst/images/_image.py python/lsst/images/_mask.py python/lsst/images/_masked_image.py \
    python/lsst/images/_visit_image.py python/lsst/images/cells/_coadd.py \
    python/lsst/images/cells/_provenance.py python/lsst/images/cells/_psf.py \
    python/lsst/images/_color_image.py python/lsst/images/_backgrounds.py \
    python/lsst/images/aperture_corrections.py python/lsst/images/cameras.py \
    python/lsst/images/psfs/_gaussian.py python/lsst/images/psfs/_legacy.py \
    python/lsst/images/psfs/_piff.py python/lsst/images/_transforms/_camera_frame_set.py \
    python/lsst/images/_transforms/_projection.py python/lsst/images/_transforms/_transform.py \
    python/lsst/images/fields/_chebyshev.py python/lsst/images/fields/_spline.py \
    python/lsst/images/fields/_sum.py python/lsst/images/fields/_product.py \
    tests/test_schema_versioning.py
git commit -m "$(cat <<'EOF'
Declare SCHEMA_NAME/SCHEMA_VERSION/MIN_READ_VERSION on every concrete ArchiveTree subclass (DM-54557)

Each of the 21 concrete ArchiveTree subclasses now carries the three
ClassVars required by the schema-versioning design. The base
validator in ArchiveTree picks them up automatically.

VisitImage and CellCoadd inherit from MaskedImage but redeclare their
own SCHEMA_NAME so the schema_url reflects the actual subclass.

Adds a class-invariants test that asserts all 21 concrete subclasses
declare the constants and that all SCHEMA_NAME values are unique.

Generated with AI

Co-Authored-By: SLAC AI
EOF
)"
```

### Task B22: Fix any round-trip tests broken by extra JSON keys

**Files:**
- Modify: any tests broken by the extra keys (varies by failure list).

- [ ] **Step 1: Re-run the suite**

Run: `.pyenv/bin/python -m pytest tests/ -x --timeout=300`

If everything passes, this task is a no-op — skip to the next group.

- [ ] **Step 2: For each failure**

Most existing JSON round-trip tests use `assert_*_equal` helpers that operate on in-memory objects, not raw JSON bytes. Those are unaffected. If a test compares serialized JSON byte-for-byte against an expected fixture string, update the expected fixture to include the three new keys at the top.

If a test compares two parsed-then-re-dumped trees, no change is needed.

- [ ] **Step 3: Commit fixes individually if any**

```bash
git add tests/<file>.py
git commit -m "$(cat <<'EOF'
Update <test> fixtures for new schema-version keys (DM-54557)

Generated with AI

Co-Authored-By: SLAC AI
EOF
)"
```

---

## Task Group C: FITS container version (`FMTVER` + `DATAMODL`)

### Task C1: Failing test for FITS write+read of new keywords

**Files:**
- Test: `tests/test_fits_format_version.py` (create)

- [ ] **Step 1: Create the test file**

```python
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

import os
import tempfile
import unittest

import astropy.io.fits

import lsst.utils.tests
from lsst.images import Box, DetectorFrame, Image
from lsst.images.fits import FitsInputArchive, FitsOutputArchive
from lsst.images.serialization import ArchiveReadError


def _write_simple_image_fits(path: str) -> None:
    image = Image(0.0, shape=(4, 4), dtype="float32")
    with FitsOutputArchive.open(path) as archive:
        tree = image.serialize(archive)
        archive.add_tree(tree)


class FitsFormatVersionTestCase(unittest.TestCase):
    def test_write_emits_fmtver_and_datamodl(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.fits")
            _write_simple_image_fits(path)
            with astropy.io.fits.open(path) as hdul:
                self.assertEqual(hdul[0].header["FMTVER"], 1)
                self.assertEqual(
                    hdul[0].header["DATAMODL"],
                    "https://images.lsst.io/schemas/image-1.0.0",
                )

    def test_read_succeeds_when_fmtver_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.fits")
            _write_simple_image_fits(path)
            # Should not raise.
            with FitsInputArchive.open(path):
                pass

    def test_read_fails_when_fmtver_too_high(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.fits")
            _write_simple_image_fits(path)
            # Bump the on-disk FMTVER beyond what we support.
            with astropy.io.fits.open(path, mode="update") as hdul:
                hdul[0].header["FMTVER"] = 2
                hdul.flush()
            with self.assertRaises(ArchiveReadError):
                with FitsInputArchive.open(path):
                    pass

    def test_read_succeeds_when_fmtver_absent(self):
        # Simulate a legacy file by stripping FMTVER.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.fits")
            _write_simple_image_fits(path)
            with astropy.io.fits.open(path, mode="update") as hdul:
                if "FMTVER" in hdul[0].header:
                    del hdul[0].header["FMTVER"]
                if "DATAMODL" in hdul[0].header:
                    del hdul[0].header["DATAMODL"]
                hdul.flush()
            with FitsInputArchive.open(path):
                pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.pyenv/bin/python -m pytest tests/test_fits_format_version.py -v`

Expected: FAIL — `KeyError: 'FMTVER'` (or similar).

### Task C2: Write `FMTVER` and `DATAMODL` in `FitsOutputArchive`

**Files:**
- Modify: `python/lsst/images/fits/_output_archive.py`

- [ ] **Step 1: Add module-level constant near the top of the file (after imports)**

```python
_FITS_FORMAT_VERSION = 1
"""Container layout version for files written by FitsOutputArchive.

Bumps when the on-disk FITS layout (HDU placement, INDX/JSON keyword
schema) changes. Independent of any data-model SCHEMA_VERSION.
"""
```

- [ ] **Step 2: Replace the TODO at line ~128 in `__init__`**

Find:

```python
        # TODO: add subformat description and version to primary HDU.
        self._primary_hdu.header.set("INDXADDR", 0, "Offset in bytes to the HDU index.")
```

Replace with:

```python
        self._primary_hdu.header.set(
            "FMTVER", _FITS_FORMAT_VERSION, "FITS container layout version."
        )
        self._primary_hdu.header.set("INDXADDR", 0, "Offset in bytes to the HDU index.")
```

(`DATAMODL` is set later in `add_tree(tree)` because it depends on the root tree's `schema_url`.)

- [ ] **Step 3: Modify `add_tree` to write `DATAMODL`**

Locate `def add_tree(self, tree: ArchiveTree) -> None:` (around line 340) and at the start of the method body, before constructing `json_hdu`, add:

```python
        self._primary_hdu.header.set(
            "DATAMODL", tree.schema_url, "Top-level data-model schema URL."
        )
```

- [ ] **Step 4: Run tests to confirm write tests pass**

Run: `.pyenv/bin/python -m pytest tests/test_fits_format_version.py::FitsFormatVersionTestCase::test_write_emits_fmtver_and_datamodl tests/test_fits_format_version.py::FitsFormatVersionTestCase::test_read_succeeds_when_fmtver_matches -v`

Expected: PASS for these two; the mismatch and absent tests still FAIL until C3.

### Task C3: Read+check `FMTVER` in `FitsInputArchive`

**Files:**
- Modify: `python/lsst/images/fits/_input_archive.py`

- [ ] **Step 1: Add module-level constant**

```python
_FITS_FORMAT_VERSION = 1
"""Container layout version this release of FitsInputArchive understands."""
```

- [ ] **Step 2: Replace the TODO around line 114 in `__init__`**

Find the existing block:

```python
        self._primary_hdu = astropy.io.fits.PrimaryHDU.readfrom(stream)
        # TODO: read and strip subformat declaration and version, once we start
        # writing those.
        #
        # TODO: do some basic checks that the file format conforms to our
        # expectations (e.g. primary HDU should have no data).
        #
        # Read and strip the addresses and sizes from the headers.  We don't
        # actually need the index address because we always want to read the
        # JSON HDU, too, and the index HDU is always the next one (but this
        # could change in the future).
        json_address: int = self._primary_hdu.header.pop("JSONADDR")
```

Replace the first TODO block with the version check:

```python
        self._primary_hdu = astropy.io.fits.PrimaryHDU.readfrom(stream)
        on_disk_fmtver: int = self._primary_hdu.header.pop("FMTVER", 1)
        # DATAMODL is informational only on read; the JSON tree's
        # schema_version/min_read_version drive data-model checks.
        self._primary_hdu.header.pop("DATAMODL", None)
        _check_format_version("fits", on_disk_fmtver, _FITS_FORMAT_VERSION)
        # TODO: do some basic checks that the file format conforms to our
        # expectations (e.g. primary HDU should have no data).
        #
        # Read and strip the addresses and sizes from the headers.  We don't
        # actually need the index address because we always want to read the
        # JSON HDU, too, and the index HDU is always the next one (but this
        # could change in the future).
        json_address: int = self._primary_hdu.header.pop("JSONADDR")
```

Add the import near the top of the file:

```python
from lsst.images.serialization._common import _check_format_version
```

- [ ] **Step 3: Run all tests for this module**

Run: `.pyenv/bin/python -m pytest tests/test_fits_format_version.py -v`

Expected: PASS for all four tests.

- [ ] **Step 4: Run the wider test suite to confirm no regressions**

Run: `.pyenv/bin/python -m pytest tests/ -x --timeout=300`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/fits/_output_archive.py python/lsst/images/fits/_input_archive.py \
    tests/test_fits_format_version.py
git commit -m "$(cat <<'EOF'
Stamp FITS files with FMTVER and DATAMODL (DM-54557)

FitsOutputArchive now writes:
  FMTVER  = 1                                    (container version)
  DATAMODL = <root tree schema_url>              (e.g. .../image-1.0.0)

FitsInputArchive reads FMTVER, runs the integer-major check, and
discards DATAMODL (informational only — the JSON tree's fields drive
data-model compatibility). Absent FMTVER defaults to 1 so legacy files
continue to read.

Generated with AI

Co-Authored-By: SLAC AI
EOF
)"
```

---

## Task Group D: NDF container version (`FORMAT_VERSION` + `DATA_MODEL`)

### Task D1: Failing test for NDF write+read

**Files:**
- Test: `tests/test_ndf_format_version.py` (create)

- [ ] **Step 1: Create the test file**

```python
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

import os
import tempfile
import unittest

import lsst.utils.tests
from lsst.images import Image
from lsst.images.serialization import ArchiveReadError

try:
    import h5py  # noqa: F401

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False


@unittest.skipUnless(HAVE_H5PY, "NDF backend requires h5py")
class NdfFormatVersionTestCase(unittest.TestCase):
    def _write_simple_ndf(self, path: str) -> None:
        from lsst.images.ndf import NdfOutputArchive
        image = Image(0.0, shape=(4, 4), dtype="float32")
        with NdfOutputArchive.open(path) as archive:
            tree = image.serialize(archive)
            archive.add_tree(tree)

    def _data_model_path(self) -> str:
        return "/MORE/LSST/DATA_MODEL"

    def _format_version_path(self) -> str:
        return "/MORE/LSST/FORMAT_VERSION"

    def test_write_emits_data_model_and_format_version(self):
        import h5py
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.sdf")
            self._write_simple_ndf(path)
            with h5py.File(path, "r") as f:
                # The NDF backend uses HDS-on-HDF5; the components live as
                # dataset entries under /MORE/LSST.
                self.assertIn("FORMAT_VERSION", f["/MORE/LSST"])
                self.assertIn("DATA_MODEL", f["/MORE/LSST"])

    def test_read_succeeds_when_format_version_matches(self):
        from lsst.images.ndf import NdfInputArchive
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.sdf")
            self._write_simple_ndf(path)
            with NdfInputArchive.open(path):
                pass

    def test_read_fails_when_format_version_too_high(self):
        import h5py
        from lsst.images.ndf import NdfInputArchive
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.sdf")
            self._write_simple_ndf(path)
            with h5py.File(path, "r+") as f:
                del f["/MORE/LSST/FORMAT_VERSION"]
                f["/MORE/LSST"].create_dataset("FORMAT_VERSION", data=2)
            with self.assertRaises(ArchiveReadError):
                with NdfInputArchive.open(path):
                    pass

    def test_read_succeeds_when_format_version_absent(self):
        import h5py
        from lsst.images.ndf import NdfInputArchive
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.sdf")
            self._write_simple_ndf(path)
            with h5py.File(path, "r+") as f:
                if "FORMAT_VERSION" in f["/MORE/LSST"]:
                    del f["/MORE/LSST/FORMAT_VERSION"]
                if "DATA_MODEL" in f["/MORE/LSST"]:
                    del f["/MORE/LSST/DATA_MODEL"]
            with NdfInputArchive.open(path):
                pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.pyenv/bin/python -m pytest tests/test_ndf_format_version.py -v`

Expected: FAIL — keys not yet written.

> Note: the exact h5py paths for HDS primitives may differ from what's shown above. If `f["/MORE/LSST"]` does not directly expose `FORMAT_VERSION` as an h5py-readable subkey, look at how existing NDF tests inspect on-disk components — see `tests/test_ndf_output_archive.py` for the pattern. Adjust the test assertions to match the actual layout. The important behavior is "the value is present and readable by `NdfInputArchive`."

### Task D2: Write `DATA_MODEL` and `FORMAT_VERSION` in `NdfOutputArchive`

**Files:**
- Modify: `python/lsst/images/ndf/_output_archive.py`

- [ ] **Step 1: Add module-level constant near the top**

```python
_NDF_FORMAT_VERSION = 1
"""Container layout version for files written by NdfOutputArchive."""
```

- [ ] **Step 2: Modify `add_tree`**

Locate `def add_tree(...)` (around line 333). Find the line `lsst = self._ensure_model_structure(self._lsst_path)` (currently around line 372). Immediately after the existing `lsst.children["JSON"] = ...` assignment, add the two new components:

```python
        import numpy as np
        lsst.children["DATA_MODEL"] = HdsPrimitive.char_scalar(
            tree.schema_url, width=max(80, len(tree.schema_url))
        )
        lsst.children["FORMAT_VERSION"] = HdsPrimitive.array(
            np.array(_NDF_FORMAT_VERSION, dtype=np.int32)
        )
```

The `HdsPrimitive` constructors used here are confirmed in
`python/lsst/images/ndf/_model.py`: `char_scalar(text, width=...)` for
the URL string, and `array(np.ndarray)` with a 0-d `int32` array for
the integer version. (The 0-d array maps to an HDS scalar `_INTEGER`.)
Add `import numpy as np` at the top of `_output_archive.py` if it's
not already imported.

- [ ] **Step 3: Run write tests**

Run: `.pyenv/bin/python -m pytest tests/test_ndf_format_version.py::NdfFormatVersionTestCase::test_write_emits_data_model_and_format_version -v`

Expected: PASS.

### Task D3: Read+check `FORMAT_VERSION` in `NdfInputArchive`

**Files:**
- Modify: `python/lsst/images/ndf/_input_archive.py`

- [ ] **Step 1: Add module-level constant**

```python
_NDF_FORMAT_VERSION = 1
"""Container layout version this release of NdfInputArchive understands."""
```

- [ ] **Step 2: Add a check to `__init__`**

After `self._read_opaque_fits_metadata()` in `__init__` (line ~74), add:

```python
        self._check_format_version()
```

Then add a private helper method to the class:

```python
    def _check_format_version(self) -> None:
        on_disk = 1
        for prefix in ("/MORE/LSST", "/LSST"):
            path = f"{prefix}/FORMAT_VERSION"
            if self._has_model_path(path):
                primitive = self._get_primitive(path)
                # HdsPrimitive.read_array() returns the numpy array we
                # wrote in D2; .item() unwraps the 0-d array to a Python
                # int.
                on_disk = int(primitive.read_array().item())
                break
        _check_format_version("ndf", on_disk, _NDF_FORMAT_VERSION)
```

Add the import:

```python
from lsst.images.serialization._common import _check_format_version
```

- [ ] **Step 3: Run all NDF format-version tests**

Run: `.pyenv/bin/python -m pytest tests/test_ndf_format_version.py -v`

Expected: PASS for all four tests.

- [ ] **Step 4: Run the wider NDF test suite to confirm no regressions**

Run: `.pyenv/bin/python -m pytest tests/test_ndf_*.py -x --timeout=300`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/ndf/_output_archive.py python/lsst/images/ndf/_input_archive.py \
    tests/test_ndf_format_version.py
git commit -m "$(cat <<'EOF'
Stamp NDF files with DATA_MODEL and FORMAT_VERSION (DM-54557)

NdfOutputArchive now writes:
  /MORE/LSST/DATA_MODEL      = <root tree schema_url>
  /MORE/LSST/FORMAT_VERSION  = 1

NdfInputArchive reads FORMAT_VERSION on open and runs the
integer-major check; absent component defaults to 1 so legacy files
continue to read. DATA_MODEL is informational only (the JSON tree's
fields drive data-model compatibility).

Generated with AI

Co-Authored-By: SLAC AI
EOF
)"
```

---

## Task Group E: Reference v1 fixtures and round-trip tests

### Task E1: Add the synthetic-fixture helper

**Files:**
- Create: `python/lsst/images/tests/_make_schema_fixtures.py`

- [ ] **Step 1: Create the helper**

```python
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

"""Generate v1 reference JSON fixtures for the schema-versioning tests.

Run from the repo root via:

    .pyenv/bin/python -m lsst.images.tests._make_schema_fixtures

This is a developer tool, not invoked from CI. It overwrites everything
under ``tests/data/schema_v1/`` (excluding the ``legacy/`` subdirectory)
so it should only be run when intentionally regenerating fixtures (e.g.
after a schema_version bump).
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

from lsst.images.json import JsonOutputArchive
from lsst.images.serialization import ArchiveTree


FIXTURE_DIR = pathlib.Path(__file__).resolve().parents[4] / "tests" / "data" / "schema_v1"


def _build_image_tree() -> dict:
    """Construct a minimal Image and serialize via JsonOutputArchive."""
    from lsst.images import Image
    image = Image(0.0, shape=(4, 4), dtype="float32")
    archive = JsonOutputArchive()
    tree = image.serialize(archive)
    archive.finish(tree)
    return tree.model_dump(mode="json")


# Add similar _build_*_tree helpers for every concrete subclass. Pattern:
#   1. Construct a minimal in-memory object using existing test factories
#      (see python/lsst/images/tests/_creation.py and _roundtrip.py for
#      patterns that produce small valid instances).
#   2. Open a JsonOutputArchive context, call obj.serialize(archive), and
#      add the tree.
#   3. Return the JSON-decodable root tree as a dict (so we can pretty-
#      print it with sort_keys=False to preserve declaration order).


_BUILDERS: dict[str, callable] = {
    "image": _build_image_tree,
    # Add one entry per concrete subclass. SCHEMA_NAME is the key; the
    # value is a no-arg builder that returns a JSON-decodable dict.
}


def main() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, builder in _BUILDERS.items():
        out = FIXTURE_DIR / f"{name}.json"
        tree = builder()
        out.write_text(json.dumps(tree, indent=2, sort_keys=False) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

The `JsonOutputArchive` pattern (confirmed by reading `python/lsst/images/json/_output_archive.py`):

1. `archive = JsonOutputArchive()` — no context manager.
2. `tree = obj.serialize(archive)` — produces an `ArchiveTree`.
3. `archive.finish(tree)` — fills `tree.indirect`.
4. `tree.model_dump(mode="json")` — dict ready for `json.dumps`.

- [ ] **Step 2: Add no test for the helper itself — its output is verified by Task E2**

### Task E2: Implement and run the helper for all 21 subclasses

**Files:**
- Modify: `python/lsst/images/tests/_make_schema_fixtures.py`
- Create: `tests/data/schema_v1/*.json` (21 files)
- Create: `tests/data/schema_v1/README.md`

- [ ] **Step 1: Implement a `_build_*_tree` function for each subclass**

For each `SCHEMA_NAME`, study existing tests and `python/lsst/images/tests/_creation.py` / `_roundtrip.py` to find the minimal way to construct a valid instance. Generally:

- `image`, `mask`, `masked_image`, `visit_image`, `cell_coadd`, `color_image`: small array-backed object.
- `gaussian_psf`, `piff_psf`, `psfex_psf`, `cell_psf`: existing test PSF factories.
- `chebyshev_field`, `spline_field`, `sum_field`, `product_field`: existing `tests/test_fields.py` factories.
- `projection`, `transform`, `camera_frame_set`: `make_random_projection` from `_creation.py`, etc.
- `detector`, `aperture_correction_map`, `background_map`, `coadd_provenance`: see corresponding `tests/test_*.py`.

Each builder should:

1. Construct the in-memory object using the smallest valid configuration.
2. Open `JsonOutputArchive.open()` as a context manager.
3. Serialize.
4. Read back the resulting JSON dict (use `tree.model_dump(mode="json")` after `add_tree`).
5. Return the dict.

- [ ] **Step 2: Run the helper**

Run: `.pyenv/bin/python -m lsst.images.tests._make_schema_fixtures`

Expected: 21 JSON files written under `tests/data/schema_v1/`.

- [ ] **Step 3: Manually inspect a couple of fixtures**

```bash
head -20 tests/data/schema_v1/image.json
head -30 tests/data/schema_v1/visit_image.json
```

Confirm that each fixture has `schema_url`, `schema_version`, `min_read_version` near the top of the JSON object.

- [ ] **Step 4: Write the README**

`tests/data/schema_v1/README.md`:

```markdown
# v1 schema fixtures

Reference JSON fixtures for every concrete `ArchiveTree` subclass,
exercising `schema_version`, `min_read_version`, and `schema_url`
stamping at the v1 release.

## Regenerating

Synthetic fixtures (everything in this directory excluding `legacy/`) are
regenerated by:

    .pyenv/bin/python -m lsst.images.tests._make_schema_fixtures

This overwrites all `*.json` files at the top level of this directory.

## Legacy fixtures

`legacy/` holds fixtures derived from real on-disk files via
`_minify_for_fixtures.py`. Each file documents its source path and the
git rev of the helper that produced it. To regenerate one, run:

    .pyenv/bin/python -c "from lsst.images.tests._minify_for_fixtures import minify; minify('<input>', '<output>')"

(See `_minify_for_fixtures.py` for per-type minify rules.)
```

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/tests/_make_schema_fixtures.py tests/data/schema_v1/
git commit -m "$(cat <<'EOF'
Add v1 reference JSON fixtures and synthetic-fixture helper (DM-54557)

One JSON fixture per concrete ArchiveTree subclass, generated by
_make_schema_fixtures.py from minimal in-memory factories. Each
fixture is pretty-printed for readable git diffs.

Generated with AI

Co-Authored-By: SLAC AI
EOF
)"
```

### Task E3: Round-trip and stamp-presence test for fixtures

**Files:**
- Test: `tests/test_schema_v1_fixtures.py` (create)

- [ ] **Step 1: Create the test file**

```python
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

import json
import pathlib
import unittest

import lsst.utils.tests
from lsst.images.tests._make_schema_fixtures import _BUILDERS, FIXTURE_DIR


class SchemaV1FixturesTestCase(unittest.TestCase):
    """Every fixture under tests/data/schema_v1/ must:

    1. Exist (parallel to a builder in _make_schema_fixtures._BUILDERS).
    2. Carry schema_url, schema_version, min_read_version at the root.
    3. Have schema_url matching https://images.lsst.io/schemas/<name>-<version>.
    4. Round-trip stably: re-serializing the tree built from the fixture
       produces equal JSON (modulo dict ordering).
    """

    def test_every_builder_has_a_fixture(self):
        for name in _BUILDERS:
            with self.subTest(name=name):
                path = FIXTURE_DIR / f"{name}.json"
                self.assertTrue(path.exists(), f"{path} missing — run _make_schema_fixtures")

    def test_fixture_has_top_level_stamps(self):
        for name in _BUILDERS:
            with self.subTest(name=name):
                path = FIXTURE_DIR / f"{name}.json"
                tree = json.loads(path.read_text())
                self.assertIn("schema_url", tree)
                self.assertIn("schema_version", tree)
                self.assertIn("min_read_version", tree)

    def test_fixture_url_matches_name_and_version(self):
        for name in _BUILDERS:
            with self.subTest(name=name):
                path = FIXTURE_DIR / f"{name}.json"
                tree = json.loads(path.read_text())
                expected = (
                    f"https://images.lsst.io/schemas/{name}-{tree['schema_version']}"
                )
                self.assertEqual(tree["schema_url"], expected)


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
```

- [ ] **Step 2: Run tests**

Run: `.pyenv/bin/python -m pytest tests/test_schema_v1_fixtures.py -v`

Expected: PASS for all 63 sub-tests (21 fixtures × 3 assertions).

- [ ] **Step 3: Commit**

```bash
git add tests/test_schema_v1_fixtures.py
git commit -m "$(cat <<'EOF'
Add round-trip and stamp-presence tests for v1 fixtures (DM-54557)

Asserts every fixture exists, has schema_url/schema_version/
min_read_version at the root, and the URL matches the
SCHEMA_NAME-SCHEMA_VERSION pattern.

Generated with AI

Co-Authored-By: SLAC AI
EOF
)"
```

### Task E4: Mutation tests for version mismatch

**Files:**
- Modify: `tests/test_schema_versioning.py`

- [ ] **Step 1: Append a mutation-test class**

```python
class FixtureMutationTestCase(unittest.TestCase):
    """Mutate a fixture in-memory and verify the read behavior."""

    def setUp(self):
        from lsst.images.tests._make_schema_fixtures import FIXTURE_DIR
        self.fixture_path = FIXTURE_DIR / "image.json"
        self.assertTrue(self.fixture_path.exists())

    def test_min_read_too_high_raises(self):
        from lsst.images import ImageSerializationModel
        from lsst.images.serialization import ArchiveReadError
        import json
        tree = json.loads(self.fixture_path.read_text())
        tree["min_read_version"] = 99
        with self.assertRaises((ArchiveReadError, __import__("pydantic").ValidationError)):
            ImageSerializationModel.model_validate(tree)

    def test_higher_major_with_low_min_read_succeeds(self):
        from lsst.images import ImageSerializationModel
        import json
        tree = json.loads(self.fixture_path.read_text())
        tree["schema_version"] = "99.0.0"
        tree["min_read_version"] = 1
        # Asymmetric escape: a 99.0.0 file that declares it's safe for
        # major-1 readers reads silently.
        instance = ImageSerializationModel.model_validate(tree)
        # And gets normalised back to in-code values.
        self.assertEqual(instance.schema_version, "1.0.0")
        self.assertEqual(instance.min_read_version, 1)

    def test_absent_fields_default_to_legacy(self):
        from lsst.images import ImageSerializationModel
        import json
        tree = json.loads(self.fixture_path.read_text())
        del tree["schema_version"]
        del tree["min_read_version"]
        del tree["schema_url"]
        instance = ImageSerializationModel.model_validate(tree)
        self.assertEqual(instance.schema_version, "1.0.0")
        self.assertEqual(instance.min_read_version, 1)
```

> Note: adjust `from lsst.images import ImageSerializationModel` to wherever `ImageSerializationModel` is publicly exported. If it's not exported, import from `lsst.images._image`.

- [ ] **Step 2: Run mutation tests**

Run: `.pyenv/bin/python -m pytest tests/test_schema_versioning.py::FixtureMutationTestCase -v`

Expected: PASS for all three.

- [ ] **Step 3: Commit**

```bash
git add tests/test_schema_versioning.py
git commit -m "$(cat <<'EOF'
Add fixture-mutation tests for version mismatch (DM-54557)

Mutate a fixture in-memory and verify:
- min_read_version above reader major is rejected.
- A higher schema_version major with a low min_read_version reads
  silently (the asymmetric escape — exercises the design's whole
  reason for two version fields).
- Absent fields default to 1.0.0/1 (legacy compatibility).

Generated with AI

Co-Authored-By: SLAC AI
EOF
)"
```

---

## Task Group F: Minify helper for legacy files

### Task F1: Stub the helper with one type implemented

**Files:**
- Create: `python/lsst/images/tests/_minify_for_fixtures.py`

- [ ] **Step 1: Create the helper module**

```python
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

"""Minify a real on-disk archive into a small JSON test fixture.

Reads a FITS or NDF file via the appropriate input archive, takes a
small subset of the in-memory object, and writes JSON via
JsonOutputArchive. Used to populate ``tests/data/schema_v1/legacy/``
with derived-from-real test data that exercises the full read path
including the absence-of-stamp legacy default.

Per top-level type the subset rule is:

  Image, Mask, MaskedImage     crop to ~16x16 px
  VisitImage                   crop image, drop secondary metadata
  ColorImage                   crop all bands
  CellCoadd                    TODO: subset cells (need >=4 cells; the
                               outer-ring problem of inputs/PSFs that
                               overlap kept cells is unsolved). A
                               candidate fallback is to morph cells in
                               place — not an accurate subset but
                               sufficient for testing.
  Detector, CameraFrameSet     keep one detector / one frame-set
  BackgroundMap, ApertureCorrectionMap
                               keep a single field/region
  *PSF, *Field, *Transform     already small; copy through

Run interactively:

    .pyenv/bin/python -c "
    from lsst.images.tests._minify_for_fixtures import minify
    minify('/path/to/real.fits', 'tests/data/schema_v1/legacy/foo.json')
    "
"""

from __future__ import annotations

import json
import pathlib

from lsst.images.json import JsonOutputArchive


def minify(in_path: str, out_path: str) -> None:
    """Read a real archive at ``in_path``, take a small subset, and write JSON.

    The dispatch is by file extension and root-tree type. Add a new
    branch when adding support for a new top-level type.
    """
    from lsst.images.fits import FitsInputArchive
    if in_path.endswith(".fits") or in_path.endswith(".fits.gz"):
        with FitsInputArchive.open(in_path) as archive:
            obj, _, _ = _read_root(archive)
    elif in_path.endswith(".sdf") or in_path.endswith(".ndf"):
        from lsst.images.ndf import NdfInputArchive
        with NdfInputArchive.open(in_path) as archive:
            obj, _, _ = _read_root(archive)
    else:
        raise ValueError(f"Unrecognised file extension: {in_path}")

    minified = _minify_object(obj)
    _write_json(minified, out_path)


def _read_root(archive):
    """Return (obj, metadata, butler_info) for the archive's root tree."""
    # Adapt to whatever public read API exists. See per-type read helpers in
    # tests/test_*.py for the pattern (e.g. Image.read_fits).
    raise NotImplementedError("Implement _read_root once we have a real legacy file to point at.")


def _minify_object(obj):
    """Dispatch on type to pick a small subset."""
    from lsst.images import Image, Mask, MaskedImage
    if isinstance(obj, (Image, Mask, MaskedImage)):
        return _minify_pixels(obj, max_size=16)
    # TODO: CellCoadd, Detector, CameraFrameSet, ColorImage, BackgroundMap,
    # ApertureCorrectionMap. CellCoadd subsetting is the hard one — see the
    # module docstring.
    raise NotImplementedError(f"No minify rule for {type(obj).__name__}")


def _minify_pixels(obj, max_size: int):
    """Crop an image-like object to (max_size x max_size) at the origin."""
    # Adapt to the actual cropping API on Image / Mask / MaskedImage.
    # See tests/test_image.py for examples.
    raise NotImplementedError("Implement crop using the actual subset API.")


def _write_json(obj, out_path: str) -> None:
    out = pathlib.Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with JsonOutputArchive.open() as archive:
        tree = obj.serialize(archive)
        archive.add_tree(tree)
        # See _make_schema_fixtures for the equivalent dump pattern.
        out.write_text(json.dumps(tree.model_dump(mode="json"), indent=2, sort_keys=False) + "\n")
```

- [ ] **Step 2: No test in this task — the helper is exercised manually**

The minify helper is intentionally left as a stub for everything except the simplest types. Filling it out requires real on-disk legacy files, which the implementer or downstream user will need to point at.

The CellCoadd implementation is flagged as a TODO inside the module docstring per the spec. Future contributors will pick a strategy (subset vs morph in place) when they have a concrete legacy file to test against.

- [ ] **Step 3: Commit the stub**

```bash
git add python/lsst/images/tests/_minify_for_fixtures.py
git commit -m "$(cat <<'EOF'
Stub _minify_for_fixtures helper for legacy-derived JSON fixtures (DM-54557)

Reads a real FITS/NDF archive and writes a small JSON fixture for
schema-versioning tests. Image/Mask/MaskedImage cropping path is
sketched; CellCoadd subsetting is flagged as TODO (the outer-ring
problem of inputs/PSFs that overlap kept cells is unsolved; the spec
notes morphing cells in place as a candidate fallback).

The minify helper is invoked manually by developers when they have a
real legacy file to derive from; not exercised by CI.

Generated with AI

Co-Authored-By: SLAC AI
EOF
)"
```

---

## Task Group G: Wrap-up

### Task G1: News fragment

**Files:**
- Create: `doc/changes/DM-54557.feature.md`

- [ ] **Step 1: Write the news fragment**

```markdown
Added schema versioning (`schema_version`, `min_read_version`, computed `schema_url`) to every top-level Pydantic serialization model, and integer-major container-format stamps (`FMTVER` for FITS, `FORMAT_VERSION` for NDF, plus `DATAMODL` / `DATA_MODEL` schema-URL keywords). Files written before this change continue to read; absent stamps are interpreted as the v1 defaults.
```

- [ ] **Step 2: Commit**

```bash
git add doc/changes/DM-54557.feature.md
git commit -m "$(cat <<'EOF'
Add news fragment for schema versioning (DM-54557)

Generated with AI

Co-Authored-By: SLAC AI
EOF
)"
```

### Task G2: Final full-suite check

- [ ] **Step 1: Run the entire test suite**

Run: `.pyenv/bin/python -m pytest tests/ --timeout=600`

Expected: PASS.

- [ ] **Step 2: Quick spot-check that the design doc and code agree on URL host**

```bash
grep -n "images.lsst.io" python/lsst/images/serialization/_common.py
grep -n "images.lsst.io" docs/superpowers/specs/2026-05-15-schema-versioning-design.md | head -5
```

Both should reference `https://images.lsst.io/schemas/`.

- [ ] **Step 3: Display the changelog so the implementer can inspect the work**

Run:

```bash
git log --oneline tickets/DM-54557 ^main
```

Expected output: roughly 9–11 commits, one per Task Group plus the spec rewrites you've already landed.
