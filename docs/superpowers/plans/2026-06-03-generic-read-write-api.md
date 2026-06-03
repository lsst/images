# Generic `read` / `write` API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `lsst.images.serialization.read(path)` and `write(obj, path)` that infer the in-memory class from the file's schema, plus a registry keyed by `(SCHEMA_NAME, SCHEMA_VERSION)` and an extra `python class:` line in `inspect`.

**Architecture:** A registry populated in `ArchiveTree.__pydantic_init_subclass__` maps `(name, version) → ArchiveTree subclass`. The in-memory class is derived from the `deserialize` return annotation via `typing.get_type_hints` and cached. `read()` uses `backend_for_path` + `get_basic_info` to dispatch; `write()` is a thin extension dispatcher.

**Tech Stack:** Python 3.13, pydantic, click, pytest, lsst.resources, lsst.utils.

**Working directory and python:** All commands run from the repo root `/sdf/home/t/timj/WORK/images`. Use `.pyenv/bin/python` and `.pyenv/bin/pytest` (the project's hidden venv); the system python lacks `zarr`. Set `PYTHONPATH=./python` so the in-tree package is found ahead of any installed copy. A reusable shell prefix:

```bash
PYTHONPATH=./python .pyenv/bin/pytest -v ...
```

**Coding conventions:**
- Top-level imports unless a circular import or expensive optional dep forces a local one. Lazy imports already used in `_inspect.py` and `_backends.py` are fine to follow.
- Keep code ruff/mypy clean (the repo runs both). Run `.pyenv/bin/ruff check python/ tests/` and `.pyenv/bin/mypy python/lsst/images/serialization` after each task.
- One sentence per line in markdown / rst.
- American English.

**File structure:**

| Path | Action | Responsibility |
| ---- | ------ | -------------- |
| `python/lsst/images/serialization/_io.py` | Create | `read`, `write`, `class_for_schema`, internal `_public_type` helper, registry storage |
| `python/lsst/images/serialization/_common.py` | Modify | Register each `ArchiveTree` subclass in `__pydantic_init_subclass__` |
| `python/lsst/images/serialization/__init__.py` | Modify | Re-export `_io` symbols |
| `python/lsst/images/cli/_inspect.py` | Modify | Add the `python class:` line |
| `tests/test_serialization_registry.py` | Create | Registry behaviour and class invariants |
| `tests/test_serialization_io.py` | Create | Generic `read` / `write`, fixture sweep |
| `tests/test_cli.py` | Modify | New `python class:` line assertions |
| `doc/changes/DM-55131.feature.md` | Modify | Append a sentence about the new API |

**Process notes:**
- TDD: write the failing test first, run it to confirm the failure, then write the minimal code, then run the test green, then commit.
- Commit after each task. Commit messages follow the existing style: `<imperative summary> (DM-55131)`.
- Don't push. Don't post on GitHub.
- `git status` shows three pre-existing untracked files (`dp1.zarr/`, `test_image_test_read_write-pd6zv4hr.fits`, `test_mask_test_read_write-55uh7sgz.fits`).  Do not stage or modify them.

---

## Task 1: Add registry storage and `class_for_schema` helper (no registration yet)

Set up the registry and lookup function in `_io.py` so later tasks have something to wire into. No `ArchiveTree` changes here yet, so the registry is empty.

**Files:**
- Create: `python/lsst/images/serialization/_io.py`
- Modify: `python/lsst/images/serialization/__init__.py`
- Test: `tests/test_serialization_registry.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_serialization_registry.py` with:

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

from lsst.images.serialization import class_for_schema


class ClassForSchemaTestCase(unittest.TestCase):
    """class_for_schema returns None for unknown (name, version)."""

    def test_unknown_returns_none(self) -> None:
        self.assertIsNone(class_for_schema("does-not-exist", "1.0.0"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=./python .pyenv/bin/pytest tests/test_serialization_registry.py -v`
Expected: FAIL with `ImportError: cannot import name 'class_for_schema'`.

- [ ] **Step 3: Create `_io.py` with the registry skeleton**

Create `python/lsst/images/serialization/_io.py`:

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
"""Generic ``read`` / ``write`` dispatchers and the schema-name registry."""

from __future__ import annotations

__all__ = ("class_for_schema", "register_schema_class")

from ._common import ArchiveTree

_REGISTRY: dict[tuple[str, str], type[ArchiveTree]] = {}
"""Map of ``(SCHEMA_NAME, SCHEMA_VERSION)`` to the registered
``ArchiveTree`` subclass."""


def class_for_schema(schema_name: str, schema_version: str) -> type[ArchiveTree] | None:
    """Return the registered ``ArchiveTree`` subclass for a schema, or ``None``.

    Parameters
    ----------
    schema_name
        Schema name (e.g. ``"visit_image"``).
    schema_version
        Schema version (e.g. ``"1.0.0"``).
    """
    return _REGISTRY.get((schema_name, schema_version))


def register_schema_class(cls: type[ArchiveTree]) -> None:
    """Register ``cls`` under ``(cls.SCHEMA_NAME, cls.SCHEMA_VERSION)``.

    No-op when the same class is registered for the same key (re-import
    during tests).  Raises `RuntimeError` when a *different* class is
    registered under an existing key.

    Intended to be called from ``ArchiveTree.__pydantic_init_subclass__``;
    not part of the public API.
    """
    key = (cls.SCHEMA_NAME, cls.SCHEMA_VERSION)
    existing = _REGISTRY.get(key)
    if existing is cls:
        return
    if existing is not None:
        raise RuntimeError(
            f"Schema {cls.SCHEMA_NAME!r} version {cls.SCHEMA_VERSION!r} "
            f"is already registered to {existing.__qualname__}; refusing to "
            f"replace it with {cls.__qualname__}."
        )
    _REGISTRY[key] = cls
```

- [ ] **Step 4: Re-export from the package `__init__.py`**

Edit `python/lsst/images/serialization/__init__.py` to add the new wildcard import after the existing ones (alphabetic placement keeps the file tidy):

```python
from ._asdf_utils import *
from ._backends import *
from ._common import *
from ._dtypes import *
from ._input_archive import *
from ._io import *
from ._output_archive import *
from ._tables import *
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `PYTHONPATH=./python .pyenv/bin/pytest tests/test_serialization_registry.py -v`
Expected: PASS.

- [ ] **Step 6: Lint and type-check**

Run: `.pyenv/bin/ruff check python/lsst/images/serialization tests/test_serialization_registry.py`
Run: `.pyenv/bin/mypy python/lsst/images/serialization`
Expected: clean for both.

- [ ] **Step 7: Commit**

```bash
git add python/lsst/images/serialization/_io.py \
        python/lsst/images/serialization/__init__.py \
        tests/test_serialization_registry.py
git commit -m "Add empty schema registry and class_for_schema lookup (DM-55131)"
```

---

## Task 2: Register every `ArchiveTree` subclass at class-creation time

Hook `register_schema_class` into the existing `__pydantic_init_subclass__`.  This wires the import-time registration without touching any subclass.

**Files:**
- Modify: `python/lsst/images/serialization/_common.py:151-170`
- Test: `tests/test_serialization_registry.py`

- [ ] **Step 1: Write the failing tests**

Append the following to `tests/test_serialization_registry.py` (and add the new import for `lsst.images`):

```python
import lsst.images  # noqa: F401  -- import for side-effect class registration
import lsst.images.cells  # noqa: F401
from lsst.images._visit_image import VisitImageSerializationModel
from lsst.images._image import ImageSerializationModel


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
        from lsst.images.serialization._io import register_schema_class

        # Re-registering the same class is a no-op.
        register_schema_class(VisitImageSerializationModel)
        # Registering a different class with the same key raises.
        class _Imposter(VisitImageSerializationModel):  # type: ignore[misc]
            pass

        # _Imposter inherits SCHEMA_NAME / SCHEMA_VERSION but is a distinct
        # class; this triggers the duplicate-key path explicitly.
        with self.assertRaises(RuntimeError):
            register_schema_class(_Imposter)
```

Note: defining `_Imposter` as a subclass triggers `__pydantic_init_subclass__`, which itself calls `register_schema_class`.  That call will raise the `RuntimeError`.  Because `class _Imposter(...)` runs the body inside the `with self.assertRaises` block, the assertion catches it correctly.  Use a fresh subclass per call (`assertRaises` exits after the first exception).

- [ ] **Step 2: Run the new tests and confirm they fail**

Run: `PYTHONPATH=./python .pyenv/bin/pytest tests/test_serialization_registry.py -v`
Expected: `test_visit_image_registered` and `test_nested_image_registered` FAIL (registry is still empty); `test_duplicate_registration_raises` may also fail.

- [ ] **Step 3: Wire registration into `ArchiveTree.__pydantic_init_subclass__`**

Edit `python/lsst/images/serialization/_common.py`.  At the top of the file (top-level imports), do not import `_io` directly to avoid a circular import — `_io` already imports `ArchiveTree` from this module.  Instead, import lazily inside the hook:

Locate the `__pydantic_init_subclass__` method (around lines 151-170).  After the existing block that injects `$id` / `title`, add the registration call.  The full updated method:

```python
@classmethod
def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
    """Inject ``$id`` and ``title`` into the subclass's JSON Schema, and
    register the subclass in the schema-name registry.

    Populates ``model_config['json_schema_extra']`` with values derived
    from the subclass's ``SCHEMA_NAME`` / ``SCHEMA_VERSION`` ClassVars,
    then registers the subclass so it can be looked up by schema name
    and version.  Subclasses that haven't declared the ClassVars are
    skipped.
    """
    super().__pydantic_init_subclass__(**kwargs)
    name = cls.__dict__.get("SCHEMA_NAME")
    version = cls.__dict__.get("SCHEMA_VERSION")
    if name is None or version is None:
        return
    json_schema_extra = cls.model_config.get("json_schema_extra") or {}
    if isinstance(json_schema_extra, dict):
        existing = dict(json_schema_extra)
        existing.setdefault("$id", f"https://images.lsst.io/schemas/{name}-{version}")
        existing.setdefault("title", name)
        cls.model_config = {**cls.model_config, "json_schema_extra": existing}
    # Local import to avoid the _io -> _common circular dependency at
    # module load time.
    from ._io import register_schema_class

    register_schema_class(cls)
```

Two behaviour changes worth flagging in the diff: the JSON-Schema injection is now guarded by a single `if isinstance(...)` (so registration still runs even when `json_schema_extra` is non-dict, which preserves the existing early-return semantics for the schema injection while not skipping registration); and the registration call is the new line at the end.

- [ ] **Step 4: Run the registry tests and confirm they pass**

Run: `PYTHONPATH=./python .pyenv/bin/pytest tests/test_serialization_registry.py -v`
Expected: PASS for all three test methods.

- [ ] **Step 5: Run the full test suite to catch any unrelated breakage**

Run: `PYTHONPATH=./python .pyenv/bin/pytest tests/ -x`
Expected: PASS.  The new registration is import-time but should not affect any existing test.

- [ ] **Step 6: Lint and type-check**

Run: `.pyenv/bin/ruff check python/lsst/images/serialization tests/test_serialization_registry.py`
Run: `.pyenv/bin/mypy python/lsst/images/serialization`
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add python/lsst/images/serialization/_common.py \
        tests/test_serialization_registry.py
git commit -m "Register ArchiveTree subclasses under (schema_name, version) (DM-55131)"
```

---

## Task 3: Add `_public_type` helper that derives the in-memory class from `deserialize`

Implement and test the helper that reads the `deserialize` return annotation and unwraps generics, then caches the result on the class.

**Files:**
- Modify: `python/lsst/images/serialization/_io.py`
- Test: `tests/test_serialization_registry.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_serialization_registry.py`:

```python
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
        from lsst.images.serialization._io import _public_type
        from lsst.images.serialization import ArchiveTree

        # Construct a stand-alone ArchiveTree subclass with an Any return.
        class _AnyTree(ArchiveTree):
            SCHEMA_NAME: str = "_any_tree_test"
            SCHEMA_VERSION: str = "1.0.0"
            MIN_READ_VERSION: int = 1

            def deserialize(self, archive, **kwargs):  # type: ignore[no-untyped-def]
                return None

        try:
            self.assertIsNone(_public_type(_AnyTree))
        finally:
            # Tidy up: pop the stand-alone class out of the registry so
            # it doesn't leak into other tests.
            from lsst.images.serialization._io import _REGISTRY

            _REGISTRY.pop(("_any_tree_test", "1.0.0"), None)
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `PYTHONPATH=./python .pyenv/bin/pytest tests/test_serialization_registry.py::PublicTypeTestCase -v`
Expected: FAIL with `ImportError: cannot import name '_public_type'`.

- [ ] **Step 3: Add `_public_type` to `_io.py`**

Edit `python/lsst/images/serialization/_io.py`.  Add to the imports and module body:

```python
from __future__ import annotations

__all__ = ("class_for_schema", "register_schema_class")

import typing
from typing import Any

from ._common import ArchiveTree
```

Add the helper at the end of the file:

```python
_PUBLIC_TYPE_ATTR = "_lsst_images_public_type"
"""Attribute name used to cache the resolved public type on each
``ArchiveTree`` subclass."""

_UNRESOLVED = object()
"""Sentinel cached when the return annotation is ``Any`` or could not be
resolved.  Distinguishes "we tried and failed" from "we have not tried"."""


def _public_type(tree_cls: type[ArchiveTree]) -> type | None:
    """Return the in-memory class produced by ``tree_cls.deserialize``.

    Derived from the return annotation of ``deserialize`` and cached on
    the class.  Returns `None` when the annotation is `Any` or cannot be
    resolved (e.g. it references a name that is not importable from the
    class's module globals).
    """
    cached = tree_cls.__dict__.get(_PUBLIC_TYPE_ATTR, None)
    if cached is _UNRESOLVED:
        return None
    if cached is not None:
        return cached  # type: ignore[no-any-return]
    try:
        hints = typing.get_type_hints(tree_cls.deserialize)
    except Exception:
        setattr(tree_cls, _PUBLIC_TYPE_ATTR, _UNRESOLVED)
        return None
    annotation = hints.get("return", Any)
    if annotation is Any:
        setattr(tree_cls, _PUBLIC_TYPE_ATTR, _UNRESOLVED)
        return None
    resolved = typing.get_origin(annotation) or annotation
    if not isinstance(resolved, type):
        setattr(tree_cls, _PUBLIC_TYPE_ATTR, _UNRESOLVED)
        return None
    setattr(tree_cls, _PUBLIC_TYPE_ATTR, resolved)
    return resolved
```

`_public_type` is intentionally not exported (no `__all__` entry).

- [ ] **Step 4: Run the tests and verify they pass**

Run: `PYTHONPATH=./python .pyenv/bin/pytest tests/test_serialization_registry.py::PublicTypeTestCase -v`
Expected: PASS.

- [ ] **Step 5: Lint and type-check**

Run: `.pyenv/bin/ruff check python/lsst/images/serialization tests/test_serialization_registry.py`
Run: `.pyenv/bin/mypy python/lsst/images/serialization`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add python/lsst/images/serialization/_io.py \
        tests/test_serialization_registry.py
git commit -m "Add _public_type helper deriving in-memory class from deserialize (DM-55131)"
```

---

## Task 4: Add a class-invariants test asserting registry coverage and concrete return annotations

A second test case in `test_serialization_registry.py` that walks every `ArchiveTree` subclass discovered after `lsst.images` (and `lsst.images.cells`) are imported, asserts every concrete subclass appears in the registry, and asserts every registered class has a resolvable concrete return type.  This is what catches regressions when somebody adds a new schema and forgets the annotation.

**Files:**
- Test: `tests/test_serialization_registry.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_serialization_registry.py`:

```python
def _all_concrete_archive_tree_subclasses() -> list[type]:
    """Walk ArchiveTree's subclass tree and return all concrete subclasses
    that declare SCHEMA_NAME (i.e. are real schema-bearing leaves)."""
    from lsst.images.serialization import ArchiveTree

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
    resolvable, concrete deserialize return annotation."""

    def test_every_subclass_registered(self) -> None:
        from lsst.images.serialization._io import _REGISTRY

        missing: list[str] = []
        for cls in _all_concrete_archive_tree_subclasses():
            key = (cls.SCHEMA_NAME, cls.SCHEMA_VERSION)
            registered = _REGISTRY.get(key)
            if registered is None or registered is not cls:
                missing.append(f"{cls.__qualname__} -> {key}")
        self.assertEqual(missing, [], f"Unregistered subclasses: {missing}")

    def test_every_registered_class_resolves_public_type(self) -> None:
        from lsst.images.serialization._io import _public_type

        unresolved: list[str] = []
        for cls in _all_concrete_archive_tree_subclasses():
            if _public_type(cls) is None:
                unresolved.append(cls.__qualname__)
        self.assertEqual(unresolved, [], f"No concrete return annotation: {unresolved}")
```

The two helpers (`_all_concrete_archive_tree_subclasses`, `ClassInvariantsTestCase`) belong at module scope under the existing test classes.

- [ ] **Step 2: Run the test**

Run: `PYTHONPATH=./python .pyenv/bin/pytest tests/test_serialization_registry.py::ClassInvariantsTestCase -v`
Expected: PASS, given the audit in the design (every concrete `ArchiveTree` subclass already has a concrete return annotation).  If the test fails, the failure list itself is the actionable diagnostic — fix the offending classes by adding a return annotation or registering them, then re-run.

- [ ] **Step 3: Commit**

```bash
git add tests/test_serialization_registry.py
git commit -m "Add class-invariants test for schema registry coverage (DM-55131)"
```

---

## Task 5: Implement the generic `read()` dispatcher

Add the `read(path, **kwargs)` function to `_io.py`, with tests against the existing `tests/data/schema_v1` JSON fixtures.

**Files:**
- Modify: `python/lsst/images/serialization/_io.py`
- Test: `tests/test_serialization_io.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_serialization_io.py`:

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

import lsst.images  # noqa: F401  -- registers schema classes
from lsst.images.serialization import (
    ArchiveReadError,
    ReadResult,
    class_for_schema,
    read,
)
from lsst.images.serialization._io import _public_type

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "schema_v1")


class GenericReadTestCase(unittest.TestCase):
    """read(path) dispatches by extension and produces the registered type."""

    def test_visit_image_json(self) -> None:
        from lsst.images import VisitImage

        path = os.path.join(DATA_DIR, "visit_image.json")
        result = read(path)
        self.assertIsInstance(result, ReadResult)
        self.assertIsInstance(result.obj, VisitImage)

    def test_image_json(self) -> None:
        from lsst.images import Image

        path = os.path.join(DATA_DIR, "image.json")
        result = read(path)
        self.assertIsInstance(result.obj, Image)


class GenericReadErrorsTestCase(unittest.TestCase):
    """Unknown schemas and bad extensions raise clean errors."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def test_unsupported_extension(self) -> None:
        path = os.path.join(self.tmp, "bogus.txt")
        with open(path, "w") as f:
            f.write("nope")
        # backend_for_path raises ValueError; read() must let it through.
        with self.assertRaises(ValueError):
            read(path)

    def test_unregistered_schema(self) -> None:
        # Write a JSON file with a fabricated schema_url so its
        # (name, version) is not in the registry.
        path = os.path.join(self.tmp, "fake.json")
        with open(path, "w") as f:
            f.write(
                '{"schema_url": "https://images.lsst.io/schemas/no-such-schema-99.0.0",'
                ' "schema_version": "99.0.0", "min_read_version": 1, "indirect": []}'
            )
        with self.assertRaises(ArchiveReadError) as ctx:
            read(path)
        self.assertIn("no-such-schema", str(ctx.exception))
        self.assertIn("99.0.0", str(ctx.exception))


class FixtureSweepTestCase(unittest.TestCase):
    """Every schema_v1 JSON fixture reads through the generic API and
    produces the in-memory type registered for its schema."""

    def test_sweep(self) -> None:
        roots = [DATA_DIR, os.path.join(DATA_DIR, "legacy")]
        for root in roots:
            if not os.path.isdir(root):
                continue
            for entry in sorted(os.listdir(root)):
                if not entry.endswith(".json"):
                    continue
                path = os.path.join(root, entry)
                with self.subTest(path=path):
                    result = read(path)
                    info = lsst.images.serialization.backend_for_path(
                        path
                    ).input_archive.get_basic_info(path)
                    cls = class_for_schema(info.schema_name, info.schema_version)
                    self.assertIsNotNone(cls)
                    expected_type = _public_type(cls)  # type: ignore[arg-type]
                    self.assertIsNotNone(expected_type)
                    self.assertIsInstance(result.obj, expected_type)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `PYTHONPATH=./python .pyenv/bin/pytest tests/test_serialization_io.py -v`
Expected: FAIL with `ImportError: cannot import name 'read' from 'lsst.images.serialization'`.

- [ ] **Step 3: Implement `read()` in `_io.py`**

Edit `python/lsst/images/serialization/_io.py`.  Update `__all__` and add the `read` function.  Place imports at the top.

```python
__all__ = ("class_for_schema", "read", "register_schema_class")
```

After the imports, add:

```python
from lsst.resources import ResourcePathExpression

from ._backends import backend_for_path
from ._common import ArchiveReadError, ReadResult
```

Add `read` at module scope (after `_public_type`):

```python
def read(
    path: ResourcePathExpression,
    **kwargs: Any,
) -> ReadResult[Any]:
    """Read an archive whose in-memory type is inferred from its schema.

    Dispatches to the FITS / NDF / JSON backend based on ``path``'s
    extension, looks up the in-memory class registered for the file's
    ``(schema_name, schema_version)``, and forwards the call to the
    per-backend ``read`` along with ``**kwargs``.

    Parameters
    ----------
    path
        File to read; convertible to `lsst.resources.ResourcePath`.
    **kwargs
        Backend- and type-specific keyword arguments.  Forwarded
        verbatim; mis-targeted arguments surface as ``TypeError`` from
        the underlying ``deserialize``.

    Returns
    -------
    ReadResult
        Named tuple of the deserialized object, its metadata, and any
        butler info, matching the per-backend ``read`` signature.

    Raises
    ------
    ValueError
        Raised by `backend_for_path` if the file extension is not
        recognised.
    ArchiveReadError
        Raised when the file's schema is not registered, or when the
        registered class does not declare a concrete deserialised type.
    """
    backend = backend_for_path(path)
    info = backend.input_archive.get_basic_info(path)
    tree_cls = class_for_schema(info.schema_name, info.schema_version)
    if tree_cls is None:
        raise ArchiveReadError(
            f"No registered schema {info.schema_name!r} version "
            f"{info.schema_version!r}; cannot determine in-memory type "
            f"for {path!r}."
        )
    public_cls = _public_type(tree_cls)
    if public_cls is None:
        raise ArchiveReadError(
            f"Schema {info.schema_name!r} version {info.schema_version!r} "
            f"does not declare a concrete deserialized type."
        )
    return backend.read(public_cls, path, **kwargs)
```

- [ ] **Step 4: Run the test suite and confirm it passes**

Run: `PYTHONPATH=./python .pyenv/bin/pytest tests/test_serialization_io.py -v`
Expected: PASS for all classes, including the parametrised `FixtureSweepTestCase.test_sweep` covering every fixture in `tests/data/schema_v1` (including `legacy/`).

- [ ] **Step 5: Lint and type-check**

Run: `.pyenv/bin/ruff check python/lsst/images/serialization tests/test_serialization_io.py`
Run: `.pyenv/bin/mypy python/lsst/images/serialization`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add python/lsst/images/serialization/_io.py tests/test_serialization_io.py
git commit -m "Add generic read() dispatcher inferring type from schema (DM-55131)"
```

---

## Task 6: Implement `write()` and round-trip tests

`write()` is a thin dispatcher.  Add it and a round-trip test for each backend extension.

**Files:**
- Modify: `python/lsst/images/serialization/_io.py`
- Test: `tests/test_serialization_io.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_serialization_io.py`:

```python
import numpy as np


class GenericWriteRoundTripTestCase(unittest.TestCase):
    """write(obj, path) dispatches by extension and round-trips."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def _make_image(self) -> "lsst.images.Image":
        from lsst.images import Box, Image

        return Image(np.arange(16, dtype=np.float32).reshape(4, 4), bbox=Box.factory[0:4, 0:4])

    def test_round_trip_fits(self) -> None:
        from lsst.images import Image
        from lsst.images.serialization import read, write

        img = self._make_image()
        path = os.path.join(self.tmp, "x.fits")
        write(img, path)
        result = read(path)
        self.assertIsInstance(result.obj, Image)
        np.testing.assert_array_equal(result.obj.array, img.array)

    def test_round_trip_json(self) -> None:
        from lsst.images import Image
        from lsst.images.serialization import read, write

        img = self._make_image()
        path = os.path.join(self.tmp, "x.json")
        write(img, path)
        result = read(path)
        self.assertIsInstance(result.obj, Image)
        np.testing.assert_array_equal(result.obj.array, img.array)

    def test_round_trip_ndf(self) -> None:
        try:
            import h5py  # noqa: F401
        except ImportError:
            self.skipTest("h5py not available.")
        from lsst.images import Image
        from lsst.images.serialization import read, write

        img = self._make_image()
        path = os.path.join(self.tmp, "x.sdf")
        write(img, path)
        result = read(path)
        self.assertIsInstance(result.obj, Image)
        np.testing.assert_array_equal(result.obj.array, img.array)
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `PYTHONPATH=./python .pyenv/bin/pytest tests/test_serialization_io.py::GenericWriteRoundTripTestCase -v`
Expected: FAIL with `ImportError: cannot import name 'write' from 'lsst.images.serialization'`.

- [ ] **Step 3: Add `write()` to `_io.py`**

Edit `python/lsst/images/serialization/_io.py`.  Update `__all__`:

```python
__all__ = ("class_for_schema", "read", "register_schema_class", "write")
```

Add `write` at module scope (next to `read`):

```python
def write(obj: Any, path: str, **kwargs: Any) -> Any:
    """Write ``obj`` to ``path``, dispatching by file extension.

    Forwards ``**kwargs`` to the per-backend ``write`` (e.g.
    ``compression_options`` for FITS).  No registry lookup is performed:
    the per-backend ``write`` already accepts any object with a
    ``serialize`` method.

    Parameters
    ----------
    obj
        Object to write; must implement ``serialize`` like the per-backend
        write functions expect.
    path
        Destination path.  The extension selects the backend.
    **kwargs
        Forwarded verbatim to the backend's ``write``.

    Returns
    -------
    Any
        Whatever the per-backend ``write`` returns (the serialised
        archive tree).
    """
    return backend_for_path(path).write(obj, path, **kwargs)
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `PYTHONPATH=./python .pyenv/bin/pytest tests/test_serialization_io.py -v`
Expected: PASS.  The NDF round-trip skips when `h5py` is unavailable.

- [ ] **Step 5: Lint and type-check**

Run: `.pyenv/bin/ruff check python/lsst/images/serialization tests/test_serialization_io.py`
Run: `.pyenv/bin/mypy python/lsst/images/serialization`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add python/lsst/images/serialization/_io.py tests/test_serialization_io.py
git commit -m "Add generic write() dispatcher and backend round-trip tests (DM-55131)"
```

---

## Task 7: Add `python class:` line to the `inspect` CLI

Extend `_inspect.py` to print the public type derived from the registered schema, and add CLI tests for both the registered and unregistered cases.

**Files:**
- Modify: `python/lsst/images/cli/_inspect.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py` (the existing `InspectTestCase` is the natural home for the new methods):

```python
    def test_inspect_fits_python_class(self) -> None:
        path = os.path.join(self.tmp, "y.fits")
        images_fits.write(self.image, path)
        result = CliRunner().invoke(main, ["inspect", path])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("python class:", result.output)
        self.assertIn("lsst.images.Image", result.output)

    def test_inspect_json_python_class(self) -> None:
        path = os.path.join(self.tmp, "y.json")
        images_json.write(self.image, path)
        result = CliRunner().invoke(main, ["inspect", path])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("python class:", result.output)
        self.assertIn("lsst.images.Image", result.output)

    def test_inspect_unregistered_schema(self) -> None:
        path = os.path.join(self.tmp, "fake.json")
        with open(path, "w") as f:
            f.write(
                '{"schema_url": "https://images.lsst.io/schemas/no-such-schema-99.0.0",'
                ' "schema_version": "99.0.0", "min_read_version": 1, "indirect": []}'
            )
        result = CliRunner().invoke(main, ["inspect", path])
        # inspect should still succeed: it reports basic info regardless
        # of registry membership.
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("python class:", result.output)
        self.assertIn("<unregistered: no-such-schema-99.0.0>", result.output)
```

The existing `InspectTestCase.setUp` already creates `self.image` and `self.tmp`; reusing them keeps the new tests concise.

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `PYTHONPATH=./python .pyenv/bin/pytest tests/test_cli.py::InspectTestCase -v`
Expected: the three new tests FAIL because `inspect` does not yet print `python class:`.

- [ ] **Step 3: Update `_inspect.py`**

Edit `python/lsst/images/cli/_inspect.py`.  Replace the body of `inspect` so the imports stay at the top of the file:

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

__all__ = ("inspect",)

import click

from ..serialization import ArchiveReadError, backend_for_path, class_for_schema
from ..serialization._io import _public_type


@click.command(name="inspect")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
def inspect(file: str) -> None:
    """Print basic information about an lsst.images file.

    Reports the schema URL, container format version, and the public
    Python class registered for the file's schema (when known) without
    deserializing pixel data.
    """
    try:
        backend = backend_for_path(file)
    except ValueError as err:
        raise click.ClickException(str(err)) from None
    try:
        info = backend.input_archive.get_basic_info(file)
    except (ArchiveReadError, ValueError) as err:
        raise click.ClickException(f"Could not read {file}: {err}") from None
    fmt = "n/a" if info.format_version is None else str(info.format_version)
    tree_cls = class_for_schema(info.schema_name, info.schema_version)
    public_cls = _public_type(tree_cls) if tree_cls is not None else None
    if public_cls is not None:
        from lsst.utils.introspection import get_full_type_name

        python_class = get_full_type_name(public_cls)
    else:
        python_class = f"<unregistered: {info.schema_name}-{info.schema_version}>"
    click.echo(f"path:           {file}")
    click.echo(f"format:         {backend.name}")
    click.echo(f"schema name:    {info.schema_name}")
    click.echo(f"schema version: {info.schema_version}")
    click.echo(f"schema URL:     {info.schema_url}")
    click.echo(f"format version: {fmt}")
    click.echo(f"python class:   {python_class}")
```

`get_full_type_name` is imported lazily so `inspect` keeps working in environments where `lsst.utils` would be expensive (or, in the future, optional).  `_public_type` is imported from the private module; the existing CLI already does similar things, and that import is intentional rather than re-exported because the helper is internal.

- [ ] **Step 4: Run the tests and verify they pass**

Run: `PYTHONPATH=./python .pyenv/bin/pytest tests/test_cli.py::InspectTestCase -v`
Expected: PASS for all methods (the original three plus the three new ones).

- [ ] **Step 5: Lint and type-check**

Run: `.pyenv/bin/ruff check python/lsst/images/cli tests/test_cli.py`
Run: `.pyenv/bin/mypy python/lsst/images/cli`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add python/lsst/images/cli/_inspect.py tests/test_cli.py
git commit -m "Show python class derived from schema in inspect output (DM-55131)"
```

---

## Task 8: Add a backend-kwargs forwarding test (FITS bbox)

Confirm that `**kwargs` flow through `read()` to the backend `deserialize`.  This is mostly insurance but covers the explicit promise in the spec.

**Files:**
- Test: `tests/test_serialization_io.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_serialization_io.py`:

```python
class GenericReadKwargsTestCase(unittest.TestCase):
    """**kwargs forwarded by read() reach the backend deserialize."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def test_bbox_subset_fits(self) -> None:
        from lsst.images import Box, Image
        from lsst.images.serialization import read, write

        img = Image(np.arange(64, dtype=np.float32).reshape(8, 8), bbox=Box.factory[0:8, 0:8])
        path = os.path.join(self.tmp, "x.fits")
        write(img, path)
        # Read a 4x4 subset.  bbox is the FITS-specific kwarg understood
        # by Image.deserialize; the generic read must forward it.
        sub = read(path, bbox=Box.factory[2:6, 2:6])
        self.assertEqual(sub.obj.array.shape, (4, 4))
        np.testing.assert_array_equal(sub.obj.array, img.array[2:6, 2:6])
```

- [ ] **Step 2: Run the test and verify it passes**

Because `read()` already forwards `**kwargs` (Task 5), this test should pass without further code changes.  Run:
`PYTHONPATH=./python .pyenv/bin/pytest tests/test_serialization_io.py::GenericReadKwargsTestCase -v`
Expected: PASS.  If it fails, the failure points at the kwargs path -- look at `read()` and make sure the `**kwargs` argument is being forwarded to `backend.read` (it should be, but treat any failure as a regression to fix here).

- [ ] **Step 3: Commit**

```bash
git add tests/test_serialization_io.py
git commit -m "Test that read() forwards backend kwargs (FITS bbox) (DM-55131)"
```

---

## Task 9: Update news fragment

The existing news fragment for DM-55131 already mentions `backend_for_path` and `get_basic_info`.  Append a sentence about the new `read` / `write` and `class_for_schema`.

**Files:**
- Modify: `doc/changes/DM-55131.feature.md`

- [ ] **Step 1: Edit the news fragment**

Replace the contents of `doc/changes/DM-55131.feature.md` with:

```
Added the ``lsst-images-admin`` command-line tool with ``convert`` (legacy FITS to new format), ``inspect`` (schema URL and format version), ``minify``, and ``extract-test-data`` subcommands.
Added ``lsst.images.serialization.backend_for_path`` and ``InputArchive.get_basic_info`` as public APIs for resolving a backend by file suffix and reading basic archive information.
Added ``lsst.images.serialization.read`` and ``lsst.images.serialization.write``, which dispatch by file suffix and infer the in-memory type from the file's schema, plus ``class_for_schema`` for looking up the registered ``ArchiveTree`` subclass for a ``(schema_name, schema_version)``.
The ``inspect`` subcommand now also reports the registered Python class for the file's schema.
```

- [ ] **Step 2: Commit**

```bash
git add doc/changes/DM-55131.feature.md
git commit -m "Document generic read/write API in news fragment (DM-55131)"
```

---

## Task 10: Final verification

Run the full test suite, ruff, and mypy from scratch to confirm no regressions and a clean code state.

**Files:** none changed.

- [ ] **Step 1: Run the full test suite**

Run: `PYTHONPATH=./python .pyenv/bin/pytest tests/ -v`
Expected: all tests pass.  Note any test that was previously skipped (e.g. those that depend on `TESTDATA_IMAGES_DIR`); these should remain skipped, not fail.

- [ ] **Step 2: Ruff**

Run: `.pyenv/bin/ruff check python/ tests/`
Run: `.pyenv/bin/ruff format --check python/ tests/`
Expected: clean.

- [ ] **Step 3: Mypy**

Run: `.pyenv/bin/mypy python/lsst/images`
Expected: clean.

- [ ] **Step 4: Verify the working tree**

Run: `git status`
Expected: working tree clean except for the three pre-existing untracked artefacts (`dp1.zarr/`, `test_image_test_read_write-pd6zv4hr.fits`, `test_mask_test_read_write-55uh7sgz.fits`) that were untracked before any work began.

- [ ] **Step 5: Verify the commit history**

Run: `git log --oneline tickets/DM-55131..HEAD` (or `git log --oneline -10`).
Expected: nine new commits, one per implementation task, each with a `(DM-55131)` suffix.  No `git push`.
