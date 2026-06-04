# Generic Reader Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `lsst.images.serialization.open()`, a path-based context-manager reader for pulling individual components (or the whole object) out of a file efficiently, layered on a new `InputArchive.open_tree` primitive that `read_tree` and the formatter also use.

**Architecture:** A single internal primitive `InputArchive.open_tree(path, tree_cls, ...)` (classmethod context manager yielding `(archive, tree)`) is implemented per backend. The user-facing `open()` resolves the tree class from the file's schema, opens via `open_tree`, and returns a `Reader[T]` exposing `get_component`, `read`, and `info`. `read_tree` and `GenericFormatter.read_from_uri` are rewired onto the same primitive. `open()`/`read()` gain a `cls` argument driving an `isinstance`/`issubclass` check and static return typing.

**Tech Stack:** Python 3.12+ (PEP 695 generics), pydantic, `lsst.resources`, `lsst.daf.butler` (FormatterV2), unittest, ruff, mypy.

**Environment:** Run Python/pytest as `PYTHONPATH=$PWD/python ~/pyenv/bin/python ...` (the installed stack shadows the local checkout). Lint with `~/pyenv/bin/ruff`. The 3 pre-existing `frozendict` mypy errors from external `cell_coadds` are environmental — ignore them.

---

## File Structure

- `python/lsst/images/serialization/_input_archive.py` — **modify**: add base `InputArchive.open_tree` (raises `NotImplementedError`).
- `python/lsst/images/fits/_input_archive.py` — **modify**: implement `FitsInputArchive.open_tree`; rewrite `read_tree` on it.
- `python/lsst/images/ndf/_input_archive.py` — **modify**: implement `NdfInputArchive.open_tree`; rewrite `read_tree` on it.
- `python/lsst/images/json/_input_archive.py` — **modify**: implement `JsonInputArchive.open_tree`; rewrite `read_tree`'s path branch on it.
- `python/lsst/images/serialization/_reader.py` — **create**: `Reader[T]` class and `open()` function.
- `python/lsst/images/serialization/__init__.py` — **modify**: add `from ._reader import *`.
- `python/lsst/images/serialization/_io.py` — **modify**: add `cls` parameter + overloads to `read()`.
- `python/lsst/images/formatters.py` — **modify**: rewire `read_from_uri` onto `open()`; delete `_open_archive_and_tree`/`_extension_from_uri`; drop now-unused imports.
- `tests/test_serialization_reader.py` — **create**: reader behaviour tests.
- `tests/test_serialization_io.py` — **modify**: add `read(..., cls=...)` typing/validation tests.
- `doc/changes/DM-55131.feature.md` — **modify**: add a line for `open()`.

---

## Task 1: Base `open_tree` + FITS implementation + FITS `read_tree`

**Files:**
- Modify: `python/lsst/images/serialization/_input_archive.py`
- Modify: `python/lsst/images/fits/_input_archive.py`
- Test: `tests/test_serialization_reader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_serialization_reader.py`:

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

from lsst.images import fits as images_fits
from lsst.images.fits import FitsInputArchive
from lsst.images.serialization import ArchiveTree, backend_for_path, class_for_schema, read

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "schema_v1")


def _visit_image():
    """A VisitImage loaded from the committed JSON fixture."""
    return read(os.path.join(DATA_DIR, "visit_image.json")).deserialized


class FitsOpenTreeTestCase(unittest.TestCase):
    """InputArchive.open_tree yields a live (archive, tree) pair."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "v.fits")
        images_fits.write(_visit_image(), self.path)

    def test_open_tree_yields_archive_and_tree(self) -> None:
        info = backend_for_path(self.path).input_archive.get_basic_info(self.path)
        tree_cls = class_for_schema(info.schema_name)
        with FitsInputArchive.open_tree(self.path, tree_cls) as (archive, tree):
            self.assertIsInstance(tree, ArchiveTree)
            proj = tree.deserialize_component("projection", archive)
            self.assertIsNotNone(proj)

    def test_read_still_works(self) -> None:
        # read() routes through read_tree, which now sits on open_tree.
        result = read(self.path)
        self.assertEqual(type(result.deserialized).__name__, "VisitImage")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m pytest tests/test_serialization_reader.py -v`
Expected: FAIL — `AttributeError: type object 'FitsInputArchive' has no attribute 'open_tree'`.

- [ ] **Step 3: Add the base `open_tree` declaration**

In `python/lsst/images/serialization/_input_archive.py`, update the imports near the top:

```python
from collections.abc import Callable
from contextlib import AbstractContextManager
from types import EllipsisType
from typing import TYPE_CHECKING, Any, TypeVar
```

Then add this method to `class InputArchive`, immediately after `get_basic_info`:

```python
    @classmethod
    def open_tree(
        cls,
        path: ResourcePathExpression,
        tree_cls: type[ArchiveTree],
        *,
        partial: bool = True,
        **backend_kwargs: Any,
    ) -> AbstractContextManager[tuple[InputArchive[P], ArchiveTree]]:
        """Open ``path``, load and validate its top-level tree, and yield
        ``(archive, tree)`` as a context manager.

        ``tree_cls`` is the un-parameterised `ArchiveTree` subclass; each
        backend parameterises it with its own pointer model.  Backend-specific
        open options (e.g. ``page_size`` for FITS) are accepted via
        ``**backend_kwargs``; ``partial`` is honoured where meaningful.

        Each concrete backend implements this.
        """
        raise NotImplementedError(f"{cls.__name__} does not implement open_tree.")
```

- [ ] **Step 4: Implement `FitsInputArchive.open_tree` and rewrite `read_tree`**

In `python/lsst/images/fits/_input_archive.py`, add this method to `class FitsInputArchive` (after `get_basic_info`):

```python
    @classmethod
    @contextmanager
    def open_tree(
        cls,
        path: ResourcePathExpression,
        tree_cls: type[ArchiveTree],
        *,
        partial: bool = True,
        **backend_kwargs: Any,
    ) -> Iterator[tuple[Self, ArchiveTree]]:
        """Open the FITS file and yield ``(archive, tree)``.

        Honours the ``page_size`` and ``partial`` open options.
        """
        page_size = backend_kwargs.pop("page_size", 2880 * 50)
        parameterized = parameterize_tree(tree_cls, PointerModel)
        with cls.open(path, page_size=page_size, partial=partial) as archive:
            tree = archive.get_tree(parameterized)
            yield archive, tree
```

Replace the body of the module-level `read_tree` function with:

```python
    if partial is None:
        partial = any(v is not None for v in kwargs.values())
    with FitsInputArchive.open_tree(path, tree_cls, page_size=page_size, partial=partial) as (
        archive,
        tree,
    ):
        obj = tree.deserialize(archive, **kwargs)
        if hasattr(obj, "_opaque_metadata"):
            obj._opaque_metadata = archive.get_opaque_metadata()
        return ReadResult(obj, tree.metadata, tree.butler_info)
```

(Leave the `read_tree` signature `def read_tree(tree_cls, path, *, page_size=2880*50, partial=None, **kwargs)` unchanged.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m pytest tests/test_serialization_reader.py tests/ -q`
Expected: PASS (full suite green — `read_tree` behaviour unchanged).

- [ ] **Step 6: Lint and type-check**

Run: `~/pyenv/bin/ruff check python/lsst/images/serialization/_input_archive.py python/lsst/images/fits/_input_archive.py tests/test_serialization_reader.py`
Expected: `All checks passed!`
Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m mypy python/lsst/images/fits/_input_archive.py 2>&1 | grep -v frozendict`
Expected: no errors in the listed file.

- [ ] **Step 7: Commit**

```bash
git add python/lsst/images/serialization/_input_archive.py python/lsst/images/fits/_input_archive.py tests/test_serialization_reader.py
git commit -m "Add InputArchive.open_tree primitive and FITS implementation (DM-55131)"
```

---

## Task 2: NDF `open_tree` + `read_tree`

**Files:**
- Modify: `python/lsst/images/ndf/_input_archive.py`
- Test: `tests/test_serialization_reader.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_serialization_reader.py` (after the FITS case):

```python
try:
    import h5py  # noqa: F401

    from lsst.images import ndf as images_ndf

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False


@unittest.skipUnless(HAVE_H5PY, "h5py is not available.")
class NdfOpenTreeTestCase(unittest.TestCase):
    """open_tree works for the NDF backend."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "v.sdf")
        images_ndf.write(_visit_image(), self.path)

    def test_open_tree_yields_archive_and_tree(self) -> None:
        from lsst.images.ndf import NdfInputArchive

        info = backend_for_path(self.path).input_archive.get_basic_info(self.path)
        tree_cls = class_for_schema(info.schema_name)
        with NdfInputArchive.open_tree(self.path, tree_cls) as (archive, tree):
            self.assertIsInstance(tree, ArchiveTree)
            self.assertIsNotNone(tree.deserialize_component("obs_info", archive))

    def test_read_still_works(self) -> None:
        self.assertEqual(type(read(self.path).deserialized).__name__, "VisitImage")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m pytest tests/test_serialization_reader.py::NdfOpenTreeTestCase -v`
Expected: FAIL — `AttributeError: ... 'NdfInputArchive' has no attribute 'open_tree'`.

- [ ] **Step 3: Implement `NdfInputArchive.open_tree` and rewrite `read_tree`**

In `python/lsst/images/ndf/_input_archive.py`, add this method to `class NdfInputArchive` (after `get_basic_info`):

```python
    @classmethod
    @contextmanager
    def open_tree(
        cls,
        path: ResourcePathExpression,
        tree_cls: type[ArchiveTree],
        *,
        partial: bool = True,
        **backend_kwargs: Any,
    ) -> Iterator[tuple[Self, ArchiveTree]]:
        """Open the NDF file and yield ``(archive, tree)``.

        Requires the symmetric LSST JSON tree; ``partial`` is accepted but
        not meaningful (h5py reads lazily regardless).
        """
        parameterized = parameterize_tree(tree_cls, NdfPointerModel)
        with cls.open(path) as archive:
            if archive._get_main_json_path() is None:
                raise ArchiveReadError(
                    f"{path!r} has no LSST JSON tree; open_tree requires the symmetric read path."
                )
            tree = archive.get_tree(parameterized)
            yield archive, tree
```

Replace the body of the module-level `read_tree` with:

```python
    with NdfInputArchive.open_tree(path, tree_cls) as (archive, tree):
        obj = tree.deserialize(archive, **kwargs)
        if hasattr(obj, "_opaque_metadata"):
            obj._opaque_metadata = archive.get_opaque_metadata()
        return ReadResult(obj, tree.metadata, tree.butler_info)
```

(Keep the `read_tree` signature `def read_tree(tree_cls, path, **kwargs)` unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m pytest tests/test_serialization_reader.py tests/ -q`
Expected: PASS.

- [ ] **Step 5: Lint and type-check**

Run: `~/pyenv/bin/ruff check python/lsst/images/ndf/_input_archive.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add python/lsst/images/ndf/_input_archive.py tests/test_serialization_reader.py
git commit -m "Add NdfInputArchive.open_tree (DM-55131)"
```

---

## Task 3: JSON `open_tree` + `read_tree`

**Files:**
- Modify: `python/lsst/images/json/_input_archive.py`
- Test: `tests/test_serialization_reader.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_serialization_reader.py`:

```python
class JsonOpenTreeTestCase(unittest.TestCase):
    """open_tree works for the JSON backend."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        from lsst.images import json as images_json

        self.path = os.path.join(tmp.name, "v.json")
        images_json.write(_visit_image(), self.path)

    def test_open_tree_yields_archive_and_tree(self) -> None:
        from lsst.images.json import JsonInputArchive

        info = backend_for_path(self.path).input_archive.get_basic_info(self.path)
        tree_cls = class_for_schema(info.schema_name)
        with JsonInputArchive.open_tree(self.path, tree_cls) as (archive, tree):
            self.assertIsInstance(tree, ArchiveTree)
            self.assertIsNotNone(tree.deserialize_component("projection", archive))

    def test_read_still_works(self) -> None:
        self.assertEqual(type(read(self.path).deserialized).__name__, "VisitImage")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m pytest tests/test_serialization_reader.py::JsonOpenTreeTestCase -v`
Expected: FAIL — `AttributeError: ... 'JsonInputArchive' has no attribute 'open_tree'`.

- [ ] **Step 3: Implement `JsonInputArchive.open_tree` and rewrite `read_tree`'s path branch**

In `python/lsst/images/json/_input_archive.py`, update imports:

```python
import re  # only if already present; otherwise skip
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Self
```

(The required additions are `Iterator` on the `collections.abc` line, `from contextlib import contextmanager`, and `Self` on the `typing` line. Do not add `re`.)

Add this method to `class JsonInputArchive` (after `get_basic_info`):

```python
    @classmethod
    @contextmanager
    def open_tree(
        cls,
        path: ResourcePathExpression,
        tree_cls: type[ArchiveTree],
        *,
        partial: bool = True,
        **backend_kwargs: Any,
    ) -> Iterator[tuple[Self, ArchiveTree]]:
        """Parse the JSON tree and yield ``(archive, tree)``.

        A no-resource context manager: JSON is fully in memory, so ``partial``
        is a no-op.  ``tree.indirect`` is released when the context exits.
        """
        parameterized = parameterize_tree(tree_cls, JsonRef)
        tree = parameterized.model_validate_json(ResourcePath(path).read())
        archive = cls(tree.indirect)
        try:
            yield archive, tree
        finally:
            tree.indirect = []
```

Replace the body of the module-level `read_tree` with (preserving the already-validated-tree branch):

```python
    if isinstance(target, ArchiveTree):
        archive = JsonInputArchive(target.indirect)
        obj = target.deserialize(archive, **kwargs)
        target.indirect = []
        return ReadResult(obj, target.metadata, target.butler_info)
    with JsonInputArchive.open_tree(target, tree_cls) as (archive, tree):
        obj = tree.deserialize(archive, **kwargs)
        return ReadResult(obj, tree.metadata, tree.butler_info)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m pytest tests/test_serialization_reader.py tests/ -q`
Expected: PASS.

- [ ] **Step 5: Lint and type-check**

Run: `~/pyenv/bin/ruff check python/lsst/images/json/_input_archive.py`
Expected: `All checks passed!`

- [ ] **Step 6: Commit**

```bash
git add python/lsst/images/json/_input_archive.py tests/test_serialization_reader.py
git commit -m "Add JsonInputArchive.open_tree (DM-55131)"
```

---

## Task 4: `Reader` + `open()` user-facing API

**Files:**
- Create: `python/lsst/images/serialization/_reader.py`
- Modify: `python/lsst/images/serialization/__init__.py`
- Test: `tests/test_serialization_reader.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_serialization_reader.py`:

```python
class ReaderApiTestCase(unittest.TestCase):
    """The user-facing serialization.open() / Reader interface."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.vi = _visit_image()
        self.fits = os.path.join(self.tmp, "v.fits")
        images_fits.write(self.vi, self.fits)

    def test_components_and_read(self) -> None:
        import lsst.images.serialization as ser

        with ser.open(self.fits) as reader:
            proj = reader.get_component("projection")
            obs = reader.get_component("obs_info")
            self.assertIsNotNone(proj)
            self.assertIsNotNone(obs)
            full = reader.read()
            self.assertEqual(type(full).__name__, "VisitImage")

    def test_info(self) -> None:
        import lsst.images.serialization as ser

        with ser.open(self.fits) as reader:
            self.assertEqual(reader.info.schema_name, "visit_image")
            self.assertEqual(reader.info.schema_version, "1.0.0")

    def test_cls_match(self) -> None:
        import lsst.images.serialization as ser
        from lsst.images import VisitImage

        with ser.open(self.fits, cls=VisitImage) as reader:
            self.assertIsInstance(reader.read(), VisitImage)

    def test_cls_mismatch_raises(self) -> None:
        import lsst.images.serialization as ser
        from lsst.images import Mask

        with self.assertRaises(TypeError):
            with ser.open(self.fits, cls=Mask):
                pass

    def test_unknown_component(self) -> None:
        import lsst.images.serialization as ser
        from lsst.images.serialization import InvalidComponentError

        with ser.open(self.fits) as reader:
            with self.assertRaises(InvalidComponentError):
                reader.get_component("does_not_exist")

    def test_use_after_close_raises(self) -> None:
        import lsst.images.serialization as ser

        with ser.open(self.fits) as reader:
            pass
        with self.assertRaises(RuntimeError):
            reader.get_component("projection")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m pytest tests/test_serialization_reader.py::ReaderApiTestCase -v`
Expected: FAIL — `AttributeError: module 'lsst.images.serialization' has no attribute 'open'`.

- [ ] **Step 3: Create `_reader.py`**

Create `python/lsst/images/serialization/_reader.py`:

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
"""User-facing ``open`` reader for incremental, component-wise reads."""

from __future__ import annotations

__all__ = ("Reader", "open")

from contextlib import AbstractContextManager, contextmanager
from typing import Any, overload

from lsst.resources import ResourcePathExpression

from ._backends import backend_for_path
from ._common import ArchiveReadError, ArchiveTree, MetadataValue
from ._input_archive import ArchiveInfo, InputArchive
from ._io import class_for_schema, public_type_for_schema


class Reader[T]:
    """A handle to an open ``lsst.images`` file.

    Returned by `open`.  Lets the caller pull individual components, or the
    whole object, out of a file that is opened once; the underlying archive
    caches dereferenced pointers so repeated reads share work.  Valid only
    inside the ``with`` block that produced it.
    """

    def __init__(
        self,
        archive: InputArchive[Any],
        tree: ArchiveTree,
        info: ArchiveInfo,
        expected_cls: type[T] | None,
    ) -> None:
        self._archive = archive
        self._tree = tree
        self._info = info
        self._expected_cls = expected_cls
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Reader is closed; use it only inside its 'with' block.")

    @property
    def info(self) -> ArchiveInfo:
        """Identifying information (schema name/version/url, format version)."""
        return self._info

    @property
    def metadata(self) -> dict[str, MetadataValue]:
        """Flexible metadata stored with the object."""
        return self._tree.metadata

    @property
    def butler_info(self) -> Any:
        """Butler dataset info stored with the object, or `None`."""
        return self._tree.butler_info

    def get_component(self, name: str, **kwargs: Any) -> Any:
        """Deserialize and return a single named component.

        Raises `~lsst.images.serialization.InvalidComponentError` for an
        unknown component name.
        """
        self._check_open()
        return self._tree.deserialize_component(name, self._archive, **kwargs)

    def read(self, **kwargs: Any) -> T:
        """Deserialize and return the whole object."""
        self._check_open()
        obj = self._tree.deserialize(self._archive, **kwargs)
        if hasattr(obj, "_opaque_metadata"):
            obj._opaque_metadata = self._archive.get_opaque_metadata()
        if self._expected_cls is not None and not isinstance(obj, self._expected_cls):
            raise TypeError(
                f"{self._info.schema_name!r} deserialised to {type(obj).__name__}, "
                f"not the requested {self._expected_cls.__name__}."
            )
        return obj  # type: ignore[return-value]


@contextmanager
def _open_reader(
    path: ResourcePathExpression,
    cls: type | None,
    partial: bool,
    backend_kwargs: dict[str, Any],
):
    backend = backend_for_path(path)
    info = backend.input_archive.get_basic_info(path)
    tree_cls = class_for_schema(info.schema_name)
    if tree_cls is None:
        raise ArchiveReadError(
            f"No registered schema {info.schema_name!r}; cannot open {path!r}."
        )
    if cls is not None:
        resolved = public_type_for_schema(info.schema_name)
        if resolved is not None and not issubclass(resolved, cls):
            raise TypeError(
                f"{path!r} has schema {info.schema_name!r} (type {resolved.__name__}), "
                f"which is not a {cls.__name__}."
            )
    with backend.input_archive.open_tree(path, tree_cls, partial=partial, **backend_kwargs) as (
        archive,
        tree,
    ):
        reader: Reader[Any] = Reader(archive, tree, info, cls)
        try:
            yield reader
        finally:
            reader._closed = True


@overload
def open[T](
    path: ResourcePathExpression, cls: type[T], *, partial: bool = ..., **backend_kwargs: Any
) -> AbstractContextManager[Reader[T]]: ...
@overload
def open(
    path: ResourcePathExpression, cls: None = ..., *, partial: bool = ..., **backend_kwargs: Any
) -> AbstractContextManager[Reader[Any]]: ...
def open(path, cls=None, *, partial=True, **backend_kwargs):
    """Open an ``lsst.images`` file for incremental, component-wise reads.

    Dispatches to the FITS / NDF / JSON backend by file extension, resolves
    the registered in-memory type from the file's schema, and returns a
    `Reader` context manager.

    Parameters
    ----------
    path
        File to read; convertible to `lsst.resources.ResourcePath`.
    cls
        Optional expected in-memory type.  When given, `open` validates that
        the file's schema resolves to a subclass of ``cls`` (raising
        `TypeError` otherwise) and the returned `Reader` is typed
        accordingly, so `Reader.read` needs no cast.
    partial
        Forwarded to the backend ``open_tree``; defaults to `True` (a reader
        is for incremental access).  A no-op for the JSON and NDF backends.
    **backend_kwargs
        Backend-specific open options (e.g. ``page_size`` for FITS).

    Raises
    ------
    ValueError
        If the file extension is not recognised.
    ArchiveReadError
        If the file's schema is not registered.
    TypeError
        If ``cls`` is given and the file's schema resolves to an
        incompatible type.
    """
    return _open_reader(path, cls, partial, backend_kwargs)
```

- [ ] **Step 4: Export from the package**

In `python/lsst/images/serialization/__init__.py`, add after the existing `from ._io import *` line:

```python
from ._reader import *
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m pytest tests/test_serialization_reader.py -v`
Expected: PASS (all `ReaderApiTestCase` tests).

- [ ] **Step 6: Lint and type-check**

Run: `~/pyenv/bin/ruff check python/lsst/images/serialization/_reader.py python/lsst/images/serialization/__init__.py`
Expected: `All checks passed!`
Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m mypy python/lsst/images/serialization/_reader.py 2>&1 | grep -v frozendict`
Expected: no errors in `_reader.py`.

- [ ] **Step 7: Commit**

```bash
git add python/lsst/images/serialization/_reader.py python/lsst/images/serialization/__init__.py tests/test_serialization_reader.py
git commit -m "Add serialization.open() reader and Reader class (DM-55131)"
```

---

## Task 5: `cls` parameter + overloads on generic `read()`

**Files:**
- Modify: `python/lsst/images/serialization/_io.py`
- Test: `tests/test_serialization_io.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_serialization_io.py` before the `if __name__` block:

```python
class ReadClsTestCase(unittest.TestCase):
    """read(path, cls=...) validates the deserialized type."""

    def test_read_cls_match(self) -> None:
        path = os.path.join(DATA_DIR, "image.json")
        result = read(path, cls=Image)
        self.assertIsInstance(result.deserialized, Image)

    def test_read_cls_mismatch_raises(self) -> None:
        from lsst.images import Mask

        path = os.path.join(DATA_DIR, "image.json")
        with self.assertRaises(TypeError):
            read(path, cls=Mask)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m pytest tests/test_serialization_io.py::ReadClsTestCase -v`
Expected: FAIL — `TypeError: read() got an unexpected keyword argument 'cls'`.

- [ ] **Step 3: Add `cls` + overloads to `read()`**

In `python/lsst/images/serialization/_io.py`, change the import line `from typing import Any, cast` to:

```python
from typing import Any, cast, overload
```

Replace the `read` function definition (the `def read(path, **kwargs) -> ReadResult[Any]:` header through its `return` statement) with:

```python
@overload
def read[T](path: ResourcePathExpression, cls: type[T], **kwargs: Any) -> ReadResult[T]: ...
@overload
def read(path: ResourcePathExpression, cls: None = ..., **kwargs: Any) -> ReadResult[Any]: ...
def read(path, cls=None, **kwargs):
    """Read an archive whose in-memory type is inferred from its schema.

    Dispatches to the appropriate backend based on ``path``'s extension,
    looks up the registered ``ArchiveTree`` subclass for the file's
    ``schema_name``, and forwards the call to the per-backend ``read_tree``
    along with ``**kwargs``.  Schema-version compatibility is enforced when
    the model validates the on-disk tree, via ``min_read_version``.

    Parameters
    ----------
    path
        File to read; convertible to `lsst.resources.ResourcePath`.
    cls
        Optional expected in-memory type.  When given, the deserialized
        object is validated with ``isinstance`` (raising `TypeError`
        otherwise) and the static return type is ``ReadResult[T]``.
    **kwargs
        Backend- and type-specific keyword arguments.  Forwarded verbatim;
        mis-targeted arguments surface as ``TypeError`` from the underlying
        ``deserialize``.

    Returns
    -------
    ReadResult
        Named tuple of the deserialized object, its metadata, and any butler
        info, matching the per-backend ``read`` signature.

    Raises
    ------
    ValueError
        Raised by `backend_for_path` if the file extension is not recognised.
    ArchiveReadError
        Raised when the file's ``schema_name`` is not registered, or
        propagated from the model's ``min_read_version`` check on
        ``model_validate*``.
    TypeError
        Raised when ``cls`` is given and the deserialized object is not an
        instance of it.
    """
    backend = backend_for_path(path)
    info = backend.input_archive.get_basic_info(path)
    tree_cls = class_for_schema(info.schema_name)
    if tree_cls is None:
        raise ArchiveReadError(
            f"No registered schema {info.schema_name!r}; cannot determine in-memory type for {path!r}."
        )
    result = cast(ReadResult[Any], backend.read_tree(tree_cls, path, **kwargs))
    if cls is not None and not isinstance(result.deserialized, cls):
        raise TypeError(
            f"{path!r} (schema {info.schema_name!r}) deserialised to "
            f"{type(result.deserialized).__name__}, not the requested {cls.__name__}."
        )
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m pytest tests/test_serialization_io.py -v`
Expected: PASS (including the new `ReadClsTestCase` and the existing read tests).

- [ ] **Step 5: Lint and type-check**

Run: `~/pyenv/bin/ruff check python/lsst/images/serialization/_io.py tests/test_serialization_io.py`
Expected: `All checks passed!`
Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m mypy python/lsst/images/serialization/_io.py 2>&1 | grep -v frozendict`
Expected: no errors in `_io.py`.

- [ ] **Step 6: Commit**

```bash
git add python/lsst/images/serialization/_io.py tests/test_serialization_io.py
git commit -m "Add cls argument and typed overloads to read() (DM-55131)"
```

---

## Task 6: Rewire `GenericFormatter.read_from_uri` onto `open()`

**Files:**
- Modify: `python/lsst/images/formatters.py`

This is a behaviour-preserving refactor; the existing `TemporaryButler` round-trip tests are the verification.

- [ ] **Step 1: Run the relevant tests to confirm they currently pass**

Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m pytest tests/test_image.py tests/test_visit_image.py tests/test_mask.py tests/test_masked_image.py -q`
Expected: PASS (baseline before refactor).

- [ ] **Step 2: Replace the read path and drop dead code**

In `python/lsst/images/formatters.py`:

Change the import line
`from .serialization import ArchiveTree, ButlerInfo, InputArchive, JsonRef, write`
to:

```python
from . import serialization as ser
from .serialization import ButlerInfo, write
```

Remove these now-unused imports:
- `from collections.abc import Iterator` → delete (check `Iterator` is unused elsewhere; it is only used by `_open_archive_and_tree`).
- `from contextlib import contextmanager` → delete (only used by `_open_archive_and_tree`).
- `from . import json as _json` → delete (only used by `_open_archive_and_tree`).

Keep `from . import fits as _fits` (still used by `_get_compression_options`) and `import astropy.io.fits`.

Delete the `_extension_from_uri` method and the entire `_open_archive_and_tree` context manager.

Replace the `read_from_uri` method with:

```python
    def read_from_uri(
        self,
        uri: ResourcePath,
        component: str | None = None,
        expected_size: int = -1,
    ) -> Any:
        kwargs = self.file_descriptor.parameters or {}
        pytype: type[Any] = self.dataset_ref.datasetType.storageClass.pytype
        with ser.open(uri, cls=pytype, partial=bool(kwargs or component)) as reader:
            if component is None:
                return reader.read(**kwargs)
            return reader.get_component(component, **kwargs)
```

- [ ] **Step 3: Run the full suite to verify no regression**

Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m pytest tests/ -q`
Expected: PASS (same pass/skip counts as before, plus the new reader tests).

- [ ] **Step 4: Lint and type-check**

Run: `~/pyenv/bin/ruff check python/lsst/images/formatters.py`
Expected: `All checks passed!` (no unused imports).
Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m mypy python/lsst/images/formatters.py 2>&1 | grep -v frozendict`
Expected: no errors in `formatters.py`.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/formatters.py
git commit -m "Read through serialization.open() in the formatter (DM-55131)"
```

---

## Task 7: News fragment

**Files:**
- Modify: `doc/changes/DM-55131.feature.md`

- [ ] **Step 1: Append a line**

Add to the end of `doc/changes/DM-55131.feature.md`:

```markdown
Added ``lsst.images.serialization.open``, a context-manager reader for pulling individual components (``reader.get_component("projection")``) or the whole object out of a file efficiently, layered on the new ``InputArchive.open_tree`` primitive. ``open`` and ``read`` accept an optional ``cls`` argument that validates the deserialized type and narrows the static return type.
```

- [ ] **Step 2: Commit**

```bash
git add doc/changes/DM-55131.feature.md
git commit -m "Add news fragment for serialization.open() (DM-55131)"
```

---

## Self-Review

**Spec coverage:**
- User-facing `open()` / `Reader` (get_component, read, info, metadata, butler_info, with-scoped) → Task 4.
- `cls` validation + typing on `open()` → Task 4; on `read()` → Task 5.
- Internal `InputArchive.open_tree` primitive (base + 3 backends) → Tasks 1–3.
- `read_tree` rests on `open_tree` → Tasks 1–3.
- Formatter rewired onto `open()`, dead code removed → Task 6.
- `TypeError` for cls mismatch → Tasks 4 & 5. Unregistered schema → `ArchiveReadError` (Task 4). Unknown component → `InvalidComponentError` (Task 4 test). Use-after-close → `RuntimeError` (Task 4).
- No `list_components` → not implemented (out of scope, per spec).
- News fragment → Task 7.

**Placeholder scan:** No TBD/TODO; every code step shows complete code; test steps show full assertions.

**Type consistency:** `open_tree(path, tree_cls, *, partial=True, **backend_kwargs)` identical across base and all three backends (FITS pulls `page_size` from `backend_kwargs`, keeping the signature uniform for Liskov/mypy). `Reader[T]` methods (`get_component`, `read`, `info`, `metadata`, `butler_info`) match between definition (Task 4) and tests. `read[T](path, cls, **kwargs) -> ReadResult[T]` overloads consistent with the `ReadResult.deserialized` field used in checks.

**Known limitation carried from spec:** JSON `open()` parses twice (get_basic_info + open_tree), matching today's `read()`; acceptable.
