# `lsst-images-admin` CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `click`-based `lsst-images-admin` command with `convert`, `inspect`, `minify`, and the existing `extract-test-data` subcommands, backed by two new shared read APIs in the `serialization` layer.

**Architecture:** First land two reusable APIs in `lsst.images.serialization` — `backend_for_path` (suffix → backend, R1) and `InputArchive.get_basic_info` (per-backend header peek returning an `ArchiveInfo` model, R2). Refactor the `minify` helper to consume them. Then add a `lsst.images.cli` package exposing the `lsst-images-admin` entry point with `convert` (legacy FITS → new format), `inspect` (schema URL + format version), and registrations of the existing `minify`/`extract-test-data` helpers.

**Tech Stack:** Python 3.12+, `click`, `pydantic`, `astropy.io.fits`, `lsst.resources`, optional `h5py`/`afw`/`cell_coadds`/`daf_butler`; tests via `unittest` run under `pytest`.

---

## Conventions for every task

- **Interpreter / path:** run everything with the repo interpreter and this checkout on `PYTHONPATH`:
  `PYTHONPATH=./python ~/pyenv/bin/python -m pytest <args>`
- **License header:** every new `.py` file starts with the standard 10-line header (copy verbatim from any existing file, e.g. `python/lsst/images/serialization/_input_archive.py` lines 1-10).
- **Lint gate (before each commit that touches `.py`):**
  `~/pyenv/bin/python -m ruff check <changed paths>` → no errors;
  `~/pyenv/bin/python -m ruff format <changed paths>` → applies formatting;
  `~/pyenv/bin/python -m mypy python/lsst/images/<changed dir>` → no *new* errors (pre-existing errors in unrelated dirs may be ignored).
- **One sentence per line** in any prose docs (Markdown/RST) touched.
- Tests are `unittest.TestCase` subclasses (repo style) but executed with `pytest`.

## File structure

| File | Responsibility |
|------|----------------|
| `python/lsst/images/serialization/_input_archive.py` (modify) | Add `ArchiveInfo` model + `InputArchive.get_basic_info` classmethod (R2 base). |
| `python/lsst/images/serialization/_backends.py` (create) | `Backend` dataclass + `backend_for_path` (R1). |
| `python/lsst/images/serialization/__init__.py` (modify) | Export `_backends`. |
| `python/lsst/images/json/_input_archive.py` (modify) | `JsonInputArchive.get_basic_info`. |
| `python/lsst/images/fits/_input_archive.py` (modify) | `FitsInputArchive.get_basic_info`. |
| `python/lsst/images/ndf/_input_archive.py` (modify) | `NdfInputArchive.get_basic_info`. |
| `python/lsst/images/tests/_minify_for_fixtures.py` (modify) | Consume R1/R2; delete private peek/reader helpers. |
| `python/lsst/images/tests/extract_legacy_test_data.py` (modify) | Make heavy imports lazy. |
| `python/lsst/images/cli/__init__.py` (create) | Export `main`. |
| `python/lsst/images/cli/_main.py` (create) | Root `click.group`; register subcommands. |
| `python/lsst/images/cli/_inspect.py` (create) | `inspect` command. |
| `python/lsst/images/cli/_convert.py` (create) | `convert` command + legacy type detection. |
| `python/lsst/images/cli/_minify.py` (create) | Thin `minify` click wrapper. |
| `pyproject.toml` (modify) | Add `click` dep + `[project.scripts]`. |
| `tests/test_serialization_basic_info.py` (create) | R2 tests. |
| `tests/test_serialization_backends.py` (create) | R1 tests. |
| `tests/test_cli.py` (create) | CLI tests (help, inspect, convert). |
| `doc/changes/DM-55131.feature.md` (create) | News fragment. |

---

## Task 1: `ArchiveInfo` model + `get_basic_info` base method (R2 base)

**Files:**
- Modify: `python/lsst/images/serialization/_input_archive.py`
- Test: `tests/test_serialization_basic_info.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_serialization_basic_info.py`:

```python
# <standard 10-line license header>
from __future__ import annotations

import unittest

from lsst.images.serialization import ArchiveInfo


class ArchiveInfoTestCase(unittest.TestCase):
    """Tests for the ArchiveInfo model and its schema_url parsing."""

    def test_from_schema_url(self) -> None:
        info = ArchiveInfo.from_schema_url(
            "https://images.lsst.io/schemas/visit_image-1.2.3", format_version=1
        )
        self.assertEqual(info.schema_name, "visit_image")
        self.assertEqual(info.schema_version, "1.2.3")
        self.assertEqual(info.schema_url, "https://images.lsst.io/schemas/visit_image-1.2.3")
        self.assertEqual(info.format_version, 1)

    def test_from_schema_url_none_format(self) -> None:
        info = ArchiveInfo.from_schema_url(
            "https://images.lsst.io/schemas/image-1.0.0", format_version=None
        )
        self.assertEqual(info.schema_name, "image")
        self.assertIsNone(info.format_version)

    def test_frozen(self) -> None:
        info = ArchiveInfo.from_schema_url("https://x/schemas/image-1.0.0", format_version=None)
        with self.assertRaises(Exception):
            info.schema_name = "other"  # type: ignore[misc]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=./python ~/pyenv/bin/python -m pytest tests/test_serialization_basic_info.py -v`
Expected: FAIL with `ImportError: cannot import name 'ArchiveInfo'`.

- [ ] **Step 3: Add the model and base method**

In `python/lsst/images/serialization/_input_archive.py`, update `__all__` to include `"ArchiveInfo"`:

```python
__all__ = ("ArchiveInfo", "InputArchive")
```

Add this import near the top imports (after the existing `import pydantic`):

```python
from lsst.resources import ResourcePathExpression
```

Add the `ArchiveInfo` class immediately above `class InputArchive`:

```python
class ArchiveInfo(pydantic.BaseModel, frozen=True):
    """Basic identifying information about an on-disk archive.

    Read from a file's headers/metadata without deserializing pixel data.
    """

    schema_url: str
    """Canonical schema URL of the top-level tree."""

    schema_name: str
    """Schema name parsed from ``schema_url``."""

    schema_version: str
    """Schema version parsed from ``schema_url``."""

    format_version: int | None
    """Container layout version (FITS ``FMTVER`` / NDF ``FORMAT_VERSION``);
    `None` for formats with no separate container version (JSON)."""

    @classmethod
    def from_schema_url(cls, schema_url: str, *, format_version: int | None) -> ArchiveInfo:
        """Build an `ArchiveInfo` by parsing a schema URL of the form
        ``.../schemas/{name}-{version}``.
        """
        tail = schema_url.rsplit("/", 1)[-1]
        name, _, version = tail.rpartition("-")
        if not name or not version:
            raise ValueError(f"Cannot parse schema name/version from URL {schema_url!r}.")
        return cls(
            schema_url=schema_url,
            schema_name=name,
            schema_version=version,
            format_version=format_version,
        )
```

Add this classmethod inside `class InputArchive`, after the class docstring and before `deserialize_pointer`:

```python
    @classmethod
    def get_basic_info(cls, path: ResourcePathExpression) -> ArchiveInfo:
        """Return basic identifying information for the archive at ``path``
        without deserializing pixel data.

        Each concrete backend reads only the headers/metadata it needs.
        """
        raise NotImplementedError(f"{cls.__name__} does not implement get_basic_info.")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=./python ~/pyenv/bin/python -m pytest tests/test_serialization_basic_info.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint**

Run: `~/pyenv/bin/python -m ruff check python/lsst/images/serialization/_input_archive.py tests/test_serialization_basic_info.py && ~/pyenv/bin/python -m ruff format python/lsst/images/serialization/_input_archive.py tests/test_serialization_basic_info.py && ~/pyenv/bin/python -m mypy python/lsst/images/serialization`
Expected: no new errors.

- [ ] **Step 6: Commit**

```bash
git add python/lsst/images/serialization/_input_archive.py tests/test_serialization_basic_info.py
git commit -m "Add ArchiveInfo model and InputArchive.get_basic_info base (DM-55131)"
```

---

## Task 2: `JsonInputArchive.get_basic_info` (R2 JSON)

**Files:**
- Modify: `python/lsst/images/json/_input_archive.py`
- Test: `tests/test_serialization_basic_info.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_serialization_basic_info.py`:

```python
import os
import tempfile

import numpy as np

from lsst.images import Box, Image
from lsst.images import json as images_json
from lsst.images.json import JsonInputArchive


class JsonBasicInfoTestCase(unittest.TestCase):
    """get_basic_info for the JSON backend."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.image = Image(np.zeros((4, 4), dtype=np.float32), bbox=Box.factory[0:4, 0:4])

    def test_json_basic_info(self) -> None:
        path = os.path.join(self.tmp, "x.json")
        images_json.write(self.image, path)
        info = JsonInputArchive.get_basic_info(path)
        self.assertEqual(info.schema_name, "image")
        self.assertEqual(info.schema_version, "1.0.0")
        self.assertEqual(info.schema_url, "https://images.lsst.io/schemas/image-1.0.0")
        self.assertIsNone(info.format_version)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=./python ~/pyenv/bin/python -m pytest tests/test_serialization_basic_info.py::JsonBasicInfoTestCase -v`
Expected: FAIL with `NotImplementedError: JsonInputArchive does not implement get_basic_info.`

- [ ] **Step 3: Implement**

In `python/lsst/images/json/_input_archive.py`, add to the imports from `..serialization` (the block that already imports `ArchiveTree`, `MetadataValue`, `ReadResult`):

```python
from ..serialization import (
    ArchiveInfo,
    ArchiveReadError,
    # ...existing names...
)
```

Add this classmethod inside `class JsonInputArchive` (after the class docstring):

```python
    @classmethod
    def get_basic_info(cls, path: ResourcePathExpression) -> ArchiveInfo:
        """Read the top-level tree's ``schema_url``; JSON has no container
        format version.
        """
        raw = from_json(ResourcePath(path).read())
        if not isinstance(raw, dict) or not raw.get("schema_url"):
            raise ArchiveReadError(
                f"{path!r} has no schema_url in its top-level JSON tree."
            )
        return ArchiveInfo.from_schema_url(raw["schema_url"], format_version=None)
```

(`from_json`, `ResourcePath`, and `ResourcePathExpression` are already imported in this module.)

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=./python ~/pyenv/bin/python -m pytest tests/test_serialization_basic_info.py::JsonBasicInfoTestCase -v`
Expected: PASS.

- [ ] **Step 5: Lint**

Run: `~/pyenv/bin/python -m ruff check python/lsst/images/json/_input_archive.py tests/test_serialization_basic_info.py && ~/pyenv/bin/python -m ruff format python/lsst/images/json/_input_archive.py && ~/pyenv/bin/python -m mypy python/lsst/images/json`
Expected: no new errors.

- [ ] **Step 6: Commit**

```bash
git add python/lsst/images/json/_input_archive.py tests/test_serialization_basic_info.py
git commit -m "Implement JsonInputArchive.get_basic_info (DM-55131)"
```

---

## Task 3: `FitsInputArchive.get_basic_info` (R2 FITS)

**Files:**
- Modify: `python/lsst/images/fits/_input_archive.py`
- Test: `tests/test_serialization_basic_info.py`

FITS stores the schema URL in the primary-header `DATAMODL` card and the container version in `FMTVER`, so the common path needs no HDU seek. A pre-`DATAMODL` file falls back to reading the JSON HDU via `JSONADDR`/`JSONSIZE`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_serialization_basic_info.py`:

```python
from lsst.images import fits as images_fits
from lsst.images.fits import FitsInputArchive


class FitsBasicInfoTestCase(unittest.TestCase):
    """get_basic_info for the FITS backend."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.image = Image(np.zeros((4, 4), dtype=np.float32), bbox=Box.factory[0:4, 0:4])

    def test_fits_basic_info(self) -> None:
        path = os.path.join(self.tmp, "x.fits")
        images_fits.write(self.image, path)
        info = FitsInputArchive.get_basic_info(path)
        self.assertEqual(info.schema_name, "image")
        self.assertEqual(info.schema_version, "1.0.0")
        self.assertEqual(info.schema_url, "https://images.lsst.io/schemas/image-1.0.0")
        self.assertEqual(info.format_version, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=./python ~/pyenv/bin/python -m pytest tests/test_serialization_basic_info.py::FitsBasicInfoTestCase -v`
Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement**

In `python/lsst/images/fits/_input_archive.py`, ensure `json`, `ResourcePath`, `ResourcePathExpression`, `astropy.io.fits`, `ArchiveInfo`, and `ArchiveReadError` are imported (add `import json` and the two serialization names if absent).

Add this classmethod inside `class FitsInputArchive` (after the class docstring, before `__init__`):

```python
    @classmethod
    def get_basic_info(cls, path: ResourcePathExpression) -> ArchiveInfo:
        """Read ``DATAMODL`` (schema URL) and ``FMTVER`` (container version)
        from the primary header, falling back to the JSON HDU if ``DATAMODL``
        is absent.
        """
        with ResourcePath(path).open("rb") as stream:
            primary = astropy.io.fits.PrimaryHDU.readfrom(stream)
            header = primary.header
            format_version = int(header.get("FMTVER", 1))
            schema_url = header.get("DATAMODL")
            if not schema_url:
                try:
                    json_address = header["JSONADDR"]
                    json_size = header["JSONSIZE"]
                except KeyError:
                    raise ArchiveReadError(
                        f"{path!r} is not an lsst.images FITS archive "
                        "(no DATAMODL or JSONADDR/JSONSIZE cards)."
                    ) from None
                stream.seek(json_address)
                json_hdu = astropy.io.fits.BinTableHDU.fromstring(stream.read(json_size))
                payload = bytes(json_hdu.data[0][0])
                schema_url = json.loads(payload.decode("utf-8")).get("schema_url")
        if not schema_url:
            raise ArchiveReadError(f"Could not determine the schema of {path!r}.")
        return ArchiveInfo.from_schema_url(schema_url, format_version=format_version)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=./python ~/pyenv/bin/python -m pytest tests/test_serialization_basic_info.py::FitsBasicInfoTestCase -v`
Expected: PASS.

- [ ] **Step 5: Lint**

Run: `~/pyenv/bin/python -m ruff check python/lsst/images/fits/_input_archive.py && ~/pyenv/bin/python -m ruff format python/lsst/images/fits/_input_archive.py && ~/pyenv/bin/python -m mypy python/lsst/images/fits`
Expected: no new errors.

- [ ] **Step 6: Commit**

```bash
git add python/lsst/images/fits/_input_archive.py tests/test_serialization_basic_info.py
git commit -m "Implement FitsInputArchive.get_basic_info (DM-55131)"
```

---

## Task 4: `NdfInputArchive.get_basic_info` (R2 NDF)

**Files:**
- Modify: `python/lsst/images/ndf/_input_archive.py`
- Test: `tests/test_serialization_basic_info.py`

NDF is HDS-on-HDF5: the top-level tree lives in a `JSON` dataset and the container version in a `FORMAT_VERSION` primitive under `/MORE/LSST` (or `/LSST`). This mirrors the now-deleted `minify._peek_ndf_top_json` walk. The test is gated on `h5py`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_serialization_basic_info.py`:

```python
try:
    import h5py  # noqa: F401

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False


@unittest.skipUnless(HAVE_H5PY, "h5py is not available.")
class NdfBasicInfoTestCase(unittest.TestCase):
    """get_basic_info for the NDF backend."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.image = Image(np.zeros((4, 4), dtype=np.float32), bbox=Box.factory[0:4, 0:4])

    def test_ndf_basic_info(self) -> None:
        from lsst.images import ndf as images_ndf
        from lsst.images.ndf import NdfInputArchive

        path = os.path.join(self.tmp, "x.sdf")
        images_ndf.write(self.image, path)
        info = NdfInputArchive.get_basic_info(path)
        self.assertEqual(info.schema_name, "image")
        self.assertEqual(info.schema_version, "1.0.0")
        self.assertEqual(info.format_version, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=./python ~/pyenv/bin/python -m pytest tests/test_serialization_basic_info.py::NdfBasicInfoTestCase -v`
Expected: FAIL with `NotImplementedError` (or skip if `h5py` absent — install it: `~/pyenv/bin/python -m pip install h5py`).

- [ ] **Step 3: Implement**

In `python/lsst/images/ndf/_input_archive.py`, ensure `json`, `numpy as np`, `ResourcePath`, `ResourcePathExpression`, `ArchiveInfo`, `ArchiveReadError` are imported. Add this classmethod inside `class NdfInputArchive` (after the class docstring):

```python
    @classmethod
    def get_basic_info(cls, path: ResourcePathExpression) -> ArchiveInfo:
        """Read the top-level ``JSON`` tree's ``schema_url`` and the
        ``FORMAT_VERSION`` primitive without deserializing pixel data.
        """
        import h5py

        ospath = ResourcePath(path).ospath
        schema_url: str | None = None
        format_version = 1

        def visit(name: str, item: object) -> bool | None:
            nonlocal schema_url
            if schema_url is not None:
                return True
            if isinstance(item, h5py.Dataset) and name.rsplit("/", 1)[-1] == "JSON":
                try:
                    payload = bytes(np.asarray(item).tobytes())
                    obj = json.loads(payload.decode("utf-8").rstrip("\x00").strip())
                except (UnicodeDecodeError, ValueError, TypeError):
                    return None
                if isinstance(obj, dict) and obj.get("schema_url"):
                    schema_url = obj["schema_url"]
                    return True
            return None

        with h5py.File(ospath, "r") as handle:
            handle.visititems(visit)
            for prefix in ("MORE/LSST", "LSST"):
                node = handle.get(f"{prefix}/FORMAT_VERSION")
                if node is not None:
                    format_version = int(np.asarray(node).item())
                    break
        if schema_url is None:
            raise ArchiveReadError(
                f"Could not locate the top-level JSON tree in {path!r}."
            )
        return ArchiveInfo.from_schema_url(schema_url, format_version=format_version)
```

(If `handle.get("MORE/LSST/FORMAT_VERSION")` returns a group rather than a dataset in the HDS layout, adjust to read the contained dataset — the gated test in Step 4 is the validation; the `/MORE/LSST` and `/LSST` prefixes match the existing `NdfInputArchive._check_format_version`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=./python ~/pyenv/bin/python -m pytest tests/test_serialization_basic_info.py::NdfBasicInfoTestCase -v`
Expected: PASS (or skip if `h5py` unavailable).

- [ ] **Step 5: Lint**

Run: `~/pyenv/bin/python -m ruff check python/lsst/images/ndf/_input_archive.py && ~/pyenv/bin/python -m ruff format python/lsst/images/ndf/_input_archive.py && ~/pyenv/bin/python -m mypy python/lsst/images/ndf`
Expected: no new errors.

- [ ] **Step 6: Commit**

```bash
git add python/lsst/images/ndf/_input_archive.py tests/test_serialization_basic_info.py
git commit -m "Implement NdfInputArchive.get_basic_info (DM-55131)"
```

---

## Task 5: `backend_for_path` (R1)

**Files:**
- Create: `python/lsst/images/serialization/_backends.py`
- Modify: `python/lsst/images/serialization/__init__.py`
- Test: `tests/test_serialization_backends.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_serialization_backends.py`:

```python
# <standard 10-line license header>
from __future__ import annotations

import unittest

from lsst.images.serialization import Backend, backend_for_path


class BackendForPathTestCase(unittest.TestCase):
    """Tests for suffix -> backend resolution."""

    def test_fits(self) -> None:
        from lsst.images.fits import FitsInputArchive

        b = backend_for_path("a/b/c.fits")
        self.assertIsInstance(b, Backend)
        self.assertEqual(b.name, "fits")
        self.assertIs(b.input_archive, FitsInputArchive)
        self.assertTrue(callable(b.read) and callable(b.write))

    def test_fits_gz(self) -> None:
        self.assertEqual(backend_for_path("c.fits.gz").name, "fits")

    def test_json(self) -> None:
        from lsst.images.json import JsonInputArchive

        b = backend_for_path("c.json")
        self.assertEqual(b.name, "json")
        self.assertIs(b.input_archive, JsonInputArchive)

    def test_ndf(self) -> None:
        self.assertEqual(backend_for_path("c.sdf").name, "ndf")
        self.assertEqual(backend_for_path("c.ndf").name, "ndf")

    def test_unknown(self) -> None:
        with self.assertRaises(ValueError) as cm:
            backend_for_path("c.txt")
        self.assertIn(".fits", str(cm.exception))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=./python ~/pyenv/bin/python -m pytest tests/test_serialization_backends.py -v`
Expected: FAIL with `ImportError: cannot import name 'Backend'`.

- [ ] **Step 3: Implement**

Create `python/lsst/images/serialization/_backends.py`:

```python
# <standard 10-line license header>
from __future__ import annotations

__all__ = ("Backend", "backend_for_path")

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING

from lsst.resources import ResourcePathExpression

if TYPE_CHECKING:
    from ._input_archive import InputArchive


@dataclasses.dataclass(frozen=True)
class Backend:
    """A file-format backend resolved from a path suffix.

    Bundles the backend's free ``read``/``write`` functions and its
    `InputArchive` subclass (whose `~InputArchive.get_basic_info` reads
    file metadata).
    """

    name: str
    read: Callable[..., object]
    write: Callable[..., object]
    input_archive: type[InputArchive]


def backend_for_path(path: ResourcePathExpression) -> Backend:
    """Return the `Backend` for ``path`` based on its file extension.

    Supported extensions: ``.fits`` / ``.fits.gz`` (FITS), ``.sdf`` /
    ``.ndf`` (NDF), and ``.json`` (JSON).  The NDF and FITS backends are
    imported lazily so optional dependencies (e.g. ``h5py``) are only
    required when actually used.

    Raises
    ------
    ValueError
        If the extension is not recognised.
    """
    s = str(path)
    if s.endswith(".fits") or s.endswith(".fits.gz"):
        from ..fits import FitsInputArchive, read, write

        return Backend("fits", read, write, FitsInputArchive)
    if s.endswith(".sdf") or s.endswith(".ndf"):
        from ..ndf import NdfInputArchive, read, write

        return Backend("ndf", read, write, NdfInputArchive)
    if s.endswith(".json"):
        from ..json import JsonInputArchive, read, write

        return Backend("json", read, write, JsonInputArchive)
    raise ValueError(
        f"Unrecognised file extension: {path!r}; "
        "expected one of .fits, .fits.gz, .sdf, .ndf, .json."
    )
```

In `python/lsst/images/serialization/__init__.py`, add after the existing `from ._input_archive import *` line:

```python
from ._backends import *
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=./python ~/pyenv/bin/python -m pytest tests/test_serialization_backends.py -v`
Expected: PASS (skips none; `.sdf`/`.ndf` resolution does not import `h5py` because `lsst.images.ndf` re-exports `NdfInputArchive`/`read`/`write` without importing `h5py` at module load — if that import fails for another reason, mark `test_ndf` to skip on `ImportError`).

- [ ] **Step 5: Lint**

Run: `~/pyenv/bin/python -m ruff check python/lsst/images/serialization/_backends.py python/lsst/images/serialization/__init__.py tests/test_serialization_backends.py && ~/pyenv/bin/python -m ruff format python/lsst/images/serialization/_backends.py tests/test_serialization_backends.py && ~/pyenv/bin/python -m mypy python/lsst/images/serialization`
Expected: no new errors.

- [ ] **Step 6: Commit**

```bash
git add python/lsst/images/serialization/_backends.py python/lsst/images/serialization/__init__.py tests/test_serialization_backends.py
git commit -m "Add backend_for_path suffix dispatch (DM-55131)"
```

---

## Task 6: Refactor `minify` onto R1/R2

**Files:**
- Modify: `python/lsst/images/tests/_minify_for_fixtures.py`
- Test: `tests/test_serialization_backends.py`

Delete the private `_read_function`, `_detect_schema_name`, `_peek_fits_top_json`, and `_peek_ndf_top_json` helpers (and any now-unused imports such as `astropy.io.fits` at module scope if no longer referenced) and route `minify` through `backend_for_path` + `get_basic_info`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_serialization_backends.py`:

```python
import os
import tempfile

import numpy as np

from lsst.images import Box, Image
from lsst.images import fits as images_fits


class MinifyDispatchTestCase(unittest.TestCase):
    """minify resolves backend and schema via the shared APIs."""

    def test_minify_unsupported_schema_uses_shared_dispatch(self) -> None:
        from lsst.images.tests._minify_for_fixtures import minify

        tmp = tempfile.mkdtemp()
        src = os.path.join(tmp, "plain.fits")
        out = os.path.join(tmp, "plain.json")
        images_fits.write(Image(np.zeros((4, 4), dtype=np.float32), bbox=Box.factory[0:4, 0:4]), src)
        # Reaching the "no subsetter" error proves backend_for_path + get_basic_info
        # ran and detected schema_name "image".
        with self.assertRaises(NotImplementedError) as cm:
            minify(src, out)
        self.assertIn("image", str(cm.exception))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=./python ~/pyenv/bin/python -m pytest tests/test_serialization_backends.py::MinifyDispatchTestCase -v`
Expected: FAIL — currently raises `ValueError`/other (old code path) or a different message; confirm it does not yet pass.

- [ ] **Step 3: Implement**

In `python/lsst/images/tests/_minify_for_fixtures.py`:

Replace the body of `minify` (the function defined at `def minify(in_path: str, out_path: str, *, schema_name: str | None = None) -> None:`) with:

```python
def minify(in_path: str, out_path: str, *, schema_name: str | None = None) -> None:
    """Read a real archive at ``in_path``, take a small subset, and write JSON.

    Parameters
    ----------
    in_path
        Path to a FITS (``.fits`` / ``.fits.gz``) or NDF (``.sdf`` / ``.ndf``)
        file to read.
    out_path
        Path to the JSON fixture to write. The parent directory is
        created if it does not exist.
    schema_name
        Top-level schema name (e.g. ``"visit_image"`` or ``"cell_coadd"``).
        If `None`, it is auto-detected from the file.

    Raises
    ------
    ValueError
        If the file extension is not recognised.
    NotImplementedError
        If the top-level type is not one this helper knows how to subset.
    """
    backend = backend_for_path(in_path)
    if schema_name is None:
        schema_name = backend.input_archive.get_basic_info(in_path).schema_name

    cls, subsetter = _dispatch(schema_name)

    obj, _, _ = backend.read(cls, in_path)
    subset = subsetter(obj)

    tree = images_json.write(subset)
    dumped = tree.model_dump(mode="json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as stream:
        stream.write(json.dumps(dumped, indent=2, sort_keys=False) + "\n")
```

Delete the functions `_read_function`, `_detect_schema_name`, `_peek_fits_top_json`, and `_peek_ndf_top_json` in their entirety.

Update the module imports: remove `from ..fits import read as fits_read` (no longer used) and add to the existing `from ..serialization import ...` (or add a new import line):

```python
from ..serialization import backend_for_path
```

If `astropy.io.fits` is no longer referenced anywhere else in the module, remove its import; otherwise leave it. (Search the file for remaining uses before deciding.)

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=./python ~/pyenv/bin/python -m pytest tests/test_serialization_backends.py::MinifyDispatchTestCase -v`
Expected: PASS.

- [ ] **Step 5: Lint**

Run: `~/pyenv/bin/python -m ruff check python/lsst/images/tests/_minify_for_fixtures.py tests/test_serialization_backends.py && ~/pyenv/bin/python -m ruff format python/lsst/images/tests/_minify_for_fixtures.py && ~/pyenv/bin/python -m mypy python/lsst/images/tests`
Expected: no new errors.

- [ ] **Step 6: Commit**

```bash
git add python/lsst/images/tests/_minify_for_fixtures.py tests/test_serialization_backends.py
git commit -m "Route minify through backend_for_path/get_basic_info (DM-55131)"
```

---

## Task 7: CLI package skeleton + entry point

**Files:**
- Create: `python/lsst/images/cli/__init__.py`, `python/lsst/images/cli/_main.py`
- Modify: `pyproject.toml`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli.py`:

```python
# <standard 10-line license header>
from __future__ import annotations

import unittest

from click.testing import CliRunner

from lsst.images.cli import main


class CliSkeletonTestCase(unittest.TestCase):
    """The root group loads and shows help with core deps only."""

    def test_group_help(self) -> None:
        result = CliRunner().invoke(main, ["--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("convert", result.output)
        self.assertIn("inspect", result.output)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=./python ~/pyenv/bin/python -m pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lsst.images.cli'`.

- [ ] **Step 3: Implement**

Create `python/lsst/images/cli/__init__.py`:

```python
# <standard 10-line license header>
from __future__ import annotations

__all__ = ("main",)

from ._main import main
```

Create `python/lsst/images/cli/_main.py`:

```python
# <standard 10-line license header>
from __future__ import annotations

__all__ = ("main",)

import click

from ._convert import convert
from ._inspect import inspect


@click.group(name="lsst-images-admin")
def main() -> None:
    """Administrative tools for lsst.images files."""


main.add_command(convert)
main.add_command(inspect)
```

To make this task self-contained, create minimal placeholder commands now (they are fully implemented in Tasks 8-9). Create `python/lsst/images/cli/_inspect.py`:

```python
# <standard 10-line license header>
from __future__ import annotations

__all__ = ("inspect",)

import click


@click.command(name="inspect")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
def inspect(file: str) -> None:
    """Print basic information about an lsst.images file."""
    raise click.ClickException("not yet implemented")
```

Create `python/lsst/images/cli/_convert.py`:

```python
# <standard 10-line license header>
from __future__ import annotations

__all__ = ("convert",)

import click


@click.command(name="convert")
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(dir_okay=False))
def convert(input: str, output: str) -> None:
    """Convert a legacy FITS file to a new lsst.images format."""
    raise click.ClickException("not yet implemented")
```

In `pyproject.toml`, add `click` to `dependencies` (after `"shapely >= 2.1",`):

```toml
    "click >= 8",
```

And add a `[project.scripts]` table (place it right after the `[project.urls]` block):

```toml
[project.scripts]
lsst-images-admin = "lsst.images.cli:main"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=./python ~/pyenv/bin/python -m pytest tests/test_cli.py -v`
Expected: PASS.

- [ ] **Step 5: Lint**

Run: `~/pyenv/bin/python -m ruff check python/lsst/images/cli tests/test_cli.py && ~/pyenv/bin/python -m ruff format python/lsst/images/cli tests/test_cli.py && ~/pyenv/bin/python -m mypy python/lsst/images/cli`
Expected: no new errors.

- [ ] **Step 6: Commit**

```bash
git add python/lsst/images/cli pyproject.toml tests/test_cli.py
git commit -m "Add lsst-images-admin CLI skeleton and entry point (DM-55131)"
```

---

## Task 8: `inspect` command

**Files:**
- Modify: `python/lsst/images/cli/_inspect.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py`:

```python
import os
import tempfile

import numpy as np

from lsst.images import Box, Image
from lsst.images import fits as images_fits
from lsst.images import json as images_json


class InspectTestCase(unittest.TestCase):
    """inspect prints schema URL and format version."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.image = Image(np.zeros((4, 4), dtype=np.float32), bbox=Box.factory[0:4, 0:4])

    def test_inspect_fits(self) -> None:
        path = os.path.join(self.tmp, "x.fits")
        images_fits.write(self.image, path)
        result = CliRunner().invoke(main, ["inspect", path])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("https://images.lsst.io/schemas/image-1.0.0", result.output)
        self.assertIn("1", result.output)  # format version

    def test_inspect_json(self) -> None:
        path = os.path.join(self.tmp, "x.json")
        images_json.write(self.image, path)
        result = CliRunner().invoke(main, ["inspect", path])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("image-1.0.0", result.output)
        self.assertIn("n/a", result.output)  # no container format version for JSON

    def test_inspect_unknown_extension(self) -> None:
        path = os.path.join(self.tmp, "x.txt")
        with open(path, "w") as stream:
            stream.write("nope")
        result = CliRunner().invoke(main, ["inspect", path])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn(".fits", result.output)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=./python ~/pyenv/bin/python -m pytest tests/test_cli.py::InspectTestCase -v`
Expected: FAIL (`not yet implemented`).

- [ ] **Step 3: Implement**

Replace `python/lsst/images/cli/_inspect.py` with:

```python
# <standard 10-line license header>
from __future__ import annotations

__all__ = ("inspect",)

import click

from ..serialization import backend_for_path


@click.command(name="inspect")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
def inspect(file: str) -> None:
    """Print basic information about an lsst.images file.

    Reports the schema URL and container format version without
    deserializing pixel data.
    """
    try:
        backend = backend_for_path(file)
    except ValueError as err:
        raise click.ClickException(str(err)) from None
    info = backend.input_archive.get_basic_info(file)
    fmt = "n/a" if info.format_version is None else str(info.format_version)
    click.echo(f"path:           {file}")
    click.echo(f"format:         {backend.name}")
    click.echo(f"schema name:    {info.schema_name}")
    click.echo(f"schema version: {info.schema_version}")
    click.echo(f"schema URL:     {info.schema_url}")
    click.echo(f"format version: {fmt}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=./python ~/pyenv/bin/python -m pytest tests/test_cli.py::InspectTestCase -v`
Expected: PASS.

- [ ] **Step 5: Lint**

Run: `~/pyenv/bin/python -m ruff check python/lsst/images/cli/_inspect.py tests/test_cli.py && ~/pyenv/bin/python -m ruff format python/lsst/images/cli/_inspect.py tests/test_cli.py && ~/pyenv/bin/python -m mypy python/lsst/images/cli`
Expected: no new errors.

- [ ] **Step 6: Commit**

```bash
git add python/lsst/images/cli/_inspect.py tests/test_cli.py
git commit -m "Implement inspect subcommand (DM-55131)"
```

---

## Task 9: `convert` command (output dispatch, detection, visit-image path)

**Files:**
- Modify: `python/lsst/images/cli/_convert.py`
- Test: `tests/test_cli.py`

The legacy type is detected from the `HIERARCH LSST BUTLER DATASETTYPE` FITS header (astropy exposes it as `header["LSST BUTLER DATASETTYPE"]`): a dataset type ending in `visit_image` is a `VisitImage` (covers `visit_image`, `preliminary_visit_image`, and difference images), and one matching the coadd pattern is a `CellCoadd`. When the header is absent or matches neither, `convert` errors and requires `--type`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py`:

```python
EXTERNAL_DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)


class ConvertDetectTestCase(unittest.TestCase):
    """Legacy type detection from the LSST BUTLER DATASETTYPE header.

    These build small FITS files carrying only the discriminating header so
    the test does not depend on a fixture happening to include it.
    """

    def _make(self, dataset_type: str | None) -> str:
        import astropy.io.fits as af

        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "x.fits")
        hdu = af.PrimaryHDU()
        if dataset_type is not None:
            hdu.header["LSST BUTLER DATASETTYPE"] = dataset_type
        hdu.writeto(path)
        return path

    def test_detect_visit_image(self) -> None:
        from lsst.images.cli._convert import detect_legacy_type

        self.assertEqual(detect_legacy_type(self._make("visit_image")), "visit_image")
        self.assertEqual(detect_legacy_type(self._make("preliminary_visit_image")), "visit_image")

    def test_detect_cell_coadd(self) -> None:
        from lsst.images.cli._convert import detect_legacy_type

        self.assertEqual(detect_legacy_type(self._make("deep_coadd_cell_predetection")), "cell_coadd")

    def test_detect_indeterminate(self) -> None:
        from lsst.images.cli._convert import detect_legacy_type

        self.assertIsNone(detect_legacy_type(self._make(None)))
        self.assertIsNone(detect_legacy_type(self._make("camera")))

    @unittest.skipUnless(EXTERNAL_DATA_DIR is not None, "TESTDATA_IMAGES_DIR is not set.")
    def test_detect_visit_image_fixture(self) -> None:
        from lsst.images.cli._convert import detect_legacy_type

        path = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "visit_image.fits")
        self.assertEqual(detect_legacy_type(path), "visit_image")


class ConvertVisitImageTestCase(unittest.TestCase):
    """convert of a legacy visit image (needs afw + testdata)."""

    @unittest.skipUnless(EXTERNAL_DATA_DIR is not None, "TESTDATA_IMAGES_DIR is not set.")
    def test_convert_visit_image_to_json(self) -> None:
        try:
            import lsst.afw.image  # noqa: F401
        except ImportError:
            self.skipTest("afw not available.")
        tmp = tempfile.mkdtemp()
        src = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "visit_image.fits")
        out = os.path.join(tmp, "converted.json")
        result = CliRunner().invoke(main, ["convert", src, out])
        self.assertEqual(result.exit_code, 0, result.output)
        info = backend_for_path(out).input_archive.get_basic_info(out)
        self.assertEqual(info.schema_name, "visit_image")

    @unittest.skipUnless(EXTERNAL_DATA_DIR is not None, "TESTDATA_IMAGES_DIR is not set.")
    def test_convert_refuses_existing_output(self) -> None:
        try:
            import lsst.afw.image  # noqa: F401
        except ImportError:
            self.skipTest("afw not available.")
        tmp = tempfile.mkdtemp()
        src = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "visit_image.fits")
        out = os.path.join(tmp, "exists.json")
        with open(out, "w") as stream:
            stream.write("{}")
        result = CliRunner().invoke(main, ["convert", src, out])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--overwrite", result.output)
```

Add the import `from lsst.images.serialization import backend_for_path` to the top of `tests/test_cli.py`.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=./python ~/pyenv/bin/python -m pytest tests/test_cli.py::ConvertDetectTestCase -v`
Expected: FAIL with `ImportError: cannot import name 'detect_legacy_type'`.

- [ ] **Step 3: Implement**

Replace `python/lsst/images/cli/_convert.py` with:

```python
# <standard 10-line license header>
from __future__ import annotations

__all__ = ("convert",)

import os

import astropy.io.fits
import click

from ..serialization import backend_for_path

_LEGACY_TYPES = ("visit_image", "cell_coadd")


def detect_legacy_type(path: str) -> str | None:
    """Return ``"visit_image"`` or ``"cell_coadd"`` from a legacy FITS file's
    ``HIERARCH LSST BUTLER DATASETTYPE`` header, or `None` if it cannot be
    determined.

    A dataset type ending in ``visit_image`` (e.g. ``visit_image``,
    ``preliminary_visit_image``, difference images) is a `VisitImage`; one
    containing ``coadd`` is a `CellCoadd`.  astropy exposes the
    ``HIERARCH LSST BUTLER DATASETTYPE`` card as ``header["LSST BUTLER DATASETTYPE"]``.
    """
    dataset_type: str | None = None
    with astropy.io.fits.open(path) as hdul:
        for hdu in hdul:
            value = hdu.header.get("LSST BUTLER DATASETTYPE")
            if value:
                dataset_type = str(value)
                break
    if dataset_type is None:
        return None
    if dataset_type.endswith("visit_image"):
        return "visit_image"
    if "coadd" in dataset_type:
        return "cell_coadd"
    return None


def _read_legacy(
    input: str,
    legacy_type: str,
    skymap: str | None,
    butler: str | None,
    collection: str | None,
):
    """Read a legacy FITS file into the corresponding lsst.images object."""
    if legacy_type == "visit_image":
        from .. import VisitImage

        return VisitImage.read_legacy(input)
    # cell_coadd handled in a later task.
    raise click.ClickException(f"Conversion of {legacy_type!r} is not yet implemented.")


@click.command(name="convert")
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(dir_okay=False))
@click.option(
    "--type",
    "type_",
    type=click.Choice(_LEGACY_TYPES),
    default=None,
    help="Legacy input type; overrides auto-detection.",
)
@click.option("--skymap", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Pickled skymap (required for cell coadds unless --butler is given).")
@click.option("--butler", default=None,
              help="Butler repository to resolve the skymap (cell coadds only).")
@click.option("--collection", default=None,
              help="Butler collection holding the skymap (required with --butler).")
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite OUTPUT if it exists.")
def convert(
    input: str,
    output: str,
    type_: str | None,
    skymap: str | None,
    butler: str | None,
    collection: str | None,
    overwrite: bool,
) -> None:
    """Convert a legacy FITS file to a new lsst.images format.

    The output format is chosen from OUTPUT's extension
    (.fits, .sdf/.ndf, .json).
    """
    try:
        backend = backend_for_path(output)
    except ValueError as err:
        raise click.ClickException(str(err)) from None

    legacy_type = type_ or detect_legacy_type(input)
    if legacy_type is None:
        raise click.ClickException(
            f"Could not determine the legacy type of {input!r}; pass --type."
        )

    if os.path.exists(output):
        if not overwrite:
            raise click.ClickException(f"{output!r} already exists; pass --overwrite to replace it.")
        os.remove(output)

    obj = _read_legacy(input, legacy_type, skymap, butler, collection)
    backend.write(obj, output)
    click.echo(f"Wrote {output} ({backend.name}, {legacy_type}).")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=./python ~/pyenv/bin/python -m pytest tests/test_cli.py::ConvertDetectTestCase tests/test_cli.py::ConvertVisitImageTestCase -v`
Expected: PASS (detection tests pass; visit-image tests pass if afw + testdata present, else skip).

- [ ] **Step 5: Lint**

Run: `~/pyenv/bin/python -m ruff check python/lsst/images/cli/_convert.py tests/test_cli.py && ~/pyenv/bin/python -m ruff format python/lsst/images/cli/_convert.py tests/test_cli.py && ~/pyenv/bin/python -m mypy python/lsst/images/cli`
Expected: no new errors. (Note: `from ..import VisitImage` must be written `from .. import VisitImage`.)

- [ ] **Step 6: Commit**

```bash
git add python/lsst/images/cli/_convert.py tests/test_cli.py
git commit -m "Implement convert detection and visit-image path (DM-55131)"
```

---

## Task 10: `convert` cell-coadd path (skymap / butler)

**Files:**
- Modify: `python/lsst/images/cli/_convert.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py`:

```python
class ConvertCellCoaddTestCase(unittest.TestCase):
    """convert of a legacy cell coadd (needs cell_coadds + testdata)."""

    @unittest.skipUnless(EXTERNAL_DATA_DIR is not None, "TESTDATA_IMAGES_DIR is not set.")
    def test_convert_cell_coadd_to_json(self) -> None:
        try:
            import lsst.cell_coadds  # noqa: F401
        except ImportError:
            self.skipTest("cell_coadds not available.")
        tmp = tempfile.mkdtemp()
        legacy_dir = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy")
        src = os.path.join(legacy_dir, "deep_coadd_cell_predetection.fits")
        skymap = os.path.join(legacy_dir, "skyMap.pickle")
        out = os.path.join(tmp, "coadd.json")
        # This fixture has no LSST BUTLER DATASETTYPE header, so pass --type.
        result = CliRunner().invoke(
            main, ["convert", src, out, "--type", "cell_coadd", "--skymap", skymap]
        )
        self.assertEqual(result.exit_code, 0, result.output)
        info = backend_for_path(out).input_archive.get_basic_info(out)
        self.assertEqual(info.schema_name, "cell_coadd")

    @unittest.skipUnless(EXTERNAL_DATA_DIR is not None, "TESTDATA_IMAGES_DIR is not set.")
    def test_convert_cell_coadd_requires_skymap(self) -> None:
        try:
            import lsst.cell_coadds  # noqa: F401
        except ImportError:
            self.skipTest("cell_coadds not available.")
        tmp = tempfile.mkdtemp()
        src = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "deep_coadd_cell_predetection.fits")
        out = os.path.join(tmp, "coadd.json")
        result = CliRunner().invoke(main, ["convert", src, out, "--type", "cell_coadd"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--skymap", result.output)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=./python ~/pyenv/bin/python -m pytest tests/test_cli.py::ConvertCellCoaddTestCase -v`
Expected: FAIL (`Conversion of 'cell_coadd' is not yet implemented.`) or skip if deps absent.

- [ ] **Step 3: Implement**

In `python/lsst/images/cli/_convert.py`, replace the `_read_legacy` function with:

```python
def _load_skymap(skymap: str | None, butler: str | None, collection: str | None, skymap_name: str):
    """Load a skymap object from a pickle path or a butler repository."""
    if skymap is not None:
        import pickle

        with open(skymap, "rb") as stream:
            return pickle.load(stream)
    if butler is not None:
        if collection is None:
            raise click.ClickException("--butler also requires --collection (the skymap's collection).")
        from lsst.daf.butler import Butler

        repo = Butler.from_config(butler)
        return repo.get("skyMap", skymap=skymap_name, collections=collection)
    raise click.ClickException(
        "Converting a cell coadd requires --skymap (a pickled skymap) or --butler."
    )


def _read_legacy(
    input: str,
    legacy_type: str,
    skymap: str | None,
    butler: str | None,
    collection: str | None,
):
    """Read a legacy FITS file into the corresponding lsst.images object."""
    if legacy_type == "visit_image":
        from .. import VisitImage

        return VisitImage.read_legacy(input)
    if legacy_type == "cell_coadd":
        from lsst.cell_coadds import MultipleCellCoadd

        from .. import get_legacy_deep_coadd_mask_planes
        from ..cells import CellCoadd

        legacy = MultipleCellCoadd.read_fits(input)
        sky = _load_skymap(skymap, butler, collection, legacy.identifiers.skymap)
        tract_info = sky[legacy.identifiers.tract]
        return CellCoadd.from_legacy(
            legacy,
            tract_info=tract_info,
            plane_map=get_legacy_deep_coadd_mask_planes(),
        )
    raise click.ClickException(f"Conversion of {legacy_type!r} is not yet implemented.")
```

The `--skymap` pickle path is the primary, dependency-light route (and what the test covers); `--butler` additionally requires `--collection` so the skymap lookup is unambiguous.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=./python ~/pyenv/bin/python -m pytest tests/test_cli.py::ConvertCellCoaddTestCase -v`
Expected: PASS if `cell_coadds` + testdata present, else skip.

- [ ] **Step 5: Lint**

Run: `~/pyenv/bin/python -m ruff check python/lsst/images/cli/_convert.py tests/test_cli.py && ~/pyenv/bin/python -m ruff format python/lsst/images/cli/_convert.py && ~/pyenv/bin/python -m mypy python/lsst/images/cli`
Expected: no new errors.

- [ ] **Step 6: Commit**

```bash
git add python/lsst/images/cli/_convert.py tests/test_cli.py
git commit -m "Implement convert cell-coadd path (DM-55131)"
```

---

## Task 11: Lazy extract-legacy imports + register `minify` / `extract-test-data`

**Files:**
- Modify: `python/lsst/images/tests/extract_legacy_test_data.py`
- Create: `python/lsst/images/cli/_minify.py`
- Modify: `python/lsst/images/cli/_main.py`
- Test: `tests/test_cli.py`

`extract_legacy_test_data.py` currently imports `afw`/`butler`/`cell_coadds` at module scope inside a `try/except ImportError: ... raise` block. Move those imports into the `extract_dp2` command body so the module imports with core deps only; `click` stays at module scope.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli.py`:

```python
class CliRegistrationTestCase(unittest.TestCase):
    """minify and extract-test-data are registered and help loads with core deps."""

    def test_subcommands_present(self) -> None:
        result = CliRunner().invoke(main, ["--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("minify", result.output)
        self.assertIn("extract-test-data", result.output)

    def test_minify_help(self) -> None:
        result = CliRunner().invoke(main, ["minify", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)

    def test_extract_test_data_help(self) -> None:
        result = CliRunner().invoke(main, ["extract-test-data", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)

    def test_import_cli_does_not_import_afw(self) -> None:
        import sys

        # Importing the CLI must not pull in afw (lazy-import contract).
        self.assertNotIn("lsst.afw", [m for m in sys.modules if m == "lsst.afw"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=./python ~/pyenv/bin/python -m pytest tests/test_cli.py::CliRegistrationTestCase -v`
Expected: FAIL — `minify`/`extract-test-data` not in help output.

- [ ] **Step 3: Implement**

In `python/lsst/images/tests/extract_legacy_test_data.py`, change the top-level import block. Replace:

```python
try:
    import click

    from lsst.afw.fits import (
        CompressionAlgorithm,
        CompressionOptions,
        DitherAlgorithm,
        QuantizationOptions,
        ScalingAlgorithm,
    )
    from lsst.cell_coadds import MultipleCellCoadd
    from lsst.daf.butler import Butler, DatasetRef
    from lsst.geom import Box2I, Extent2I, Point2I
    from lsst.utils import getPackageDir
except ImportError as err:
    err.add_note(
        "Updating the test data requires a full Rubin development enviroment with at least "
        "'click', 'afw', 'obs_base', 'meas_extensions_psfex', 'meas_extensions_piff' and 'cell_coadds' "
        "importable. This is not necessary for just running the tests."
    )
    raise
```

with:

```python
import click

if TYPE_CHECKING:
    from lsst.daf.butler import Butler, DatasetRef
```

Add `from typing import TYPE_CHECKING` to the imports. Then, inside `extract_dp2` (the function body, at the top, before any use), add the heavy imports with the original install hint:

```python
    try:
        from lsst.afw.fits import (
            CompressionAlgorithm,
            CompressionOptions,
            DitherAlgorithm,
            QuantizationOptions,
            ScalingAlgorithm,
        )
        from lsst.cell_coadds import MultipleCellCoadd  # noqa: F401
        from lsst.daf.butler import Butler
        from lsst.geom import Box2I, Extent2I, Point2I  # noqa: F401
        from lsst.utils import getPackageDir
    except ImportError as err:
        err.add_note(
            "Updating the test data requires a full Rubin development enviroment with at least "
            "'afw', 'obs_base', 'meas_extensions_psfex', 'meas_extensions_piff' and 'cell_coadds' "
            "importable. This is not necessary for just running the tests."
        )
        raise
```

Any module-scope helper functions (`extract_visit_image`, `extract_cell_coadd`, etc.) that reference `CompressionOptions`, `MultipleCellCoadd`, `Box2I`, etc. must import those names locally too (they are only invoked from `extract_dp2`). Move each helper's required heavy imports to the top of that helper's body. Verify by importing the module with afw absent:
`PYTHONPATH=./python ~/pyenv/bin/python -c "import lsst.images.tests.extract_legacy_test_data"` → must succeed with core deps only.

Create `python/lsst/images/cli/_minify.py`:

```python
# <standard 10-line license header>
from __future__ import annotations

__all__ = ("minify",)

import click


@click.command(name="minify")
@click.argument("input", type=click.Path(exists=True, dir_okay=False))
@click.argument("output", type=click.Path(dir_okay=False))
@click.option("--schema-name", default=None, help="Top-level schema name; auto-detected if omitted.")
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite OUTPUT if it exists.")
def minify(input: str, output: str, schema_name: str | None, overwrite: bool) -> None:
    """Subset a real archive into a small JSON test fixture."""
    import os

    from ..tests._minify_for_fixtures import minify as _minify

    if os.path.exists(output) and not overwrite:
        raise click.ClickException(f"{output!r} already exists; pass --overwrite to replace it.")
    _minify(input, output, schema_name=schema_name)
    click.echo(f"Wrote {output}.")
```

In `python/lsst/images/cli/_main.py`, register the two helpers. Update it to:

```python
# <standard 10-line license header>
from __future__ import annotations

__all__ = ("main",)

import click

from ..tests.extract_legacy_test_data import extract_test_data
from ._convert import convert
from ._inspect import inspect
from ._minify import minify


@click.group(name="lsst-images-admin")
def main() -> None:
    """Administrative tools for lsst.images files."""


main.add_command(convert)
main.add_command(inspect)
main.add_command(minify)
main.add_command(extract_test_data, name="extract-test-data")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=./python ~/pyenv/bin/python -m pytest tests/test_cli.py -v`
Expected: PASS (all CLI tests).

- [ ] **Step 5: Lint**

Run: `~/pyenv/bin/python -m ruff check python/lsst/images/cli python/lsst/images/tests/extract_legacy_test_data.py tests/test_cli.py && ~/pyenv/bin/python -m ruff format python/lsst/images/cli python/lsst/images/tests/extract_legacy_test_data.py && ~/pyenv/bin/python -m mypy python/lsst/images/cli`
Expected: no new errors.

- [ ] **Step 6: Commit**

```bash
git add python/lsst/images/cli python/lsst/images/tests/extract_legacy_test_data.py tests/test_cli.py
git commit -m "Register minify and extract-test-data; make extract imports lazy (DM-55131)"
```

---

## Task 12: News fragment + docs

**Files:**
- Create: `doc/changes/DM-55131.feature.md`
- Test: full suite green.

- [ ] **Step 1: Write the news fragment**

Create `doc/changes/DM-55131.feature.md`:

```markdown
Added the ``lsst-images-admin`` command-line tool with ``convert`` (legacy FITS to new format), ``inspect`` (schema URL and format version), ``minify``, and ``extract-test-data`` subcommands.
Added ``lsst.images.serialization.backend_for_path`` and ``InputArchive.get_basic_info`` as public APIs for resolving a backend by file suffix and reading basic archive information.
```

- [ ] **Step 2: Run the full test suite**

Run: `PYTHONPATH=./python ~/pyenv/bin/python -m pytest tests/test_serialization_basic_info.py tests/test_serialization_backends.py tests/test_cli.py -v`
Expected: PASS (with environment-gated tests skipping as appropriate).

- [ ] **Step 3: Verify the entry point installs**

Run: `~/pyenv/bin/python -m pip install -e . >/dev/null && lsst-images-admin --help`
Expected: the help text lists `convert`, `inspect`, `minify`, `extract-test-data`. (If editable install is undesirable in this environment, instead run `PYTHONPATH=./python ~/pyenv/bin/python -m lsst.images.cli --help` after adding an `if __name__ == "__main__": main()` guard to `_main.py` — optional.)

- [ ] **Step 4: Lint full changed set**

Run: `~/pyenv/bin/python -m ruff check python/lsst/images tests && ~/pyenv/bin/python -m ruff format --check python/lsst/images/cli python/lsst/images/serialization`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add doc/changes/DM-55131.feature.md
git commit -m "Add news fragment for lsst-images-admin CLI (DM-55131)"
```

---

## Self-review notes (resolved)

- **Spec coverage:** R1 → Task 5; R2 (model + base) → Task 1; R2 per backend → Tasks 2-4; minify refactor → Task 6; CLI skeleton + entry point + `click` dep → Task 7; `inspect` → Task 8; `convert` (FITS-only input, output-by-extension, overwrite, detection, visit image) → Task 9; `convert` cell coadd (skymap/butler) → Task 10; `extract-test-data` name kept + `minify` registration + lazy heavy imports → Task 11; testing across tasks; future-work items are explicitly out of scope.
- **Type consistency:** `ArchiveInfo(schema_url, schema_name, schema_version, format_version)` and `ArchiveInfo.from_schema_url(schema_url, *, format_version)` used identically in Tasks 1-4, 6, 8-10; `Backend(name, read, write, input_archive)` used identically in Tasks 5, 6, 8-10; `detect_legacy_type` and `_read_legacy` signatures consistent across Tasks 9-10.
- **Open detail now closed:** the legacy discriminator is the `HIERARCH LSST BUTLER DATASETTYPE` header — ending in `visit_image` → `VisitImage`, containing `coadd` → `CellCoadd`. Verified that the `visit_image`/`preliminary_visit_image` fixtures carry it; the `deep_coadd_cell_predetection.fits` fixture does **not** carry the header, so the cell-coadd `convert` tests pass `--type cell_coadd` explicitly and detection unit tests build synthetic headers.
- **Known follow-ups flagged inline (not blockers):** NDF `FORMAT_VERSION` node access (Task 4 Step 3 note) is validated by the gated NDF test. The `--butler` path requires `--collection`; only the `--skymap` pickle path is covered by automated tests (the `--butler` path needs a populated repo).
