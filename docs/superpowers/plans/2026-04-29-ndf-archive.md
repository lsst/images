# NDF Archive Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `python/lsst/images/ndf/` subpackage that reads and writes HDF5 files using the Starlink HDS-on-HDF5 mapping (so files are interpretable by Starlink/KAPPA), supporting `Image`, `MaskedImage`, and `VisitImage`.

**Architecture:** Two-layer subpackage. `_hds.py` knows only the HDS-on-HDF5 conventions (h5py groups + `HDSTYPE`/`HDSNDIMS` attrs, dimension reversal, `_CHAR*N` storage). `_output_archive.py`/`_input_archive.py` know NDF semantics and route the `OutputArchive`/`InputArchive` abstract API onto HDS structures, with non-image content hoisted into `MORE/LSST/*` JSON. Sibling to existing `fits/` and `json/` subpackages — same `write()`/`read()` and `Formatter` shape.

**Tech Stack:** Python 3.12+, `h5py` (new dep), `numpy`, `pydantic 2`, `astropy`, `starlink-pyast`, `lsst-resources`, `lsst-daf-butler` (formatter only).

**Reference spec:** `docs/superpowers/specs/2026-04-29-ndf-archive-design.md` — read it before starting.

---

## File Structure

**Created:**

- `python/lsst/images/ndf/__init__.py` — module docstring, re-exports
- `python/lsst/images/ndf/_hds.py` — HDS-on-HDF5 helpers (private, format-only)
- `python/lsst/images/ndf/_common.py` — `NdfPointerModel`, small shared helpers
- `python/lsst/images/ndf/_output_archive.py` — `NdfOutputArchive` + `write()`
- `python/lsst/images/ndf/_input_archive.py` — `NdfInputArchive` + `read()`
- `python/lsst/images/ndf/formatters.py` — daf_butler formatters
- `tests/test_ndf_hds.py` — unit tests for `_hds.py`
- `tests/test_ndf_layout.py` — on-disk layout sanity tests
- `tests/test_ndf_starlink_ingest.py` — read-only ingest of `example.sdf`
- `tests/data/example.sdf` — the existing root-level SDF moved here

**Modified:**

- `requirements.txt` — add `h5py`
- `pyproject.toml` — add `h5py` to `dependencies`
- `python/lsst/images/tests/_roundtrip.py` — add `RoundtripNdf`
- `python/lsst/images/tests/__init__.py` — re-export `RoundtripNdf`
- `tests/test_image.py` — add NDF round-trip cases
- `tests/test_masked_image.py` — add NDF round-trip cases (compatible + incompatible mask)
- `tests/test_visit_image.py` — add NDF round-trip cases

---

## Phase 1 — `_hds.py` foundation

### Task 1: Add `h5py` dependency and empty subpackage skeleton

**Files:**
- Modify: `requirements.txt`
- Modify: `pyproject.toml:30-40` (the `dependencies` list)
- Create: `python/lsst/images/ndf/__init__.py`
- Create: `python/lsst/images/ndf/_hds.py`

- [ ] **Step 1: Confirm `h5py` is not already a dep**

Run: `grep -n h5py requirements.txt pyproject.toml`
Expected: no matches (or only optional-dep matches; if present already, skip pyproject/requirements edits).

- [ ] **Step 2: Add `h5py` to `requirements.txt`**

Add `h5py` on its own line (alphabetical position). Run `grep h5py requirements.txt` and expect a single hit.

- [ ] **Step 3: Add `h5py` to `pyproject.toml` dependencies**

In the `[project] -> dependencies` array, add `"h5py"` (alphabetical with other entries). Mirror the version-spec style used by neighbours (e.g. `"h5py >= 3.10"` if neighbours have version pins; otherwise unpinned).

- [ ] **Step 4: Create empty subpackage skeleton**

Create `python/lsst/images/ndf/__init__.py`:

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


"""Archive implementations that write HDS-on-HDF5 files compatible with the
Starlink NDF data model.

Files written by this archive are valid NDF files readable by Starlink tools
(KAPPA, ``hdstrace``, etc.). The HDS-on-HDF5 format is described in
arxiv:1502.04029; the NDF data model in arxiv:1410.7513.
"""
```

Create `python/lsst/images/ndf/_hds.py` with the standard license header, a module docstring (`"""HDS-on-HDF5 read/write helpers."""`), and `from __future__ import annotations`.

- [ ] **Step 5: Verify the package imports**

Run: `python -c "import lsst.images.ndf"`
Expected: success, no output.

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pyproject.toml python/lsst/images/ndf/
git commit -m "Add h5py dep and empty ndf subpackage skeleton (DM-54817)"
```

---

### Task 2: `_hds.py` — primitive type system + array round-trip

**Files:**
- Modify: `python/lsst/images/ndf/_hds.py`
- Create: `tests/test_ndf_hds.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ndf_hds.py`:

```python
# (license header)
from __future__ import annotations

import tempfile
import unittest

import h5py
import numpy as np

from lsst.images.ndf import _hds


class HdsPrimitiveTestCase(unittest.TestCase):
    def test_real_array_round_trip(self):
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                _hds.write_array(f, "DATA", data)
            with h5py.File(tmp.name, "r") as f:
                ds = f["DATA"]
                self.assertEqual(ds.attrs["HDSTYPE"], "_REAL")
                self.assertEqual(ds.attrs["HDSNDIMS"], 2)
                self.assertEqual(ds.attrs["HDS_DATASET_IS_DEFINED"], True)
                np.testing.assert_array_equal(_hds.read_array(ds), data)

    def test_double_array_round_trip(self):
        data = np.linspace(0, 1, 6, dtype=np.float64).reshape(2, 3)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                _hds.write_array(f, "DATA", data)
            with h5py.File(tmp.name, "r") as f:
                self.assertEqual(f["DATA"].attrs["HDSTYPE"], "_DOUBLE")
                np.testing.assert_array_equal(_hds.read_array(f["DATA"]), data)

    def test_ubyte_and_integer(self):
        data_u = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
        data_i = np.array([10, 20, 30], dtype=np.int32)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                _hds.write_array(f, "Q", data_u)
                _hds.write_array(f, "I", data_i)
            with h5py.File(tmp.name, "r") as f:
                self.assertEqual(f["Q"].attrs["HDSTYPE"], "_UBYTE")
                self.assertEqual(f["I"].attrs["HDSTYPE"], "_INTEGER")
                np.testing.assert_array_equal(_hds.read_array(f["Q"]), data_u)
                np.testing.assert_array_equal(_hds.read_array(f["I"]), data_i)

    def test_unsupported_dtype_raises(self):
        data = np.array([1.0], dtype=np.complex128)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp, h5py.File(tmp.name, "w") as f:
            with self.assertRaises(NotImplementedError):
                _hds.write_array(f, "X", data)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_ndf_hds.py -v`
Expected: FAIL with `AttributeError` on `_hds.write_array` / `_hds.read_array`.

- [ ] **Step 3: Implement the type system + write/read primitives**

Append to `python/lsst/images/ndf/_hds.py`:

```python
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

import h5py
import numpy as np

__all__ = (
    "create_structure",
    "open_structure",
    "iter_children",
    "write_array",
    "read_array",
    "write_char_array",
    "read_char_array",
    "HDS_TO_NUMPY",
    "NUMPY_TO_HDS",
)


HDS_TO_NUMPY: dict[str, np.dtype] = {
    "_REAL": np.dtype(np.float32),
    "_DOUBLE": np.dtype(np.float64),
    "_UBYTE": np.dtype(np.uint8),
    "_INTEGER": np.dtype(np.int32),
    "_WORD": np.dtype(np.int16),
}

NUMPY_TO_HDS: dict[np.dtype, str] = {
    np.dtype(np.float32): "_REAL",
    np.dtype(np.float64): "_DOUBLE",
    np.dtype(np.uint8): "_UBYTE",
    np.dtype(np.int32): "_INTEGER",
}


def write_array(
    parent: h5py.Group,
    name: str,
    data: np.ndarray,
    *,
    hdstype: str | None = None,
    compression: str | None = None,
) -> h5py.Dataset:
    """Write a numpy C-order array as an HDS primitive.

    The HDF5 dataset has the array's natural shape (C-order). Combined with
    HDF5's native byte ordering, this matches the Fortran-on-disk layout
    required by HDS for an NDF whose Fortran-order shape is the reverse of
    ``data.shape``.
    """
    if hdstype is None:
        try:
            hdstype = NUMPY_TO_HDS[data.dtype]
        except KeyError:
            raise NotImplementedError(f"No HDS write support for dtype {data.dtype!r}.") from None
    kwargs: dict = {}
    if compression is not None:
        kwargs["compression"] = compression
    ds = parent.create_dataset(name, data=data, **kwargs)
    ds.attrs["HDSTYPE"] = hdstype
    ds.attrs["HDSNDIMS"] = data.ndim
    ds.attrs["HDS_DATASET_IS_DEFINED"] = True
    return ds


def read_array(dataset: h5py.Dataset) -> np.ndarray:
    """Read an HDS primitive into a C-order numpy array.

    Validates ``HDSTYPE`` is in the supported set and that ``HDSNDIMS``
    matches the dataset's HDF5 ndim.
    """
    hdstype = dataset.attrs.get("HDSTYPE")
    if not isinstance(hdstype, (bytes, str)):
        raise ValueError(f"Dataset {dataset.name!r} has no HDSTYPE attribute.")
    if isinstance(hdstype, bytes):
        hdstype = hdstype.decode("ascii")
    if hdstype.startswith("_CHAR"):
        raise ValueError(f"Use read_char_array for _CHAR primitives at {dataset.name!r}.")
    if hdstype not in HDS_TO_NUMPY:
        raise NotImplementedError(f"HDS type {hdstype!r} not supported for read.")
    expected_dtype = HDS_TO_NUMPY[hdstype]
    if dataset.dtype != expected_dtype:
        raise ValueError(
            f"Dataset {dataset.name!r} has HDF5 dtype {dataset.dtype} "
            f"but HDSTYPE {hdstype!r} expects {expected_dtype}."
        )
    return dataset[()]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_ndf_hds.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/ndf/_hds.py tests/test_ndf_hds.py
git commit -m "Add HDS primitive write/read helpers (DM-54817)"
```

---

### Task 3: `_hds.py` — `_CHAR*N` arrays

**Files:**
- Modify: `python/lsst/images/ndf/_hds.py`
- Modify: `tests/test_ndf_hds.py`

- [ ] **Step 1: Write the failing test**

Append to `HdsPrimitiveTestCase` in `tests/test_ndf_hds.py`:

```python
    def test_char_array_round_trip(self):
        lines = ["Begin FrameSet", "Nframe = 5", "End FrameSet"]
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                _hds.write_char_array(f, "DATA", lines, width=80)
            with h5py.File(tmp.name, "r") as f:
                ds = f["DATA"]
                self.assertEqual(ds.attrs["HDSTYPE"], "_CHAR*80")
                self.assertEqual(ds.attrs["HDSNDIMS"], 1)
                self.assertEqual(_hds.read_char_array(ds), lines)

    def test_char_array_pads_and_strips(self):
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                _hds.write_char_array(f, "X", ["short"], width=80)
            with h5py.File(tmp.name, "r") as f:
                # Raw data should be space-padded to 80 characters.
                self.assertEqual(f["X"][0], b"short" + b" " * 75)
                # read_char_array strips trailing spaces.
                self.assertEqual(_hds.read_char_array(f["X"]), ["short"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_ndf_hds.py::HdsPrimitiveTestCase::test_char_array_round_trip -v`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement `_CHAR*N` helpers**

Append to `python/lsst/images/ndf/_hds.py`:

```python
def write_char_array(
    parent: h5py.Group,
    name: str,
    lines: Sequence[str],
    *,
    width: int = 80,
) -> h5py.Dataset:
    """Write a sequence of strings as a 1D HDS ``_CHAR*N`` primitive.

    Each string is padded to ``width`` with trailing spaces (HDS
    convention) and truncated if longer. Reader returns strings with
    trailing spaces stripped.
    """
    encoded = np.array(
        [line.encode("ascii", errors="replace").ljust(width)[:width] for line in lines],
        dtype=f"|S{width}",
    )
    ds = parent.create_dataset(name, data=encoded)
    ds.attrs["HDSTYPE"] = f"_CHAR*{width}"
    ds.attrs["HDSNDIMS"] = 1
    ds.attrs["HDS_DATASET_IS_DEFINED"] = True
    return ds


def read_char_array(dataset: h5py.Dataset) -> list[str]:
    """Read an HDS ``_CHAR*N`` 1D primitive as a list of stripped strings."""
    hdstype = dataset.attrs.get("HDSTYPE")
    if isinstance(hdstype, bytes):
        hdstype = hdstype.decode("ascii")
    if not isinstance(hdstype, str) or not hdstype.startswith("_CHAR*"):
        raise ValueError(f"Dataset {dataset.name!r} is not _CHAR*N (got HDSTYPE={hdstype!r}).")
    raw = dataset[()]
    return [item.decode("ascii").rstrip(" ") for item in raw]
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_ndf_hds.py -v`
Expected: PASS (all four tests).

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/ndf/_hds.py tests/test_ndf_hds.py
git commit -m "Add HDS _CHAR*N read/write helpers (DM-54817)"
```

---

### Task 4: `_hds.py` — structure groups

**Files:**
- Modify: `python/lsst/images/ndf/_hds.py`
- Modify: `tests/test_ndf_hds.py`

- [ ] **Step 1: Write the failing test**

Append a new test class to `tests/test_ndf_hds.py`:

```python
class HdsStructureTestCase(unittest.TestCase):
    def test_create_open_structure(self):
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                ndf = _hds.create_structure(f, "ROOT", "NDF")
                _hds.create_structure(ndf, "DATA_ARRAY", "ARRAY")
            with h5py.File(tmp.name, "r") as f:
                root, root_type = _hds.open_structure(f, "ROOT")
                self.assertEqual(root_type, "NDF")
                child_names = sorted(name for name, _ in _hds.iter_children(root))
                self.assertEqual(child_names, ["DATA_ARRAY"])
                _, child_type = _hds.open_structure(root, "DATA_ARRAY")
                self.assertEqual(child_type, "ARRAY")

    def test_open_structure_missing_hdstype_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                f.create_group("BAD")
            with h5py.File(tmp.name, "r") as f:
                with self.assertRaises(ValueError):
                    _hds.open_structure(f, "BAD")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_ndf_hds.py::HdsStructureTestCase -v`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement structure helpers**

Append to `python/lsst/images/ndf/_hds.py`:

```python
def create_structure(parent: h5py.Group, name: str, hdstype: str) -> h5py.Group:
    """Create a named HDS structure (h5py group with HDSTYPE attribute)."""
    group = parent.create_group(name)
    group.attrs["HDSTYPE"] = hdstype
    return group


def open_structure(parent: h5py.Group, name: str) -> tuple[h5py.Group, str]:
    """Open a child structure by name. Returns (group, hdstype). Raises
    ``ValueError`` if the child is not a group with an ``HDSTYPE`` attribute.
    """
    obj = parent[name]
    if not isinstance(obj, h5py.Group):
        raise ValueError(f"{parent.name}/{name} is a dataset, not a structure.")
    hdstype = obj.attrs.get("HDSTYPE")
    if isinstance(hdstype, bytes):
        hdstype = hdstype.decode("ascii")
    if not isinstance(hdstype, str):
        raise ValueError(f"Group {obj.name!r} has no HDSTYPE attribute.")
    return obj, hdstype


def iter_children(group: h5py.Group) -> Iterator[tuple[str, h5py.Group | h5py.Dataset]]:
    """Iterate over a structure's direct children as ``(name, child)`` pairs."""
    for name, child in group.items():
        yield name, child
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_ndf_hds.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/ndf/_hds.py tests/test_ndf_hds.py
git commit -m "Add HDS structure helpers (DM-54817)"
```

---

### Task 5: `_hds.py` — sanity test against `example.sdf`

**Files:**
- Create: `tests/data/`
- Move: `example.sdf` → `tests/data/example.sdf`
- Modify: `tests/test_ndf_hds.py`

- [ ] **Step 1: Move `example.sdf` under `tests/data/`**

```bash
mkdir -p tests/data
git mv example.sdf tests/data/example.sdf
```

(Use `git mv` even though it's untracked — it'll do a `mv` and stage the new path.) If `git mv` complains because the file is untracked, fall back to `mv example.sdf tests/data/example.sdf`.

- [ ] **Step 2: Write the sanity test**

Append to `tests/test_ndf_hds.py`:

```python
import os


class HdsExampleSdfTestCase(unittest.TestCase):
    """Validate _hds against a Starlink-generated NDF.

    The example file was produced by CCDPACK and contains a single
    top-level NDF structure named BIAS1 with DATA_ARRAY, WCS, and MORE.FITS
    components.
    """

    EXAMPLE = os.path.join(os.path.dirname(__file__), "data", "example.sdf")

    def test_top_level_structure(self):
        with h5py.File(self.EXAMPLE, "r") as f:
            # The example wraps a single NDF in a top-level container; verify
            # we can iterate the root and find the BIAS1 NDF.
            children = dict(_hds.iter_children(f))
            self.assertIn("BIAS1", children)
            bias1, hdstype = _hds.open_structure(f, "BIAS1")
            self.assertEqual(hdstype, "NDF")

    def test_data_array_present(self):
        with h5py.File(self.EXAMPLE, "r") as f:
            bias1, _ = _hds.open_structure(f, "BIAS1")
            data_array, _ = _hds.open_structure(bias1, "DATA_ARRAY")
            data = data_array["DATA"]
            self.assertIn(data.attrs["HDSTYPE"], ("_REAL", "_DOUBLE", "_INTEGER"))
            self.assertEqual(data.attrs["HDSNDIMS"], 2)
            self.assertEqual(_hds.read_array(data).shape, data.shape)

    def test_wcs_present(self):
        with h5py.File(self.EXAMPLE, "r") as f:
            bias1, _ = _hds.open_structure(f, "BIAS1")
            wcs, hdstype = _hds.open_structure(bias1, "WCS")
            self.assertEqual(hdstype, "WCS")
            lines = _hds.read_char_array(wcs["DATA"])
            self.assertTrue(any(line.startswith("Begin FrameSet") for line in lines))
            self.assertTrue(any(line.startswith("End FrameSet") for line in lines))

    def test_more_fits_present(self):
        with h5py.File(self.EXAMPLE, "r") as f:
            bias1, _ = _hds.open_structure(f, "BIAS1")
            more, _ = _hds.open_structure(bias1, "MORE")
            cards = _hds.read_char_array(more["FITS"])
            # Sample a few cards we know are in the example.
            self.assertTrue(any(c.startswith("NAXIS   =") for c in cards))
            self.assertTrue(any(c.startswith("OBSTYPE = 'BIAS'") for c in cards))
```

- [ ] **Step 3: Run the test**

Run: `pytest tests/test_ndf_hds.py::HdsExampleSdfTestCase -v`
Expected: PASS. If a test fails because the example's actual layout differs (e.g. top-level wraps the NDF in an extra container, or `DATA_ARRAY` is wrapped differently), adjust the test to match the actual layout — this is the source of truth and reveals real layout details we need to handle. The point of this task is to *anchor* the convention to a real Starlink file.

- [ ] **Step 4: Commit**

```bash
git add tests/data/example.sdf tests/test_ndf_hds.py
git commit -m "Move example.sdf to tests/data/ and add sanity tests (DM-54817)"
```

---

## Phase 2 — Output archive

### Task 6: `_common.py` — `NdfPointerModel`

**Files:**
- Create: `python/lsst/images/ndf/_common.py`

- [ ] **Step 1: Write the failing test**

Add a new test file `tests/test_ndf_common.py`:

```python
# (license header)
from __future__ import annotations

import unittest

from lsst.images.ndf._common import NdfPointerModel


class NdfPointerModelTestCase(unittest.TestCase):
    def test_round_trips_through_json(self):
        original = NdfPointerModel(ref="/MORE/LSST/PSF")
        json_bytes = original.model_dump_json().encode()
        recovered = NdfPointerModel.model_validate_json(json_bytes)
        self.assertEqual(recovered, original)

    def test_join_path_to_hdf5(self):
        # Helper for routing JSON Pointer paths into MORE/LSST/<UPPER_PATH>.
        from lsst.images.ndf._common import json_pointer_to_hdf5_path

        self.assertEqual(json_pointer_to_hdf5_path(""), "/MORE/LSST/JSON")
        self.assertEqual(json_pointer_to_hdf5_path("/psf"), "/MORE/LSST/PSF")
        self.assertEqual(
            json_pointer_to_hdf5_path("/psf/coefficients"), "/MORE/LSST/PSF_COEFFICIENTS"
        )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_ndf_common.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `_common.py`**

Create `python/lsst/images/ndf/_common.py`:

```python
# (license header)
from __future__ import annotations

__all__ = ("NdfPointerModel", "json_pointer_to_hdf5_path")

import pydantic


class NdfPointerModel(pydantic.BaseModel, serialize_by_alias=True):
    """Reference to an NDF-archive sub-tree by HDF5 path.

    Used by :class:`NdfOutputArchive`/:class:`NdfInputArchive` to point to
    sub-trees that have been hoisted out of the main JSON tree into separate
    HDS components.
    """

    ref: str = pydantic.Field(alias="$ref")
    """HDF5 absolute path (e.g. ``/MORE/LSST/PSF``)."""

    model_config = pydantic.ConfigDict(populate_by_name=True)


def json_pointer_to_hdf5_path(json_pointer: str) -> str:
    """Translate an RFC-6901 JSON Pointer to the HDF5 path used by the
    NDF archive for the corresponding hoisted sub-tree.

    The empty pointer (root) maps to the main JSON tree at
    ``/MORE/LSST/JSON``. Any non-empty pointer is uppercased and its
    ``/`` separators replaced with ``_`` to form a single component
    name under ``/MORE/LSST/``. Mirrors the FITS archive's ``EXTNAME``
    convention.
    """
    if not json_pointer:
        return "/MORE/LSST/JSON"
    flattened = json_pointer.lstrip("/").upper().replace("/", "_")
    return f"/MORE/LSST/{flattened}"
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_ndf_common.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/ndf/_common.py tests/test_ndf_common.py
git commit -m "Add NdfPointerModel and JSON-Pointer-to-HDF5-path helper (DM-54817)"
```

---

### Task 7: `NdfOutputArchive` skeleton + `serialize_direct`

**Files:**
- Create: `python/lsst/images/ndf/_output_archive.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ndf_common.py` (or create `tests/test_ndf_output_archive.py` — either works; the rest of the plan assumes `tests/test_ndf_output_archive.py`):

```python
# (license header)
from __future__ import annotations

import tempfile
import unittest

import h5py
import numpy as np
import pydantic

from lsst.images.ndf._output_archive import NdfOutputArchive


class TinyTree(pydantic.BaseModel):
    name: str


class NdfOutputArchiveBasicsTestCase(unittest.TestCase):
    def test_serialize_direct_passes_through(self):
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                tree = arch.serialize_direct(
                    None, lambda nested: TinyTree(name="hello")
                )
                self.assertEqual(tree.name, "hello")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_ndf_output_archive.py::NdfOutputArchiveBasicsTestCase -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the skeleton**

Create `python/lsst/images/ndf/_output_archive.py`:

```python
# (license header)
from __future__ import annotations

__all__ = ("NdfOutputArchive", "write")

from collections.abc import Callable, Hashable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

import astropy.table
import astropy.units
import h5py
import numpy as np
import pydantic

from .._transforms import FrameSet
from ..fits._common import FitsOpaqueMetadata, ExtensionKey
from ..serialization import (
    ArchiveTree,
    ArrayReferenceModel,
    ButlerInfo,
    InlineArrayModel,
    MetadataValue,
    NestedOutputArchive,
    NumberType,
    OutputArchive,
    TableColumnModel,
    TableModel,
    no_header_updates,
)
from . import _hds
from ._common import NdfPointerModel, json_pointer_to_hdf5_path

if TYPE_CHECKING:
    import astropy.io.fits


class NdfOutputArchive(OutputArchive[NdfPointerModel]):
    """An :class:`~lsst.images.serialization.OutputArchive` implementation
    that writes HDS-on-HDF5 files compatible with the Starlink NDF data model.

    Parameters
    ----------
    file
        An open ``h5py.File`` opened in a writable mode. The archive does
        not close the file; the caller is responsible for that.
    compression_options
        Optional dict passed through to ``h5py.create_dataset`` for image
        arrays (e.g. ``{"compression": "gzip", "compression_opts": 4}``).
        Ignored for ``_CHAR*N`` and small arrays.
    opaque_metadata
        Optional :class:`FitsOpaqueMetadata`; if its primary-HDU header is
        non-empty its cards are written to ``/MORE/FITS``.
    """

    def __init__(
        self,
        file: h5py.File,
        compression_options: Mapping[str, Any] | None = None,
        opaque_metadata: FitsOpaqueMetadata | None = None,
    ) -> None:
        self._file = file
        self._compression_options = dict(compression_options) if compression_options else {}
        self._opaque_metadata = opaque_metadata if opaque_metadata is not None else FitsOpaqueMetadata()
        self._frame_sets: list[tuple[FrameSet, NdfPointerModel]] = []
        self._pointers: dict[Hashable, NdfPointerModel] = {}
        # Initialise the top-level NDF structure if not already present.
        if "HDSTYPE" not in self._file.attrs:
            self._file.attrs["HDSTYPE"] = "NDF"

    def serialize_direct[T: pydantic.BaseModel](
        self, name: str | None, serializer: Callable[[OutputArchive[NdfPointerModel]], T]
    ) -> T:
        if name is None:
            return serializer(self)
        nested = NestedOutputArchive[NdfPointerModel](name, self)
        return serializer(nested)

    def serialize_pointer[T: ArchiveTree](
        self, name: str, serializer: Callable[[OutputArchive[NdfPointerModel]], T], key: Hashable
    ) -> NdfPointerModel:
        # Implemented in Task 9.
        raise NotImplementedError

    def serialize_frame_set[T: ArchiveTree](
        self, name: str, frame_set: FrameSet, serializer: Callable[[OutputArchive], T], key: Hashable
    ) -> NdfPointerModel:
        # Implemented in Task 9.
        raise NotImplementedError

    def iter_frame_sets(self) -> Iterator[tuple[FrameSet, NdfPointerModel]]:
        return iter(self._frame_sets)

    def add_array(
        self,
        array: np.ndarray,
        *,
        name: str | None = None,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> ArrayReferenceModel | InlineArrayModel:
        # Implemented in Task 8.
        raise NotImplementedError

    def add_table(
        self,
        table: astropy.table.Table,
        *,
        name: str | None = None,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> TableModel:
        # Implemented in Task 10.
        raise NotImplementedError

    def add_structured_array(
        self,
        array: np.ndarray,
        *,
        name: str | None = None,
        units: Mapping[str, astropy.units.Unit] | None = None,
        descriptions: Mapping[str, str] | None = None,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> TableModel:
        # Implemented in Task 10.
        raise NotImplementedError
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_ndf_output_archive.py::NdfOutputArchiveBasicsTestCase -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/ndf/_output_archive.py tests/test_ndf_output_archive.py
git commit -m "Add NdfOutputArchive skeleton with serialize_direct (DM-54817)"
```

---

### Task 8: `add_array` — recognised-component routing + hoisting

**Files:**
- Modify: `python/lsst/images/ndf/_output_archive.py`
- Modify: `tests/test_ndf_output_archive.py`

- [ ] **Step 1: Confirm the routing names by tracing existing serialize() calls**

Run: `grep -n "add_array\|serialize_frame_set" python/lsst/images/_image.py python/lsst/images/_masked_image.py python/lsst/images/_visit_image.py python/lsst/images/_transforms/_projection.py 2>&1 | head -30`

Record the actual `name=` string each top-level type uses for its image array, mask, variance, and projection. The plan below assumes `"image"`, `"mask"`, `"variance"`, and `"projection"` — substitute whatever the code actually uses.

- [ ] **Step 2: Write failing tests for routing**

Append to `tests/test_ndf_output_archive.py`:

```python
class NdfOutputArchiveAddArrayTestCase(unittest.TestCase):
    def test_top_level_image_routes_to_data_array(self):
        data = np.arange(20, dtype=np.float32).reshape(4, 5)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                ref = arch.add_array(data, name="image")
                self.assertEqual(ref.source, "ndf:/DATA_ARRAY/DATA")
            with h5py.File(tmp.name, "r") as f:
                ds = f["/DATA_ARRAY/DATA"]
                self.assertEqual(ds.attrs["HDSTYPE"], "_REAL")
                np.testing.assert_array_equal(ds[()], data)
                self.assertEqual(f["/DATA_ARRAY"].attrs["HDSTYPE"], "ARRAY")

    def test_top_level_variance_routes_to_variance(self):
        data = np.full((3, 3), 0.5, dtype=np.float64)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                ref = arch.add_array(data, name="variance")
                self.assertEqual(ref.source, "ndf:/VARIANCE/DATA")
            with h5py.File(tmp.name, "r") as f:
                self.assertEqual(f["/VARIANCE"].attrs["HDSTYPE"], "ARRAY")
                self.assertEqual(f["/VARIANCE/DATA"].attrs["HDSTYPE"], "_DOUBLE")

    def test_top_level_compatible_mask_routes_to_quality(self):
        data = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                ref = arch.add_array(data, name="mask")
                self.assertEqual(ref.source, "ndf:/QUALITY/QUALITY")
            with h5py.File(tmp.name, "r") as f:
                self.assertEqual(f["/QUALITY"].attrs["HDSTYPE"], "QUALITY")
                self.assertEqual(f["/QUALITY/QUALITY"].attrs["HDSTYPE"], "_UBYTE")
                self.assertEqual(f["/QUALITY/BADBITS"][()], 0xFF)

    def test_top_level_incompatible_mask_routes_to_more_lsst(self):
        # 3D mask array (multi-plane uint8) doesn't fit NDF QUALITY.
        data = np.zeros((3, 4, 2), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                ref = arch.add_array(data, name="mask")
                self.assertEqual(ref.source, "ndf:/MORE/LSST/MASK/DATA")

    def test_nested_array_hoists(self):
        data = np.array([[1.0, 2.0]], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                ref = arch.add_array(data, name="psf/coefficients")
                self.assertEqual(ref.source, "ndf:/MORE/LSST/PSF_COEFFICIENTS")
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/test_ndf_output_archive.py::NdfOutputArchiveAddArrayTestCase -v`
Expected: FAIL with `NotImplementedError` (the stub is in place).

- [ ] **Step 4: Implement `add_array`**

Replace the `add_array` stub in `_output_archive.py`:

```python
    _COMPATIBLE_MASK_DTYPES = (np.dtype(np.uint8),)

    def add_array(
        self,
        array: np.ndarray,
        *,
        name: str | None = None,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> ArrayReferenceModel:
        # Recognised top-level names (no "/" in the path) go to standard NDF
        # locations. Nested or non-canonical names hoist under MORE/LSST.
        path: str
        ensure_array_wrapper: str | None = None
        if name == "image":
            self._ensure_array_structure("/DATA_ARRAY")
            path = "/DATA_ARRAY/DATA"
            self._write_origin_for_array("/DATA_ARRAY", array)
        elif name == "variance":
            self._ensure_array_structure("/VARIANCE")
            path = "/VARIANCE/DATA"
            self._write_origin_for_array("/VARIANCE", array)
        elif name == "mask":
            if array.ndim == 2 and array.dtype in self._COMPATIBLE_MASK_DTYPES:
                self._ensure_quality_structure()
                path = "/QUALITY/QUALITY"
            else:
                self._ensure_struct("/MORE/LSST/MASK", "STRUCT")
                path = "/MORE/LSST/MASK/DATA"
        else:
            # Anything else hoists.
            assert name is not None, "Anonymous arrays are not supported in the NDF archive."
            path = json_pointer_to_hdf5_path(f"/{name}" if not name.startswith("/") else name)
        parent_path, leaf = path.rsplit("/", 1)
        parent = self._ensure_path(parent_path)
        _hds.write_array(parent, leaf, array, **self._compression_options)
        return ArrayReferenceModel(source=f"ndf:{path}", datatype=NumberType.from_numpy(array.dtype))

    def _ensure_path(self, path: str) -> h5py.Group:
        """Walk/create groups for an HDF5 absolute path."""
        if path in ("", "/"):
            return self._file
        parts = path.lstrip("/").split("/")
        cursor: h5py.Group = self._file
        for part in parts:
            if part not in cursor:
                cursor = _hds.create_structure(cursor, part, "EXT")
            else:
                cursor = cursor[part]
        return cursor

    def _ensure_struct(self, path: str, hdstype: str) -> h5py.Group:
        if path in self._file:
            return self._file[path]
        parent_path, leaf = path.rsplit("/", 1)
        parent = self._ensure_path(parent_path or "/")
        return _hds.create_structure(parent, leaf, hdstype)

    def _ensure_array_structure(self, path: str) -> h5py.Group:
        return self._ensure_struct(path, "ARRAY")

    def _ensure_quality_structure(self) -> h5py.Group:
        if "/QUALITY" in self._file:
            return self._file["/QUALITY"]
        group = _hds.create_structure(self._file, "QUALITY", "QUALITY")
        # BADBITS: scalar _UBYTE, default 0xFF (treat all defined bits as bad).
        _hds.write_array(group, "BADBITS", np.uint8(0xFF))
        return group

    def _write_origin_for_array(self, struct_path: str, array: np.ndarray) -> None:
        """Write a placeholder ORIGIN of zeros; the caller (write()) overwrites
        with bbox-derived values via :meth:`set_array_origin` once the bbox is
        known. v1 just writes zeros if no origin is supplied later.
        """
        struct = self._file[struct_path]
        if "ORIGIN" not in struct:
            _hds.write_array(struct, "ORIGIN", np.zeros(array.ndim, dtype=np.int32))

    def set_array_origin(self, struct_path: str, origin_xy: tuple[int, ...]) -> None:
        """Overwrite the ORIGIN of an ARRAY structure with the supplied
        Fortran-axis-order origin (e.g. ``(x_min, y_min)``).
        """
        struct = self._file[struct_path]
        if "ORIGIN" in struct:
            del struct["ORIGIN"]
        _hds.write_array(struct, "ORIGIN", np.asarray(origin_xy, dtype=np.int32))
```

(Note: bbox-derived ORIGIN handling is wired up in Task 11's `write()` after the top-level archive tree is built.)

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_ndf_output_archive.py::NdfOutputArchiveAddArrayTestCase -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add python/lsst/images/ndf/_output_archive.py tests/test_ndf_output_archive.py
git commit -m "Add add_array routing for recognised top-level names (DM-54817)"
```

---

### Task 9: `serialize_pointer` and `serialize_frame_set`

**Files:**
- Modify: `python/lsst/images/ndf/_output_archive.py`
- Modify: `tests/test_ndf_output_archive.py`

- [ ] **Step 1: Write failing tests**

Append:

```python
class NdfOutputArchivePointerTestCase(unittest.TestCase):
    def test_serialize_pointer_writes_subtree_and_returns_pointer(self):
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                ptr = arch.serialize_pointer(
                    "psf",
                    lambda nested: TinyTree(name="gaussian"),
                    key=("psf", 1),
                )
                self.assertEqual(ptr.ref, "/MORE/LSST/PSF")
                # Same key returns the cached pointer without re-serializing.
                ptr2 = arch.serialize_pointer(
                    "psf",
                    lambda nested: TinyTree(name="OTHER"),
                    key=("psf", 1),
                )
                self.assertEqual(ptr, ptr2)
            with h5py.File(tmp.name, "r") as f:
                lines = _hds.read_char_array(f["/MORE/LSST/PSF"])
                self.assertIn("hello".replace("hello", '"name":"gaussian"'), "".join(lines))

    def test_serialize_frame_set_records_for_iter(self):
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                # A real FrameSet would come from starlink-pyast; for the unit
                # test, any sentinel works because iter_frame_sets just records
                # what it was given.
                fake_frame_set = object()
                ptr = arch.serialize_frame_set(
                    "projection",
                    fake_frame_set,
                    lambda nested: TinyTree(name="proj"),
                    key=("frame_set", 1),
                )
                self.assertEqual(ptr.ref, "/WCS/DATA")
                recorded = list(arch.iter_frame_sets())
                self.assertEqual(len(recorded), 1)
                self.assertIs(recorded[0][0], fake_frame_set)
                self.assertEqual(recorded[0][1].ref, "/WCS/DATA")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ndf_output_archive.py::NdfOutputArchivePointerTestCase -v`
Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement `serialize_pointer` and `serialize_frame_set`**

Replace the two stubs in `_output_archive.py`:

```python
    def serialize_pointer[T: ArchiveTree](
        self, name: str, serializer: Callable[[OutputArchive[NdfPointerModel]], T], key: Hashable
    ) -> NdfPointerModel:
        if pointer := self._pointers.get(key):
            return pointer
        # Compute hoist destination (HDF5 path) and write the sub-tree there.
        path = self._hoist_path_for(name)
        nested = NestedOutputArchive[NdfPointerModel](name, self)
        sub_tree = serializer(nested)
        json_bytes = sub_tree.model_dump_json().encode()
        # Store as a 1D _CHAR*N array, splitting on linefeed if the JSON is
        # long enough that a single 80-char string would be too small.
        lines = self._wrap_json_for_storage(json_bytes.decode())
        parent_path, leaf = path.rsplit("/", 1)
        parent = self._ensure_path(parent_path)
        if leaf in parent:
            del parent[leaf]
        _hds.write_char_array(parent, leaf, lines, width=max(80, max(len(l) for l in lines)))
        pointer = NdfPointerModel(ref=path)
        self._pointers[key] = pointer
        return pointer

    def serialize_frame_set[T: ArchiveTree](
        self, name: str, frame_set: FrameSet, serializer: Callable[[OutputArchive], T], key: Hashable
    ) -> NdfPointerModel:
        # Top-level "projection" goes to /WCS/DATA; otherwise hoist like a
        # normal pointer.
        if name == "projection":
            path = "/WCS/DATA"
            self._ensure_struct("/WCS", "WCS")
        else:
            path = self._hoist_path_for(name)
        # Run the serializer to populate any nested arrays/etc., even though
        # for /WCS we discard the resulting Pydantic tree (the AST FrameSet
        # text dump replaces it on disk; the in-memory tree retains the
        # pointer).
        nested = NestedOutputArchive[NdfPointerModel](name, self)
        _ = serializer(nested)
        # Defer AST text-dump emission to the write() function, which has
        # access to starlink-pyast Channel and the actual FrameSet object.
        # Here we just record the (frame_set, pointer) pair.
        pointer = NdfPointerModel(ref=path)
        self._frame_sets.append((frame_set, pointer))
        self._pointers[key] = pointer
        return pointer

    def _hoist_path_for(self, name: str) -> str:
        # ``name`` arrives as a JSON Pointer relative to the root. Convert to
        # an absolute pointer first so json_pointer_to_hdf5_path handles it.
        if not name.startswith("/"):
            name = "/" + name
        return json_pointer_to_hdf5_path(name)

    @staticmethod
    def _wrap_json_for_storage(json_text: str, max_line_width: int = 80) -> list[str]:
        # Naive wrap: the simplest reader-compatible thing is to store the
        # whole JSON as a single fixed-width-padded element if it fits, else
        # break on raw bytes (HDS readers don't interpret line breaks).
        if len(json_text) <= max_line_width:
            return [json_text]
        return [
            json_text[i : i + max_line_width] for i in range(0, len(json_text), max_line_width)
        ]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_ndf_output_archive.py -v`
Expected: PASS for all output-archive tests.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/ndf/_output_archive.py tests/test_ndf_output_archive.py
git commit -m "Add serialize_pointer and serialize_frame_set with WCS routing (DM-54817)"
```

---

### Task 10: `add_table` and `add_structured_array` (hoisted JSON for v1)

**Files:**
- Modify: `python/lsst/images/ndf/_output_archive.py`
- Modify: `tests/test_ndf_output_archive.py`

- [ ] **Step 1: Write a failing test**

Append:

```python
class NdfOutputArchiveAddTableTestCase(unittest.TestCase):
    def test_add_table_returns_inline_table_model(self):
        import astropy.table

        t = astropy.table.Table({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                model = arch.add_table(t, name="some_table")
                self.assertEqual(len(model.columns), 2)
                # v1 stores tables inline in the JSON tree.
                self.assertIsInstance(model.columns[0].data, InlineArrayModel)
```

(Add `from lsst.images.serialization import InlineArrayModel` at the top of the test file.)

- [ ] **Step 2: Run the test, expect FAIL with NotImplementedError**

- [ ] **Step 3: Implement `add_table` / `add_structured_array`**

Reuse the JSON archive's strategy verbatim — both methods just produce inline Pydantic `TableModel`s for v1; binary-table hoisting is a follow-up:

```python
    def add_table(
        self,
        table: astropy.table.Table,
        *,
        name: str | None = None,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> TableModel:
        columns = TableColumnModel.from_table(table, inline=True)
        return TableModel(columns=columns, meta=table.meta)

    def add_structured_array(
        self,
        array: np.ndarray,
        *,
        name: str | None = None,
        units: Mapping[str, astropy.units.Unit] | None = None,
        descriptions: Mapping[str, str] | None = None,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> TableModel:
        columns = TableColumnModel.from_record_array(array, inline=True)
        for c in columns:
            if units and (unit := units.get(c.name)):
                c.unit = unit
            if descriptions and (description := descriptions.get(c.name)):
                c.description = description
        return TableModel(columns=columns)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_ndf_output_archive.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/ndf/_output_archive.py tests/test_ndf_output_archive.py
git commit -m "Add inline add_table and add_structured_array (DM-54817)"
```

---

### Task 11: Module-level `write()` function

**Files:**
- Modify: `python/lsst/images/ndf/_output_archive.py`
- Modify: `tests/test_ndf_output_archive.py`

- [ ] **Step 1: Write the failing test**

```python
class NdfWriteFunctionTestCase(unittest.TestCase):
    def test_write_image_produces_valid_ndf_layout(self):
        from lsst.images import Box, Image
        from lsst.images.ndf import write

        image = Image(np.arange(20, dtype=np.float32).reshape(4, 5), bbox=Box.factory[10:14, 20:25])
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            tree = write(image, tmp.name)
            self.assertIsNotNone(tree)
            with h5py.File(tmp.name, "r") as f:
                self.assertEqual(f.attrs["HDSTYPE"], "NDF")
                self.assertEqual(f["/DATA_ARRAY"].attrs["HDSTYPE"], "ARRAY")
                np.testing.assert_array_equal(f["/DATA_ARRAY/DATA"][()], image.array)
                self.assertEqual(list(f["/DATA_ARRAY/ORIGIN"][()]), [20, 10])
                # Main JSON tree is at /MORE/LSST/JSON
                self.assertIn("MORE", f)
                self.assertIn("LSST", f["/MORE"])
                self.assertIn("JSON", f["/MORE/LSST"])
```

- [ ] **Step 2: Run the test, expect FAIL (`write` doesn't exist yet)**

- [ ] **Step 3: Implement `write()`**

Append to `_output_archive.py`:

```python
def write(
    obj: Any,
    filename: str | None = None,
    *,
    metadata: dict[str, MetadataValue] | None = None,
    butler_info: ButlerInfo | None = None,
    update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    compression_options: Mapping[str, Any] | None = None,
    compression_seed: int | None = None,
) -> ArchiveTree:
    """Write an object with a ``serialize`` method to an NDF (HDS-on-HDF5) file."""
    import starlink.Ast

    if filename is None:
        # Build the in-memory tree without writing to disk. Used by tests that
        # just want the Pydantic representation. We still need an open HDF5
        # file for the archive to write hoisted blobs into; use an in-memory
        # h5py.File("core" driver).
        h5_file = h5py.File("inmem.sdf", "w", driver="core", backing_store=False)
        opened_locally = True
    else:
        h5_file = h5py.File(filename, "w")
        opened_locally = True
    try:
        opaque_metadata = getattr(obj, "_opaque_metadata", None)
        if not isinstance(opaque_metadata, FitsOpaqueMetadata):
            opaque_metadata = FitsOpaqueMetadata()
        # Allow the caller to add provenance to the primary header via update_header.
        primary_header = opaque_metadata.headers.get(ExtensionKey()) or astropy.io.fits.Header()
        update_header(primary_header)
        if len(primary_header):
            opaque_metadata.headers[ExtensionKey()] = primary_header
        archive = NdfOutputArchive(
            h5_file,
            compression_options=compression_options,
            opaque_metadata=opaque_metadata,
        )
        archive_default_name = getattr(obj, "_archive_default_name", None)
        if archive_default_name is not None:
            tree = archive.serialize_direct(archive_default_name, obj.serialize)
        else:
            tree = obj.serialize(archive)
        if metadata is not None:
            tree.metadata.update(metadata)
        if butler_info is not None:
            tree.butler_info = butler_info
        # Write any FrameSets via starlink-pyast.
        for frame_set, pointer in archive.iter_frame_sets():
            _write_frame_set_to_path(h5_file, pointer.ref, frame_set)
        # Write the main JSON tree.
        json_text = tree.model_dump_json()
        json_lines = NdfOutputArchive._wrap_json_for_storage(
            json_text, max_line_width=max(80, len(json_text))
        )
        more_lsst = archive._ensure_path("/MORE/LSST")
        if "JSON" in more_lsst:
            del more_lsst["JSON"]
        _hds.write_char_array(more_lsst, "JSON", json_lines, width=max(80, len(json_text)))
        # Write opaque FITS cards.
        primary = opaque_metadata.headers.get(ExtensionKey())
        if primary is not None and len(primary):
            cards = [card.image for card in primary.cards]
            more = archive._ensure_path("/MORE")
            if "FITS" in more:
                del more["FITS"]
            _hds.write_char_array(more, "FITS", cards, width=80)
        # Write bbox-derived ORIGIN for DATA_ARRAY/VARIANCE if applicable.
        bbox = getattr(obj, "bbox", None)
        if bbox is not None:
            origin_xy = (bbox.x_min, bbox.y_min)
            for struct_path in ("/DATA_ARRAY", "/VARIANCE"):
                if struct_path in h5_file:
                    archive.set_array_origin(struct_path, origin_xy)
        return tree
    finally:
        if opened_locally:
            h5_file.close()


def _write_frame_set_to_path(file: h5py.File, path: str, frame_set: Any) -> None:
    """Serialize a starlink.Ast.FrameSet to a list of strings and store at path."""
    import starlink.Ast

    if not isinstance(frame_set, starlink.Ast.FrameSet):
        # Test sentinels and other oddities — write a placeholder so the layout
        # remains valid HDS even if the FrameSet isn't real.
        lines = [str(frame_set)]
    else:
        channel = starlink.Ast.Channel()
        lines = []
        channel.write(frame_set, sink=lines.append)
    parent_path, leaf = path.rsplit("/", 1)
    parent = file[parent_path]
    if leaf in parent:
        del parent[leaf]
    _hds.write_char_array(parent, leaf, lines, width=max(80, max((len(l) for l in lines), default=80)))
```

(Note: this task's `write()` carries some intentional rough edges — e.g., the test sentinel branch in `_write_frame_set_to_path`. Real FrameSets always go through the AST channel path; the sentinel exists only so `serialize_frame_set` unit tests don't need a real FrameSet.)

- [ ] **Step 4: Run the test**

Run: `pytest tests/test_ndf_output_archive.py::NdfWriteFunctionTestCase -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/ndf/_output_archive.py tests/test_ndf_output_archive.py
git commit -m "Add ndf.write() top-level function (DM-54817)"
```

---

## Phase 3 — Input archive

### Task 12: `NdfInputArchive` open + `get_tree`

**Files:**
- Create: `python/lsst/images/ndf/_input_archive.py`
- Create: `tests/test_ndf_input_archive.py`

- [ ] **Step 1: Write the failing test**

```python
# (license header)
from __future__ import annotations

import tempfile
import unittest

import h5py
import numpy as np

from lsst.images import Box, Image
from lsst.images.ndf import write
from lsst.images.ndf._input_archive import NdfInputArchive


class NdfInputArchiveOpenTestCase(unittest.TestCase):
    def test_open_round_trips_image_tree(self):
        image = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5), bbox=Box.factory[10:14, 20:25]
        )
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            written_tree = write(image, tmp.name)
            with NdfInputArchive.open(tmp.name) as arch:
                tree = arch.get_tree(type(written_tree))
                self.assertEqual(
                    tree.model_dump_json(), written_tree.model_dump_json()
                )
```

- [ ] **Step 2: Run the test, expect FAIL (`NdfInputArchive` doesn't exist)**

- [ ] **Step 3: Implement `NdfInputArchive.open` and `get_tree`**

Create `python/lsst/images/ndf/_input_archive.py`:

```python
# (license header)
from __future__ import annotations

__all__ = ("NdfInputArchive", "read")

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Self

import astropy.io.fits
import astropy.table
import h5py
import numpy as np

from lsst.resources import ResourcePath, ResourcePathExpression

from .._transforms import FrameSet
from ..fits._common import FitsOpaqueMetadata, ExtensionKey
from ..serialization import (
    ArchiveReadError,
    ArchiveTree,
    ArrayReferenceModel,
    InlineArrayModel,
    InputArchive,
    ReadResult,
    TableModel,
    no_header_updates,
)
from . import _hds
from ._common import NdfPointerModel

if TYPE_CHECKING:
    pass


_LOG = logging.getLogger(__name__)


class NdfInputArchive(InputArchive[NdfPointerModel]):
    """An :class:`~lsst.images.serialization.InputArchive` implementation
    that reads HDS-on-HDF5 NDF files.

    Instances of this class should only be constructed via the :meth:`open`
    context manager.
    """

    def __init__(self, file: h5py.File) -> None:
        self._file = file
        self._opaque_metadata = FitsOpaqueMetadata()
        self._read_opaque_fits_metadata()
        self._deserialized_pointer_cache: dict[str, Any] = {}
        self._frame_set_cache: dict[str, FrameSet] = {}

    @classmethod
    @contextmanager
    def open(cls, path: ResourcePathExpression) -> Iterator[Self]:
        rp = ResourcePath(path)
        # Materialise locally if remote (fsspec-direct h5py is a follow-up).
        with rp.as_local() as local:
            with h5py.File(local.ospath, "r") as f:
                yield cls(f)

    def get_tree[T: ArchiveTree](self, model_type: type[T]) -> T:
        if "/MORE/LSST/JSON" not in self._file:
            raise ArchiveReadError(
                "File has no /MORE/LSST/JSON tree; use ndf.read() with auto-detect for "
                "Starlink-only files."
            )
        lines = _hds.read_char_array(self._file["/MORE/LSST/JSON"])
        json_bytes = "".join(lines).encode()
        return model_type.model_validate_json(json_bytes)

    def deserialize_pointer[U: ArchiveTree, V](
        self,
        pointer: NdfPointerModel,
        model_type: type[U],
        deserializer: Callable[[U, InputArchive[NdfPointerModel]], V],
    ) -> V:
        # Implemented in Task 13.
        raise NotImplementedError

    def get_frame_set(self, ref: NdfPointerModel) -> FrameSet:
        # Implemented in Task 13.
        raise NotImplementedError

    def get_array(
        self,
        model: ArrayReferenceModel | InlineArrayModel,
        *,
        slices: tuple[slice, ...] | EllipsisType = ...,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> np.ndarray:
        # Implemented in Task 13.
        raise NotImplementedError

    def get_table(
        self,
        model: TableModel,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> astropy.table.Table:
        # Inline-only for v1 (paralleling JsonInputArchive).
        result = astropy.table.Table(meta=model.meta)
        for column_model in model.columns:
            if not isinstance(column_model.data, InlineArrayModel):
                raise ArchiveReadError("Only inline tables are supported in NDF archives in v1.")
            result[column_model.name] = astropy.table.Column(
                column_model.data.data,
                name=column_model.name,
                dtype=column_model.data.datatype.to_numpy(),
                unit=column_model.unit,
                description=column_model.description,
                meta=column_model.meta,
            )
        return result

    def get_structured_array(
        self,
        model: TableModel,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> np.ndarray:
        return self.get_table(model, strip_header).as_array()

    def get_opaque_metadata(self) -> FitsOpaqueMetadata:
        return self._opaque_metadata

    def _read_opaque_fits_metadata(self) -> None:
        # Implemented in Task 14.
        pass
```

- [ ] **Step 4: Run the test**

Run: `pytest tests/test_ndf_input_archive.py::NdfInputArchiveOpenTestCase -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/ndf/_input_archive.py tests/test_ndf_input_archive.py
git commit -m "Add NdfInputArchive open() and get_tree (DM-54817)"
```

---

### Task 13: `get_array`, `deserialize_pointer`, `get_frame_set`

**Files:**
- Modify: `python/lsst/images/ndf/_input_archive.py`
- Modify: `tests/test_ndf_input_archive.py`

- [ ] **Step 1: Write failing tests**

```python
class NdfInputArchiveDataTestCase(unittest.TestCase):
    def test_get_array_reads_image_array(self):
        image = Image(np.arange(20, dtype=np.float32).reshape(4, 5))
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            tree = write(image, tmp.name)
            with NdfInputArchive.open(tmp.name) as arch:
                arr = arch.get_array(tree.image.array)
                np.testing.assert_array_equal(arr, image.array)
```

- [ ] **Step 2: Run, expect FAIL (`NotImplementedError`)**

- [ ] **Step 3: Implement `get_array`, `deserialize_pointer`, `get_frame_set`**

Replace the three stubs:

```python
    def deserialize_pointer[U: ArchiveTree, V](
        self,
        pointer: NdfPointerModel,
        model_type: type[U],
        deserializer: Callable[[U, InputArchive[NdfPointerModel]], V],
    ) -> V:
        if (cached := self._deserialized_pointer_cache.get(pointer.ref)) is not None:
            return cached
        if pointer.ref == "/WCS/DATA":
            # WCS pointers resolve to FrameSet via get_frame_set, not to a JSON model.
            frame_set = self._read_frame_set("/WCS/DATA")
            self._frame_set_cache[pointer.ref] = frame_set
            self._deserialized_pointer_cache[pointer.ref] = frame_set
            return frame_set  # type: ignore[return-value]
        if pointer.ref not in self._file:
            raise ArchiveReadError(f"Pointer reference {pointer.ref!r} not in file.")
        ds = self._file[pointer.ref]
        lines = _hds.read_char_array(ds)
        json_bytes = "".join(lines).encode()
        model = model_type.model_validate_json(json_bytes)
        result = deserializer(model, self)
        self._deserialized_pointer_cache[pointer.ref] = result
        return result

    def get_frame_set(self, ref: NdfPointerModel) -> FrameSet:
        try:
            return self._frame_set_cache[ref.ref]
        except KeyError:
            raise AssertionError(
                f"Frame set at {ref.ref!r} must be deserialized via deserialize_pointer first."
            ) from None

    def get_array(
        self,
        model: ArrayReferenceModel | InlineArrayModel,
        *,
        slices: tuple[slice, ...] | EllipsisType = ...,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> np.ndarray:
        if isinstance(model, InlineArrayModel):
            return np.array(model.data, dtype=model.datatype.to_numpy())[slices]
        if not isinstance(model.source, str) or not model.source.startswith("ndf:"):
            raise ArchiveReadError(f"NdfInputArchive cannot handle source={model.source!r}.")
        path = model.source[len("ndf:") :]
        if path not in self._file:
            raise ArchiveReadError(f"Array reference {path!r} not in file.")
        ds = self._file[path]
        if slices is ...:
            return _hds.read_array(ds)
        # h5py supports slice-on-read. Validate the dataset matches an HDS primitive.
        return ds[slices]

    def _read_frame_set(self, path: str) -> FrameSet:
        import starlink.Ast

        ds = self._file[path]
        lines = _hds.read_char_array(ds)
        # FrameSets sometimes have lines longer than 80 chars; lines from
        # _hds.read_char_array are already stripped of trailing spaces.
        iter_lines = iter(lines)

        def source() -> str | None:
            try:
                return next(iter_lines)
            except StopIteration:
                return None

        channel = starlink.Ast.Channel()
        return channel.read(source=source)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_ndf_input_archive.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/ndf/_input_archive.py tests/test_ndf_input_archive.py
git commit -m "Add get_array, deserialize_pointer, get_frame_set (DM-54817)"
```

---

### Task 14: Read opaque FITS metadata from `/MORE/FITS`

**Files:**
- Modify: `python/lsst/images/ndf/_input_archive.py`
- Modify: `tests/test_ndf_input_archive.py`

- [ ] **Step 1: Write the failing test**

```python
class NdfInputArchiveOpaqueMetadataTestCase(unittest.TestCase):
    def test_more_fits_round_trips_via_opaque_metadata(self):
        from lsst.images.fits._common import FitsOpaqueMetadata

        image = Image(np.arange(4, dtype=np.float32).reshape(2, 2))
        original_header = astropy.io.fits.Header()
        original_header["FOO"] = ("bar", "test card")
        opaque = FitsOpaqueMetadata()
        opaque.add_header(original_header, name="", ver=1)
        image._opaque_metadata = opaque
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            write(image, tmp.name)
            with NdfInputArchive.open(tmp.name) as arch:
                recovered = arch.get_opaque_metadata()
                primary = recovered.headers[ExtensionKey()]
                self.assertEqual(primary["FOO"], "bar")
```

(Add `import astropy.io.fits` and `from lsst.images.fits._common import ExtensionKey` at the top of the test file.)

- [ ] **Step 2: Run, expect FAIL (header missing — `_read_opaque_fits_metadata` is a no-op)**

- [ ] **Step 3: Implement `_read_opaque_fits_metadata`**

Replace the stub in `_input_archive.py`:

```python
    def _read_opaque_fits_metadata(self) -> None:
        if "/MORE/FITS" not in self._file:
            return
        cards = _hds.read_char_array(self._file["/MORE/FITS"])
        header = astropy.io.fits.Header.fromstring("".join(c.ljust(80) for c in cards))
        self._opaque_metadata.add_header(header, name="", ver=1)
```

- [ ] **Step 4: Run**

Run: `pytest tests/test_ndf_input_archive.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/ndf/_input_archive.py tests/test_ndf_input_archive.py
git commit -m "Read /MORE/FITS into FitsOpaqueMetadata (DM-54817)"
```

---

### Task 15: Module-level `read()` with auto-detect for Starlink-only files

**Files:**
- Modify: `python/lsst/images/ndf/_input_archive.py`
- Modify: `tests/test_ndf_input_archive.py`

- [ ] **Step 1: Write tests for both round-trip and auto-detect paths**

```python
class NdfReadFunctionTestCase(unittest.TestCase):
    def test_read_round_trips_image(self):
        from lsst.images.ndf import read

        image = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5), bbox=Box.factory[10:14, 20:25]
        )
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            write(image, tmp.name)
            result = read(Image, tmp.name)
            self.assertIsInstance(result.deserialized, Image)
            np.testing.assert_array_equal(result.deserialized.array, image.array)
            self.assertEqual(result.deserialized.bbox, image.bbox)

    def test_read_starlink_only_file_auto_detects_image(self):
        # The packaged example.sdf has no /MORE/LSST/JSON, no /QUALITY,
        # no /VARIANCE — so auto-detect should return an Image.
        from lsst.images.ndf import read

        example_path = os.path.join(
            os.path.dirname(__file__), "data", "example.sdf"
        )
        result = read(Image, example_path)
        self.assertIsInstance(result.deserialized, Image)
        self.assertEqual(result.deserialized.array.shape, (128, 128))
```

(Add `import os` if not already imported.)

- [ ] **Step 2: Run, expect FAIL (`read` not defined)**

- [ ] **Step 3: Implement `read()` with auto-detect**

Append to `_input_archive.py`:

```python
def read[T: Any](cls: type[T], path: ResourcePathExpression, **kwargs: Any) -> ReadResult[T]:
    """Read an NDF (HDS-on-HDF5) file.

    If the file has a ``/MORE/LSST/JSON`` tree, it's used as the source of
    truth and ``cls.deserialize`` is called with the parsed tree. Otherwise
    the reader falls back to auto-detection, supporting only ``Image`` and
    ``MaskedImage`` from a minimal recognised-component set
    (``DATA_ARRAY``, ``VARIANCE``, ``QUALITY``, ``WCS``, ``MORE.FITS``).
    """
    with NdfInputArchive.open(path) as archive:
        if "/MORE/LSST/JSON" in archive._file:
            tree_type = cls._get_archive_tree_type(NdfPointerModel)
            tree = archive.get_tree(tree_type)
            obj = cls.deserialize(tree, archive, **kwargs)
            obj._opaque_metadata = archive.get_opaque_metadata()
            return ReadResult(obj, tree.metadata, tree.butler_info)
        # Auto-detect path.
        return _read_auto_detect(cls, archive, **kwargs)


def _read_auto_detect[T: Any](
    cls: type[T], archive: NdfInputArchive, **kwargs: Any
) -> ReadResult[T]:
    """Construct an Image/MaskedImage from a Starlink-only NDF file.

    Components handled:
        DATA_ARRAY/DATA, VARIANCE/DATA, QUALITY/QUALITY, WCS, MORE.FITS

    Anything else is logged at WARNING level and dropped.
    """
    from lsst.images import Box, Image, Mask, MaskedImage, MaskSchema, MaskPlane, Projection

    f = archive._file
    # Walk the top-level looking for an NDF, allowing a one-level container.
    ndf_group = f
    if "HDSTYPE" not in f.attrs or (f.attrs.get("HDSTYPE") not in ("NDF", b"NDF")):
        # Maybe a container with one child NDF.
        children = dict(_hds.iter_children(f))
        ndf_candidates = [name for name, child in children.items()
                          if isinstance(child, h5py.Group)
                          and child.attrs.get("HDSTYPE") in ("NDF", b"NDF")]
        if len(ndf_candidates) == 1:
            ndf_group = f[ndf_candidates[0]]
        else:
            raise ArchiveReadError(
                f"Could not locate top-level NDF in {archive._file.filename!r}."
            )
    # DATA_ARRAY (required).
    if "DATA_ARRAY" not in ndf_group or "DATA" not in ndf_group["DATA_ARRAY"]:
        raise ArchiveReadError("File is HDS but contains no image DATA_ARRAY/DATA.")
    data_arr = _hds.read_array(ndf_group["DATA_ARRAY/DATA"])
    if "ORIGIN" in ndf_group["DATA_ARRAY"]:
        origin = _hds.read_array(ndf_group["DATA_ARRAY/ORIGIN"])
        bbox = Box(int(origin[0]), int(origin[1]), data_arr.shape[1], data_arr.shape[0])
    else:
        bbox = Box(0, 0, data_arr.shape[1], data_arr.shape[0])
    # WCS: not populated in the auto-detect path for v1. The symmetric
    # write→read path uses the Pydantic tree's pointer/get_frame_set
    # machinery, which round-trips WCS faithfully. The auto-detect path is
    # for Starlink-only files; building a `Projection` from a bare
    # `starlink.Ast.FrameSet` requires picking the right constructor in
    # `_transforms/_projection.py` and is left as a follow-up. We log a
    # warning so the user knows WCS data was present but dropped.
    projection = None
    if "WCS" in ndf_group and "DATA" in ndf_group["WCS"]:
        _LOG.warning(
            "Starlink WCS present in %s but auto-detect ingest does not yet "
            "build a Projection from it; dropping. Round-trip writes from "
            "lsst.images.ndf preserve WCS via the Pydantic tree.",
            archive._file.filename,
        )
    # Warn-and-drop unrecognised components.
    recognised = {"DATA_ARRAY", "VARIANCE", "QUALITY", "WCS", "MORE"}
    for name in ndf_group:
        if name not in recognised:
            _LOG.warning(
                "Ignoring unrecognised NDF component %s/%s during auto-detect read.",
                ndf_group.name,
                name,
            )
    if "VARIANCE" in ndf_group or "QUALITY" in ndf_group:
        # MaskedImage path.
        variance = (_hds.read_array(ndf_group["VARIANCE/DATA"])
                    if "VARIANCE" in ndf_group else np.zeros_like(data_arr))
        if "QUALITY" in ndf_group and "QUALITY" in ndf_group["QUALITY"]:
            mask_arr = _hds.read_array(ndf_group["QUALITY/QUALITY"])
        else:
            mask_arr = np.zeros(data_arr.shape, dtype=np.uint8)
        schema = MaskSchema([MaskPlane(name="BAD", description="Bad pixel.")])
        masked = MaskedImage(
            image=Image(data_arr, bbox=bbox, projection=projection),
            mask=Mask(mask_arr, schema=schema, bbox=bbox),
            variance=Image(variance, bbox=bbox),
        )
        if not isinstance(masked, cls):
            raise ArchiveReadError(
                f"Auto-detect produced MaskedImage but caller asked for {cls.__name__}."
            )
        return ReadResult(masked, {}, None)
    image = Image(data_arr, bbox=bbox, projection=projection)
    if not isinstance(image, cls):
        raise ArchiveReadError(
            f"Auto-detect produced Image but caller asked for {cls.__name__}."
        )
    return ReadResult(image, {}, None)
```

(If `Projection.from_frame_set` doesn't exist, use whatever helper `_transforms/_ast.py` exposes. The plan worker should grep `_transforms/_ast.py` for the right entry point.)

- [ ] **Step 4: Run**

Run: `pytest tests/test_ndf_input_archive.py -v`
Expected: PASS. The auto-detect test against `example.sdf` may need adjustment if the package defines `Projection.from_frame_set` differently — adjust call as needed.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/ndf/_input_archive.py tests/test_ndf_input_archive.py
git commit -m "Add ndf.read() with auto-detect for Starlink-only files (DM-54817)"
```

---

## Phase 4 — Formatter and exports

### Task 16: `formatters.py` — daf_butler formatter classes

**Files:**
- Create: `python/lsst/images/ndf/formatters.py`

- [ ] **Step 1: Mirror the FITS formatter classes**

Read `python/lsst/images/fits/formatters.py` first; the NDF formatter is a near-copy with `ndf.read`/`ndf.write` swapped in for `fits.read`/`fits.write` and `default_extension` set to `.sdf`. The structure (`GenericFormatter` → `ImageFormatter` → `MaskedImageFormatter` → `VisitImageFormatter`) is identical.

Create `python/lsst/images/ndf/formatters.py`:

```python
# (license header)
from __future__ import annotations

__all__ = ("GenericFormatter", "ImageFormatter", "MaskedImageFormatter", "VisitImageFormatter")

import enum
from typing import Any, ClassVar

from lsst.daf.butler import DatasetProvenance, FormatterV2
from lsst.resources import ResourcePath

from .._geom import Box
from .._image import Image
from .._mask import Mask
from .._masked_image import MaskedImageSerializationModel
from .._observation_summary_stats import ObservationSummaryStats
from .._transforms import Projection, ProjectionSerializationModel
from .._visit_image import VisitImageSerializationModel
from ..serialization import ButlerInfo
from ._common import NdfPointerModel
from ._input_archive import NdfInputArchive, read
from ._output_archive import write


class GenericFormatter(FormatterV2):
    """Butler interface to the NDF (HDS-on-HDF5) archive."""

    default_extension: ClassVar[str] = ".sdf"
    can_read_from_uri: ClassVar[bool] = True
    supported_write_parameters: ClassVar[frozenset[str]] = frozenset()

    butler_provenance: DatasetProvenance | None = None

    def read_from_uri(self, uri: ResourcePath, component: str | None = None, expected_size: int = -1) -> Any:
        pytype = self.dataset_ref.datasetType.storageClass.pytype
        kwargs = self.file_descriptor.parameters or {}
        return read(pytype, uri, **kwargs).deserialized

    def write_local_file(self, in_memory_dataset: Any, uri: ResourcePath) -> None:
        butler_info = ButlerInfo(
            dataset=self.dataset_ref.to_simple(),
            provenance=self.butler_provenance if self.butler_provenance is not None else DatasetProvenance(),
        )
        write(in_memory_dataset, uri.ospath, butler_info=butler_info)

    def add_provenance(
        self, in_memory_dataset: Any, /, *, provenance: DatasetProvenance | None = None
    ) -> Any:
        self.butler_provenance = provenance
        return in_memory_dataset


class ComponentSentinel(enum.Enum):
    UNRECOGNIZED_COMPONENT = enum.auto()
    INVALID_COMPONENT_MODEL = enum.auto()


class ImageFormatter(GenericFormatter):
    """Specialised butler interface to NDF serialization for image-like
    objects with ``projection``/``bbox`` components."""

    def read_from_uri(self, uri: ResourcePath, component: str | None = None, expected_size: int = -1) -> Any:
        pytype: Any = self.file_descriptor.storageClass.pytype
        if component is None:
            result = read(pytype, uri, bbox=self.pop_bbox_from_parameters()).deserialized
        else:
            with NdfInputArchive.open(uri) as archive:
                tree = archive.get_tree(pytype._get_archive_tree_type(NdfPointerModel))
                result = self.read_component(component, tree, archive)
                if result is ComponentSentinel.UNRECOGNIZED_COMPONENT:
                    raise NotImplementedError(
                        f"Unrecognized component {component!r} for {type(self).__name__}."
                    )
                if result is ComponentSentinel.INVALID_COMPONENT_MODEL:
                    raise NotImplementedError(
                        f"Invalid serialization model for component {component!r} for {type(self).__name__}."
                    )
        self.check_unhandled_parameters()
        return result

    def pop_bbox_from_parameters(self) -> Box | None:
        parameters = self.file_descriptor.parameters or {}
        return parameters.pop("bbox", None)

    def check_unhandled_parameters(self) -> None:
        if self.file_descriptor.parameters:
            raise RuntimeError(f"Parameters {list(self.file_descriptor.parameters.keys())} not recognized.")

    def read_component(self, component: str, tree: Any, archive: NdfInputArchive) -> Any:
        from astro_metadata_translator import ObservationInfo

        match component:
            case "projection":
                if isinstance(p := getattr(tree, "projection", None), ProjectionSerializationModel):
                    return Projection.deserialize(p, archive)
                return ComponentSentinel.INVALID_COMPONENT_MODEL
            case "bbox":
                if isinstance(bbox := getattr(tree, "bbox", None), Box):
                    return bbox
                return ComponentSentinel.INVALID_COMPONENT_MODEL
            case "obs_info":
                if isinstance(oi := getattr(tree, "obs_info", None), ObservationInfo):
                    return oi
                return ComponentSentinel.INVALID_COMPONENT_MODEL
        return ComponentSentinel.UNRECOGNIZED_COMPONENT


class MaskedImageFormatter(ImageFormatter):
    def read_component(self, component: str, tree: Any, archive: NdfInputArchive) -> Any:
        match super().read_component(component, tree, archive):
            case ComponentSentinel():
                pass
            case handled:
                return handled
        if not isinstance(tree, MaskedImageSerializationModel):
            return ComponentSentinel.INVALID_COMPONENT_MODEL
        match component:
            case "image":
                return Image.deserialize(tree.image, archive, bbox=self.pop_bbox_from_parameters())
            case "mask":
                return Mask.deserialize(tree.mask, archive, bbox=self.pop_bbox_from_parameters())
            case "variance":
                return Image.deserialize(tree.variance, archive, bbox=self.pop_bbox_from_parameters())
        return ComponentSentinel.UNRECOGNIZED_COMPONENT


class VisitImageFormatter(MaskedImageFormatter):
    def read_component(self, component: str, tree: Any, archive: NdfInputArchive) -> Any:
        match super().read_component(component, tree, archive):
            case ComponentSentinel():
                pass
            case handled:
                return handled
        if not isinstance(tree, VisitImageSerializationModel):
            return ComponentSentinel.INVALID_COMPONENT_MODEL
        match component:
            case "psf":
                return tree.deserialize_psf(archive)
            case "summary_stats":
                if isinstance(ss := getattr(tree, "summary_stats", None), ObservationSummaryStats):
                    return ss
                return ComponentSentinel.INVALID_COMPONENT_MODEL
        return ComponentSentinel.UNRECOGNIZED_COMPONENT
```

- [ ] **Step 2: Sanity-check imports**

Run: `python -c "from lsst.images.ndf import formatters; print(formatters.GenericFormatter)"`
Expected: success, prints the class.

- [ ] **Step 3: Commit**

```bash
git add python/lsst/images/ndf/formatters.py
git commit -m "Add NDF formatter classes for daf_butler (DM-54817)"
```

---

### Task 17: `__init__.py` re-exports

**Files:**
- Modify: `python/lsst/images/ndf/__init__.py`

- [ ] **Step 1: Update `__init__.py`**

Replace the current `__init__.py` body (after the license header and module docstring) with:

```python
from ._common import *
from ._input_archive import *
from ._output_archive import *
```

- [ ] **Step 2: Verify the public surface**

Run:

```bash
python -c "from lsst.images import ndf; print(sorted(name for name in dir(ndf) if not name.startswith('_')))"
```

Expected: a list including `NdfInputArchive`, `NdfOutputArchive`, `NdfPointerModel`, `read`, `write`.

- [ ] **Step 3: Commit**

```bash
git add python/lsst/images/ndf/__init__.py
git commit -m "Wire ndf subpackage __init__ re-exports (DM-54817)"
```

---

## Phase 5 — Test integration

### Task 18: `RoundtripNdf` in `tests/_roundtrip.py`

**Files:**
- Modify: `python/lsst/images/tests/_roundtrip.py`
- Modify: `python/lsst/images/tests/__init__.py`

- [ ] **Step 1: Add the class**

In `python/lsst/images/tests/_roundtrip.py`, after `RoundtripJson`, add:

```python
class RoundtripNdf[T](RoundtripBase):
    def inspect(self) -> Any:
        """Open the NDF file with h5py."""
        import h5py
        return self._exit_stack.enter_context(h5py.File(self.filename, "r"))

    def _get_extension(self) -> str:
        return ".sdf"

    def _write(self, obj: Any, filename: str) -> ArchiveTree:
        from .. import ndf
        return ndf.write(obj, filename)

    def _read(self, obj_type: Any, filename: str) -> ReadResult:
        from .. import ndf
        return ndf.read(obj_type, filename)
```

Add `"RoundtripNdf"` to the module's `__all__` tuple at line 14.

In `python/lsst/images/tests/__init__.py`, ensure `RoundtripNdf` is re-exported (mirror how `RoundtripFits`/`RoundtripJson` are exposed).

- [ ] **Step 2: Verify import**

Run: `python -c "from lsst.images.tests import RoundtripNdf; print(RoundtripNdf)"`
Expected: prints the class.

- [ ] **Step 3: Commit**

```bash
git add python/lsst/images/tests/_roundtrip.py python/lsst/images/tests/__init__.py
git commit -m "Add RoundtripNdf for tests (DM-54817)"
```

---

### Task 19: Round-trip integration in existing test files

**Files:**
- Modify: `tests/test_image.py`
- Modify: `tests/test_masked_image.py`
- Modify: `tests/test_visit_image.py`

For each file, add a test method that mirrors the existing `RoundtripFits` / `RoundtripJson` usage but uses `RoundtripNdf`. Example for `tests/test_image.py`:

- [ ] **Step 1: Add the test method to `tests/test_image.py`**

Add `RoundtripNdf` to the test-utility import at line 24-30 (the `from lsst.images.tests import (...)` block). Then, inside `ImageTestCase`, add a new test method that mirrors the existing FITS round-trip pattern at line 102:

```python
    def test_round_trip_ndf(self):
        """NDF archive round-trip for Image."""
        rng = np.random.default_rng(123)
        image = Image(
            rng.normal(100.0, 8.0, size=(200, 251)),
            dtype=np.float64,
            unit=u.nJy,
            start=(5, 8),
            metadata={"hello": "world"},
        )
        with RoundtripNdf(self, image) as roundtrip:
            assert_images_equal(self, image, roundtrip.result)
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_image.py::ImageTestCase::test_round_trip_ndf -v`
Expected: PASS.

- [ ] **Step 3: Repeat for `tests/test_masked_image.py`**

Add `RoundtripNdf` to the imports at the top of the file. The existing `MaskedImageTestCase.setUp` (lines 33-56) already constructs `self.masked_image` with a 2-plane mask schema (`BAD`, `HUNGRY`) on a uint8 dtype — that's already a "compatible" mask. Add inside the class:

```python
    def test_round_trip_ndf_compatible_mask(self):
        """NDF round-trip for the default-setup MaskedImage (2 planes ≤ 8)."""
        with RoundtripNdf(self, self.masked_image) as roundtrip:
            assert_masked_images_equal(
                self, roundtrip.result, self.masked_image, expect_view=False
            )

    def test_round_trip_ndf_incompatible_mask(self):
        """NDF round-trip for a >8-plane mask (forces 3D mask array, hoisted to MORE/LSST/MASK)."""
        rng = np.random.default_rng(7)
        planes = [MaskPlane(f"P{i}", f"plane {i}") for i in range(12)]
        wide = MaskedImage(
            Image(
                rng.normal(100.0, 8.0, size=(50, 60)),
                dtype=np.float64,
                unit=u.nJy,
                start=(0, 0),
            ),
            mask_schema=MaskSchema(planes),
            obs_info=self.obs_info,
        )
        wide.variance.array = rng.normal(64.0, 0.5, size=wide.bbox.shape)
        with RoundtripNdf(self, wide) as roundtrip:
            assert_masked_images_equal(self, roundtrip.result, wide, expect_view=False)
```

(`u` is already imported as `import astropy.units as u`. Add `from lsst.images.tests import RoundtripNdf` to the existing tests import block.)

- [ ] **Step 4: Repeat for `tests/test_visit_image.py`**

Inspect the existing test file to find the local `setUp`-built `self.visit_image` or equivalent. Mirror it:

```python
    def test_round_trip_ndf(self):
        """NDF round-trip for VisitImage."""
        with RoundtripNdf(self, self.visit_image) as roundtrip:
            assert_masked_images_equal(
                self, roundtrip.result, self.visit_image, expect_view=False
            )
            self.assertEqual(roundtrip.result.summary_stats, self.visit_image.summary_stats)
            self.assertEqual(type(roundtrip.result.psf), type(self.visit_image.psf))
```

(If `test_visit_image.py` uses a different fixture name or builder, substitute the local equivalent. If summary stats / psf assertions need different equality helpers, follow the patterns in the existing FITS round-trip test in the same file.)

- [ ] **Step 5: Run all touched tests**

Run: `pytest tests/test_image.py tests/test_masked_image.py tests/test_visit_image.py -v -k ndf`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/test_image.py tests/test_masked_image.py tests/test_visit_image.py
git commit -m "Add NDF round-trip tests for Image/MaskedImage/VisitImage (DM-54817)"
```

---

### Task 20: Layout sanity tests

**Files:**
- Create: `tests/test_ndf_layout.py`

- [ ] **Step 1: Write the test**

```python
# (license header)
from __future__ import annotations

import tempfile
import unittest

import h5py
import numpy as np

from lsst.images import Box, Image, Mask, MaskedImage, MaskPlane, MaskSchema
from lsst.images.ndf import write


class NdfLayoutTestCase(unittest.TestCase):
    """Open files written by NdfOutputArchive with raw h5py and verify the
    on-disk layout matches the spec.
    """

    def test_image_layout(self):
        image = Image(np.arange(20, dtype=np.float32).reshape(4, 5), bbox=Box.factory[10:14, 20:25])
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            write(image, tmp.name)
            with h5py.File(tmp.name, "r") as f:
                self.assertEqual(f.attrs["HDSTYPE"], "NDF")
                self.assertEqual(f["/DATA_ARRAY"].attrs["HDSTYPE"], "ARRAY")
                self.assertEqual(f["/DATA_ARRAY/DATA"].attrs["HDSTYPE"], "_REAL")
                self.assertEqual(f["/DATA_ARRAY/DATA"].attrs["HDSNDIMS"], 2)
                self.assertEqual(f["/DATA_ARRAY/DATA"].shape, (4, 5))
                self.assertEqual(list(f["/DATA_ARRAY/ORIGIN"][()]), [20, 10])
                self.assertEqual(f["/MORE/LSST"].attrs["HDSTYPE"], "EXT")
                self.assertIn("JSON", f["/MORE/LSST"])

    def test_masked_image_compatible_mask_layout(self):
        """Compatible mask (uint8 + ≤8 planes) → native /QUALITY component."""
        rng = np.random.default_rng(11)
        masked = MaskedImage(
            Image(rng.normal(100.0, 8.0, size=(40, 50)), dtype=np.float64, start=(0, 0)),
            mask_schema=MaskSchema([MaskPlane("BAD", "bad pixel.")]),
        )
        masked.variance.array = rng.normal(64.0, 0.5, size=masked.bbox.shape)
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            write(masked, tmp.name)
            with h5py.File(tmp.name, "r") as f:
                self.assertEqual(f["/QUALITY"].attrs["HDSTYPE"], "QUALITY")
                self.assertEqual(f["/QUALITY/QUALITY"].attrs["HDSTYPE"], "_UBYTE")
                self.assertEqual(int(f["/QUALITY/BADBITS"][()]), 0xFF)
                self.assertEqual(f["/VARIANCE"].attrs["HDSTYPE"], "ARRAY")
                self.assertEqual(f["/VARIANCE/DATA"].attrs["HDSTYPE"], "_DOUBLE")
                self.assertNotIn("MASK", f.get("/MORE/LSST", {}))

    def test_masked_image_incompatible_mask_layout(self):
        """Incompatible mask (>8 planes → 3D array) → hoisted to /MORE/LSST/MASK."""
        rng = np.random.default_rng(12)
        planes = [MaskPlane(f"P{i}", f"plane {i}") for i in range(12)]
        masked = MaskedImage(
            Image(rng.normal(100.0, 8.0, size=(40, 50)), dtype=np.float64, start=(0, 0)),
            mask_schema=MaskSchema(planes),
        )
        masked.variance.array = rng.normal(64.0, 0.5, size=masked.bbox.shape)
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            write(masked, tmp.name)
            with h5py.File(tmp.name, "r") as f:
                self.assertNotIn("QUALITY", f)
                self.assertEqual(f["/MORE/LSST/MASK"].attrs["HDSTYPE"], "STRUCT")
                self.assertEqual(f["/MORE/LSST/MASK/DATA"].attrs["HDSTYPE"], "_UBYTE")
                # 12 planes / 8 bits per byte = 2-byte trailing axis.
                self.assertEqual(f["/MORE/LSST/MASK/DATA"].shape, (40, 50, 2))
```

(Fill in the construction details by reading `tests/test_masked_image.py` for how `MaskedImage` is built in tests.)

- [ ] **Step 2: Run**

Run: `pytest tests/test_ndf_layout.py -v`
Expected: PASS for the test_image_layout case at minimum; complete the masked-image cases by following the construction patterns in `tests/test_masked_image.py`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ndf_layout.py
git commit -m "Add NDF layout sanity tests (DM-54817)"
```

---

### Task 21: Starlink-ingest test

**Files:**
- Create: `tests/test_ndf_starlink_ingest.py`

- [ ] **Step 1: Write the test**

```python
# (license header)
from __future__ import annotations

import os
import unittest

import numpy as np

from lsst.images import Image
from lsst.images.ndf import read


EXAMPLE = os.path.join(os.path.dirname(__file__), "data", "example.sdf")


class NdfStarlinkIngestTestCase(unittest.TestCase):
    """Read a Starlink-generated NDF (CCDPACK BIAS frame) via the auto-detect path."""

    def test_round_trips_to_image(self):
        result = read(Image, EXAMPLE)
        self.assertIsInstance(result.deserialized, Image)
        self.assertEqual(result.deserialized.array.shape, (128, 128))

    def test_opaque_fits_metadata_recovered(self):
        result = read(Image, EXAMPLE)
        opaque = result.deserialized._opaque_metadata
        from lsst.images.fits._common import ExtensionKey

        primary = opaque.headers[ExtensionKey()]
        self.assertEqual(primary["OBSTYPE"], "BIAS")
        self.assertEqual(primary["NAXIS1"], 128)
        self.assertEqual(primary["NAXIS2"], 128)
```

- [ ] **Step 2: Run**

Run: `pytest tests/test_ndf_starlink_ingest.py -v`
Expected: PASS. If the example file's actual top-level structure differs from "single NDF named BIAS1," `_read_auto_detect` may need to handle the container case — see Task 15's container-walking logic.

- [ ] **Step 3: Commit**

```bash
git add tests/test_ndf_starlink_ingest.py
git commit -m "Add Starlink-NDF ingest test against example.sdf (DM-54817)"
```

---

### Task 22: Cross-archive consistency tests

**Files:**
- Modify: `tests/test_image.py` (or wherever cross-archive comparison lives)

- [ ] **Step 1: Add the test**

In `tests/test_image.py`, inside `ImageTestCase`:

```python
    def test_fits_ndf_consistency(self):
        """Writing via FITS and via NDF, then reading back, produces equal Images."""
        rng = np.random.default_rng(321)
        image = Image(
            rng.normal(100.0, 8.0, size=(60, 80)),
            dtype=np.float64,
            unit=u.nJy,
            start=(0, 0),
        )
        with RoundtripFits(self, image) as fits_rt, RoundtripNdf(self, image) as ndf_rt:
            assert_images_equal(self, image, fits_rt.result)
            assert_images_equal(self, image, ndf_rt.result)
            assert_images_equal(self, fits_rt.result, ndf_rt.result)
```

In `tests/test_masked_image.py`, inside `MaskedImageTestCase`:

```python
    def test_fits_ndf_consistency(self):
        """FITS and NDF backends produce equal MaskedImages on round-trip."""
        with RoundtripFits(self, self.masked_image) as fits_rt, \
             RoundtripNdf(self, self.masked_image) as ndf_rt:
            assert_masked_images_equal(self, self.masked_image, fits_rt.result, expect_view=False)
            assert_masked_images_equal(self, self.masked_image, ndf_rt.result, expect_view=False)
            assert_masked_images_equal(self, fits_rt.result, ndf_rt.result, expect_view=False)
```

(For `tests/test_visit_image.py`, add an analogous test using the file's existing `self.visit_image` fixture.)

- [ ] **Step 2: Run all newly-touched tests**

Run: `pytest tests/test_image.py tests/test_masked_image.py tests/test_visit_image.py tests/test_ndf_layout.py tests/test_ndf_starlink_ingest.py tests/test_ndf_hds.py tests/test_ndf_common.py tests/test_ndf_output_archive.py tests/test_ndf_input_archive.py -v`
Expected: all PASS.

- [ ] **Step 3: Run the whole test suite**

Run: `pytest tests/ -v`
Expected: all PASS — no regressions in existing FITS or JSON archive tests.

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "Add cross-archive consistency tests for FITS vs NDF (DM-54817)"
```

---

## Notes for the implementer

- **`_archive_default_name` discovery (Task 8 step 1):** before locking in routing names, grep `python/lsst/images/_image.py`, `_masked_image.py`, `_visit_image.py` for what `serialize()` actually passes to `add_array` / `serialize_frame_set`. The plan assumes `"image"`, `"variance"`, `"mask"`, `"projection"`. If the code uses different strings, substitute throughout `_output_archive.py`.

- **WCS in auto-detect (Task 15):** the v1 plan deliberately drops WCS in the auto-detect (Starlink-only) path with a logged warning, because building a `Projection` from a bare AST `FrameSet` requires picking the right constructor in `_transforms/_projection.py` and that's left for a follow-up. The symmetric write→read path preserves WCS faithfully via the Pydantic tree's `serialize_frame_set`/`get_frame_set` machinery, so this v1 cut only affects ingest of files we didn't write.

- **`example.sdf` actual top-level shape (Task 5 step 3 + Task 21):** the example file may have a single top-level NDF, OR it may wrap a top-level container with one NDF child named `BIAS1`. The `_read_auto_detect` function in Task 15 handles both cases; the test in Task 21 may need a slight adjustment depending on which is real. Use the Task 5 sanity tests (which traverse the actual file) as the source of truth.

- **TDD discipline:** every implementation step has a failing test BEFORE the code that makes it pass. Don't merge tests + code in one step.

- **Frequent commits:** each task ends in a commit. Don't bundle multiple tasks into one commit even if they're small — easier to bisect and review.

- **Don't add features beyond the spec:** the spec's "Deferred to later tickets" list is the floor for what NOT to build. Especially: no IRQ writing, no compression beyond pass-through, no fsspec-direct h5py reads, no ColorImage support.

