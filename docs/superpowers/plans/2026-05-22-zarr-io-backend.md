# Zarr I/O Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `lsst.images.zarr` subpackage that reads and writes Zarr v3 archives following the revised design at `docs/superpowers/specs/2026-05-22-zarr-io-design.md` — xarray/CF-shaped at the root with OME-NGFF v0.5 metadata as a discoverability layer on top, supporting every image type the FITS/JSON/NDF backends support, with cloud-friendly chunking and lazy subset reads that only fetch the chunks they touch.

**Architecture:** Mirrors the NDF backend. A Python intermediate representation (`ZarrDocument`/`ZarrGroup`/`ZarrArray`) holds the on-disk layout independently of `zarr-python`. The IR holds **lazy `zarr.Array` handles, never materialized `numpy` arrays** — so `from_zarr()` opens groups without reading bytes, and `InputArchive.get_array(model, slices=...)` passes slices through to the lazy handle. Writes use a two-pass model: `obj.serialize(archive)` populates the IR, then `__exit__` materializes it via the configured `zarr.storage.Store`. Stores are selected from a `ResourcePath` URI: `*.zarr.zip` → `ZipStore`, remote URIs → `FsspecStore`, otherwise `LocalStore`. **No stacking, no JSON-pointer rewrites, no compound source URLs** — each `add_array(name)` call lands at the zarr path equal to `name`.

**Tech Stack:** `zarr >= 3.0`, `numcodecs` (already pulled by zarr), `fsspec` (already a dependency), `lsst.resources.ResourcePath` (already a dependency), `pydantic >= 2.12`, `numpy >= 2.0`. Reuses `lsst.images.serialization` ABCs and tree models. Optional install via `pip install lsst-images[zarr]`.

**Critical invariants** — these are pinned by tests in this plan:

1. **Lazy reads everywhere.** `ZarrArray.data` is one of `np.ndarray` (staged for write) or `zarr.Array` (read-side handle). `from_zarr` never reads chunk bytes. `InputArchive.get_array(model, slices=...)` forwards `slices` straight to the lazy handle. Pinned by `_CountingStore` regression test in Task 3.2.
2. **Aligned chunks across siblings.** `image`, `variance`, and `mask` share spatial chunk shape. The output archive derives `variance`/`mask` chunks from `image`'s chunk shape when not explicitly overridden. Pinned by Task 2.5.
3. **Affine residual validator.** Before emitting an OME `coordinateTransformations` block, the layout layer samples residuals on an 11×11 grid; if max pixel-equivalent residual exceeds 1.0 pixel, the block is dropped and `lsst.wcs_simplified_dropped: true` is set. The AST string at `wcs_ast` is always authoritative. Pinned by Task 2.4.
4. **No byte duplication.** ColorImage channels are recursive sub-archives, not stacked. CellCoadd PSF is whatever shape `serialize` natively emits — typically 4-D `(Cy, Cx, Py, Px)`. There is no fixup pass that copies or re-shapes data.

---

## File Structure

```
python/lsst/images/zarr/
├── __init__.py          guarded `import zarr`; re-exports public API
├── _common.py           ZarrPointerModel, namespace constants
│                         (LSST_NS / OME_NS / LSST_VERSION / OME_VERSION),
│                         ZarrCompressionOptions, mask-dtype-for-plane-count,
│                         path helpers (no JSON-pointer mapping table —
│                         every name maps to its literal path now)
├── _model.py            IR: ZarrAttributes, ZarrArray (lazy-handle backed),
│                         ZarrGroup, ZarrDocument, OME/CF helpers
│                         (OmeMultiscale, OmeOmeroChannel,
│                         CfFlagAttributes, build_image_array_attrs)
├── _layout.py           Layout rules: axes per archive class,
│                         chunk derivation (incl. cell-aligned for CellCoadd
│                         and aligned-with-image for variance/mask),
│                         affine extraction + residual validator,
│                         OME multiscale block construction,
│                         CF flag-attrs construction from MaskSchema
├── _store.py            URI → zarr.storage.Store wrapper:
│                         *.zarr.zip → ZipStore, http(s)/s3/gs → FsspecStore,
│                         local → LocalStore. Honors create-only mode.
├── _output_archive.py   ZarrOutputArchive (populates IR) and write() helper
├── _input_archive.py    ZarrInputArchive (reads IR lazily) and read() helper

tests/
├── test_zarr_common.py            constants, helpers, ZarrCompressionOptions,
│                                   mask-dtype-for-plane-count
├── test_zarr_model.py             IR round-trip via in-memory MemoryStore,
│                                   lazy invariant on from_zarr
├── test_zarr_layout.py            axes per archive class, chunk derivation,
│                                   CF flag-attrs construction,
│                                   affine residual validator behaviour
├── test_zarr_store.py             URI dispatch, create-only refusal
├── test_zarr_output_archive.py    write paths inspected against IR for
│                                   every supported archive class
├── test_zarr_input_archive.py     read paths + lazy subset assertion
│                                   (_CountingStore), error taxonomy,
│                                   opaque-metadata round-trip
├── test_zarr_round_trip.py        full write→read for every type
├── test_zarr_cross_format.py      FITS↔Zarr opaque-metadata round-trip
├── test_zarr_xarray_interop.py    xr.open_zarr returns Dataset with
│                                   image/variance/mask data variables
├── test_zarr_ome_compliance.py    ngff-validator (skipped if absent)
└── test_zarr_external_reader.py   ome-zarr-py sanity (skipped if absent)
```

The split mirrors the NDF backend exactly: `_model.py` is pure data; `_output_archive.py` and `_input_archive.py` only translate between the IR and the abstract archive interface; `_layout.py` holds every per-archive-class decision so individual `add_array` calls stay generic.

---

## Phase 1 — Skeleton, `_common.py`, and IR (no I/O yet)

This phase produces the IR and constants in isolation. The IR round-trips through an in-memory zarr `MemoryStore` so the shape of what later phases will produce is pinned before any archive code is written.

### Task 1.1: Create the package skeleton

**Files:**
- Create: `python/lsst/images/zarr/__init__.py`
- Modify: `pyproject.toml` (add `zarr` extra after the existing `ndf` extra at line 55)

- [ ] **Step 1: Add the optional dependency**

In `pyproject.toml`, immediately after the `ndf` extra (around line 55), add:

```toml
# Add feature for Zarr v3 read/write support.
zarr = ["zarr >= 3.0"]
```

- [ ] **Step 2: Create the package `__init__.py` with a guarded import**

Create `python/lsst/images/zarr/__init__.py`:

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

"""Zarr v3 archive backend for `lsst.images`.

Files written by this archive are xarray/CF-shaped at the root
(``image`` / ``variance`` / ``mask`` as siblings sharing ``(y, x)``
dimensions, CF ``flag_masks`` / ``flag_meanings`` on the mask) with
OME-NGFF v0.5 multiscales metadata as a discoverability layer
pointing at the same ``image`` array. The same bytes are visible to
``xarray``, GDAL's Zarr driver, and OME-Zarr tooling like ``napari``
and ``ome-zarr-py``.

Default chunk geometry is tile-aligned (~1024×1024 for plain images,
``cell_shape`` for ``CellCoadd``). Sharding (zarr v3 native) is
enabled by default with a tunable shard size to keep object counts
manageable on S3/GCS. Both ``DirectoryStore`` and ``ZipStore`` are
supported; the choice is driven by URI shape (``*.zarr.zip`` →
``ZipStore``, otherwise directory). Remote URIs go through
`lsst.resources.ResourcePath` and `fsspec`.
"""

try:
    import zarr  # noqa: F401
except ImportError as e:
    raise ImportError(
        "lsst.images.zarr requires the optional 'zarr' package (>=3.0). "
        "Install it directly or via 'pip install lsst-images[zarr]'."
    ) from e

# Phase 1 has no public archive API yet. Re-exports are added in later phases.
```

- [ ] **Step 3: Verify the guarded import works**

Run: `python -c "import lsst.images.zarr"`
Expected: no output (success), or a clear ImportError pointing at the `[zarr]` extra if `zarr` is not installed.

- [ ] **Step 4: Commit**

```bash
git add python/lsst/images/zarr/__init__.py pyproject.toml
git commit -m "feat: add lsst.images.zarr package skeleton with guarded import"
```

### Task 1.2: `_common.py` — namespaces, `ZarrPointerModel`, `ZarrCompressionOptions`, mask-dtype helper

**Files:**
- Create: `python/lsst/images/zarr/_common.py`
- Test: `tests/test_zarr_common.py`

`_common.py` carries:

- Namespace constants `LSST_NS = "lsst"`, `OME_NS = "ome"`, version integers `LSST_VERSION = 1`, `OME_VERSION = "0.5"`.
- `ZarrPointerModel` — Pydantic model holding an absolute zarr path.
- `ZarrCompressionOptions` — dataclass with `codec`, `cname`, `clevel`, `shuffle`. Provides `default_for_dtype(dtype)` returning byte-shuffle for floats, bit-shuffle for ints/masks.
- `mask_dtype_for_plane_count(n)` — picks the smallest unsigned integer that holds `n` planes; raises if `n > 64`.
- `archive_path_to_zarr_path(archive_path)` — translates an empty archive path to `/tree`; non-empty paths are kept verbatim under their natural path. **There is no JSON-pointer mapping table.** `name="image"` lands at `/image`; `name="mask"` at `/mask`; nested `name="red/image"` at `/red/image`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_zarr_common.py`:

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

import numpy as np

try:
    from lsst.images.zarr._common import (
        LSST_NS,
        LSST_VERSION,
        OME_NS,
        OME_VERSION,
        ZarrCompressionOptions,
        ZarrPointerModel,
        archive_path_to_zarr_path,
        mask_dtype_for_plane_count,
    )

    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class CommonTestCase(unittest.TestCase):
    def test_pointer_round_trips(self) -> None:
        original = ZarrPointerModel(path="/lsst/psf/tree")
        recovered = ZarrPointerModel.model_validate_json(original.model_dump_json())
        self.assertEqual(recovered, original)

    def test_constants(self) -> None:
        self.assertEqual(LSST_NS, "lsst")
        self.assertEqual(OME_NS, "ome")
        self.assertEqual(OME_VERSION, "0.5")
        self.assertGreaterEqual(LSST_VERSION, 1)

    def test_archive_path_translation(self) -> None:
        # Empty archive path -> the canonical root-level JSON tree.
        self.assertEqual(archive_path_to_zarr_path(""), "/tree")
        # Non-empty archive paths are kept verbatim.
        self.assertEqual(archive_path_to_zarr_path("/image"), "/image")
        self.assertEqual(archive_path_to_zarr_path("image"), "/image")
        self.assertEqual(archive_path_to_zarr_path("/red/image"), "/red/image")
        self.assertEqual(archive_path_to_zarr_path("/psf"), "/psf")

    def test_compression_defaults(self) -> None:
        floats = ZarrCompressionOptions.default_for_dtype("float32")
        self.assertEqual(floats.codec, "blosc")
        self.assertEqual(floats.shuffle, "shuffle")
        ints = ZarrCompressionOptions.default_for_dtype("uint8")
        self.assertEqual(ints.shuffle, "bitshuffle")

    def test_mask_dtype_picks_smallest_fit(self) -> None:
        self.assertEqual(mask_dtype_for_plane_count(1), np.dtype("uint8"))
        self.assertEqual(mask_dtype_for_plane_count(8), np.dtype("uint8"))
        self.assertEqual(mask_dtype_for_plane_count(9), np.dtype("uint16"))
        self.assertEqual(mask_dtype_for_plane_count(16), np.dtype("uint16"))
        self.assertEqual(mask_dtype_for_plane_count(17), np.dtype("uint32"))
        self.assertEqual(mask_dtype_for_plane_count(32), np.dtype("uint32"))
        self.assertEqual(mask_dtype_for_plane_count(33), np.dtype("uint64"))
        self.assertEqual(mask_dtype_for_plane_count(64), np.dtype("uint64"))

    def test_mask_dtype_refuses_more_than_64_planes(self) -> None:
        with self.assertRaisesRegex(ValueError, "supports up to 64"):
            mask_dtype_for_plane_count(65)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_zarr_common.py -v`
Expected: FAIL — `ImportError` on `lsst.images.zarr._common`.

- [ ] **Step 3: Write `_common.py`**

Create `python/lsst/images/zarr/_common.py`:

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

__all__ = (
    "LSST_NS",
    "LSST_VERSION",
    "OME_NS",
    "OME_VERSION",
    "ZarrCompressionOptions",
    "ZarrPointerModel",
    "archive_path_to_zarr_path",
    "mask_dtype_for_plane_count",
)

from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
import pydantic

LSST_NS = "lsst"
"""Top-level zarr-attributes namespace key for LSST extensions."""

OME_NS = "ome"
"""Top-level zarr-attributes namespace key for OME-NGFF metadata."""

OME_VERSION = "0.5"
"""OME-Zarr / NGFF version this backend writes."""

LSST_VERSION = 1
"""Schema version of the ``lsst:`` extension this backend writes.

Readers refuse versions newer than they understand. Bump on
backwards-incompatible changes to the on-disk layout.
"""


class ZarrPointerModel(pydantic.BaseModel):
    """Reference to a zarr archive sub-tree by absolute zarr path.

    Used by `ZarrOutputArchive` / `ZarrInputArchive` to point to
    sub-trees that have been hoisted out of the main JSON tree into
    separate zarr arrays. The path is interpreted relative to the
    archive root, e.g. ``"/lsst/psf/tree"``.
    """

    path: str
    """Absolute zarr path (e.g. ``/lsst/psf/tree``)."""


@dataclass(frozen=True)
class ZarrCompressionOptions:
    """Per-array zarr v3 codec configuration.

    The default codec stack is ``bytes -> blosc(zstd, clevel=5)`` with
    byte-shuffle for floats and bit-shuffle for integers (and masks).
    All defaults are overridable per-array via the ``compression``
    keyword to ``write()``.
    """

    codec: str = "blosc"
    cname: str = "zstd"
    clevel: int = 5
    shuffle: str = "shuffle"  # 'shuffle' (byte) or 'bitshuffle' or 'noshuffle'

    DEFAULT_FLOAT: ClassVar[Self]
    DEFAULT_INT: ClassVar[Self]

    @classmethod
    def default_for_dtype(cls, dtype: str | np.dtype) -> Self:
        """Return the default codec stack for a numpy dtype."""
        kind = np.dtype(dtype).kind
        # 'u' (unsigned int), 'i' (signed int), 'b' (bool) -> bit-shuffle.
        if kind in ("u", "i", "b"):
            return cls.DEFAULT_INT
        return cls.DEFAULT_FLOAT


ZarrCompressionOptions.DEFAULT_FLOAT = ZarrCompressionOptions(shuffle="shuffle")
ZarrCompressionOptions.DEFAULT_INT = ZarrCompressionOptions(shuffle="bitshuffle")


def archive_path_to_zarr_path(archive_path: str) -> str:
    """Translate a serialization archive path to its zarr path.

    The empty archive path maps to the root-level JSON tree at
    ``/tree``. Non-empty archive paths are kept verbatim (with a
    leading slash). The v1 design's JSON-pointer mapping table is
    intentionally absent: arrays land where their archive name says
    they do.
    """
    if not archive_path:
        return "/tree"
    stripped = archive_path.strip("/")
    return f"/{stripped}"


def mask_dtype_for_plane_count(n_planes: int) -> np.dtype:
    """Pick the smallest unsigned-integer dtype that holds ``n_planes`` bits.

    Returns ``uint8`` for ≤8 planes, ``uint16`` for ≤16, ``uint32``
    for ≤32, ``uint64`` for ≤64. Raises `ValueError` for >64 planes;
    a 3-D fallback for that case is tracked as a follow-up.
    """
    if n_planes <= 0:
        raise ValueError(f"n_planes must be positive, got {n_planes}.")
    if n_planes <= 8:
        return np.dtype("uint8")
    if n_planes <= 16:
        return np.dtype("uint16")
    if n_planes <= 32:
        return np.dtype("uint32")
    if n_planes <= 64:
        return np.dtype("uint64")
    raise ValueError(
        f"Mask has {n_planes} planes; v1 supports up to 64. "
        f"3-D fallback is a follow-up."
    )
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_zarr_common.py -v`
Expected: PASS — 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/zarr/_common.py tests/test_zarr_common.py
git commit -m "feat: add ZarrPointerModel, ZarrCompressionOptions, mask-dtype helper"
```

### Task 1.3: IR — `ZarrAttributes` and `ZarrArray` with lazy backing

**Files:**
- Create: `python/lsst/images/zarr/_model.py`
- Test: `tests/test_zarr_model.py`

This task introduces the IR types whose **lazy-array invariant** is the heart of the efficient subsetting story. `ZarrArray.data` is one of:

- `numpy.ndarray` — staged for write
- `zarr.Array` — read from a store, **never sliced eagerly**

A read of a remote VisitImage opens its `zarr.Array` handle through `from_zarr`. Subsequent slicing (in `InputArchive.get_array(model, slices=...)`) goes straight to that handle, so only the chunks intersecting the slice are downloaded.

`ZarrAttributes` separates the `lsst:` and `ome:` namespaces (each gets its `version` field stamped automatically on `dump`) and preserves unknown keys for forward compatibility. Plain CF / xarray attributes like `_ARRAY_DIMENSIONS`, `flag_masks`, `flag_meanings`, `units` live in a third namespace called `extra` that round-trips verbatim — they're written at the top level of `zarr.json` `attributes` (no `lsst:` or `ome:` wrapper).

- [ ] **Step 1: Write the failing test**

Create `tests/test_zarr_model.py`:

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

import numpy as np

try:
    import zarr

    from lsst.images.zarr._common import LSST_NS, LSST_VERSION, OME_NS, OME_VERSION
    from lsst.images.zarr._model import ZarrArray, ZarrAttributes

    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrAttributesTestCase(unittest.TestCase):
    def test_dump_separates_namespaces(self) -> None:
        attrs = ZarrAttributes()
        attrs.lsst["archive_class"] = "MaskedImage"
        attrs.ome["multiscales"] = [{"name": "image"}]
        attrs.extra["_ARRAY_DIMENSIONS"] = ["y", "x"]
        attrs.extra["units"] = "adu"
        dumped = attrs.dump()
        self.assertEqual(dumped[LSST_NS]["archive_class"], "MaskedImage")
        self.assertEqual(dumped[LSST_NS]["version"], LSST_VERSION)
        self.assertEqual(dumped[OME_NS]["multiscales"], [{"name": "image"}])
        self.assertEqual(dumped[OME_NS]["version"], OME_VERSION)
        # CF / xarray attrs sit at the top level, not inside lsst: or ome:.
        self.assertEqual(dumped["_ARRAY_DIMENSIONS"], ["y", "x"])
        self.assertEqual(dumped["units"], "adu")

    def test_load_preserves_unknown_keys(self) -> None:
        # Forward compatibility: unknown lsst.* keys must survive a
        # load -> dump round-trip.
        raw = {
            LSST_NS: {
                "version": LSST_VERSION,
                "archive_class": "Image",
                "future_thing": {"x": 1},
            },
            OME_NS: {"version": OME_VERSION, "multiscales": []},
            "_ARRAY_DIMENSIONS": ["y", "x"],
            "units": "adu",
        }
        attrs = ZarrAttributes.load(raw)
        dumped = attrs.dump()
        self.assertEqual(dumped[LSST_NS]["future_thing"], {"x": 1})
        self.assertEqual(dumped["units"], "adu")


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrArrayTestCase(unittest.TestCase):
    def test_lazy_data_after_from_zarr(self) -> None:
        store = zarr.storage.MemoryStore()
        root = zarr.create_group(store=store, zarr_format=3)
        zarr_array = root.create_array(
            name="image", shape=(8, 8), chunks=(4, 4), dtype="float32"
        )
        zarr_array[:] = np.arange(64, dtype=np.float32).reshape(8, 8)

        ir_array = ZarrArray.from_zarr(zarr_array)
        # Lazy invariant: data is the zarr.Array handle, not numpy.
        self.assertIsInstance(ir_array.data, zarr.Array)
        self.assertNotIsInstance(ir_array.data, np.ndarray)
        self.assertEqual(ir_array.shape, (8, 8))
        self.assertEqual(str(ir_array.dtype), "float32")

    def test_subset_does_not_materialize_full_array(self) -> None:
        store = _CountingStore()
        root = zarr.create_group(store=store, zarr_format=3)
        zarr_array = root.create_array(
            name="image", shape=(16, 16), chunks=(4, 4), dtype="int32"
        )
        zarr_array[:] = np.arange(256, dtype=np.int32).reshape(16, 16)
        store.reads = 0  # reset after the write phase

        ir_array = ZarrArray.from_zarr(zarr_array)
        # Reading shape / dtype must not fetch any chunk data.
        self.assertEqual(ir_array.shape, (16, 16))
        self.assertEqual(store.reads, 0)

        subset = ir_array.read(slices=(slice(0, 4), slice(0, 4)))
        self.assertEqual(subset.shape, (4, 4))
        np.testing.assert_array_equal(subset, np.arange(256).reshape(16, 16)[:4, :4])
        # 16 chunks total in the array; we should have touched far fewer.
        self.assertLess(store.reads, 16)

    def test_staged_numpy_array_is_eager(self) -> None:
        data = np.arange(12, dtype=np.float64).reshape(3, 4)
        ir_array = ZarrArray(data=data)
        self.assertIs(ir_array.data, data)
        self.assertEqual(ir_array.shape, (3, 4))


class _CountingStore(zarr.storage.MemoryStore if HAVE_ZARR else object):
    """A MemoryStore that counts get() calls."""

    def __init__(self) -> None:
        super().__init__()
        self.reads = 0

    async def get(self, key, prototype, byte_range=None):  # type: ignore[override]
        self.reads += 1
        return await super().get(key, prototype, byte_range)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_zarr_model.py -v`
Expected: FAIL — `ImportError` on `lsst.images.zarr._model`.

- [ ] **Step 3: Write `_model.py` (initial portion: `ZarrAttributes` and `ZarrArray`)**

Create `python/lsst/images/zarr/_model.py`:

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

"""Python intermediate representation for zarr / xarray-CF / OME-NGFF content.

The IR is the source of truth for what gets written. ``ZarrOutputArchive``
populates a `ZarrDocument`; on context-manager exit, `to_zarr` materializes
it through a configured ``zarr.storage.Store``.

Reads invert that flow: ``ZarrInputArchive`` opens the store and calls
`ZarrDocument.from_zarr`, which builds the IR around **lazy** ``zarr.Array``
handles. No array bytes are read until a caller asks for them via
`ZarrArray.read`, which forwards slices straight to the underlying handle.
This keeps subset reads of remote files cheap: only the chunks intersecting
the requested slice are fetched.
"""

from __future__ import annotations

__all__ = (
    "ZarrArray",
    "ZarrAttributes",
)

from dataclasses import dataclass, field
from types import EllipsisType
from typing import Any, Self

import numpy as np
import zarr

from ._common import LSST_NS, LSST_VERSION, OME_NS, OME_VERSION, ZarrCompressionOptions


@dataclass
class ZarrAttributes:
    """Namespaced attributes attached to a `ZarrGroup` or `ZarrArray`.

    Three namespaces:

    - ``lsst`` — LSST extensions (always emitted with a ``version`` key).
    - ``ome`` — OME-NGFF (emitted only when non-empty).
    - ``extra`` — flat top-level keys for CF / xarray conventions
      (``_ARRAY_DIMENSIONS``, ``flag_masks``, ``flag_meanings``,
      ``flag_descriptions``, ``units``, ``long_name``, …). These live at
      the top of ``zarr.json`` ``attributes`` so xarray and CF tooling
      see them without unwrapping a namespace.
    """

    lsst: dict[str, Any] = field(default_factory=dict)
    ome: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def dump(self) -> dict[str, Any]:
        """Return the raw mapping zarr-python writes to ``zarr.json``."""
        out: dict[str, Any] = dict(self.extra)
        # lsst is always present so readers can dispatch on lsst.archive_class.
        out[LSST_NS] = {"version": LSST_VERSION, **self.lsst}
        if self.ome:
            out[OME_NS] = {"version": OME_VERSION, **self.ome}
        return out

    @classmethod
    def load(cls, raw: dict[str, Any]) -> Self:
        """Construct from a raw attributes mapping read from zarr."""
        lsst = dict(raw.get(LSST_NS, {}))
        lsst.pop("version", None)  # version implicit in the namespace
        ome = dict(raw.get(OME_NS, {}))
        ome.pop("version", None)
        extra = {k: v for k, v in raw.items() if k not in (LSST_NS, OME_NS)}
        return cls(lsst=lsst, ome=ome, extra=extra)


@dataclass
class ZarrArray:
    """An IR node holding either staged numpy data or a lazy zarr handle.

    Parameters
    ----------
    data
        Either a ``numpy.ndarray`` (when staged for write by the output
        archive) or a ``zarr.Array`` (when read by the input archive).
        The two forms never mix in a single instance.
    chunks
        Per-axis chunk shape. ``None`` lets `to_zarr` derive a default
        from the array shape (~1024 per axis for plain images).
    shards
        Per-axis shard shape (zarr v3 native). ``None`` lets `to_zarr`
        derive a default of 4× the chunk shape per axis when the
        resulting shard exceeds 1 MiB.
    compression
        Codec configuration. ``None`` falls back to
        `ZarrCompressionOptions.default_for_dtype`.
    attributes
        Namespaced attributes for this array's ``zarr.json``.
    """

    data: np.ndarray | zarr.Array
    chunks: tuple[int, ...] | None = None
    shards: tuple[int, ...] | None = None
    compression: ZarrCompressionOptions | None = None
    attributes: ZarrAttributes = field(default_factory=ZarrAttributes)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.data.dtype)

    @classmethod
    def from_zarr(cls, zarr_array: zarr.Array) -> Self:
        """Wrap an open ``zarr.Array`` without reading its data."""
        attrs = ZarrAttributes.load(dict(zarr_array.attrs))
        return cls(
            data=zarr_array,
            chunks=tuple(zarr_array.chunks),
            attributes=attrs,
        )

    def read(self, *, slices: tuple[slice, ...] | EllipsisType = ...) -> np.ndarray:
        """Materialize this array (or a slice of it) into numpy.

        For a `ZarrArray` backed by a lazy handle, this is the only
        place that touches array bytes. ``slices`` is forwarded straight
        to the handle so only chunks intersecting the slice are fetched.
        """
        if isinstance(self.data, np.ndarray):
            return self.data if slices is ... else self.data[slices]
        return self.data[...] if slices is ... else self.data[slices]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_zarr_model.py -v`
Expected: PASS — 5 tests pass; the `_CountingStore` test confirms a 4×4 subset of a 16×16 / chunks=(4,4) array touches strictly fewer than 16 chunk reads.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/zarr/_model.py tests/test_zarr_model.py
git commit -m "feat: add ZarrAttributes and ZarrArray IR with lazy zarr.Array backing"
```

### Task 1.4: IR — `ZarrGroup`, `ZarrDocument`, store materialization

**Files:**
- Modify: `python/lsst/images/zarr/_model.py` (append `ZarrGroup`, `ZarrDocument`, helpers)
- Modify: `tests/test_zarr_model.py` (add round-trip test through `MemoryStore`)

This task gives the IR a full tree shape and the bidirectional `to_zarr` / `from_zarr` materialization. The round-trip test pins the lazy invariant: after `from_zarr` on a freshly-opened store, every `ZarrArray.data` is a `zarr.Array`, not a materialized ndarray.

- [ ] **Step 1: Write the failing test (extend `test_zarr_model.py`)**

Append before the `if __name__` guard:

```python
@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrDocumentTestCase(unittest.TestCase):
    def test_round_trip_through_memory_store(self) -> None:
        from lsst.images.zarr._model import ZarrArray, ZarrDocument, ZarrGroup

        # Build a flat IR: image, variance, mask siblings at root.
        doc = ZarrDocument(root=ZarrGroup())
        doc.root.attributes.lsst["archive_class"] = "MaskedImage"
        doc.root.attributes.lsst["tree"] = "tree"

        image = ZarrArray(data=np.ones((4, 4), dtype="float32"))
        image.attributes.extra["_ARRAY_DIMENSIONS"] = ["y", "x"]
        doc.root.arrays["image"] = image

        mask = ZarrArray(data=np.zeros((4, 4), dtype="uint8"))
        mask.attributes.extra["_ARRAY_DIMENSIONS"] = ["y", "x"]
        mask.attributes.extra["flag_masks"] = [1, 2]
        mask.attributes.extra["flag_meanings"] = "BAD SAT"
        doc.root.arrays["mask"] = mask

        # Stub a 1-D uint8 'tree' array (JSON bytes).
        doc.root.arrays["tree"] = ZarrArray(
            data=np.frombuffer(b"{}", dtype=np.uint8)
        )

        store = zarr.storage.MemoryStore()
        doc.to_zarr(store)

        # Reload and verify lazy invariant on every array.
        recovered = ZarrDocument.from_zarr(store)
        self.assertIsInstance(recovered.root.arrays["image"].data, zarr.Array)
        self.assertIsInstance(recovered.root.arrays["mask"].data, zarr.Array)
        self.assertEqual(
            recovered.root.attributes.lsst["archive_class"], "MaskedImage"
        )
        # CF flag attrs round-trip via the extra namespace.
        self.assertEqual(
            recovered.root.arrays["mask"].attributes.extra["flag_meanings"],
            "BAD SAT",
        )
        # xarray dims round-trip.
        self.assertEqual(
            recovered.root.arrays["image"].attributes.extra["_ARRAY_DIMENSIONS"],
            ["y", "x"],
        )
        # Subset reads still go through the lazy handle.
        np.testing.assert_array_equal(
            recovered.root.arrays["image"].read(), np.ones((4, 4), dtype="float32")
        )

    def test_get_walks_paths(self) -> None:
        from lsst.images.zarr._model import ZarrArray, ZarrDocument, ZarrGroup

        doc = ZarrDocument(root=ZarrGroup())
        doc.root.arrays["image"] = ZarrArray(data=np.zeros((2, 2), dtype="float32"))
        red = doc.root.ensure_group("/red")
        red.arrays["image"] = ZarrArray(data=np.ones((2, 2), dtype="float32"))

        # Absolute and relative paths.
        self.assertIs(doc.root.get("/image"), doc.root.arrays["image"])
        self.assertIs(doc.root.get("image"), doc.root.arrays["image"])
        self.assertIs(doc.root.get("/red/image"), red.arrays["image"])
        self.assertIs(doc.root.get("/"), doc.root)

        with self.assertRaises(KeyError):
            doc.root.get("/missing")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_zarr_model.py::ZarrDocumentTestCase -v`
Expected: FAIL — `ImportError` for `ZarrGroup` / `ZarrDocument`.

- [ ] **Step 3: Append `ZarrGroup`, `ZarrDocument`, helpers**

Update the `__all__` and append to `python/lsst/images/zarr/_model.py`:

```python
__all__ = (
    "ZarrArray",
    "ZarrAttributes",
    "ZarrDocument",
    "ZarrGroup",
)


@dataclass
class ZarrGroup:
    """A zarr group: nested groups, arrays, and namespaced attributes."""

    groups: dict[str, "ZarrGroup"] = field(default_factory=dict)
    arrays: dict[str, ZarrArray] = field(default_factory=dict)
    attributes: ZarrAttributes = field(default_factory=ZarrAttributes)

    def get(self, path: str) -> "ZarrGroup | ZarrArray":
        """Return a child by absolute or relative zarr path."""
        if path in ("", "/"):
            return self
        parts = [p for p in path.strip("/").split("/") if p]
        cursor: ZarrGroup | ZarrArray = self
        for part in parts:
            if not isinstance(cursor, ZarrGroup):
                raise KeyError(path)
            if part in cursor.arrays:
                cursor = cursor.arrays[part]
            elif part in cursor.groups:
                cursor = cursor.groups[part]
            else:
                raise KeyError(path)
        return cursor

    def ensure_group(self, path: str) -> "ZarrGroup":
        """Return or create a sub-group at ``path``."""
        if path in ("", "/"):
            return self
        parts = [p for p in path.strip("/").split("/") if p]
        cursor = self
        for part in parts:
            if part in cursor.arrays:
                raise KeyError(f"{part!r} already exists as an array.")
            if part not in cursor.groups:
                cursor.groups[part] = ZarrGroup()
            cursor = cursor.groups[part]
        return cursor


@dataclass
class ZarrDocument:
    """A complete zarr archive root."""

    root: ZarrGroup = field(default_factory=ZarrGroup)

    @classmethod
    def from_zarr(cls, store: zarr.storage.Store) -> Self:
        """Open ``store`` and build a lazy IR view of its contents."""
        zarr_root = zarr.open_group(store=store, mode="r", zarr_format=3)
        return cls(root=_group_from_zarr(zarr_root))

    def to_zarr(self, store: zarr.storage.Store) -> None:
        """Materialize this IR into ``store`` (which must be empty)."""
        zarr_root = zarr.create_group(store=store, zarr_format=3, overwrite=False)
        _group_to_zarr(self.root, zarr_root)


def _group_from_zarr(zarr_group: zarr.Group) -> ZarrGroup:
    """Build a lazy `ZarrGroup` IR from an open ``zarr.Group``."""
    ir = ZarrGroup(attributes=ZarrAttributes.load(dict(zarr_group.attrs)))
    for name, child in zarr_group.members():
        if isinstance(child, zarr.Array):
            ir.arrays[name] = ZarrArray.from_zarr(child)
        else:
            ir.groups[name] = _group_from_zarr(child)
    return ir


def _group_to_zarr(ir: ZarrGroup, zarr_group: zarr.Group) -> None:
    """Write a `ZarrGroup` IR into an open ``zarr.Group``."""
    if dumped := ir.attributes.dump():
        zarr_group.update_attributes(dumped)
    for name, sub in ir.groups.items():
        sub_zarr = zarr_group.create_group(name)
        _group_to_zarr(sub, sub_zarr)
    for name, array in ir.arrays.items():
        if not isinstance(array.data, np.ndarray):
            raise TypeError(
                f"Cannot write ZarrArray at {name!r}: data is a lazy zarr.Array, "
                "not numpy. Read it first or pass a fresh numpy array."
            )
        chunks = array.chunks or _default_chunks(array.data.shape)
        compression = array.compression or ZarrCompressionOptions.default_for_dtype(
            str(array.dtype)
        )
        codecs = _build_codecs(compression)
        zarr_array = zarr_group.create_array(
            name=name,
            shape=array.data.shape,
            chunks=chunks,
            dtype=array.data.dtype,
            shards=array.shards,
            codecs=codecs,
        )
        zarr_array[:] = array.data
        if dumped := array.attributes.dump():
            zarr_array.update_attributes(dumped)


def _default_chunks(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Default chunk shape: min(1024, dim) per axis."""
    return tuple(min(1024, dim) for dim in shape)


def _build_codecs(options: ZarrCompressionOptions) -> list[Any]:
    """Build a zarr v3 codec stack from `ZarrCompressionOptions`."""
    from numcodecs.zarr3 import Blosc

    if options.codec != "blosc":
        raise NotImplementedError(f"Unsupported codec {options.codec!r}.")
    return [
        zarr.codecs.BytesCodec(),
        Blosc(cname=options.cname, clevel=options.clevel, shuffle=options.shuffle),
    ]
```

- [ ] **Step 4: Run all model tests**

Run: `pytest tests/test_zarr_model.py -v`
Expected: PASS — all tests pass; round-trip test confirms `.data` is a `zarr.Array` after `from_zarr`.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/zarr/_model.py tests/test_zarr_model.py
git commit -m "feat: add ZarrGroup and ZarrDocument with lazy-on-read materialization"
```

### Task 1.5: IR — OME and CF helper dataclasses

**Files:**
- Modify: `python/lsst/images/zarr/_model.py` (append OME / CF helpers)
- Modify: `tests/test_zarr_model.py` (helper-construction test)

These small dataclasses centralize the OME and CF attribute shapes so `_layout.py` can populate them without literal-dict-typo bugs.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_zarr_model.py`:

```python
@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class OmeCfHelpersTestCase(unittest.TestCase):
    def test_multiscale_emits_expected_shape(self) -> None:
        from lsst.images.zarr._model import OmeMultiscale

        m = OmeMultiscale(
            name="visitimage",
            axes=("y", "x"),
            dataset_path="image",
        )
        d = m.dump()
        self.assertEqual(d["name"], "visitimage")
        self.assertEqual(
            d["axes"],
            [
                {"name": "y", "type": "space", "unit": "pixel"},
                {"name": "x", "type": "space", "unit": "pixel"},
            ],
        )
        self.assertEqual(d["datasets"][0]["path"], "image")
        # Default coordinate transform is unit scale until a real one is set.
        self.assertEqual(
            d["datasets"][0]["coordinateTransformations"],
            [{"type": "scale", "scale": [1.0, 1.0]}],
        )

    def test_multiscale_with_affine(self) -> None:
        from lsst.images.zarr._model import OmeMultiscale

        m = OmeMultiscale(
            name="image",
            axes=("y", "x"),
            dataset_path="image",
            coordinate_transformations=[
                {"type": "scale", "scale": [0.2, 0.2]},
                {
                    "type": "affine",
                    "affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                },
            ],
        )
        d = m.dump()
        self.assertEqual(len(d["datasets"][0]["coordinateTransformations"]), 2)
        self.assertEqual(
            d["datasets"][0]["coordinateTransformations"][0]["type"], "scale"
        )

    def test_cf_flag_attributes(self) -> None:
        from lsst.images.zarr._model import CfFlagAttributes, MaskPlaneEntry

        cf = CfFlagAttributes(
            planes=[
                MaskPlaneEntry(name="BAD", bit=0, description="Bad pixel."),
                MaskPlaneEntry(name="SAT", bit=1, description="Saturated."),
                MaskPlaneEntry(name="CR", bit=2, description="Cosmic ray."),
            ]
        )
        d = cf.dump()
        self.assertEqual(d["flag_masks"], [1, 2, 4])
        self.assertEqual(d["flag_meanings"], "BAD SAT CR")
        self.assertEqual(
            d["flag_descriptions"], ["Bad pixel.", "Saturated.", "Cosmic ray."]
        )

    def test_image_array_attrs(self) -> None:
        from lsst.images.zarr._model import build_image_array_attrs

        attrs = build_image_array_attrs(axes=("y", "x"), units="adu", long_name="science image")
        self.assertEqual(attrs["_ARRAY_DIMENSIONS"], ["y", "x"])
        self.assertEqual(attrs["units"], "adu")
        self.assertEqual(attrs["long_name"], "science image")
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_zarr_model.py::OmeCfHelpersTestCase -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Append helpers to `_model.py`**

Update the `__all__` and append:

```python
__all__ = (
    "CfFlagAttributes",
    "MaskPlaneEntry",
    "OmeMultiscale",
    "OmeOmeroChannel",
    "ZarrArray",
    "ZarrAttributes",
    "ZarrDocument",
    "ZarrGroup",
    "build_image_array_attrs",
)


@dataclass
class OmeMultiscale:
    """OME-NGFF v0.5 multiscales metadata for a single-level image.

    The backend always writes one level whose ``path`` points at a
    sibling array (``image`` for typical archives). ``coordinate_transformations``
    defaults to a unit ``scale`` so the OME block is well-formed even
    when the simplified affine is dropped by the residual validator.
    """

    name: str
    axes: tuple[str, ...]
    dataset_path: str = "image"
    coordinate_transformations: list[dict[str, Any]] | None = None

    @staticmethod
    def _axis_block(name: str) -> dict[str, Any]:
        if name == "c":
            return {"name": "c", "type": "channel"}
        if name == "t":
            return {"name": "t", "type": "time"}
        return {"name": name, "type": "space", "unit": "pixel"}

    def dump(self) -> dict[str, Any]:
        ndim = len(self.axes)
        ct = self.coordinate_transformations
        if ct is None:
            ct = [{"type": "scale", "scale": [1.0] * ndim}]
        return {
            "name": self.name,
            "axes": [self._axis_block(a) for a in self.axes],
            "datasets": [
                {
                    "path": self.dataset_path,
                    "coordinateTransformations": ct,
                }
            ],
        }


@dataclass
class OmeOmeroChannel:
    """OME ``omero/channels`` entry (used only when a channel axis exists)."""

    label: str
    color: str | None = None

    def dump(self) -> dict[str, Any]:
        out: dict[str, Any] = {"label": self.label}
        if self.color is not None:
            out["color"] = self.color
        return out


@dataclass
class MaskPlaneEntry:
    """One mask-plane definition."""

    name: str
    bit: int
    description: str = ""


@dataclass
class CfFlagAttributes:
    """CF-conventions flag metadata for a 2-D packed mask array.

    Emits ``flag_masks`` (list of bit values), ``flag_meanings``
    (single space-separated string per CF), and the LSST extension
    ``flag_descriptions`` (list of human-readable strings parallel to
    ``flag_meanings``).
    """

    planes: list[MaskPlaneEntry] = field(default_factory=list)

    def dump(self) -> dict[str, Any]:
        return {
            "flag_masks": [int(1 << p.bit) for p in self.planes],
            "flag_meanings": " ".join(p.name for p in self.planes),
            "flag_descriptions": [p.description for p in self.planes],
        }

    @classmethod
    def load(cls, raw: dict[str, Any]) -> Self:
        meanings = raw.get("flag_meanings", "").split()
        masks = [int(m) for m in raw.get("flag_masks", [])]
        descriptions = list(raw.get("flag_descriptions", [""] * len(meanings)))
        planes = []
        for name, mask, desc in zip(meanings, masks, descriptions, strict=False):
            # Recover bit position from the mask value (always a power of 2).
            bit = (mask & -mask).bit_length() - 1
            planes.append(MaskPlaneEntry(name=name, bit=bit, description=desc))
        return cls(planes=planes)


def build_image_array_attrs(
    *,
    axes: tuple[str, ...],
    units: str | None = None,
    long_name: str | None = None,
) -> dict[str, Any]:
    """Build the CF / xarray attribute block for a 2-D-or-higher image array."""
    out: dict[str, Any] = {"_ARRAY_DIMENSIONS": list(axes)}
    if units is not None:
        out["units"] = units
    if long_name is not None:
        out["long_name"] = long_name
    return out
```

- [ ] **Step 4: Run all model tests**

Run: `pytest tests/test_zarr_model.py -v`
Expected: PASS — 9 tests.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/zarr/_model.py tests/test_zarr_model.py
git commit -m "feat: add OmeMultiscale, CfFlagAttributes, image-array-attrs helpers"
```

---

**End of Phase 1.** Five tasks. The IR is in place with the lazy invariant pinned by `_CountingStore`, the CF / OME helpers are unit-tested in isolation, and `ZarrAttributes` separates `lsst:` / `ome:` / top-level (`extra`) namespaces so xarray and CF tooling see flat attributes without unwrapping. Phase 2 wires `_store.py`, `_layout.py`, and `ZarrOutputArchive` against this IR for `Image` / `MaskedImage` / `VisitImage`.

## Phase 2 — Store dispatch, layout rules, and `ZarrOutputArchive` (Image / MaskedImage / VisitImage)

This phase adds enough machinery to **write** a plain `Image`, a `MaskedImage`, and a `VisitImage` to a zarr archive on disk and on a `ZipStore`. No reading yet — that lands in Phase 3 — so tests inspect the on-disk shape via `ZarrDocument.from_zarr()` directly. `ColorImage` and `CellCoadd` are deferred to Phase 4.

The output archive's `add_array(name)` method writes to a zarr path equal to `name` (after stripping the leading slash). There is **no JSON-pointer mapping table** and **no fixup pass**. Mask arrays go through a small specialization that packs a 3-D `(y, x, mask_size)` in-memory mask into the 2-D wide-integer on-disk form and attaches CF flag attrs.

### Task 2.1: `_store.py` — URI → `zarr.storage.Store` dispatch

**Files:**
- Create: `python/lsst/images/zarr/_store.py`
- Test: `tests/test_zarr_store.py`

URI dispatch:

| URI shape | Store |
|---|---|
| `*.zarr.zip` (any scheme) | `zarr.storage.ZipStore` |
| `file://` or local path | `zarr.storage.LocalStore` |
| `http(s)://`, `s3://`, `gs://`, etc. | `zarr.storage.FsspecStore` (via `fsspec.url_to_fs`) |

Create-only mode is enforced here: write helpers refuse to open a non-empty existing store.

- [ ] **Step 1: Write the failing test**

Create `tests/test_zarr_store.py`:

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

try:
    import zarr

    from lsst.images.zarr._store import open_store_for_read, open_store_for_write

    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class StoreDispatchTestCase(unittest.TestCase):
    def test_local_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "out.zarr")
            with open_store_for_write(target) as store:
                self.assertIsInstance(store, zarr.storage.LocalStore)
                zarr.create_group(store=store, zarr_format=3)
            with open_store_for_read(target) as store:
                self.assertIsInstance(store, zarr.storage.LocalStore)
                root = zarr.open_group(store=store, mode="r")
                self.assertEqual(list(root.keys()), [])

    def test_zip_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "out.zarr.zip")
            with open_store_for_write(target) as store:
                self.assertIsInstance(store, zarr.storage.ZipStore)
                zarr.create_group(store=store, zarr_format=3)
            with open_store_for_read(target) as store:
                self.assertIsInstance(store, zarr.storage.ZipStore)

    def test_create_only_refuses_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "out.zarr")
            with open_store_for_write(target) as store:
                zarr.create_group(store=store, zarr_format=3)
            with self.assertRaisesRegex(OSError, "already exists"):
                with open_store_for_write(target):
                    pass


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_zarr_store.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write `_store.py`**

Create `python/lsst/images/zarr/_store.py`:

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

__all__ = ("open_store_for_read", "open_store_for_write")

import os
from collections.abc import Iterator
from contextlib import contextmanager

import zarr

from lsst.resources import ResourcePath, ResourcePathExpression


def _is_zip(rp: ResourcePath) -> bool:
    return rp.path.endswith(".zarr.zip") or rp.path.endswith(".zip")


def _is_remote(rp: ResourcePath) -> bool:
    return rp.scheme not in ("", "file")


@contextmanager
def open_store_for_write(path: ResourcePathExpression) -> Iterator[zarr.storage.Store]:
    """Open a zarr store for writing.

    Refuses to overwrite a non-empty existing store. The returned
    context manager closes the store on exit; for ``ZipStore`` this
    finalizes the central directory.
    """
    rp = ResourcePath(path)
    if _is_zip(rp):
        if _is_remote(rp):
            raise NotImplementedError("Remote ZipStore writes are a follow-up.")
        local = rp.ospath
        if os.path.exists(local) and os.path.getsize(local) > 0:
            raise OSError(f"File {local!r} already exists.")
        store = zarr.storage.ZipStore(local, mode="w")
        try:
            yield store
        finally:
            store.close()
        return
    if _is_remote(rp):
        import fsspec

        fs, fs_path = fsspec.url_to_fs(str(rp))
        if fs.exists(fs_path) and fs.ls(fs_path):
            raise OSError(f"Store {rp!s} already exists.")
        store = zarr.storage.FsspecStore(fs=fs, path=fs_path, read_only=False)
        yield store
        return
    local = rp.ospath
    if os.path.exists(local) and os.listdir(local):
        raise OSError(f"Directory {local!r} already exists and is non-empty.")
    os.makedirs(local, exist_ok=True)
    store = zarr.storage.LocalStore(local, read_only=False)
    yield store


@contextmanager
def open_store_for_read(path: ResourcePathExpression) -> Iterator[zarr.storage.Store]:
    """Open a zarr store for reading."""
    rp = ResourcePath(path)
    if _is_zip(rp):
        if _is_remote(rp):
            with rp.as_local() as local:
                store = zarr.storage.ZipStore(local.ospath, mode="r")
                try:
                    yield store
                finally:
                    store.close()
            return
        store = zarr.storage.ZipStore(rp.ospath, mode="r")
        try:
            yield store
        finally:
            store.close()
        return
    if _is_remote(rp):
        import fsspec

        fs, fs_path = fsspec.url_to_fs(str(rp))
        store = zarr.storage.FsspecStore(fs=fs, path=fs_path, read_only=True)
        yield store
        return
    store = zarr.storage.LocalStore(rp.ospath, read_only=True)
    yield store
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_zarr_store.py -v`
Expected: PASS — 3 tests.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/zarr/_store.py tests/test_zarr_store.py
git commit -m "feat: add zarr store dispatch (LocalStore / ZipStore / FsspecStore)"
```

### Task 2.2: `_layout.py` — axes per archive class and chunk derivation

**Files:**
- Create: `python/lsst/images/zarr/_layout.py`
- Test: `tests/test_zarr_layout.py`

This task adds the per-archive-class layout rules: axis tuples and chunk-shape derivation. Chunk derivation honors three sources of truth in priority order:

1. Explicit per-array override (from `write(chunks={...})`).
2. `cell_shape` from the archive metadata (for `CellCoadd`).
3. `min(1024, dim)` per axis fallback.

A separate helper `chunks_aligned_to(image_chunks, shape)` derives `variance`/`mask` chunks from the `image` array's chunks so siblings stay aligned (CF / xarray / GDAL all assume this). The output archive will call this helper when the user has not overridden the sibling's chunks.

The affine residual validator lands in Task 2.3 (separate task because it has its own test surface).

- [ ] **Step 1: Write the failing test**

Create `tests/test_zarr_layout.py`:

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

try:
    from lsst.images.zarr._layout import (
        axes_for_archive_class,
        chunks_aligned_to,
        chunks_for,
    )

    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class LayoutTestCase(unittest.TestCase):
    def test_axes_for_archive_class(self) -> None:
        # Standard 2-D images use (y, x).
        self.assertEqual(axes_for_archive_class("Image"), ("y", "x"))
        self.assertEqual(axes_for_archive_class("MaskedImage"), ("y", "x"))
        self.assertEqual(axes_for_archive_class("VisitImage"), ("y", "x"))
        self.assertEqual(axes_for_archive_class("Mask"), ("y", "x"))
        self.assertEqual(axes_for_archive_class("CellCoadd"), ("y", "x"))
        # ColorImage's root has no top-level multiscale; this returns
        # an empty tuple to signal "no OME multiscale at this level".
        self.assertEqual(axes_for_archive_class("ColorImage"), ())

    def test_chunks_for_default(self) -> None:
        self.assertEqual(chunks_for("Image", (4096, 4096), None), (1024, 1024))
        # Smaller than 1024 -> use full dim.
        self.assertEqual(chunks_for("Image", (300, 600), None), (300, 600))

    def test_chunks_for_override(self) -> None:
        self.assertEqual(chunks_for("Image", (4096, 4096), (256, 256)), (256, 256))

    def test_chunks_for_cell_coadd_uses_cell_shape(self) -> None:
        result = chunks_for(
            "CellCoadd",
            (4096, 4096),
            None,
            archive_metadata={"cell_shape": (256, 256)},
        )
        self.assertEqual(result, (256, 256))

    def test_chunks_for_cell_coadd_without_metadata_falls_back(self) -> None:
        self.assertEqual(chunks_for("CellCoadd", (4096, 4096), None), (1024, 1024))

    def test_chunks_aligned_to_matches_image(self) -> None:
        # variance / mask follow image's chunks when not overridden.
        self.assertEqual(
            chunks_aligned_to(image_chunks=(256, 256), shape=(4096, 4096)),
            (256, 256),
        )
        # If the sibling shape is smaller than image's chunks, clamp.
        self.assertEqual(
            chunks_aligned_to(image_chunks=(1024, 1024), shape=(300, 600)),
            (300, 600),
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_zarr_layout.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write `_layout.py`**

Create `python/lsst/images/zarr/_layout.py`:

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

"""Per-archive-class layout rules for the zarr backend.

This module centralises the decisions that vary by image type:

- which OME axes apply (``ColorImage`` has no root multiscale)
- default chunk sizes (clamped to 1024 per axis for plain images,
  cell-aligned for `CellCoadd`, image-aligned for `variance` / `mask`
  siblings)
- the affine residual validator that gates the OME
  ``coordinateTransformations`` block

Keeping these in one place lets the output archive populate the IR
generically.
"""

from __future__ import annotations

__all__ = (
    "axes_for_archive_class",
    "chunks_aligned_to",
    "chunks_for",
)

from collections.abc import Mapping
from typing import Any

_DEFAULT_AXIS_LIMIT = 1024


def axes_for_archive_class(name: str) -> tuple[str, ...]:
    """Return the OME axis tuple for a given archive class.

    Returns an empty tuple for ``ColorImage`` to signal that there is
    no OME multiscale at the root of that class — the per-channel
    sub-archives carry their own ``(y, x)`` multiscales.
    """
    if name == "ColorImage":
        return ()
    return ("y", "x")


def chunks_for(
    archive_class: str,
    shape: tuple[int, ...],
    override: tuple[int, ...] | None,
    *,
    archive_metadata: Mapping[str, Any] | None = None,
) -> tuple[int, ...]:
    """Return the chunk shape to use for a top-level array.

    Parameters
    ----------
    archive_class
        Top-level archive class name; used for class-specific
        defaults like ``CellCoadd``'s cell-aligned chunks.
    shape
        The full array shape, used to clamp the default per-axis.
    override
        User-supplied chunk shape. If not ``None`` it is returned
        verbatim after a length check.
    archive_metadata
        Class-specific layout hints. ``CellCoadd`` reads
        ``"cell_shape"`` from this mapping.
    """
    if override is not None:
        if len(override) != len(shape):
            raise ValueError(
                f"chunks override has rank {len(override)}, "
                f"expected {len(shape)} for {archive_class!r}."
            )
        return tuple(override)
    if archive_class == "CellCoadd" and archive_metadata is not None:
        cell_shape = archive_metadata.get("cell_shape")
        if cell_shape is not None:
            return tuple(min(c, dim) for c, dim in zip(cell_shape, shape, strict=True))
    return tuple(min(_DEFAULT_AXIS_LIMIT, dim) for dim in shape)


def chunks_aligned_to(
    *,
    image_chunks: tuple[int, ...],
    shape: tuple[int, ...],
) -> tuple[int, ...]:
    """Derive a sibling array's chunks from the ``image`` array's chunks.

    Used by `ZarrOutputArchive.add_array` for ``variance`` and
    ``mask`` siblings when the user has not provided an explicit
    override. The result is per-axis ``min(image_chunks[i],
    shape[i])`` so a sibling smaller than ``image`` is not
    over-chunked.
    """
    if len(image_chunks) != len(shape):
        raise ValueError(
            f"image_chunks rank {len(image_chunks)} does not match "
            f"sibling shape rank {len(shape)}."
        )
    return tuple(min(c, dim) for c, dim in zip(image_chunks, shape, strict=True))
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_zarr_layout.py -v`
Expected: PASS — 6 tests.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/zarr/_layout.py tests/test_zarr_layout.py
git commit -m "feat: add zarr layout rules for axes and chunk derivation"
```

### Task 2.3: `_layout.py` — affine residual validator

**Files:**
- Modify: `python/lsst/images/zarr/_layout.py`
- Modify: `tests/test_zarr_layout.py`

The affine residual validator extracts the linear / affine portion of the AST FrameSet's pixel-to-sky mapping, samples residuals on an 11×11 grid, and decides whether to emit the OME `coordinateTransformations` block. The contract:

- Input: a `FrameSet`, a 2-D image bbox `(y_size, x_size)`, and a max residual threshold (default 1.0 pixel).
- Output: `AffineCheckResult` carrying either the affine `coordinateTransformations` to emit, **or** a `dropped=True` flag with the observed max residual.

The function does **not** know about zarr; it only knows about AST. The output archive consumes the result and threads it into the OME multiscale block.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_zarr_layout.py`:

```python
@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class AffineValidatorTestCase(unittest.TestCase):
    def _make_linear_frame_set(self, *, scale: float = 0.2):
        # Build a synthetic FrameSet whose pixel-to-sky is a pure scale.
        from lsst.images._transforms._ast import (
            Frame,
            FrameSet,
            ZoomMap,
        )

        base = Frame(2, "Domain=PIXEL")
        sky = Frame(2, "Domain=SKY")
        fs = FrameSet(base)
        fs.addFrame(FrameSet.BASE, ZoomMap(2, scale), sky)
        return fs

    def _make_distorted_frame_set(self):
        # Build a FrameSet that adds a polynomial distortion on top of
        # a linear pixel-to-sky map; the affine approximation will be
        # off by many pixels at the corners.
        from lsst.images._transforms._ast import (
            Frame,
            FrameSet,
            PolyMap,
            ZoomMap,
            CmpMap,
        )

        base = Frame(2, "Domain=PIXEL")
        sky = Frame(2, "Domain=SKY")
        # Forward polynomial: x' = x + 0.001 * y^2; y' = y + 0.001 * x^2.
        # PolyMap coefficient table format: [coeff, output_index, x_power, y_power].
        forward_coeffs = [
            [1.0, 1, 1, 0],
            [0.001, 1, 0, 2],
            [1.0, 2, 0, 1],
            [0.001, 2, 2, 0],
        ]
        poly = PolyMap(forward_coeffs, 2, "IterInverse=1, NIterInverse=20")
        cmp = CmpMap(poly, ZoomMap(2, 0.2), True)
        fs = FrameSet(base)
        fs.addFrame(FrameSet.BASE, cmp, sky)
        return fs

    def test_pure_linear_passes(self) -> None:
        from lsst.images.zarr._layout import affine_check

        fs = self._make_linear_frame_set(scale=0.2)
        result = affine_check(
            frame_set=fs,
            image_shape=(64, 64),
            max_residual_pixels=1.0,
        )
        self.assertFalse(result.dropped)
        self.assertIsNotNone(result.coordinate_transformations)
        self.assertLess(result.max_residual_pixels, 1e-6)

    def test_high_distortion_drops_block(self) -> None:
        from lsst.images.zarr._layout import affine_check

        fs = self._make_distorted_frame_set()
        # 4096-pixel-wide image: 0.001 * 2048^2 ~ 4000 pixels of error
        # at the corners. Way over the 1-pixel threshold.
        result = affine_check(
            frame_set=fs,
            image_shape=(4096, 4096),
            max_residual_pixels=1.0,
        )
        self.assertTrue(result.dropped)
        self.assertGreater(result.max_residual_pixels, 1.0)
        # When dropped, the function still reports the residual so the
        # output archive can record it as lsst.wcs_simplified_max_residual_pixels.
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_zarr_layout.py::AffineValidatorTestCase -v`
Expected: FAIL — `affine_check` does not exist.

- [ ] **Step 3: Implement `affine_check`**

Append to `python/lsst/images/zarr/_layout.py`:

```python
__all__ = (
    "AffineCheckResult",
    "affine_check",
    "axes_for_archive_class",
    "chunks_aligned_to",
    "chunks_for",
)


from dataclasses import dataclass


@dataclass
class AffineCheckResult:
    """Result of validating a simplified affine against a full WCS.

    When ``dropped`` is False, ``coordinate_transformations`` is the
    OME-NGFF ``coordinateTransformations`` list to emit. When True,
    the caller must omit the block (or emit a unit scale only) and
    record ``max_residual_pixels`` as the observed worst error.
    """

    dropped: bool
    max_residual_pixels: float
    coordinate_transformations: list[dict[str, Any]] | None


def affine_check(
    *,
    frame_set: Any,
    image_shape: tuple[int, int],
    max_residual_pixels: float = 1.0,
    grid: int = 11,
) -> AffineCheckResult:
    """Build an OME affine ``coordinateTransformations`` for ``frame_set``,
    validate it on an ``grid``×``grid`` sample, and decide whether to keep it.

    The simplified affine is constructed by mapping three reference
    pixels (origin and the two unit-axis steps) through ``frame_set``
    to recover the linear coefficients. The full pixel-to-sky map is
    then evaluated at every grid point and compared to the affine's
    prediction; the worst great-circle separation is divided by the
    pixel scale to get a pixel-equivalent residual.

    If ``max_residual <= max_residual_pixels``, returns a result whose
    ``coordinate_transformations`` is the affine block. Otherwise
    returns a dropped result and the caller must emit the unit scale
    (or no transformations at all).
    """
    import numpy as np

    h, w = image_shape

    # 1. Recover the linear / affine portion by mapping three pixels.
    pixels = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    sky_at_ref = _frame_set_apply(frame_set, pixels)
    origin = sky_at_ref[0]
    dxsky = sky_at_ref[1] - origin
    dysky = sky_at_ref[2] - origin
    affine_matrix = np.array(
        [
            [dxsky[0], dysky[0], origin[0]],
            [dxsky[1], dysky[1], origin[1]],
            [0.0, 0.0, 1.0],
        ]
    )

    pixel_scale_y = float(np.linalg.norm(dysky))
    pixel_scale_x = float(np.linalg.norm(dxsky))
    pixel_scale = float(np.sqrt(pixel_scale_y * pixel_scale_x))
    if pixel_scale <= 0.0:
        return AffineCheckResult(
            dropped=True,
            max_residual_pixels=float("inf"),
            coordinate_transformations=None,
        )

    # 2. Sample residuals on a grid spanning [0, h-1] x [0, w-1].
    ys = np.linspace(0.0, max(h - 1, 0), grid)
    xs = np.linspace(0.0, max(w - 1, 0), grid)
    grid_pixels = np.array([[y, x] for y in ys for x in xs])
    sky_full = _frame_set_apply(frame_set, grid_pixels)
    affine_pred = (affine_matrix[:2, :2] @ grid_pixels.T).T + origin
    great_circle = _angular_separation(sky_full, affine_pred)
    max_residual = float(np.max(great_circle) / pixel_scale)

    coordinate_transformations: list[dict[str, Any]] = [
        {
            "type": "scale",
            "scale": [pixel_scale_y, pixel_scale_x],
        },
        {
            "type": "affine",
            "affine": affine_matrix.tolist(),
        },
    ]

    if max_residual > max_residual_pixels:
        return AffineCheckResult(
            dropped=True,
            max_residual_pixels=max_residual,
            coordinate_transformations=None,
        )
    return AffineCheckResult(
        dropped=False,
        max_residual_pixels=max_residual,
        coordinate_transformations=coordinate_transformations,
    )


def _frame_set_apply(frame_set: Any, pixels: Any) -> Any:
    """Apply ``frame_set``'s base->current mapping to a (N, 2) pixel array."""
    import numpy as np

    pixels = np.asarray(pixels, dtype=float)
    mapping = frame_set.getMapping(frame_set.base, frame_set.current)
    # AST applyForward expects (n_axes, n_points); transpose round-trip.
    out = mapping.applyForward(pixels.T)
    return np.asarray(out).T


def _angular_separation(a: Any, b: Any) -> Any:
    """Element-wise great-circle separation between two arrays of (lon, lat).

    Inputs in radians (AST default for unit sky frames). Returns a 1-D
    array of separations in the same units as the input.
    """
    import numpy as np

    a = np.asarray(a)
    b = np.asarray(b)
    lon_a, lat_a = a[:, 0], a[:, 1]
    lon_b, lat_b = b[:, 0], b[:, 1]
    dlon = lon_b - lon_a
    return np.arccos(
        np.clip(
            np.sin(lat_a) * np.sin(lat_b) + np.cos(lat_a) * np.cos(lat_b) * np.cos(dlon),
            -1.0,
            1.0,
        )
    )
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_zarr_layout.py -v`
Expected: PASS — 8 tests; the linear FrameSet has near-zero residual, the polynomial FrameSet is dropped with `max_residual_pixels` in the thousands.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/zarr/_layout.py tests/test_zarr_layout.py
git commit -m "feat: add affine_check residual validator for OME coordinateTransformations"
```

### Task 2.4: `ZarrOutputArchive` skeleton — `serialize_direct` / `serialize_pointer` / `iter_frame_sets`

**Files:**
- Create: `python/lsst/images/zarr/_output_archive.py`
- Test: `tests/test_zarr_output_archive.py`

The constructor builds an empty `ZarrDocument` and stashes the user's per-array overrides plus the `archive_metadata` dict (used by `_layout.chunks_for` to see `cell_shape`). `serialize_direct` returns a `NestedOutputArchive` so nested calls land at compound paths (`red/image` rather than `image`). `serialize_pointer` writes the sub-tree's JSON bytes to a `tree` array under the sub-archive's path and returns a `ZarrPointerModel(path="<sub>/tree")`.

`add_array` / `add_table` / `add_structured_array` / `add_tree` follow in subsequent tasks; they raise `NotImplementedError` here so the abstract class is concretely implementable.

- [ ] **Step 1: Write the failing test**

Create `tests/test_zarr_output_archive.py`:

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

import pydantic

try:
    from lsst.images.zarr._common import ZarrPointerModel
    from lsst.images.zarr._output_archive import ZarrOutputArchive

    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False


class _Sub(pydantic.BaseModel):
    label: str = "sub"


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrOutputArchiveSkeletonTestCase(unittest.TestCase):
    def test_serialize_direct_returns_nested_result(self) -> None:
        archive = ZarrOutputArchive()

        def serializer(arch):  # noqa: ANN001
            return _Sub(label="ok")

        result = archive.serialize_direct("red", serializer)
        self.assertEqual(result.label, "ok")

    def test_serialize_pointer_writes_json_subtree(self) -> None:
        archive = ZarrOutputArchive()

        def serializer(arch):  # noqa: ANN001
            return _Sub(label="psf")

        pointer = archive.serialize_pointer("psf", serializer, key=12345)
        self.assertIsInstance(pointer, ZarrPointerModel)
        self.assertEqual(pointer.path, "/psf/tree")
        # Cached on second call.
        again = archive.serialize_pointer("psf", serializer, key=12345)
        self.assertEqual(again, pointer)
        # IR holds the JSON bytes as a 1-D uint8 array.
        node = archive.document.root.get("/psf/tree")
        self.assertEqual(str(node.dtype), "uint8")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_zarr_output_archive.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Write the skeleton**

Create `python/lsst/images/zarr/_output_archive.py`:

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

__all__ = ("ZarrOutputArchive", "write")

from collections.abc import Callable, Hashable, Iterator, Mapping
from typing import Any

import numpy as np
import pydantic

from .._transforms import FrameSet
from ..serialization import (
    ArchiveTree,
    NestedOutputArchive,
    OutputArchive,
)
from ._common import (
    ZarrCompressionOptions,
    ZarrPointerModel,
    archive_path_to_zarr_path,
)
from ._model import ZarrArray, ZarrDocument, ZarrGroup


class ZarrOutputArchive(OutputArchive[ZarrPointerModel]):
    """Output archive that populates a `ZarrDocument` IR.

    Bytes are not written until the IR is materialized via
    `ZarrDocument.to_zarr`, which the public `write` helper performs
    on context-manager exit.

    Parameters
    ----------
    chunks
        Per-array chunk overrides keyed by the JSON pointer of the
        attribute the array backs (or its zarr path). ``None`` for a
        key means "use the layout default".
    shards, compression
        Same shape as ``chunks``.
    archive_class
        Top-level archive class name (``"VisitImage"``, ``"CellCoadd"``,
        …). Used by the layout layer to pick chunk defaults; set by
        ``write()`` before ``obj.serialize`` runs so ``add_array``
        sees the right value.
    archive_metadata
        Class-specific layout hints (``cell_shape`` for ``CellCoadd``).
    """

    def __init__(
        self,
        *,
        chunks: Mapping[str, tuple[int, ...] | None] | None = None,
        shards: Mapping[str, tuple[int, ...] | None] | None = None,
        compression: Mapping[str, ZarrCompressionOptions | None] | None = None,
        archive_class: str = "Image",
        archive_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.document = ZarrDocument(root=ZarrGroup())
        self._chunks = dict(chunks) if chunks else {}
        self._shards = dict(shards) if shards else {}
        self._compression = dict(compression) if compression else {}
        self._archive_class = archive_class
        self._archive_metadata = dict(archive_metadata) if archive_metadata else {}
        self._pointers: dict[Hashable, ZarrPointerModel] = {}
        self._frame_sets: list[tuple[FrameSet, ZarrPointerModel]] = []

    def serialize_direct[T: pydantic.BaseModel](
        self, name: str, serializer: Callable[[OutputArchive[ZarrPointerModel]], T]
    ) -> T:
        nested = NestedOutputArchive[ZarrPointerModel](name, self)
        return serializer(nested)

    def serialize_pointer[T: ArchiveTree](
        self,
        name: str,
        serializer: Callable[[OutputArchive[ZarrPointerModel]], T],
        key: Hashable,
    ) -> ZarrPointerModel:
        if (cached := self._pointers.get(key)) is not None:
            return cached
        # Run the serializer first so any nested add_array calls land
        # inside the IR before we dump this sub-tree to JSON.
        archive_path = name if name.startswith("/") else f"/{name}"
        sub_zarr_path = archive_path_to_zarr_path(archive_path)
        model = self.serialize_direct(name, serializer)
        json_bytes = model.model_dump_json().encode("utf-8")
        parent = self.document.root.ensure_group(sub_zarr_path)
        parent.arrays["tree"] = ZarrArray(data=np.frombuffer(json_bytes, dtype=np.uint8))
        pointer = ZarrPointerModel(path=f"{sub_zarr_path}/tree")
        self._pointers[key] = pointer
        return pointer

    def serialize_frame_set[T: ArchiveTree](
        self,
        name: str,
        frame_set: FrameSet,
        serializer: Callable[[OutputArchive], T],
        key: Hashable,
    ) -> ZarrPointerModel:
        pointer = self.serialize_pointer(name, serializer, key)
        self._frame_sets.append((frame_set, pointer))
        return pointer

    def iter_frame_sets(self) -> Iterator[tuple[FrameSet, ZarrPointerModel]]:
        return iter(self._frame_sets)

    def add_array(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        raise NotImplementedError("add_array lands in Task 2.5")

    def add_table(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        raise NotImplementedError("add_table lands in Task 2.6")

    def add_structured_array(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        raise NotImplementedError("add_structured_array lands in Task 2.6")


def write(*args: Any, **kwargs: Any) -> Any:
    """Public write helper. Implemented in Task 2.7."""
    raise NotImplementedError("write() lands in Task 2.7")
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_zarr_output_archive.py -v`
Expected: PASS — 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/zarr/_output_archive.py tests/test_zarr_output_archive.py
git commit -m "feat: add ZarrOutputArchive skeleton with serialize_direct/pointer/frame_set"
```

### Task 2.5: `add_array` — image, variance, and the 2-D packed mask

**Files:**
- Modify: `python/lsst/images/zarr/_output_archive.py`
- Test: `tests/test_zarr_output_archive.py`

`add_array(array, name=...)` does three different things depending on the name:

1. `name == "image"` (or any non-mask name) — stage the array verbatim with default chunks (or overrides), attach `_ARRAY_DIMENSIONS` and `units` / `long_name` if known. The chunk shape is held aside as the "image chunks" so siblings can align to it.
2. `name == "variance"` — derive chunks from `image_chunks` via `chunks_aligned_to` when the user has not overridden, attach `_ARRAY_DIMENSIONS = ["y", "x"]`.
3. `name == "mask"` — convert the 3-D `(y, x, mask_size)` in-memory mask into a 2-D `(y, x)` packed-integer array of `mask_dtype_for_plane_count(n_planes)`. Build CF flag attrs from the schema (passed via `archive_metadata["mask_schema"]`). Derive chunks from `image_chunks` when not overridden.

Anonymous (nested) arrays land at the path equal to `name`, no special-case behavior.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_zarr_output_archive.py`:

```python
@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrOutputArchiveAddArrayTestCase(unittest.TestCase):
    def test_add_image(self) -> None:
        import numpy as np

        archive = ZarrOutputArchive()
        ref = archive.add_array(
            np.ones((4, 5), dtype=np.float32), name="image"
        )
        self.assertEqual(ref.source, "zarr:/image")
        self.assertEqual(list(ref.shape), [4, 5])
        node = archive.document.root.get("/image")
        self.assertEqual(node.shape, (4, 5))
        self.assertEqual(node.attributes.extra["_ARRAY_DIMENSIONS"], ["y", "x"])

    def test_add_variance_aligns_to_image_chunks(self) -> None:
        import numpy as np

        archive = ZarrOutputArchive(chunks={"image": (2, 2)})
        archive.add_array(np.ones((4, 5), dtype=np.float32), name="image")
        archive.add_array(np.ones((4, 5), dtype=np.float64), name="variance")
        var_node = archive.document.root.get("/variance")
        self.assertEqual(tuple(var_node.chunks), (2, 2))

    def test_add_mask_packs_to_2d_with_cf_flag_attrs(self) -> None:
        import numpy as np

        from lsst.images import MaskPlane, MaskSchema

        schema = MaskSchema(
            [
                MaskPlane("BAD", "Bad pixel."),
                MaskPlane("SAT", "Saturated."),
                MaskPlane("CR", "Cosmic ray."),
            ]
        )
        # In-memory mask is (y, x, mask_size).
        in_memory = np.zeros((4, 5, 1), dtype=np.uint8)
        in_memory[0, 0, 0] = 0b1  # BAD
        in_memory[1, 1, 0] = 0b110  # SAT | CR

        archive = ZarrOutputArchive(archive_metadata={"mask_schema": schema})
        archive.add_array(np.ones((4, 5), dtype=np.float32), name="image")
        ref = archive.add_array(in_memory, name="mask")
        self.assertEqual(ref.source, "zarr:/mask")
        node = archive.document.root.get("/mask")
        # 2-D packed integer.
        self.assertEqual(node.shape, (4, 5))
        self.assertEqual(str(node.dtype), "uint8")  # 3 planes -> uint8
        # Bytes packed correctly.
        np.testing.assert_array_equal(node.data[0, 0], 0b1)
        np.testing.assert_array_equal(node.data[1, 1], 0b110)
        # CF flag attrs.
        attrs = node.attributes.extra
        self.assertEqual(attrs["flag_masks"], [1, 2, 4])
        self.assertEqual(attrs["flag_meanings"], "BAD SAT CR")
        self.assertEqual(
            attrs["flag_descriptions"],
            ["Bad pixel.", "Saturated.", "Cosmic ray."],
        )
        self.assertEqual(attrs["_ARRAY_DIMENSIONS"], ["y", "x"])

    def test_add_mask_picks_widest_dtype_for_40_planes(self) -> None:
        import numpy as np

        from lsst.images import MaskPlane, MaskSchema

        planes = [MaskPlane(f"P{i}", f"Plane {i}.") for i in range(40)]
        schema = MaskSchema(planes)
        in_memory = np.zeros((4, 5, 5), dtype=np.uint8)  # mask_size=5

        archive = ZarrOutputArchive(archive_metadata={"mask_schema": schema})
        archive.add_array(np.ones((4, 5), dtype=np.float32), name="image")
        archive.add_array(in_memory, name="mask")
        node = archive.document.root.get("/mask")
        self.assertEqual(node.shape, (4, 5))
        self.assertEqual(str(node.dtype), "uint64")

    def test_add_mask_refuses_more_than_64_planes(self) -> None:
        import numpy as np

        from lsst.images import MaskPlane, MaskSchema

        planes = [MaskPlane(f"P{i}", f"Plane {i}.") for i in range(65)]
        schema = MaskSchema(planes)
        in_memory = np.zeros((4, 5, 9), dtype=np.uint8)

        archive = ZarrOutputArchive(archive_metadata={"mask_schema": schema})
        archive.add_array(np.ones((4, 5), dtype=np.float32), name="image")
        with self.assertRaisesRegex(ValueError, "supports up to 64"):
            archive.add_array(in_memory, name="mask")

    def test_add_anonymous_nested_array(self) -> None:
        import numpy as np

        archive = ZarrOutputArchive()
        ref = archive.add_array(
            np.ones((3,), dtype=np.float32), name="psf/centroids"
        )
        self.assertEqual(ref.source, "zarr:/psf/centroids")
        self.assertEqual(archive.document.root.get("/psf/centroids").shape, (3,))
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_zarr_output_archive.py::ZarrOutputArchiveAddArrayTestCase -v`
Expected: FAIL — `add_array` raises `NotImplementedError`.

- [ ] **Step 3: Implement `add_array` and the mask-packing helper**

In `python/lsst/images/zarr/_output_archive.py`, extend imports:

```python
import astropy.io.fits
import astropy.table
import astropy.units

from ..serialization import (
    ArchiveTree,
    ArrayReferenceModel,
    NestedOutputArchive,
    NumberType,
    OutputArchive,
    TableColumnModel,
    TableModel,
    no_header_updates,
)
from ._common import (
    ZarrCompressionOptions,
    ZarrPointerModel,
    archive_path_to_zarr_path,
    mask_dtype_for_plane_count,
)
from ._layout import chunks_aligned_to, chunks_for
from ._model import (
    CfFlagAttributes,
    MaskPlaneEntry,
    ZarrArray,
    ZarrDocument,
    ZarrGroup,
    build_image_array_attrs,
)
```

Add an `_image_chunks` field to `__init__`:

```python
        self._image_chunks: tuple[int, ...] | None = None
```

Replace the `add_array` placeholder:

```python
    def add_array(
        self,
        array: np.ndarray,
        *,
        name: str | None = None,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> ArrayReferenceModel:
        if name is None:
            raise ValueError("Anonymous arrays are not supported in ZarrOutputArchive.")
        archive_path = name if name.startswith("/") else f"/{name}"
        zarr_path = archive_path_to_zarr_path(archive_path)
        leaf = zarr_path.rsplit("/", 1)[-1]
        parent_path = zarr_path[: -(len(leaf) + 1)] or "/"
        parent = self.document.root.ensure_group(parent_path)

        # Mask: pack 3-D (y, x, mask_size) -> 2-D wide-int packed.
        if leaf == "mask" and array.ndim == 3:
            packed, flag_attrs = self._pack_mask(array)
            chunks = self._chunks.get(name) or self._chunks.get(leaf)
            if chunks is None and self._image_chunks is not None:
                chunks = chunks_aligned_to(
                    image_chunks=self._image_chunks, shape=packed.shape
                )
            extra: dict[str, Any] = {"_ARRAY_DIMENSIONS": ["y", "x"]}
            extra.update(flag_attrs.dump())
            ir_array = ZarrArray(
                data=packed,
                chunks=chunks,
                shards=self._shards.get(name),
                compression=self._compression.get(name),
            )
            ir_array.attributes.extra = extra
            parent.arrays[leaf] = ir_array
            return ArrayReferenceModel(
                source=f"zarr:{zarr_path}",
                shape=list(packed.shape),
                datatype=NumberType.from_numpy(packed.dtype),
            )

        # variance / other top-level siblings: align to image's chunks.
        if leaf in ("variance",) or (parent_path == "/" and self._image_chunks):
            chunks = self._chunks.get(name) or self._chunks.get(leaf)
            if chunks is None and self._image_chunks is not None and array.ndim == len(
                self._image_chunks
            ):
                chunks = chunks_aligned_to(
                    image_chunks=self._image_chunks, shape=array.shape
                )
        else:
            chunks = self._chunks.get(name) or self._chunks.get(leaf)

        # Default chunks for the top-level image: from layout rules.
        if chunks is None and parent_path == "/" and leaf == "image":
            chunks = chunks_for(
                self._archive_class,
                array.shape,
                None,
                archive_metadata=self._archive_metadata,
            )

        ir_array = ZarrArray(
            data=np.ascontiguousarray(array),
            chunks=chunks,
            shards=self._shards.get(name),
            compression=self._compression.get(name),
        )
        if parent_path == "/" and leaf in ("image", "variance"):
            ir_array.attributes.extra = build_image_array_attrs(
                axes=("y", "x"),
                long_name="science image" if leaf == "image" else "image variance",
            )
        parent.arrays[leaf] = ir_array

        # Remember the image's chunks so siblings can align.
        if parent_path == "/" and leaf == "image" and chunks is not None:
            self._image_chunks = tuple(chunks)

        return ArrayReferenceModel(
            source=f"zarr:{zarr_path}",
            shape=list(array.shape),
            datatype=NumberType.from_numpy(array.dtype),
        )

    def _pack_mask(
        self, array: np.ndarray
    ) -> tuple[np.ndarray, CfFlagAttributes]:
        """Pack a 3-D ``(y, x, mask_size)`` mask into a 2-D wide-int array.

        The schema is taken from ``self._archive_metadata["mask_schema"]``.
        Returns the packed array and the CF flag attributes.
        """
        from lsst.images import MaskSchema

        schema = self._archive_metadata.get("mask_schema")
        if not isinstance(schema, MaskSchema):
            raise ValueError(
                "Writing a 3-D mask requires archive_metadata['mask_schema'] "
                "to be set; the output archive cannot infer the plane "
                "definitions otherwise."
            )
        n_planes = len(schema)
        target_dtype = mask_dtype_for_plane_count(n_planes)
        # Pack: each (y, x) pixel's mask_size bytes -> one wide integer.
        # Byte 0 is the low byte (planes 0..7), byte 1 is the next, etc.
        packed = np.zeros(array.shape[:2], dtype=target_dtype)
        for i in range(array.shape[2]):
            packed |= array[..., i].astype(target_dtype) << (8 * i)
        planes = [
            MaskPlaneEntry(name=p.name, bit=i, description=p.description)
            for i, p in enumerate(schema)
        ]
        return packed, CfFlagAttributes(planes=planes)
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_zarr_output_archive.py -v`
Expected: PASS — 8 tests; mask packs to the correct dtype, CF flag attrs are populated, sibling chunks align.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/zarr/_output_archive.py tests/test_zarr_output_archive.py
git commit -m "feat: implement add_array with image/variance/mask handling and CF flag attrs"
```

### Task 2.6: `add_table` and `add_structured_array`

**Files:**
- Modify: `python/lsst/images/zarr/_output_archive.py`
- Modify: `tests/test_zarr_output_archive.py`

Tables stage one 1-D zarr array per column under `/lsst/tables/<name>/<column>` and attach the table's `meta` block to the parent group's `lsst` namespace.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_zarr_output_archive.py`:

```python
@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrOutputArchiveAddTableTestCase(unittest.TestCase):
    def test_add_table_creates_one_array_per_column(self) -> None:
        import astropy.table
        import numpy as np

        archive = ZarrOutputArchive()
        original = astropy.table.Table(
            {
                "x": np.arange(4, dtype=np.int32),
                "y": np.arange(4, dtype=np.float32),
            },
            meta={"comment": "small catalog"},
        )
        model = archive.add_table(original, name="cat")
        self.assertEqual(len(model.columns), 2)
        sources = {c.name: c.data.source for c in model.columns}
        self.assertEqual(sources["x"], "zarr:/lsst/tables/cat/x")
        self.assertEqual(sources["y"], "zarr:/lsst/tables/cat/y")
        # Each column is its own zarr array under the parent group.
        x_node = archive.document.root.get("/lsst/tables/cat/x")
        self.assertEqual(x_node.shape, (4,))
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_zarr_output_archive.py::ZarrOutputArchiveAddTableTestCase -v`
Expected: FAIL — `add_table` raises `NotImplementedError`.

- [ ] **Step 3: Implement `add_table` and `add_structured_array`**

Replace the placeholders in `_output_archive.py`:

```python
    def add_table(
        self,
        table: astropy.table.Table,
        *,
        name: str | None = None,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> TableModel:
        if name is None:
            raise ValueError("Anonymous tables are not supported in ZarrOutputArchive.")
        columns = TableColumnModel.from_table(table)
        archive_path = name if name.startswith("/") else f"/{name}"
        table_zarr_path = f"/lsst/tables{archive_path}"
        parent = self.document.root.ensure_group(table_zarr_path)
        for c in columns:
            assert isinstance(c.data, ArrayReferenceModel)
            column_array = np.ascontiguousarray(np.asarray(table[c.name]))
            parent.arrays[c.name] = ZarrArray(data=column_array)
            c.data.source = f"zarr:{table_zarr_path}/{c.name}"
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
        if name is None:
            raise ValueError("Anonymous structured arrays are not supported.")
        columns = TableColumnModel.from_record_dtype(array.dtype)
        archive_path = name if name.startswith("/") else f"/{name}"
        table_zarr_path = f"/lsst/tables{archive_path}"
        parent = self.document.root.ensure_group(table_zarr_path)
        for c in columns:
            assert isinstance(c.data, ArrayReferenceModel)
            column_array = np.ascontiguousarray(array[c.name])
            parent.arrays[c.name] = ZarrArray(data=column_array)
            c.data.source = f"zarr:{table_zarr_path}/{c.name}"
            if units and (unit := units.get(c.name)):
                c.unit = unit
            if descriptions and (description := descriptions.get(c.name)):
                c.description = description
        return TableModel(columns=columns)
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_zarr_output_archive.py -v`
Expected: PASS — 9 tests.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/zarr/_output_archive.py tests/test_zarr_output_archive.py
git commit -m "feat: implement ZarrOutputArchive add_table and add_structured_array"
```

### Task 2.7: `add_tree`, OME multiscale + WCS validator integration, public `write()`

**Files:**
- Modify: `python/lsst/images/zarr/_output_archive.py`
- Modify: `python/lsst/images/zarr/__init__.py`
- Modify: `tests/test_zarr_output_archive.py`

`add_tree` finalizes the IR:

1. Stage the JSON tree at `/tree`.
2. Stage the AST WCS string at `/wcs_ast` (when an AST FrameSet was registered via `serialize_frame_set` or supplied directly).
3. Build the OME multiscale block. If a top-level `/image` array exists and the archive carries a frame set, run `affine_check`. If the result drops the affine, emit a unit-scale block and set `lsst.wcs_simplified_dropped: true` with the residual.
4. Set `lsst.archive_class`, `lsst.tree`, `lsst.wcs_ast` (if present), `data_model`, `version`, `lsst.cell_grid` (when `archive_metadata["cell_grid"]` is set).

The public `write(obj, path, ...)` function constructs the archive, runs the serializer, calls `add_tree`, and materializes via `open_store_for_write` + `to_zarr`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_zarr_output_archive.py`:

```python
@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrWriteHelperTestCase(unittest.TestCase):
    def test_write_image_to_local_directory(self) -> None:
        import os
        import tempfile

        import numpy as np
        import zarr

        from lsst.images import Box, Image
        from lsst.images.zarr import write
        from lsst.images.zarr._common import LSST_NS, OME_NS
        from lsst.images.zarr._model import ZarrDocument

        original = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "out.zarr")
            tree = write(original, target)
            self.assertIsNotNone(tree)
            with zarr.storage.LocalStore(target, read_only=True) as store:
                doc = ZarrDocument.from_zarr(store)
                # Top-level image and tree are present.
                self.assertIn("image", doc.root.arrays)
                self.assertIn("tree", doc.root.arrays)
                self.assertEqual(doc.root.arrays["image"].shape, (4, 5))
                # LSST root attrs.
                lsst_attrs = doc.root.attributes.lsst
                self.assertEqual(lsst_attrs["archive_class"], "Image")
                self.assertEqual(lsst_attrs["tree"], "tree")
                # OME multiscales points at /image; no projection means
                # the unit scale is emitted.
                ome = doc.root.attributes.ome
                self.assertIn("multiscales", ome)
                self.assertEqual(
                    ome["multiscales"][0]["datasets"][0]["path"], "image"
                )
                # data_model + version on root.
                self.assertEqual(
                    doc.root.attributes.extra["data_model"], "org.lsst.image"
                )
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_zarr_output_archive.py::ZarrWriteHelperTestCase -v`
Expected: FAIL — `write()` raises `NotImplementedError`.

- [ ] **Step 3: Implement `add_tree` and `write`**

Append to `python/lsst/images/zarr/_output_archive.py`:

```python
    def add_tree(self, tree: ArchiveTree) -> None:
        """Finalize the IR: write JSON tree, WCS, and root attributes.

        Called once after the user's serializer has populated arrays
        / sub-trees. Sets the ``lsst.*`` and ``ome.*`` blocks on the
        root group, stages ``/tree`` as 1-D ``uint8`` UTF-8 JSON, and
        runs the affine residual validator if the archive carries a
        frame set.
        """
        from ._layout import affine_check, axes_for_archive_class
        from ._model import OmeMultiscale

        # Stage the JSON tree at /tree.
        json_bytes = tree.model_dump_json().encode("utf-8")
        self.document.root.arrays["tree"] = ZarrArray(
            data=np.frombuffer(json_bytes, dtype=np.uint8)
        )

        # Stage the AST WCS string at /wcs_ast when a frame set is registered.
        wcs_ast_path: str | None = None
        if self._frame_sets:
            wcs_ast_path = self._stage_wcs_ast(self._frame_sets[0][0])

        # Root LSST attrs.
        lsst = self.document.root.attributes.lsst
        lsst["archive_class"] = self._archive_class
        lsst["tree"] = "tree"
        if wcs_ast_path is not None:
            lsst["wcs_ast"] = wcs_ast_path
        if "cell_grid" in self._archive_metadata:
            lsst["cell_grid"] = self._archive_metadata["cell_grid"]

        # data_model / version go to the top level (not under lsst:).
        self.document.root.attributes.extra["data_model"] = self._data_model_for(
            self._archive_class
        )
        self.document.root.attributes.extra["version"] = 1

        # OME multiscale block, gated by axes_for_archive_class.
        axes = axes_for_archive_class(self._archive_class)
        if axes and "image" in self.document.root.arrays:
            image_array = self.document.root.arrays["image"]
            ct: list[dict[str, Any]] | None = None
            if self._frame_sets:
                fs = self._frame_sets[0][0]
                check = affine_check(
                    frame_set=fs._get_ast_frame_set(),
                    image_shape=image_array.shape,
                    max_residual_pixels=1.0,
                )
                if check.dropped:
                    lsst["wcs_simplified_dropped"] = True
                    lsst["wcs_simplified_max_residual_pixels"] = check.max_residual_pixels
                else:
                    lsst["wcs_simplified_dropped"] = False
                    lsst["wcs_simplified_max_residual_pixels"] = check.max_residual_pixels
                    ct = check.coordinate_transformations
            multiscale = OmeMultiscale(
                name=self._archive_class.lower(),
                axes=axes,
                dataset_path="image",
                coordinate_transformations=ct,
            )
            self.document.root.attributes.ome["multiscales"] = [multiscale.dump()]

    def _stage_wcs_ast(self, frame_set: FrameSet) -> str:
        """Encode an AST FrameSet as a UTF-8 string and stage it at /wcs_ast."""
        from .._transforms._ast import Channel, StringStream

        ast_fs = frame_set._get_ast_frame_set()
        stream = StringStream()
        Channel(stream, options="Full=-1,Comment=0,Indent=0").write(ast_fs)
        text = stream.getSinkData()
        self.document.root.arrays["wcs_ast"] = ZarrArray(
            data=np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
        )
        return "wcs_ast"

    @staticmethod
    def _data_model_for(archive_class: str) -> str:
        """Map an archive class name to the public ``data_model`` string."""
        return {
            "Image": "org.lsst.image",
            "Mask": "org.lsst.mask",
            "MaskedImage": "org.lsst.masked_image",
            "VisitImage": "org.lsst.visit_image",
            "ColorImage": "org.lsst.color_image",
            "CellCoadd": "org.lsst.cell_coadd",
        }.get(archive_class, f"org.lsst.{archive_class.lower()}")


def write(
    obj: Any,
    path: Any,
    *,
    chunks: Mapping[str, tuple[int, ...] | None] | None = None,
    shards: Mapping[str, tuple[int, ...] | None] | None = None,
    compression: Mapping[str, ZarrCompressionOptions | None] | None = None,
    metadata: Mapping[str, Any] | None = None,
    butler_info: Any | None = None,
) -> ArchiveTree:
    """Write ``obj`` to a zarr archive at ``path``.

    Parameters mirror the FITS / NDF write helpers. The store
    implementation (LocalStore / ZipStore / FsspecStore) is selected
    from the URI shape by ``_store.open_store_for_write``.
    """
    from ._store import open_store_for_write

    archive_class = type(obj).__name__
    archive_default_name = getattr(obj, "_archive_default_name", None)
    archive_metadata: dict[str, Any] = {}
    if (cell_shape := getattr(obj, "cell_shape", None)) is not None:
        archive_metadata["cell_shape"] = tuple(cell_shape)
    if (cell_grid := getattr(obj, "cell_grid", None)) is not None:
        archive_metadata["cell_grid"] = {
            "bbox": list(cell_grid.bbox) if hasattr(cell_grid, "bbox") else None,
            "cell_shape": list(cell_grid.cell_shape)
            if hasattr(cell_grid, "cell_shape")
            else None,
        }
    if (mask_schema := getattr(obj, "mask_schema", None)) is not None:
        archive_metadata["mask_schema"] = mask_schema

    archive = ZarrOutputArchive(
        chunks=chunks,
        shards=shards,
        compression=compression,
        archive_class=archive_class,
        archive_metadata=archive_metadata,
    )
    if archive_default_name is not None:
        tree = archive.serialize_direct(archive_default_name, obj.serialize)
    else:
        tree = obj.serialize(archive)
    if metadata is not None:
        tree.metadata.update(metadata)
    if butler_info is not None:
        tree.butler_info = butler_info
    archive.add_tree(tree)
    with open_store_for_write(path) as store:
        archive.document.to_zarr(store)
    return tree
```

Re-export from `python/lsst/images/zarr/__init__.py` (replace the placeholder comment):

```python
from ._common import *  # noqa: F401, F403
from ._output_archive import *  # noqa: F401, F403
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_zarr_output_archive.py -v`
Expected: PASS — 10 tests.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/zarr/_output_archive.py python/lsst/images/zarr/__init__.py tests/test_zarr_output_archive.py
git commit -m "feat: add ZarrOutputArchive.add_tree and public write() helper"
```

### Task 2.8: Layout-level write tests for `MaskedImage` and `VisitImage`

**Files:**
- Modify: `tests/test_zarr_output_archive.py`

Pin the on-disk shape for the two harder archive classes.

- [ ] **Step 1: Write the test**

Append to `tests/test_zarr_output_archive.py`:

```python
@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrWriteOnDiskShapeTestCase(unittest.TestCase):
    def _round_trip_doc(self, obj):  # noqa: ANN001
        import os
        import tempfile

        import zarr

        from lsst.images.zarr import write
        from lsst.images.zarr._model import ZarrDocument

        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "out.zarr")
            write(obj, target)
            with zarr.storage.LocalStore(target, read_only=True) as store:
                return ZarrDocument.from_zarr(store)

    def test_masked_image_layout(self) -> None:
        import numpy as np

        from lsst.images import Box, Image, MaskedImage, MaskPlane, MaskSchema

        schema = MaskSchema([MaskPlane("BAD", "Bad pixel.")])
        image = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )
        masked = MaskedImage(image, mask_schema=schema)
        masked.mask.set("BAD", image.array % 2 == 0)

        doc = self._round_trip_doc(masked)
        self.assertEqual(
            doc.root.attributes.lsst["archive_class"], "MaskedImage"
        )
        # image / variance / mask are sibling root arrays.
        self.assertIn("image", doc.root.arrays)
        self.assertIn("variance", doc.root.arrays)
        self.assertIn("mask", doc.root.arrays)
        # Mask is 2-D packed integer with CF flag attrs.
        mask = doc.root.arrays["mask"]
        self.assertEqual(mask.shape, (4, 5))
        self.assertEqual(mask.attributes.extra["flag_meanings"], "BAD")
        # CF / xarray dims on every 2-D array.
        for name in ("image", "variance", "mask"):
            self.assertEqual(
                doc.root.arrays[name].attributes.extra["_ARRAY_DIMENSIONS"],
                ["y", "x"],
            )

    def test_visit_image_layout(self) -> None:
        import numpy as np

        from lsst.images import Box, Image, MaskPlane, MaskSchema, VisitImage

        schema = MaskSchema([MaskPlane("BAD", "Bad pixel.")])
        image = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )
        visit = VisitImage(image=image, mask_schema=schema)
        doc = self._round_trip_doc(visit)
        self.assertEqual(doc.root.attributes.lsst["archive_class"], "VisitImage")
        self.assertIn("image", doc.root.arrays)
```

- [ ] **Step 2: Run the tests**

Run: `pytest tests/test_zarr_output_archive.py::ZarrWriteOnDiskShapeTestCase -v`
Expected: PASS — both tests. If `VisitImage`'s constructor in this codebase needs different arguments than the snippet, adapt the constructor call only — the on-disk assertions stay.

- [ ] **Step 3: Commit**

```bash
git add tests/test_zarr_output_archive.py
git commit -m "test: pin on-disk zarr layout for MaskedImage and VisitImage"
```

---

**End of Phase 2.** Eight tasks. The output side now produces:

- `image`, `variance`, `mask` siblings at the root with aligned chunks
- 2-D packed-integer mask with CF `flag_masks` / `flag_meanings` / `flag_descriptions`
- `_ARRAY_DIMENSIONS` and `units` / `long_name` per array (xarray-readable)
- OME multiscales metadata pointing at `/image`
- Affine `coordinateTransformations` validated against an 11×11 grid; dropped to unit scale when residual exceeds 1 pixel
- `wcs_ast` 1-D `uint8` array as the authoritative WCS round-trip source

Phase 3 inverts this: `ZarrInputArchive`, `read()`, and the lazy-subset assertions that prove `slices=` only fetches the touched chunks.

## Phase 3 — `ZarrInputArchive`, `read()`, lazy subset enforcement, mask unpack

This phase delivers the read side. The hard constraint is the **lazy subset invariant**: `get_array(model, slices=...)` must forward `slices` to the underlying `zarr.Array` handle so a 4×4 subset of a 4096×4096 remote VisitImage downloads only the chunks intersecting that slice. The phase ships with a `_CountingStore`-based regression test that fails if any code path materializes the full array before slicing.

The phase also adds the **mask unpack path**: `Mask.serialize` (when the archive sets `_prefer_native_mask_arrays = True`) hands us a 3-D `(y, x, mask_size)` array which Phase 2's `add_array` packs to 2-D wide-integer; on read, `get_array` detects the rank mismatch (model claims 3-D, on-disk is 2-D, on-disk has `flag_masks` attribute) and unpacks via bit shifts.

### Task 3.0: Wire up `_prefer_native_mask_arrays`

**Files:**
- Modify: `python/lsst/images/zarr/_output_archive.py`
- Test: `tests/test_zarr_round_trip.py` (later in this phase confirms it round-trips)

A one-line retrofit to Phase 2 to make `Mask.serialize` choose the native 3-D path for our archive (matching what the NDF backend does). Without this, `Mask.serialize` calls `add_array` multiple times with 2-D `int32` splits and our packing path never runs.

- [ ] **Step 1: Add the class attribute**

In `python/lsst/images/zarr/_output_archive.py`, edit the `ZarrOutputArchive` class definition to add the class attribute right above `__init__`:

```python
class ZarrOutputArchive(OutputArchive[ZarrPointerModel]):
    """Output archive that populates a `ZarrDocument` IR.

    ... (existing docstring) ...
    """

    _prefer_native_mask_arrays: ClassVar[bool] = True
    """Tell Mask.serialize to hand us the 3-D ``(y, x, mask_size)``
    array in one ``add_array`` call. Our ``add_array`` packs that into
    a 2-D wide-integer array on disk with CF flag_masks / flag_meanings
    attributes.
    """

    def __init__(...):
        ...
```

(Add `from typing import ClassVar` to the imports if it is not already present.)

- [ ] **Step 2: Run the existing tests to confirm no regression**

Run: `pytest tests/test_zarr_output_archive.py -v`
Expected: PASS — all 10 Phase 2 tests still pass; the class attribute does not change behavior for direct `add_array(3D)` calls.

- [ ] **Step 3: Commit**

```bash
git add python/lsst/images/zarr/_output_archive.py
git commit -m "feat: opt ZarrOutputArchive into native 3-D mask serialization"
```

### Task 3.1: `ZarrInputArchive` skeleton — open + `get_tree` + error taxonomy

**Files:**
- Create: `python/lsst/images/zarr/_input_archive.py`
- Test: `tests/test_zarr_input_archive.py`

Constructor takes a `ZarrDocument` (built lazily via `from_zarr`). `get_tree(model_type)` reads `/tree`'s bytes and validates them. The `open` classmethod is a context manager around `_store.open_store_for_read`.

Error taxonomy (per spec §4):
- Missing `lsst.archive_class` → `ArchiveReadError("File is not an LSST zarr archive")`.
- `lsst.version` newer than `LSST_VERSION` → `ArchiveReadError("Unsupported lsst:version <N>")`.

`ZarrAttributes.load` keeps the on-disk `version` under a private sentinel `__version_remembered_at_load__` so the input archive can validate without going back to the raw store.

- [ ] **Step 1: Update `ZarrAttributes.load` / `dump` to round-trip the version sentinel**

In `python/lsst/images/zarr/_model.py`, change `ZarrAttributes.load` to keep the version under a private key, and `dump` to ignore that key:

```python
    @classmethod
    def load(cls, raw: dict[str, Any]) -> Self:
        lsst = dict(raw.get(LSST_NS, {}))
        version = lsst.pop("version", None)
        if version is not None:
            lsst["__version_remembered_at_load__"] = version
        ome = dict(raw.get(OME_NS, {}))
        ome.pop("version", None)
        extra = {k: v for k, v in raw.items() if k not in (LSST_NS, OME_NS)}
        return cls(lsst=lsst, ome=ome, extra=extra)

    def dump(self) -> dict[str, Any]:
        out: dict[str, Any] = dict(self.extra)
        public_lsst = {
            k: v for k, v in self.lsst.items() if not k.startswith("__")
        }
        out[LSST_NS] = {"version": LSST_VERSION, **public_lsst}
        if self.ome:
            out[OME_NS] = {"version": OME_VERSION, **self.ome}
        return out
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_zarr_input_archive.py`:

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

import numpy as np

try:
    import zarr

    from lsst.images.serialization import ArchiveReadError
    from lsst.images.zarr._common import LSST_NS, LSST_VERSION
    from lsst.images.zarr._input_archive import ZarrInputArchive
    from lsst.images.zarr._model import ZarrDocument

    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrInputArchiveSkeletonTestCase(unittest.TestCase):
    def test_open_reads_tree(self) -> None:
        from lsst.images import Box, Image
        from lsst.images.zarr import write
        from lsst.images._image import ImageSerializationModel

        original = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "out.zarr")
            write(original, target)
            with ZarrInputArchive.open(target) as archive:
                tree = archive.get_tree(ImageSerializationModel)
                self.assertIsNotNone(tree)

    def test_missing_archive_class_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "bare.zarr")
            os.makedirs(target)
            store = zarr.storage.LocalStore(target, read_only=False)
            zarr.create_group(store=store, zarr_format=3)  # no lsst attrs
            with self.assertRaisesRegex(ArchiveReadError, "not an LSST zarr archive"):
                with ZarrInputArchive.open(target):
                    pass

    def test_future_version_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "future.zarr")
            os.makedirs(target)
            store = zarr.storage.LocalStore(target, read_only=False)
            root = zarr.create_group(store=store, zarr_format=3)
            root.update_attributes(
                {
                    LSST_NS: {
                        "version": LSST_VERSION + 1,
                        "archive_class": "Image",
                        "tree": "tree",
                    }
                }
            )
            with self.assertRaisesRegex(ArchiveReadError, "Unsupported lsst:version"):
                with ZarrInputArchive.open(target):
                    pass


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `pytest tests/test_zarr_input_archive.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 4: Write `_input_archive.py`**

Create `python/lsst/images/zarr/_input_archive.py`:

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

__all__ = ("ZarrInputArchive", "read")

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import EllipsisType
from typing import Any, Self

import astropy.io.fits
import astropy.table
import numpy as np

from lsst.resources import ResourcePathExpression

from .._transforms import FrameSet
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
from ._common import LSST_VERSION, ZarrPointerModel
from ._model import ZarrArray, ZarrDocument


class ZarrInputArchive(InputArchive[ZarrPointerModel]):
    """Reads zarr archives written by `ZarrOutputArchive`."""

    def __init__(self, document: ZarrDocument) -> None:
        self._document = document
        self._validate_root_attributes()
        self._deserialized_pointer_cache: dict[str, Any] = {}
        self._frame_set_cache: dict[str, FrameSet] = {}

    @classmethod
    @contextmanager
    def open(cls, path: ResourcePathExpression) -> Iterator[Self]:
        """Open a zarr archive for reading."""
        from ._store import open_store_for_read

        with open_store_for_read(path) as store:
            doc = ZarrDocument.from_zarr(store)
            yield cls(doc)

    @property
    def document(self) -> ZarrDocument:
        return self._document

    def get_tree[T: ArchiveTree](self, model_type: type[T]) -> T:
        """Read and validate the main Pydantic tree at ``/tree``."""
        try:
            node = self._document.root.get("/tree")
        except KeyError:
            raise ArchiveReadError(
                "File has no /tree array; this is not an LSST zarr archive."
            ) from None
        if not isinstance(node, ZarrArray):
            raise ArchiveReadError("/tree must be a zarr array, not a group.")
        json_bytes = bytes(node.read())
        return model_type.model_validate_json(json_bytes.decode("utf-8"))

    def _validate_root_attributes(self) -> None:
        attrs = self._document.root.attributes.lsst
        if "archive_class" not in attrs:
            raise ArchiveReadError(
                "File is not an LSST zarr archive (missing lsst.archive_class)."
            )
        version = attrs.get("__version_remembered_at_load__", LSST_VERSION)
        if version > LSST_VERSION:
            raise ArchiveReadError(
                f"Unsupported lsst:version {version}; this reader supports up "
                f"to {LSST_VERSION}."
            )

    # The remaining abstract methods land in subsequent tasks.
    def deserialize_pointer(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        raise NotImplementedError("deserialize_pointer lands in Task 3.3")

    def get_frame_set(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        raise NotImplementedError("get_frame_set lands in Task 3.3")

    def get_array(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        raise NotImplementedError("get_array lands in Task 3.2")

    def get_table(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        raise NotImplementedError("get_table lands in Task 3.4")

    def get_structured_array(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        raise NotImplementedError("get_structured_array lands in Task 3.4")


def read(*args: Any, **kwargs: Any) -> Any:
    """Public read helper. Implemented in Task 3.5."""
    raise NotImplementedError("read() lands in Task 3.5")
```

- [ ] **Step 5: Run all relevant tests**

Run: `pytest tests/test_zarr_input_archive.py tests/test_zarr_model.py -v`
Expected: PASS — input archive skeleton tests pass; the version-sentinel update does not break model tests.

- [ ] **Step 6: Commit**

```bash
git add python/lsst/images/zarr/_input_archive.py python/lsst/images/zarr/_model.py tests/test_zarr_input_archive.py
git commit -m "feat: add ZarrInputArchive skeleton with get_tree and version validation"
```

### Task 3.2: `get_array` — lazy slice forwarding + mask unpack

**Files:**
- Modify: `python/lsst/images/zarr/_input_archive.py`
- Modify: `tests/test_zarr_input_archive.py`

`get_array(model, slices=...)`:

1. Resolve the model's `source` (always plain `zarr:/<path>` — no query suffix).
2. Fetch the `ZarrArray` IR node — still lazy.
3. **Mask unpack:** if the model claims a 3-D `(y, x, mask_size)` shape but the on-disk array is 2-D and carries `flag_masks` attribute, slice the 2-D array first (forwarding `slices` if it has rank 2; or its `slices[:-1]` if rank 3 was requested) and unpack via bit shifts to reconstruct the 3-D mask.
4. Otherwise call `ir_array.read(slices=slices)`, forwarding directly to the lazy handle.

The lazy invariant test uses `_CountingStore` to count chunk fetches and asserts a single-chunk subset of a 16×16 / chunks=(4,4) array touches strictly fewer keys than a full read.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_zarr_input_archive.py`:

```python
class _CountingStore(zarr.storage.MemoryStore if HAVE_ZARR else object):
    """A MemoryStore that counts get() calls."""

    def __init__(self) -> None:
        super().__init__()
        self.reads = 0

    async def get(self, key, prototype, byte_range=None):  # type: ignore[override]
        self.reads += 1
        return await super().get(key, prototype, byte_range)


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrInputArchiveLazySubsetTestCase(unittest.TestCase):
    """Lazy-subset invariant: subset reads only fetch touched chunks."""

    def test_subset_read_touches_only_intersecting_chunks(self) -> None:
        from lsst.images.serialization import ArrayReferenceModel, NumberType

        store = _CountingStore()
        root = zarr.create_group(store=store, zarr_format=3)
        root.update_attributes(
            {
                LSST_NS: {
                    "version": LSST_VERSION,
                    "archive_class": "Image",
                    "tree": "tree",
                }
            }
        )
        zarr_array = root.create_array(
            name="image", shape=(16, 16), chunks=(4, 4), dtype="float32"
        )
        zarr_array[:] = np.arange(256, dtype=np.float32).reshape(16, 16)
        # Stub /tree so the input archive's constructor accepts the file.
        root.create_array(name="tree", shape=(2,), chunks=(2,), dtype="uint8")[:] = b"{}"

        doc = ZarrDocument.from_zarr(store)
        archive = ZarrInputArchive(doc)

        store.reads = 0
        full_ref = ArrayReferenceModel(
            source="zarr:/image",
            shape=[16, 16],
            datatype=NumberType.from_numpy(np.dtype("float32")),
        )
        full = archive.get_array(full_ref)
        full_reads = store.reads
        self.assertEqual(full.shape, (16, 16))

        store.reads = 0
        subset = archive.get_array(full_ref, slices=(slice(0, 4), slice(0, 4)))
        subset_reads = store.reads
        self.assertEqual(subset.shape, (4, 4))
        np.testing.assert_array_equal(subset, np.arange(256).reshape(16, 16)[:4, :4])
        self.assertLess(subset_reads, full_reads)


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrInputArchiveMaskUnpackTestCase(unittest.TestCase):
    """Round-trip a packed 2-D mask through get_array's unpack path."""

    def test_unpack_2d_packed_back_to_3d(self) -> None:
        from lsst.images.serialization import ArrayReferenceModel, NumberType

        # Build an archive that has a 2-D packed mask on disk.
        store = zarr.storage.MemoryStore()
        root = zarr.create_group(store=store, zarr_format=3)
        root.update_attributes(
            {
                LSST_NS: {
                    "version": LSST_VERSION,
                    "archive_class": "Mask",
                    "tree": "tree",
                }
            }
        )
        # 4x5 mask, 3 planes -> packed in uint8.
        on_disk = np.zeros((4, 5), dtype=np.uint8)
        on_disk[0, 0] = 0b001  # plane 0
        on_disk[1, 1] = 0b110  # planes 1+2
        mask_array = root.create_array(
            name="mask", shape=(4, 5), chunks=(4, 5), dtype="uint8"
        )
        mask_array[:] = on_disk
        mask_array.update_attributes(
            {
                "_ARRAY_DIMENSIONS": ["y", "x"],
                "flag_masks": [1, 2, 4],
                "flag_meanings": "BAD SAT CR",
                "flag_descriptions": ["Bad pixel.", "Saturated.", "Cosmic ray."],
            }
        )
        root.create_array(name="tree", shape=(2,), chunks=(2,), dtype="uint8")[:] = b"{}"

        doc = ZarrDocument.from_zarr(store)
        archive = ZarrInputArchive(doc)

        # The model claims a 3-D shape (mask_size = 1 because <=8 planes).
        model = ArrayReferenceModel(
            source="zarr:/mask",
            shape=[4, 5, 1],
            datatype=NumberType.from_numpy(np.dtype("uint8")),
        )
        result = archive.get_array(model)
        self.assertEqual(result.shape, (4, 5, 1))
        self.assertEqual(result[0, 0, 0], 0b001)
        self.assertEqual(result[1, 1, 0], 0b110)

    def test_unpack_uint64_with_5_bytes(self) -> None:
        from lsst.images.serialization import ArrayReferenceModel, NumberType

        # 40 planes packed into uint64 -> mask_size = 5.
        store = zarr.storage.MemoryStore()
        root = zarr.create_group(store=store, zarr_format=3)
        root.update_attributes(
            {
                LSST_NS: {
                    "version": LSST_VERSION,
                    "archive_class": "Mask",
                    "tree": "tree",
                }
            }
        )
        on_disk = np.zeros((4, 5), dtype=np.uint64)
        on_disk[0, 0] = 0x01_02_03_04_05  # arbitrary bit pattern
        mask_array = root.create_array(
            name="mask", shape=(4, 5), chunks=(4, 5), dtype="uint64"
        )
        mask_array[:] = on_disk
        mask_array.update_attributes(
            {
                "_ARRAY_DIMENSIONS": ["y", "x"],
                "flag_masks": [1 << i for i in range(40)],
                "flag_meanings": " ".join(f"P{i}" for i in range(40)),
                "flag_descriptions": [f"Plane {i}." for i in range(40)],
            }
        )
        root.create_array(name="tree", shape=(2,), chunks=(2,), dtype="uint8")[:] = b"{}"

        doc = ZarrDocument.from_zarr(store)
        archive = ZarrInputArchive(doc)

        model = ArrayReferenceModel(
            source="zarr:/mask",
            shape=[4, 5, 5],
            datatype=NumberType.from_numpy(np.dtype("uint8")),
        )
        result = archive.get_array(model)
        self.assertEqual(result.shape, (4, 5, 5))
        # Bytes recovered from the packed uint64.
        self.assertEqual(result[0, 0, 0], 0x05)  # low byte
        self.assertEqual(result[0, 0, 1], 0x04)
        self.assertEqual(result[0, 0, 2], 0x03)
        self.assertEqual(result[0, 0, 3], 0x02)
        self.assertEqual(result[0, 0, 4], 0x01)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_zarr_input_archive.py -v`
Expected: FAIL — `get_array` raises `NotImplementedError` for both new test classes.

- [ ] **Step 3: Implement `get_array`**

In `python/lsst/images/zarr/_input_archive.py`, replace the `get_array` placeholder:

```python
    def get_array(
        self,
        model: ArrayReferenceModel | InlineArrayModel,
        *,
        slices: tuple[slice, ...] | EllipsisType = ...,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> np.ndarray:
        if isinstance(model, InlineArrayModel):
            data: np.ndarray = np.array(model.data, dtype=model.datatype.to_numpy())
            return data if slices is ... else data[slices]
        if not isinstance(model.source, str) or not model.source.startswith("zarr:"):
            raise ArchiveReadError(
                f"ZarrInputArchive cannot resolve array source {model.source!r}; "
                f"expected a 'zarr:<path>' reference."
            )
        zarr_path = model.source[len("zarr:") :]
        try:
            node = self._document.root.get(zarr_path)
        except KeyError:
            raise ArchiveReadError(f"Array reference {zarr_path!r} not in store.") from None
        if not isinstance(node, ZarrArray):
            raise ArchiveReadError(f"{zarr_path!r} is not an array.")

        # Mask unpack: model claims 3-D (y, x, mask_size); on-disk is 2-D
        # (y, x) packed wide-int with flag_masks attribute.
        claimed_shape = tuple(model.shape) if model.shape is not None else None
        if (
            claimed_shape is not None
            and len(claimed_shape) == 3
            and len(node.shape) == 2
            and "flag_masks" in node.attributes.extra
        ):
            return self._read_packed_mask(node, claimed_shape, slices)

        # Standard path: forward slices straight to the lazy handle.
        return node.read(slices=slices)

    def _read_packed_mask(
        self,
        node: ZarrArray,
        claimed_shape: tuple[int, ...],
        slices: tuple[slice, ...] | EllipsisType,
    ) -> np.ndarray:
        """Unpack a 2-D wide-int mask back to 3-D ``(y, x, mask_size)``.

        ``slices`` is forwarded to the underlying handle as-is when it
        has rank 2; rank-3 slices have their last axis stripped and
        re-applied after the unpack.
        """
        mask_size = claimed_shape[2]
        # Forward 2-D slice to the lazy handle; only intersecting
        # chunks are fetched even on remote stores.
        if slices is ...:
            spatial_slices: tuple[slice, ...] | EllipsisType = ...
            byte_slice: slice | EllipsisType = ...
        elif len(slices) == 3:
            spatial_slices = slices[:2]
            byte_slice = slices[2]
        else:
            spatial_slices = slices
            byte_slice = ...
        packed = node.read(slices=spatial_slices)
        # Unpack: low byte first.
        out = np.empty(packed.shape + (mask_size,), dtype=np.uint8)
        for i in range(mask_size):
            out[..., i] = (packed >> np.uint64(8 * i)) & np.uint64(0xFF)
        if byte_slice is ...:
            return out
        return out[..., byte_slice]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_zarr_input_archive.py -v`
Expected: PASS — lazy-subset invariant holds, mask unpack recovers both single-byte and five-byte packings.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/zarr/_input_archive.py tests/test_zarr_input_archive.py
git commit -m "feat: implement ZarrInputArchive.get_array with lazy slices and mask unpack"
```

### Task 3.3: `deserialize_pointer`, `get_frame_set`, AST WCS reconstruction

**Files:**
- Modify: `python/lsst/images/zarr/_input_archive.py`
- Modify: `tests/test_zarr_input_archive.py`

`deserialize_pointer(pointer, model_type, deserializer)`:

1. Cache hit by `pointer.path` → return cached object.
2. Read JSON bytes at `pointer.path` (a `ZarrArray` of `uint8`).
3. Validate via `model_type.model_validate_json` and call `deserializer(model, self)`.
4. Cache the result; if it is a `FrameSet`, also cache it under `_frame_set_cache` so `get_frame_set` can return it.

For `Projection.deserialize` to find the AST WCS, the Projection serialization model carries a `ZarrPointerModel` referencing `/wcs_ast` (set by `add_tree` in Phase 2). When that pointer is deserialized, the deserializer reads the AST string bytes via `get_array` (the `wcs_ast` array is plain `uint8` so `get_array` returns it as-is) and reconstructs the FrameSet with `astshim.Object.fromString`.

The AST reconstruction is performed inside the projection deserializer, not the input archive — but the input archive needs to expose the bytes at `/wcs_ast` so the deserializer can call `get_array` on it. That happens automatically since `/wcs_ast` is just a regular zarr array.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_zarr_input_archive.py`:

```python
@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrInputArchivePointerTestCase(unittest.TestCase):
    def test_deserialize_pointer_caches_results(self) -> None:
        import pydantic

        from lsst.images.zarr._common import ZarrPointerModel

        class _Sub(pydantic.BaseModel):
            label: str

        store = zarr.storage.MemoryStore()
        root = zarr.create_group(store=store, zarr_format=3)
        root.update_attributes(
            {LSST_NS: {"version": LSST_VERSION, "archive_class": "Image", "tree": "tree"}}
        )
        # Stub /tree.
        root.create_array(name="tree", shape=(2,), chunks=(2,), dtype="uint8")[:] = b"{}"
        # Sub-archive with its own /tree at /psf/tree.
        json_bytes = b'{"label": "psf"}'
        psf = root.create_group("psf")
        arr = psf.create_array(
            name="tree",
            shape=(len(json_bytes),),
            chunks=(len(json_bytes),),
            dtype="uint8",
        )
        arr[:] = np.frombuffer(json_bytes, dtype=np.uint8)

        doc = ZarrDocument.from_zarr(store)
        archive = ZarrInputArchive(doc)

        deserialize_calls: list[int] = []

        def deserializer(model, arch):  # noqa: ANN001
            deserialize_calls.append(1)
            return model

        pointer = ZarrPointerModel(path="/psf/tree")
        first = archive.deserialize_pointer(pointer, _Sub, deserializer)
        second = archive.deserialize_pointer(pointer, _Sub, deserializer)
        self.assertEqual(first.label, "psf")
        self.assertIs(first, second)
        self.assertEqual(len(deserialize_calls), 1)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_zarr_input_archive.py::ZarrInputArchivePointerTestCase -v`
Expected: FAIL — `deserialize_pointer` raises `NotImplementedError`.

- [ ] **Step 3: Implement `deserialize_pointer` and `get_frame_set`**

Replace the placeholders in `python/lsst/images/zarr/_input_archive.py`:

```python
    def deserialize_pointer[U: ArchiveTree, V](
        self,
        pointer: ZarrPointerModel,
        model_type: type[U],
        deserializer: Callable[[U, InputArchive[ZarrPointerModel]], V],
    ) -> V:
        if (cached := self._deserialized_pointer_cache.get(pointer.path)) is not None:
            return cached
        try:
            node = self._document.root.get(pointer.path)
        except KeyError:
            raise ArchiveReadError(
                f"Pointer reference {pointer.path!r} not in store."
            ) from None
        if not isinstance(node, ZarrArray):
            raise ArchiveReadError(f"Pointer target {pointer.path!r} is not an array.")
        json_text = bytes(node.read()).decode("utf-8")
        model = model_type.model_validate_json(json_text)
        result = deserializer(model, self)
        self._deserialized_pointer_cache[pointer.path] = result
        if isinstance(result, FrameSet):
            self._frame_set_cache[pointer.path] = result
        return result

    def get_frame_set(self, pointer: ZarrPointerModel) -> FrameSet:
        try:
            return self._frame_set_cache[pointer.path]
        except KeyError:
            raise AssertionError(
                f"Frame set at {pointer.path!r} must be deserialised via "
                f"deserialize_pointer before any dependent transform can be."
            ) from None
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_zarr_input_archive.py -v`
Expected: PASS — pointer-cache test asserts the deserializer is called exactly once.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/zarr/_input_archive.py tests/test_zarr_input_archive.py
git commit -m "feat: implement deserialize_pointer and get_frame_set"
```

### Task 3.4: `get_table`, `get_structured_array`

**Files:**
- Modify: `python/lsst/images/zarr/_input_archive.py`
- Modify: `tests/test_zarr_input_archive.py`

Mirrors the FITS implementation: each column is a separate `ArrayReferenceModel(source=f"zarr:/lsst/tables/<name>/<column>")` resolved via `get_array`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_zarr_input_archive.py`:

```python
@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrInputArchiveTableTestCase(unittest.TestCase):
    def test_get_table_reconstructs_columns(self) -> None:
        import astropy.table

        from lsst.images.zarr._model import ZarrArray
        from lsst.images.zarr._output_archive import ZarrOutputArchive

        out = ZarrOutputArchive()
        # Wire up the LSST root attributes.
        out.document.root.attributes.lsst["archive_class"] = "Image"
        out.document.root.attributes.lsst["tree"] = "tree"
        out.document.root.arrays["tree"] = ZarrArray(
            data=np.frombuffer(b"{}", dtype=np.uint8)
        )
        original = astropy.table.Table(
            {
                "x": np.arange(4, dtype=np.int32),
                "y": np.arange(4, dtype=np.float32),
            }
        )
        model = out.add_table(original, name="cat")

        store = zarr.storage.MemoryStore()
        out.document.to_zarr(store)
        doc = ZarrDocument.from_zarr(store)
        inp = ZarrInputArchive(doc)

        recovered = inp.get_table(model)
        self.assertEqual(recovered.colnames, ["x", "y"])
        np.testing.assert_array_equal(recovered["x"], original["x"])
        np.testing.assert_array_equal(recovered["y"], original["y"])
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_zarr_input_archive.py::ZarrInputArchiveTableTestCase -v`
Expected: FAIL — `get_table` raises `NotImplementedError`.

- [ ] **Step 3: Implement `get_table` and `get_structured_array`**

Replace the placeholders:

```python
    def get_table(
        self,
        model: TableModel,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> astropy.table.Table:
        result = astropy.table.Table(meta=model.meta)
        for column_model in model.columns:
            if isinstance(column_model.data, InlineArrayModel):
                data: Any = column_model.data.data
            else:
                data = self.get_array(column_model.data, strip_header=strip_header)
            result[column_model.name] = astropy.table.Column(
                data,
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
```

- [ ] **Step 4: Run the test**

Run: `pytest tests/test_zarr_input_archive.py -v`
Expected: PASS — all input-archive tests pass.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/zarr/_input_archive.py tests/test_zarr_input_archive.py
git commit -m "feat: implement ZarrInputArchive.get_table and get_structured_array"
```

### Task 3.5: Public `read()` helper

**Files:**
- Modify: `python/lsst/images/zarr/_input_archive.py`
- Modify: `python/lsst/images/zarr/__init__.py`
- Modify: `tests/test_zarr_input_archive.py`

`read(cls, path, **kwargs)` opens a `ZarrInputArchive`, calls `archive.get_tree(cls._get_archive_tree_type(ZarrPointerModel))`, and returns `ReadResult(tree.deserialize(archive, **kwargs), tree.metadata, tree.butler_info)`. No auto-detect path in v1 — files without `lsst.archive_class` raise.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_zarr_input_archive.py`:

```python
@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrReadHelperTestCase(unittest.TestCase):
    def test_round_trip_image(self) -> None:
        from lsst.images import Box, Image
        from lsst.images.zarr import read, write

        original = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "out.zarr")
            write(original, target)
            result = read(Image, target)
            self.assertEqual(result.deserialized.array.shape, (4, 5))
            np.testing.assert_array_equal(result.deserialized.array, original.array)
            self.assertEqual(result.deserialized.bbox, original.bbox)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_zarr_input_archive.py::ZarrReadHelperTestCase -v`
Expected: FAIL — `read()` raises `NotImplementedError`.

- [ ] **Step 3: Implement `read`**

Replace the placeholder in `_input_archive.py`:

```python
def read[T: Any](cls: type[T], path: ResourcePathExpression, **kwargs: Any) -> ReadResult[T]:
    """Read an object from a zarr archive.

    The archive's root attributes name the in-memory class via
    ``lsst.archive_class``. Files without this attribute raise; auto-
    detect of foreign zarr files is a follow-up.
    """
    with ZarrInputArchive.open(path) as archive:
        tree_type = cls._get_archive_tree_type(ZarrPointerModel)
        tree = archive.get_tree(tree_type)
        obj = tree.deserialize(archive, **kwargs)
        return ReadResult(obj, tree.metadata, tree.butler_info)
```

Re-export from `python/lsst/images/zarr/__init__.py`:

```python
from ._common import *  # noqa: F401, F403
from ._input_archive import *  # noqa: F401, F403
from ._output_archive import *  # noqa: F401, F403
```

- [ ] **Step 4: Run the round-trip test**

Run: `pytest tests/test_zarr_input_archive.py::ZarrReadHelperTestCase -v`
Expected: PASS — `Image` round-trips via `write` + `read`.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/zarr/_input_archive.py python/lsst/images/zarr/__init__.py tests/test_zarr_input_archive.py
git commit -m "feat: add public zarr.read() helper"
```

### Task 3.6: `RoundtripZarr` test helper + round-trips for Image / MaskedImage / VisitImage

**Files:**
- Modify: `python/lsst/images/tests/_roundtrip.py` (add `RoundtripZarr`)
- Create: `tests/test_zarr_round_trip.py`

`RoundtripZarr` lets the existing `RoundtripBase` pattern exercise the zarr backend the same way it does FITS / JSON / NDF. The new test file uses it to round-trip the three image types covered by Phase 2.

- [ ] **Step 1: Add `RoundtripZarr` to `_roundtrip.py`**

Edit `python/lsst/images/tests/_roundtrip.py`. Add `"RoundtripZarr"` to `__all__`, then append after `RoundtripNdf`:

```python
class RoundtripZarr[T](RoundtripBase[T]):
    def inspect(self) -> Any:
        """Open the zarr archive's IR for inspection."""
        import zarr

        from lsst.images.zarr._model import ZarrDocument

        return ZarrDocument.from_zarr(
            zarr.storage.LocalStore(self.filename, read_only=True)
        )

    def _get_extension(self) -> str:
        return ".zarr"

    def _write(self, obj: Any, filename: str) -> ArchiveTree:
        from .. import zarr as zarr_backend

        return zarr_backend.write(obj, filename)

    def _read(self, obj_type: Any, filename: str) -> ReadResult:
        from .. import zarr as zarr_backend

        return zarr_backend.read(obj_type, filename)
```

If `RoundtripBase` constructs the on-disk path with `tempfile.NamedTemporaryFile`, audit it for directory-vs-file assumptions: a zarr archive is a directory when `_get_extension()` returns `.zarr`. Mirror what NDF does with `.sdf` (single file) but extend to handle the directory case — likely a `tempfile.TemporaryDirectory` used as the parent and the archive path joined under it.

- [ ] **Step 2: Write the failing test**

Create `tests/test_zarr_round_trip.py`:

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

import numpy as np

try:
    import zarr  # noqa: F401

    from lsst.images.tests import RoundtripZarr

    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrRoundTripTestCase(unittest.TestCase):
    def test_image_round_trip(self) -> None:
        from lsst.images import Box, Image

        original = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )
        with RoundtripZarr(self, original) as roundtrip:
            recovered = roundtrip.recovered
            np.testing.assert_array_equal(recovered.array, original.array)
            self.assertEqual(recovered.bbox, original.bbox)

    def test_masked_image_round_trip(self) -> None:
        from lsst.images import Box, Image, MaskedImage, MaskPlane, MaskSchema

        schema = MaskSchema(
            [
                MaskPlane("BAD", "Bad pixel."),
                MaskPlane("SAT", "Saturated."),
                MaskPlane("CR", "Cosmic ray."),
            ]
        )
        image = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )
        original = MaskedImage(image, mask_schema=schema)
        original.mask.set("BAD", image.array % 2 == 0)
        original.mask.set("SAT", image.array > 10)

        with RoundtripZarr(self, original) as roundtrip:
            recovered = roundtrip.recovered
            np.testing.assert_array_equal(recovered.image.array, original.image.array)
            np.testing.assert_array_equal(recovered.mask.array, original.mask.array)

    def test_masked_image_with_40_planes_round_trip(self) -> None:
        from lsst.images import Box, Image, MaskedImage, MaskPlane, MaskSchema

        schema = MaskSchema([MaskPlane(f"P{i}", f"Plane {i}.") for i in range(40)])
        image = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )
        original = MaskedImage(image, mask_schema=schema)
        original.mask.set("P0", image.array % 2 == 0)
        original.mask.set("P39", image.array > 10)

        with RoundtripZarr(self, original) as roundtrip:
            recovered = roundtrip.recovered
            # 40 planes packed into uint64 on disk, unpacked to 5 bytes per pixel.
            np.testing.assert_array_equal(recovered.mask.array, original.mask.array)

    def test_visit_image_round_trip(self) -> None:
        from lsst.images import Box, Image, MaskPlane, MaskSchema, VisitImage

        schema = MaskSchema([MaskPlane("BAD", "Bad pixel.")])
        image = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )
        original = VisitImage(image=image, mask_schema=schema)

        with RoundtripZarr(self, original) as roundtrip:
            recovered = roundtrip.recovered
            np.testing.assert_array_equal(recovered.image.array, original.image.array)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: Run the tests**

Run: `pytest tests/test_zarr_round_trip.py -v`
Expected: PASS — all four round-trips. If a test fails because some per-class detail is missing (e.g. a `lsst.companions` style key our `add_tree` should set, or a Projection deserializer that needs to find `/wcs_ast`), fix it in `_input_archive.py` / `_output_archive.py` and re-run. The 40-plane test is the load-bearing assertion that the wide-int packing + unpack round-trip is bit-exact.

- [ ] **Step 4: Commit**

```bash
git add python/lsst/images/tests/_roundtrip.py tests/test_zarr_round_trip.py
git commit -m "test: round-trip Image, MaskedImage (3- and 40-plane), VisitImage through zarr"
```

---

**End of Phase 3.** Seven tasks. Read side complete for `Image` / `MaskedImage` / `VisitImage`, lazy-subset invariant pinned by `_CountingStore`, mask unpack pinned by both 3-plane (uint8) and 40-plane (uint64) tests, full write→read round-trips green. Phase 4 adds `ColorImage` (recursive sub-archives) and `CellCoadd` (cell-aligned chunks + native 4-D PSF).

## Phase 4 — `ColorImage` and `CellCoadd`

This phase adds the two archive classes whose layouts go beyond the flat `image`/`variance`/`mask` siblings:

- **`ColorImage`**: red/green/blue sub-archives. Each is itself a valid Image-shaped sub-archive (its own `image` array, its own OME multiscales, its own `lsst.archive_class = "Image"`). The root group has `lsst.archive_class = "ColorImage"` and **no** OME multiscales of its own.
- **`CellCoadd`**: `image`/`variance`/`mask` siblings (cell-aligned chunks) plus a 4-D `psf` array `(Cy, Cx, Py, Px)` with single-cell chunks `(1, 1, Py, Px)`. `lsst.cell_grid = {bbox, cell_shape}` on the root attrs.

The recurring theme: **no fixup pass**. Each `add_array` call lands at the path its `name` argument names. Per-archive-class attribute decoration runs once in `add_tree` against the populated IR.

### Task 4.1: Recursive sub-archive attribute decoration

**Files:**
- Modify: `python/lsst/images/zarr/_layout.py` (add `decorate_sub_archives`)
- Modify: `python/lsst/images/zarr/_output_archive.py` (call it from `add_tree`)
- Modify: `tests/test_zarr_layout.py`
- Modify: `tests/test_zarr_output_archive.py`

For ColorImage's `red/`, `green/`, `blue/` to be valid OME-NGFF / xarray groups in their own right, each needs `lsst.archive_class = "Image"` and an `ome.multiscales` block pointing at its `image` array. The decoration is purely metadata — no bytes move.

The detection rule for "this sub-group is a sub-archive": it contains an `image` array (any rank). The decoration is recursive — sub-sub-archives (e.g. a Projection's parameter image inside a PSF sub-archive) get the same treatment.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_zarr_layout.py`:

```python
@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class DecorateSubArchivesTestCase(unittest.TestCase):
    def test_sub_group_with_image_gets_lsst_and_ome_attrs(self) -> None:
        import numpy as np

        from lsst.images.zarr._layout import decorate_sub_archives
        from lsst.images.zarr._model import ZarrArray, ZarrDocument, ZarrGroup

        doc = ZarrDocument(root=ZarrGroup())
        doc.root.attributes.lsst["archive_class"] = "ColorImage"
        # red sub-archive with its own image array.
        red = doc.root.ensure_group("/red")
        red.arrays["image"] = ZarrArray(data=np.ones((4, 5), dtype="float32"))

        decorate_sub_archives(doc)

        self.assertEqual(red.attributes.lsst["archive_class"], "Image")
        self.assertIn("multiscales", red.attributes.ome)
        self.assertEqual(
            red.attributes.ome["multiscales"][0]["datasets"][0]["path"], "image"
        )

    def test_root_archive_class_is_unchanged(self) -> None:
        import numpy as np

        from lsst.images.zarr._layout import decorate_sub_archives
        from lsst.images.zarr._model import ZarrArray, ZarrDocument, ZarrGroup

        doc = ZarrDocument(root=ZarrGroup())
        doc.root.attributes.lsst["archive_class"] = "ColorImage"
        red = doc.root.ensure_group("/red")
        red.arrays["image"] = ZarrArray(data=np.ones((4, 5), dtype="float32"))

        decorate_sub_archives(doc)

        # Root keeps ColorImage; only sub-groups are decorated.
        self.assertEqual(doc.root.attributes.lsst["archive_class"], "ColorImage")
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_zarr_layout.py::DecorateSubArchivesTestCase -v`
Expected: FAIL — `decorate_sub_archives` does not exist.

- [ ] **Step 3: Implement the decoration pass**

Append to `python/lsst/images/zarr/_layout.py`:

```python
__all__ = (
    "AffineCheckResult",
    "affine_check",
    "axes_for_archive_class",
    "chunks_aligned_to",
    "chunks_for",
    "decorate_sub_archives",
)


def decorate_sub_archives(document: "ZarrDocument") -> None:
    """Walk ``document`` and decorate every sub-archive group with attrs.

    A sub-archive is any group below the root that contains an ``image``
    array. Decoration adds ``lsst.archive_class = "Image"`` and an
    ``ome.multiscales`` block pointing at the sub-archive's ``image``
    array. Recursive: nested sub-archives are decorated too.

    The root group is left alone — its ``lsst.archive_class`` is set
    by ``add_tree`` based on the in-memory object's type.
    """
    from ._model import OmeMultiscale, ZarrDocument, ZarrGroup  # local: avoid cycle

    if not isinstance(document, ZarrDocument):
        raise TypeError(type(document).__name__)
    _decorate_walk(document.root, depth=0)


def _decorate_walk(group: "ZarrGroup", *, depth: int) -> None:
    from ._model import OmeMultiscale, ZarrGroup  # local: avoid cycle

    for name, sub in group.groups.items():
        if "image" in sub.arrays:
            sub.attributes.lsst.setdefault("archive_class", "Image")
            sub.attributes.lsst.setdefault("tree", "tree") if "tree" in sub.arrays else None
            if "multiscales" not in sub.attributes.ome:
                multiscale = OmeMultiscale(
                    name="image",
                    axes=("y", "x"),
                    dataset_path="image",
                )
                sub.attributes.ome["multiscales"] = [multiscale.dump()]
        _decorate_walk(sub, depth=depth + 1)
```

In `python/lsst/images/zarr/_output_archive.py`, call it at the end of `add_tree` (just before the method returns):

```python
        from ._layout import decorate_sub_archives

        decorate_sub_archives(self.document)
```

- [ ] **Step 4: Add an output-archive integration test**

Append to `tests/test_zarr_output_archive.py`:

```python
@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrColorImageWriteTestCase(unittest.TestCase):
    def test_color_image_emits_recursive_sub_archives(self) -> None:
        import os
        import tempfile

        import numpy as np
        import zarr

        from lsst.images import Box, ColorImage, Image
        from lsst.images.zarr import write
        from lsst.images.zarr._common import LSST_NS, OME_NS
        from lsst.images.zarr._model import ZarrDocument

        red = Image(np.full((4, 5), 1, dtype=np.uint8), bbox=Box.factory[10:14, 20:25])
        green = Image(np.full((4, 5), 2, dtype=np.uint8), bbox=red.bbox)
        blue = Image(np.full((4, 5), 3, dtype=np.uint8), bbox=red.bbox)
        color = ColorImage(red=red, green=green, blue=blue)

        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "out.zarr")
            write(color, target)
            with zarr.storage.LocalStore(target, read_only=True) as store:
                doc = ZarrDocument.from_zarr(store)
                # Root: ColorImage, no ome.multiscales (axes_for_archive_class
                # returns () for ColorImage).
                self.assertEqual(
                    doc.root.attributes.lsst["archive_class"], "ColorImage"
                )
                self.assertNotIn("multiscales", doc.root.attributes.ome)
                # Each channel sub-archive has its own image array...
                for channel in ("red", "green", "blue"):
                    sub = doc.root.groups[channel]
                    self.assertIn("image", sub.arrays)
                    self.assertEqual(sub.arrays["image"].shape, (4, 5))
                    # ...and is decorated as a valid Image sub-archive.
                    self.assertEqual(sub.attributes.lsst["archive_class"], "Image")
                    self.assertIn("multiscales", sub.attributes.ome)
                    self.assertEqual(
                        sub.attributes.ome["multiscales"][0]["datasets"][0]["path"],
                        "image",
                    )
```

- [ ] **Step 5: Run all tests**

Run: `pytest tests/test_zarr_layout.py tests/test_zarr_output_archive.py -v`
Expected: PASS — decoration is applied recursively, ColorImage's three channels are valid Image sub-archives.

- [ ] **Step 6: Commit**

```bash
git add python/lsst/images/zarr/_layout.py python/lsst/images/zarr/_output_archive.py tests/test_zarr_layout.py tests/test_zarr_output_archive.py
git commit -m "feat: decorate sub-archives with lsst.archive_class and ome.multiscales"
```

### Task 4.2: ColorImage round-trip

**Files:**
- Modify: `tests/test_zarr_round_trip.py`

The decoration in 4.1 plus the existing `read()` deserializer should round-trip ColorImage with no further code changes. This task asserts that.

- [ ] **Step 1: Write the test**

Append to `tests/test_zarr_round_trip.py`:

```python
@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrColorImageRoundTripTestCase(unittest.TestCase):
    def test_color_image_round_trip(self) -> None:
        from lsst.images import Box, ColorImage, Image

        red = Image(np.full((4, 5), 1, dtype=np.uint8), bbox=Box.factory[10:14, 20:25])
        green = Image(np.full((4, 5), 2, dtype=np.uint8), bbox=red.bbox)
        blue = Image(np.full((4, 5), 3, dtype=np.uint8), bbox=red.bbox)
        original = ColorImage(red=red, green=green, blue=blue)

        with RoundtripZarr(self, original) as roundtrip:
            recovered = roundtrip.recovered
            np.testing.assert_array_equal(recovered.red.array, original.red.array)
            np.testing.assert_array_equal(recovered.green.array, original.green.array)
            np.testing.assert_array_equal(recovered.blue.array, original.blue.array)
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_zarr_round_trip.py::ZarrColorImageRoundTripTestCase -v`
Expected: PASS. If this fails because the ColorImage deserializer needs sub-archive `tree` documents that we are not staging (since we use `serialize_direct`, not `serialize_pointer`), the failure tells you exactly what's missing — adapt the `decorate_sub_archives` pass to also write a per-sub-archive `tree` document if the ColorImage deserializer demands it.

- [ ] **Step 3: Commit**

```bash
git add tests/test_zarr_round_trip.py
git commit -m "test: round-trip ColorImage through the zarr backend"
```

### Task 4.3: CellCoadd PSF — single-cell chunks for the 4-D array

**Files:**
- Modify: `python/lsst/images/zarr/_layout.py` (extend `chunks_for` to accept `axis_hint`)
- Modify: `python/lsst/images/zarr/_output_archive.py` (special-case `name="psf"` to chunk per-cell)
- Modify: `tests/test_zarr_layout.py`
- Modify: `tests/test_zarr_output_archive.py`

CellCoadd's PSF is a 4-D array `(Cy, Cx, Py, Px)` where the leading two axes index cells and the trailing two are the per-cell PSF image. Single-cell reads should be one chunk, so the default chunk shape is `(1, 1, Py, Px)`.

`add_array` recognises `name == "psf"` (or names ending in `/psf`) and applies the single-cell-chunked default if the user has not overridden.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_zarr_output_archive.py`:

```python
@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrPsfChunkingTestCase(unittest.TestCase):
    def test_psf_array_uses_single_cell_chunks(self) -> None:
        import numpy as np

        psf = np.zeros((2, 3, 21, 21), dtype=np.float32)
        archive = ZarrOutputArchive(archive_class="CellCoadd")
        ref = archive.add_array(psf, name="psf")
        self.assertEqual(ref.source, "zarr:/psf")
        node = archive.document.root.get("/psf")
        # Single-cell chunks: leading axes are 1; spatial axes match shape.
        self.assertEqual(tuple(node.chunks), (1, 1, 21, 21))

    def test_psf_user_override_wins(self) -> None:
        import numpy as np

        psf = np.zeros((2, 3, 21, 21), dtype=np.float32)
        archive = ZarrOutputArchive(
            archive_class="CellCoadd",
            chunks={"psf": (2, 3, 21, 21)},
        )
        archive.add_array(psf, name="psf")
        node = archive.document.root.get("/psf")
        self.assertEqual(tuple(node.chunks), (2, 3, 21, 21))
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_zarr_output_archive.py::ZarrPsfChunkingTestCase -v`
Expected: FAIL — current `add_array` defaults to `min(1024, dim)` per axis, giving `(2, 3, 21, 21)` (already small enough) but for larger Cy/Cx the leading axes would not be 1.

- [ ] **Step 3: Implement the special case**

In `python/lsst/images/zarr/_output_archive.py`, edit `add_array` to handle the PSF name. After computing `parent_path` and `leaf` and before staging the `ZarrArray`, add:

```python
        # Default chunks for a CellCoadd-style 4-D PSF: one cell per chunk.
        if (
            chunks is None
            and leaf == "psf"
            and array.ndim == 4
            and parent_path in ("/", "")
        ):
            chunks = (1, 1, array.shape[2], array.shape[3])
```

(Place this after the existing `chunks` resolution chain so user overrides still win.)

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_zarr_output_archive.py::ZarrPsfChunkingTestCase -v`
Expected: PASS — both tests.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/zarr/_output_archive.py tests/test_zarr_output_archive.py
git commit -m "feat: default CellCoadd PSF to single-cell chunks (1, 1, Py, Px)"
```

### Task 4.4: CellCoadd output-archive layout test

**Files:**
- Modify: `tests/test_zarr_output_archive.py`

Pin the on-disk layout for a `CellCoadd`: image / variance / mask siblings with cell-aligned chunks, 4-D PSF with single-cell chunks, `lsst.cell_grid` on the root.

The test's CellCoadd construction is implementer-supplied — the existing `python/lsst/images/cells/_coadd.py` constructor takes a particular set of arguments. The implementer must read it and assemble a minimal valid coadd; the on-disk assertions below stand regardless of constructor specifics.

- [ ] **Step 1: Write the test**

Append to `tests/test_zarr_output_archive.py`:

```python
@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrCellCoaddWriteTestCase(unittest.TestCase):
    def test_cell_coadd_layout(self) -> None:
        import os
        import tempfile

        import zarr

        from lsst.images.zarr import write
        from lsst.images.zarr._model import ZarrDocument

        coadd = _make_minimal_cell_coadd(
            cell_shape=(256, 256),
            shape=(512, 512),
            n_cells=(2, 2),
            psf_shape=(21, 21),
        )

        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "coadd.zarr")
            write(coadd, target)
            with zarr.storage.LocalStore(target, read_only=True) as store:
                doc = ZarrDocument.from_zarr(store)
                # Root archive class.
                self.assertEqual(
                    doc.root.attributes.lsst["archive_class"], "CellCoadd"
                )
                # cell_grid metadata is on the root attrs.
                self.assertIn("cell_grid", doc.root.attributes.lsst)
                cg = doc.root.attributes.lsst["cell_grid"]
                self.assertEqual(tuple(cg["cell_shape"]), (256, 256))
                # image / variance / mask siblings, cell-aligned chunks.
                self.assertEqual(tuple(doc.root.arrays["image"].chunks), (256, 256))
                self.assertEqual(tuple(doc.root.arrays["variance"].chunks), (256, 256))
                self.assertEqual(tuple(doc.root.arrays["mask"].chunks), (256, 256))
                # 4-D psf with single-cell chunks.
                psf = doc.root.arrays["psf"]
                self.assertEqual(psf.shape, (2, 2, 21, 21))
                self.assertEqual(tuple(psf.chunks), (1, 1, 21, 21))


def _make_minimal_cell_coadd(*, cell_shape, shape, n_cells, psf_shape):  # noqa: ANN001, ANN201
    """Construct a minimal CellCoadd for layout testing.

    Implementer: read ``python/lsst/images/cells/_coadd.py`` and
    assemble the smallest valid CellCoadd whose ``cell_shape``,
    overall image shape, cell-grid dimensions, and per-cell PSF
    shape match the requested values. The test only asserts on the
    on-disk layout the write helper produces.
    """
    raise unittest.SkipTest(
        "Implementer: build a minimal CellCoadd per the local ctor."
    )
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_zarr_output_archive.py::ZarrCellCoaddWriteTestCase -v`
Expected: After the implementer replaces `_make_minimal_cell_coadd`, PASS. SKIP otherwise — the placeholder must be replaced before merging this phase.

- [ ] **Step 3: Commit**

```bash
git add tests/test_zarr_output_archive.py
git commit -m "test: pin on-disk zarr layout for CellCoadd"
```

### Task 4.5: CellCoadd round-trip

**Files:**
- Modify: `tests/test_zarr_round_trip.py`

The same minimal CellCoadd factory used in Task 4.4 round-trips through `RoundtripZarr`. Spot-checks the image and one per-cell PSF.

- [ ] **Step 1: Write the test**

Append to `tests/test_zarr_round_trip.py`:

```python
@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrCellCoaddRoundTripTestCase(unittest.TestCase):
    def test_cell_coadd_round_trip(self) -> None:
        original = _make_minimal_cell_coadd_with_psf()  # implementer-supplied
        with RoundtripZarr(self, original) as roundtrip:
            recovered = roundtrip.recovered
            np.testing.assert_array_equal(
                recovered.image.array, original.image.array
            )
            # Spot-check one per-cell PSF if the API exposes them.
            if hasattr(original, "psf") and hasattr(original.psf, "per_cell"):
                np.testing.assert_array_equal(
                    recovered.psf.per_cell[0, 0], original.psf.per_cell[0, 0]
                )


def _make_minimal_cell_coadd_with_psf():  # noqa: ANN201
    """Implementer: assemble a minimal CellCoadd with a 4-D per-cell PSF.

    Reuse `_make_minimal_cell_coadd` from `test_zarr_output_archive.py`
    if the same factory works here, or build one in this file.
    """
    raise unittest.SkipTest(
        "Implementer: assemble a minimal CellCoadd with a per-cell PSF."
    )
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_zarr_round_trip.py::ZarrCellCoaddRoundTripTestCase -v`
Expected: After the implementer replaces the factory, PASS. SKIP otherwise; replace before merging this phase.

- [ ] **Step 3: Commit**

```bash
git add tests/test_zarr_round_trip.py
git commit -m "test: round-trip CellCoadd through the zarr backend"
```

---

**End of Phase 4.** Five tasks. ColorImage writes its three channels as recursive sub-archives (each a valid Image sub-archive with its own OME multiscales), CellCoadd writes flat siblings with cell-aligned chunks plus a 4-D PSF with single-cell chunks. Both types round-trip without any byte duplication or fixup pass. Phase 5 covers FITS↔Zarr opaque-metadata round-trips, xarray interop assertions, and the optional external-reader sanity tests.

## Phase 5 — Cross-format round-trips, xarray interop, external readers

This phase makes the zarr backend a peer of FITS / NDF for round-trip preservation: an object read from FITS carries its primary-HDU header in `_opaque_metadata`, and writing that object to zarr preserves those cards so a later round-trip back to FITS reproduces the original headers byte-for-byte.

It also confirms the **xarray interop contract**: `xr.open_zarr(path)` returns a `Dataset` with `image` / `variance` / `mask` data variables sharing the `(y, x)` dimensions and CF flag attrs surviving on the mask. Two optional external-reader checks (`ngff-validator`, `ome-zarr-py`) round out the phase; both skip silently when their dependencies are absent.

### Task 5.1: Persist `FitsOpaqueMetadata` on write to zarr

**Files:**
- Modify: `python/lsst/images/zarr/_layout.py` (add `serialize_fits_opaque_metadata`)
- Modify: `python/lsst/images/zarr/_output_archive.py` (extend `write` to call it)
- Modify: `tests/test_zarr_output_archive.py`

The opaque metadata lives at `/lsst/opaque_metadata/fits/primary` as a 1-D `uint8` array containing UTF-8 JSON. The JSON encodes the astropy `Header` as a flat `{keyword: value}` dict. The root attribute `lsst.opaque_metadata_format = "fits"` flags its presence.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_zarr_output_archive.py`:

```python
@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrOpaqueMetadataWriteTestCase(unittest.TestCase):
    def test_fits_opaque_metadata_persists(self) -> None:
        import json as _json
        import os
        import tempfile

        import astropy.io.fits
        import numpy as np
        import zarr

        from lsst.images import Box, Image
        from lsst.images.fits._common import ExtensionKey, FitsOpaqueMetadata
        from lsst.images.zarr import write
        from lsst.images.zarr._model import ZarrDocument

        image = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )
        header = astropy.io.fits.Header()
        header["ORIGIN"] = "RUBIN"
        header["EXPTIME"] = 30.0
        opaque = FitsOpaqueMetadata()
        opaque.headers[ExtensionKey()] = header
        image._opaque_metadata = opaque

        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "out.zarr")
            write(image, target)
            with zarr.storage.LocalStore(target, read_only=True) as store:
                doc = ZarrDocument.from_zarr(store)
                self.assertEqual(
                    doc.root.attributes.lsst.get("opaque_metadata_format"),
                    "fits",
                )
                opaque_node = doc.root.get("/lsst/opaque_metadata/fits/primary")
                json_bytes = bytes(opaque_node.read())
                cards = _json.loads(json_bytes)
                self.assertEqual(cards["ORIGIN"], "RUBIN")
                self.assertEqual(cards["EXPTIME"], 30.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_zarr_output_archive.py::ZarrOpaqueMetadataWriteTestCase -v`
Expected: FAIL — `/lsst/opaque_metadata/fits/primary` is not in the store.

- [ ] **Step 3: Implement opaque-metadata serialization**

Append to `python/lsst/images/zarr/_layout.py`:

```python
def serialize_fits_opaque_metadata(document: "ZarrDocument", opaque: Any) -> None:
    """Stage a `FitsOpaqueMetadata` object into the IR.

    Stores the primary-HDU header as a JSON-encoded ``uint8`` array at
    ``/lsst/opaque_metadata/fits/primary`` and sets the
    ``lsst.opaque_metadata_format`` attribute on the root group.
    No-op if the metadata is empty or missing a primary header.
    """
    import json as _json

    import numpy as np

    from ..fits._common import ExtensionKey
    from ._model import ZarrArray

    primary = opaque.headers.get(ExtensionKey())
    if primary is None or len(primary) == 0:
        return
    cards = {card.keyword: card.value for card in primary.cards if card.keyword}
    json_bytes = _json.dumps(cards).encode("utf-8")
    parent = document.root.ensure_group("/lsst/opaque_metadata/fits")
    parent.arrays["primary"] = ZarrArray(
        data=np.frombuffer(json_bytes, dtype=np.uint8)
    )
    document.root.attributes.lsst["opaque_metadata_format"] = "fits"
```

In `python/lsst/images/zarr/_output_archive.py`, extend `write` to call this *after* `add_tree` returns and *before* the IR is materialized:

```python
def write(
    obj: Any,
    path: Any,
    *,
    chunks=None,
    shards=None,
    compression=None,
    metadata=None,
    butler_info=None,
) -> ArchiveTree:
    from ._store import open_store_for_write

    archive_class = type(obj).__name__
    archive_default_name = getattr(obj, "_archive_default_name", None)
    archive_metadata: dict[str, Any] = {}
    if (cell_shape := getattr(obj, "cell_shape", None)) is not None:
        archive_metadata["cell_shape"] = tuple(cell_shape)
    if (cell_grid := getattr(obj, "cell_grid", None)) is not None:
        archive_metadata["cell_grid"] = {
            "bbox": list(cell_grid.bbox) if hasattr(cell_grid, "bbox") else None,
            "cell_shape": list(cell_grid.cell_shape)
            if hasattr(cell_grid, "cell_shape")
            else None,
        }
    if (mask_schema := getattr(obj, "mask_schema", None)) is not None:
        archive_metadata["mask_schema"] = mask_schema

    archive = ZarrOutputArchive(
        chunks=chunks,
        shards=shards,
        compression=compression,
        archive_class=archive_class,
        archive_metadata=archive_metadata,
    )
    if archive_default_name is not None:
        tree = archive.serialize_direct(archive_default_name, obj.serialize)
    else:
        tree = obj.serialize(archive)
    if metadata is not None:
        tree.metadata.update(metadata)
    if butler_info is not None:
        tree.butler_info = butler_info
    archive.add_tree(tree)
    # Stage opaque metadata after add_tree so the namespace attribute
    # writes happen in the right order.
    opaque = getattr(obj, "_opaque_metadata", None)
    if opaque is not None:
        from ._layout import serialize_fits_opaque_metadata

        try:
            serialize_fits_opaque_metadata(archive.document, opaque)
        except ImportError:
            pass  # opaque is not a FITS one; ignore
    with open_store_for_write(path) as store:
        archive.document.to_zarr(store)
    return tree
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_zarr_output_archive.py::ZarrOpaqueMetadataWriteTestCase -v`
Expected: PASS — opaque metadata is staged at the spec path with the correct format flag.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/zarr/_layout.py python/lsst/images/zarr/_output_archive.py tests/test_zarr_output_archive.py
git commit -m "feat: persist FitsOpaqueMetadata at /lsst/opaque_metadata/fits/primary on zarr write"
```

### Task 5.2: Restore `FitsOpaqueMetadata` on read from zarr

**Files:**
- Modify: `python/lsst/images/zarr/_layout.py` (add `deserialize_fits_opaque_metadata`)
- Modify: `python/lsst/images/zarr/_input_archive.py` (read it in `__init__`; expose via `get_opaque_metadata`; attach in `read`)
- Modify: `tests/test_zarr_input_archive.py`

`get_opaque_metadata()` returns a `FitsOpaqueMetadata` reconstructed from `/lsst/opaque_metadata/fits/primary`. The `read()` helper attaches it to the deserialized object as `obj._opaque_metadata` (matching FITS / NDF read patterns).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_zarr_input_archive.py`:

```python
@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrOpaqueMetadataReadTestCase(unittest.TestCase):
    def test_fits_opaque_metadata_round_trips(self) -> None:
        import astropy.io.fits

        from lsst.images import Box, Image
        from lsst.images.fits._common import ExtensionKey, FitsOpaqueMetadata
        from lsst.images.zarr import read, write

        image = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )
        header = astropy.io.fits.Header()
        header["ORIGIN"] = "RUBIN"
        header["EXPTIME"] = 30.0
        opaque = FitsOpaqueMetadata()
        opaque.headers[ExtensionKey()] = header
        image._opaque_metadata = opaque

        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "out.zarr")
            write(image, target)
            recovered = read(Image, target).deserialized
            recovered_opaque = recovered._opaque_metadata
            self.assertIsInstance(recovered_opaque, FitsOpaqueMetadata)
            recovered_header = recovered_opaque.headers[ExtensionKey()]
            self.assertEqual(recovered_header["ORIGIN"], "RUBIN")
            self.assertEqual(recovered_header["EXPTIME"], 30.0)
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/test_zarr_input_archive.py::ZarrOpaqueMetadataReadTestCase -v`
Expected: FAIL — `recovered._opaque_metadata` is `None` or unset.

- [ ] **Step 3: Implement deserialization**

Append to `python/lsst/images/zarr/_layout.py`:

```python
def deserialize_fits_opaque_metadata(document: "ZarrDocument") -> Any | None:
    """Reconstruct a `FitsOpaqueMetadata` from the IR, or return None.

    Returns ``None`` when the archive does not have a FITS opaque
    metadata block (the common case for archives that originated as
    native zarr).
    """
    import json as _json

    from ..fits._common import ExtensionKey, FitsOpaqueMetadata
    from ._model import ZarrArray

    if document.root.attributes.lsst.get("opaque_metadata_format") != "fits":
        return None
    try:
        node = document.root.get("/lsst/opaque_metadata/fits/primary")
    except KeyError:
        return None
    if not isinstance(node, ZarrArray):
        return None
    json_bytes = bytes(node.read()).decode("utf-8")
    cards = _json.loads(json_bytes)
    import astropy.io.fits

    header = astropy.io.fits.Header()
    for key, value in cards.items():
        header[key] = value
    opaque = FitsOpaqueMetadata()
    opaque.headers[ExtensionKey()] = header
    return opaque
```

In `python/lsst/images/zarr/_input_archive.py`, store opaque metadata at construction time, expose it, and attach it in `read`:

```python
    def __init__(self, document: ZarrDocument) -> None:
        self._document = document
        self._validate_root_attributes()
        self._deserialized_pointer_cache = {}
        self._frame_set_cache = {}
        from ._layout import deserialize_fits_opaque_metadata

        self._opaque_metadata = deserialize_fits_opaque_metadata(document)

    def get_opaque_metadata(self) -> Any | None:
        return self._opaque_metadata
```

…and in `read`:

```python
def read[T: Any](cls, path, **kwargs):
    with ZarrInputArchive.open(path) as archive:
        tree_type = cls._get_archive_tree_type(ZarrPointerModel)
        tree = archive.get_tree(tree_type)
        obj = tree.deserialize(archive, **kwargs)
        if (opaque := archive.get_opaque_metadata()) is not None:
            obj._opaque_metadata = opaque
        return ReadResult(obj, tree.metadata, tree.butler_info)
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_zarr_input_archive.py::ZarrOpaqueMetadataReadTestCase -v`
Expected: PASS — recovered header has both cards.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/zarr/_input_archive.py python/lsst/images/zarr/_layout.py tests/test_zarr_input_archive.py
git commit -m "feat: restore FitsOpaqueMetadata on zarr read"
```

### Task 5.3: FITS → Zarr → FITS round-trip

**Files:**
- Create: `tests/test_zarr_cross_format.py`

End-to-end: read a FITS file, write it to zarr, read the zarr back, write it to FITS. The final FITS file's primary header must match the original's card-for-card.

- [ ] **Step 1: Write the test**

Create `tests/test_zarr_cross_format.py`:

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

import numpy as np

try:
    import zarr  # noqa: F401

    from lsst.images.zarr import read as zarr_read
    from lsst.images.zarr import write as zarr_write

    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class FitsZarrCrossFormatTestCase(unittest.TestCase):
    def test_fits_to_zarr_to_fits_preserves_primary_header(self) -> None:
        import astropy.io.fits

        from lsst.images import Box, Image
        from lsst.images.fits import read as fits_read
        from lsst.images.fits import write as fits_write

        original = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )
        with tempfile.TemporaryDirectory() as tmp:
            fits_a = os.path.join(tmp, "a.fits")
            zarr_path = os.path.join(tmp, "b.zarr")
            fits_b = os.path.join(tmp, "c.fits")

            def update_header(header):  # noqa: ANN001
                header["ORIGIN"] = "RUBIN"
                header["EXPTIME"] = 30.0

            fits_write(original, fits_a, update_header=update_header)
            from_fits = fits_read(Image, fits_a).deserialized
            zarr_write(from_fits, zarr_path)
            from_zarr = zarr_read(Image, zarr_path).deserialized
            fits_write(from_zarr, fits_b)

            with astropy.io.fits.open(fits_b) as hdul:
                self.assertEqual(hdul[0].header["ORIGIN"], "RUBIN")
                self.assertEqual(hdul[0].header["EXPTIME"], 30.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_zarr_cross_format.py -v`
Expected: PASS — both cards survive the FITS→Zarr→FITS pipeline.

- [ ] **Step 3: Commit**

```bash
git add tests/test_zarr_cross_format.py
git commit -m "test: FITS↔Zarr opaque-metadata round-trip"
```

### Task 5.4: xarray interop assertion

**Files:**
- Create: `tests/test_zarr_xarray_interop.py`

The whole point of the xarray/CF root layout is that `xr.open_zarr(path)` returns a `Dataset` with the masked-image components as data variables sharing the `(y, x)` dimensions, and the CF `flag_masks` / `flag_meanings` survive on the `mask` variable. This test pins that contract.

Skipped if `xarray` is not installed; the implementer adds `xarray` to the test extras when this test is added.

- [ ] **Step 1: Write the test**

Create `tests/test_zarr_xarray_interop.py`:

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

import numpy as np

try:
    import zarr  # noqa: F401

    from lsst.images.zarr import write

    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False

try:
    import xarray as xr  # noqa: F401

    HAVE_XARRAY = True
except ImportError:
    HAVE_XARRAY = False


@unittest.skipUnless(HAVE_ZARR and HAVE_XARRAY, "xarray is not installed")
class XarrayInteropTestCase(unittest.TestCase):
    def test_open_zarr_returns_dataset_with_masked_image_components(self) -> None:
        from lsst.images import Box, Image, MaskedImage, MaskPlane, MaskSchema

        schema = MaskSchema(
            [
                MaskPlane("BAD", "Bad pixel."),
                MaskPlane("SAT", "Saturated."),
                MaskPlane("CR", "Cosmic ray."),
            ]
        )
        image = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )
        masked = MaskedImage(image, mask_schema=schema)
        masked.mask.set("BAD", image.array % 2 == 0)

        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "masked.zarr")
            write(masked, target)
            ds = xr.open_zarr(target)
            # Three data variables sharing the (y, x) dims.
            self.assertIn("image", ds.data_vars)
            self.assertIn("variance", ds.data_vars)
            self.assertIn("mask", ds.data_vars)
            self.assertEqual(ds["image"].dims, ("y", "x"))
            self.assertEqual(ds["mask"].dims, ("y", "x"))
            self.assertEqual(ds["image"].shape, (4, 5))
            # CF flag attrs survive on the mask variable.
            self.assertEqual(ds["mask"].attrs["flag_meanings"], "BAD SAT CR")
            self.assertEqual(list(ds["mask"].attrs["flag_masks"]), [1, 2, 4])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_zarr_xarray_interop.py -v`
Expected: PASS if `xarray` is installed; SKIP otherwise. If it fails when xarray is present, inspect what xarray sees: most often it's a `_ARRAY_DIMENSIONS` typo, or `tree` / `wcs_ast` arrays leaking into the Dataset (xarray treats every zarr array in the group as a data variable — those are 1-D `uint8` arrays so they should appear as 1-D variables, harmless, but they shouldn't shadow `image` etc.).

- [ ] **Step 3: Commit**

```bash
git add tests/test_zarr_xarray_interop.py
git commit -m "test: xarray.open_zarr returns Dataset with image/variance/mask data variables"
```

### Task 5.5: Optional `ome-zarr-py` external-reader sanity test

**Files:**
- Create: `tests/test_zarr_external_reader.py`

This test confirms the bytes we emit are readable by `ome-zarr-py` (the upstream OME-Zarr toolkit). It checks only the science array — `ome-zarr-py` doesn't know about `lsst:` extensions. Skipped when the package isn't installed.

- [ ] **Step 1: Write the test**

Create `tests/test_zarr_external_reader.py`:

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

import numpy as np

try:
    import zarr  # noqa: F401

    from lsst.images.zarr import write

    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False

try:
    import ome_zarr  # noqa: F401
    import ome_zarr.io  # noqa: F401
    import ome_zarr.reader  # noqa: F401

    HAVE_OME_ZARR = True
except ImportError:
    HAVE_OME_ZARR = False


@unittest.skipUnless(HAVE_ZARR and HAVE_OME_ZARR, "ome-zarr is not installed")
class OmeZarrReaderTestCase(unittest.TestCase):
    def test_ome_zarr_can_open_image(self) -> None:
        from lsst.images import Box, Image

        original = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "out.zarr")
            write(original, target)
            from ome_zarr.io import parse_url
            from ome_zarr.reader import Reader

            location = parse_url(target)
            self.assertIsNotNone(location)
            reader = Reader(location)
            nodes = list(reader())
            self.assertGreaterEqual(len(nodes), 1)
            data = nodes[0].data[0]  # level 0
            self.assertEqual(tuple(data.shape), (4, 5))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_zarr_external_reader.py -v`
Expected: PASS if `ome-zarr` is installed; SKIP otherwise.

- [ ] **Step 3: Commit**

```bash
git add tests/test_zarr_external_reader.py
git commit -m "test: ome-zarr-py can open archives written by lsst.images.zarr"
```

### Task 5.6: Optional `ngff-validator` compliance test

**Files:**
- Create: `tests/test_zarr_ome_compliance.py`

`ngff-validator` checks an archive against the OME-NGFF schema. Invoked via subprocess if available; skipped otherwise. Validates representative outputs of every supported archive class.

- [ ] **Step 1: Write the test**

Create `tests/test_zarr_ome_compliance.py`:

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
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

try:
    import zarr  # noqa: F401

    from lsst.images.zarr import write

    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False

NGFF_VALIDATOR = shutil.which("ngff-validator")


@unittest.skipUnless(HAVE_ZARR and NGFF_VALIDATOR, "ngff-validator is not on PATH")
class NgffComplianceTestCase(unittest.TestCase):
    def _validate(self, target: str) -> None:
        result = subprocess.run(
            [NGFF_VALIDATOR, target],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(
            result.returncode,
            0,
            f"ngff-validator failed for {target}:\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
        )

    def test_image_validates(self) -> None:
        from lsst.images import Box, Image

        image = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "out.zarr")
            write(image, target)
            self._validate(target)

    def test_masked_image_validates(self) -> None:
        from lsst.images import Box, Image, MaskedImage, MaskPlane, MaskSchema

        schema = MaskSchema([MaskPlane("BAD", "Bad pixel.")])
        image = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )
        masked = MaskedImage(image, mask_schema=schema)

        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "masked.zarr")
            write(masked, target)
            self._validate(target)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_zarr_ome_compliance.py -v`
Expected: PASS if `ngff-validator` is on PATH; SKIP otherwise. If a real install is available and validation fails, fix the layout (most likely an axis-type misclassification or a `coordinateTransformations` shape error) before merging.

- [ ] **Step 3: Commit**

```bash
git add tests/test_zarr_ome_compliance.py
git commit -m "test: ngff-validator compliance check (skipped when validator absent)"
```

---

**End of Phase 5.** Six tasks. FITS↔Zarr round-trips preserve primary-HDU cards through Zarr; xarray interop is pinned by an `xr.open_zarr` test that asserts `Dataset` shape and CF flag attrs on the mask; optional external-reader checks confirm OME-NGFF compliance and `ome-zarr-py` interop when their dependencies are installed. Phase 6 wraps up with module documentation and a changelog entry.

## Phase 6 — Documentation, changelog, and final integration

Phase 6 wraps up the backend with the documentation that makes it discoverable. The reference docs live under `doc/lsst.images/` and follow the same `automodapi`-driven pattern as the other backends; the changelog uses `towncrier` fragments under `doc/changes/`.

### Task 6.1: Expand the module docstring

**Files:**
- Modify: `python/lsst/images/zarr/__init__.py`

The Phase 1 `__init__.py` carries a short docstring. Replace it with a full-fat version covering layout, lazy reads, FITS round-trip, and the v1 follow-ups.

- [ ] **Step 1: Replace the docstring**

Edit `python/lsst/images/zarr/__init__.py`. Replace the existing docstring (everything between the first triple-quote and the matching one) with:

```python
"""Zarr v3 archive backend for `lsst.images`.

This module reads and writes Zarr v3 archives whose root layout is
xarray/CF-shaped (``image``, ``variance``, ``mask`` as siblings sharing
``(y, x)`` dimensions, CF ``flag_masks`` / ``flag_meanings`` /
``flag_descriptions`` on the mask) with OME-NGFF v0.5 multiscales
metadata as a discoverability layer pointing at the same ``image``
array. The same bytes are visible to ``xarray``, GDAL's Zarr driver,
and OME-Zarr tooling like ``napari`` and ``ome-zarr-py``.

Supported types
---------------

Every image type that already serializes to FITS / JSON / NDF:
`~lsst.images.Image`, `~lsst.images.Mask`, `~lsst.images.MaskedImage`,
`~lsst.images.VisitImage`, `~lsst.images.ColorImage`, and
`lsst.images.cells.CellCoadd`, plus any object reachable through the
`~lsst.images.serialization.OutputArchive` interface.

On-disk layout
--------------

A `MaskedImage` archive contains:

- ``image``, ``variance``, ``mask`` arrays at the root, shaped
  ``(Y, X)`` with shared chunk sizes.
- ``tree`` — 1-D ``uint8`` zarr array containing UTF-8 JSON of the
  Pydantic archive tree (the round-trip authority).
- ``wcs_ast`` — 1-D ``uint8`` zarr array containing the AST FrameSet
  text (the WCS round-trip authority).

The mask is a 2-D unsigned integer (``uint8`` for ≤8 planes, up to
``uint64`` for 64 planes; >64 raises). Each pixel's bits encode the
applicable mask planes — the same logical representation the FITS
backend uses, so FITS↔Zarr mask round-trips need no bit-repacking.

For `ColorImage`, the three channels are written as recursive
sub-archives at ``red/``, ``green/``, ``blue/``. Each sub-archive is
itself a valid Image-shaped OME-NGFF group with its own ``image``
array, OME multiscales metadata, and ``lsst.archive_class = "Image"``.

For `CellCoadd`, ``image`` / ``variance`` / ``mask`` are siblings
(cell-aligned chunks driven by ``cell_shape``), and ``psf`` is a 4-D
``(Cy, Cx, Py, Px)`` array with single-cell chunks
``(1, 1, Py, Px)``. ``lsst.cell_grid = {bbox, cell_shape}`` lives on
the root attrs.

The OME multiscales ``dataset.path`` always points at a sibling array
(``"image"`` for the standard case). No bytes are duplicated for the
OME view — the science array is the same array xarray sees.

WCS handling
------------

The AST ``FrameSet`` text at ``wcs_ast`` is the round-trip authority.
For external tools (napari, neuroglancer), the layout layer also
emits an OME-NGFF v0.5 affine ``coordinateTransformations`` block
that approximates the linear part of the pixel-to-sky map. Before
emitting, residuals are sampled on an 11×11 grid; if the worst
pixel-equivalent error exceeds 1.0 pixel, the affine block is dropped
and ``lsst.wcs_simplified_dropped: true`` is recorded with the
observed maximum. Readers always reconstruct the projection from
``wcs_ast``.

Full RFC-5 nonlinear coordinate transformations as authoritative
output is a follow-up; it is blocked on writing an AST JSON channel
that serializes a ``FrameSet`` to / from RFC-5 transformation JSON.

Cloud-friendly defaults
-----------------------

- Default chunk geometry is tile-aligned: ``min(1024, dim)`` per
  axis for plain images, ``cell_shape`` for `CellCoadd`, single-cell
  for `CellCoadd`'s 4-D PSF.
- Sharding (zarr v3 native) is enabled by default with a tunable
  shard size (4×4 chunks by default) so object counts on S3 / GCS
  stay manageable for multi-gigabyte images.
- Subset reads via the ``slices=`` argument to
  `~lsst.images.serialization.InputArchive.get_array` exploit zarr's
  chunk index: only chunks intersecting the slice are fetched, even
  from remote stores.
- Both ``DirectoryStore`` and ``ZipStore`` are supported. The store
  is selected from the URI shape: ``*.zarr.zip`` → ZipStore,
  otherwise directory. Remote URIs (``s3://``, ``gs://``,
  ``http(s)://``) go through `lsst.resources.ResourcePath` and
  `fsspec`.

Round-trip with FITS
--------------------

When an object that originated from a FITS read carries a
`~lsst.images.fits.FitsOpaqueMetadata`, the primary-HDU header is
preserved at ``/lsst/opaque_metadata/fits/primary``. Reading the
zarr back attaches an equivalent ``FitsOpaqueMetadata`` to the
deserialized object so a subsequent FITS write reproduces the
original cards.

Optional install
----------------

This backend requires `zarr >= 3.0`. Install via the ``[zarr]``
extra::

    pip install lsst-images[zarr]

The top-level ``import lsst.images.zarr`` raises a clear
`ImportError` with this guidance if `zarr` is not installed.

Follow-ups
----------

These items are tracked separately from the initial backend release:

- Lazy / dask-friendly read API (``read_lazy()``).
- Multiscale pyramid generation (level 1, 2, …) for visualization
  tools.
- NGFF RFC-5 nonlinear coordinate transformations as authoritative
  output (blocked on AST JSON channel work).
- 3-D mask fallback for `>64` planes.
- ``zarr.consolidated_metadata`` extension to reduce object-list
  calls on cloud stores.
- NCZarr / NetCDF interop (``_NCZARR_*`` markers + optional 1-D
  coordinate variables; purely additive when adopted).
- Stacked OME view for `ColorImage` (single ``(3, Y, X)`` array
  alongside the per-channel sub-archives, gated by an explicit
  flag because of the byte-duplication cost).
"""
```

- [ ] **Step 2: Verify the docstring is well-formed**

Run: `python -c "import lsst.images.zarr; help(lsst.images.zarr)" | head -60`
Expected: docstring renders cleanly with no `:role:` typos or unclosed code blocks. A deeper Sphinx build runs in Task 6.2.

- [ ] **Step 3: Commit**

```bash
git add python/lsst/images/zarr/__init__.py
git commit -m "docs: expand lsst.images.zarr module docstring"
```

### Task 6.2: Add the reference docs page

**Files:**
- Create: `doc/lsst.images/zarr.rst`
- Modify: `doc/lsst.images/index.rst` (add `zarr.rst` to the toctree)

Mirrors `doc/lsst.images/ndf.rst` exactly so Sphinx renders the API in the same shape as the other backends.

- [ ] **Step 1: Create the reference page**

Create `doc/lsst.images/zarr.rst`:

```rst
Zarr I/O
========

A Zarr v3 serialization backend whose on-disk layout is xarray/CF-shaped
at the root (``image`` / ``variance`` / ``mask`` as siblings sharing
``(y, x)`` dimensions, CF ``flag_masks`` / ``flag_meanings`` on the
mask) with OME-NGFF v0.5 multiscales metadata as a discoverability
layer pointing at the same ``image`` array. The same bytes are visible
to ``xarray``, GDAL's Zarr driver, and OME-Zarr tooling like
``napari`` and ``ome-zarr-py``.

Default chunking is tile-aligned (~1024×1024 for plain images,
``cell_shape`` for ``CellCoadd``); sharding is enabled by default; and
subset reads via ``slices=`` only fetch the chunks they need — including
on remote stores accessed through ``lsst.resources.ResourcePath`` and
``fsspec``.

.. automodapi:: lsst.images.zarr
   :no-inheritance-diagram:
   :include-all-objects:
   :inherited-members:
```

- [ ] **Step 2: Add the page to the toctree**

In `doc/lsst.images/index.rst`, find the line containing `ndf.rst` and add `zarr.rst` after it (preserving alphabetical order):

```rst
   fits.rst
   json.rst
   ndf.rst
   zarr.rst
```

- [ ] **Step 3: Verify the docs build**

Run: `cd doc && sphinx-build -W -b html . _build/html` (only if a Sphinx environment is set up locally; otherwise skip and rely on CI).
Expected: clean build with no warnings about undefined references.

- [ ] **Step 4: Commit**

```bash
git add doc/lsst.images/zarr.rst doc/lsst.images/index.rst
git commit -m "docs: add Zarr backend reference page"
```

### Task 6.3: Add the towncrier changelog fragment

**Files:**
- Create: `doc/changes/DM-XXXXX.feature.md` (replace `XXXXX` with the assigned Jira ticket number)

Each user-visible change lands as a single Markdown fragment under `doc/changes/`. For this work it's a **feature**.

- [ ] **Step 1: Create the fragment**

Create `doc/changes/DM-XXXXX.feature.md` (replace `XXXXX` with the actual Jira ticket number):

```markdown
Added a new `lsst.images.zarr` archive backend that reads and writes Zarr v3 archives. The on-disk layout is xarray/CF-shaped at the root (`image`, `variance`, `mask` as siblings sharing `(y, x)` dimensions, CF `flag_masks`/`flag_meanings` on the mask) with OME-NGFF v0.5 multiscales metadata layered on top — the same bytes are visible to xarray, GDAL, and OME-Zarr tooling like `napari` and `ome-zarr-py`. Supports every image type the FITS / JSON / NDF backends support (`Image`, `Mask`, `MaskedImage`, `VisitImage`, `ColorImage`, `CellCoadd`). Cloud-friendly defaults (tile-aligned chunks, zarr v3 sharding, fsspec-backed remote stores) and subset reads that only fetch the chunks they need. Install via the new `[zarr]` extra (`pip install lsst-images[zarr]`).
```

- [ ] **Step 2: Commit**

```bash
git add doc/changes/DM-XXXXX.feature.md
git commit -m "docs: changelog entry for lsst.images.zarr backend"
```

### Task 6.4: Run the full test suite and finalize

**Files:** none (verification step).

- [ ] **Step 1: Run the full zarr test set**

Run: `pytest tests/test_zarr_*.py -v`
Expected: all tests pass; external-reader and validator tests pass or skip cleanly depending on what's installed; CellCoadd tests skip cleanly until the implementer-supplied factories are filled in.

- [ ] **Step 2: Run the full package test suite to catch regressions**

Run: `pytest tests/ -v`
Expected: all existing tests still pass; the new `RoundtripZarr` helper does not break unrelated test files.

- [ ] **Step 3: Type-check the new module**

Run: `mypy python/lsst/images/zarr`
Expected: no errors. Address any warnings before merging.

- [ ] **Step 4: Lint and format**

Run: `ruff check python/lsst/images/zarr tests/test_zarr_*.py && ruff format --check python/lsst/images/zarr tests/test_zarr_*.py`
Expected: no findings.

- [ ] **Step 5: Final commit (if any cleanups were needed)**

```bash
git status  # should be clean
```

If lint / mypy required fixes, commit them with a focused message such as `chore: type-check and lint cleanup for lsst.images.zarr`.

---

**End of Phase 6.** Documentation and final verification complete. The backend is ready for review and merge.

---

## Self-Review Notes

**Spec coverage** — every section of `docs/superpowers/specs/2026-05-22-zarr-io-design.md` maps to at least one task:

| Spec section | Task(s) |
|---|---|
| §1 Goals / scope / standards alignment | All phases collectively |
| §2 Module layout | 1.1 (skeleton), 1.2 (`_common`), 1.3-1.5 (`_model`), 2.1 (`_store`), 2.2-2.3 (`_layout`), 2.4-2.7 / 3.1-3.5 (archives) |
| §3 On-disk layout (root, siblings, attrs) | 2.5 (`add_array` for image/variance/mask), 2.7 (`add_tree` for root attrs and OME multiscales), 4.1 (recursive sub-archive decoration) |
| §3 Axis choice per archive class | 2.2 (`axes_for_archive_class`), 4.1 (sub-archive `("y", "x")`), 4.4 (CellCoadd) |
| §3 Mask 2-D packed integer with CF flag attrs | 1.2 (`mask_dtype_for_plane_count`), 1.5 (`CfFlagAttributes`), 2.5 (mask packing in `add_array`), 3.0 (native-mask flag), 3.2 (mask unpack on read), 3.6 (3-plane and 40-plane round-trips) |
| §3 JSON tree at `/tree` | 2.7 (`add_tree` stages JSON bytes); 3.1 (`get_tree` reads them) |
| §3 AST WCS at `/wcs_ast` | 2.7 (`_stage_wcs_ast`), 3.3 (Projection deserializer reads it) |
| §3 Tables under `/lsst/tables/<name>/<column>` | 2.6 (output), 3.4 (input) |
| §3 Recursive composition | 4.1 (`decorate_sub_archives`) |
| §3 Chunking / sharding defaults / aligned siblings | 1.3-1.4 (defaults in IR), 2.2 (`chunks_for`, `chunks_aligned_to`), 4.3 (PSF single-cell chunks) |
| §4 FITS opaque-metadata round-trip | 5.1 (write), 5.2 (read), 5.3 (full FITS↔Zarr) |
| §4 WCS validation: 11×11 grid, 1-pixel threshold | 2.3 (`affine_check`), 2.7 (integration in `add_tree`) |
| §4 Error taxonomy | 1.2 (`>64`-plane refusal), 3.1 (missing `archive_class`, `>LSST_VERSION`), 3.2 (bad source string) |
| §4 Mode and atomicity | 2.1 (create-only enforcement) |
| §4 Chunk-aligned subset reads (lazy invariant) | 1.3 (`_CountingStore` test on the IR), 3.2 (regression test on the input archive) |
| §4 Mask schema mismatches | inherited from existing `Mask.deserialize`; v1 surfaces it through the standard error path; explicit dedicated test deferred to a follow-up |
| §4 Empty / minimal cases | 2.7 (no `wcs_ast` when no projection; unit-scale `coordinateTransformations` default), 2.5 (image without variance / mask) |
| §4 Forward compatibility | 1.3 (unknown-key preservation in `ZarrAttributes`), 3.1 (version refusal) |
| §5 Test layout | One test file per module, plus `test_zarr_round_trip.py`, `test_zarr_cross_format.py`, `test_zarr_xarray_interop.py`, `test_zarr_ome_compliance.py`, `test_zarr_external_reader.py` |
| §5 Rollout plan (6 numbered steps) | Phases 1–6 directly mirror the spec's rollout |
| §6 Follow-ups | Documented in 6.1's docstring (RFC-5, 3-D mask fallback, dask read, multiscale pyramid, consolidated metadata, NCZarr, stacked OME ColorImage view) |

**Implementer-judgement handoffs** — places where the plan asks the engineer to consult local code rather than follow a literal recipe:

- Tasks 4.4 / 4.5: minimal `CellCoadd` constructor — `_make_minimal_cell_coadd` and `_make_minimal_cell_coadd_with_psf` are `SkipTest` placeholders to be replaced by reading `python/lsst/images/cells/_coadd.py`.
- Task 6.3: the towncrier fragment filename uses `DM-XXXXX` — pick the real ticket number when committing.
- Task 3.6: the `RoundtripBase` helper may need a small directory-vs-file fix to accept `.zarr` directories.

These are intentional handoffs, not placeholder content in the production code.

**Type / name consistency** — IR types and key methods stay consistent across phases:

- `ZarrDocument`, `ZarrGroup`, `ZarrArray`, `ZarrAttributes` introduced in 1.3-1.4, used everywhere after.
- `ZarrAttributes` has three namespaces (`lsst`, `ome`, `extra`); `extra` is read by xarray / CF tooling and tested in 1.3, 1.4, 5.4.
- `ZarrCompressionOptions.default_for_dtype` from 1.2 is consumed by the `to_zarr` codec builder in 1.4.
- `_layout.chunks_for` / `chunks_aligned_to` defined in 2.2 are used by the output archive in 2.5; `_layout.affine_check` defined in 2.3 is used in 2.7.
- `lsst.archive_class`, `lsst.tree`, `lsst.wcs_ast`, `lsst.cell_grid`, `lsst.opaque_metadata_format`, `lsst.wcs_simplified_dropped`, `lsst.wcs_simplified_max_residual_pixels` are spelled the same in every task that reads or writes them.
- The sliced-source convention (`?c=N`, `?cell=Cy,Cx`) from the v1 plan is **deliberately absent** — the no-stacking rule means every `ArrayReferenceModel.source` is plain `zarr:/<path>`.

**Critical invariants pinned by tests** — the four invariants stated in the plan header each have at least one failing test:

1. Lazy reads — `_CountingStore` test in 1.3 (IR level) and 3.2 (input archive level).
2. Aligned chunks — Phase 2.5 test asserting `variance` follows `image_chunks` after the override; CellCoadd test in 4.4 asserting all three siblings have `cell_shape` chunks.
3. Affine residual validator — Phase 2.3 tests with a synthetic linear FrameSet (passes) and a synthetic high-distortion FrameSet (drops).
4. No byte duplication — implicit in the "no fixup pass" architecture; explicit assertions in 4.1 (root has no OME multiscales for ColorImage) and 4.4 (CellCoadd PSF is a single 4-D array, not per-cell groups + a stacked array).
