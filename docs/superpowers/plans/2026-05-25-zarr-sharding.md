# Zarr v3 Default Sharding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable automatic zarr v3 sharding for bulk pixel arrays and drop the per-axis chunk default from 1024 to 256, with no public API changes.

**Architecture:** A pure `default_shards(chunks, shape, dtype, *, target_bytes)` helper lives next to `chunks_for` in `_layout.py`. Two module-level constants (chunk-axis limit and target shard bytes) live in `_common.py`; the shard target reads `LSST_IMAGES_ZARR_TARGET_SHARD_BYTES` once at import. `ZarrOutputArchive.add_array` calls the helper at the same point chunks are decided, so the IR's `ZarrArray.shards` is populated whenever the caller didn't supply an override. The model writer (`_group_to_zarr`) is unchanged.

**Tech Stack:** Python 3.12, zarr v3 (`zarr-python` 3.x), numpy, unittest. Project uses `.pyenv/bin/python` to run; system Python lacks zarr.

**Design spec:** `docs/superpowers/specs/2026-05-25-zarr-sharding-design.md`

---

## File Structure

| Path                                              | Change   | Responsibility                              |
|---------------------------------------------------|----------|---------------------------------------------|
| `python/lsst/images/zarr/_common.py`              | modify   | add `DEFAULT_CHUNK_AXIS_LIMIT`, `DEFAULT_TARGET_SHARD_BYTES` |
| `python/lsst/images/zarr/_layout.py`              | modify   | drop hardcoded 1024, read from `_common`; add `default_shards` |
| `python/lsst/images/zarr/_output_archive.py`      | modify   | call `default_shards` alongside `chunks_for` at the two existing sites |
| `tests/test_zarr_layout.py`                       | modify   | update existing chunk-default test; add `default_shards` unit tests |
| `tests/test_zarr_common.py`                       | modify   | add subprocess-based env-var tests                                  |
| `tests/test_zarr_round_trip.py`                   | modify   | add a 300×300 round-trip that asserts on-disk `shards` is set       |
| `tests/test_zarr_output_archive.py`               | modify   | add a CellCoadd PSF shard-defaulting test                           |
| `tests/test_zarr_store.py`                        | modify   | add a sharded write/read through `ZipStore`                         |

No new files.

---

## Task 1: Lower the chunk-axis default to 256 (test-first)

**Files:**
- Modify: `tests/test_zarr_layout.py:56-60` (`test_chunks_for_default`)
- Modify: `python/lsst/images/zarr/_common.py` (add constant + export)
- Modify: `python/lsst/images/zarr/_layout.py:50` (replace hardcoded 1024)

- [ ] **Step 1: Update the existing chunk-default test to expect 256**

In `tests/test_zarr_layout.py`, replace `test_chunks_for_default` (currently lines 56–60) with:

```python
    def test_chunks_for_default(self) -> None:
        # Plain images clamp to the per-axis chunk limit (256 by default).
        self.assertEqual(chunks_for("Image", (4096, 4096), None), (256, 256))
        # Smaller than the limit -> use full dim.
        self.assertEqual(chunks_for("Image", (200, 100), None), (200, 100))
```

Also update `test_chunks_for_cell_coadd_without_metadata_falls_back` (currently lines 73–74):

```python
    def test_chunks_for_cell_coadd_without_metadata_falls_back(self) -> None:
        self.assertEqual(chunks_for("CellCoadd", (4096, 4096), None), (256, 256))
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.pyenv/bin/python -m pytest tests/test_zarr_layout.py::LayoutTestCase::test_chunks_for_default -v`

Expected: FAIL — actual is `(1024, 1024)`, expected is `(256, 256)`.

- [ ] **Step 3: Add `DEFAULT_CHUNK_AXIS_LIMIT` to `_common.py`**

Edit `python/lsst/images/zarr/_common.py`:

Add to `__all__` (currently lines 14–23):

```python
__all__ = (
    "DEFAULT_CHUNK_AXIS_LIMIT",
    "LSST_NS",
    "LSST_VERSION",
    "OME_NS",
    "OME_VERSION",
    "ZarrCompressionOptions",
    "ZarrPointerModel",
    "archive_path_to_zarr_path",
    "mask_dtype_for_plane_count",
)
```

After `LSST_VERSION = 1` (line 40) and its docstring, add:

```python
DEFAULT_CHUNK_AXIS_LIMIT = 256
"""Per-axis cap on the auto-derived chunk shape for plain image arrays.

Used by `lsst.images.zarr._layout.chunks_for` when the caller does not
supply an explicit override and the archive class does not have a
class-specific chunk rule. Chunks of ~256 elements per spatial axis
trade some compression ratio for cutout-friendly partial reads.
"""
```

- [ ] **Step 4: Read the constant from `_layout.py`**

Edit `python/lsst/images/zarr/_layout.py`.

Add a new import. The existing import section already pulls from `..fits._common` and from `._model`; add a sibling line:

```python
from ._common import DEFAULT_CHUNK_AXIS_LIMIT
```

Delete the line `_DEFAULT_AXIS_LIMIT = 1024` (currently the only module-level numeric constant in this file; sits just before `axes_for_archive_class`).

In `chunks_for`, replace the `_DEFAULT_AXIS_LIMIT` reference at the very end of the function:

```python
    return tuple(min(_DEFAULT_AXIS_LIMIT, dim) for dim in shape)
```

with

```python
    return tuple(min(DEFAULT_CHUNK_AXIS_LIMIT, dim) for dim in shape)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `.pyenv/bin/python -m pytest tests/test_zarr_layout.py -v`

Expected: PASS for `test_chunks_for_default`, `test_chunks_for_cell_coadd_without_metadata_falls_back`, and all other tests in the file.

- [ ] **Step 6: Commit**

```bash
git add python/lsst/images/zarr/_common.py python/lsst/images/zarr/_layout.py tests/test_zarr_layout.py
git commit -m "feat(zarr): drop chunk-axis default 1024 -> 256, centralize constant"
```

---

## Task 2: Add `DEFAULT_TARGET_SHARD_BYTES` constant with env-var override

**Files:**
- Modify: `python/lsst/images/zarr/_common.py` (add constant + env-var read)
- Modify: `tests/test_zarr_common.py` (add subprocess tests)

- [ ] **Step 1: Inspect `tests/test_zarr_common.py` to see existing test conventions**

Run: `head -30 tests/test_zarr_common.py`

Review what's there. The new tests will follow the same `unittest.TestCase` style.

- [ ] **Step 2: Write the env-var subprocess tests**

Append to `tests/test_zarr_common.py` (before the `if __name__ == "__main__":` block):

```python
import subprocess
import sys


class TargetShardBytesEnvVarTestCase(unittest.TestCase):
    """`DEFAULT_TARGET_SHARD_BYTES` reads from env var at import time."""

    def _import_in_subprocess(self, env_value: str | None) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env.pop("LSST_IMAGES_ZARR_TARGET_SHARD_BYTES", None)
        if env_value is not None:
            env["LSST_IMAGES_ZARR_TARGET_SHARD_BYTES"] = env_value
        code = (
            "from lsst.images.zarr._common import DEFAULT_TARGET_SHARD_BYTES;"
            "print(DEFAULT_TARGET_SHARD_BYTES)"
        )
        return subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_unset_uses_default(self) -> None:
        result = self._import_in_subprocess(None)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(16 * 1024 * 1024))

    def test_set_value_overrides(self) -> None:
        result = self._import_in_subprocess("1234567")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "1234567")

    def test_garbage_value_fails_at_import(self) -> None:
        result = self._import_in_subprocess("not-a-number")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("LSST_IMAGES_ZARR_TARGET_SHARD_BYTES", result.stderr)
```

If the file does not already import `os`, add `import os` to the imports at the top.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `.pyenv/bin/python -m pytest tests/test_zarr_common.py::TargetShardBytesEnvVarTestCase -v`

Expected: FAIL — `DEFAULT_TARGET_SHARD_BYTES` does not exist yet (`ImportError`).

- [ ] **Step 4: Add the constant to `_common.py`**

Edit `python/lsst/images/zarr/_common.py`.

Add `DEFAULT_TARGET_SHARD_BYTES` to `__all__` so it becomes:

```python
__all__ = (
    "DEFAULT_CHUNK_AXIS_LIMIT",
    "DEFAULT_TARGET_SHARD_BYTES",
    "LSST_NS",
    "LSST_VERSION",
    "OME_NS",
    "OME_VERSION",
    "ZarrCompressionOptions",
    "ZarrPointerModel",
    "archive_path_to_zarr_path",
    "mask_dtype_for_plane_count",
)
```

Add `import os` near the other stdlib imports.

After the `DEFAULT_CHUNK_AXIS_LIMIT` block added in Task 1, append:

```python
def _read_target_shard_bytes() -> int:
    """Read `LSST_IMAGES_ZARR_TARGET_SHARD_BYTES` or return the default.

    Parsed as a base-10 integer. A non-integer value raises ``ValueError``
    at import time — silent typos are worse than loud failure.
    """
    raw = os.environ.get("LSST_IMAGES_ZARR_TARGET_SHARD_BYTES")
    if raw is None:
        return 16 * 1024 * 1024
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"LSST_IMAGES_ZARR_TARGET_SHARD_BYTES={raw!r} is not a base-10 integer."
        ) from exc


DEFAULT_TARGET_SHARD_BYTES: int = _read_target_shard_bytes()
"""Target uncompressed byte size for an auto-derived shard.

Read from ``LSST_IMAGES_ZARR_TARGET_SHARD_BYTES`` once at import time;
defaults to 16 MiB. Used by `lsst.images.zarr._layout.default_shards` to
decide how many chunks to combine into a shard.
"""
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.pyenv/bin/python -m pytest tests/test_zarr_common.py::TargetShardBytesEnvVarTestCase -v`

Expected: PASS for all three subtests.

- [ ] **Step 6: Run the full common test file as a regression check**

Run: `.pyenv/bin/python -m pytest tests/test_zarr_common.py -v`

Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add python/lsst/images/zarr/_common.py tests/test_zarr_common.py
git commit -m "feat(zarr): add DEFAULT_TARGET_SHARD_BYTES with env-var override"
```

---

## Task 3: Add `default_shards` helper (test-first)

**Files:**
- Modify: `python/lsst/images/zarr/_layout.py` (add helper + export)
- Modify: `tests/test_zarr_layout.py` (add new test case)

- [ ] **Step 1: Write the unit tests**

Append a new test class to `tests/test_zarr_layout.py` (before the `if __name__ == "__main__":` block, after the existing `LayoutTestCase`):

```python
@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class DefaultShardsTestCase(unittest.TestCase):
    """The `default_shards` byte-budget rule."""

    TARGET = 16 * 1024 * 1024  # 16 MiB

    def test_4k_float32_image_uses_byte_budget(self) -> None:
        result = default_shards(
            chunks=(256, 256),
            shape=(4096, 4096),
            dtype=np.dtype("float32"),
            target_bytes=self.TARGET,
        )
        self.assertEqual(result, (2048, 2048))

    def test_3d_mask_plane_axis_untouched(self) -> None:
        # chunks already cover the plane axis; growable axes are y, x only.
        result = default_shards(
            chunks=(8, 256, 256),
            shape=(8, 4096, 4096),
            dtype=np.dtype("uint8"),
            target_bytes=self.TARGET,
        )
        self.assertEqual(result, (8, 1536, 1536))

    def test_tiny_single_chunk_returns_none(self) -> None:
        result = default_shards(
            chunks=(40,),
            shape=(40,),
            dtype=np.dtype("uint8"),
            target_bytes=self.TARGET,
        )
        self.assertIsNone(result)

    def test_chunks_equal_shape_returns_none(self) -> None:
        result = default_shards(
            chunks=(1024, 1024),
            shape=(1024, 1024),
            dtype=np.dtype("float32"),
            target_bytes=self.TARGET,
        )
        self.assertIsNone(result)

    def test_already_big_chunk_returns_none(self) -> None:
        # 4096*4096*4 = 64 MiB > 16 MiB target.
        result = default_shards(
            chunks=(4096, 4096),
            shape=(8192, 8192),
            dtype=np.dtype("float32"),
            target_bytes=self.TARGET,
        )
        self.assertIsNone(result)

    def test_k_le_one_returns_none(self) -> None:
        # chunk_bytes = 256*256*4 = 256 KiB; ratio = 4 with one growable axis;
        # k = round(4) = 4 -> not this boundary. Construct a case where
        # ratio is just above 1: 256 KiB chunk, 384 KiB target -> ratio 1.5,
        # k = round(1.5) = 2 -> sharded. Use 256 KiB chunk, 320 KiB target
        # -> ratio 1.25, k = round(1.25) = 1 -> None.
        chunk_bytes = 256 * 256 * 4
        result = default_shards(
            chunks=(256, 256),
            shape=(4096, 4096),
            dtype=np.dtype("float32"),
            target_bytes=int(chunk_bytes * 1.25),
        )
        self.assertIsNone(result)

    def test_cap_at_array_bounds(self) -> None:
        # 600x600 float32; chunk_bytes = 256 KiB; ratio = 64; k = 8.
        # Uncapped shard would be (2048, 2048) but the array is only
        # 3 chunks per axis (ceil(600/256) = 3), so the cap is (768, 768).
        result = default_shards(
            chunks=(256, 256),
            shape=(600, 600),
            dtype=np.dtype("float32"),
            target_bytes=self.TARGET,
        )
        self.assertEqual(result, (768, 768))

    def test_cell_coadd_psf(self) -> None:
        # (25, 25, 150, 150) float32 with (1, 1, 150, 150) chunks.
        # chunk_bytes = 90 KiB; ratio ~= 186; growable axes are 0 and 1
        # (cell-grid axes). k = round(sqrt(186)) = 14.
        result = default_shards(
            chunks=(1, 1, 150, 150),
            shape=(25, 25, 150, 150),
            dtype=np.dtype("float32"),
            target_bytes=self.TARGET,
        )
        self.assertEqual(result, (14, 14, 150, 150))

    def test_mismatched_ndim_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank"):
            default_shards(
                chunks=(256, 256),
                shape=(4096, 4096, 4096),
                dtype=np.dtype("float32"),
                target_bytes=self.TARGET,
            )

    def test_zero_itemsize_returns_none(self) -> None:
        # void(0) has itemsize 0; defensive guard against degenerate dtypes.
        result = default_shards(
            chunks=(256, 256),
            shape=(4096, 4096),
            dtype=np.dtype("V0"),
            target_bytes=self.TARGET,
        )
        self.assertIsNone(result)
```

Update the import block at the top of `tests/test_zarr_layout.py` (currently lines 25–32) to also import `default_shards`:

```python
try:
    from lsst.images.zarr._layout import (
        affine_check,
        axes_for_archive_class,
        chunks_aligned_to,
        chunks_for,
        decorate_sub_archives,
        default_shards,
    )
    from lsst.images.zarr._model import ZarrArray, ZarrDocument, ZarrGroup

    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `.pyenv/bin/python -m pytest tests/test_zarr_layout.py::DefaultShardsTestCase -v`

Expected: All ten subtests FAIL with `ImportError` for `default_shards`.

- [ ] **Step 3: Implement `default_shards` in `_layout.py`**

Add `"default_shards"` to the `__all__` tuple at the top of `python/lsst/images/zarr/_layout.py` (currently lines 29–38), keeping alphabetical order:

```python
__all__ = (
    "AffineCheckResult",
    "affine_check",
    "axes_for_archive_class",
    "chunks_aligned_to",
    "chunks_for",
    "decorate_sub_archives",
    "default_shards",
    "deserialize_fits_opaque_metadata",
    "serialize_fits_opaque_metadata",
)
```

Add `import math` to the imports at the top of the file (alongside `numpy`).

After `chunks_aligned_to` (currently ends at line 118), add:

```python
def default_shards(
    *,
    chunks: tuple[int, ...],
    shape: tuple[int, ...],
    dtype: np.dtype,
    target_bytes: int,
) -> tuple[int, ...] | None:
    """Derive a default shard shape from ``chunks``, ``shape``, and ``dtype``.

    Returns ``None`` when sharding would be a no-op: the array is
    already a single chunk per axis, the chunk is already at least
    ``target_bytes`` big, or the byte budget rounds to ``k == 1``
    chunks per growable axis.

    The rule grows only axes whose ``chunks[i] < shape[i]`` (the
    others already cover the full extent), uses one uniform multiplier
    ``k = round(ratio ** (1 / num_growable_axes))`` to stay close to
    the byte budget, and caps each axis at ``chunks[i] * ceil(shape[i]
    / chunks[i])`` so a small array does not get a shard larger than
    itself. Every shard axis is an integer multiple of the
    corresponding chunk axis, as required by zarr v3.

    Parameters
    ----------
    chunks
        Chunk shape, one int per axis.
    shape
        Array shape, one int per axis.
    dtype
        Array dtype; only ``itemsize`` is consulted.
    target_bytes
        Target uncompressed shard size. Typically
        `DEFAULT_TARGET_SHARD_BYTES`.

    Raises
    ------
    ValueError
        If ``len(chunks) != len(shape)``.
    """
    if len(chunks) != len(shape):
        raise ValueError(
            f"chunks rank {len(chunks)} does not match shape rank {len(shape)}."
        )
    itemsize = dtype.itemsize
    if itemsize == 0:
        return None
    chunk_bytes = math.prod(chunks) * itemsize
    if chunk_bytes >= target_bytes:
        return None
    growable = [i for i in range(len(shape)) if chunks[i] < shape[i]]
    if not growable:
        return None
    ratio = target_bytes / chunk_bytes
    k = max(1, round(ratio ** (1.0 / len(growable))))
    if k <= 1:
        return None
    shard = list(chunks)
    for i in growable:
        n_chunks_axis = math.ceil(shape[i] / chunks[i])
        shard[i] = min(chunks[i] * k, chunks[i] * n_chunks_axis)
    return tuple(shard)
```

Note: the helper takes its arguments keyword-only to match the style of `chunks_aligned_to` and to make calls at the use sites self-documenting.

- [ ] **Step 4: Update the test calls to use keyword arguments**

The unit tests written in Step 1 already pass arguments by keyword (`chunks=..., shape=..., dtype=..., target_bytes=...`). No change needed.

- [ ] **Step 5: Run the new tests to verify they pass**

Run: `.pyenv/bin/python -m pytest tests/test_zarr_layout.py::DefaultShardsTestCase -v`

Expected: All ten subtests PASS.

- [ ] **Step 6: Run the full layout test file as a regression check**

Run: `.pyenv/bin/python -m pytest tests/test_zarr_layout.py -v`

Expected: All tests pass (existing tests should be unaffected).

- [ ] **Step 7: Commit**

```bash
git add python/lsst/images/zarr/_layout.py tests/test_zarr_layout.py
git commit -m "feat(zarr): add default_shards helper for byte-budget shard sizing"
```

---

## Task 4: Wire `default_shards` into `ZarrOutputArchive.add_array`

**Files:**
- Modify: `python/lsst/images/zarr/_output_archive.py` (mask path ~L188, generic path ~L226)
- Modify: `tests/test_zarr_round_trip.py` (add a sharded round-trip test)

- [ ] **Step 1: Write the failing round-trip test**

Append to `tests/test_zarr_round_trip.py` (inside the `ZarrRoundTripTestCase` class, after `test_image_round_trip`):

```python
    def test_image_round_trip_writes_shards(self) -> None:
        # 300x300 float32: chunks (256, 256) -> shard (512, 512) by the
        # byte-budget rule (target 16 MiB, ratio ~64, k ~ 8 capped at the
        # 2-chunk-per-axis ceiling of 256 * 2 = 512).
        import zarr as _zarr

        from lsst.images.zarr._store import open_store_for_read

        original = Image(
            np.zeros((300, 300), dtype=np.float32),
            bbox=Box.factory[0:300, 0:300],
        )
        with RoundtripZarr(self, original) as roundtrip:
            with open_store_for_read(roundtrip.filename) as store:
                root = _zarr.open_group(store=store, mode="r", zarr_format=3)
                image_arr = root["image"]
                self.assertEqual(tuple(image_arr.chunks), (256, 256))
                self.assertEqual(tuple(image_arr.shards), (512, 512))
                # Single-chunk metadata arrays must NOT be sharded.
                lsst_json_arr = root["lsst_json"]
                self.assertIsNone(lsst_json_arr.shards)
            # Data round-trip is preserved.
            np.testing.assert_array_equal(roundtrip.result.array, original.array)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.pyenv/bin/python -m pytest tests/test_zarr_round_trip.py::ZarrRoundTripTestCase::test_image_round_trip_writes_shards -v`

Expected: FAIL — `image_arr.shards` is `None` because the archive doesn't populate shards yet.

- [ ] **Step 3: Wire `default_shards` into the mask-path branch of `add_array`**

Edit `python/lsst/images/zarr/_output_archive.py`. The existing imports at the top (currently lines 40–53) are multi-line tuples. Update them in place.

Change the `_common` import block to add `DEFAULT_TARGET_SHARD_BYTES`:

```python
from ._common import (
    DEFAULT_TARGET_SHARD_BYTES,
    ZarrCompressionOptions,
    ZarrPointerModel,
    archive_path_to_zarr_path,
    mask_dtype_for_plane_count,
)
```

Change the `_layout` import block to add `default_shards`:

```python
from ._layout import (
    affine_check,
    axes_for_archive_class,
    chunks_aligned_to,
    chunks_for,
    decorate_sub_archives,
    default_shards,
    serialize_fits_opaque_metadata,
)
```

In the mask branch (currently lines 180–200), replace:

```python
            chunks = self._chunks.get(name) or self._chunks.get(leaf)
            if chunks is None and self._image_chunks is not None:
                chunks = chunks_aligned_to(image_chunks=self._image_chunks, shape=packed.shape)
            extra: dict[str, Any] = {"_ARRAY_DIMENSIONS": ["y", "x"]}
            extra.update(flag_attrs.dump())
            ir_array = ZarrArray(
                data=packed,
                chunks=chunks,
                shards=self._shards.get(name),
                compression=self._compression.get(name),
            )
```

with:

```python
            chunks = self._chunks.get(name) or self._chunks.get(leaf)
            if chunks is None and self._image_chunks is not None:
                chunks = chunks_aligned_to(image_chunks=self._image_chunks, shape=packed.shape)
            shards = self._shards.get(name) or self._shards.get(leaf)
            if shards is None and chunks is not None:
                shards = default_shards(
                    chunks=tuple(chunks),
                    shape=tuple(packed.shape),
                    dtype=packed.dtype,
                    target_bytes=DEFAULT_TARGET_SHARD_BYTES,
                )
            extra: dict[str, Any] = {"_ARRAY_DIMENSIONS": ["y", "x"]}
            extra.update(flag_attrs.dump())
            ir_array = ZarrArray(
                data=packed,
                chunks=chunks,
                shards=shards,
                compression=self._compression.get(name),
            )
```

- [ ] **Step 4: Wire `default_shards` into the generic branch of `add_array`**

In the generic branch (currently lines 202–231), find this block:

```python
        ir_array = ZarrArray(
            data=np.ascontiguousarray(array),
            chunks=chunks,
            shards=self._shards.get(name),
            compression=self._compression.get(name),
        )
```

Replace with:

```python
        shards = self._shards.get(name) or self._shards.get(leaf)
        if shards is None and chunks is not None:
            shards = default_shards(
                chunks=tuple(chunks),
                shape=tuple(array.shape),
                dtype=array.dtype,
                target_bytes=DEFAULT_TARGET_SHARD_BYTES,
            )
        ir_array = ZarrArray(
            data=np.ascontiguousarray(array),
            chunks=chunks,
            shards=shards,
            compression=self._compression.get(name),
        )
```

Note: `chunks is not None` guards the unusual case where neither `_chunks` nor any layout default fired — `default_shards` only makes sense once we have a chunk shape. In practice `chunks` is non-`None` for `image`, `variance`, `mask`, and `psf`; for table columns and structured-array columns it is `None` (those use `add_table` / `add_structured_array`, which do not pass through this branch).

- [ ] **Step 5: Run the round-trip test to verify it passes**

Run: `.pyenv/bin/python -m pytest tests/test_zarr_round_trip.py::ZarrRoundTripTestCase::test_image_round_trip_writes_shards -v`

Expected: PASS — `image_arr.shards == (512, 512)` and `lsst_json_arr.shards is None`.

- [ ] **Step 6: Run all zarr round-trip tests as a regression check**

Run: `.pyenv/bin/python -m pytest tests/test_zarr_round_trip.py -v`

Expected: All tests pass — existing data-equality assertions are unchanged.

- [ ] **Step 7: Run the full zarr suite**

Run: `.pyenv/bin/python -m pytest tests/ -k zarr -v`

Expected: All zarr-related tests pass.

- [ ] **Step 8: Commit**

```bash
git add python/lsst/images/zarr/_output_archive.py tests/test_zarr_round_trip.py
git commit -m "feat(zarr): default-shard image, variance, mask in ZarrOutputArchive"
```

---

## Task 5: Verify CellCoadd PSF gets sharded

**Files:**
- Modify: `tests/test_zarr_output_archive.py` (extend `ZarrPsfChunkingTestCase`)

- [ ] **Step 1: Write a PSF shard-defaulting test**

Append to `tests/test_zarr_output_archive.py` inside `ZarrPsfChunkingTestCase` (currently lines 277–298), after `test_psf_user_override_wins`:

```python
    def test_psf_array_gets_default_shards(self) -> None:
        # 25x25 cells of 150x150 float32: chunk_bytes = 90 KiB,
        # ratio ~ 186, k = round(sqrt(186)) = 14 -> shard (14, 14, 150, 150).
        psf = np.zeros((25, 25, 150, 150), dtype=np.float32)
        archive = ZarrOutputArchive(archive_class="CellCoadd")
        archive.add_array(psf, name="psf")
        node = archive.document.root.get("/psf")
        self.assertEqual(tuple(node.shards), (14, 14, 150, 150))

    def test_psf_user_shard_override_wins(self) -> None:
        psf = np.zeros((25, 25, 150, 150), dtype=np.float32)
        archive = ZarrOutputArchive(
            archive_class="CellCoadd",
            shards={"psf": (5, 5, 150, 150)},
        )
        archive.add_array(psf, name="psf")
        node = archive.document.root.get("/psf")
        self.assertEqual(tuple(node.shards), (5, 5, 150, 150))

    def test_small_psf_skips_sharding(self) -> None:
        # 2x3 cells of 21x21 float32: chunk_bytes = 1764 B, ratio ~9295,
        # but ceil(2/1) * ceil(3/1) = 6 cells total -> capped shard equals
        # the array; effective shard becomes (2, 3, 21, 21) which equals
        # shape, so no sharding is meaningful. The byte-budget rule still
        # produces a tuple — verify it is the capped value, not None.
        psf = np.zeros((2, 3, 21, 21), dtype=np.float32)
        archive = ZarrOutputArchive(archive_class="CellCoadd")
        archive.add_array(psf, name="psf")
        node = archive.document.root.get("/psf")
        # Either way is acceptable: shards=(2,3,21,21) (capped) or shards=None.
        # The default rule returns the capped value; assert that.
        self.assertEqual(tuple(node.shards), (2, 3, 21, 21))
```

- [ ] **Step 2: Run the new tests**

Run: `.pyenv/bin/python -m pytest tests/test_zarr_output_archive.py::ZarrPsfChunkingTestCase -v`

Expected: PASS for all subtests, including the two existing ones (`test_psf_array_uses_single_cell_chunks` and `test_psf_user_override_wins`). For the small PSF, the helper computes `k = round(sqrt(16777216 / 1764)) = 98`, then caps each growable axis at the cell-grid extent (2 and 3), yielding shard `(2, 3, 21, 21)` — the whole array fits in one shard, which is the desired outcome (6 chunks bundled into one file).

- [ ] **Step 3: Run the full output-archive test file as a regression check**

Run: `.pyenv/bin/python -m pytest tests/test_zarr_output_archive.py -v`

Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_zarr_output_archive.py
git commit -m "test(zarr): cover CellCoadd PSF shard defaulting and overrides"
```

---

## Task 6: Verify sharded write round-trips through `ZipStore`

**Files:**
- Modify: `tests/test_zarr_store.py` (add a sharded round-trip via zip)

- [ ] **Step 1: Write the test**

Append to `tests/test_zarr_store.py` inside `StoreDispatchTestCase`, after `test_create_only_refuses_existing`:

```python
    def test_zip_store_round_trips_sharded_array(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "out.zarr.zip")
            data = np.arange(300 * 300, dtype=np.float32).reshape(300, 300)
            with open_store_for_write(target) as store:
                group = zarr.create_group(store=store, zarr_format=3)
                arr = group.create_array(
                    name="image",
                    shape=data.shape,
                    chunks=(256, 256),
                    shards=(512, 512),
                    dtype=data.dtype,
                )
                arr[:] = data
            with open_store_for_read(target) as store:
                group = zarr.open_group(store=store, mode="r", zarr_format=3)
                image = group["image"]
                self.assertEqual(tuple(image.chunks), (256, 256))
                self.assertEqual(tuple(image.shards), (512, 512))
                np.testing.assert_array_equal(image[...], data)
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `.pyenv/bin/python -m pytest tests/test_zarr_store.py::StoreDispatchTestCase::test_zip_store_round_trips_sharded_array -v`

Expected: PASS — `ZipStore` handles sharded arrays without special handling on our side.

If this fails (zarr-python 3.x not honoring shards through `ZipStore`), stop and discuss with the user before proceeding. The spec assumes this works; failure would be a real finding worth surfacing.

- [ ] **Step 3: Run the full store test file**

Run: `.pyenv/bin/python -m pytest tests/test_zarr_store.py -v`

Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_zarr_store.py
git commit -m "test(zarr): round-trip a sharded array through ZipStore"
```

---

## Task 7: Final regression sweep and changelog

**Files:**
- Modify: `doc/changes/` (add changelog fragment if the project uses one)

- [ ] **Step 1: Run the full zarr test suite**

Run: `.pyenv/bin/python -m pytest tests/ -k zarr -v`

Expected: All zarr tests pass.

- [ ] **Step 2: Run the full project test suite**

Run: `.pyenv/bin/python -m pytest tests/ -v`

Expected: All tests pass; no unrelated regressions.

- [ ] **Step 3: Check whether a changelog fragment is required**

Run: `ls doc/changes/ 2>/dev/null && head -20 doc/changes/README.rst 2>/dev/null || echo "no changelog dir"`

If `doc/changes/` exists, follow the existing fragment-naming convention. If a `.rst`/`.md` template can be found in nearby commits (look at `git log --oneline -- doc/changes/`), match that style.

If a fragment is required, create one summarising:
- "Default sharding now enabled for image, variance, mask, and PSF arrays in zarr archives. The per-axis chunk default has been lowered from 1024 to 256 to better suit cutout-style science access patterns. Public API is unchanged. Tunable via `LSST_IMAGES_ZARR_TARGET_SHARD_BYTES`."

If no changelog system is detected, skip this step.

- [ ] **Step 4: Verify mypy passes**

Run: `.pyenv/bin/python -m mypy python/lsst/images/zarr/`

Expected: No new type errors. (The zarr module was clean as of commit `9c2f01e`; this change adds only typed code.)

- [ ] **Step 5: Final commit (only if a changelog fragment was added)**

```bash
git add doc/changes/<fragment>
git commit -m "docs: changelog fragment for zarr default sharding"
```

- [ ] **Step 6: Sanity check the full diff**

Run: `git log --oneline origin/main..HEAD`

Expected: 5–6 commits — one per task above, in order.

Run: `git diff --stat origin/main..HEAD`

Expected: Three production files (`_common.py`, `_layout.py`, `_output_archive.py`) and four test files modified, plus possibly a changelog fragment. No other source files changed.

---

## Self-review notes

- **Spec coverage**: Architecture (Task 1, Task 2 add constants; Task 3 adds helper; Task 4 wires it in), the `default_shards` rule (Task 3), per-array behaviour (Tasks 4 + 5), error handling (covered by Task 3 unit tests), backward compatibility (Task 4 step 6 regression sweep, Task 6 zip round-trip), all five testing categories from the spec (Task 3 unit tests, Task 2 env-var test, Task 4 round-trip integration, Task 5 PSF-specific round-trip, Task 6 zip round-trip).
- **No placeholders**: every code step shows the full code; every test step shows the full assertion.
- **Type consistency**: the helper signature is `default_shards(*, chunks, shape, dtype, target_bytes)` everywhere — task 3 (definition), task 3 (unit tests), task 4 (call sites). The constant name `DEFAULT_TARGET_SHARD_BYTES` and chunk constant `DEFAULT_CHUNK_AXIS_LIMIT` are used consistently across `_common.py`, `_layout.py`, and `_output_archive.py`.
- **Spec deviation**: the spec's claim that "object dtype has itemsize 0" is incorrect — `np.dtype('O').itemsize == 8`. The `itemsize == 0` guard still exists (it triggers on `np.dtype('V0')`), and the unit test in Task 3 covers it via void(0) instead of object.
