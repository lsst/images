# Zarr v3 Sharding & Smaller Chunk Defaults

Date: 2026-05-25
Status: approved (brainstorming complete; awaiting implementation plan)

## Background

The zarr v3 backend currently writes arrays without shards. The
`shards` field is plumbed through `ZarrArray` and `ZarrOutputArchive`
all the way to `zarr.create_array`, but the archive never populates
it, so every chunk becomes a separate object on disk / in cloud
storage. The default per-axis chunk limit is 1024, which produces ~4
MiB float32 chunks — fine for full-image reads but on the larger end
for cutout-style science access.

Modern zarr v3 guidance for cloud-backed stores is:

- small-ish *logical chunks* sized for science access patterns;
- larger *physical shards* sized to amortise S3 / GCS request cost;
- avoid `.zarr.zip` for cloud distribution — keep it for packaging
  and local export.

This spec covers the first two. Zip support stays as it is today
(useful for tests and packaging); we are not deprecating it.

## Goals

- Default sharding "just works" with no public API changes — the
  caller does not have to think about chunk-vs-shard ratios.
- Smaller chunk default for science access (256² for plain images).
- One internal knob (`DEFAULT_TARGET_SHARD_BYTES`) and one env-var
  escape hatch for tuning without code changes.
- Old archives continue to read; round-trip data equality is
  preserved.

## Non-goals

- Changing `ZarrCompressionOptions` defaults.
- Re-tuning the `CellCoadd` cell-aligned chunk rule.
- Reading `shards` metadata back into the IR (`ZarrArray.from_zarr`
  still ignores it; the input archive slices through `zarr.Array`).
- Adding any new kwarg to public `write_zarr`.
- Deprecating `ZipStore`.

## Architecture

Three files are touched. No public API additions or renames.

```
python/lsst/images/zarr/
  _common.py         # +DEFAULT_CHUNK_AXIS_LIMIT (was hardcoded 1024 in _layout)
                     # +DEFAULT_TARGET_SHARD_BYTES (env-overridable, read once at import)
  _layout.py         # chunks_for: clamp constant moves to _common, value 1024 → 256
                     # +default_shards(chunks, shape, dtype, *, target_bytes) helper
  _output_archive.py # call default_shards alongside chunks_for in add_array;
                     # IR node gets shards populated when caller did not override
```

`_model.py` is **not** modified. `_group_to_zarr` continues to pass
`shards=array.shards` through to `zarr.create_array`. By the time the
IR reaches the writer, every array's `shards` is either explicitly
set by the caller, populated by the default helper, or `None` (for
tiny single-chunk arrays).

### Why eager defaulting in the archive layer

This pattern mirrors the existing `chunks_for` /
`chunks_aligned_to` helpers in `_layout.py`. The archive sets
shape-derived defaults at IR-construction time, the model writer
stays a dumb serialiser, and tests can assert IR-level shape
decisions without driving zarr. Lazy defaulting in `_group_to_zarr`
was considered and rejected — it would push policy logic into the
writer and make the IR's effective layout invisible until write
time.

## Constants

In `_common.py`:

- `DEFAULT_CHUNK_AXIS_LIMIT: int = 256` — replaces the hardcoded
  `_DEFAULT_AXIS_LIMIT = 1024` currently in `_layout.py`.
- `DEFAULT_TARGET_SHARD_BYTES: int` — `16 * 1024 * 1024` by default.
  At import time read `LSST_IMAGES_ZARR_TARGET_SHARD_BYTES`; if set,
  parse as base-10 int. A `ValueError` from `int()` propagates and
  fails import — silent typos are worse than loud failure. No
  `1MiB`-style suffix parsing.

`chunks_for` in `_layout.py` reads `DEFAULT_CHUNK_AXIS_LIMIT` from
`_common`. `chunks_aligned_to` is unchanged — it derives sibling
chunks from `image_chunks`, so it follows the new default
automatically.

The `CellCoadd` cell-aligned branch and the 4-D PSF branch
(`(1, 1, h, w)`) in `chunks_for` are unchanged — those are
class-specific layout rules, not default-clamp questions.

## The `default_shards` rule

Pure function, no archive-class arg, no `archive_metadata` arg:

```python
def default_shards(
    chunks: tuple[int, ...],
    shape: tuple[int, ...],
    dtype: np.dtype,
    *,
    target_bytes: int,
) -> tuple[int, ...] | None:
    if len(chunks) != len(shape):
        raise ValueError("chunks and shape rank mismatch")
    itemsize = dtype.itemsize
    if itemsize == 0:
        return None  # object dtype etc.
    chunk_bytes = math.prod(chunks) * itemsize
    if chunk_bytes >= target_bytes:
        return None  # one chunk already big enough
    growable = [i for i in range(len(shape)) if chunks[i] < shape[i]]
    if not growable:
        return None  # array fits in one chunk per axis
    ratio = target_bytes / chunk_bytes
    k = round(ratio ** (1.0 / len(growable)))
    if k <= 1:
        return None  # rounding produced a no-op shard
    shard = list(chunks)
    for i in growable:
        n_chunks_axis = math.ceil(shape[i] / chunks[i])
        shard[i] = min(chunks[i] * k, chunks[i] * n_chunks_axis)
    return tuple(shard)
```

Properties of the rule:

- **Integer-multiple alignment per axis**: every shard axis is
  `chunks[i] * m` for some `m ≥ 1`. zarr v3 requires this.
- **Spatial-only growth falls out for free**: a 3-D mask
  `(8, 4096, 4096)` chunked `(8, 256, 256)` has `growable = [1, 2]`,
  so the plane axis is left alone.
- **Tiny arrays skip sharding**: a `(N, 80)` FITS-card array, a
  single-chunk `lsst_json`, or any array whose chunks already cover
  every axis returns `None`.
- **CellCoadd PSF** `(25, 25, h, w)` chunked `(1, 1, h, w)` has
  `growable = [0, 1]`, so it shards the cell-grid axes only — no
  class-specific rule needed.
- **Cap at array bounds**: small arrays do not get shards larger
  than the array itself.

### Worked examples (target = 16 MiB)

| array              | shape              | chunks            | dtype   | result                   |
|--------------------|--------------------|-------------------|---------|--------------------------|
| `image` (4k×4k)    | (4096, 4096)       | (256, 256)        | float32 | shard `(2048, 2048)`     |
| `mask` (3-D, 4k)   | (8, 4096, 4096)    | (8, 256, 256)     | uint8   | shard `(8, 1536, 1536)`  |
| `variance` (4k×4k) | (4096, 4096)       | (256, 256)        | float32 | shard `(2048, 2048)`     |
| CellCoadd `psf`    | (25, 25, 150, 150) | (1, 1, 150, 150)  | float32 | shard `(14, 14, 150, 150)` |
| small image        | (600, 600)         | (256, 256)        | float32 | shard `(768, 768)` (capped) |
| `lsst_json`        | (N,)               | (N,)              | uint8   | `None`                   |
| `wcs_ast`          | (M,)               | (M,)              | uint8   | `None`                   |
| FITS primary       | (N, 80)            | (N, 80)           | uint8   | `None`                   |

## Per-array behaviour in `ZarrOutputArchive`

The pattern at every site that decides chunks today
(`_output_archive.py:183` for the MaskedImage path,
`_output_archive.py:202-241` for `add_array`):

```python
chunks = self._chunks.get(name) or self._chunks.get(leaf) or <derived>
shards = self._shards.get(name) or self._shards.get(leaf)
if shards is None:
    shards = default_shards(
        chunks, packed.shape, packed.dtype,
        target_bytes=DEFAULT_TARGET_SHARD_BYTES,
    )
ZarrArray(data=..., chunks=chunks, shards=shards, ...)
```

Coverage by call site:

| call site                                    | what gets sharded                          |
|----------------------------------------------|--------------------------------------------|
| MaskedImage path (`_output_archive.py:~183`) | `image`, `variance`, `mask`                |
| `add_array` generic (`_output_archive.py:~228`) | top-level sibling arrays                |
| `add_array` PSF branch (`_output_archive.py:~223`) | CellCoadd `psf` 4-D                  |
| JSON tree (`_output_archive.py:~142`, `:~337`) | `lsst_json` — helper returns `None`      |
| `wcs_ast` (`_output_archive.py:~406`)        | helper returns `None`                      |
| `serialize_fits_opaque_metadata` (`_layout.py:~281`) | helper returns `None`              |

Bulk pixel arrays (`image`, `variance`, `mask`, `psf`) and any
user-supplied extra arrays large enough to qualify gain `shards`.
Everything tiny / single-chunk is auto-`None`.

User overrides remain unchanged: passing `shards={"image": (...)}` to
`write_zarr` still wins because the override is consulted before the
default helper.

## Error handling

- `default_shards` raises `ValueError` on mismatched ndim between
  `chunks` and `shape`, mirroring `chunks_aligned_to`. All other
  inputs are total — no exceptions on well-formed numeric data.
- `dtype.itemsize == 0` (object dtype) → `None`. Defensive guard;
  object dtypes are not written today.
- Env-var parse failure raises at import.

## Backward compatibility

- **Reading old archives**: unaffected. `ZarrArray.from_zarr` does
  not consult `shards`. The input archive slices through
  `zarr.Array`.
- **Round-trip equality**: byte-equal data round-trips unchanged.
  Tests asserting array equality continue to pass.
- **On-disk file counts**: any test asserting a specific file count
  on disk needs updating. None known today.
- **Old test fixtures** (e.g. `dp1.zarr/`): readable as before; the
  change is write-side only.
- **ZipStore**: unchanged. `zarr.storage.ZipStore` accepts sharded
  arrays the same way as `LocalStore` — shards inside a zip are
  nested keys, no special handling.

### Performance note

A 4k×4k float32 image full-read goes from 16 chunks to 256 chunks
when the chunk default drops 1024 → 256. Sharding keeps the I/O
profile identical (4 GETs, same wire bytes), but per-chunk decode
runs 16× more often. Expected to be invisible: blosc-zstd decode
is fast and concurrent. If a benchmark regresses, the fallback is
to bump `DEFAULT_CHUNK_AXIS_LIMIT` to 512.

## Testing

### Unit tests for `default_shards` (new file `tests/test_zarr_layout.py`)

- 4k×4k float32 with `(256, 256)` chunks → `(2048, 2048)`.
- 3-D mask `(8, 4096, 4096)` uint8 with `(8, 256, 256)` chunks →
  `(8, 1536, 1536)` — plane axis untouched.
- Tiny 1-D single-chunk array → `None`.
- `chunks == shape` (single-chunk of any size) → `None`.
- `chunk_bytes >= target_bytes` (already-big chunk) → `None`.
- `k <= 1` boundary → `None`.
- Cap at array bounds: shape `(600, 600)`, chunks `(256, 256)`,
  ratio 64 → shard `(768, 768)`, not `(2048, 2048)`.
- Mismatched ndim raises `ValueError`.
- `dtype.itemsize == 0` → `None`.

### Env-var test (`tests/test_zarr_layout.py`)

- Set `LSST_IMAGES_ZARR_TARGET_SHARD_BYTES`, re-import the module
  in a subprocess (cleanest way to re-run import-time init), assert
  the constant changed.
- Garbage value raises at import.

### Round-trip / integration (extend existing zarr round-trip tests)

- Assert one large-image round-trip writes an `image` array whose
  on-disk metadata has non-`None` `shards` and shards are integer
  multiples of chunks per axis.
- Assert `lsst_json` and `wcs_ast` arrays come back with `shards`
  unset (or `None` in metadata).
- CellCoadd round-trip: assert PSF `psf` array's `shards != chunks`
  (i.e. the byte-budget rule actually fired).
- Existing data-equality round-trip checks are unmodified and
  continue to gate correctness.

### Zip round-trip (extend `tests/test_zarr_store.py`)

- Add one assertion to an existing zip test that a sharded
  write/read round-trips through `ZipStore` cleanly.

### Verification command

`.pyenv/bin/python -m pytest tests/ -k zarr` is the gate for the
implementation phase.
