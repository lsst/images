# NDF/HDS name shrinker for over-long archive path components

## Problem

The NDF backend stores hoisted sub-trees and arrays as HDS components under
`/MORE/LSST/...`.
HDS limits every component name to 16 characters.
`archive_path_to_hdf5_path_components` (`ndf/_common.py`) currently uppercases
each path component and raises `ValueError` when any component exceeds that
limit.
Real data hits this: writing `cell_coadd.fits` to NDF fails because the archive
path `/noise_realizations/0` contains the component `NOISE_REALIZATIONS` (18
characters).

The same branch also added version disambiguation for repeated archive names
(`OutputArchive._register_name`): when a logical name is registered more than
once, hierarchical backends append `_{version}` to the leaf component (e.g.
`data` written twice becomes `data` and `data_2`).
Any shrinking mechanism must not undo that disambiguation.

## Key invariant

The reader never re-derives a path from an archive name.
`get_array` takes the stored `ndf:<path>` source string verbatim and looks it up
in the hierarchy; `deserialize_pointer` uses `NdfPointerModel.path` directly.
The only requirement is therefore:

> the path written into the JSON must equal the path of the node in the HDS
> hierarchy.

Both the JSON `source`/`pointer.path` and the on-disk structure are built from
the output of `archive_path_to_hdf5_path` / `archive_path_to_hdf5_path_components`.
That single chokepoint is where shrinking happens, and fixing it propagates to
both sides automatically.
The reader needs no changes and no ability to reverse the shrink.

## Approach

Port the deterministic, stateless shrinking scheme from
`lsst.daf.butler.name_shrinker.NameShrinker` (we copy the logic; we do not depend
on `daf_butler`, and we never need to un-shrink).

For a component longer than the limit, keep a readable prefix and append a hash
of the full component:

```
_shrink_hds_name(name, max_length=16, hash_size=4):
    name = name.upper()
    if len(name) <= max_length:
        return name
    digest = blake2b(name.encode("ascii"), digest_size=hash_size).hexdigest().upper()
    trunc = max_length - 2 * hash_size - 1          # 7 at the defaults
    return f"{name[:trunc]}_{digest}"               # exactly max_length characters
```

- `hash_size=4` gives an 8-hex-character (32-bit) digest, leaving a 7-character
  readable prefix at full width.
  32 bits keeps collisions negligible for the realistic number of distinct
  over-long names in a file (birthday probability ~1e-6 at 100 such names).
  No collision registry or guard is used; we accept daf_butler's level of trust
  in the digest.
- The component is uppercased before hashing so the on-disk token and its digest
  are self-consistent.
- Components at or under 16 characters pass through unchanged (uppercased only),
  preserving today's readable layout (`/MORE/LSST/PSF/COEFFICIENTS`) and every
  existing file.

### Version-aware shrinking

Version disambiguation and shrinking must compose so that the visible version
number survives. `noise_realizations_99` shrinks as "shrink `noise_realizations`
into `16 - len("_99")` characters, then append `_99`":

```
shrink_versioned_component(base, version, max_length=16, hash_size=4):
    suffix = f"_{version}" if version > 1 else ""
    return _shrink_hds_name(base, max_length - len(suffix), hash_size) + suffix
```

The hash is taken over the **base** name only, so:

- version 1 (no suffix) produces the same token whether routed through the plain
  translator or the versioned helper;
- versions of the same base share the base hash but differ by the visible
  `_{version}` suffix, so they never collide;
- different base names differ by hash.

Version numbers stay well under 100 in practice, so the suffix is at most three
characters and the readable prefix never collapses; the helper asserts the
result is within `max_length`.

### Applying the version structurally (not by detection)

Today the callers glue `_{version}` onto the path string and then translate.
If we kept that and then blindly truncated, we would silently destroy the
version distinction — exactly the heuristic to avoid.
Instead, the version is applied through `shrink_versioned_component` at the
component that carries it, producing a token already within the limit that the
translator then passes through untouched:

- **`add_array`** — the version sits on the leaf of the caller's `name`.
  Split the leaf off, run it through `shrink_versioned_component`, reattach, then
  translate.
- **`add_structured_array`** — the version sits on the `name` base, which may
  then gain a `/{column}` child (so the versioned component is *not* the leaf for
  multi-column tables).
  Apply `shrink_versioned_component` to the leaf of `name` before appending the
  column component.

`serialize_pointer` is unaffected: it dedupes by object identity and never
versions, so its path is shrunk by the plain per-component translator only.

## Changes

1. `ndf/_common.py`
   - Add `_shrink_hds_name` and `shrink_versioned_component`.
   - Change `archive_path_to_hdf5_path_components` to map each component through
     `_shrink_hds_name` instead of raising. (`archive_path_to_hdf5_path` needs no
     change; it composes the result.)
2. `ndf/_output_archive.py`
   - `add_array`: replace `archive_path = f"{archive_path}_{version}"` with the
     leaf-level `shrink_versioned_component` application described above.
   - `add_structured_array`: replace `name = f"{name}_{version}"` likewise.
3. Tests (`tests/test_ndf_common.py` and NDF round-trip tests)
   - Replace `test_archive_path_to_hdf5_path_rejects_long_components` (which
     asserts the raise) with assertions on the shrunk form.
   - Add: components ≤16 pass through unchanged; shrink is deterministic and
     ≤16; two distinct long names produce distinct tokens; the same base at
     different versions produces distinct tokens that keep the visible `_{n}`
     suffix.
   - Add a round-trip test that writes `cell_coadd.fits` (the reported failing
     case) to NDF and reads it back, asserting the noise-realization arrays
     survive.

## Out of scope

- Reversing a shrunk name back to the original (not needed; the reader uses the
  stored path).
- A collision-detection registry or dynamic hash sizing.
- Changes to the FITS backend, which encodes versions via `EXTVER` and has no
  16-character limit.
