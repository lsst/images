# NDF archive — follow-up tickets

Open items from PR #33 (DM-54817) review that the reviewer agreed are
better handled separately. Each section below is sized to become one
Jira ticket; the four can be sequenced in a single follow-up PR or
spread out as preferred.

Cross-references are to GitHub PR comments on
`https://github.com/lsst/images/pull/33` and to current code in this
repo on branch `tickets/DM-54817`.

---

## Ticket A — Type-driven array routing in `add_array`

**Review thread:** PR #33 `_output_archive.py:295` (commit `4f59cf6`).

### Problem

`NdfOutputArchive.add_array` (`python/lsst/images/ndf/_output_archive.py:417-468`)
routes to NDF slots by exact string match on the `name` kwarg —
`name == "image"` → `/DATA_ARRAY`, `name == "variance"` → `/VARIANCE`,
`name == "mask"` → `/QUALITY` or `/MORE/LSST/MASK`. That works for the
top-level `MaskedImage` case because its `serialize()` happens to pass
those exact strings, but it fails for any caller that nests an image or
mask deeper in an archive path. The concrete example reviewer raised is
a hypothetical stamps-style container: a sub-NDF at `STAMP1` whose own
mask serializes as `name="stamp1/mask"`, which would miss the
`elif name == "mask"` clause entirely and not get the 8-bit Starlink-
compatible `QUALITY` placement.

The deeper problem is that the archive is being told *what to name the
array*, not *what kind of thing it is*. The producer (e.g. `Mask`)
already knows it is a mask; that information should flow through the
archive interface rather than being re-inferred from a name string.

### Proposed approach

1. Add an optional `kind` (or `component`) parameter to
   `OutputArchive.add_array` in `python/lsst/images/serialization/_archive.py`
   (the abstract base), with values drawn from a small `StrEnum`:
   `ArrayKind.IMAGE`, `ArrayKind.VARIANCE`, `ArrayKind.MASK`,
   `ArrayKind.OTHER` (or `None` defaulting to OTHER). Plumb it through
   the `Image` / `Mask` / `MaskedImage` serializers in
   `python/lsst/images/_image.py`, `_mask.py`, `_masked_image.py`.
2. In `NdfOutputArchive.add_array`, dispatch on `kind` instead of
   `name`. The current routing keeps working for top-level
   `MaskedImage` (because `Image` would pass `kind=IMAGE`, etc.) and
   *also* works for nested stamps because the producer sets `kind`
   regardless of the path it's serialized at.
3. The `name` argument continues to drive *where* the array lands
   (relative archive path); `kind` only decides which native NDF slot
   to use when at the right depth.
4. FITS and JSON archives ignore `kind` (FITS already uses HDU names;
   JSON uses inline arrays).

### Scope

Medium. Touches one base-class signature, three serializers in
`lsst.images`, and the NDF output archive. Existing tests should
continue to pass with no semantic changes; add a new test that
exercises a name like `"stamp1/mask"` going to the right place.

### Dependencies

Independent. Touches files that Ticket B also wants to extend, so
land first if doing both.

---

## Ticket B — Hierarchical native-NDF WCS writes

**Review thread:** PR #33 `_output_archive.py:129` (commit `d8520aa`).

### Problem

`NdfOutputArchive.add_tree` writes the canonical `/WCS` HDS structure
only for the top-level NDF (plus a special carve-out for the native
mask at `/MORE/LSST/MASK`). Anywhere an NDF subtree gets serialized
into a deeper archive path, its native `WCS` component is silently not
written — Starlink tools see a typed NDF without sky coordinates even
when the LSST object had a projection.

The most-visible example today is `ColorImage`: each channel is its
own `<NDF>` and currently gets a 2-D pixel-only `WCS` from the
`_wcs_ndf_paths=("/RED", "/GREEN", "/BLUE")` explicit list. The same
infrastructure should work generically for any sub-NDF: a future stamp
container, a `VisitImage` companion image, etc.

### Proposed approach

1. Replace the explicit `_wcs_ndf_paths` list with discovery. After
   the serializer runs, walk `self._document.root` looking for nodes
   with `CLASS="NDF"` (i.e. `Ndf` instances) and write a canonical
   `WCS` to each whose corresponding LSST-side object has a
   `projection`.
2. The hard bit is "which LSST projection goes with which sub-NDF."
   Options, in increasing order of invasiveness:
   - **Producer-registers:** add an explicit
     `NdfOutputArchive.register_projection(archive_path, projection, bbox)`
     called by serializers that know their sub-tree is going to be a
     native NDF. The top-level call in `write()` continues to use the
     object's own `.projection`/`.bbox`. The downside is each producer
     needs to know whether it ends up at an NDF slot.
   - **Base-class hook:** add an optional
     `OutputArchive.add_projection(name, projection, bbox)` method on
     the abstract base. FITS and JSON ignore it; NDF records the
     `(archive_path → projection)` mapping. After serialization
     `add_tree` resolves the archive paths to HDS paths via the
     existing layout helpers. This is closer to what the reviewer
     described as "real-HDS locator" semantics.
3. The native-mask 3-D WCS carve-out (parent sky frame plus
   `mask-byte` axis) generalises to "any sub-NDF whose data array has
   one more dim than the parent" or to an explicit producer-declared
   shape, depending on which option above is chosen.
4. Update the spec doc's writer-routing section to document the
   new behaviour.

### Scope

Medium-large. Either option requires changes to the
`OutputArchive` base class and to at least one serializer. The
discovery pass itself is short.

### Dependencies

Best done after Ticket A — both want a richer
producer→archive information channel, and doing them together (a
single base-class extension that carries kind + projection) avoids
churn.

---

## Ticket C — Opaque-metadata abstraction beyond `FitsOpaqueMetadata`

**Review thread:** PR #33 `_output_archive.py:60` (commit `af07296`).

### Problem

The NDF archive currently reuses
`lsst.images.fits._common.FitsOpaqueMetadata` for objects that need to
preserve a FITS-style header alongside the structured data
(`python/lsst/images/ndf/_output_archive.py:36,98-100,165-169`,
`python/lsst/images/ndf/_input_archive.py:71-74,184-195`). That works
because NDF chose to keep FITS cards in `/MORE/FITS`, but it bakes a
FITS-specific type into the public surface of a non-FITS archive and
blocks adding more backends.

The author has signalled (PR thread on this comment) that the right
fix is a broader rethink of how opaque metadata is presented to
callers — hiding the FITS-shaped abstraction from non-FITS code paths.

### Proposed approach

1. Introduce a backend-neutral opaque-metadata interface in
   `python/lsst/images/serialization/`: an abstract class
   (e.g. `OpaqueMetadata`) with `get`/`set` methods keyed by an
   extension name + version (or whatever the FITS-side
   `ExtensionKey` provides today). The FITS archive provides a
   `FitsOpaqueMetadata` subclass that wraps `astropy.io.fits.Header`
   directly; the NDF archive provides an `NdfOpaqueMetadata` subclass
   (or reuses the FITS one with a documented role) that knows how to
   serialize to NDF `_CHAR*80` records.
2. Update the top-level `write`/`read` paths and the
   `_opaque_metadata` attribute on `Image`/`MaskedImage` to refer to
   the abstract type. Reading a FITS file populates the FITS subclass;
   reading an NDF populates whatever the NDF backend produces.
3. Round-tripping a FITS-sourced object through an NDF file (currently
   the only real use case) needs the FITS header content to come out
   identical on the FITS side. Add a cross-archive round-trip test
   in `tests/test_masked_image.py` or similar.
4. Update the NDF design spec to drop the
   "no parallel `NdfOpaqueMetadata`" note from the writer-routing
   section.

### Scope

Medium. Mostly mechanical once the interface is decided. Touches
`python/lsst/images/serialization/`, both FITS and NDF archives, and
the `Image`/`MaskedImage` callers that read/write the `_opaque_metadata`
attribute.

### Dependencies

Independent of Tickets A and B. Useful to do early if a third archive
backend is on the horizon, otherwise it can wait.

---

## Ticket D — Generalize the mask serialization carve-out into the archive interface

**Review thread:** PR #33 `_mask.py:1003` (commit `fd64c1f`).

### Problem

`Mask._serialize` (around `python/lsst/images/_mask.py:600-680`)
contains explicit branching between "compatible mask" (single-byte,
≤8 planes) and "incompatible mask" (wider dtype or more planes),
because the NDF archive can only store the former natively in
`QUALITY`. The FITS and JSON archives have no such constraint; the
branching exists purely to feed the NDF backend's limitation.

The reviewer's point is that this leaks an archive-specific quirk
into the data class. The mask should hand the archive its native
3-D representation and let the archive decide whether to collapse
it (NDF) or store it as-is (FITS/JSON).

### Proposed approach

1. Drop the mask-shape branching from `Mask._serialize`. Always pass
   the full 3-D `(mask-byte, y, x)` array to
   `archive.add_array(..., kind=ArrayKind.MASK)` (relies on
   Ticket A's `kind` parameter).
2. Move the NDF-specific collapse-to-`QUALITY` logic that currently
   lives in `NdfOutputArchive.add_array` (`_collapse_mask_to_quality`,
   `_set_quality_array`) so it triggers on the new `kind=MASK`
   signal. The archive sees only the full mask and decides whether to
   also emit a collapsed 8-bit `QUALITY` based on shape and dtype.
3. The FITS archive can still use the native 3-D form (a 3-D image
   HDU with a sibling JSON schema entry) — no behaviour change.
4. Confirm the round-trip tests for both compatible and incompatible
   masks still pass.

### Scope

Small once Ticket A has landed (this is mostly a code-move). The
reviewer explicitly said "not necessarily on this ticket" — i.e.
a known follow-up.

### Dependencies

Requires Ticket A's `kind` parameter. Should land at the same time
as (or shortly after) Ticket A, to keep the abstract base-class
extension well-motivated by a concrete consumer.

---

## Suggested sequencing

A → B → D → C, or A+D together followed by B and then C.

A unlocks D and motivates B; the three together establish the
"producer-tells-archive-what-this-is" pattern. C is independent and
can interleave at any point but pays for itself most when a third
backend appears.
