# Schema-version compatibility: use cases and trade-offs (DM-54557)

A working note on the open question raised in review of
[`2026-05-15-schema-versioning-design.md`](./2026-05-15-schema-versioning-design.md):
the proposed ASDF-style symmetric major-bump rule may be too conservative.

> At first glance, I'm suspicious of the symmetry between newer/older major
> bumps; seems likely we'd work very hard on "new code + old file" and not as
> hard on "old code + new file" […] I could imagine us wanting to do a major
> bump that blocks "old code + new file" without also blocking "new code +
> old file".

This document walks through concrete kinds of schema change and shows how
each candidate compatibility scheme behaves, then makes a recommendation.

## 1. Use cases

For each case, "old code" is a release that pre-dates the change; "new
code" is a release that includes it. "File" means a JSON tree (or
container).

### Case A — Add an *optional* field (with default)

*Example: add `quality_score: float = 0.0` to `VisitImageSerializationModel`.*

- Old code reading new file: extra field — Pydantic's default is
  `extra="ignore"`, so it works.
- New code reading old file: field absent → default applies. Works.

Truly compatible in both directions. **Minor bump**, no asymmetry.

### Case B — Add a *required* field (no sensible default)

*Example: add a required `band: str` to `VisitImageSerializationModel`.*

- Old code reading new file: extra field, silently dropped. Works in the
  sense that nothing raises, but the consumer loses information.
- New code reading old file: required field missing → **fails validation**.

This is the asymmetric case the reviewer is worried about. We want new
code to keep reading old files (perhaps with a default or backfill), but
old code reading new files should fail loudly.

### Case C — Remove a previously-required field

*Example: drop `deprecated_index: int` from a model.*

- Old code reading new file: required field missing → **fails validation**.
- New code reading old file: extra field, ignored → works.

Asymmetric in the opposite direction from Case B.

### Case D — Rename or retype a field

*Example: rename `psf` → `point_spread_function`, or change `version: int`
to `version: str`.*

- Both directions break. Truly symmetric: both old-reads-new and
  new-reads-old fail.

### Case E — Add a new variant to a discriminated union

*Example: add `RhinoPSFSerializationModel` to the `PSF` union; some files
now contain it.*

- Old code reading new file containing the new variant → fails (unknown
  discriminator).
- New code reading old file → works (existing variants are still valid).

Asymmetric, like Case B. Especially common in `lsst.images`, which is
heavy on discriminated unions over `PSF`, `Field`, `Transform`, etc.

### Case F — Change semantics of an existing field

*Example: `flux` was in counts, now in nJy.*

- Both directions parse successfully but produce **silently wrong** values.
- Symmetric in spirit (both directions are broken), but no validation
  catches it. Demands a major bump for safety.

## 2. Schemes considered

### Scheme 1 — Current spec (single `version`, ASDF rule)

A single `major.minor.patch`. Reader rule:

- Different major → reject (both directions).
- Newer minor on disk → warn.
- Otherwise silent.

| Case | Bump | Old reads new | New reads old |
|------|------|---------------|---------------|
| A (optional add) | 1.0.0 → 1.1.0 | warn | silent ✓ |
| B (required add) | 1.0.0 → 2.0.0 | rejected (could've worked) | rejected (would want to work, needs migration) ✗ |
| C (required drop) | 1.0.0 → 2.0.0 | rejected (correct) | rejected (could've worked + migration) ✗ |
| D (rename / retype) | 1.0.0 → 2.0.0 | rejected | rejected ✓ |
| E (union variant) | 1.0.0 → 2.0.0 | rejected (correct) | rejected (would want to work) ✗ |
| F (semantics) | 1.0.0 → 2.0.0 | rejected | rejected ✓ |

**Pain point:** Cases B and E are the kinds of breaking change we'll
introduce most often, and they cost us "new code reads old file" forever
— exactly the case we most care about supporting.

### Scheme 2 — Two version axes (`schema_version` + `min_reader_version`)

The file carries two strings:

- `schema_version` — identifies the shape (bumped on *every* change).
- `min_reader_version` — smallest `schema_version` of code that can safely
  read this file.

The reader carries:

- Its own `schema_version` (in-code).
- A `min_writer_version` — smallest file `schema_version` it knows how to
  interpret.

Read rules:

1. If `file.min_reader_version > reader.schema_version` → **reject** (the
   file declares we're too old to read it).
2. If `file.schema_version < reader.min_writer_version` → **reject** (we
   no longer know how to interpret this old shape).
3. Otherwise, same major (with `min_writer_version` defaulting to
   `<major>.0.0`): silent read; newer minor than reader's: warn.

| Case | `schema_version` bump | `min_reader_version` change | Old reads new | New reads old |
|------|---------------------|-----------------------------|---------------|---------------|
| A (optional add) | minor | unchanged (still 1.0.0) | works | works |
| B (required add) | major | bumped to 2.0.0 | rejected (correct) | works (`min_writer_version` stays 1.0.0; reader has backfill code) |
| C (required drop) | minor | unchanged | works (extra field ignored) | works |
| D (rename / retype) | major | bumped to 2.0.0 *and* `min_writer_version` bumped | rejected | rejected |
| E (union variant) | major | bumped to 2.0.0 only when a file actually contains the new variant | rejected only when relevant | works |
| F (semantics) | major | bumped to 2.0.0 *and* `min_writer_version` bumped | rejected | rejected |

**Power gained:** Cases B and E — by far the most common kinds of
breaking change — let new code keep reading old files. The cost is that
some "won't read this" decisions now require us to also bump
`min_writer_version`, and require explicit backward-compat code in the
new reader for older shapes (a lightweight migration story for additive
changes).

### Scheme 3 — Defer the asymmetry

Keep the spec as written and don't promise the symmetric rejection
forever — when we later need asymmetry, we add a migration framework.
Simplest scheme for now; defers everything to "we'll figure it out when
it hurts."

## 3. JSON-on-disk shapes

Scheme 1:

```json
{
  "schema_url": "https://images.lsst.io/schemas/image-1.0.0",
  "schema_version": "1.0.0",
  "data": { ... }
}
```

Scheme 2:

```json
{
  "schema_url": "https://images.lsst.io/schemas/image-2.0.0",
  "schema_version": "2.0.0",
  "min_reader_version": "2.0.0",
  "data": { ... }
}
```

`min_reader_version` is omitted from the wire form (or set equal to
`schema_version`) until we've actually used the asymmetry — so the common
case stays one extra string.

## 4. Recommendation

**Scheme 2**, because:

- It is the only scheme that addresses the reviewer's concern. Schemes 1
  and 3 do not.
- The two breaking changes we'll make most often (Case B: add a required
  field; Case E: add a union variant) are exactly the ones where Scheme 2
  preserves "new code reads old file" — at the cost of a few lines of
  backfill in the reader.
- The on-disk cost is one extra string field. The reader logic is a
  one-line addition to the existing `_check_compat` helper.
- Scheme 2 degrades gracefully to Scheme 1 if we choose: keeping
  `min_reader_version == schema_version` always reproduces the symmetric
  ASDF behavior. So adopting Scheme 2 doesn't lock us in.

Counter-argument for Scheme 1/3: the cost of Scheme 2 isn't really the
extra field, it's the *discipline* of remembering to set
`min_reader_version` correctly on every change, plus writing
backfill/migration code for the cases where we want new code to read old
files. ASDF chose the conservative rule for a reason — it's hard to get
asymmetric compatibility right, and easy to claim more compatibility than
you actually have. If we don't intend to write the backfill code, Scheme
2 is just Scheme 1 with extra ceremony.

The recommendation stands on the assumption that we *do* want to write
that backfill code at least sometimes — particularly for union-variant
additions, which are common and where the backfill is trivial.

## 5. Open questions

- Should `min_reader_version` apply *per-subclass* (so a model that has
  not had any breaking change just emits its own `schema_version`), or
  should there be a top-level `min_reader_version` for the whole tree?
  Per-subclass is more precise but adds noise to common files; top-level
  is coarser but simpler. Probably per-subclass, mirroring `schema_version`.
- How does this interact with the per-backend container version? FITS
  and NDF container versions probably don't need this asymmetry —
  container layout changes tend to be all-or-nothing. Probably keep
  Scheme 1 for containers, Scheme 2 for data models.
- Do we want a CI check that flags `schema_version` bumps without
  matching `min_reader_version` decisions? Likely yes; out of scope for
  this design but worth noting.
