# Schema fixtures

Committed reference fixtures for the serialization data models, one per schema version.
Each fixture is the instance-level twin of the frozen schema document of the same version under `schemas/` at the repository root.

The layout is `{name}/{name}-{version}[-{variant}].json`, where `{version}` is `X.Y.Z` for a frozen schema or `X.Y.Z.dev` for one still in development.
A filename carries only the release part of the version; the exact development counter lives in the `schema_version` stamp inside the file.

Every fixture here is read through its live model on every test run, so a model change that alters what a file looks like shows up as fixture drift.
Run `lsst-images-admin fixtures refresh` after changing a development schema, and `lsst-images-admin fixtures freeze` when finalizing one.

## Variants

A `-{variant}` suffix marks a further fixture of the same schema at the same version, so a composite model can cover more than one branch of its optional and union fields, and a fixture derived from a real file can sit beside a synthetic one.
`visit_image` has `dp1` and `dp2`; `difference_image` has `dp2`.
Each variant is checked, refreshed, frozen, and version-paired independently.

Adding one means writing the file, listing its `(name, variant)` pair in `tests/test_serialization_io.py`, and canonicalizing it with `fixtures refresh` while the version is still in development.
Generate the file by serializing a model rather than hand-authoring it: a hand-edited payload value round-trips cleanly, so nothing but review would catch it.

## Reserved variant names

`as_shipped` preserves bytes exactly as a real shipped file produced them, and is never rewritten by the tooling.
`canonical` is its canonicalized twin; comparing the two pins how a shipped file is normalized on read.
`fixtures refresh` regenerates the twin only while its version is in development.
Once finalized, the twin is immutable and a mismatch requires a schema-version bump.

`cell_coadd-1.0.0` is the only schema that is both frozen and shipped, and it shipped before validated schema management existed, so shipped files carry `cell_shape: [4, 4]` where the code now writes `{"y": 4, "x": 4}`.
Both spellings must keep reading, and the `as_shipped` fixture is what records that.

## Retired fixtures

A fixture under a schema's `retired/` subdirectory is expected to be *rejected* by current code with an `ArchiveReadError`.
A fixture is retired when the current model genuinely can no longer validate that shape and no migration covers the gap; retiring it is how a read contract ends, asserting the new behavior rather than merely no longer testing the old one.
`MIN_READ_VERSION` does not drive this on its own: it gates old readers refusing new files, and says nothing about new code refusing old ones.

## Coverage

Run `lsst-images-admin fixtures coverage` to see what these fixtures exercise, which is the question to ask when finalizing a version, because a frozen fixture cannot be widened without a `SCHEMA_VERSION` bump.
A schema embedded in another schema's tree is exercised by the container's fixture as well, since every embedded tree carries its own version stamp, so a sub-model's coverage is the union over the whole tree rather than just its own fixtures.
Gaps in the report are information, not failures: nothing obliges a composite model to hold every sub-schema its own schema admits.

`psfex_psf` has no fixture: it needs PSFEx data this package cannot construct.
The exemption is recorded in `tests/test_schema_fixtures.py`.

`tests/data/detector.json` is separate sample data, not a schema fixture, and is not part of this tree.

## Fixtures derived from real files

The `cell_coadd` `as_shipped` fixture, the `visit_image` `dp1` / `dp2` variants, and the
`difference_image` `dp2` variant (it has no `dp1`) come from real on-disk files via
`lsst.images.tests._minify_for_fixtures`, which reads a real archive, takes a small representative
subset, and writes it back as JSON:

    python -c "
    from lsst.images.tests._minify_for_fixtures import minify
    minify('cell_example.fits', 'tests/data/schemas/cell_coadd/cell_coadd-1.0.0-as_shipped.json')
    minify('dp1.fits', 'tests/data/schemas/visit_image/visit_image-1.0.0.dev-dp1.json')
    minify('dp2.fits', 'tests/data/schemas/visit_image/visit_image-1.0.0.dev-dp2.json')
    minify('difference_image_dp2.fits', 'tests/data/schemas/difference_image/difference_image-1.0.0.dev-dp2.json')
    "

`CellCoadd` regeneration works with just this package installed; `VisitImage` needs a full Rubin environment so the real PSF can be read.
