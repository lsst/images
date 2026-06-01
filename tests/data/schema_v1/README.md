# v1 schema fixtures

Reference JSON fixtures for every concrete `ArchiveTree` subclass that
exercises `schema_version` / `min_read_version` / `schema_url` stamping
at the v1 release.

## Regenerating

These fixtures are produced by:

    .pyenv/bin/python -m lsst.images.tests._make_schema_fixtures

The helper overwrites every `<schema_name>.json` file in this
directory. Run it after any deliberate `SCHEMA_VERSION` /
`MIN_READ_VERSION` bump.

## Coverage

Builders for every concrete `ArchiveTree` subclass live in
`python/lsst/images/tests/_make_schema_fixtures.py`. Five subclasses
require external test data and have builders that raise
`NotImplementedError`:

- `piff_psf` — needs Piff data.
- `psfex_psf` — needs PSFEx data.
- `cell_coadd` — needs `lsst.cell_coadds` and a real coadd file.
- `camera_frame_set` — needs an AST representation from a legacy camera.
- `spline_field` — `SplineField.serialize` writes its data as an array
  reference, which the JSON archive doesn't currently store inline.
  Fixtures for this would need to come from a FITS archive instead.

Tests over these fixtures should skip the missing names rather than
hard-fail.

## Legacy fixtures

`legacy/` holds fixtures derived from real on-disk files (converted from
legacy formats) via `_minify_for_fixtures.py`, which reads a real archive,
takes a small representative subset, and writes it back out as JSON:

- `cell_coadd.json` — a `CellCoadd` morphed onto a tiny cell grid: a small
  block of cells (including a missing cell) decimated to a few pixels each,
  with the grid topology, mask schema, band and provenance shape preserved.
- `visit_image_dp1.json`, `visit_image_dp2.json` — `VisitImage`s cropped to
  a ~16x16 corner.  They keep the real Piff PSF and detector frames, but to
  stay small: the PSF's field interpolation is truncated to order 0 (the
  field-averaged PSF), the projection's pixel->sky mapping is replaced by its
  affine approximation over the box, and amplifiers and aperture corrections
  are trimmed to a couple of representative entries.  All of these are
  schema-identical to the full versions.

These exercise the read path on data that the synthetic builders above
cannot reproduce. Regenerate them by pointing `minify` at the real files:

    python -c "
    from lsst.images.tests._minify_for_fixtures import minify
    minify('cell_example.fits', 'tests/data/schema_v1/legacy/cell_coadd.json')
    minify('dp1.fits', 'tests/data/schema_v1/legacy/visit_image_dp1.json')
    minify('dp2.fits', 'tests/data/schema_v1/legacy/visit_image_dp2.json')
    "

`CellCoadd` regeneration works with just this package installed;
`VisitImage` needs a full Rubin environment so the real PSF can be read.
