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

`legacy/` is reserved for fixtures derived from real on-disk files via
`_minify_for_fixtures.py`. None ship with v1; populate them when there
are real legacy files to point at.
