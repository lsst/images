## Why

The test suite still contains 14 files that use `unittest.TestCase`, making it
inconsistent with the pytest free-function style already adopted in prior
batches.  Converting the straightforward-to-moderate subset now keeps the
migration moving while deferring genuinely complex cases (dynamic subTest loops,
deeply nested context managers) to focused follow-on tickets.

## What Changes

- Convert 14 test files from `unittest.TestCase` to pytest free-functions and
  fixtures, following the style rules in `openspec/specs/pytest-test-style/`.
- Files in scope:

  **Easy / mechanical**
  - `tests/test_psfs.py`
  - `tests/test_fuzz.py`
  - `tests/test_color_image.py`
  - `tests/test_difference_image_extras.py`
  - `tests/test_ndf_hds.py`
  - `tests/test_serialization_basic_info.py`

  **Moderate**
  - `tests/test_image.py`
  - `tests/test_mask.py`
  - `tests/test_masked_image.py`
  - `tests/test_fields.py`
  - `tests/test_serialization_io.py`
  - `tests/test_serialization_reader.py`
  - `tests/test_formatter_cache.py`
  - `tests/test_transforms.py`

- **Deferred** (separate tickets): `test_cameras.py`, `test_cell_coadd.py`,
  `test_from_hdu_list.py`, `test_cli.py`, `test_diagram.py`,
  `test_ndf_input_archive.py`, `test_ndf_output_archive.py`.

- `setUp` / `setUpClass` involving I/O → `@pytest.fixture(scope="session")`
  with `pytest.skip()` guards inside the body; cheap object construction →
  module-level factory functions (not fixtures).
- Optional-dependency skip guards on individual test functions → named
  module-level skip markers (`skip_no_h5py`, `skip_no_afw`, etc.); guards
  on session fixtures → `pytest.skip()` inside the fixture body (marks on
  fixtures are prohibited by pytest 9+).
- `subTest` over fixed enumerable lists → `@pytest.mark.parametrize`.
- `subTest` used as a soft-failure guard inside `with RoundtripX(...):` blocks →
  removed; the guarded inner block becomes a `pytest.skip(...)` call or a
  separate skip-guarded helper function, preserving the intent without relying
  on `subTest`'s continue-on-failure semantics.

## Capabilities

### New Capabilities
*(none)*

### Modified Capabilities
- `pytest-test-style`: Extend the existing spec to cover the 14 files in this
  batch and add three clarifying sub-rules: (1) prefer factory functions over
  fixtures for cheap construction; (2) use named `skip_no_*` module-level
  constants for optional-dependency guards on test functions; (3) use
  `pytest.skip()` inside session fixture bodies for guards on shared
  expensive objects.

## Future Work (out of scope)

Prior-batch converted files may use factory functions that do file I/O on
every call, or `@pytest.mark.skipif` on individual tests where a shared
session fixture with an internal `pytest.skip()` would be cleaner and faster.
A follow-on pass should retrofit those files — particularly `test_visit_image.py`,
`test_geom.py`, and `test_schema_versioning.py`.  This is not urgent; the
existing style is correct, this is a quality and performance improvement.

## Impact

- `tests/` — 14 files rewritten; no change to production code or test coverage.
- `openspec/specs/pytest-test-style/spec.md` — minor addendum to existing spec.
- CI: same `pytest -r a -v -n 3`; all tests pass or skip cleanly on unchanged
  optional-dependency availability.
