## Why

Six test files still use `unittest.TestCase` subclasses, preventing the
`tests/` directory from reaching full pytest-native style.  The prior three
batches (batches 1–3 plus the skip-guard-consistency fix) established stable
patterns for every conversion situation present in these files; completing
them now finishes the migration for all files except `test_cell_coadd.py`,
which is deferred to a separate change.

## What Changes

- `tests/test_ndf_input_archive.py` — four `@skipUnless(HAVE_H5PY)`-guarded
  classes, no `setUp`; mechanical free-function conversion with `skip_no_h5py`.
- `tests/test_ndf_output_archive.py` — six `@skipUnless(HAVE_H5PY)`-guarded
  classes, no `setUp`; same pattern as `test_ndf_input_archive.py`.
- `tests/test_diagram.py` — six classes, only two with trivial `setUp` (cheap
  objects); purely mechanical conversion plus two factory-or-inline patterns.
- `tests/test_cameras.py` — two classes: `CamerasTestCase` needs a
  session fixture (reads legacy files, dual skip guards); `ReadoutCornerTestCase`
  is trivial.
- `tests/test_from_hdu_list.py` — five classes: one (`FromHduListTestCase`) has
  a `setUp` that builds a reusable `MaskedImage` HDU list; the other four have
  no `setUp`.
- `tests/test_cli.py` — eight classes; `setUp` uses `tempfile.TemporaryDirectory`
  + `addCleanup` → `tmp_path`; external-data tests get session fixtures; one
  `subTest` loop becomes `@pytest.mark.parametrize`.

All seven files remove `import unittest` and any `if __name__ == "__main__": unittest.main()` guard.

## Capabilities

### New Capabilities

None — this change implements existing requirements.

### Modified Capabilities

- `pytest-test-style`: No new requirements are added; all conversion patterns
  are already codified.  The spec is not modified.

## Impact

- `tests/test_ndf_input_archive.py`
- `tests/test_ndf_output_archive.py`
- `tests/test_diagram.py`
- `tests/test_cameras.py`
- `tests/test_from_hdu_list.py`
- `tests/test_cli.py`
- No changes to `python/` production code or public API.
- No new dependencies.
