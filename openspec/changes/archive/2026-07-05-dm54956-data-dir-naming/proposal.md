## Why

Test files use the name `DATA_DIR` for two completely different things — sometimes an
environment-variable path to external on-disk Rubin data, sometimes a path to the
in-repo committed fixtures under `tests/data/schema_v1/`. This ambiguity makes test
code harder to read and grep, and was explicitly flagged as a TODO in `test_visit_image.py`.

## What Changes

- Rename every module-level `DATA_DIR` that holds `os.environ.get("TESTDATA_IMAGES_DIR")`
  to `EXTERNAL_DATA_DIR` (8 files).
- Rename every module-level `DATA_DIR` that holds an `os.path.join(__file__, ...)` local
  path to `LOCAL_DATA_DIR` (4 files).
- Update all use-sites of the renamed variables within the same files.
- Fix the docstring typo `EXXTERNAL_DATA_DIR` → `EXTERNAL_DATA_DIR` in
  `test_difference_image_extras.py`.
- Remove the now-resolved `TODO[DM-54956]` comment from `test_visit_image.py`.

## Capabilities

### New Capabilities

- `data-dir-naming`: Consistent, unambiguous module-level constant names for the two
  distinct kinds of data directory used in the test suite.

### Modified Capabilities

<!-- No spec-level behavior changes; this is a pure rename within tests/. -->

## Impact

- Affects 12 test files under `tests/` (no production code changes).
- No functional change — pure rename; tests must continue to pass unchanged.
- Docstrings and comments that reference the old name must be updated in the same commit.
