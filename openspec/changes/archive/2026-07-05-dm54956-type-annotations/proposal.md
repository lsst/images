## Why

Several test files have incomplete or incorrect type annotations, violating the
project's 100%-typed-coverage requirement enforced by mypy. Two classes of problem
were found: (a) `tmp_path` parameters annotated as `pytest.MonkeyPatch` (the wrong
pytest type entirely — MonkeyPatch is a different fixture), and (b) entire test files
(`test_ndf_input_archive.py`, `test_ndf_output_archive.py`) and individual functions
in `test_serialization_reader.py` and `test_transforms.py` with no annotations at
all. Both classes were flagged by `TODO[DM-54956]` comments.

## What Changes

- Fix 8 `tmp_path: pytest.MonkeyPatch` annotations → `tmp_path: pathlib.Path` in
  `test_serialization_io.py` (6) and `test_fuzz.py` (2).
- Add missing return type `-> VisitImage` to the `visit_image` fixture in
  `test_serialization_reader.py`.
- Add missing `visit_image: VisitImage` and `tmp_path: pathlib.Path` parameter
  annotations to 15 test functions in `test_serialization_reader.py`.
- Add missing return types to 2 fixtures and missing parameter annotations to
  2 test functions in `test_transforms.py`.
- Add `-> None` return annotations to all 23 test functions in
  `test_ndf_input_archive.py`.
- Add `-> None` return annotations to all 28 test functions in
  `test_ndf_output_archive.py`.
- Remove the now-resolved `TODO[DM-54956]` markers from affected files.

## Capabilities

### New Capabilities

- `test-type-coverage`: All test functions and fixtures in `tests/` carry complete
  type annotations (parameters and return types).

### Modified Capabilities

<!-- No spec-level behavior changes; this is a typing cleanup within tests/. -->

## Impact

- Affects 5 test files under `tests/` (no production code changes).
- No functional change — annotations only; tests must continue to pass unchanged.
- mypy must report no new errors after each file is annotated.
- `pathlib` may need to be added to the imports of files that currently only have it
  transitively.
