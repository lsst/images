## Why

Test files across the suite use Python's `tempfile` module (`TemporaryDirectory`,
`NamedTemporaryFile`, `mkdtemp`) to create scratch space, despite pytest providing
a built-in `tmp_path` fixture that is cleaner, automatically cleaned up, and
standard practice in modern pytest codebases. One site (`test_serialization_backends.py`)
uses `tempfile.mkdtemp()` without any cleanup, creating a resource leak.
The pattern was flagged by `TODO[DM-54956]` in `test_diagram.py`.

## What Changes

- Replace all `tempfile.TemporaryDirectory` and `tempfile.NamedTemporaryFile` usages
  inside test functions with the `tmp_path: pathlib.Path` pytest fixture parameter.
- Fix the `tempfile.mkdtemp()` resource leak in `test_serialization_backends.py`.
- For non-fixture helper functions (`_write_archive`, `make_hdu_list`, `_cutdown`)
  that currently create their own temp directories, add a `tmp_path` parameter and
  have callers pass it through.
- Remove `import tempfile` from every file where it is no longer needed after the
  replacements.
- Remove the now-resolved `TODO[DM-54956]` comment from `test_diagram.py` line 677.

## Capabilities

### New Capabilities

- `tmp-path-convention`: Test temporary files are managed exclusively via pytest's
  `tmp_path` fixture, not the `tempfile` stdlib module.

### Modified Capabilities

<!-- No spec-level behavior changes; this is an internal test infrastructure cleanup. -->

## Impact

- Affects 13 test files under `tests/` (no production code changes).
- No functional change to what is tested — only how scratch files are created.
- The `delete_on_close=False` + `tmp.close()` boilerplate (used in NDF tests as a
  Windows compatibility workaround) is eliminated entirely.
- Tests must continue to pass unchanged after each file is converted.
