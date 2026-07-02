## Why

Nine test files have already been migrated from `unittest.TestCase` to pytest
free functions, establishing a clear style baseline.  Thirty-two files remain
unconverted; this change tackles the next straightforward batch — files whose
structural simplicity (no `setUp`/`setUpClass` shared state, or only trivially
extractable `setUp`, no Mixin inheritance, small line counts) makes conversion
low-risk and mechanical.  Doing it in bounded batches keeps diffs reviewable
and avoids accumulating legacy style debt.

## What Changes

- Ten test files in `tests/` are rewritten from `unittest.TestCase` subclasses
  to pytest top-level `def test_*()` free functions.
- `setUp` methods that set a single shared object become `@pytest.fixture`
  functions; methods with no `setUp` need no fixture at all.
- `self.assert*` calls become plain `assert` statements or `pytest.raises`.
- `@unittest.skipUnless` decorators become `@pytest.mark.skipif`.
- `with self.subTest(...)` loops are kept as plain Python loops with `assert`
  (only promoted to `@pytest.mark.parametrize` where the mapping is natural
  and adds clarity).
- `if __name__ == "__main__": unittest.main()` guards are removed.
- `import unittest` is removed from each converted file.

## Capabilities

### New Capabilities

None — this is a pure style refactor with no new public API.

### Modified Capabilities

- `pytest-test-style`: The ten files listed below become conformant with the
  existing spec.  No requirement text changes; this change is an implementation
  of existing requirements on a new batch of files.

## Impact

**Files converted (10):**

| File | Lines | Tests | Key conversion note |
|---|---|---|---|
| `tests/test_schema_v1_fixtures.py` | 53 | 3 | No `setUp`; `subTest` stays as a loop |
| `tests/test_ndf_starlink_ingest.py` | 73 | 3 | No `setUp`; class skip → `@pytest.mark.skipif` per function |
| `tests/test_serialization_backends.py` | 76 | 6 | No `setUp`; 2 classes → flat free functions |
| `tests/test_fits_date_header.py` | 82 | 3 | `setUp` sets one `Image`; becomes a `@pytest.fixture`; `subTest` stays as loop |
| `tests/test_ndf_model.py` | 98 | 2 | No `setUp`; class skip → `@pytest.mark.skipif` per function |
| `tests/test_ndf_format_version.py` | 97 | 4 | No `setUp`; structural twin of already-converted `test_fits_format_version.py` |
| `tests/test_schema_v1_legacy_fixtures.py` | 108 | 2 | No `setUp`; soft skip stays inline |
| `tests/test_detached_archive.py` | 142 | 11 | Two minimal `setUp`s; one uses `addCleanup` → `@pytest.fixture` with `yield` |
| `tests/test_serialization_registry.py` | 165 | 11 | No `setUp`; `mock.patch` stays in-function; `_REGISTRY` cleanup stays inline |
| `tests/test_ndf_layout.py` | 277 | 4 | No `setUp`; 3 classes → `@pytest.mark.skipif` per function; module-level helpers stay |

**No public API changes.** No new dependencies. Tests must pass identically
after conversion (`pytest -r a -v -n 3`).
