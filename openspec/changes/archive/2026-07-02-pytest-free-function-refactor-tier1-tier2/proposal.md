## Why

The test suite is written entirely in `unittest.TestCase` style, which adds
boilerplate, makes fixtures harder to compose, and prevents use of pytest's
richer assertion introspection. Converting to pytest free-functions and fixtures
reduces noise and aligns the codebase with modern Python testing conventions.

## What Changes

- Nine test files are refactored from `unittest.TestCase` subclasses to
  top-level `def test_*()` functions and `@pytest.fixture` fixtures.
- `import unittest` and `if __name__ == "__main__": unittest.main()` guards are
  removed from all converted files.
- `self.assertXxx(...)` calls are replaced with plain `assert` statements.
- `setUp` methods become `@pytest.fixture` functions (or inline locals where
  only one test uses the object).
- `self.assertRaises(Exc)` becomes `pytest.raises(Exc)`.
- `@unittest.skipUnless` / `@unittest.skipIf` decorators become
  `@pytest.mark.skipif`.
- `self.maxDiff = None` assignments are dropped (pytest diffs are unlimited).
- `with self.subTest(...)` loops are unrolled into `@pytest.mark.parametrize`
  where natural, or left as plain loops with inline `assert`.

Scope is limited to two tiers of files identified during exploration:

**Tier 1 — no setUp, pure assertions (6 files):**
`test_utils.py`, `test_json_schema.py`, `test_fits_format_version.py`,
`test_fits_output_archive.py`, `test_geom.py`, `test_schema_versioning.py`

**Tier 2 — setUp creates simple objects, no tempdir (3 files):**
`test_ndf_common.py`, `test_legacy.py`, `test_polygon.py`

Files that rely on roundtrip-mixin base classes, external testdata, or the
Butler integration are explicitly excluded.

## Capabilities

### New Capabilities

- `pytest-test-style`: Convention that test files use pytest free-functions and
  fixtures rather than `unittest.TestCase`; establishes the patterns and
  mechanical translation rules applied in this change.

### Modified Capabilities

## Impact

- **Test files changed**: 9 files under `tests/`
- **No production code touched**
- **No new dependencies**: pytest is already required
- **CI**: no changes needed; pytest already discovers and runs both styles
