## Why

The shared test helpers in `lsst.images.tests` (`_checks.py` and `_roundtrip.py`) thread a
`tc: unittest.TestCase` parameter through every function and class so they can call
`tc.assertEqual(...)`, `tc.assertTrue(...)`, etc. This coupling to the `unittest` API adds
noise to every call site, makes function signatures misleading (the `tc` argument is an
implementation detail, not meaningful domain data), and blocks a gradual migration toward
idiomatic pytest style. Since pytest runs all tests and re-raises plain `AssertionError`
natively, bare `assert` statements provide identical behavior with less boilerplate.

## What Changes

- Remove the `tc: unittest.TestCase` parameter from all 27 public functions in `_checks.py`
  and from `RoundtripBase.__init__` in `_roundtrip.py` (**BREAKING** for any downstream
  caller that passes `self` or another `TestCase` as the first argument)
- Replace all `tc.assert*` calls in `_checks.py` (115 call sites) with bare `assert`
  statements or `pytest.raises` / `re.fullmatch` as appropriate
- Replace all `self._tc.assert*` calls in `_roundtrip.py` (6 call sites) with bare `assert`
- Replace `raise unittest.SkipTest(...)` (3 call sites in `_roundtrip.py`) with `pytest.skip(...)`
- Remove `import unittest` from both helper files; add `import pytest` and `import re`
  where needed
- Update the `tc` parameter entry in every NumPy-style docstring across both files
- Update all 213 call sites in `tests/test_*.py` that pass `self` as the first argument to
  a helper function or `Roundtrip*` constructor

## Capabilities

### New Capabilities

None — this is a pure refactor with no new user-visible behavior.

### Modified Capabilities

- `test-helper-api`: The public API of `lsst.images.tests` changes: all assertion helpers
  and `RoundtripBase` subclasses drop their leading `tc` / `self` argument.

## Impact

- **`python/lsst/images/tests/_checks.py`**: all 27 public functions lose their `tc`
  parameter; `import unittest` removed; `import pytest` and `import re` added
- **`python/lsst/images/tests/_roundtrip.py`**: `RoundtripBase.__init__` loses `tc`;
  `import unittest` removed; `import pytest` added; `unittest.SkipTest` → `pytest.skip`
- **`tests/test_*.py`** (40 files): ~156 `assert_*/compare_*/check_*` call sites drop
  their leading `self` argument; ~57 `Roundtrip*(self, ...)` constructors drop `self`
- No changes to production (non-test) code
- No changes to `unittest.TestCase` class structure or `setUp`/`subTest` usage in tests —
  those are out of scope
