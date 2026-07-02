## Context

`lsst.images.tests` exposes a shared test-helper library (`_checks.py`, `_roundtrip.py`)
used by every test file in the package. Every public function in `_checks.py` and the
`RoundtripBase` context manager in `_roundtrip.py` accept a `tc: unittest.TestCase` as
their first positional argument and route all assertions through it (e.g.
`tc.assertEqual(a, b)`). This was the natural pattern when the test files were written as
pure `unittest.TestCase` subclasses, but pytest runs the suite and natively understands
plain `AssertionError`; the `tc` coupling is now pure boilerplate. The `unittest` module is
not otherwise needed in these helper files.

471 tests collect and pass cleanly under pytest.

## Goals / Non-Goals

**Goals:**
- Remove `tc: unittest.TestCase` from all 27 public function signatures in `_checks.py`
  and from `RoundtripBase.__init__` in `_roundtrip.py`
- Convert all `tc.assert*` / `self._tc.assert*` calls to bare `assert` (or
  `pytest.raises` / `re.fullmatch` for the two special cases)
- Replace `raise unittest.SkipTest(...)` with `pytest.skip(...)` throughout `_roundtrip.py`
- Remove `import unittest` from both helper files
- Remove the `tc` entry from every NumPy-style docstring in both files
- Update the ~213 call sites in `tests/test_*.py` that pass `self` as the first argument

**Non-Goals:**
- Converting `unittest.TestCase` classes in `tests/test_*.py` to bare pytest functions
- Converting `setUp` / `tearDown` / `setUpClass` to pytest fixtures
- Converting `self.subTest(...)` blocks to `@pytest.mark.parametrize`
- Changing `self.assert*` calls that appear directly in test methods (only the helpers
  are in scope)
- Any changes to production (non-test) source code

## Decisions

### 1. Bare `assert` over a pytest-specific assertion library

The `tc.assert*` family maps cleanly to plain Python `assert` statements. pytest rewrites
`assert` expressions at collection time to produce rich failure messages, so no information
is lost. Using bare `assert` keeps the helpers free of any test-framework dependency except
for the two cases below.

### 2. `pytest.raises` for the single `tc.assertRaises` usage

`_checks.py` line 821 (`compare_psf_to_legacy`) uses `with tc.assertRaises(InvalidPsfError):`
as a context manager. The direct replacement is `with pytest.raises(InvalidPsfError):`.
This is the only place `pytest` itself is needed in `_checks.py`.

### 3. `re.fullmatch` for the single `tc.assertRegex` usage

`check_archive_tree_class_invariants` calls `tc.assertRegex(cls.SCHEMA_VERSION, r"^\d+\.\d+\.\d+$")`.
The pattern is fully anchored, so `assert re.fullmatch(r"^\d+\.\d+\.\d+$", cls.SCHEMA_VERSION)`
is semantically equivalent and more explicit. This requires `import re` in `_checks.py`.

### 4. `assert_equal_allow_nan` rewritten as an explicit conditional

The current implementation catches `AssertionError` from `tc.assertEqual`. Without `tc`,
the cleanest equivalent is a direct conditional:

```python
def assert_equal_allow_nan(a: float, b: float) -> None:
    if not (a == b or (math.isnan(a) and math.isnan(b))):
        raise AssertionError(f"{a!r} != {b!r}")
```

This avoids importing pytest for this single function and is clearer about the NaN-equality
semantics.

### 5. `pytest.skip` for `unittest.SkipTest` in `_roundtrip.py`

`TemporaryButler.__enter__` and `RoundtripBase.get` raise `unittest.SkipTest` in three
places. pytest treats `unittest.SkipTest` as a skip signal, so this is not a correctness
issue today, but once `import unittest` is removed the change is necessary. The replacement
is `pytest.skip(msg)`, which raises `pytest.skip.Exception` — the canonical pytest skip
mechanism.

### 6. `RoundtripBase` drops `tc` but keeps `_tc` attribute name as internal dead code during transition

No — the attribute `self._tc` is simply removed entirely. The six `self._tc.assert*` calls
are each replaced in-place with the appropriate bare `assert`.

### 7. Call-site update strategy: mechanical search-and-replace per file

Each of the 40 test files is updated individually. The pattern is uniform:
- `assert_foo(self, ...)` → `assert_foo(...)`
- `RoundtripFits(self, ...)` → `RoundtripFits(...)`

No logic changes; only the leading `self` argument is removed. Files that don't import or
call any of the affected helpers need no changes.

## Risks / Trade-offs

- **Failure message quality**: `tc.assertEqual(a, b)` generates a diff-style message
  showing both values. pytest's assert rewriting also generates rich messages for
  `assert a == b`, so no regression is expected. For `assert_equal_allow_nan`, the
  explicit `raise AssertionError(f"{a!r} != {b!r}")` preserves a useful message.
  → No mitigation needed beyond running the full suite after the change.

- **`TypeError` instead of `SkipTest` if a caller accidentally passes `self`**: After the
  change, `assert_images_equal(self, a, b)` will interpret `self` as `a`, leading to a
  confusing `TypeError` or wrong comparison rather than a clear error. This is a
  short-lived risk that disappears once all call sites are updated atomically.
  → Mitigation: update helpers and all call sites in a single commit; run tests before
  committing.

- **Downstream breakage**: Any downstream package that imports `lsst.images.tests` helpers
  and passes a `tc` argument will break. The proposal notes this is not a stable API, so
  this is acceptable.
  → No mitigation required per project decision.

## Migration Plan

1. Update `_checks.py`: drop `tc` params, rewrite assertions, update docstrings,
   swap imports.
2. Update `_roundtrip.py`: drop `tc` param from `RoundtripBase`, rewrite assertions,
   swap `unittest.SkipTest` → `pytest.skip`, swap imports.
3. Update all 40 `tests/test_*.py` files: drop leading `self` from each affected call site.
4. Run `pytest -r a -v -n 3` — all 471 tests must pass.
5. Run `ruff check --fix python/ tests/` and `mypy python/` — no new errors.

All steps happen in a single branch/commit set; no staged rollout is needed since this is
a pure refactor with no behavior change.

## Open Questions

None — all technical decisions resolved during exploration.
