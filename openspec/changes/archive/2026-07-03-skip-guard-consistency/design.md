## Context

The `pytest-test-style` spec already mandates that session-scoped fixtures call
`pytest.skip()` internally when an optional dependency is absent, so that every test
consuming the fixture is automatically skipped without needing its own decorator.  Three
test files violate this principle in different ways:

- `test_fields.py`: `make_legacy_fields()` is a module-level factory that calls afw
  constructors directly.  If `HAVE_LEGACY` is `False` the names are undefined and the call
  raises `NameError`.  Five tests guard against this with `@skip_no_legacy`, but the skip
  belongs in the factory.
- `test_polygon.py`: one test carries a raw `@pytest.mark.skipif(not have_legacy, ...)`
  expression instead of a named constant, inconsistent with every other file.
- `test_visit_image.py`: nine tests carry `@skip_no_test_data` even though they
  exclusively receive external-data objects via `legacy_dataset_context`,
  `legacy_visit_image`, or `legacy_nJy_dataset_context`—all of which already suppress
  collection when `TESTDATA_IMAGES_DIR` is absent (via `params=[] if dir is None else
  [...]`).  Additionally, two tests use an inline `@pytest.mark.skipif(not HAVE_H5PY, ...)`
  instead of the named `skip_no_h5py` constant defined in the same file.

## Goals / Non-Goals

**Goals:**

- Factory functions that return legacy objects embed their own `pytest.skip()` so
  consumers need no skip decorator.
- All per-function skip decorators that guard a condition already handled by an injected
  fixture or factory are removed.
- Inline `skipif` expressions are replaced with named constants to match the rest of the
  codebase.
- The existing correct pattern (`@skip_no_h5py` on NDF tests, `@skip_no_legacy` on tests
  that call legacy APIs directly) is left untouched.

**Non-Goals:**

- Converting `@skip_no_legacy` decorators that are genuinely needed (tests that call
  legacy APIs inline with no intervening factory or fixture).
- Converting butler-skip decorators in `test_formatter_cache.py`; `@skip_no_butler` on
  test functions is correct because those tests call the butler API directly.
- Touching any file that is already fully compliant.
- Changing any non-skip behaviour, test logic, or assertions.

## Decisions

### D1: Add `pytest.skip()` to `make_legacy_fields()`, not a wrapper fixture

`make_legacy_fields()` is called from five test functions each of which needs a fresh
dict (the factory pattern, not the fixture pattern).  Converting it to a fixture would
require adding it as a parameter to every caller and changing call-sites from
`make_legacy_fields()` to `legacy_fields`.  Instead, the factory gains a single guard at
its top:

```python
def make_legacy_fields() -> dict:
    if not HAVE_LEGACY:
        pytest.skip("This test requires lsst.afw.math to be importable.")
    ...
```

This preserves the factory call pattern while making the skip propagate automatically.

*Alternative considered*: convert to a session fixture.  Rejected because (a) the dict
contains mutable numpy arrays that tests may modify, and session scope would cause
cross-test contamination, and (b) it would change five call-sites unnecessarily.

### D2: Replace `skip_no_test_data` decorators by relying on empty-params fixture suppression

`legacy_dataset_context` and `legacy_nJy_dataset_context` are parametrized with
`params=[] if EXTERNAL_DATA_DIR is None else [...]`.  When params is empty, pytest
collects zero instances of every test that injects that fixture — the tests simply do not
exist in the run.  `legacy_visit_image` is a dependent of `legacy_dataset_context` and
inherits the same behaviour.  The nine `@skip_no_test_data` decorators therefore provide
no additional protection and should be removed.

The `assert EXTERNAL_DATA_DIR is not None, "Guaranteed by decorator."` line in
`test_convert_unit` was relying on the decorator as a runtime invariant.  With the
decorator gone the assert is no longer guaranteed; it should be removed along with its
comment.  (`legacy_nJy_dataset_context` already ensures the test cannot be reached when
the directory is absent.)

*Alternative considered*: convert to `pytest.skip()` inside the fixtures.  The empty-
params approach is already in place and is idiomatic pytest; switching to a different skip
mechanism would be gratuitous churn.

### D3: Named constant for `have_legacy` in `test_polygon.py`

The inline `@pytest.mark.skipif(not have_legacy, reason="lsst legacy packages could not
be imported.")` is replaced with a module-level `skip_no_legacy` constant following the
naming convention used by every other file.  No behavioural change.

### D4: Named constant for h5py in `test_visit_image.py`

Two tests (lines 374 and 382) use `@pytest.mark.skipif(not HAVE_H5PY, reason="h5py is
not installed")` inline.  `test_visit_image.py` already imports `HAVE_H5PY`; it needs
only a `skip_no_h5py = pytest.mark.skipif(...)` line and the two decorators updated.

## Risks / Trade-offs

- **`make_legacy_fields()` is now a skip-raising function, not a pure factory.**  Any
  future caller outside of tests should not call it without afw present.  Risk is low: the
  function lives in a test file and its name makes its purpose clear.  → No mitigation
  needed beyond the docstring update.

- **Removing `@skip_no_test_data` makes the skip mechanism implicit.**  A reader of
  `test_convert_unit` must know that `legacy_nJy_dataset_context` uses empty params to
  understand why no data-dir guard is needed.  → The fixture docstring (already present)
  explains this; no additional documentation needed.
