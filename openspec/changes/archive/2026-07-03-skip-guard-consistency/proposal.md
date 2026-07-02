## Why

The existing `pytest-test-style` spec already requires that session-scoped fixtures embed
their skip logic internally, eliminating `@pytest.mark.skipif` decorators from test
functions that receive legacy objects via those fixtures.  Several pytest-style test files
still carry per-function skip decorators that are redundant with—or missing from—the
factories and fixtures the tests depend on, violating both the existing spec and the
principle that skip guards live at the point of dependency, not at every consumer.

## What Changes

- `make_legacy_fields()` in `tests/test_fields.py` gains an internal `pytest.skip()` call
  when `HAVE_LEGACY` is false, so callers no longer need `@skip_no_legacy` decorators.
- The five `@skip_no_legacy` decorators on tests in `test_fields.py` that call
  `make_legacy_fields()` are removed.
- `tests/test_polygon.py` gains a named `skip_no_legacy` module-level constant and uses it
  instead of the inline `@pytest.mark.skipif(not have_legacy, ...)` expression.
- The nine `@skip_no_test_data` decorators in `tests/test_visit_image.py` are removed;
  those tests already receive `TESTDATA_IMAGES_DIR`-dependent objects exclusively via
  `legacy_dataset_context`, `legacy_visit_image`, or `legacy_nJy_dataset_context` fixtures,
  which suppress test collection entirely when the directory is absent.
- The one `assert EXTERNAL_DATA_DIR is not None, "Guaranteed by decorator."` guard in
  `test_convert_unit` is removed along with its comment.
- Two inline `@pytest.mark.skipif(not HAVE_H5PY, ...)` decorators in
  `tests/test_visit_image.py` are replaced with the named `skip_no_h5py` constant.

## Capabilities

### New Capabilities

*(none)*

### Modified Capabilities

- `pytest-test-style`: extend the rule that fixtures embed skip logic to cover
  module-level factory functions that return legacy objects; clarify that skip decorators
  on test functions are eliminated whenever the test obtains all legacy-dependent objects
  via a fixture or factory that already embeds the skip.

## Impact

- `tests/test_fields.py` — `make_legacy_fields()`, five test functions
- `tests/test_polygon.py` — module-level skip constant, one test function
- `tests/test_visit_image.py` — `skip_no_test_data` usage (nine decorators), one inline
  assertion, two inline h5py decorators
- `openspec/specs/pytest-test-style/spec.md` — new requirements (added at archive time)
