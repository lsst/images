## Why

`tests/test_visit_image.py` (`VisitImageTestCase`) still uses `unittest.TestCase`
with a monolithic `setUpClass` that constructs all shared state in one block,
making it harder to understand which objects a given test actually depends on.
Converting it to pytest free-functions with scoped fixtures aligns it with the
project's `pytest-test-style` spec and makes fixture dependencies explicit.

## What Changes

- `tests/test_visit_image.py`: The `VisitImageTestCase` class is replaced by
  pytest free-functions and session-scoped fixtures.  The legacy test classes
  (`VisitImageLegacyTestMixin`, `VisitImageLegacyTestCase`,
  `PreliminaryVisitImageLegacyTestCase`, `DifferenceImageLegacyTestCase`) are
  left unchanged for now.
- `unittest` import and `if __name__ == "__main__": unittest.main()` are removed
  from the converted section; the `unittest` import is retained only while the
  legacy classes remain.
- A new `visit_image_components` session-scoped fixture (a `dict`) supplies the
  ten shared objects that the converted tests need to construct `VisitImage`
  instances.
- Two further session-scoped fixtures — `visit_image` and
  `simplest_visit_image` — build the two canonical `VisitImage` objects used
  across most tests.
- The `_make_sum_background_visit_image` helper becomes a module-level function
  accepting the components dict; it uses its own local `rng` rather than a
  shared one.
- The `_check_sum_background_round_trip` helper becomes a module-level
  function.
- `self.subTest()` blocks in `test_read_write` are replaced with plain
  assertions (no parametrize needed — they are sequential phases of one
  round-trip, not independent iterations).
- `@unittest.skipUnless(HAVE_H5PY, …)` decorators are replaced with
  `@pytest.mark.skipif(not HAVE_H5PY, …)`.

## Capabilities

### New Capabilities

- `visit-image-test-fixtures`: Session-scoped pytest fixtures for
  `VisitImageTestCase` — `visit_image_components` (dict), `visit_image`, and
  `simplest_visit_image` — with explicit, documented fixture dependencies.

### Modified Capabilities

- `pytest-test-style`: Add a requirement that session-scoped fixtures are used
  when shared state is expensive or logically shared across all tests in a
  file, with fixture dependencies declared explicitly via function parameters.

## Impact

- `tests/test_visit_image.py`: modified (partial — `VisitImageTestCase` only)
- `openspec/specs/pytest-test-style/spec.md`: one new requirement added
- No changes to `python/lsst/images/` source or public API
- No new external dependencies
