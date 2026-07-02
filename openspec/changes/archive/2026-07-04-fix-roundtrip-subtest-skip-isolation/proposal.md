## Why

Two pytest-converted test functions — `test_read_write` in `tests/test_visit_image.py`
and `test_roundtrip` in `tests/test_cell_coadd.py` (the latter not yet converted) —
contain `RoundtripBase.get(component=...)` calls inside a `with RoundtripX(...):`
block without any isolation boundary.  When `lsst.daf.butler` is absent,
`get(component=...)` calls `pytest.skip()`, which propagates out of the `with`
block and skips the entire test function — including substantial butler-free
assertions that would otherwise always run.  These assertions are silently
dead in non-butler CI environments.

The root cause is that the original `unittest.TestCase` tests wrapped these calls
in `with self.subTest():` blocks, which *did* contain the skip.  The pytest
conversion for `test_visit_image.py` (change `convert-visit-image-test-to-pytest`,
decision D5) removed those wrappers on the grounds that the subtests were "sequential
phases, not independent iterations", which was correct reasoning for the
parametrize question but missed the load-bearing skip-containment role the wrappers
served.

## What Changes

- `tests/test_visit_image.py` — split `test_read_write` into two functions, both
  passing `storage_class="VisitImage"`:
  - `test_read_write`: no `get(component=...)` calls; exercises FITS header checks,
    bbox-only subimage read, `assert_visit_images_equal`, opaque metadata, and
    background spot-check — all of which always run regardless of butler availability
  - `test_read_write_components`: all component `get()` calls; when butler is absent,
    the first `get(component=...)` cleanly skips the whole function
- `tests/test_cell_coadd.py` — convert `CellCoaddTestCase` to pytest free-function
  style (the last remaining `unittest.TestCase` in the test suite); apply the same
  split pattern to `test_roundtrip`, both passing `storage_class="CellCoadd"`:
  - `test_roundtrip`: no `get(component=...)` calls; butler-free assertions
  - `test_roundtrip_components`: all component `get()` calls
- `python/lsst/images/tests/_roundtrip.py` — update docstrings on `RoundtripBase`,
  `RoundtripBase.get`, and `TemporaryButler` to remove stale references to
  `unittest.SkipTest`, `unittest.TestCase.subTest`, and `self.subTest`

## Capabilities

### New Capabilities

None — this change fixes existing behaviour and completes an existing migration.

### Modified Capabilities

- `pytest-test-style`: Add an explicit requirement that `get(component=...)` calls
  SHALL NOT appear in the same test function as assertions that must always execute
  regardless of butler availability; the two concerns SHALL live in separate test
  functions.

## Impact

- `tests/test_visit_image.py` — `test_read_write` split into two functions
- `tests/test_cell_coadd.py` — full `unittest.TestCase` → pytest free-function
  conversion; `test_roundtrip` split into two functions
- `python/lsst/images/tests/_roundtrip.py` — docstring-only updates
- `openspec/specs/pytest-test-style/spec.md` — new requirement added
- No changes to test logic, public API, or production code
