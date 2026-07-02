## Context

`RoundtripBase.get(component=...)` calls `pytest.skip()` when `self.butler is None`
(i.e. when `lsst.daf.butler` is not installed).  This is documented in both the
method docstring and the class docstring, which explicitly says component-read calls
"generally means these tests should be nested within a `subTest` context".

The original `unittest.TestCase` tests used `with self.subTest():` to contain these
skips — a failed or skipped subTest allowed the outer test to continue.  During the
pytest conversions, these wrappers were removed on the grounds that the inner blocks
were "sequential phases, not independent iterations" (D5 in
`convert-visit-image-test-to-pytest`).  That reasoning was correct for the
*parametrize* question, but missed the skip-containment purpose: without any boundary,
`pytest.skip()` propagates up the call stack and exits the entire test function.

Two functions are affected:

| File | Function | First unsafe `get()` | Butler-free assertions after |
|---|---|---|---|
| `test_visit_image.py` | `test_read_write` | line 428 (`"masked_image"`) | ~65 lines (492–508) |
| `test_cell_coadd.py` | `test_roundtrip` (unconverted) | line 130 (`"psf"`) | lines 174–187 |

## Goals / Non-Goals

**Goals:**
- Ensure `assert_visit_images_equal`, opaque metadata checks, and background
  spot-checks in `test_read_write` always execute regardless of butler availability.
- Ensure `assert_cell_coadds_equal` and `compare_cell_coadd_to_legacy` in
  `test_roundtrip` always execute regardless of butler availability.
- Complete the pytest migration of `test_cell_coadd.py` (the last remaining
  `unittest.TestCase` file in the test suite).
- Add a spec requirement that prevents this class of defect from recurring.

**Non-Goals:**
- Changing any assertion logic or test coverage beyond what the split requires.
- Converting any file other than `test_visit_image.py` and `test_cell_coadd.py`.
- Modifying `python/` production code.

## Decisions

### D1: Split each affected test into two functions, not three

Each affected test has two logical regions:
1. Component-read-free: assertions on `roundtrip.result`, FITS headers, etc.
   that always execute regardless of butler availability
2. Component-read-dependent: `get(component=...)` calls that fire `pytest.skip()`
   when butler is absent

**Decision**: split into exactly two functions per affected test.  **Both**
functions pass `storage_class=...` to `RoundtripX` — this ensures that when
butler *is* available, both functions exercise the butler adapter path (put/get
through `GenericFormatter`), which is additional coverage worth keeping.  The
split criterion is solely whether the function calls `get(component=...)`:
functions that do not call `get(component=...)` are never at risk of an
unwanted skip; functions that do will cleanly skip their entire body when
butler is absent, because the first `get(component=...)` fires `pytest.skip()`.

Alternative considered: remove `storage_class` from the butler-free half so
`_run_without_butler()` is always used.  Rejected: this loses the butler-path
exercise (put → get cycle, metadata round-trip) for the half of the test that
checks equality and headers — coverage we want when butler is present.

Alternative considered: keep one function and wrap component calls in a
`try/except Skipped`.  Rejected: fragile, non-idiomatic, and the skip exception
type is a pytest internal.

Alternative considered: a session fixture that yields a pre-entered
`RoundtripFits` object, shared across the two functions.  Rejected: the
`RoundtripFits` context manager manages a temp file whose lifetime must be tied
to the `with` block; sharing it across two test functions would require a
session-scoped fixture with `yield`, which adds complexity with no benefit —
both functions can write their own temp file independently.

### D2: The split criterion is `get(component=...)`, not `storage_class`

`RoundtripBase.get()` skips on `self.butler is None`, which is true when either
`HAVE_BUTLER` is False or `storage_class` was not supplied.  Passing
`storage_class` is necessary but not sufficient to prevent a skip — the butler
package must also be importable.  The safe invariant is therefore: a test
function that calls `get(component=...)` should contain *only* component-read
assertions, so that when the skip fires it wastes nothing important.

### D3: `alternates` dict is not passed across the split in `test_roundtrip`

In the original `test_roundtrip`, `alternates` is populated inside the `subTest`
block and then passed to `compare_cell_coadd_to_legacy(..., alternates=alternates)`
outside it.  In the split:
- `test_roundtrip` (no component reads) calls `compare_cell_coadd_to_legacy` with
  `alternates={}` (the default, which means no alternate-component comparison is
  done) — this is the existing behaviour when butler is absent anyway.
- `test_roundtrip_components` populates `alternates` and passes the full dict
  inside the `with` block before the context exits.

This is not a regression: the butler-absent path already produced `alternates={}`
because the `subTest` was skipped.

### D4: `CellCoaddTestCase` → pytest free-functions follows established batch patterns

The full conversion of `test_cell_coadd.py` follows the patterns established and
codified in batches 1–4:

- `setUpClass` (expensive: reads files, requires `lsst.cell_coadds`) →
  `cell_coadd_data` session fixture that calls `pytest.skip()` for missing
  `DATA_DIR` or missing `lsst.cell_coadds` import.
- `make_psf_points(self, bbox)` → module-level `make_psf_points(cell_coadd, bbox)`
  with internal `rng = np.random.default_rng(44)` (per spec: local rng in helpers).
- `setUp` (cheap: creates rng, calls `make_psf_points`) → tests call
  `make_psf_points` inline; no `setUp` fixture needed.
- `@unittest.skipUnless(HAVE_H5PY)` → `skip_no_h5py` module-level constant.
- `test_to_legacy_exposure` calls `cell_coadd.to_legacy_exposure()` which does
  `from lsst.afw.image import ...` inline → needs `@skip_no_afw` or
  `pytest.importorskip("lsst.afw.image")` at top of test body, in addition to the
  skip already embedded in the `cell_coadd_data` fixture.
- `with self.subTest(extname=extname):` loop in `test_fits_compression` → plain
  `for` loop with `extname` retained as named local (loop count is runtime-dependent,
  cannot use `@pytest.mark.parametrize`).
- `if __name__ == "__main__": unittest.main()` → removed.

### D5: Spec delta is an ADDED requirement, not a MODIFIED one

The existing `pytest-test-style` spec has a requirement about `subTest` inside
`RoundtripX` blocks, but it focuses on afw-dependent inner blocks and pure skips —
it does not cover the butler-absence / `get(component=...)` scenario.  The new
requirement is genuinely additive; the existing requirement text need not change.

### D6: `_roundtrip.py` docstring updates are docstring-only

`RoundtripBase` class docstring, `RoundtripBase.get` docstring, and
`TemporaryButler` class docstring all reference `unittest.SkipTest`,
`~unittest.TestCase.subTest`, and `self.subTest`.  These references are stale
now that the codebase uses `pytest.skip()` directly and no `subTest` wrapper
is needed.  The fix is purely docstring — no behavioural change to `_roundtrip.py`.

## Risks / Trade-offs

- **`test_read_write_components` and `test_roundtrip_components` are no-ops without
  butler** → Accepted: they were already effectively no-ops (the original test skipped
  past them).  The split makes this explicit and restores the butler-free assertions.

- **Two temp files written per original test** → Each function creates its own
  `RoundtripFits` temp file.  This is a minor I/O cost with no correctness impact;
  both files are cleaned up by the context manager.

- **`test_cell_coadd.py` conversion size** → The file is small (279 lines, 1 class,
  8 tests).  Risk of transcription error is low compared to prior batch files.
