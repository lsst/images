## Context

Three prior batches have converted the majority of `tests/` to pytest
free-function style, establishing stable patterns for every conversion
scenario.  Six files remain.  The `openspec/specs/pytest-test-style/spec.md`
spec codifies all required patterns; no new patterns need to be invented for
this batch.

Current state of the six files:

| File | Classes | setUp? | Key complexity |
|---|---|---|---|
| `test_ndf_input_archive.py` | 4 | No | All guarded `@skipUnless(HAVE_H5PY)` |
| `test_ndf_output_archive.py` | 6 | No | All guarded `@skipUnless(HAVE_H5PY)` |
| `test_diagram.py` | 6 | 2 trivial | Large (625 lines, 47 tests) |
| `test_cameras.py` | 2 | `setUpClass` | Dual skip (DATA_DIR + HAVE_AFW) |
| `test_from_hdu_list.py` | 5 | 1 class | `setUp` builds reusable HDU list |
| `test_cli.py` | 8 | 3 classes | `addCleanup`, external data, subTest loop |

## Goals / Non-Goals

**Goals:**
- Convert all six files to pytest free-function style satisfying the
  `pytest-test-style` spec.
- Remove all `unittest.TestCase` subclasses, `import unittest`, and
  `if __name__ == "__main__": unittest.main()` blocks from the six files.
- All tests pass (or skip) after conversion; no regressions.
- Zero ruff lint errors after conversion.

**Non-Goals:**
- Converting `test_cell_coadd.py` (deferred to a separate change).
- Changing test coverage, test logic, or assert semantics.
- Modifying `python/` production code.

## Decisions

### D1: NDF files — `skip_no_h5py` constant, no class-level grouping

Both NDF files have all classes guarded with `@unittest.skipUnless(HAVE_H5PY,
...)`.  **Decision**: define `skip_no_h5py = pytest.mark.skipif(not HAVE_H5PY,
reason="h5py is not installed")` at module level, apply it to each test
function, and remove the class wrappers.  No `pytestmark` class grouping is
needed because no classes survive.

Alternative considered: keep a thin class with `pytestmark` for grouping.
Rejected: unnecessary indirection; flat free functions are simpler.

### D2: `test_diagram.py` — inline cheap objects, no session fixtures

`EmitterTestCase.setUp` creates `self.graph = build_graph(Parent)` — pure
in-memory construction with no I/O.  `DiagramCliTestCase.setUp` creates
`self.runner = CliRunner()` and computes a `fixture` path string.

**Decision**: construct both objects inline at the start of each test function,
or extract a `make_graph()` / `make_runner()` module-level factory if more than
two tests share the same construction call.  No `@pytest.fixture` is needed
because objects are cheap (spec: "Cheap-to-construct objects are provided by
module-level factories").

### D3: `test_cameras.py` — dual skip in session fixture body

`CamerasTestCase` carries two class-level decorators: `@skipUnless(DATA_DIR is
not None, ...)` and `@skipUnless(HAVE_AFW, ...)`.  **Decision**: create a
single session fixture `legacy_camera_data` that:
1. Calls `pytest.skip("TESTDATA_IMAGES_DIR is not in the environment.")` if
   `DATA_DIR is None`.
2. Calls `pytest.skip("lsst.afw could not be imported.")` if `not HAVE_AFW`.
3. Returns a `dict` with `legacy_camera`, `legacy_detector` keys.

Downstream test functions inject `legacy_camera_data` and need no skip
decorators.

### D4: `test_from_hdu_list.py` — setUp with reusable HDU list becomes session fixture

`FromHduListTestCase.setUp` builds a `MaskedImage` HDU list used by most tests
in the class.  The construction involves creating arrays and headers (no disk
I/O), so it is cheap.  **Decision**: use a module-level factory
`make_hdu_list()` rather than a session fixture, per spec preference for cheap
objects.  The four other classes (no `setUp`) convert mechanically.

### D5: `test_cli.py` — `addCleanup` → `tmp_path`, external data → session fixture

Three classes use `tempfile.TemporaryDirectory()` + `self.addCleanup(tmp.cleanup)`.
**Decision**: replace with pytest's built-in `tmp_path` fixture injected by
parameter name into each test function.

`ConvertVisitImageTestCase` and `ConvertCellCoaddTestCase` gate tests on both
`EXTERNAL_DATA_DIR` and an optional import (`lsst.afw.image` /
`lsst.cell_coadds`).  **Decision**: the `@unittest.skipUnless(EXTERNAL_DATA_DIR
is not None, ...)` guard becomes a session fixture `external_data_dir` that
calls `pytest.skip(...)` when `EXTERNAL_DATA_DIR` is `None`.  The inline
`self.skipTest("afw not available.")` / `self.skipTest("cell_coadds not
available.")` become `pytest.importorskip(...)` called at the top of the test
body, since no object derived from the import is passed via a fixture.

`ConvertDetectTestCase._make()` helper becomes a module-level helper function
`_make_detect_fixture(dataset_type)`.  The one `@skipUnless` method in that
class injects `external_data_dir`.

`CliRegistrationTestCase.test_short_help_alias` uses a `subTest` loop over 10
arg lists — **Decision**: `@pytest.mark.parametrize("args", [...])`.

### D6: No spec delta needed

All patterns applied in this batch are already in `pytest-test-style/spec.md`.
No `specs/` delta files are required.

## Risks / Trade-offs

- **Large file size of `test_diagram.py`** — 625 lines, 47 tests across 6
  classes.  Risk of transcription error.  Mitigation: convert class by class,
  run `ruff` and `pytest` after each class.

- **`test_cli.py` `ConvertDetectTestCase._make()` uses `tempfile.mkdtemp()`
  without cleanup** — the original code has the same leak.  Mitigation: replace
  with `tmp_path` consistently; pass `tmp_path` to `_make_detect_fixture` or
  use `tmp_path` directly in the test body.

- **`test_cli.py` `mock.patch` and `mock` import** — `from unittest import mock`
  must be retained (it is `unittest.mock`, not `unittest.TestCase`); only the
  `import unittest` and class usage are removed.
