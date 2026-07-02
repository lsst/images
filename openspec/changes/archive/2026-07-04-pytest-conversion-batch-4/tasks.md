## 1. Convert test_ndf_input_archive.py

- [x] 1.1 Add `skip_no_h5py = pytest.mark.skipif(not HAVE_H5PY, reason="h5py is not installed")` at module level; remove `import unittest`
- [x] 1.2 Convert `NdfInputArchiveOpenTestCase` (2 tests) — strip class, apply `@skip_no_h5py`, replace `self.assert*` / `self.assertRaises` with `assert` / `pytest.raises`
- [x] 1.3 Convert `NdfInputArchiveDataTestCase` (7 tests) — strip class, apply `@skip_no_h5py`, replace all `self.assert*` with plain `assert`
- [x] 1.4 Convert `NdfInputArchiveOpaqueMetadataTestCase` (2 tests) — strip class, apply `@skip_no_h5py`, replace `self.assert*` with plain `assert`
- [x] 1.5 Convert `NdfReadFunctionTestCase` (12 tests) — strip class, apply `@skip_no_h5py`, replace `self.assert*` / `self.assertRaises` with `assert` / `pytest.raises`
- [x] 1.6 Remove `if __name__ == "__main__": unittest.main()` block; add imperative docstrings to all test functions and any fixtures

## 2. Convert test_ndf_output_archive.py

- [x] 2.1 Add `skip_no_h5py = pytest.mark.skipif(not HAVE_H5PY, reason="h5py is not installed")` at module level; remove `import unittest`
- [x] 2.2 Convert `NdfOutputArchiveBasicsTestCase` (2 tests) — strip class, apply `@skip_no_h5py`, replace `self.assert*` with plain `assert`
- [x] 2.3 Convert `NdfOutputArchiveAddArrayTestCase` (8 tests) — strip class, apply `@skip_no_h5py`, replace `self.assert*` / `self.assertRaisesRegex` with `assert` / `pytest.raises(..., match=...)`
- [x] 2.4 Convert `NdfOutputArchivePointerTestCase` (5 tests) — strip class, apply `@skip_no_h5py`, replace `self.assert*` with plain `assert`
- [x] 2.5 Convert `NdfOutputArchiveAddTableTestCase` — strip class, apply `@skip_no_h5py`, replace `self.assert*` with plain `assert`
- [x] 2.6 Convert `NdfWriteWcsTestCase` — strip class, apply `@skip_no_h5py`, replace `self.assert*` with plain `assert`
- [x] 2.7 Convert `NdfWriteFunctionTestCase` — strip class, apply `@skip_no_h5py`, replace `self.assert*` / `self.assertRaises` with `assert` / `pytest.raises`
- [x] 2.8 Remove `if __name__ == "__main__": unittest.main()` block; add imperative docstrings to all test functions

## 3. Convert test_diagram.py

- [x] 3.1 Remove `import unittest`; remove `if __name__ == "__main__": unittest.main()`
- [x] 3.2 Convert `BuildGraphTestCase` (no setUp, 10 tests) — strip class, replace `self.assert*` with plain `assert`; add imperative docstrings
- [x] 3.3 Convert `RealModelTestCase` (no setUp, 3 tests) — strip class, replace `self.assert*` with plain `assert`; add imperative docstrings
- [x] 3.4 Convert `PolicyTestCase` (no setUp, 7 tests) — strip class, replace `self.assert*` with plain `assert`; add imperative docstrings
- [x] 3.5 Convert `InstanceGraphTestCase` (no setUp, 6 tests) — strip class, replace `self.assert*` with plain `assert`; add imperative docstrings
- [x] 3.6 Convert `EmitterTestCase` — `setUp` creates `self.graph = build_graph(Parent)` (cheap); extract `make_graph` module-level factory (`def make_graph() -> DiagramGraph: return build_graph(Parent)`); replace `self.graph` with `make_graph()` inline; strip class; replace `self.assert*` / `self.assertRaises` with `assert` / `pytest.raises`; add imperative docstrings
- [x] 3.7 Convert `DiagramCliTestCase` — `setUp` creates `self.runner = CliRunner()` and `self.fixture` path (cheap); both constructed inline in each test; `self.invoke(...)` helper becomes module-level `_invoke(runner, *args)` function; strip class; replace `self.assert*` / `self.assertNotEqual` with plain `assert`; add imperative docstrings

## 4. Convert test_cameras.py

- [x] 4.1 Remove `import unittest`; remove `if __name__ == "__main__": unittest.main()`
- [x] 4.2 Add `legacy_camera_data` session fixture: checks `DATA_DIR` and `HAVE_AFW` with `pytest.skip()` in its body; reads `camera.fits` and `visit_image.fits`; returns a `dict` with `legacy_camera` and `legacy_detector` keys
- [x] 4.3 Convert `CamerasTestCase` (3 tests) — strip class; inject `legacy_camera_data` fixture; replace `self.legacy_camera` / `self.legacy_detector` with `legacy_camera_data["legacy_camera"]` / `legacy_camera_data["legacy_detector"]`; replace `self.subTest(detector_id=...)` with a plain `for` loop retaining `legacy_detector_1` as the named loop variable; remove class-level `@unittest.skipUnless` decorators; add imperative docstrings
- [x] 4.4 Convert `ReadoutCornerTestCase` (3 tests) — strip class; replace `self.assert*` / `self.assertIs` with plain `assert`; add imperative docstrings

## 5. Convert test_from_hdu_list.py

- [x] 5.1 Remove `import unittest`; remove `if __name__ == "__main__": unittest.main()`
- [x] 5.2 Convert `ReadOffsetWcsTestCase` (3 tests, no setUp) — strip class; replace `self.assert*` / `self.assertIsNone` with plain `assert`; add imperative docstrings
- [x] 5.3 Convert `ReadYx0TestCase` (3 tests, no setUp) — strip class; replace `self.assert*` / `self.assertRaises` with `assert` / `pytest.raises`; add imperative docstrings
- [x] 5.4 Convert `FromHduListTestCase` — `setUp` constructs a `MaskedImage` HDU list (no disk I/O, cheap); extract `make_hdu_list()` module-level factory returning the constructed list; strip class; inject nothing (call factory inline); replace `self.assert*` / `self.assertRaises` with `assert` / `pytest.raises`; add imperative docstrings
- [x] 5.5 Convert `LegacyMaskBranchTestCase` (no setUp) — strip class; replace `self.assert*` with plain `assert`; add imperative docstrings
- [x] 5.6 Convert `LegacyPlaneRetentionTestCase` — strip class; promote `_schema_index` static helper to module-level function `_schema_index(mask, name)`; replace `self.assert*` with plain `assert`; add imperative docstrings

## 6. Convert test_cli.py

- [x] 6.1 Remove `import unittest`; keep `from unittest import mock` (mock is unrelated to TestCase); remove `if __name__ == "__main__": unittest.main()` if present
- [x] 6.2 Convert `CliSkeletonTestCase` (2 tests, no setUp) — strip class; replace `self.assert*` with plain `assert`; add imperative docstrings
- [x] 6.3 Convert `InspectTestCase` — `setUp` builds a `TemporaryDirectory` + `Image` (cheap); replace `self.addCleanup(tmp.cleanup)` pattern with `tmp_path` fixture injected by parameter; construct `image` inline at test-function level; strip class; replace `self.assert*` with plain `assert`; add imperative docstrings
- [x] 6.4 Convert `ReformatTestCase` — same `addCleanup` → `tmp_path` replacement as above; strip class; replace `self.assert*` with plain `assert`; add imperative docstrings
- [x] 6.5 Convert `ConvertDetectTestCase` — promote `_make(dataset_type)` to module-level `_make_detect_fixture(tmp_path, dataset_type)` accepting `tmp_path` as first arg (or use `tempfile.mkdtemp()` inline); convert 3 tests without skip; convert `test_detect_visit_image_fixture` to inject an `external_data_dir` session fixture; strip class; replace `self.assert*` with plain `assert`; add imperative docstrings
- [x] 6.6 Add `external_data_dir` session fixture: calls `pytest.skip("TESTDATA_IMAGES_DIR is not set.")` when `EXTERNAL_DATA_DIR is None`; returns `EXTERNAL_DATA_DIR`
- [x] 6.7 Convert `ConvertVisitImageTestCase` (2 tests) — strip class; inject `external_data_dir` fixture; replace `self.skipTest("afw not available.")` with `pytest.importorskip("lsst.afw.image")` at top of test body; replace `self.assert*` with plain `assert`; add imperative docstrings
- [x] 6.8 Convert `ConvertCellCoaddTestCase` (2 tests) — strip class; inject `external_data_dir` fixture; replace `self.skipTest("cell_coadds not available.")` with `pytest.importorskip("lsst.cell_coadds")` at top of test body; replace `self.assert*` with plain `assert`; add imperative docstrings
- [x] 6.9 Convert `ConvertPreserveQuantizationTestCase` — `setUp` only creates a `TemporaryDirectory`; replace with `tmp_path`; promote `_make_input` to module-level `_make_cli_input(tmp_path)` helper; strip class; replace `self.assert*` with plain `assert`; add imperative docstrings
- [x] 6.10 Convert `CliRegistrationTestCase` — `test_short_help_alias` subTest loop over 10 arg lists → `@pytest.mark.parametrize("args", [...], ids=[...])` with the arg lists as params; other 4 tests have no subTest; strip class; replace `self.assert*` with plain `assert`; add imperative docstrings
- [x] 6.11 Convert `ConvertSafetyTestCase` — `setUp` → `tmp_path`; promote `_make_input` to module-level helper; strip class; replace `self.assert*` with plain `assert`; add imperative docstrings

## 7. Verification

- [x] 7.1 Run `ruff check --fix python/ tests/` and `ruff format python/ tests/` — confirm zero errors
- [x] 7.2 Run `pytest tests/ -r a -v -n 3` — confirm all tests pass or skip (no new failures)
- [x] 7.3 Confirm no `unittest.TestCase` remains in the six converted files: `grep -l "class.*TestCase\|import unittest" tests/test_ndf_input_archive.py tests/test_ndf_output_archive.py tests/test_diagram.py tests/test_cameras.py tests/test_from_hdu_list.py tests/test_cli.py` should return nothing
