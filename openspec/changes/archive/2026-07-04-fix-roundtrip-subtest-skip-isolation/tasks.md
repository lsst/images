## 1. Update the pytest-test-style spec

- [x] 1.1 Merge `specs/pytest-test-style/spec.md` delta into `openspec/specs/pytest-test-style/spec.md` — append the new `### Requirement: get(component=...) calls SHALL NOT share a test function with always-run assertions` block (with its three scenarios) at the end of the file

## 2. Fix test_visit_image.py — split test_read_write

- [x] 2.1 Rename the existing `test_read_write` function to `test_read_write_components`; keep `RoundtripFits(visit_image, "VisitImage")`; remove from it: the `fits = roundtrip.inspect()` FITS header block (lines 415–421), the bbox-only subimage `roundtrip.get(bbox=subbox)` check (lines 423–425), and all post-`with` assertions (lines 492–508); update the docstring to imperative: "Verify component reads, storage-class override, and error cases round-trip correctly."
- [x] 2.2 Insert a new `test_read_write` function before `test_read_write_components` with signature `def test_read_write(visit_image_components: dict[str, Any]) -> None:`; use `RoundtripFits(visit_image, "VisitImage")` (keep `storage_class`); populate it with: the `roundtrip.inspect()` FITS header block, the bbox-only `roundtrip.get(bbox=subbox)` subimage check (no `component` arg — does not trigger skip), and all post-`with` assertions (lines 492–508: `assert_visit_images_equal`, opaque metadata checks, background spot-check); add imperative docstring: "Verify a VisitImage round-trips through FITS with correct compression, WCS, and equality."
- [x] 2.3 Verify `test_read_write_components` retains all `roundtrip.get(component=...)` calls (lines 428–490 of the original) and the `pytest.raises` error-case blocks

## 3. Convert test_cell_coadd.py to pytest free-functions

- [x] 3.1 Add imports: `import pytest`; remove `import unittest` and `from typing import Any` if `Any` is no longer used after conversion (it is used in `alternates: dict[str, Any]`, so retain it); remove `if __name__ == "__main__": unittest.main()`
- [x] 3.2 Add module-level skip constants: `skip_no_h5py = pytest.mark.skipif(not HAVE_H5PY, reason="h5py is not installed")` and `skip_no_afw = pytest.mark.skipif(not HAVE_AFW, reason="lsst.afw could not be imported.")` (add `HAVE_AFW` detection block matching existing pattern); remove the `@unittest.skipUnless` decorator from the class
- [x] 3.3 Add `cell_coadd_data` session fixture: calls `pytest.skip("TESTDATA_IMAGES_DIR is not in the environment.")` if `DATA_DIR is None`; imports and instantiates `MultipleCellCoadd` inside a `try/except ImportError` that calls `pytest.skip("lsst.cell_coadds could not be imported.")`; reads `deep_coadd_cell_predetection.fits` and `skyMap.pickle`; constructs `CellCoadd.from_legacy(...)`; returns a dict with keys `filename`, `plane_map`, `missing_cell`, `legacy_cell_coadd`, `skymap`, `cell_coadd`
- [x] 3.4 Convert `make_psf_points` to a module-level function `make_psf_points(cell_coadd: CellCoadd, bbox: Box) -> YX[np.ndarray]` with `rng = np.random.default_rng(44)` created internally; remove the `self.rng` parameter; update all call sites in converted test functions to pass `cell_coadd` explicitly
- [x] 3.5 Convert `test_from_legacy` — signature `def test_from_legacy(cell_coadd_data: dict[str, Any]) -> None:`; destructure `cell_coadd`, `missing_cell`, `legacy_cell_coadd`, `skymap`, `plane_map` from fixture dict; replace `self.assertEqual` with `assert ==`; call `make_psf_points(cell_coadd, cell_coadd.bbox)` inline; add imperative docstring
- [x] 3.6 Convert `test_roundtrip` — split into two functions, both using `RoundtripFits(cell_coadd, "CellCoadd")`:
  - `test_roundtrip`: signature `def test_roundtrip(cell_coadd_data: dict[str, Any]) -> None:`; contains only non-component-read calls: the bbox-only subimage check (`roundtrip.get(bbox=subbox)`, no `component` arg), the FITS header loop (`roundtrip.inspect()`), and post-`with` assertions (bbox/missing-cell self-consistency, `assert_cell_coadds_equal`, `compare_cell_coadd_to_legacy` with `alternates={}`); replace `self.assertEqual`/`self.assertIsNone` with plain `assert`; add imperative docstring
  - `test_roundtrip_components`: signature `def test_roundtrip_components(cell_coadd_data: dict[str, Any]) -> None:`; contains all `get(component=...)` calls (subpsf, "bbox", "sky_projection", all alternates dict entries, "components", "backgrounds"); calls `compare_cell_coadd_to_legacy(..., alternates=alternates)` after populating `alternates` but still inside the `with` block; replace `self.assertEqual`/`self.assertIsNone` with plain `assert`; add imperative docstring
- [x] 3.7 Convert `test_fits_compression` — signature `def test_fits_compression(cell_coadd_data: dict[str, Any]) -> None:`; replace `with self.subTest(extname=extname):` with a plain `for` loop retaining `extname` as the named loop variable; replace `self.assertEqual` with `assert ==`; add imperative docstring
- [x] 3.8 Convert `test_fits_json_consistency` — signature `def test_fits_json_consistency(cell_coadd_data: dict[str, Any]) -> None:`; no assertion changes needed (already uses `assert_*` helpers); add imperative docstring
- [x] 3.9 Convert `test_to_legacy` — signature `def test_to_legacy(cell_coadd_data: dict[str, Any]) -> None:`; add imperative docstring; call `make_psf_points` inline
- [x] 3.10 Convert `test_to_legacy_exposure` — signature `def test_to_legacy_exposure(cell_coadd_data: dict[str, Any]) -> None:`; add `@skip_no_afw` decorator (calls `cell_coadd.to_legacy_exposure()` which does `from lsst.afw.image import ...` inline); replace `self.assertEqual` with `assert ==`; call `make_psf_points` inline; add imperative docstring
- [x] 3.11 Convert `test_round_trip_ndf` — apply `@skip_no_h5py`; signature `def test_round_trip_ndf(cell_coadd_data: dict[str, Any]) -> None:`; add imperative docstring
- [x] 3.12 Convert `test_fits_ndf_consistency` — apply `@skip_no_h5py`; signature `def test_fits_ndf_consistency(cell_coadd_data: dict[str, Any]) -> None:`; add imperative docstring
- [x] 3.13 Remove the `CellCoaddTestCase` class definition entirely

## 4. Update _roundtrip.py docstrings

- [x] 4.1 In `python/lsst/images/tests/_roundtrip.py`, update `TemporaryButler` class docstring: replace "Raised when the context manager is entered if `lsst.daf.butler` could not be imported. This is typically handled by using this context manager within a `unittest.TestCase.subTest` context..." with "Skips the current test when the context manager is entered if `lsst.daf.butler` could not be imported." (remove the `Raises` section or recast it as a `Notes` item)
- [x] 4.2 In `RoundtripBase.get` docstring: replace "This requires the roundtrip to use a butler, raising `unittest.SkipTest` otherwise; this generally means these tests should be nested within a `~unittest.TestCase.subTest` context." with "This requires the roundtrip to use a butler; `pytest.skip()` is called otherwise. Place calls to this method in a dedicated test function that contains only component-read assertions, so the skip does not suppress unrelated always-run assertions."

## 5. Verification

- [x] 5.1 Run `ruff check --fix python/ tests/` and `ruff format python/ tests/` — confirm zero errors
- [x] 5.2 Run `pytest tests/test_visit_image.py tests/test_cell_coadd.py -r a -v` — confirm all tests pass or skip; confirm `test_read_write` and `test_roundtrip` appear as independent passing nodes; confirm `test_read_write_components` and `test_roundtrip_components` appear (passing with butler, skipped without)
- [x] 5.3 Confirm `grep -l "class.*TestCase\|import unittest" tests/` returns nothing (no remaining `unittest.TestCase` files)
- [x] 5.4 Run `pytest tests/ -r a -v -n 3` — confirm no regressions across the full suite
