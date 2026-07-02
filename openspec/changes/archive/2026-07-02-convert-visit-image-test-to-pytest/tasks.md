## 1. Add session-scoped fixtures

- [x] 1.1 Add `visit_image_components` session fixture returning a `dict` with keys `mask_schema`, `obs_info`, `summary_stats`, `gaussian_psf`, `aperture_corrections`, `detector`, `image`, `variance`, `polygon`, `sky_projection` — extracted verbatim from `setUpClass`
- [x] 1.2 Add `visit_image` session fixture depending on `visit_image_components`: construct the fully-populated `VisitImage`, add the `"standard"` `ChebyshevField` background, and attach the `FitsOpaqueMetadata` (with `PLATFORM`/`LSST BUTLER ID` primary header)
- [x] 1.3 Add `simplest_visit_image` session fixture depending on `visit_image_components`: construct the minimal `VisitImage` with required arguments only

## 2. Convert helper methods to module-level functions

- [x] 2.1 Convert `_make_sum_background_visit_image` to a module-level function `_make_sum_background_visit_image(c, vi)` accepting the components dict and the `visit_image`; replace `self.rng` with a local `np.random.default_rng(42)`; replace all `self.*` attribute access with `c[...]` / `vi.*`
- [x] 2.2 Convert `_check_sum_background_round_trip` to a module-level function `_check_sum_background_round_trip(result, original)` replacing `self.assertIsInstance` / `self.assertEqual` / `self.assertIsNot` with plain `assert` statements

## 3. Convert test methods to free functions

- [x] 3.1 Convert `test_basics` — signature `def test_basics(simplest_visit_image, visit_image_components):`; replace all `self.assert*` with plain `assert` / `pytest.raises`; replace `self.rng` usage with a local `rng = np.random.default_rng(501)`
- [x] 3.2 Convert `test_copy_and_slice` — signature `def test_copy_and_slice(visit_image, visit_image_components):`; replace `self.assert*` with plain `assert`
- [x] 3.3 Convert `test_obs_info` — signature `def test_obs_info(visit_image):`; replace `self.assert*` / `self.maxDiff = None` with plain `assert`
- [x] 3.4 Convert `test_summary_stats` — signature `def test_summary_stats(visit_image_components):`; replace `self.assert*` with plain `assert`
- [x] 3.5 Convert `test_round_trip_ndf` — signature with `@pytest.mark.skipif(not HAVE_H5PY, reason="h5py is not installed")` decorator and `def test_round_trip_ndf(visit_image):`
- [x] 3.6 Convert `test_fits_ndf_consistency` — same `skipif` decorator, signature `def test_fits_ndf_consistency(visit_image):`
- [x] 3.7 Convert `test_fits_json_consistency` — signature `def test_fits_json_consistency(visit_image):`
- [x] 3.8 Convert `test_read_write` — signature `def test_read_write(visit_image, visit_image_components):`; remove all `with self.subTest():` wrappers, replacing with plain sequential assertions; replace `self.assert*` with plain `assert` / `pytest.raises`
- [x] 3.9 Convert `test_sum_background_round_trip_fits` — signature `def test_sum_background_round_trip_fits(visit_image, visit_image_components):`; call `_make_sum_background_visit_image` and `_check_sum_background_round_trip`
- [x] 3.10 Convert `test_sum_background_round_trip_ndf` — same `skipif` decorator, same pattern as 3.9

## 4. Clean up imports and module boilerplate

- [x] 4.1 Add `import pytest` to the import block
- [x] 4.2 Remove `if __name__ == "__main__": unittest.main()` block
- [x] 4.3 Remove `ClassVar` from the `typing` import (no longer needed after class removal); retain `unittest` import and `Any` as long as the legacy classes remain in the file
- [x] 4.4 Remove the `VisitImageTestCase` class definition (the `class` line and `setUpClass`) once all methods have been converted

## 5. Update the pytest-test-style spec

- [x] 5.1 Sync the three new requirements from `specs/pytest-test-style/spec.md` into `openspec/specs/pytest-test-style/spec.md` (session-scope rule, explicit dependency parameter rule, local rng rule)

## 6. Verify

- [x] 6.1 Run `pytest tests/test_visit_image.py -r a -v` and confirm all converted tests pass (or skip on missing optional deps)
- [x] 6.2 Run `ruff check python/ tests/` and `ruff format --check python/ tests/` — no errors
