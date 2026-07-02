## 1. test_fields.py — factory skip guard and decorator removal

- [x] 1.1 Add `if not HAVE_LEGACY: pytest.skip(...)` at the top of `make_legacy_fields()` in `tests/test_fields.py`
- [x] 1.2 Remove `@skip_no_legacy` from `test_chebyshev_roundtrip`
- [x] 1.3 Remove `@skip_no_legacy` from `test_product_roundtrip`
- [x] 1.4 Remove `@skip_no_legacy` from `test_spline_simple`
- [x] 1.5 Remove `@skip_no_legacy` from `test_spline_one_nan`
- [x] 1.6 Remove `@skip_no_legacy` from `test_chebyshev1_function2`

## 2. test_polygon.py — named skip constant

- [x] 2.1 Add `skip_no_legacy = pytest.mark.skipif(not have_legacy, reason="lsst legacy packages could not be imported.")` at module level in `tests/test_polygon.py`
- [x] 2.2 Replace the inline `@pytest.mark.skipif(not have_legacy, ...)` on `test_polygon_legacy` with `@skip_no_legacy`

## 3. test_visit_image.py — remove redundant skip_no_test_data decorators

- [x] 3.1 Remove `@skip_no_test_data` from `test_legacy_errors`
- [x] 3.2 Remove `@skip_no_test_data` from `test_component_reads`
- [x] 3.3 Remove `@skip_no_test_data` from `test_legacy_obs_info`
- [x] 3.4 Remove `@skip_no_test_data` from `test_aperture_corrections_to_legacy`
- [x] 3.5 Remove `@skip_no_test_data` from `test_read_legacy_headers`
- [x] 3.6 Remove `@skip_no_test_data` from `test_from_legacy_headers`
- [x] 3.7 Remove `@skip_no_test_data` from `test_rewrite`
- [x] 3.8 Remove `@skip_no_test_data` from `test_butler_converters`
- [x] 3.9 Remove `@skip_no_test_data` from `test_convert_unit`
- [x] 3.10 Remove the `assert EXTERNAL_DATA_DIR is not None, "Guaranteed by decorator."` line and its surrounding comment in `test_convert_unit`
- [x] 3.11 Remove the `skip_no_test_data` module-level constant definition from `tests/test_visit_image.py`

## 4. test_visit_image.py — named h5py skip constant

- [x] 4.1 Add `skip_no_h5py = pytest.mark.skipif(not HAVE_H5PY, reason="h5py is not installed")` at module level in `tests/test_visit_image.py`
- [x] 4.2 Replace the inline `@pytest.mark.skipif(not HAVE_H5PY, ...)` on `test_round_trip_ndf` (line ~374) with `@skip_no_h5py`
- [x] 4.3 Replace the inline `@pytest.mark.skipif(not HAVE_H5PY, ...)` on `test_fits_ndf_consistency` (line ~382) with `@skip_no_h5py`
- [x] 4.4 Replace the inline `@pytest.mark.skipif(not HAVE_H5PY, ...)` on the third h5py-guarded test (line ~519) with `@skip_no_h5py`

## 5. Verify

- [x] 5.1 Run `pytest tests/test_fields.py tests/test_polygon.py tests/test_visit_image.py -r a -v` and confirm all tests pass or skip with the expected reasons
- [x] 5.2 Run `ruff check --fix python/ tests/` and confirm no lint errors
