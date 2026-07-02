## 1. Spec update

- [x] 1.1 Add the two new sub-rules (factory preference, `skip_no_*` naming) to `openspec/specs/pytest-test-style/spec.md`

## 2. Easy conversions (no setUp, or trivial setUp)

- [x] 2.1 Convert `tests/test_psfs.py` — strip `self.`, `@skip_no_afw` + `@skip_no_data_dir` guards, extract inline `subTest("NDF round-trip")` into a `pytest.skip` call
- [x] 2.2 Convert `tests/test_fuzz.py` — `ShuffleBlocksTestCase` (no setUp) and `FuzzMaskedImageCommandTestCase` (setUp → `tmp` fixture with `tmp_path`)
- [x] 2.3 Convert `tests/test_ndf_hds.py` — three classes all `@skip_no_h5py`-guarded, no setUp; purely mechanical
- [x] 2.4 Convert `tests/test_difference_image_extras.py` — `setUpClass` → session fixture loading legacy objects; `@staticmethod` helpers → module-level; whole class guarded by `skip_no_data_dir`
- [x] 2.5 Convert `tests/test_serialization_basic_info.py` — four small independent classes; `setUp` → `tmp_path` fixture or inline; `@skip_no_h5py` guard on NDF class

## 3. Moderate conversions (setUp → factory or fixture)

- [x] 3.1 Convert `tests/test_color_image.py` — `setUp` → `make_color_image()` factory; `assert_color_images_equal` → module-level function; `subTest` loop over 3 channels → plain `for` loop
- [x] 3.2 Convert `tests/test_serialization_io.py` — `FixtureSweepTestCase.test_sweep` → `@pytest.mark.parametrize("entry", sorted(EXPECTED_TYPES))` with per-entry piff skip; other classes → factory or `tmp_path`
- [x] 3.3 Convert `tests/test_serialization_reader.py` — per-class `setUp` (write file) → session fixture loading VisitImage + per-test `tmp_path` for file writes; `_check_components_and_read` → module-level helper
- [x] 3.4 Convert `tests/test_formatter_cache.py` — butler guard → `skip_no_butler`; `setUp` loads JSON fixture → session fixture; `_reset_cache` staticmethod → module-level helper called in per-test fixture
- [x] 3.5 Convert `tests/test_image.py` — `setUp`-free (uses `lsst.utils.tests.getTempFilePath`); strip `self.`; extract afw inner `subTest` block into `test_read_write_legacy_read()` with `@skip_no_afw`
- [x] 3.6 Convert `tests/test_mask.py` — `setUp` → `make_mask(rng)` factory + `make_mask_planes()` module-level helper; `getTempFilePath` stays; extract afw `subTest` into `test_legacy_legacy_read()` with `@skip_no_afw`
- [x] 3.7 Convert `tests/test_masked_image.py` — `setUp` builds complex `MaskedImage` → `make_masked_image()` factory (since tests mutate it); extract afw `subTest` into `test_fits_roundtrip_legacy_read()` with `@skip_no_afw`
- [x] 3.8 Convert `tests/test_fields.py` — three classes: `FieldTestCase.setUp` → `make_fields()` factory; `check_*` helpers → module-level; `FieldLegacyTestCase.setUp` → `make_legacy_fields()` factory; `FieldLegacyDataTestCase` (no setUp) guarded by `skip_no_data_dir`
- [x] 3.9 Convert `tests/test_transforms.py` — single `TransformTestCase`; `compare_to_legacy_camera` `@staticmethod` → module-level function; `FrameSetTestHolder`/`FrameSetTestHolderModel` preserved as-is; no setUp

## 4. Verification

- [x] 4.1 Run `ruff check --fix python/ tests/` and `ruff format python/ tests/` — confirm zero errors
- [x] 4.2 Run `pytest tests/ -r a -v -n 3` — confirm all tests pass or skip (no new failures)
- [x] 4.3 Confirm no `unittest.TestCase` remains in the 14 converted files: `grep -l "class.*TestCase" tests/` shows only the deferred files
