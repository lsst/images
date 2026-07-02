## 1. Fix wrong pytest.MonkeyPatch annotations (Category A)

- [x] 1.1 In `tests/test_serialization_io.py`: change `tmp_path: pytest.MonkeyPatch` → `tmp_path: Path` on lines 86, 95, 130, 140, 151, 161; ensure `from pathlib import Path` is present; remove the `TODO[DM-54956]` comment on line 85
- [x] 1.2 In `tests/test_fuzz.py`: change `tmp_path: pytest.MonkeyPatch` → `tmp_path: Path` on lines 91 and 115; ensure `from pathlib import Path` is present
- [x] 1.3 Run `mypy tests/test_serialization_io.py tests/test_fuzz.py` and resolve any newly surfaced errors

## 2. Annotate test_serialization_reader.py (Category B)

- [x] 2.1 Add return type `-> VisitImage` to the `visit_image` fixture (line 41); add `VisitImage` to imports if not already present
- [x] 2.2 Add `visit_image: VisitImage` and `tmp_path: pathlib.Path` annotations to the following 15 test functions: `test_fits_open_tree_yields_archive_tree_and_info` (line 67), `test_fits_read_still_works` (80), `test_ndf_open_tree_yields_archive_tree_and_info` (91), `test_ndf_read_still_works` (104), `test_json_open_tree_yields_archive_tree_and_info` (113), `test_json_read_still_works` (125), `test_reader_api_components_and_read_fits` (147), `test_reader_api_components_and_read_json` (154), `test_reader_api_components_and_read_ndf` (162), `test_reader_api_info` (169), `test_reader_api_cls_match` (181), `test_reader_api_cls_mismatch_raises` (192), `test_reader_api_unknown_component` (206), `test_reader_api_use_after_close_raises` (220), `test_fits_open_reads_file_once` (232)
- [x] 2.3 Remove the `TODO[DM-54956]` comment from `test_serialization_reader.py` line 66
- [x] 2.4 Run `mypy tests/test_serialization_reader.py` and resolve any newly surfaced errors

## 3. Annotate test_transforms.py (Category B)

- [x] 3.1 Add return type annotation to `legacy_camera_fixture` fixture (line 56); use `Any` if the lsst.afw type is optional
- [x] 3.2 Add return type annotation to `legacy_detector_wcs_fixture` fixture (line 73); use `Any` or the appropriate concrete type
- [x] 3.3 Add fixture parameter type to `test_camera` (line 288) matching the return type of `legacy_camera_fixture`
- [x] 3.4 Add fixture parameter type to `test_detector_wcs` (line 336) matching the return type of `legacy_detector_wcs_fixture`
- [x] 3.5 Run `mypy tests/test_transforms.py` and resolve any newly surfaced errors

## 4. Add -> None to test_ndf_input_archive.py (Category B)

- [x] 4.1 Add `-> None` return type to all 23 test functions in `tests/test_ndf_input_archive.py` (completed as part of dm54956-tempfile-to-tmp-path)
- [x] 4.2 Run `mypy tests/test_ndf_input_archive.py` and resolve any newly surfaced errors

## 5. Add -> None to test_ndf_output_archive.py (Category B)

- [x] 5.1 Add `-> None` return type to all 28 test functions in `tests/test_ndf_output_archive.py` (completed as part of dm54956-tempfile-to-tmp-path)
- [x] 5.2 Run `mypy tests/test_ndf_output_archive.py` and resolve any newly surfaced errors

## 6. Verification

- [x] 6.1 Run `mypy python/ tests/` (full suite) and confirm no annotation errors in affected files
- [x] 6.2 Run `pytest -r a -v -n 3` and confirm all tests pass
