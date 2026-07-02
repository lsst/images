## 1. Fix resource leak (highest priority)

- [x] 1.1 In `tests/test_serialization_backends.py` line 76: replace `tempfile.mkdtemp()` with `tmp_path` fixture parameter; remove `import tempfile`

## 2. Convert TemporaryDirectory in simple test functions

- [x] 2.1 In `tests/test_fits_date_header.py`: replace all 3 `tempfile.TemporaryDirectory` usages (lines 33, 54, 71) with `tmp_path`; add `tmp_path: Path` params; remove `import tempfile`
- [x] 2.2 In `tests/test_fits_format_version.py`: replace all 4 `tempfile.TemporaryDirectory` usages (lines 33, 43, 52, 68) with `tmp_path`; add `tmp_path: Path` params; remove `import tempfile`
- [x] 2.3 In `tests/test_ndf_format_version.py`: replace all 4 `tempfile.TemporaryDirectory` usages (lines 45, 58, 70, 91) with `tmp_path`; add `tmp_path: Path` params; remove `import tempfile`
- [x] 2.4 In `tests/test_serialization_basic_info.py`: replace all 4 `tempfile.TemporaryDirectory` usages (lines 97, 111, 126, 141) with `tmp_path`; add `tmp_path: Path` params; remove `import tempfile`
- [x] 2.5 In `tests/test_diagram.py` line 678: replace `tempfile.TemporaryDirectory` with `tmp_path`; add `tmp_path: Path` param; remove `import tempfile`; remove resolved TODO comment at line 677
- [x] 2.6 In `tests/test_masked_image.py` line 245: replace `NamedTemporaryFile(delete_on_close=False)` with `tmp_path`; add `tmp_path: Path` param; remove `import tempfile`
- [x] 2.7 In `tests/test_from_hdu_list.py` lines 412 and 434: replace `tempfile.TemporaryDirectory` with `tmp_path` in these two test functions

## 3. Convert TemporaryDirectory in helper functions (Pattern E)

- [x] 3.1 In `tests/test_fits_output_archive.py`: add `tmp_path: pathlib.Path` parameter to `_write_archive()`; replace internal `TemporaryDirectory` with `tmp_path`; update all call sites within the file to pass `tmp_path`; add `tmp_path: Path` to affected test function signatures; remove `import tempfile`
- [x] 3.2 In `tests/test_from_hdu_list.py`: add `tmp_path: pathlib.Path` parameter to `make_hdu_list()` and `_cutdown()`; replace internal `TemporaryDirectory` with `tmp_path`; update all test function call sites to pass `tmp_path`; add `tmp_path: Path` to affected test function signatures; remove `import tempfile`

## 4. Convert NamedTemporaryFile in NDF test files

- [x] 4.1 In `tests/test_ndf_model.py`: replace 2 `NamedTemporaryFile` usages (lines 63, 89) with `tmp_path`; add `tmp_path: Path` params; remove `import tempfile`
- [x] 4.2 In `tests/test_ndf_hds.py`: replace all 17 `NamedTemporaryFile` usages (lines 49, 66, 79, 93, 110, 118, 132, 146, 162, 177, 185, 214, 242, 262, 275, 289, and the double-with form at 110/177/185) with `tmp_path`; add `tmp_path: Path` params; remove `import tempfile`
- [x] 4.3 In `tests/test_ndf_output_archive.py`: replace all 29 `NamedTemporaryFile` usages (lines 72–668) with `tmp_path`; add `tmp_path: Path` params to all affected test functions; remove `import tempfile`
- [x] 4.4 In `tests/test_ndf_input_archive.py`: replace all 20 `NamedTemporaryFile(delete_on_close=False)` usages (lines 60–441) with `tmp_path`; add `tmp_path: Path` params; remove `import tempfile`

## 5. Verification

- [x] 5.1 Run `grep -r 'import tempfile' tests/` and confirm no output
- [x] 5.2 Run `grep -r 'mkdtemp\|TemporaryDirectory\|NamedTemporaryFile' tests/` and confirm no output
- [x] 5.3 Run `pytest -r a -v -n 3` and confirm all tests pass
