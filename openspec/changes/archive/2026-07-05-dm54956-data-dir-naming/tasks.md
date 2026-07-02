## 1. Rename DATA_DIR → EXTERNAL_DATA_DIR (env-var files)

- [x] 1.1 In `tests/test_masked_image.py`: rename `DATA_DIR` → `EXTERNAL_DATA_DIR` at assignment (line 47) and all use-sites
- [x] 1.2 In `tests/test_fields.py`: rename `DATA_DIR` → `EXTERNAL_DATA_DIR` at assignment (line 58) and all use-sites
- [x] 1.3 In `tests/test_transforms.py`: rename `DATA_DIR` → `EXTERNAL_DATA_DIR` at assignment (line 49) and all use-sites
- [x] 1.4 In `tests/test_psfs.py`: rename `DATA_DIR` → `EXTERNAL_DATA_DIR` at assignment (line 38) and all use-sites
- [x] 1.5 In `tests/test_image.py`: rename `DATA_DIR` → `EXTERNAL_DATA_DIR` at assignment (line 49) and all use-sites
- [x] 1.6 In `tests/test_cell_coadd.py`: rename `DATA_DIR` → `EXTERNAL_DATA_DIR` at assignment (line 57) and all use-sites (including the docstring reference on line 65)
- [x] 1.7 In `tests/test_mask.py`: rename `DATA_DIR` → `EXTERNAL_DATA_DIR` at assignment (line 40) and all use-sites
- [x] 1.8 In `tests/test_cameras.py`: rename `DATA_DIR` → `EXTERNAL_DATA_DIR` at assignment (line 23) and all use-sites

## 2. Rename DATA_DIR → LOCAL_DATA_DIR (in-repo fixture files)

- [x] 2.1 In `tests/test_detached_archive.py`: rename `DATA_DIR` → `LOCAL_DATA_DIR` at assignment (line 32) and all use-sites
- [x] 2.2 In `tests/test_serialization_reader.py`: rename `DATA_DIR` → `LOCAL_DATA_DIR` at assignment (line 37) and all use-sites
- [x] 2.3 In `tests/test_formatter_cache.py`: rename `DATA_DIR` → `LOCAL_DATA_DIR` at assignment (line 33) and all use-sites
- [x] 2.4 In `tests/test_serialization_io.py`: rename `DATA_DIR` → `LOCAL_DATA_DIR` at assignment (line 38) and all use-sites

## 3. Fix docstring typo and remove resolved TODO

- [x] 3.1 In `tests/test_difference_image_extras.py` line 57: fix typo `EXXTERNAL_DATA_DIR` → `EXTERNAL_DATA_DIR` in docstring
- [x] 3.2 In `tests/test_visit_image.py` lines 79–80: remove the `TODO[DM-54956]` comment (the naming is now standardized)

## 4. Verification

- [x] 4.1 Run `grep -r '\bDATA_DIR\b' tests/` and confirm no output (no stale references)
- [x] 4.2 Run `pytest -r a -v -n 3` and confirm all tests pass
