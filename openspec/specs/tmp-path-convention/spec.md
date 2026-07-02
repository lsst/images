### Requirement: Test functions use tmp_path for temporary files
Test functions that need a temporary file or directory SHALL receive pytest's
`tmp_path: pathlib.Path` fixture parameter instead of creating a
`tempfile.TemporaryDirectory` or `tempfile.NamedTemporaryFile` internally.

#### Scenario: Test function creates a temporary FITS file
- **WHEN** a test function needs a writable temporary `.fits` path
- **THEN** it MUST declare `tmp_path: pathlib.Path` as a parameter and use `tmp_path / "x.fits"` as the path

#### Scenario: Test function creates a temporary NDF file
- **WHEN** a test function needs a writable temporary `.sdf` path
- **THEN** it MUST declare `tmp_path: pathlib.Path` as a parameter and use `tmp_path / "x.sdf"` as the path

### Requirement: No tempfile.mkdtemp calls exist in test files
The `tempfile.mkdtemp()` function SHALL NOT be used in any test file, as it
creates directories that are never cleaned up.

#### Scenario: mkdtemp resource leak is eliminated
- **WHEN** `grep -r 'mkdtemp' tests/` is run after implementation
- **THEN** it MUST produce no output

### Requirement: Non-fixture helper functions accept tmp_path as a parameter
Module-level helper functions in test files that previously created their own
`tempfile.TemporaryDirectory` SHALL instead accept a `tmp_path: pathlib.Path`
parameter and use it for all temporary paths. Callers (test functions) SHALL
pass their own `tmp_path` fixture value.

#### Scenario: Helper function _write_archive accepts tmp_path
- **WHEN** a test function calls `_write_archive()` in `test_fits_output_archive.py`
- **THEN** it MUST pass its own `tmp_path` as an argument, and `_write_archive` MUST use it instead of creating a `TemporaryDirectory`

#### Scenario: Helper functions make_hdu_list and _cutdown accept tmp_path
- **WHEN** a test function calls `make_hdu_list()` or `_cutdown()` in `test_from_hdu_list.py`
- **THEN** it MUST pass its own `tmp_path` as an argument, and the helper MUST use it instead of creating a `TemporaryDirectory`

### Requirement: tempfile is not imported in test files that no longer use it
After conversion, any test file that no longer calls any `tempfile` API SHALL NOT
import the `tempfile` module.

#### Scenario: No spurious tempfile imports remain
- **WHEN** `grep -r 'import tempfile' tests/` is run after implementation
- **THEN** it MUST produce no output (assuming all files are fully converted)
