### Requirement: External data directory constant is named EXTERNAL_DATA_DIR
Every test module that reads the `TESTDATA_IMAGES_DIR` environment variable SHALL
assign the result to a module-level constant named `EXTERNAL_DATA_DIR`, not `DATA_DIR`
or any other name.

#### Scenario: Env-var constant uses canonical name
- **WHEN** a test file assigns `os.environ.get("TESTDATA_IMAGES_DIR", None)` to a module-level constant
- **THEN** that constant MUST be named `EXTERNAL_DATA_DIR`

#### Scenario: All use-sites of the external path use the canonical name
- **WHEN** a test function or fixture references the external data path
- **THEN** it MUST use `EXTERNAL_DATA_DIR`, not `DATA_DIR`

### Requirement: Local data directory constant is named LOCAL_DATA_DIR
Every test module that computes a path to the in-repo committed fixtures
(i.e., a path derived from `os.path.dirname(__file__)`) SHALL assign that path
to a module-level constant named `LOCAL_DATA_DIR`, not `DATA_DIR` or any other name.

#### Scenario: In-repo path constant uses canonical name
- **WHEN** a test file assigns an `os.path.join(os.path.dirname(__file__), ...)` value to a module-level constant
- **THEN** that constant MUST be named `LOCAL_DATA_DIR`

#### Scenario: All use-sites of the local path use the canonical name
- **WHEN** a test function or fixture references the in-repo fixture path
- **THEN** it MUST use `LOCAL_DATA_DIR`, not `DATA_DIR`

### Requirement: No stale DATA_DIR references remain in test files
After the rename, the name `DATA_DIR` SHALL NOT appear as a variable name or
reference in any file under `tests/`.

#### Scenario: Grep finds no stale DATA_DIR variables
- **WHEN** `grep -r '\bDATA_DIR\b' tests/` is run after implementation
- **THEN** it MUST produce no output

### Requirement: Docstrings and comments use the correct constant name
Any docstring or inline comment in a test file that refers to the data directory
constant by name SHALL use `EXTERNAL_DATA_DIR` or `LOCAL_DATA_DIR` as appropriate.

#### Scenario: Typo in test_difference_image_extras.py is corrected
- **WHEN** `test_difference_image_extras.py` is read
- **THEN** the string `EXXTERNAL_DATA_DIR` SHALL NOT appear anywhere in the file
