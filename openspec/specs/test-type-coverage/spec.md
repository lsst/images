### Requirement: tmp_path parameters are annotated pathlib.Path
Any test function or fixture that receives the pytest `tmp_path` built-in fixture
SHALL annotate it as `tmp_path: pathlib.Path`, not `pytest.MonkeyPatch` or any
other type.

#### Scenario: Correct annotation in test_serialization_io.py
- **WHEN** `mypy tests/test_serialization_io.py` is run
- **THEN** it MUST report no errors related to `tmp_path` parameter types

#### Scenario: Correct annotation in test_fuzz.py
- **WHEN** `mypy tests/test_fuzz.py` is run
- **THEN** it MUST report no errors related to `tmp_path` parameter types

### Requirement: Fixture functions carry complete return type annotations
Every `@pytest.fixture` function in the affected test files SHALL declare a return
type annotation.

#### Scenario: visit_image fixture in test_serialization_reader.py is annotated
- **WHEN** `mypy tests/test_serialization_reader.py` is run
- **THEN** it MUST report no missing-return-type errors on the `visit_image` fixture

#### Scenario: legacy_camera_fixture in test_transforms.py is annotated
- **WHEN** `mypy tests/test_transforms.py` is run
- **THEN** it MUST report no missing-return-type errors on `legacy_camera_fixture` or `legacy_detector_wcs_fixture`

### Requirement: Test function parameters are fully annotated in affected files
Every `def test_*` function in `test_serialization_reader.py` and
`test_transforms.py` SHALL annotate all parameters (including fixture parameters
received from pytest).

#### Scenario: test_serialization_reader.py functions fully annotated
- **WHEN** `mypy tests/test_serialization_reader.py` is run
- **THEN** it MUST report no missing-annotation errors on any test function

#### Scenario: test_transforms.py test functions fully annotated
- **WHEN** `mypy tests/test_transforms.py` is run
- **THEN** it MUST report no missing-annotation errors on `test_camera` or `test_detector_wcs`

### Requirement: All test functions carry explicit return type annotations
Every `def test_*` function in `tests/` SHALL declare `-> None` as its return type.

#### Scenario: test_ndf_input_archive.py fully annotated
- **WHEN** `mypy tests/test_ndf_input_archive.py` is run
- **THEN** it MUST report no missing-return-type errors on any test function

#### Scenario: test_ndf_output_archive.py fully annotated
- **WHEN** `mypy tests/test_ndf_output_archive.py` is run
- **THEN** it MUST report no missing-return-type errors on any test function

### Requirement: No TODO[DM-54956] annotation markers remain in affected files
After this change, all `TODO[DM-54956]` comments that relate to type annotations
SHALL be removed from the affected test files.

#### Scenario: Annotation TODO removed from test_serialization_reader.py
- **WHEN** `grep 'TODO\[DM-54956\]' tests/test_serialization_reader.py` is run
- **THEN** it MUST produce no output

#### Scenario: Annotation TODO removed from test_serialization_io.py
- **WHEN** `grep 'TODO\[DM-54956\]' tests/test_serialization_io.py` is run
- **THEN** it MUST produce no output
