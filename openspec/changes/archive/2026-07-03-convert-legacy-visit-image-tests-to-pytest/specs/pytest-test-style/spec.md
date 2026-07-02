## ADDED Requirements

### Requirement: test_convert_unit runs on all nJy-unit legacy datasets
`test_convert_unit` SHALL be parametrized over all datasets whose image unit is
`u.nJy` (currently `visit_image` and `difference_image`) rather than running
only against the `visit_image` dataset.  Datasets whose unit is not `u.nJy`
(currently `preliminary_visit_image`) SHALL be excluded from this test's
parameter list.

#### Scenario: test_convert_unit runs for visit_image dataset
- **WHEN** `TESTDATA_IMAGES_DIR` is set and `test_convert_unit` is collected
- **THEN** it SHALL produce a test node `test_convert_unit[visit_image]` that
  passes

#### Scenario: test_convert_unit runs for difference_image dataset
- **WHEN** `TESTDATA_IMAGES_DIR` is set and `test_convert_unit` is collected
- **THEN** it SHALL produce a test node `test_convert_unit[difference_image]`
  that passes

#### Scenario: test_convert_unit does not run for preliminary_visit_image dataset
- **WHEN** `TESTDATA_IMAGES_DIR` is set and `test_convert_unit` is collected
- **THEN** it SHALL NOT produce a test node named
  `test_convert_unit[preliminary_visit_image]`
