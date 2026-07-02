## Why

`tests/test_visit_image.py` still contains `VisitImageLegacyTestMixin` and three
`unittest.TestCase` subclasses (`VisitImageLegacyTestCase`,
`PreliminaryVisitImageLegacyTestCase`, `DifferenceImageLegacyTestCase`) that use
a mixin pattern with monolithic `setUpClass` blocks.  Converting them to pytest
free-functions with parametrized fixtures aligns the file with the project's
`pytest-test-style` spec, makes fixture dependencies explicit per test, and
extends `test_convert_unit` to also cover `DifferenceImage`.

## What Changes

- `tests/test_visit_image.py`: The mixin class `VisitImageLegacyTestMixin` and
  its three concrete `unittest.TestCase` subclasses are replaced by:
  - Two session-scoped parametrized fixtures (`legacy_dataset_context`,
    `legacy_visit_image`) that cover the three datasets
    (`visit_image`, `preliminary_visit_image`, `difference_image`).
  - A set of pytest free-functions replicating every mixin test method, running
    against all three datasets via the parametrized fixtures.
  - `test_convert_unit` promoted to a free-function parametrized over
    `["visit_image", "difference_image"]` only (skipping
    `preliminary_visit_image` because that dataset is in `u.electron`, not
    `u.nJy`).
  - A module-level helper `_check_legacy_obs_info(obs_info)` replacing the mixin
    helper method `check_legacy_obs_info`.
- `import unittest`, `ClassVar`, and `typing.Any` imports are removed once the
  last `unittest.TestCase` subclass is gone.
- `if __name__ == "__main__": unittest.main()` block removed.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `pytest-test-style`: No new requirements needed — existing requirements
  already cover parametrized fixtures, session scope, and fine-grained fixture
  dependencies.  No delta spec required.
- `visit-image-test-fixtures`: No new requirements needed — existing
  requirements already cover `visit_image_components`, `make_visit_image`, and
  the module-level helpers.  No delta spec required.

## Impact

- `tests/test_visit_image.py`: modified (legacy section only; the already-
  converted free-function section is untouched)
- No changes to `python/lsst/images/` source or public API
- No new external dependencies
- Test count increases: `test_convert_unit` now runs on two datasets instead of
  one
