## Requirements

### Requirement: Test helper functions accept no test-case object
All public functions in `lsst.images.tests._checks` and `lsst.images.tests._roundtrip`
SHALL NOT accept a `unittest.TestCase` instance as a parameter. Assertions SHALL be
expressed as bare `assert` statements, `pytest.raises(...)`, or explicit
`raise AssertionError(...)` so that they work with any Python test runner without
a `unittest` dependency.

#### Scenario: Helper called without a test-case argument
- **WHEN** a caller invokes `assert_images_equal(a, b)` (two positional arguments)
- **THEN** the function SHALL compare the two images and raise `AssertionError` on mismatch

#### Scenario: Helper raises AssertionError on inequality
- **WHEN** two values passed to an assertion helper are not equal
- **THEN** the helper SHALL raise `AssertionError` with a message identifying the mismatch

#### Scenario: RoundtripBase constructed without a test-case argument
- **WHEN** a caller constructs `RoundtripFits(obj, "StorageClass")` (no leading `self`)
- **THEN** the context manager SHALL complete the write/read cycle and expose `result`

#### Scenario: Skip raised via pytest mechanism
- **WHEN** `TemporaryButler` is entered and `lsst.daf.butler` is not importable
- **THEN** `pytest.skip(...)` SHALL be called, causing pytest to record a skip

### Requirement: assert_equal_allow_nan uses explicit raise
`assert_equal_allow_nan` SHALL compare two floats for equality treating NaN as equal to
NaN, and SHALL raise `AssertionError` with the message `f"{a!r} != {b!r}"` when the values
are neither equal nor both NaN.

#### Scenario: Equal values pass
- **WHEN** both arguments are equal finite floats
- **THEN** the function SHALL return without raising

#### Scenario: Both NaN passes
- **WHEN** both arguments are `float('nan')`
- **THEN** the function SHALL return without raising

#### Scenario: Unequal non-NaN values fail
- **WHEN** the two arguments are unequal and neither is NaN
- **THEN** the function SHALL raise `AssertionError` with message `f"{a!r} != {b!r}"`
