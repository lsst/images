## ADDED Requirements

### Requirement: Fourteen additional test files conform to pytest free-function style
The fourteen files listed in the proposal for this batch SHALL be fully
converted to pytest free-function style, satisfying all existing requirements
in the `pytest-test-style` spec (no `unittest.TestCase` subclasses, no `import
unittest`, factory functions or `@pytest.fixture` for shared objects,
`@pytest.mark.skipif` for optional-dependency guards, plain `assert`
statements, imperative docstrings, no banner comments).

#### Scenario: Each converted file passes ruff and pytest
- **WHEN** any of the fourteen converted files is linted with `ruff check` and
  run with `pytest -r a -v -n 3`
- **THEN** it SHALL produce no lint errors and all its tests SHALL pass (or
  skip when the relevant optional dependency is absent)

### Requirement: Optional-dependency skip markers use named module-level constants
When a test file conditionally skips tests based on an optional import, the
skip condition SHALL be captured in a named module-level marker constant of the
form `skip_no_<dep>` (e.g. `skip_no_h5py`, `skip_no_afw`) rather than
repeating the condition inline on every affected function.

#### Scenario: h5py-guarded tests use skip_no_h5py
- **WHEN** a converted test file guards tests against missing `h5py`
- **THEN** it SHALL define `skip_no_h5py = pytest.mark.skipif(not HAVE_H5PY, reason="h5py is not installed")`
  at module level and apply `@skip_no_h5py` to each guarded function

#### Scenario: External-data-guarded tests use a named skip marker
- **WHEN** a converted test file guards tests against a missing external data
  directory (e.g. `TESTDATA_IMAGES_DIR`)
- **THEN** it SHALL define a named skip marker constant at module level and
  apply it consistently rather than repeating the `pytest.mark.skipif`
  expression on every function

### Requirement: subTest inside a RoundtripX context manager is replaced without losing test isolation
A `with self.subTest():` block nested inside a `with RoundtripX(...):` block
SHALL be replaced in a way that preserves the outer test's ability to pass
when the inner optional assertion is skipped.  The preferred approach is to
extract the guarded block into a separate test function decorated with an
appropriate skip marker, so the outer test's assertions about `roundtrip.result`
are independent of the inner optional block.

#### Scenario: afw-dependent inner block becomes its own test function
- **WHEN** a `with self.subTest():` block inside `with RoundtripFits(...) as roundtrip:`
  guards an afw import and afw-specific assertions
- **THEN** the converted code SHALL extract that block into a separate
  `def test_*_legacy_read(...)` function that takes the roundtrip result as input
  or re-reads the file, so the outer test does not depend on afw being present

#### Scenario: Pure skip inside subTest becomes pytest.skip
- **WHEN** a `with self.subTest():` block inside a `RoundtripX` context serves
  only to prevent a `self.skipTest(...)` from aborting the outer test
- **THEN** the converted code SHALL replace it with `pytest.skip(...)` called
  directly, which skips only the current test function (not the process)

### Requirement: Session fixtures use pytest.skip internally for optional-dependency guards
A session-scoped fixture whose construction depends on an optional import or
external data directory SHALL call `pytest.skip(reason)` inside its body when
the prerequisite is absent, rather than using a `@pytest.mark.skipif` decorator
on the fixture.  pytest 9+ explicitly prohibits marks on fixtures, and
`pytest.skip()` inside the body propagates the skip to every test that requests
the fixture, with the correct reason string, including through fixture chains.

#### Scenario: Session fixture skips on missing external data
- **WHEN** a session fixture reads from `TESTDATA_IMAGES_DIR` and the variable
  is not set
- **THEN** the fixture body SHALL call `pytest.skip("TESTDATA_IMAGES_DIR is not set")`
  and every test that injects that fixture SHALL be automatically skipped —
  no `@pytest.mark.skipif` decorator needed on the test functions

#### Scenario: Session fixture skips on missing optional import
- **WHEN** a session fixture requires an optional package (e.g. `lsst.afw`) and
  that package is not importable
- **THEN** the fixture body SHALL use `pytest.importorskip("lsst.afw.image")` or
  a try/except with `pytest.skip(...)` so the skip propagates to all consumers

#### Scenario: Fixture chain propagates the earliest skip reason
- **WHEN** fixture B depends on fixture A, and fixture A calls `pytest.skip()`
- **THEN** tests requesting fixture B SHALL be skipped with fixture A's reason,
  without any additional skip guard in fixture B or in the test functions

### Requirement: Test functions SHALL NOT use skip_no_data_dir decorators
Any test whose execution depends on `TESTDATA_IMAGES_DIR` being set SHALL
obtain that data via a session-scoped fixture that performs the skip check
internally.  No `@pytest.mark.skipif(DATA_DIR is None, ...)` decorator (or
named equivalent such as `skip_no_data_dir`) SHALL appear on a test function.
`skip_no_*` module-level markers remain acceptable for guarding against missing
optional *imports* (e.g. `skip_no_h5py`, `skip_no_butler`) but not for missing
external test data.

#### Scenario: External-data test uses a fixture, not a decorator
- **WHEN** a test reads a file from `TESTDATA_IMAGES_DIR`
- **THEN** it SHALL accept a session-scoped fixture as a parameter; that fixture
  SHALL call `pytest.skip("TESTDATA_IMAGES_DIR is not in the environment.")` in
  its body when the directory is absent, and the test function SHALL carry no
  `@skip_no_data_dir` or equivalent decorator

#### Scenario: No skip_no_data_dir defined at module level
- **WHEN** a converted test file has tests that read from `TESTDATA_IMAGES_DIR`
- **THEN** it SHALL NOT define a `skip_no_data_dir` constant at module level
