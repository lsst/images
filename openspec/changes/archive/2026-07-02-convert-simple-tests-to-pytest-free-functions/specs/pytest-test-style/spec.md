## ADDED Requirements

### Requirement: Ten additional test files conform to pytest free-function style
The ten files listed in the proposal SHALL be fully converted to pytest
free-function style, satisfying all existing requirements in the
`pytest-test-style` spec (no `unittest.TestCase` subclasses, no `import
unittest`, `@pytest.fixture` for shared objects, `@pytest.mark.skipif` for
optional-dependency guards, plain `assert` statements, imperative docstrings,
no banner comments).

#### Scenario: Each converted file passes ruff and pytest
- **WHEN** any of the ten converted files is linted with `ruff check` and run
  with `pytest -r a -v -n 3`
- **THEN** it SHALL produce no lint errors and all its tests SHALL pass (or
  skip when the relevant optional dependency is absent)

### Requirement: Converted subTest loops preserve iteration identity on failure
Every `with self.subTest(key=val)` loop that is converted SHALL ensure that a
test failure identifies which iteration failed, either by using
`@pytest.mark.parametrize` (which embeds the parameter value in the test node
name) or by retaining the iteration variable as a named local in the loop body
so that pytest's assertion rewriting displays it in the traceback.

#### Scenario: Large or independent-entry loops use parametrize
- **WHEN** a `subTest` loop iterates over a collection where failures in
  different iterations are likely to be independent (e.g. one fixture file vs
  another, one component name vs another) and continue-on-failure materially
  helps diagnosis
- **THEN** the converted test SHALL use `@pytest.mark.parametrize` so each
  entry becomes a separate pytest node with its value embedded in the test ID

#### Scenario: Small loops retain iteration variable in traceback
- **WHEN** a `subTest` loop iterates over a small (≤3 item) fixed collection
  where a single root cause typically explains all failures
- **THEN** the converted test MAY use a plain `for` loop with `assert`,
  provided the loop variable is a named local that pytest will display on
  failure

#### Scenario: Per-item pytest.skip replaces per-item self.skipTest
- **WHEN** a `subTest` loop calls `self.skipTest(...)` to skip individual
  iterations whose data is absent
- **THEN** the converted test SHALL use `@pytest.mark.parametrize` with
  `pytest.skip(...)` called inline, so each entry is independently skippable
