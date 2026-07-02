## MODIFIED Requirements

### Requirement: Session fixtures use pytest.skip internally for optional-dependency guards
A session-scoped fixture whose construction depends on an optional import or
external data directory SHALL call `pytest.skip(reason)` inside its body when
the prerequisite is absent, rather than using a `@pytest.mark.skipif` decorator
on the fixture.  pytest 9+ explicitly prohibits marks on fixtures (`pytest.PytestRemovedIn9Warning`
that becomes an error in pytest 10), and `pytest.skip()` inside the body
propagates the skip to every test that requests the fixture, with the correct
reason string, including through fixture chains.

Module-level factory functions that return legacy objects (objects whose
construction requires an optional import) SHALL likewise call `pytest.skip()`
at the top of their body when the optional dependency is absent.  A test that
calls such a factory obtains the same automatic skip propagation as a test that
injects a fixture.

#### Scenario: Session fixture skips on missing external data
- **WHEN** a session fixture reads from `TESTDATA_IMAGES_DIR` and the variable
  is not set
- **THEN** the fixture body SHALL call `pytest.skip("TESTDATA_IMAGES_DIR is not set")`
  and every test that injects that fixture SHALL be automatically skipped with
  that reason — no `@pytest.mark.skipif` decorator needed on the test functions

#### Scenario: Session fixture skips on missing optional import
- **WHEN** a session fixture requires an optional package (e.g. `lsst.afw`) and
  that package is not importable
- **THEN** the fixture body SHALL use `pytest.importorskip("lsst.afw.image")` or
  a try/except with `pytest.skip(...)` so the skip propagates to all consumers

#### Scenario: Fixture chain propagates the earliest skip reason
- **WHEN** fixture B depends on fixture A, and fixture A calls `pytest.skip()`
- **THEN** tests requesting fixture B SHALL be skipped with fixture A's reason,
  without any additional skip guard in fixture B or in the test functions

#### Scenario: Factory function skips on missing optional import
- **WHEN** a module-level factory function requires an optional package (e.g.
  `lsst.afw.math`) and that package is not importable
- **THEN** the factory body SHALL call `pytest.skip(reason)` at the top, before
  any reference to names that would be undefined, so any test that calls the
  factory is automatically skipped

#### Scenario: Test calling a skip-aware factory needs no skip decorator
- **WHEN** a test function calls a module-level factory that embeds `pytest.skip()`
  for a missing optional dependency
- **THEN** the test function SHALL carry no `@pytest.mark.skipif` or `@skip_no_*`
  decorator for that dependency — the factory's internal skip is sufficient

## ADDED Requirements

### Requirement: Test functions SHALL NOT carry skip decorators redundant with injected fixtures or factories
A test function SHALL NOT carry a `@pytest.mark.skipif` or `@skip_no_*` decorator
for a condition that is already handled by a fixture the test injects or a
factory the test calls.  Skip guards SHALL exist at the point of dependency
(fixture or factory body), not repeated at every consumer.

`skip_no_*` module-level markers remain the correct approach when a test calls an
optional-import API directly and inline, with no intervening fixture or factory —
for example, tests that call `.to_legacy()` or `.from_legacy()` methods without
obtaining the legacy object from a factory.

#### Scenario: Decorator removed when fixture embeds the skip
- **WHEN** a test injects a session-scoped fixture that calls `pytest.skip()` when
  an optional dependency is absent
- **THEN** the test function SHALL carry no `@skip_no_*` decorator for that
  dependency

#### Scenario: Decorator removed when factory embeds the skip
- **WHEN** a test calls a module-level factory that calls `pytest.skip()` when an
  optional dependency is absent
- **THEN** the test function SHALL carry no `@skip_no_*` decorator for that
  dependency

#### Scenario: Decorator kept when test calls legacy API directly
- **WHEN** a test function calls an optional-import API inline (e.g.
  `obj.to_legacy()`, `LegacyClass(...)`) without going through a factory or
  fixture that already embeds the skip
- **THEN** the test function SHALL carry a `@skip_no_*` decorator (or equivalent
  `pytestmark`) so it is skipped when the dependency is absent
