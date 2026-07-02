## ADDED Requirements

### Requirement: Shared state expensive or logically global uses session-scoped fixtures
When a test file's shared objects are expensive to construct (e.g. require file
I/O), those objects SHALL be provided by `@pytest.fixture(scope="session")`
fixtures.

#### Scenario: File-read object is session-scoped
- **WHEN** a fixture reads a file from disk (e.g. `read(path, Type)`) and the
  resulting object is used by multiple test functions without mutation
- **THEN** that fixture SHALL carry `scope="session"`

### Requirement: Cheap-to-construct objects are provided by module-level factories
Objects that are cheap to construct (i.e. no file I/O, no network, no
significant computation) and that tests may benefit from owning independently
SHALL be provided by module-level factory functions rather than session-scoped
fixtures.  Tests call the factory directly to obtain a fresh instance.

#### Scenario: Cheap object factory is a plain function, not a fixture
- **WHEN** a test needs a `VisitImage` or similarly cheap object
- **THEN** it SHALL call a module-level factory function (e.g.
  `make_visit_image(c)`) rather than receiving a shared instance via a
  session-scoped fixture

#### Scenario: Factory returns a fresh instance on every call
- **WHEN** a factory function is called twice
- **THEN** the two returned objects SHALL NOT be the same object (`is` check
  fails), ensuring test isolation

### Requirement: Fixture dependencies are declared via function parameters
A fixture that depends on another fixture SHALL declare that dependency by
accepting the dependency fixture's name as a function parameter, not by
constructing the dependency inline.

#### Scenario: Dependent fixture lists its dependency as a parameter
- **WHEN** a fixture requires objects from `visit_image_components`
- **THEN** it SHALL declare `visit_image_components` as a parameter and SHALL
  NOT re-construct those objects itself

### Requirement: Local rng instances are used for per-call randomness
Helper functions or fixtures that require random data but whose specific values
do not affect test correctness SHALL create a local `np.random.default_rng`
instance with an explicit seed rather than consuming from a shared fixture rng.

#### Scenario: Helper function uses its own seeded rng
- **WHEN** a module-level helper function needs random arrays
- **THEN** it SHALL call `rng = np.random.default_rng(<seed>)` at the start of
  its body and SHALL NOT accept an `rng` argument from outside
