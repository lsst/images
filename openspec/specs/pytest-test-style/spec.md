### Requirement: Test files use pytest free-functions instead of unittest.TestCase
Test files in `tests/` SHALL be written as top-level `def test_*()` functions
rather than methods on `unittest.TestCase` subclasses, except where a class is
strictly required (e.g., to hold a `pytestmark` for a group of skip-guarded tests).

#### Scenario: No TestCase subclasses in converted files
- **WHEN** a test file is converted under this change
- **THEN** it SHALL contain no subclasses of `unittest.TestCase`

#### Scenario: No unittest import in converted files
- **WHEN** a test file is converted under this change
- **THEN** it SHALL NOT import `unittest` and SHALL NOT contain
  `if __name__ == "__main__": unittest.main()`

### Requirement: setUp methods become pytest fixtures
Where a `unittest.TestCase.setUp` method created shared objects, those objects
SHALL be provided as `@pytest.fixture` functions in the same file, injected by
parameter name into the test functions that need them.

#### Scenario: Shared object becomes a named fixture
- **WHEN** multiple test functions in a file need the same constructed object
- **THEN** that object SHALL be returned by a `@pytest.fixture` function
  declared in the same file

#### Scenario: Single-use object stays inline
- **WHEN** only one test function needs a particular object
- **THEN** it MAY be constructed inline in that function rather than as a fixture

### Requirement: Assertions use plain assert statements
All assertion calls inherited from `unittest.TestCase` (e.g. `assertEqual`,
`assertTrue`, `assertRaises`) SHALL be replaced with plain Python `assert`
statements or `pytest.raises` context managers.

#### Scenario: Equality assertion translation
- **WHEN** the original test uses `self.assertEqual(a, b)`
- **THEN** the converted test SHALL use `assert a == b`

#### Scenario: Exception assertion translation
- **WHEN** the original test uses `with self.assertRaises(Exc)`
- **THEN** the converted test SHALL use `with pytest.raises(Exc)`

#### Scenario: Exception message assertion preserved
- **WHEN** the original test inspects `str(ctx.exception)` after `assertRaises`
- **THEN** the converted test SHALL use `pytest.raises(Exc)` with either
  `match=` or the `as exc_info` form to preserve that assertion

### Requirement: Optional-dependency skips use pytest.mark.skipif
Skip decorators based on optional imports SHALL use `@pytest.mark.skipif`
rather than `@unittest.skipUnless` or `@unittest.skipIf`.

#### Scenario: Class-level optional-dep skip becomes per-function decorator
- **WHEN** the original test class has `@unittest.skipUnless(HAVE_X, reason)`
- **THEN** each converted test function in that logical group SHALL carry
  `@pytest.mark.skipif(not HAVE_X, reason=reason)`

### Requirement: subTest loops use parametrize where natural
`with self.subTest(key=val)` loops SHALL be replaced with
`@pytest.mark.parametrize` when the loop variable maps cleanly to a single
parameter; otherwise they may remain as plain Python loops with `assert`.

#### Scenario: Mode-parametrised subTest becomes parametrize
- **WHEN** the original test iterates over a fixed list of modes with `subTest`
- **THEN** the converted test SHALL use `@pytest.mark.parametrize` over that list

### Requirement: No comment-section headers between test functions
Test files SHALL NOT use banner or divider comments (e.g. lines of dashes,
`# ---…---` rules, or `# Section name` headings) to group test functions.
Function names and docstrings are the only navigation aid needed.

#### Scenario: No divider comments in converted files
- **WHEN** a test file is converted or written
- **THEN** it SHALL NOT contain comment lines whose purpose is solely to
  visually separate groups of test functions (e.g. `# ----`, `# XY tests`,
  `# ---------------------------------------------------------------------------`)

### Requirement: Docstrings use imperative mood
Every test function and fixture in a converted file SHALL have a docstring
whose first sentence opens with an imperative-mood verb (e.g. ``Verify``,
``Test``, ``Return``, ``Assert``).  Declarative sentences that describe what
the code *does* rather than commanding it to act are not acceptable.

#### Scenario: Test function docstring is imperative
- **WHEN** a test function is written or converted
- **THEN** its docstring SHALL begin with an imperative verb such as
  ``Verify``, ``Test``, or ``Assert``

#### Scenario: Fixture docstring is imperative
- **WHEN** a fixture function is written or converted
- **THEN** its docstring SHALL begin with an imperative verb, typically
  ``Return``

#### Scenario: Declarative opening is not acceptable
- **WHEN** a docstring opens with a noun phrase or third-person verb
  (e.g. ``"Repeated names get…"``, ``"NdfPointerModel round-trips…"``,
  ``"A freshly-written FITS carries…"``)
- **THEN** it SHALL be rewritten to open with an imperative verb
  (e.g. ``"Verify repeated names get…"``,
  ``"Verify NdfPointerModel round-trips…"``,
  ``"Verify a freshly-written FITS carries…"``)

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

### Requirement: Shared state expensive or logically global uses session-scoped fixtures
When a test file's shared objects are expensive to construct (e.g. require file
I/O), those objects SHALL be provided by `@pytest.fixture(scope="session")`
fixtures.

#### Scenario: File-read object is session-scoped
- **WHEN** a fixture reads a file from disk (e.g. `read(path, Type)`) and the
  resulting object is used by multiple test functions without mutation
- **THEN** that fixture SHALL carry `scope="session"`

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

### Requirement: get(component=...) calls SHALL NOT share a test function with always-run assertions
A test function that calls `roundtrip.get(component=...)` or
`roundtrip.get(storageClass=...)` SHALL NOT contain assertions that must always
execute regardless of butler availability.  When `lsst.daf.butler` is absent,
`get(component=...)` calls `pytest.skip()`, which exits the entire test function.
Butler-free assertions that must always run SHALL live in a separate test function
that contains no `get(component=...)` or `get(storageClass=...)` calls.

Both the component-read function and the butler-free function MAY pass
`storage_class=...` to `RoundtripX`; `storage_class` alone does not cause a skip
and its presence ensures full butler-path coverage when the butler is available.

#### Scenario: Always-run assertions are in a component-read-free function
- **WHEN** a test needs to assert on `roundtrip.result`, FITS headers, or other
  properties that do not require named component reads
- **THEN** those assertions SHALL be placed in a test function that contains no
  `roundtrip.get(component=...)` calls, so that `pytest.skip()` is never triggered

#### Scenario: Component reads are isolated in their own function
- **WHEN** a test needs to call `roundtrip.get(component=...)` or
  `roundtrip.get(storageClass=...)`
- **THEN** those calls SHALL be placed in a dedicated test function containing
  only component-read assertions; when butler is absent, the first
  `get(component=...)` call will `pytest.skip()` the entire function cleanly,
  wasting no coverage that should always run

#### Scenario: No always-run assertions are silently skipped by component reads
- **WHEN** `lsst.daf.butler` is not installed and a test function calls
  `roundtrip.get(component=...)`
- **THEN** only the assertions in that specific test function SHALL be skipped;
  assertions in the companion component-read-free function SHALL execute normally
