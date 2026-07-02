## ADDED Requirements

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
