## Context

The test suite mixes two styles: nine files have already been converted to
pytest free functions; thirty-two files still use `unittest.TestCase`.  The
existing converted files (`test_legacy.py`, `test_polygon.py`,
`test_ndf_common.py`, `test_schema_versioning.py`, etc.) form the reference
style: top-level `def test_*()` functions, `@pytest.fixture` for shared
objects, `@pytest.mark.skipif` for optional-dependency guards, plain `assert`
statements, imperative docstrings, no banner comments.

This change converts the next ten files — selected because they have no complex
`setUpClass`, no Mixin inheritance, and require only straightforward mechanical
translation.

## Goals / Non-Goals

**Goals:**
- Convert all ten target files to pytest free-function style.
- Preserve identical test coverage and semantics (no test logic changes).
- Conform each file to `openspec/specs/pytest-test-style/spec.md`.
- Pass `pytest -r a -v -n 3` without failures or new skips.
- Pass `ruff check` and `ruff format` without errors.

**Non-Goals:**
- Converting any of the remaining 22 unconverted files.
- Adding new tests or changing what is being tested.
- Changing any production code.
- Modifying `conftest.py` (no new shared fixtures needed; all fixtures are
  file-local in the converted pattern).

## Decisions

### D1: Conversion is strictly mechanical — no logic changes

Each converted file must behave identically to the original.  The only
permitted changes are syntactic: removing `class`/`self`/`setUp`, rewriting
assertions, replacing skip decorators.  If a translation requires a logic
decision, it is an error, not a design choice.

*Alternative considered*: opportunistically improve tests during conversion.
Rejected — mixing refactor with functional change makes diffs hard to review
and risks introducing regressions.

### D2: `setUp` with a single attribute becomes a single `@pytest.fixture`

Where `setUp` sets `self.foo = Foo()`, the fixture is:

```python
@pytest.fixture
def foo() -> Foo:
    """Return a Foo instance for testing."""
    return Foo()
```

Each test that previously accessed `self.foo` gains `foo` as a parameter.

*Alternative*: inline the construction in every test.  Rejected when multiple
tests share the object, as it introduces duplication and diverges from the
pattern established by `test_legacy.py` and `test_polygon.py`.

### D3: `addCleanup` becomes `yield`-based fixture; fixtures compose via injection

`test_detached_archive.py`'s `ComponentProbeTestCase.setUp` creates a
`tempfile.TemporaryDirectory` and calls `self.addCleanup(tmp.cleanup)`.  The
converted fixture uses `yield` and delegates temp-directory cleanup to pytest's
built-in `tmp_path` fixture.  Where a fixture needs a `DetachedArchive`, it
receives the `detached_archive` fixture by injection rather than constructing
its own:

```python
@pytest.fixture
def component_probe_env(
    detached_archive: DetachedArchive, tmp_path: Path
) -> Iterator[tuple[str, object, DetachedArchive]]:
    """Return (tmpdir, visit_image, archive) for component-probe tests."""
    visit_image = read(os.path.join(DATA_DIR, "visit_image.json"))
    yield str(tmp_path), visit_image, detached_archive
```

This is the idiomatic pytest equivalent and follows the fixture-composition
pattern: fixtures express their own dependencies via parameters rather than
constructing them directly.

### D4: `subTest` loops — case-by-case, with iteration identity always preserved

`subTest` serves two distinct purposes: (a) continue-on-failure (a failed
inner block does not abort the outer loop) and (b) labelling each iteration
so failures are identifiable.  pytest provides different tools for each.
Every converted subTest loop MUST preserve the ability to identify which
iteration failed; continue-on-failure is preserved where it materially helps.

The analysis of the seven subTest usages across the ten files yields four
distinct patterns:

**Pattern A — `@pytest.mark.parametrize` (continue-on-failure + per-node
identity):** Used when the iterable is a module-level collection and
continue-on-failure genuinely matters (i.e., one bad item should not mask
others).

- `test_schema_v1_fixtures.py` both loops: `_FIXTURES` is an auto-discovered
  glob of ~18 JSON files.  Each file becomes a separate test node with a
  stable name derived from its stem:
  ```python
  @pytest.mark.parametrize("path", _FIXTURES, ids=lambda p: p.stem)
  def test_fixture_has_top_level_stamps(path: Path) -> None: ...
  ```
- `test_schema_v1_legacy_fixtures.py`: a 2-item hardcoded tuple where
  per-item skip is the key behavior.  Each name becomes a test node; the
  body calls `pytest.skip()` when the file is absent:
  ```python
  @pytest.mark.parametrize("name", ["visit_image_dp1.json", "visit_image_dp2.json"])
  def test_legacy_visit_image(name: str) -> None:
      path = LEGACY_DIR / name
      if not path.exists():
          pytest.skip(f"{path} not present")
      ...
  ```
- `test_detached_archive.py` `test_free_components`: 9 independent entries;
  each component is parametrized so a single failure does not hide others and
  can be re-run in isolation.
- `test_detached_archive.py` `test_pixel_components_need_file`: the 3-item
  pixel-component tuple is also parametrized — consistent treatment of all
  component-iteration loops in the same file, and each failure is individually
  identifiable.  The component lists are inlined directly into the
  `@pytest.mark.parametrize` call rather than stored as module-level constants.

**Pattern B — plain loop, iteration variable in traceback locals:** Used when
the collection is small, the failures are unlikely to be independent, or
restructuring for parametrize would require non-trivial setup refactoring.
pytest's assertion rewriting captures all local variables in the frame, so
`component`, `schema_name`, etc. appear automatically in the failure output.

- `test_serialization_registry.py` line 63: 3-item tuple inside a
  `try/finally` that restores global `_REGISTRY` state.  The `finally` runs
  regardless of whether the failure is a subTest soft-fail or a hard raise, so
  the continue-on-failure behavior is not needed for safety.  Plain loop; the
  `try/finally` structure is preserved unchanged.
- `test_fits_date_header.py` line 45: iterates over HDUs from an open
  `astropy.io.fits.open` file handle; parametrizing over HDU indices would
  require pre-opening the file or using indirect fixtures, both of which are
  disproportionate.  Plain loop; `index` and `extname` appear in traceback
  locals.

*Rejected alternative for Pattern B*: collect failures and call `pytest.fail()`
once at the end (the manual "accumulate errors" pattern).  Accepted as a last
resort but avoided here because the loop bodies are short, the collections are
small, and the pattern adds verbosity without clear benefit in these cases.

*Key invariant*: no subTest loop may be converted to a plain `assert` loop
without ensuring the iteration variable is available as a local that pytest
will display on failure.

### D5: `@unittest.skipUnless(HAVE_X, reason)` becomes a named mark applied per function

Files where the entire class is gated on `HAVE_H5PY` (all ndf tests) define a
named mark constant and apply it as a decorator to each test function:

```python
skip_no_h5py = pytest.mark.skipif(not HAVE_H5PY, reason="h5py is not installed")

@skip_no_h5py
def test_something() -> None: ...
```

This is preferred over a module-level `pytestmark` assignment because it is
explicit at each function, easier to grep, and avoids the special-case
`pytestmark` name which applies implicitly to everything in the module (making
it easy to accidentally skip helper functions or fixtures).  It is also
preferred over repeating the full `@pytest.mark.skipif(not HAVE_H5PY, ...)`
decorator because the named constant avoids repetition while remaining readable.

*Alternative considered*: module-level `pytestmark = pytest.mark.skipif(...)`.
Rejected in favour of the named-constant-per-function approach above.

### D6: Module-level helper functions stay at module level

`test_ndf_layout.py` has `_cls()`, `_hds_type()`, and `_hds_shape()` helper
functions that are not test functions.  These stay at module level unchanged.

### D7: Local class definitions inside test methods stay local

`test_serialization_registry.py` defines `_EntryPointTree` and
`_FakeEntryPoint` inside `test_entry_point_provider_loaded_on_miss`.  These
stay as local classes inside the converted free function.

## Risks / Trade-offs

- **Semantic drift during translation** → Mitigation: run the full test suite
  (`pytest -r a -v -n 3`) after each file conversion before moving to the
  next.
- **Fixture scope confusion** (`function` vs `session`)  → All new fixtures
  use the default `function` scope, matching the originals' per-test `setUp`
  behaviour.
- **subTest continue-on-failure lost for plain loops** → Accepted for small
  (≤3 item) collections where a single root cause typically explains all
  failures.  For larger collections (e.g. the 9 free components and 3 pixel
  components in `test_detached_archive.py`, the ~18-item `_FIXTURES` glob)
  continue-on-failure is preserved via `@pytest.mark.parametrize`.
- **`Roundtrip*` context managers and assertion failures** → `RoundtripFits`,
  `RoundtripJson`, and `RoundtripNdf` are `contextmanager`-backed; their
  `__exit__` runs cleanup when a `with` body raises, so test-body assertion
  failures do not leak temp files.  However, an `AssertionError` raised inside
  `__enter__` (from its own internal checks) would bypass `__exit__`.  None of
  the ten target files have `Roundtrip*` inside a `subTest` loop — the
  Roundtrip calls are outside any loops in all affected tests.

## Open Questions

None — conversion criteria and per-file approach are fully determined by the
exploration analysis and the existing spec.
