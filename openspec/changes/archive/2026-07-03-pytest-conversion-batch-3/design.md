## Context

Prior batches converted ~18 test files to pytest free-function style (commits
`f436e62`, `8d9a1fb`, `de858cd`).  The same `openspec/specs/pytest-test-style/`
spec governs all conversions.  This batch covers 14 remaining files that range
from trivially mechanical to moderate complexity.  No production code changes;
the risk envelope is purely "did we preserve every test's intent?"

## Goals / Non-Goals

**Goals:**
- Remove all `unittest.TestCase` subclasses from the 14 in-scope files.
- Apply the established pytest style (free functions, `@pytest.fixture`,
  `@pytest.mark.skipif`, bare `assert`, imperative docstrings, no banner
  comments, `@pytest.mark.parametrize` for enumerable subTest loops).
- Add two clarifying sub-rules to `openspec/specs/pytest-test-style/spec.md`:
  factory-function preference and `skip_no_*` naming.

**Non-Goals:**
- Converting the seven deferred files (`test_cameras`, `test_cell_coadd`,
  `test_from_hdu_list`, `test_cli`, `test_diagram`, `test_ndf_input_archive`,
  `test_ndf_output_archive`).
- Changing any test's logic, coverage, or external behavior.
- Altering production source files.

## Decisions

### setUp → factory function vs. fixture

**Decision:** Prefer module-level factory functions (plain `def make_X()`) over
`@pytest.fixture` when construction is cheap (no file I/O, no significant
computation).  Use `@pytest.fixture(scope="session")` only when construction
involves file I/O or a heavy legacy object load.

**Rationale:** Factory functions give each test a fresh independent instance
without pytest's fixture injection machinery, which is unnecessary overhead for
objects like a small `Image` or `MaskedImage`.  Session fixtures are appropriate
for objects like a loaded `VisitImage` from disk in `test_serialization_reader`
or a legacy legacy object in `test_difference_image_extras` (setUpClass).

**Alternatives considered:** Fixture-for-everything (prior batch default) — fine
but heavier than necessary for trivial objects; leads to many single-parameter
functions where a `make_X()` call is clearer.

### Optional-dependency skip guards

**Decision:** Use a named module-level `pytestmark`-style skip constant:
```python
skip_no_h5py = pytest.mark.skipif(not HAVE_H5PY, reason="h5py is not installed")
skip_no_afw  = pytest.mark.skipif(DATA_DIR is None or not HAVE_AFW, reason="...")
```
Apply with `@skip_no_h5py` on each affected function.

**Rationale:** Avoids repeating the condition and message string on every
function; makes bulk-skipped groups visually obvious; mirrors how `pytestmark`
is used for class-level skips in already-converted files.

**Alternatives considered:** Inline `@pytest.mark.skipif(not HAVE_H5PY, ...)` on
every function — correct but verbose and error-prone to keep in sync.

### subTest inside `with RoundtripX(...):` blocks

**Decision:** Remove the `with self.subTest():` wrapper.  If the inner block
contains a `self.skipTest(...)`, replace with `pytest.skip(...)`.  If it
contains afw-import-guarded logic (the pattern in `test_image`, `test_mask`,
`test_masked_image`), extract that block into a separate helper function or a
separate test function decorated with `@skip_no_afw`.

**Rationale:** The `RoundtripX` context manager is a *resource* manager, not a
test isolation boundary.  `subTest` inside it was used either (a) to soft-skip
an optional assertion without failing the whole test, or (b) to provide a
per-assertion label.  In pytest, (a) is better expressed as `pytest.skip()`,
and (b) is unnecessary because bare `assert` already gives good tracebacks.
Extracting to a separate function is safe because `RoundtripX.__exit__` writes
`roundtrip.result` regardless of exceptions in the `with` body — the outer
test still sees the fully-written result.

**Alternatives considered:** Keep `subTest` via `unittest`-compat shim — adds
the `unittest` import back, defeating the purpose; not acceptable per spec.

### FixtureSweepTestCase → parametrize

**Decision:** `test_serialization_io.FixtureSweepTestCase.test_sweep` iterates
over JSON fixture files and calls `subTest(entry=entry)`.  Convert to
`@pytest.mark.parametrize("entry", sorted(EXPECTED_TYPES))` with a
`pytest.importorskip`-style skip for the piff fixture when piff is absent.

**Rationale:** Each fixture file is an independent test node; a failure in one
should not suppress the rest.  This is the textbook `parametrize` use-case.

### TransformTestCase helper method

**Decision:** `TransformTestCase.compare_to_legacy_camera` is a `@staticmethod`
that does not use `self`.  Convert to a module-level function.  The
`FrameSetTestHolder` and `FrameSetTestHolderModel` classes in the same file are
production serialization helpers, not test infrastructure; they remain
unchanged.

## Risks / Trade-offs

- **subTest extraction changes failure isolation**: removing `subTest` from
  `test_masked_image.test_fits_roundtrip` means the afw-read assertion failure
  will propagate and mark the outer test as failed rather than just the subTest.
  Mitigation: extract to a separate `test_fits_roundtrip_legacy_read` function
  guarded by `@skip_no_afw`, so the outer test is fully independent.

- **Factory function ownership**: tests that receive a shared object from a
  factory and mutate it (e.g. `test_mask` sets bits in the mask) must call the
  factory per-test, not once.  The factory approach naturally enforces this.
  Mitigation: review each factory call site to confirm it's called inside each
  test function, not at module scope.

- **Scope of session fixtures**: a session-scoped fixture loaded from disk is
  shared across all tests in the session.  Tests that mutate the object (e.g.
  call `.add_plane()`) must work on a `.copy()`.  Mitigation: audit each
  session-fixture consumer for mutation.

## Migration Plan

Each file is an independent atomic unit:
1. Rewrite file to pytest style.
2. Run `ruff check --fix` and `ruff format` on the file.
3. Run `pytest tests/<file> -v` to confirm pass/skip.
4. Commit.

No rollback strategy needed beyond `git revert`; no production code is touched.

## Open Questions

*(none — all decisions resolved during exploration)*

## Out of Scope / Future Work

### Retrofit session fixtures with pytest.skip guards to prior batches

During implementation of this batch we established that `pytest.skip()` called
inside a session-scoped fixture body propagates the skip to all consuming tests
— and that `@pytest.mark.skipif` on a fixture is explicitly prohibited by pytest
9+ (raises `PytestRemovedIn9Warning`, becomes an error in pytest 10).

The tests converted in **prior batches** (commits `f436e62`, `8d9a1fb`,
`de858cd`) may still contain:
- Factory functions that do file I/O on every call instead of once per session.
- `@pytest.mark.skipif` decorators on individual test functions where a shared
  session fixture with an internal `pytest.skip()` guard would be cleaner.

A follow-on pass should audit and retrofit those files.  The key files to
review are those in prior batches that have `TESTDATA_IMAGES_DIR` or `HAVE_AFW`
checks, particularly `test_visit_image.py`, `test_geom.py`, and
`test_schema_versioning.py`.  This is **not** urgent — the existing style works
correctly; this is a quality improvement for consistency and performance.
