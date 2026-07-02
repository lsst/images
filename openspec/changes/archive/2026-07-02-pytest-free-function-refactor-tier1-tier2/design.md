## Context

All 41 test files in `tests/` currently subclass `unittest.TestCase`. The full
suite runs under pytest, so both styles coexist transparently. This change
targets a well-scoped batch of 9 files (Tier 1 and Tier 2) that have no
external testdata dependencies, no roundtrip-mixin inheritance, and no Butler
integration — making them straightforward to convert mechanically.

The conversion is purely cosmetic from pytest's perspective: behavior and
coverage are unchanged.

## Goals / Non-Goals

**Goals:**
- Replace `unittest.TestCase` subclasses with top-level `def test_*()` functions
  in the 9 targeted files.
- Replace `setUp` methods with `@pytest.fixture` functions (or inline locals).
- Replace all `self.assertXxx` calls with plain `assert` expressions.
- Replace `@unittest.skipUnless` / `@unittest.skipIf` with `@pytest.mark.skipif`.
- Remove `import unittest` and `if __name__ == "__main__": unittest.main()` from
  converted files.
- Pass the full test suite (`pytest -r a -v -n 3`) with no regressions.

**Non-Goals:**
- Converting any test file that uses roundtrip mixins, external testdata, or
  Butler integration (those are a separate batch).
- Changing test logic, coverage, or adding new tests.
- Introducing `conftest.py` shared fixtures (fixtures stay file-local for now).
- Converting `test_schema_versioning.py`'s `ArchiveTreeClassInvariantsTestCase`
  `with self.subTest(...)` loop to `@pytest.mark.parametrize` — preserving it as
  a plain loop with `assert` is sufficient and simpler.

## Decisions

### D1: Fixture scope — file-local `@pytest.fixture`, not `conftest.py`

Each converted file defines its own fixtures. No `conftest.py` is created or
modified.

**Rationale**: The setUp objects are small and file-specific (e.g.,
`HdsNameShrinker`, a `Polygon`, an `rng`). Sharing them across files would
couple tests unnecessarily. Keeping fixtures local preserves the current
isolation model.

**Alternative considered**: A shared `conftest.py` with common numpy `rng`
fixtures. Rejected — the seeds differ between files, so sharing would require
parametrising the seed anyway, adding complexity for no gain.

### D2: `setUp` → fixture granularity

For test classes where *every* test method uses `self.X`, `X` becomes a
fixture injected by name. For the rare case where only one test uses the
object, it is constructed inline in the test function.

**Rationale**: Minimises the number of fixtures introduced while keeping
individual tests self-contained.

### D3: `self.assertRaises` → `pytest.raises`

Used as a context manager: `with pytest.raises(SomeError):`. The `as ctx` form
is used only when the test inspects `str(ctx.exception)`, matching the existing
`assertRaises` usage.

**Rationale**: Direct mechanical translation; no behaviour change.

### D4: `@unittest.skipUnless(HAVE_H5PY, ...)` → `@pytest.mark.skipif`

The existing module-level `HAVE_H5PY` bool is retained. Class-level skips
become `@pytest.mark.skipif(not HAVE_H5PY, reason="h5py is not installed")` on
the group of free-functions (via a class-less marker or per-function decorator).

Since pytest does not support class-level `pytestmark` without a class, the
cleanest approach for files where entire classes are h5py-gated is to apply
`@pytest.mark.skipif` to each function in that logical group, or to keep a thin
class only for the marker — but the simplest mechanical translation is to
decorate each function individually.

**Rationale**: Consistent with how the other already-pytest test files in the
ecosystem handle optional-dep skips.

### D5: `self.maxDiff = None` — drop it

pytest's assertion rewriting already shows full diffs unconditionally. The
`maxDiff` attribute is only meaningful to `unittest.TestCase`. Simply remove it.

### D6: `with self.subTest(mode=mode)` in `test_json_schema.py`

The two `subTest` loops (over `"validation"` and `"serialization"`) are
unrolled into `@pytest.mark.parametrize("mode", ["validation", "serialization"])`.

**Rationale**: `parametrize` is the idiomatic pytest equivalent and produces
better output — each mode appears as a separate test item.

## Risks / Trade-offs

- **[Risk] Silent behavioural change from assert translation** → Mitigation:
  Run the full suite before and after each file conversion; any difference in
  pass/fail is immediately visible.

- **[Risk] `pytest.raises` without `match=` loses message-content assertions**
  → Mitigation: Where the original used `assertRaises` with an `as ctx` block
  checking `str(ctx.exception)`, the pytest version uses
  `pytest.raises(Exc, match=...)` or the `as exc_info` form to preserve the
  assertion.

- **[Trade-off] Per-function `@pytest.mark.skipif` is verbose for large
  h5py-gated groups** → Accepted. The alternative (keeping a wrapper class)
  defeats the purpose of the refactor. The verbosity is bounded; `test_ndf_common.py`
  has only two logical groups.
