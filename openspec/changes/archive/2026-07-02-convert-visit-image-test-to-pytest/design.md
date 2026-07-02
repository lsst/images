## Context

`tests/test_visit_image.py` contains `VisitImageTestCase`, a `unittest.TestCase`
subclass whose `setUpClass` builds ~10 shared objects in one monolithic block.
The tests that follow use these objects via `self.*` attributes without any
declaration of what each test actually needs.  The rest of the file (legacy
test classes) is left unchanged by this change.

The project spec (`pytest-test-style`) requires new and converted test files to
use pytest free-functions with `@pytest.fixture`, plain `assert` statements,
imperative docstrings, and `@pytest.mark.skipif` for optional-dependency guards.

## Goals / Non-Goals

**Goals:**

- Replace `VisitImageTestCase` with pytest free-functions and three
  session-scoped fixtures.
- Make each test's fixture dependencies explicit via function parameters.
- Eliminate `unittest.TestCase` assertions (`assertEqual`, `assertRaises`,
  `assertIsInstance`, etc.) from the converted functions.
- Preserve all existing test logic and coverage exactly.

**Non-Goals:**

- Converting the legacy test classes (`VisitImageLegacyTestMixin` and its
  three concrete subclasses) — deferred to a later change.
- Extracting fixture factories into `lsst.images.tests` — not warranted until
  downstream packages need them.
- Changing what is tested or adding new tests.

## Decisions

### D1 — Three session-scoped fixtures, medium granularity

**Decision:** Provide three fixtures — `visit_image_components` (dict),
`visit_image`, and `simplest_visit_image` — all at `scope="session"`.

**Rationale:** Session scope amortises construction cost across the whole test
run; the objects are immutable or only ever read by the tests that use them.
Medium granularity (one components dict + two canonical objects) is simpler
than a fixture per leaf object while still making dependencies legible from
each test's parameter list.

**Alternative considered:** One fat `visit_image_state` fixture returning a
namespace.  Rejected because several tests only need a single component (e.g.
`test_summary_stats` needs only `summary_stats`); a bundle fixture would
obscure that.

**Alternative considered:** Fine-grained fixtures (one per leaf object).
Rejected as unnecessarily verbose for test-internal objects that have no
reuse outside this file.

### D2 — `visit_image_components` returns a `dict`, not a dataclass or SimpleNamespace

**Decision:** `visit_image_components` returns a plain `dict[str, Any]`.

**Rationale:** Test-internal fixtures don't need mypy coverage (the 100% mypy
requirement applies to `python/lsst/images/`, not `tests/`).  A `dict` is
concise and avoids boilerplate.  Keys are string literals that read naturally
at call sites: `c["image"]`, `c["gaussian_psf"]`.

### D3 — No `rng` or `det_frame` in `visit_image_components`

**Decision:** `visit_image_components` does not expose `rng` or `det_frame`.

**Rationale:** `rng` is consumed during fixture construction (to build
`sky_projection`) and has no meaningful reuse at test time.  `test_basics`
needs a fresh `tract_proj` — it creates its own local `rng = np.random.default_rng(501)`
(a distinct seed to avoid any accidental coupling).  `_make_sum_background_visit_image`
uses its own local `rng = np.random.default_rng(42)`.  `det_frame` is a
construction intermediate; tests that need a bounding box derive it from
`visit_image.image.sky_projection.pixel_frame.bbox`, matching the existing
`_make_sum_background_visit_image` logic.

### D4 — `_make_sum_background_visit_image` and `_check_sum_background_round_trip` become module-level helpers

**Decision:** Both private helpers become module-level functions prefixed with
`_`, accepting the components dict and/or a `VisitImage` argument explicitly.

**Rationale:** They are not fixtures (they produce fresh objects each call or
perform assertions).  Module-level helpers with explicit arguments are
straightforward to call from test functions.

### D5 — `self.subTest()` in `test_read_write` becomes plain assertions

**Decision:** The `with self.subTest():` blocks in `test_read_write` are
removed; the assertions inside become plain sequential code.

**Rationale:** The subtests are sequential phases of one round-trip — each
depends on state established by the previous phase.  They are not independent
iterations.  pytest's assertion rewriting provides sufficient failure context
without subtest labelling.

### D6 — `unittest` import retained while legacy classes remain

**Decision:** The `import unittest` statement and `ClassVar` typing import are
kept until the legacy classes are converted.  The `if __name__ == "__main__"`
block is removed.

**Rationale:** The legacy classes still use `unittest.TestCase` and
`@unittest.skipUnless`; removing the import would break them.

## Risks / Trade-offs

**Session-scoped `visit_image` may accumulate state if a test mutates it** →
Inspection of all test functions confirms none mutate the shared `visit_image`
or `simplest_visit_image`; `test_copy_and_slice` calls `.copy()` explicitly.
The `_make_sum_background_visit_image` helper constructs a fresh object each
call, so both sum-background tests are independent.

**`rng` seed change in `test_basics`** → The `tract_proj` in `test_basics` is
only used to trigger a `TypeError`; its actual sky-coordinate values do not
matter.  Any seed works.

## Open Questions

None.
