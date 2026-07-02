## Context

`tests/test_visit_image.py` has been partially converted in a prior change: the
`VisitImageTestCase` class was replaced by pytest free-functions and three
session-scoped fixtures (`visit_image_components`, and the factory functions
`make_visit_image` / `make_simplest_visit_image`).  The remaining legacy section
consists of:

- `VisitImageLegacyTestMixin` — 8 test methods and 1 helper method, all using
  `self.*` attributes set by `setUpClass`.
- `VisitImageLegacyTestCase` — loads `visit_image.fits` (unit `u.nJy`); adds
  `test_convert_unit`.
- `PreliminaryVisitImageLegacyTestCase` — loads `preliminary_visit_image.fits`
  (unit `u.electron`).
- `DifferenceImageLegacyTestCase` — loads `difference_image.fits` (unit
  `u.nJy`; `storage_class = "DifferenceImage"`).

All three concrete classes are guarded by
`@unittest.skipUnless(EXTERNAL_DATA_DIR is not None, ...)`.

An explore-mode experiment confirmed that `test_convert_unit` runs cleanly
against `DifferenceImage` data with zero code changes, making it a good
candidate to parametrize across both `nJy`-unit datasets.

The project `pytest-test-style` spec and `visit-image-test-fixtures` spec
already cover all requirements; no spec additions are needed.

## Goals / Non-Goals

**Goals:**

- Replace `VisitImageLegacyTestMixin` and its three `unittest.TestCase`
  subclasses with pytest free-functions and two session-scoped parametrized
  fixtures.
- Make each test's fixture dependencies explicit (tests that only need the
  loaded file, vs. those that also need the constructed `VisitImage` object).
- Extend `test_convert_unit` to run on `DifferenceImage` as well as
  `VisitImage`.
- Remove `import unittest`, `ClassVar`, and `if __name__ == "__main__"` once
  all `unittest.TestCase` subclasses are gone.
- Preserve all existing test logic and coverage exactly; add only the new
  `DifferenceImage` coverage for `test_convert_unit`.

**Non-Goals:**

- Modifying any test logic or adding new assertions beyond promoting
  `test_convert_unit` to two datasets.
- Extracting fixtures into `conftest.py` or `lsst.images.tests`.
- Converting `test_convert_unit` to also run on `preliminary_visit_image`
  (that dataset is in `u.electron`; the test is structured around `u.nJy`).

## Decisions

### D1 — Two parametrized session-scoped fixtures, not three

**Decision:** Provide two session-scoped fixtures:

1. `legacy_dataset_context` — parametrized over
   `["visit_image", "preliminary_visit_image", "difference_image"]`; yields a
   `dict` containing `filename`, `legacy_exposure`, `plane_map`, `unit`,
   `storage_class`, and `read_cls` (either `VisitImage` or `DifferenceImage`).

2. `legacy_visit_image` — depends on `legacy_dataset_context`; calls
   `context["read_cls"].read_legacy(filename, preserve_quantization=True,
   plane_map=plane_map)` and yields the result.

Tests declare only the fixture(s) they actually need: tests that only operate
on the file (e.g., `test_legacy_errors`, `test_component_reads`) take
`legacy_dataset_context`; tests that need the constructed object take
`legacy_visit_image` (which carries `legacy_dataset_context` along implicitly).

**Rationale:** Two fixtures mirror the two real I/O operations (read the raw
FITS with afw; read it as a new `VisitImage`/`DifferenceImage`).  Finer
granularity (e.g., separate `legacy_plane_map`, `legacy_unit` fixtures) would
be over-engineering for objects that cost nothing to bundle.

**Alternative considered:** Three explicit named fixtures (one per dataset, no
parametrize).  Rejected because it triples the fixture boilerplate and makes it
harder to add a fourth dataset later.

### D2 — `test_convert_unit` parametrized over `["visit_image", "difference_image"]` only

**Decision:** `test_convert_unit` uses `@pytest.mark.parametrize` with an
explicit list of two param IDs, independent of the `legacy_dataset_context`
fixture.  It receives a dedicated fixture `legacy_nJy_context` (or takes the
params via an `indirect`-style approach using a separate
`legacy_convert_unit_context` fixture scoped to those two datasets).

The cleanest implementation: define a separate session fixture
`legacy_nJy_dataset_context` parametrized over only
`["visit_image", "difference_image"]` and have `test_convert_unit` use it.
This avoids any `pytest.skip` inside the test body and keeps the test scope
declarative.

**Rationale:** `preliminary_visit_image` uses `u.electron`; the test
unconditionally calls `convert_unit(u.nJy, copy=False)` as a no-op, which
would fail on an electron-unit image.  Explicitly scoping the fixture to `nJy`
datasets is cleaner than a runtime `pytest.skip`.

**Alternative considered:** A single `legacy_dataset_context` with a
`pytest.skip(...)` call inside `test_convert_unit` when `unit != u.nJy`.
Rejected because it hides the scope from the test signature and is harder to
understand at a glance.

### D3 — `_check_legacy_obs_info` becomes a module-level helper

**Decision:** The mixin helper method `check_legacy_obs_info` becomes a
module-level function `_check_legacy_obs_info(obs_info)` using plain `assert`
statements.

**Rationale:** It is called in two test functions.  A module-level helper with
an explicit argument is the standard pattern established by the prior conversion
(cf. `_check_sum_background_round_trip`).

### D4 — `with self.subTest()` blocks in `test_rewrite` become plain assertions

**Decision:** The two `with self.subTest():` blocks in `test_rewrite` are
removed; the assertions inside become sequential plain code.

**Rationale:** The subtests are sequential phases of one round-trip (each
depends on state from the previous phase), not independent iterations.  pytest's
assertion rewriting provides sufficient failure context.  This matches the
decision made for the same pattern in the prior conversion (D5 of the archived
change).

### D5 — Skip guard moves to `@pytest.mark.skipif` per function

**Decision:** The class-level `@unittest.skipUnless(EXTERNAL_DATA_DIR is not
None, ...)` decorator is replaced with `@pytest.mark.skipif(EXTERNAL_DATA_DIR
is None, reason=...)` on each test function in the legacy group.

**Rationale:** Consistent with the `pytest-test-style` spec requirement.
A module-level `pytestmark` is an alternative but is less explicit; per-function
decorators match the pattern used elsewhere in the converted file.

### D6 — `import unittest` and `ClassVar` removed after conversion

**Decision:** Once the last `unittest.TestCase` subclass is gone, `import
unittest`, `ClassVar` from `typing`, and `if __name__ == "__main__":
unittest.main()` are removed.

**Rationale:** Nothing else in the file needs them.  `Any` from `typing` is
retained because it appears in the existing `visit_image_components` fixture
signature.

## Risks / Trade-offs

**Session-scoped `legacy_visit_image` may accumulate state if a test mutates
it** → Inspection confirms no test mutates the shared object: `test_rewrite`
and `test_butler_converters` operate via `RoundtripFits`/`TemporaryButler`
context managers on the shared image without modifying it; `test_convert_unit`
calls `.copy()` explicitly before any mutation.

**`compare_photo_calib_to_legacy` in `test_convert_unit` uses
`self.legacy_exposure.getPhotoCalib()`** → Under parametrize, this becomes
`legacy_dataset_context["legacy_exposure"].getPhotoCalib()`.  The difference
image has its own `PhotoCalib` that is distinct from the visit image's; the
`visit_summary.find(detector).getPhotoCalib()` comparison is self-consistent
for each dataset independently, so no cross-contamination occurs.

**Parametrized test IDs** → pytest embeds the param ID in the node name, e.g.,
`test_legacy_errors[visit_image]`, `test_legacy_errors[preliminary_visit_image]`,
`test_legacy_errors[difference_image]`.  This is more informative than the
previous unnamed `TestCase` subclasses.

## Open Questions

None.
