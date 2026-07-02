### Requirement: visit_image_components fixture provides shared construction objects
A session-scoped pytest fixture named `visit_image_components` SHALL be defined
in `tests/test_visit_image.py` and SHALL return a `dict` containing the ten
shared objects needed to construct `VisitImage` instances in the converted
tests: `mask_schema`, `obs_info`, `summary_stats`, `gaussian_psf`,
`aperture_corrections`, `detector`, `image`, `variance`, `polygon`, and
`sky_projection`.

#### Scenario: Fixture is session-scoped
- **WHEN** pytest collects `tests/test_visit_image.py`
- **THEN** `visit_image_components` SHALL be decorated with
  `@pytest.fixture(scope="session")`

#### Scenario: Dict contains all required keys
- **WHEN** `visit_image_components` is evaluated
- **THEN** the returned dict SHALL contain exactly the keys `mask_schema`,
  `obs_info`, `summary_stats`, `gaussian_psf`, `aperture_corrections`,
  `detector`, `image`, `variance`, `polygon`, and `sky_projection`

#### Scenario: rng and det_frame are not exposed
- **WHEN** `visit_image_components` is evaluated
- **THEN** the returned dict SHALL NOT contain keys `rng` or `det_frame`

### Requirement: make_visit_image is a module-level factory for a fully-populated VisitImage
A module-level function `make_visit_image` SHALL be defined in
`tests/test_visit_image.py`, SHALL accept a single `dict` argument (the
`visit_image_components` dict), and SHALL return a freshly constructed
`VisitImage` with all optional fields populated: `variance`, `bounds` (a
`Polygon`), `aperture_corrections`, `summary_stats`, a `ChebyshevField`
background entry named `"standard"` (marked as subtracted), and
`_opaque_metadata` set to a `FitsOpaqueMetadata` instance carrying a primary
header with `PLATFORM` and (stripped) `LSST BUTLER ID` keys.

#### Scenario: Returns a fresh instance on every call
- **WHEN** `make_visit_image(c)` is called twice with the same components dict
- **THEN** the two returned objects SHALL NOT be the same object

#### Scenario: Background is present and marked as subtracted
- **WHEN** `make_visit_image(c)` is called
- **THEN** the returned object's `backgrounds.keys()` SHALL equal `{"standard"}`
  and `backgrounds.subtracted.name` SHALL equal `"standard"`

#### Scenario: Opaque metadata carries primary header keys
- **WHEN** `make_visit_image(c)` is called
- **THEN** `_opaque_metadata.headers[ExtensionKey()]["PLATFORM"]` SHALL equal
  `"lsstcam"`

### Requirement: make_simplest_visit_image is a module-level factory for a minimal VisitImage
A module-level function `make_simplest_visit_image` SHALL be defined in
`tests/test_visit_image.py`, SHALL accept a single `dict` argument (the
`visit_image_components` dict), and SHALL return a freshly constructed
`VisitImage` with only the required arguments: `image`, `psf`, `mask_schema`,
`sky_projection`, `detector`, `obs_info`, and `band`.  No `variance`, `bounds`,
`aperture_corrections`, or `summary_stats` SHALL be passed, so `variance`
defaults to a unit-fill array.

#### Scenario: Returns a fresh instance on every call
- **WHEN** `make_simplest_visit_image(c)` is called twice with the same
  components dict
- **THEN** the two returned objects SHALL NOT be the same object

#### Scenario: Default variance is unit-fill
- **WHEN** `make_simplest_visit_image(c)` is called
- **THEN** the returned object's `variance.array[0, 0]` SHALL equal `1.0`

### Requirement: _make_sum_background_visit_image is a module-level helper
A module-level function `_make_sum_background_visit_image` SHALL be defined in
`tests/test_visit_image.py`, SHALL accept a single `dict` argument (the
`visit_image_components` dict), and SHALL return a freshly constructed
`VisitImage` whose subtracted background is a `SumField` of two `SplineField`
operands.  It SHALL use its own local `rng = np.random.default_rng(42)` and
SHALL derive the bounding box from the passed `visit_image` argument's
`image.sky_projection.pixel_frame.bbox`.

#### Scenario: Returns a VisitImage with a SumField background
- **WHEN** `_make_sum_background_visit_image(c, vi)` is called with valid
  components dict `c` and a `VisitImage` `vi`
- **THEN** the returned object SHALL be a `VisitImage` whose
  `backgrounds.subtracted.field` is a `SumField` with exactly two `SplineField`
  operands

#### Scenario: Uses an independent local rng
- **WHEN** `_make_sum_background_visit_image` is called multiple times
- **THEN** each call SHALL produce the same `SplineField` coefficients
  (because the local rng is re-seeded to 42 each call)

### Requirement: _LegacyDatasetContext dataclass holds all legacy fixture fields
A module-private `@dataclasses.dataclass` named `_LegacyDatasetContext` SHALL be
defined in `tests/test_visit_image.py` with the following fields, in order:
`filename: str`, `plane_map: dict[str, MaskPlane]`, `unit: astropy.units.Unit`,
`storage_class: str`, `read_cls: type[VisitImage]`, `legacy_exposure: LegacyExposure`
(where `LegacyExposure` is a module-level alias for `lsst.afw.image.Exposure` guarded
by `try/except ImportError` falling back to `Any`), and `visit_image: VisitImage`.

#### Scenario: Dataclass fields are accessible as attributes
- **WHEN** a `_LegacyDatasetContext` instance is constructed
- **THEN** each field SHALL be accessible via attribute access (e.g. `ctx.filename`,
  `ctx.visit_image`) without raising `AttributeError`

#### Scenario: Optional-dependency alias falls back to Any
- **WHEN** `lsst.afw.image` cannot be imported
- **THEN** `LegacyExposure` SHALL resolve to `Any` via the `type LegacyExposure = Any`
  statement, allowing the dataclass to be defined and annotated without error

### Requirement: legacy_dataset_context fixture returns _LegacyDatasetContext
The `legacy_dataset_context` fixture SHALL return `_LegacyDatasetContext` (not
`dict[str, Any]`). It SHALL populate `legacy_exposure` by reading the file with
`ExposureFitsReader`, and SHALL populate `visit_image` by calling
`read_cls.read_legacy(filename, preserve_quantization=True, plane_map=plane_map)`.

#### Scenario: visit_image is pre-read inside the fixture
- **WHEN** `legacy_dataset_context` is evaluated
- **THEN** the returned object's `visit_image` field SHALL be a `VisitImage` (or
  `DifferenceImage`) instance, read with `preserve_quantization=True`

#### Scenario: Fixture skips when lsst.afw.image is absent
- **WHEN** `lsst.afw.image` cannot be imported
- **THEN** the fixture SHALL call `pytest.skip(...)` so dependent tests are skipped

### Requirement: legacy_nJy_dataset_context fixture returns _LegacyDatasetContext
The `legacy_nJy_dataset_context` fixture SHALL return `_LegacyDatasetContext` (not
`dict[str, Any]`), using the same field population logic as `legacy_dataset_context`.

#### Scenario: visit_image is pre-read inside the nJy fixture
- **WHEN** `legacy_nJy_dataset_context` is evaluated
- **THEN** the returned object's `visit_image` field SHALL be a `VisitImage` instance,
  read with `preserve_quantization=True`

### Requirement: legacy_visit_image fixture is removed
The `legacy_visit_image` fixture SHALL NOT exist in `tests/test_visit_image.py`.
Tests that previously depended on it SHALL instead use `ctx.visit_image` from the
`legacy_dataset_context` fixture.

#### Scenario: No legacy_visit_image fixture in module
- **WHEN** pytest collects `tests/test_visit_image.py`
- **THEN** no fixture named `legacy_visit_image` SHALL be registered

### Requirement: All legacy context consumers use attribute access
Every test function and fixture that previously accepted `legacy_dataset_context:
dict[str, Any]` or `legacy_visit_image: VisitImage` SHALL be updated to accept
`legacy_dataset_context: _LegacyDatasetContext` and access fields via attributes
(`ctx.filename`, `ctx.visit_image`, etc.). The `legacy_visit_image` parameter SHALL
be removed from all function signatures.

#### Scenario: Tests that received both fixtures now receive only legacy_dataset_context
- **WHEN** `test_legacy_obs_info`, `test_read_legacy_headers`, `test_from_legacy_headers`,
  `test_rewrite`, and `test_butler_converters` are collected
- **THEN** none of them SHALL have `legacy_visit_image` in their parameter list

#### Scenario: test_aperture_corrections_to_legacy receives legacy_dataset_context
- **WHEN** `test_aperture_corrections_to_legacy` is collected
- **THEN** it SHALL accept `legacy_dataset_context: _LegacyDatasetContext` and use
  `legacy_dataset_context.visit_image` in place of the old `legacy_visit_image` argument
