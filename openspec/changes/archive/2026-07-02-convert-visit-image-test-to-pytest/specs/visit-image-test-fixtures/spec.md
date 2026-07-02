## ADDED Requirements

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
