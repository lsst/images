## ADDED Requirements

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
