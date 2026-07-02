## Why

`legacy_dataset_context` and `legacy_nJy_dataset_context` in `tests/test_visit_image.py` return `dict[str, Any]`, which silences type-checking on all field accesses and forces every consumer to use string-key subscript syntax. Converting these to a typed dataclass eliminates the `Any` escape hatch, enables mypy to catch typos and wrong-type uses, and removes the now-redundant `legacy_visit_image` fixture (whose sole role is to carry one field that can live on the context object instead).

## What Changes

- Introduce `_LegacyDatasetContext` dataclass in `tests/test_visit_image.py` with fields: `filename`, `plane_map`, `unit`, `storage_class`, `read_cls`, `legacy_exposure`, and `visit_image`.
- Change `legacy_dataset_context` fixture return type from `dict[str, Any]` to `_LegacyDatasetContext`; populate `visit_image` (previously produced by the separate `legacy_visit_image` fixture) inside the fixture body.
- Change `legacy_nJy_dataset_context` fixture return type from `dict[str, Any]` to `_LegacyDatasetContext`; it already populates `visit_image`, so it maps directly onto the same dataclass.
- Delete the `legacy_visit_image` fixture (now redundant).
- Update all eight consumers (`test_legacy_errors`, `test_component_reads`, `test_legacy_obs_info`, `test_read_legacy_headers`, `test_from_legacy_headers`, `test_rewrite`, `test_butler_converters`, `test_aperture_corrections_to_legacy`) to use attribute access (`ctx.visit_image`, `ctx.filename`, …) and drop the now-unnecessary `legacy_visit_image` parameter where present.
- Add a `try/except ImportError` guard for `lsst.afw.image.Exposure` so the dataclass field type annotation works when the optional Rubin stack is absent (following the pattern already used in `test_masked_image.py`).

## Capabilities

### New Capabilities

- `legacy-dataset-context-dataclass`: Typed dataclass replacing `dict[str, Any]` fixture returns in `tests/test_visit_image.py`, covering both legacy-dataset context fixtures and consolidating the pre-read `VisitImage` into the context object.

### Modified Capabilities

## Impact

- `tests/test_visit_image.py` only — no production code changes.
- `from typing import Any` import may become unused and should be removed if so.
- `legacy_visit_image` fixture is deleted; any future downstream test file that imports it would break (none currently do).
