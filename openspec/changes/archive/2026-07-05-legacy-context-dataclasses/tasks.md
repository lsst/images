## 1. Add dataclass and optional-dependency alias

- [x] 1.1 Add `import dataclasses` to the imports in `tests/test_visit_image.py`
- [x] 1.2 Add a `try/except ImportError` block to define `LegacyExposure` as `lsst.afw.image.Exposure` or fall back to `type LegacyExposure = Any`
- [x] 1.3 Define `_LegacyDatasetContext` as a `@dataclasses.dataclass` with fields: `filename: str`, `plane_map: dict[str, MaskPlane]`, `unit: astropy.units.Unit`, `storage_class: str`, `read_cls: type[VisitImage]`, `legacy_exposure: LegacyExposure`, `visit_image: VisitImage`

## 2. Update legacy_dataset_context fixture

- [x] 2.1 Change the return type annotation of `legacy_dataset_context` from `dict[str, Any]` to `_LegacyDatasetContext`
- [x] 2.2 After reading `legacy_exposure`, also read `visit_image` via `read_cls.read_legacy(filename, preserve_quantization=True, plane_map=plane_map)`
- [x] 2.3 Replace the `return ctx` dict return with `return _LegacyDatasetContext(...)` constructed from the local variables
- [x] 2.4 Update the fixture docstring to reflect the new return type (remove the `Keys:` enumeration)

## 3. Update legacy_nJy_dataset_context fixture

- [x] 3.1 Change the return type annotation of `legacy_nJy_dataset_context` from `dict[str, Any]` to `_LegacyDatasetContext`
- [x] 3.2 Replace the `return ctx` dict return with `return _LegacyDatasetContext(...)` constructed from the local variables (same shape as `legacy_dataset_context`)
- [x] 3.3 Update the fixture docstring to reflect the new return type

## 4. Delete legacy_visit_image fixture

- [x] 4.1 Remove the `legacy_visit_image` fixture definition entirely from `tests/test_visit_image.py`

## 5. Update all consumers to use attribute access

- [x] 5.1 Update `test_legacy_errors`: change parameter type to `_LegacyDatasetContext`, replace `ctx["key"]` with `ctx.key` throughout
- [x] 5.2 Update `test_component_reads`: change parameter type to `_LegacyDatasetContext`, replace `ctx["key"]` with `ctx.key` throughout
- [x] 5.3 Update `test_legacy_obs_info`: remove `legacy_visit_image` parameter, change `legacy_dataset_context` type to `_LegacyDatasetContext`, replace `ctx["key"]` and bare `legacy_visit_image` references with `ctx.key` and `ctx.visit_image`
- [x] 5.4 Update `test_read_legacy_headers`: same as 5.3
- [x] 5.5 Update `test_from_legacy_headers`: same as 5.3
- [x] 5.6 Update `test_rewrite`: same as 5.3
- [x] 5.7 Update `test_butler_converters`: same as 5.3
- [x] 5.8 Update `test_aperture_corrections_to_legacy`: change parameter from `legacy_visit_image: VisitImage` to `legacy_dataset_context: _LegacyDatasetContext`, use `legacy_dataset_context.visit_image` in the body
- [x] 5.9 Update `test_convert_unit`: change `legacy_nJy_dataset_context` parameter type to `_LegacyDatasetContext`, replace `ctx["key"]` with `ctx.key` throughout

## 6. Clean up imports

- [x] 6.1 Remove `from typing import Any` if it is no longer used elsewhere in the file (check `visit_image_components` return type and any other remaining `Any` usages first)

## 7. Verify

- [x] 7.1 Run `pytest tests/test_visit_image.py -x` and confirm no failures (tests requiring external data will skip)
- [x] 7.2 Run `mypy python/` and confirm no new type errors
- [x] 7.3 Run `ruff check --fix python/ tests/` and `ruff format python/ tests/` and confirm clean
