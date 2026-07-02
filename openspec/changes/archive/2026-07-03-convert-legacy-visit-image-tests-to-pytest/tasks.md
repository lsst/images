## 1. Add parametrized session-scoped fixtures

- [x] 1.1 Add `legacy_dataset_context` session fixture parametrized over `["visit_image", "preliminary_visit_image", "difference_image"]`; each param yields a `dict` with keys `filename`, `legacy_exposure`, `plane_map`, `unit`, `storage_class`, `read_cls` — extracted from the three `setUpClass` methods; wrap the `ExposureFitsReader` import in a `try/except ImportError` that calls `pytest.skip(...)` rather than `raise unittest.SkipTest`
- [x] 1.2 Add `legacy_visit_image` session fixture depending on `legacy_dataset_context`; call `context["read_cls"].read_legacy(context["filename"], preserve_quantization=True, plane_map=context["plane_map"])` and return the result
- [x] 1.3 Add `legacy_nJy_dataset_context` session fixture parametrized over `["visit_image", "difference_image"]` only (same structure as `legacy_dataset_context` but without `preliminary_visit_image`) — used exclusively by `test_convert_unit`

## 2. Convert the mixin helper to a module-level function

- [x] 2.1 Convert `check_legacy_obs_info` to a module-level function `_check_legacy_obs_info(obs_info)` replacing `self.assertIsInstance` / `self.assertEqual` with plain `assert` statements

## 3. Convert mixin test methods to free functions

- [x] 3.1 Convert `test_legacy_errors` — signature `def test_legacy_errors(legacy_dataset_context):`; replace all `self.assert*` with plain `assert` / `pytest.raises`; add `@pytest.mark.skipif(EXTERNAL_DATA_DIR is None, reason=...)`
- [x] 3.2 Convert `test_component_reads` — signature `def test_component_reads(legacy_dataset_context):`; call `_check_legacy_obs_info`; replace `self.assert*` with plain `assert`; add skip guard
- [x] 3.3 Convert `test_obs_info` — renamed `test_legacy_obs_info` to avoid collision with existing `test_obs_info`; signature `def test_legacy_obs_info(legacy_visit_image, legacy_dataset_context):`; replace `self.assert*` / `self.maxDiff = None` with plain `assert`; add skip guard
- [x] 3.4 Convert `test_aperture_corrections_to_legacy` — signature `def test_aperture_corrections_to_legacy(legacy_visit_image):`; replace `self.assert*` with plain `assert`; add skip guard
- [x] 3.5 Convert `test_read_legacy_headers` — signature `def test_read_legacy_headers(legacy_visit_image, legacy_dataset_context):`; replace `self.assert*` with plain `assert`; add skip guard
- [x] 3.6 Convert `test_from_legacy_headers` — signature `def test_from_legacy_headers(legacy_visit_image, legacy_dataset_context):`; replace `self.assert*` with plain `assert`; add skip guard
- [x] 3.7 Convert `test_rewrite` — signature `def test_rewrite(legacy_visit_image, legacy_dataset_context):`; remove both `with self.subTest():` wrappers, making assertions sequential plain code; replace all `self.assert*` with plain `assert`; add skip guard
- [x] 3.8 Convert `test_butler_converters` — signature `def test_butler_converters(legacy_visit_image, legacy_dataset_context):`; replace `raise unittest.SkipTest(...)` with `pytest.skip(...)`; replace `self.assert*` with plain `assert`; add skip guard

## 4. Add test_convert_unit as a parametrized free function

- [x] 4.1 Add `test_convert_unit` as a free function with signature `def test_convert_unit(legacy_nJy_dataset_context):`, decorated with `@pytest.mark.skipif(EXTERNAL_DATA_DIR is None, reason=...)`; replace all `self.assert*` with plain `assert` / `pytest.raises`; access `legacy_exposure` and `visit_image` from the fixture dict

## 5. Clean up imports and remove old classes

- [x] 5.1 Remove `VisitImageLegacyTestMixin`, `VisitImageLegacyTestCase`, `PreliminaryVisitImageLegacyTestCase`, and `DifferenceImageLegacyTestCase` class definitions once all methods have been converted
- [x] 5.2 Remove `import unittest` from the import block
- [x] 5.3 Remove `ClassVar` from the `typing` import (retain `Any`)
- [x] 5.4 Remove `if __name__ == "__main__": unittest.main()` block if present

## 6. Sync the spec delta to the main spec

- [x] 6.1 Merge the `test_convert_unit` scope requirement from `specs/pytest-test-style/spec.md` into `openspec/specs/pytest-test-style/spec.md`

## 7. Verify

- [x] 7.1 Run `pytest tests/test_visit_image.py -r a -v` and confirm all converted tests pass (or skip on missing optional deps / external data)
- [x] 7.2 Run `ruff check python/ tests/` and `ruff format --check python/ tests/` — no errors
- [x] 7.3 Confirm `test_convert_unit[visit_image]` and `test_convert_unit[difference_image]` both appear and pass; confirm no `test_convert_unit[preliminary_visit_image]` node exists
