## 1. Convert no-setUp, no-optional-dep files

- [x] 1.1 Convert `tests/test_schema_v1_fixtures.py`: remove `unittest.TestCase`; replace the two `subTest` loops with `@pytest.mark.parametrize("path", _FIXTURES, ids=lambda p: p.stem)` — each fixture file becomes a separate test node and continue-on-failure is automatic; rewrite `assertIn`/`assertEqual` as `assert`; add imperative docstrings; remove `import unittest` and `unittest.main()` guard
- [x] 1.2 Convert `tests/test_serialization_backends.py`: flatten 2 classes into free functions; rewrite assertions; add imperative docstrings; remove `import unittest` and `unittest.main()` guard

## 2. Convert no-setUp, optional-dep (h5py) files

- [x] 2.1 Convert `tests/test_ndf_starlink_ingest.py`: replace `@unittest.skipUnless(HAVE_H5PY, ...)` with module-level `pytestmark = pytest.mark.skipif(not HAVE_H5PY, reason="h5py is not installed")`; rewrite assertions; add imperative docstrings; remove `import unittest` and `unittest.main()` guard
- [x] 2.2 Convert `tests/test_ndf_model.py`: same `pytestmark` pattern as 2.1; rewrite assertions; add imperative docstrings; remove `import unittest` and `unittest.main()` guard
- [x] 2.3 Convert `tests/test_ndf_format_version.py`: same `pytestmark` pattern; rewrite assertions (structural twin of already-converted `test_fits_format_version.py`); add imperative docstrings; remove `import unittest` and `unittest.main()` guard
- [x] 2.4 Convert `tests/test_ndf_layout.py`: module-level `pytestmark` for `HAVE_H5PY`; flatten 3 classes into free functions; keep module-level `_cls` / `_hds_type` / `_hds_shape` helpers unchanged; rewrite assertions; add imperative docstrings; remove `import unittest` and `unittest.main()` guard

## 3. Convert files with trivial setUp

- [x] 3.1 Convert `tests/test_fits_date_header.py`: extract `setUp`'s `self.image` into a `@pytest.fixture`; keep the HDU `subTest` loop as a plain `for` loop with `assert` — `index` and `extname` are locals that pytest will display on failure; rewrite other assertions; add imperative docstrings; remove `import unittest` and `unittest.main()` guard
- [x] 3.2 Convert `tests/test_schema_v1_legacy_fixtures.py`: no `setUp`; replace the `subTest` loop with `@pytest.mark.parametrize("name", ["visit_image_dp1.json", "visit_image_dp2.json"])` — each filename becomes a test node and is independently skippable; replace `self.skipTest(...)` with `pytest.skip(...)`; rewrite assertions; add imperative docstrings; remove `import unittest` and `unittest.main()` guard
- [x] 3.3 Convert `tests/test_detached_archive.py`:
  - Extract `DetachedArchiveTestCase.setUp` (`self.archive = DetachedArchive()`) into a `@pytest.fixture` named `detached_archive`
  - Extract `ComponentProbeTestCase.setUp` (creates `TemporaryDirectory` with `addCleanup`, reads fixture, constructs `DetachedArchive`) into a `yield`-based `@pytest.fixture` that cleans up via `with tempfile.TemporaryDirectory() as tmpdir: ... yield ...`
  - Replace the `FREE_COMPONENTS` `subTest` loop with `@pytest.mark.parametrize("component", FREE_COMPONENTS)` — 9 independent entries, each becomes a test node so continue-on-failure is preserved
  - Keep the `PIXEL_COMPONENTS` `subTest` loop as a plain `for` loop with `pytest.raises` — 3-item tuple, `component` is a local visible in tracebacks
  - Rewrite all other assertions; add imperative docstrings; remove `import unittest` and `unittest.main()` guard

## 4. Convert files with no setUp but richer structure

- [x] 4.1 Convert `tests/test_serialization_registry.py`: flatten 4 classes into free functions; keep local class definitions (`_EntryPointTree`, `_FakeEntryPoint`) inside `test_entry_point_provider_loaded_on_miss`; keep `_all_concrete_archive_tree_subclasses` as a module-level helper; keep `_REGISTRY` save/restore inside the `try/finally` in `test_builtin_provider_loaded_on_miss` and convert its `subTest` loop to a plain `for` loop with `assert` — `schema_name` is a local visible in tracebacks, and the `finally` restores registry state regardless of whether the failure is a soft subTest fail or a hard raise; rewrite all assertions; add imperative docstrings; remove `import unittest` and `unittest.main()` guard

## 5. Verification

- [x] 5.1 Run `pytest -r a -v -n 3 tests/test_schema_v1_fixtures.py tests/test_serialization_backends.py tests/test_ndf_starlink_ingest.py tests/test_ndf_model.py tests/test_ndf_format_version.py tests/test_ndf_layout.py tests/test_fits_date_header.py tests/test_schema_v1_legacy_fixtures.py tests/test_detached_archive.py tests/test_serialization_registry.py` and confirm all tests pass or skip as expected
- [x] 5.2 Run `ruff check --fix python/ tests/` and `ruff format python/ tests/` and confirm no errors remain
- [x] 5.3 Commit the ten converted files on the sandbox branch
