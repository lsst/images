## 1. Tier 1 — no setUp, pure assertions

- [x] 1.1 Convert `tests/test_utils.py`: remove `unittest.TestCase`, replace `assertEqual` with `assert ==`, drop `import unittest` and `__main__` guard
- [x] 1.2 Convert `tests/test_json_schema.py`: remove `TestCase`, unroll `subTest` loop into `@pytest.mark.parametrize("mode", ["validation", "serialization"])`, extract `_check` as a module-level helper, replace `assertIsInstance`/`assertEqual`/`assertIn` with `assert`
- [x] 1.3 Convert `tests/test_fits_format_version.py`: remove `TestCase`, replace `assertRaises`/`assertEqual` with `pytest.raises`/`assert`
- [x] 1.4 Convert `tests/test_fits_output_archive.py`: remove `TestCase`, promote `_write_archive` to a module-level helper, replace `assertEqual` with `assert ==`
- [x] 1.5 Convert `tests/test_geom.py`: remove `TestCase` from `XYYXTestCase`, `IntervalTestCase`, `BoxTestCase`; replace all `assertXxx` calls with `assert`; replace `assertRaises` with `pytest.raises`
- [x] 1.6 Convert `tests/test_schema_versioning.py`: remove `TestCase` from all six classes; replace all `assertXxx` with `assert`; replace `assertRaises` with `pytest.raises`; keep the `subTest` loop in `ArchiveTreeClassInvariantsTestCase` as a plain `for` loop with `assert`; convert `FixtureMutationTestCase.setUp` to a `fixture_path` fixture

## 2. Tier 2 — setUp becomes fixture

- [x] 2.1 Convert `tests/test_ndf_common.py`: add `@pytest.fixture def shrinker()` returning `HdsNameShrinker()`; convert `NdfPointerModelTestCase` and `HdsNameShrinkerTestCase` to free-functions accepting `shrinker`; replace `@unittest.skipUnless` with `@pytest.mark.skipif(not HAVE_H5PY, reason="h5py is not installed")` on each function group
- [x] 2.2 Convert `tests/test_legacy.py`: add `@pytest.fixture def rng()` returning `np.random.default_rng(500)`; drop `self.maxDiff = None`; replace `@unittest.skipIf(skip_all, ...)` with `@pytest.mark.skipif(skip_all, reason="lsst legacy packages could not be imported.")` on each test function; replace `assertXxx` with `assert`
- [x] 2.3 Convert `tests/test_polygon.py`: add `@pytest.fixture def polygon()` and `@pytest.fixture def regions()` (returning a namedtuple or simple object with `a`/`b`/`c`/`d`); convert `SimplePolygonTestCase` and `RegionTestCase` to free-functions; replace `@unittest.skipUnless(have_legacy, ...)` with `@pytest.mark.skipif(not have_legacy, ...)` on `test_legacy`; replace all `assertXxx` with `assert`

## 3. Verification

- [x] 3.1 Run `pytest tests/test_utils.py tests/test_json_schema.py tests/test_fits_format_version.py tests/test_fits_output_archive.py tests/test_geom.py tests/test_schema_versioning.py tests/test_ndf_common.py tests/test_legacy.py tests/test_polygon.py -v` and confirm all tests pass
- [x] 3.2 Run `ruff check --fix python/ tests/` and `ruff format python/ tests/` on the converted files; fix any issues
- [x] 3.3 Run full suite `pytest -r a -v -n 3` and confirm no regressions
