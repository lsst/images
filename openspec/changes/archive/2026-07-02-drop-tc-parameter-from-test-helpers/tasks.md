## 1. Update `_checks.py`

- [x] 1.1 Remove `import unittest`; add `import re` and `import pytest`
- [x] 1.2 Drop `tc: unittest.TestCase` from all 27 public function signatures
- [x] 1.3 Remove the `tc` parameter entry from every NumPy-style docstring in the file
- [x] 1.4 Rewrite `assert_equal_allow_nan`: replace the `try/except tc.assertEqual` block with an explicit `if not (a == b or (math.isnan(a) and math.isnan(b))): raise AssertionError(f"{a!r} != {b!r}")`
- [x] 1.5 Replace all `tc.assertEqual(a, b)` / `tc.assertEqual(a, b, msg)` with `assert a == b` / `assert a == b, msg`
- [x] 1.6 Replace all `tc.assertTrue(x)` / `tc.assertTrue(x, msg=...)` with `assert x` / `assert x, msg`
- [x] 1.7 Replace `tc.assertIs(a, b)` with `assert a is b`; `tc.assertIsNot(a, b)` with `assert a is not b`
- [x] 1.8 Replace `tc.assertIsNone(x)` with `assert x is None`; `tc.assertIsInstance(x, T)` with `assert isinstance(x, T)`
- [x] 1.9 Replace `tc.assertIn(a, b)` with `assert a in b`
- [x] 1.10 Replace `tc.assertGreater(a, b)` with `assert a > b`; `tc.assertGreaterEqual(a, b)` with `assert a >= b`; `tc.assertLessEqual(a, b)` with `assert a <= b`
- [x] 1.11 Replace `with tc.assertRaises(InvalidPsfError):` with `with pytest.raises(InvalidPsfError):` (one occurrence, `compare_psf_to_legacy`)
- [x] 1.12 Replace `tc.assertRegex(cls.SCHEMA_VERSION, r"^\d+\.\d+\.\d+$")` with `assert re.fullmatch(r"^\d+\.\d+\.\d+$", cls.SCHEMA_VERSION)` (one occurrence, `check_archive_tree_class_invariants`)

## 2. Update `_roundtrip.py`

- [x] 2.1 Remove `import unittest`; add `import pytest`
- [x] 2.2 Drop `tc: unittest.TestCase` parameter from `RoundtripBase.__init__`; remove the `self._tc = tc` assignment; remove the `tc` docstring entry
- [x] 2.3 Replace `self._tc.assertEqual(self.result.metadata[k], self._test_metadata[k])` with `assert self.result.metadata[k] == self._test_metadata[k]` (in `__enter__`)
- [x] 2.4 Replace `self._tc.assertEqual(DatasetRef.from_simple(...), self.ref)` with `assert DatasetRef.from_simple(...) == self.ref` (in `_run_with_butler`)
- [x] 2.5 Replace `self._tc.assertEqual(self.result.butler_provenance.quantum_id, quantum_id)` with `assert self.result.butler_provenance.quantum_id == quantum_id` (in `_run_with_butler`)
- [x] 2.6 Replace `self._tc.assertTrue(self.filename.endswith(...), ...)` with `assert self.filename.endswith(...), ...` (in `_run_with_butler`)
- [x] 2.7 Replace `self._tc.assertIsNone(reader.butler_info)` with `assert reader.butler_info is None` (in `_run_without_butler`)
- [x] 2.8 Replace the three `raise unittest.SkipTest(msg)` calls with `pytest.skip(msg)` (two in `RoundtripBase.get`, one in `TemporaryButler.__enter__`)

## 3. Update test files — helpers with `Roundtrip*` constructors and helper calls

- [x] 3.1 `tests/test_image.py`: drop leading `self` from all `Roundtrip*(self, ...)` and `assert_*(self, ...)` / `compare_*(self, ...)` calls
- [x] 3.2 `tests/test_mask.py`: same
- [x] 3.3 `tests/test_masked_image.py`: same
- [x] 3.4 `tests/test_visit_image.py`: same
- [x] 3.5 `tests/test_cell_coadd.py`: same
- [x] 3.6 `tests/test_psfs.py`: same
- [x] 3.7 `tests/test_fields.py`: same
- [x] 3.8 `tests/test_cameras.py`: same
- [x] 3.9 `tests/test_transforms.py`: same
- [x] 3.10 `tests/test_color_image.py`: same
- [x] 3.11 `tests/test_difference_image_extras.py`: same

## 4. Update test files — helper calls only (no `Roundtrip*`)

- [x] 4.1 `tests/test_from_hdu_list.py`: drop leading `self` from all `assert_*(self, ...)` / `compare_*(self, ...)` calls
- [x] 4.2 `tests/test_geom.py`: same
- [x] 4.3 `tests/test_polygon.py`: same
- [x] 4.4 `tests/test_schema_versioning.py`: same (includes `check_archive_tree_class_invariants(self, cls)`)

## 5. Update test files — `Roundtrip*` constructor only (no helper calls)

- [x] 5.1 `tests/test_ndf_layout.py`: drop leading `self` from all `Roundtrip*(self, ...)` constructors

## 6. Verify

- [x] 6.1 Run `pytest -r a -v -n 3 --cov=lsst.images --cov-report=term` — all 471 tests must pass
- [x] 6.2 Run `ruff check --fix python/ tests/` — no new lint errors
- [ ] 6.3 Run `mypy python/` — no new type errors (mypy not available in sandbox; deferred to CI)
