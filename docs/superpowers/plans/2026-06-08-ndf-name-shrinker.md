# NDF/HDS Name Shrinker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the NDF backend write archive paths whose components exceed the 16-character HDS limit by deterministically shrinking over-long components, while preserving version-disambiguation suffixes and failing loudly on the rare shrink collision.

**Architecture:** A pure, stateless shrink (prefix + blake2b digest, ported from `lsst.daf.butler.name_shrinker`) is applied per path component at the single `ndf/_common.py` translation chokepoint. Version suffixes are applied through a version-aware shrink helper so the visible `_N` survives. `NdfOutputArchive` holds a per-write full-path registry that raises if two different archive entries shrink to the same HDS path. The reader is unchanged — it matches the stored JSON path verbatim.

**Tech Stack:** Python 3.13, h5py, numpy, pydantic, `unittest`/`pytest`, ruff + mypy.

**Spec:** `docs/superpowers/specs/2026-06-08-ndf-name-shrinker-design.md`

**Conventions:**
- Run tests with: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m pytest <path> -v`
  (the installed stack shadows the local checkout, so `PYTHONPATH` is required).
- Lint must stay clean: `~/pyenv/bin/ruff check python tests` and `~/pyenv/bin/ruff format python tests`.
- Type-check: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m mypy python/lsst/images/ndf`.
- One sentence per line in prose; American spelling; docstrings describe the code as it is today (no history references).

---

## File Structure

- `python/lsst/images/ndf/_common.py` — **modify.** Add the pure `_shrink_hds_name` and `shrink_versioned_component` helpers; change `archive_path_to_hdf5_path_components` to shrink instead of raise. This stays the single translation chokepoint.
- `python/lsst/images/ndf/_output_archive.py` — **modify.** Add the per-write collision registry (`_hdf5_path_owners`), the `_register_hdf5_path` guard, and the `_versioned_archive_path` helper; rewire `add_array` and `add_structured_array` to apply versions through the shrink helper and register their node paths.
- `tests/test_ndf_common.py` — **modify.** Replace the "rejects long components" test; add shrink, passthrough, version-aware, and determinism tests.
- `tests/test_ndf_output_archive.py` — **modify.** Add long-name hoisting, the `/noise_realizations/0` regression, versioned-repeat, collision-guard, and write/read round-trip tests.
- `tests/test_cell_coadd.py` — **modify.** Add a real end-to-end CellCoadd FITS/NDF round-trip and cross-backend consistency check (the original reported failure), mirroring the NDF tests in `test_visit_image.py`.

---

## Task 1: Pure component shrink in `_common.py`

**Files:**
- Modify: `python/lsst/images/ndf/_common.py`
- Test: `tests/test_ndf_common.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ndf_common.py`. First extend the import block at the top (lines 16-21) so the helper is importable (`shrink_versioned_component` is added to this block in Task 2):

```python
try:
    from lsst.images.ndf._common import (
        NdfPointerModel,
        _shrink_hds_name,
        archive_path_to_hdf5_path,
    )

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False
```

Then add this test class at the end of the file:

```python
@unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
class ShrinkHdsNameTestCase(unittest.TestCase):
    """Tests for the pure HDS component shrinker."""

    def test_short_names_pass_through_uppercased(self):
        self.assertEqual(_shrink_hds_name("psf"), "PSF")
        self.assertEqual(_shrink_hds_name("a" * 16), "A" * 16)

    def test_long_names_are_shrunk_to_the_limit(self):
        shrunk = _shrink_hds_name("noise_realizations")
        self.assertEqual(len(shrunk), 16)
        self.assertTrue(shrunk.startswith("NOISE_R"))
        self.assertEqual(shrunk, shrunk.upper())

    def test_shrink_is_deterministic(self):
        self.assertEqual(
            _shrink_hds_name("noise_realizations"),
            _shrink_hds_name("noise_realizations"),
        )

    def test_distinct_long_names_get_distinct_tokens(self):
        self.assertNotEqual(
            _shrink_hds_name("noise_realization_field"),
            _shrink_hds_name("noise_realization_other"),
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m pytest tests/test_ndf_common.py::ShrinkHdsNameTestCase -v`
Expected: collection/import error — `cannot import name '_shrink_hds_name'`.

- [ ] **Step 3: Implement `_shrink_hds_name`**

In `python/lsst/images/ndf/_common.py`, add `import hashlib` below the `from __future__` line (line 12) so it reads:

```python
from __future__ import annotations

import hashlib
```

Add the function above `archive_path_to_hdf5_path` (before line 31):

```python
def _shrink_hds_name(name: str, max_length: int = 16, hash_size: int = 4) -> str:
    """Shrink an HDS component name to fit the HDS length limit.

    The name is uppercased.  Names at or under ``max_length`` are returned
    unchanged.  Longer names are replaced by a readable prefix and an
    underscore-separated `blake2b` digest of the full uppercased name, so the
    result is exactly ``max_length`` characters and distinct inputs almost
    never collide.  ``hash_size`` is the digest length in bytes; it occupies
    ``hash_size * 2 + 1`` characters of the result.
    """
    name = name.upper()
    if len(name) <= max_length:
        return name
    digest = hashlib.blake2b(name.encode("ascii"), digest_size=hash_size).hexdigest().upper()
    trunc = max_length - 2 * hash_size - 1
    shrunk = f"{name[:trunc]}_{digest}"
    assert len(shrunk) == max_length
    return shrunk
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m pytest tests/test_ndf_common.py::ShrinkHdsNameTestCase -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/ndf/_common.py tests/test_ndf_common.py
git commit -m "Add pure HDS component name shrinker (DM-55183)"
```

---

## Task 2: Version-aware shrink helper

**Files:**
- Modify: `python/lsst/images/ndf/_common.py`
- Test: `tests/test_ndf_common.py`

- [ ] **Step 1: Write the failing test**

Add `shrink_versioned_component` to the import block edited in Task 1 Step 1, so it reads:

```python
    from lsst.images.ndf._common import (
        NdfPointerModel,
        _shrink_hds_name,
        archive_path_to_hdf5_path,
        shrink_versioned_component,
    )
```

Then add this test class to `tests/test_ndf_common.py`:

```python
@unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
class ShrinkVersionedComponentTestCase(unittest.TestCase):
    """Tests for version-aware HDS component shrinking."""

    def test_version_one_matches_plain_shrink(self):
        self.assertEqual(
            shrink_versioned_component("noise_realizations", 1),
            _shrink_hds_name("noise_realizations"),
        )

    def test_short_versioned_name_keeps_visible_suffix(self):
        self.assertEqual(shrink_versioned_component("data", 2), "DATA_2")

    def test_long_versioned_name_preserves_suffix_within_limit(self):
        shrunk = shrink_versioned_component("noise_realizations", 99)
        self.assertEqual(len(shrunk), 16)
        self.assertTrue(shrunk.endswith("_99"))

    def test_same_base_different_versions_are_distinct(self):
        self.assertNotEqual(
            shrink_versioned_component("noise_realizations", 2),
            shrink_versioned_component("noise_realizations", 3),
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m pytest tests/test_ndf_common.py::ShrinkVersionedComponentTestCase -v`
Expected: FAIL — `shrink_versioned_component` not defined (or stub returns wrong value).

- [ ] **Step 3: Implement `shrink_versioned_component`**

In `python/lsst/images/ndf/_common.py`, add directly below `_shrink_hds_name`:

```python
def shrink_versioned_component(
    base: str, version: int, max_length: int = 16, hash_size: int = 4
) -> str:
    """Shrink a component while preserving a visible version suffix.

    When ``version`` is greater than one a ``_{version}`` suffix is reserved at
    the tail and the ``base`` is shrunk into the remaining characters, so the
    version number stays readable in Starlink tools.  Version one (the first
    occurrence) is shrunk exactly like an unversioned component.
    """
    suffix = f"_{version}" if version > 1 else ""
    return _shrink_hds_name(base, max_length - len(suffix), hash_size) + suffix
```

If a stub was added in Task 1, replace it with this.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m pytest tests/test_ndf_common.py -v`
Expected: PASS (all classes, including Task 1).

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/ndf/_common.py tests/test_ndf_common.py
git commit -m "Add version-aware HDS component shrinker (DM-55183)"
```

---

## Task 3: Shrink in the translator instead of raising

**Files:**
- Modify: `python/lsst/images/ndf/_common.py:45-54` (`archive_path_to_hdf5_path_components`)
- Test: `tests/test_ndf_common.py`

- [ ] **Step 1: Update the existing tests**

In `tests/test_ndf_common.py`, replace `test_archive_path_to_hdf5_path_rejects_long_components` (lines 39-41) with a test that asserts shrinking instead of raising, and add a passthrough assertion to the existing `test_archive_path_to_hdf5_path`:

```python
    def test_archive_path_to_hdf5_path(self):
        self.assertEqual(archive_path_to_hdf5_path(""), "/MORE/LSST/JSON")
        self.assertEqual(archive_path_to_hdf5_path("/psf"), "/MORE/LSST/PSF")
        self.assertEqual(archive_path_to_hdf5_path("/psf/coefficients"), "/MORE/LSST/PSF/COEFFICIENTS")

    def test_archive_path_shrinks_long_components(self):
        result = archive_path_to_hdf5_path("/psf/this_component_is_too_long")
        self.assertTrue(result.startswith("/MORE/LSST/PSF/"))
        leaf = result.rsplit("/", 1)[-1]
        self.assertLessEqual(len(leaf), 16)
        # The short parent component is untouched; only the long leaf shrinks.
        self.assertEqual(result.split("/")[4], "PSF")

    def test_archive_path_shrink_round_trips_to_same_value(self):
        self.assertEqual(
            archive_path_to_hdf5_path("/noise_realizations/0"),
            archive_path_to_hdf5_path("/noise_realizations/0"),
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m pytest tests/test_ndf_common.py::NdfPointerModelTestCase -v`
Expected: FAIL — `archive_path_to_hdf5_path("/psf/this_component_is_too_long")` still raises `ValueError`.

- [ ] **Step 3: Replace the raise with a shrink**

In `python/lsst/images/ndf/_common.py`, replace the body of `archive_path_to_hdf5_path_components` (lines 45-54) with:

```python
def archive_path_to_hdf5_path_components(archive_path: str) -> list[str]:
    """Return HDS-compatible path components for an archive path.

    Each component is uppercased; components longer than the 16-character HDS
    limit are deterministically shrunk by `_shrink_hds_name`.
    """
    return [
        _shrink_hds_name(component)
        for component in archive_path.strip("/").split("/")
        if component
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m pytest tests/test_ndf_common.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/ndf/_common.py tests/test_ndf_common.py
git commit -m "Shrink over-long NDF path components instead of raising (DM-55183)"
```

---

## Task 4: Version-aware hoisting and collision guard in `add_array`

**Files:**
- Modify: `python/lsst/images/ndf/_output_archive.py` — import (line 51), `__init__` (around line 297), new helpers, `add_array` else-branch (lines 527-543)
- Test: `tests/test_ndf_output_archive.py`

- [ ] **Step 1: Write the failing tests**

Add these tests to the `NdfOutputArchiveAddArrayTestCase` class in `tests/test_ndf_output_archive.py` (it already imports `h5py`, `np`, `tempfile`, and `NdfOutputArchive`):

```python
    def test_long_hoisted_component_is_shrunk(self):
        # Regression for the cell_coadd failure: the /noise_realizations/0
        # archive path contains an 18-character component.
        data = np.array([[1.0, 2.0]], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                ref = arch.add_array(data, name="noise_realizations/0")
                # The reported path is exactly what is stored in the JSON.
                self.assertTrue(ref.source.startswith("ndf:/MORE/LSST/"))
                self.assertTrue(ref.source.endswith("/DATA_ARRAY/DATA"))
            with h5py.File(tmp.name, "r") as f:
                # Every HDS component is within the limit.
                hdf5_path = ref.source[len("ndf:") :]
                for component in hdf5_path.strip("/").split("/"):
                    self.assertLessEqual(len(component), 16)
                # The node the JSON points at actually exists.
                self.assertIn(hdf5_path, f)

    def test_repeated_long_name_gets_distinct_versioned_paths(self):
        data = np.array([[1.0]], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                first = arch.add_array(data, name="noise_realizations_value")
                second = arch.add_array(data, name="noise_realizations_value")
                self.assertNotEqual(first.source, second.source)
                # The second occurrence keeps a visible _2 version suffix.
                second_leaf = second.source[len("ndf:") :].split("/")[-3]
                self.assertTrue(second_leaf.endswith("_2"))
            with h5py.File(tmp.name, "r") as f:
                self.assertIn(first.source[len("ndf:") :], f)
                self.assertIn(second.source[len("ndf:") :], f)
```

Note: in `ndf:/MORE/LSST/<TOKEN>/DATA_ARRAY/DATA`, splitting on `/` gives
`["", "MORE", "LSST", "<TOKEN>", "DATA_ARRAY", "DATA"]`, so `<TOKEN>` is index `-3`.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m pytest tests/test_ndf_output_archive.py::NdfOutputArchiveAddArrayTestCase -v`
Expected: PASS for `test_long_hoisted_component_is_shrunk` (Task 3 already fixed the raise) but FAIL for `test_repeated_long_name_gets_distinct_versioned_paths` — the current code glues a raw `_2` then the translator hashes the whole thing, so the leaf does not end in `_2`.

- [ ] **Step 3: Add imports, registry, and helpers**

In `python/lsst/images/ndf/_output_archive.py`, extend the `_common` import (line 51) to include `shrink_versioned_component`:

```python
from ._common import (
    NdfPointerModel,
    archive_path_to_hdf5_path,
    archive_path_to_hdf5_path_components,
    shrink_versioned_component,
)
```

In `__init__`, add the registry alongside the other per-write state (after `self._pointers` on line 297):

```python
        self._hdf5_path_owners: dict[str, str] = {}
```

Add these two helpers to the class (place them next to `_archive_path_to_hdf5_path`, after line 695):

```python
    def _register_hdf5_path(self, hdf5_path: str, logical_id: str) -> None:
        """Record that ``logical_id`` owns ``hdf5_path``; raise on collision.

        ``logical_id`` is the un-shrunk, version-applied archive path, which is
        unique per logical write.  Two different archive entries shrinking to
        the same HDS path would silently clobber one another, so this fails
        loudly instead.
        """
        previous = self._hdf5_path_owners.get(hdf5_path)
        if previous is not None and previous != logical_id:
            raise ValueError(
                f"NDF/HDS name collision: archive entries {previous!r} and {logical_id!r} "
                f"both map to {hdf5_path!r} after 16-character shrinking; increase hash_size."
            )
        self._hdf5_path_owners[hdf5_path] = logical_id

    def _versioned_archive_path(self, name: str, version: int) -> tuple[str, str]:
        """Return ``(archive_path, logical_id)`` for a hoisted name.

        ``archive_path`` has any version suffix applied to its leaf through the
        version-aware shrinker (so the suffix survives the later per-component
        shrink); ``logical_id`` is the un-shrunk version-applied path used for
        collision detection.
        """
        archive_path = name if name.startswith("/") else f"/{name}"
        if version <= 1:
            return archive_path, archive_path
        head, sep, leaf = archive_path.rpartition("/")
        logical_id = f"{archive_path}_{version}"
        versioned = f"{head}{sep}{shrink_versioned_component(leaf, version)}"
        return versioned, logical_id
```

- [ ] **Step 4: Rewire the `add_array` else-branch**

Replace the else-branch (lines 527-543, from `else:` through `sub_ndf_path = self._archive_path_to_hdf5_path(archive_path)`) with:

```python
        else:
            # Hoisted numeric arrays are wrapped as sub-NDFs under
            # /MORE/LSST/<UPPER_PATH> so Starlink tools (KAPPA `display`,
            # `hdstrace`, etc.) can inspect them just like the main image.
            # The sub-NDF has the canonical layout: top-level group with
            # CLASS="NDF" containing a DATA_ARRAY structure (CLASS="ARRAY")
            # with DATA + ORIGIN primitives.  Over-long components are shrunk
            # to fit the 16-character HDS limit; repeated names get a version
            # suffix on their leaf so siblings stay distinct.
            archive_path, logical_id = self._versioned_archive_path(name, version)
            sub_ndf_path = self._archive_path_to_hdf5_path(archive_path)
            self._register_hdf5_path(sub_ndf_path, logical_id)
```

Leave the following lines (`sub_ndf = self._document.ensure_ndf(sub_ndf_path)` onward) unchanged.

- [ ] **Step 5: Register the `serialize_pointer` target in the guard**

`serialize_pointer` hoists JSON sub-trees through the same translator, so two
distinct long pointer names could shrink to the same target. Register it. In
`serialize_pointer` (around lines 429-430), after:

```python
        archive_path = name if name.startswith("/") else f"/{name}"
        target_path = self._archive_path_to_hdf5_path(archive_path)
```

add:

```python
        self._register_hdf5_path(target_path, archive_path)
```

(`serialize_pointer` never versions, so the un-shrunk `archive_path` is the
logical identity. Re-serializing the same object returns early via the
`_pointers` cache, so this line is not reached twice for one object.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m pytest tests/test_ndf_output_archive.py::NdfOutputArchiveAddArrayTestCase tests/test_ndf_output_archive.py::NdfOutputArchivePointerTestCase -v`
Expected: PASS (all add-array and pointer tests, including the two new add-array tests; existing pointer tests must still pass, proving the registration does not false-positive on distinct names).

- [ ] **Step 7: Commit**

```bash
git add python/lsst/images/ndf/_output_archive.py tests/test_ndf_output_archive.py
git commit -m "Apply version-aware shrink and collision guard in add_array (DM-55183)"
```

---

## Task 5: Collision guard fires on a forced clash

**Files:**
- Test: `tests/test_ndf_output_archive.py`

- [ ] **Step 1: Write the failing test**

This proves the guard raises rather than silently clobbering. Force two distinct
names to the same token by monkeypatching the shrinker to a constant. Add to
`NdfOutputArchiveAddArrayTestCase` and add `from unittest import mock` to the
imports at the top of the file (alongside `import unittest`):

```python
    def test_colliding_shrunk_names_raise(self):
        data = np.array([[1.0]], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                # Force both long names to shrink to the same HDS token.
                with mock.patch(
                    "lsst.images.ndf._common._shrink_hds_name",
                    side_effect=lambda name, *a, **k: name.upper() if len(name) <= 16 else "CLASH",
                ):
                    arch.add_array(data, name="long_component_name_one")
                    with self.assertRaisesRegex(ValueError, "name collision"):
                        arch.add_array(data, name="long_component_name_two")
```

Note: `_archive_path_to_hdf5_path` resolves `archive_path_to_hdf5_path` /
`archive_path_to_hdf5_path_components` from `_common`, which call
`_common._shrink_hds_name`, so patching there takes effect. Confirm the patch
target resolves by checking `_common.archive_path_to_hdf5_path_components` calls
`_shrink_hds_name` by its module-global name (it does, from Task 3).

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m pytest "tests/test_ndf_output_archive.py::NdfOutputArchiveAddArrayTestCase::test_colliding_shrunk_names_raise" -v`
Expected: FAIL only if the guard is missing. Since Task 4 already added the guard, this test should PASS immediately — it is a guard-coverage test. If it does not pass, the guard wiring in Task 4 is wrong; fix it before continuing.

- [ ] **Step 3: (No new implementation)**

The guard was implemented in Task 4. If Step 2 passed, proceed. If it failed, the bug is in `_register_hdf5_path` or its call site — re-read Task 4 Steps 3-4.

- [ ] **Step 4: Re-run to confirm green**

Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m pytest tests/test_ndf_output_archive.py::NdfOutputArchiveAddArrayTestCase -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_ndf_output_archive.py
git commit -m "Cover NDF shrink collision guard (DM-55183)"
```

---

## Task 6: Version-aware hoisting and guard in `add_structured_array`

**Files:**
- Modify: `python/lsst/images/ndf/_output_archive.py:743-761` (`add_structured_array`)
- Test: `tests/test_ndf_output_archive.py`

- [ ] **Step 1: Write the failing test**

Find the test class that exercises `add_structured_array` (search the file for
`add_structured_array`); if none has a long-name case, add this test there (or to
a new `NdfOutputArchiveStructuredArrayTestCase` mirroring the add-array class
setup):

```python
    def test_structured_array_long_name_is_shrunk_and_versioned(self):
        dtype = np.dtype([("alpha", "f8"), ("beta", "i4")])
        arr = np.zeros(3, dtype=dtype)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                first = arch.add_structured_array(arr, name="catalog_of_long_named_sources")
                second = arch.add_structured_array(arr, name="catalog_of_long_named_sources")
                for model in (first, second):
                    for column in model.columns:
                        token = column.data.source[len("ndf:") :]
                        for component in token.strip("/").split("/"):
                            self.assertLessEqual(len(component), 16)
                # The two structured arrays land in distinct sub-trees.
                self.assertNotEqual(
                    first.columns[0].data.source,
                    second.columns[0].data.source,
                )
            with h5py.File(tmp.name, "r") as f:
                for model in (first, second):
                    for column in model.columns:
                        self.assertIn(column.data.source[len("ndf:") :], f)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m pytest "tests/test_ndf_output_archive.py" -k structured_array_long_name -v`
Expected: FAIL — the current `name = f"{name}_{version}"` plus whole-string hashing means the two structured arrays either collide or the version suffix is not preserved distinctly.

- [ ] **Step 3: Rewire `add_structured_array`**

In `python/lsst/images/ndf/_output_archive.py`, replace the version-gluing and
per-column path construction (lines 744-761). Change:

```python
        name, version = self._register_name(name)
        if version > 1:
            name = f"{name}_{version}"
        columns = TableColumnModel.from_record_dtype(array.dtype)
        for c in columns:
            column_path = name if len(columns) == 1 else f"{name}/{c.name}"
            archive_path = column_path if column_path.startswith("/") else f"/{column_path}"
            sub_ndf_path = self._archive_path_to_hdf5_path(archive_path)
            column_array = np.asarray(array[c.name])
            sub_ndf = self._document.ensure_ndf(sub_ndf_path)
            sub_ndf.set_array_component(
                "DATA_ARRAY",
                column_array,
                origin=np.zeros(column_array.ndim, dtype=np.int64),
                compression_options=self._compression_options,
            )
            assert isinstance(c.data, ArrayReferenceModel)
            c.data.source = f"ndf:{sub_ndf_path}/DATA_ARRAY/DATA"
```

to:

```python
        name, version = self._register_name(name)
        base_path, base_logical = self._versioned_archive_path(name, version)
        columns = TableColumnModel.from_record_dtype(array.dtype)
        for c in columns:
            if len(columns) == 1:
                archive_path = base_path
                logical_id = base_logical
            else:
                archive_path = f"{base_path}/{c.name}"
                logical_id = f"{base_logical}/{c.name}"
            sub_ndf_path = self._archive_path_to_hdf5_path(archive_path)
            self._register_hdf5_path(sub_ndf_path, logical_id)
            column_array = np.asarray(array[c.name])
            sub_ndf = self._document.ensure_ndf(sub_ndf_path)
            sub_ndf.set_array_component(
                "DATA_ARRAY",
                column_array,
                origin=np.zeros(column_array.ndim, dtype=np.int64),
                compression_options=self._compression_options,
            )
            assert isinstance(c.data, ArrayReferenceModel)
            c.data.source = f"ndf:{sub_ndf_path}/DATA_ARRAY/DATA"
```

`_versioned_archive_path` already normalizes the leading slash, so the separate
`if ... startswith("/")` normalization is no longer needed.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m pytest "tests/test_ndf_output_archive.py" -k structured_array -v`
Expected: PASS (new test plus any pre-existing structured-array tests).

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/ndf/_output_archive.py tests/test_ndf_output_archive.py
git commit -m "Apply version-aware shrink and guard in add_structured_array (DM-55183)"
```

---

## Task 7: End-to-end write/read round-trip through a long-named hoist

**Files:**
- Test: `tests/test_ndf_output_archive.py`

This is the always-on, dependency-free regression (no test data or
`lsst.cell_coadds` needed); Task 8 adds the heavier real-CellCoadd reproduction
on top.

- [ ] **Step 1: Write the failing test**

This guards the core invariant — the stored JSON path resolves on read. Add to
`NdfOutputArchiveAddArrayTestCase`. It reopens the file with `NdfInputArchive`
and reads the array back via `get_array`:

```python
    def test_long_name_round_trips_through_input_archive(self):
        from lsst.images.ndf import NdfInputArchive

        data = np.arange(6, dtype=np.float32).reshape(2, 3)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                ref = arch.add_array(data, name="noise_realizations/0")
            with NdfInputArchive.open(tmp.name) as inp:
                read_back = inp.get_array(ref)
        np.testing.assert_array_equal(read_back, data)
```

Confirm `NdfInputArchive` is exported from `lsst.images.ndf` (it is, per
`ndf/__init__.py`); if the import differs, use the path the package exposes.

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m pytest "tests/test_ndf_output_archive.py::NdfOutputArchiveAddArrayTestCase::test_long_name_round_trips_through_input_archive" -v`
Expected: PASS (Tasks 3-4 make the write succeed and the reader matches the stored path verbatim). If it FAILS, the JSON `source` and the on-disk node path disagree — inspect `ref.source` versus the actual HDF5 layout.

- [ ] **Step 3: (No new implementation expected)**

If Step 2 passed, the invariant holds. If it failed, reconcile the path written
into `ref.source` (Task 4) with the node created by `ensure_ndf`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_ndf_output_archive.py
git commit -m "Round-trip a long hoisted NDF name through read (DM-55183)"
```

---

## Task 8: CellCoadd FITS/NDF round-trip in `test_cell_coadd.py`

This is the real-world reproduction of the reported bug: a `CellCoadd` serialize
hoists `noise_realizations/<n>` arrays, which is exactly what crashed. It mirrors
the existing NDF tests in `test_visit_image.py` (`test_round_trip_ndf`,
`test_fits_ndf_consistency`) and the zarr additions in commit `b53ab6a`. The case
gracefully skips unless `TESTDATA_IMAGES_DIR`, `lsst.cell_coadds`, and `h5py` are
all available (the class is already gated on `TESTDATA_IMAGES_DIR` and
`lsst.cell_coadds` via `setUpClass`).

**Files:**
- Modify: `tests/test_cell_coadd.py`

- [ ] **Step 1: Add the `RoundtripNdf` import and an `h5py` gate**

In `tests/test_cell_coadd.py`, add `RoundtripNdf` to the
`from lsst.images.tests import (...)` block (keep the list alphabetised):

```python
from lsst.images.tests import (
    DP2_COADD_DATA_ID,
    DP2_COADD_MISSING_CELL,
    RoundtripFits,
    RoundtripJson,
    RoundtripNdf,
    assert_cell_coadds_equal,
    assert_masked_images_equal,
    assert_psfs_equal,
    compare_cell_coadd_to_legacy,
)
```

Then, immediately after that import block and before
`DATA_DIR = os.environ.get(...)`, add:

```python
try:
    import h5py  # noqa: F401

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False
```

- [ ] **Step 2: Write the failing tests**

Add these two methods to `CellCoaddTestCase` (e.g. directly after
`test_fits_json_consistency`):

```python
    @unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
    def test_round_trip_ndf(self) -> None:
        """NDF round-trip for CellCoadd, exercising hoisted long-named arrays."""
        with RoundtripNdf(self, self.cell_coadd, "CellCoadd") as roundtrip:
            assert_cell_coadds_equal(self, roundtrip.result, self.cell_coadd, expect_view=False)

    @unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
    def test_fits_ndf_consistency(self) -> None:
        """FITS and NDF backends produce equal CellCoadds on round-trip."""
        with (
            RoundtripFits(self, self.cell_coadd) as fits_rt,
            RoundtripNdf(self, self.cell_coadd) as ndf_rt,
        ):
            assert_cell_coadds_equal(self, self.cell_coadd, fits_rt.result, expect_view=False)
            assert_cell_coadds_equal(self, self.cell_coadd, ndf_rt.result, expect_view=False)
            assert_cell_coadds_equal(self, fits_rt.result, ndf_rt.result, expect_view=False)
```

- [ ] **Step 3: Run the tests**

Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m pytest tests/test_cell_coadd.py -k "ndf" -v`
Expected, depending on the environment:
- With `TESTDATA_IMAGES_DIR`, `lsst.cell_coadds`, and `h5py` present: PASS — this proves the shrinker fix lets a real CellCoadd serialize to NDF and read back equal to FITS. (Before Tasks 1-6 it would FAIL with the `16-character HDS limit` `ValueError`.)
- Otherwise: SKIPPED. That is acceptable — the always-on regression coverage lives in Task 7.

If the tests are skipped in your environment, confirm there are no collection or
import errors in the output (the file must still import cleanly).

- [ ] **Step 4: Commit**

```bash
git add tests/test_cell_coadd.py
git commit -m "Round-trip CellCoadd through NDF and compare with FITS (DM-55183)"
```

---

## Task 9: Full NDF suite, lint, and type-check

**Files:** none (verification only).

- [ ] **Step 1: Run the full NDF test suite**

Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m pytest tests/test_ndf_common.py tests/test_ndf_output_archive.py tests/test_ndf_input_archive.py tests/test_ndf_layout.py tests/test_cell_coadd.py -v`
Expected: PASS (no regressions). `test_cell_coadd.py` cases that need `lsst.cell_coadds` will skip if it is not importable — that is acceptable.

- [ ] **Step 2: Lint**

Run: `~/pyenv/bin/ruff check python/lsst/images/ndf tests/test_ndf_common.py tests/test_ndf_output_archive.py`
Then: `~/pyenv/bin/ruff format --check python/lsst/images/ndf tests/test_ndf_common.py tests/test_ndf_output_archive.py`
Expected: no errors. If `format --check` reports changes, run without `--check` and re-commit.

- [ ] **Step 3: Type-check**

Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m mypy python/lsst/images/ndf/_common.py python/lsst/images/ndf/_output_archive.py`
Expected: no new errors versus the pre-change baseline.

- [ ] **Step 4: (Optional) reproduce the original failing command**

If `../cell_coadd.fits` is available locally, confirm the reported crash is gone:
Run: `PYTHONPATH=$PWD/python ~/pyenv/bin/python -m lsst.images.cli reformat ../cell_coadd.fits /tmp/cell.sdf`
(or `bin/lsst-images-admin reformat ../cell_coadd.fits /tmp/cell.sdf`)
Expected: completes without the `16-character HDS limit` `ValueError`. Remove `/tmp/cell.sdf` afterward.

- [ ] **Step 5: Commit any formatting fixups**

```bash
git add -A
git commit -m "Tidy formatting after NDF name shrinker (DM-55183)" || echo "nothing to commit"
```
