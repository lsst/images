# NDF HDS Name-Length Limit Constant (DAT__SZNAM) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hard-coded HDS name-length limit (currently `16`, actually `15`) with a single named constant `DAT__SZNAM`, defined once in `_hds.py` and used consistently across the NDF package and its tests, so the value can be changed in one place and error messages report it correctly.

**Architecture:** Define `DAT__SZNAM = 15` in `ndf/_hds.py` (the existing home for Starlink/HDS constants), citing Starlink `dat_par.h`. Thread it as the default `max_length` of the two shrink helpers in `_common.py`, interpolate it into the runtime collision message in `_output_archive.py`, reword the literal "16-character" docstrings/comments to reference the limit symbolically, and key the NDF test assertions off the constant instead of the literal `16`. The HDS *object name* limit is the only one enforced; HDS type tags are derived from already-shrunk names, so a single name-length constant bounds them too.

**Tech Stack:** Python 3.13, h5py, numpy, pydantic, `unittest`/`pytest`, ruff + mypy.

**Background / why 15:** Starlink `dat_par.h` (`~/star/include/dat_par.h`, `DAT_PAR`) defines `DAT__SZNAM 15` (object name), `DAT__SZGRP 15` (group name), `DAT__SZTYP 15` (type string). The earlier implementation used `16` by mistake. All three are currently `15`; only the object-name limit is enforced here, so we define the single `DAT__SZNAM` (the design decision for this work). The other two are mentioned in a comment for future reference but not introduced as code.

**Atomicity note:** The constant, the production defaults that consume it, and the test assertions that encode the limit value are coupled — the test suite can only be green at `15` once all three change together. Task 1 therefore makes all of those edits and lands them in a single green commit (no intermediate red commit).

**Conventions:**
- Run tests with: `PYTHONPATH=./python ~/pyenv/bin/python -m pytest <path> -v` (the installed stack shadows the local checkout, so `PYTHONPATH` is required; this exact prefix is allowlisted).
- Lint must stay clean: `~/pyenv/bin/ruff check <files>` and `~/pyenv/bin/ruff format --check <files>`.
- Type-check: `PYTHONPATH=./python ~/pyenv/bin/python -m mypy <files>` — ignore the 3 pre-existing `frozendict`/`lsst.cell_coadds` stub errors; only new errors matter.
- One sentence per line in prose/docstrings; American spelling; docstrings describe the code as it is today (no history references).
- Do NOT `git add -A`; stage only the files each task lists.

---

## File Structure

- `python/lsst/images/ndf/_hds.py` — **modify.** Add the `DAT__SZNAM` constant (single source of truth) next to the other module-level Starlink constants. Not added to `__all__` (consistent with the sibling `NDF_AST_DATA_WIDTH` constants, which are also internal and consumed by direct import).
- `python/lsst/images/ndf/_common.py` — **modify.** Import `DAT__SZNAM`; use it as the default `max_length` of `_shrink_hds_name` and `shrink_versioned_component`; reword the two "16-character" docstrings.
- `python/lsst/images/ndf/_output_archive.py` — **modify.** Reword the one "16-character" comment and interpolate `_hds.DAT__SZNAM` into the collision `ValueError` message so it reports the live value.
- `tests/test_ndf_common.py` — **modify.** Import `DAT__SZNAM`; key the boundary/length assertions off it instead of the literal `16`; relax the limit-sensitive `startswith("NOISE_R")` assertion.
- `tests/test_ndf_output_archive.py` — **modify.** Import `DAT__SZNAM`; key the `len(component) <= …` and mock-threshold assertions off it.
- `docs/superpowers/specs/2026-06-08-ndf-name-shrinker-design.md` — **modify (final task).** Update the design-of-record's "16" references to `15` / `DAT__SZNAM` for accuracy.

The change is a single cohesive refactor: Task 1 lands the constant, its production consumers, and the matching test edits together (one green commit); Task 2 verifies end-to-end; Task 3 updates the design doc.

---

## Task 1: Parameterize the HDS name limit as `DAT__SZNAM` (constant + production + tests, atomic)

**Files:**
- Modify: `python/lsst/images/ndf/_hds.py` (add constant near `NDF_AST_DATA_WIDTH`, ~line 83-84)
- Modify: `python/lsst/images/ndf/_common.py` (import + two default args + two docstrings)
- Modify: `python/lsst/images/ndf/_output_archive.py` (one comment + one error message)
- Modify: `tests/test_ndf_common.py` (import + boundary/length assertions)
- Modify: `tests/test_ndf_output_archive.py` (import + length/mock-threshold assertions)

- [ ] **Step 1: Add the constant to `_hds.py`**

In `python/lsst/images/ndf/_hds.py`, find the existing constants:

```python
NDF_AST_DATA_WIDTH = 32
NDF_AST_DATA_MIN_WIDTH = 16
```

Add directly below them:

```python
# HDS object-name length limit, from the Starlink DAT_PAR include
# (``dat_par.h``): ``DAT__SZNAM 15`` ("Size of object name").
# The sibling limits ``DAT__SZGRP`` (group name) and ``DAT__SZTYP`` (type
# string) are also 15 today; only the object-name limit is enforced here,
# because HDS type tags are derived from already-shrunk component names.
DAT__SZNAM = 15
```

Do NOT add `DAT__SZNAM` to `__all__` — the sibling `NDF_AST_DATA_WIDTH`/`NDF_AST_DATA_MIN_WIDTH` constants are not in `__all__` either; module constants here are consumed by direct import within the package.

- [ ] **Step 2: Use the constant in `_common.py`**

In `python/lsst/images/ndf/_common.py`, the current imports are:

```python
from __future__ import annotations

import hashlib

import pydantic
```

Add a relative import of the constant (after the third-party `pydantic` import, as a first-party group):

```python
from __future__ import annotations

import hashlib

import pydantic

from ._hds import DAT__SZNAM
```

(`_hds.py` imports nothing from the other `ndf` modules, so this introduces no circular import.)

Change the signature of `_shrink_hds_name` from:

```python
def _shrink_hds_name(name: str, max_length: int = 16, hash_size: int = 4) -> str:
```
to:
```python
def _shrink_hds_name(name: str, max_length: int = DAT__SZNAM, hash_size: int = 4) -> str:
```

Change the signature of `shrink_versioned_component` from:

```python
def shrink_versioned_component(base: str, version: int, max_length: int = 16, hash_size: int = 4) -> str:
```
to:
```python
def shrink_versioned_component(base: str, version: int, max_length: int = DAT__SZNAM, hash_size: int = 4) -> str:
```

In `archive_path_to_hdf5_path`, reword the docstring sentence that currently reads:

```
    ``/MORE/LSST/``. This mirrors the serialization path while keeping HDS
    component names within their 16-character limit.
```
to:
```
    ``/MORE/LSST/``. This mirrors the serialization path while keeping HDS
    component names within the HDS object-name limit (`DAT__SZNAM`).
```

In `archive_path_to_hdf5_path_components`, reword the docstring sentence that currently reads:

```
    Each component is uppercased; components longer than the 16-character HDS
    limit are deterministically shrunk by `_shrink_hds_name`.
```
to:
```
    Each component is uppercased; components longer than the HDS object-name
    limit (`DAT__SZNAM`) are deterministically shrunk by `_shrink_hds_name`.
```

- [ ] **Step 3: Use the constant in `_output_archive.py`**

`python/lsst/images/ndf/_output_archive.py` already does `from . import _hds` (around line 50), so `_hds.DAT__SZNAM` is available with no new import.

Reword the comment in the `add_array` hoisted-array else-branch that currently reads:

```
            # to fit the 16-character HDS limit; repeated names get a version
```
to:
```
            # to fit the HDS object-name limit (DAT__SZNAM); repeated names get a version
```

In `_register_hdf5_path`, change the `ValueError` message so it interpolates the live limit. The current message is:

```python
            raise ValueError(
                f"NDF/HDS name collision: archive entries {previous!r} and {logical_id!r} "
                f"both map to {hdf5_path!r} after 16-character shrinking; rename one of them "
                f"or increase hash_size to reduce the hash-collision probability."
            )
```
Change it to:

```python
            raise ValueError(
                f"NDF/HDS name collision: archive entries {previous!r} and {logical_id!r} "
                f"both map to {hdf5_path!r} after shrinking to the {_hds.DAT__SZNAM}-character "
                f"HDS name limit; rename one of them or increase hash_size to reduce the "
                f"hash-collision probability."
            )
```

- [ ] **Step 4: Import the constant in `tests/test_ndf_common.py`**

The test module imports the shrinker symbols inside a `try` block. Add `DAT__SZNAM` from `_hds` to that block. The block currently is:

```python
try:
    from lsst.images.ndf._common import (
        NdfPointerModel,
        _shrink_hds_name,
        archive_path_to_hdf5_path,
        shrink_versioned_component,
    )

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False
```

Change it to add the `_hds` import (inside the same `try`, since `_hds` requires `h5py`):

```python
try:
    from lsst.images.ndf._common import (
        NdfPointerModel,
        _shrink_hds_name,
        archive_path_to_hdf5_path,
        shrink_versioned_component,
    )
    from lsst.images.ndf._hds import DAT__SZNAM

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False
```

- [ ] **Step 5: Update the assertions in `tests/test_ndf_common.py`**

In `test_archive_path_shrinks_long_components`, change:
```python
        self.assertLessEqual(len(leaf), 16)
```
to:
```python
        self.assertLessEqual(len(leaf), DAT__SZNAM)
```

In `test_short_names_pass_through_uppercased`, the current body is:
```python
    def test_short_names_pass_through_uppercased(self):
        self.assertEqual(_shrink_hds_name("psf"), "PSF")
        self.assertEqual(_shrink_hds_name("a" * 16), "A" * 16)
```
Replace it with one that exercises the boundary symbolically (a name exactly at the limit passes through; one over the limit is shrunk):
```python
    def test_short_names_pass_through_uppercased(self):
        self.assertEqual(_shrink_hds_name("psf"), "PSF")
        # A name exactly at the limit passes through unchanged (uppercased).
        self.assertEqual(_shrink_hds_name("a" * DAT__SZNAM), "A" * DAT__SZNAM)
        # One character over the limit is shrunk to the limit.
        self.assertEqual(len(_shrink_hds_name("a" * (DAT__SZNAM + 1))), DAT__SZNAM)
```

In `test_long_names_are_shrunk_to_the_limit`, the current body is:
```python
    def test_long_names_are_shrunk_to_the_limit(self):
        shrunk = _shrink_hds_name("noise_realizations")
        self.assertEqual(len(shrunk), 16)
        self.assertTrue(shrunk.startswith("NOISE_R"))
        self.assertEqual(shrunk, shrunk.upper())
```
Replace it with (the `startswith` prefix is limit-sensitive — at 15 the kept prefix is shorter, since `trunc = 15 - 2*4 - 1 = 6` keeps `"NOISE_"` not `"NOISE_R"` — so assert a shorter, limit-independent prefix):
```python
    def test_long_names_are_shrunk_to_the_limit(self):
        shrunk = _shrink_hds_name("noise_realizations")
        self.assertEqual(len(shrunk), DAT__SZNAM)
        self.assertTrue(shrunk.startswith("NOISE"))
        self.assertEqual(shrunk, shrunk.upper())
```

In `test_long_versioned_name_preserves_suffix_within_limit`, change:
```python
        shrunk = shrink_versioned_component("noise_realizations", 99)
        self.assertEqual(len(shrunk), 16)
        self.assertTrue(shrunk.endswith("_99"))
```
to:
```python
        shrunk = shrink_versioned_component("noise_realizations", 99)
        self.assertEqual(len(shrunk), DAT__SZNAM)
        self.assertTrue(shrunk.endswith("_99"))
```

- [ ] **Step 6: Import the constant in `tests/test_ndf_output_archive.py`**

This module imports the NDF public API inside a `try` block guarded by `HAVE_H5PY` (it imports `h5py` and `from lsst.images.ndf import (...)`). Add, inside that same `try` block, an import of the constant alongside the other `lsst.images.ndf` imports:

```python
    from lsst.images.ndf._hds import DAT__SZNAM
```

(If the existing import is e.g. `from lsst.images.ndf import (NdfInputArchive, NdfOutputArchive, ...)`, add the `_hds` line right after it, still inside the `try` so it is gated by `HAVE_H5PY`.)

- [ ] **Step 7: Update the assertions in `tests/test_ndf_output_archive.py`**

In `test_long_hoisted_component_is_shrunk`, change:
```python
                    self.assertLessEqual(len(component), 16)
```
to:
```python
                    self.assertLessEqual(len(component), DAT__SZNAM)
```

In `test_structured_array_long_name_is_shrunk_and_versioned`, change:
```python
                            self.assertLessEqual(len(component), 16)
```
to:
```python
                            self.assertLessEqual(len(component), DAT__SZNAM)
```

In `test_colliding_shrunk_names_raise`, the mock currently is:
```python
                    side_effect=lambda name, *a, **k: name.upper() if len(name) <= 16 else "CLASH",
```
Change the threshold to the constant so the stub mirrors the real passthrough rule:
```python
                    side_effect=lambda name, *a, **k: name.upper() if len(name) <= DAT__SZNAM else "CLASH",
```

- [ ] **Step 8: Run the full NDF + cell-coadd suite (expect all green)**

Run: `PYTHONPATH=./python ~/pyenv/bin/python -m pytest tests/test_ndf_common.py tests/test_ndf_output_archive.py tests/test_ndf_input_archive.py tests/test_ndf_layout.py tests/test_ndf_model.py tests/test_ndf_hds.py tests/test_ndf_format_version.py tests/test_ndf_starlink_ingest.py tests/test_cell_coadd.py -q`
Expected: all pass (cell-coadd tests skip if `TESTDATA_IMAGES_DIR`/`lsst.cell_coadds` are absent). NO failures. If a test fails, fix the assertion to key off `DAT__SZNAM` (do not revert the production change); the production limit of 15 is correct.

Sanity check the constant resolved into the defaults:
Run: `PYTHONPATH=./python ~/pyenv/bin/python -c "from lsst.images.ndf._hds import DAT__SZNAM; from lsst.images.ndf._common import _shrink_hds_name; print(DAT__SZNAM, len(_shrink_hds_name('noise_realizations')))"`
Expected: `15 15`.

- [ ] **Step 9: Lint and type-check**

Run: `~/pyenv/bin/ruff check python/lsst/images/ndf/_hds.py python/lsst/images/ndf/_common.py python/lsst/images/ndf/_output_archive.py tests/test_ndf_common.py tests/test_ndf_output_archive.py`
Run: `~/pyenv/bin/ruff format --check python/lsst/images/ndf/_hds.py python/lsst/images/ndf/_common.py python/lsst/images/ndf/_output_archive.py tests/test_ndf_common.py tests/test_ndf_output_archive.py` (if it reports changes, run without `--check` and re-stage)
Run: `PYTHONPATH=./python ~/pyenv/bin/python -m mypy python/lsst/images/ndf/_common.py python/lsst/images/ndf/_output_archive.py python/lsst/images/ndf/_hds.py` (ignore the 3 pre-existing `frozendict` errors)
Expected: clean (no new errors).

- [ ] **Step 10: Commit**

```bash
git add python/lsst/images/ndf/_hds.py python/lsst/images/ndf/_common.py python/lsst/images/ndf/_output_archive.py tests/test_ndf_common.py tests/test_ndf_output_archive.py
git commit -m "Define DAT__SZNAM and enforce the 15-char HDS name limit (DM-55183)"
```

---

## Task 2: Verify the limit change end-to-end

**Files:** none (verification only).

- [ ] **Step 1: Confirm no stray name-limit literal remains in the NDF package or its tests**

Run: `grep -rn "16-char\|16 char\|16-character\|16 characters\|<= 16\|== 16\|len.*16\|\* 16" python/lsst/images/ndf/ tests/test_ndf_common.py tests/test_ndf_output_archive.py`
Expected: NO matches that refer to the HDS name limit. (Unrelated `16`s such as `NDF_AST_DATA_MIN_WIDTH = 16`, `np.int16`, or `_WORD` mappings in `_hds.py` are fine and expected; the patterns above are narrowed to limit-style phrasings to avoid those. If a limit-style match remains, fix it under Task 1 and re-commit.)

- [ ] **Step 2: Confirm the constant is the single source and equals 15**

Run: `grep -rn "DAT__SZNAM" python/lsst/images/ndf/`
Expected: exactly one assignment (`DAT__SZNAM = 15` in `_hds.py`) plus import/use sites in `_common.py` and `_output_archive.py`. The literal `15` should appear only in that one assignment as a name-limit (nowhere else hard-coded).

- [ ] **Step 3: Run the full NDF suite, lint, and type-check together**

Run: `PYTHONPATH=./python ~/pyenv/bin/python -m pytest tests/test_ndf_common.py tests/test_ndf_output_archive.py tests/test_ndf_input_archive.py tests/test_ndf_layout.py tests/test_ndf_model.py tests/test_ndf_hds.py tests/test_ndf_format_version.py tests/test_ndf_starlink_ingest.py tests/test_cell_coadd.py -q`
Run: `~/pyenv/bin/ruff check python/lsst/images/ndf tests/test_ndf_common.py tests/test_ndf_output_archive.py`
Run: `PYTHONPATH=./python ~/pyenv/bin/python -m mypy python/lsst/images/ndf/_common.py python/lsst/images/ndf/_output_archive.py python/lsst/images/ndf/_hds.py`
Expected: tests pass (cell-coadd may skip), ruff clean, mypy clean apart from the 3 pre-existing `frozendict` stub errors.

- [ ] **Step 4: (Optional) Reproduce the real conversion at the new limit**

If `../cell_coadd.fits` is available, confirm the end-to-end conversion still works with the 15-char limit and produces only ≤15-char components:
Run: `rm -f /tmp/cell_szname.sdf; PYTHONPATH=./python ~/pyenv/bin/python bin/lsst-images-admin reformat ../cell_coadd.fits /tmp/cell_szname.sdf`
Expected: exits 0, "Wrote /tmp/cell_szname.sdf (ndf)."
Then verify every on-disk HDS component name is ≤ 15 characters:
Run:
```bash
PYTHONPATH=./python ~/pyenv/bin/python - <<'PY'
import h5py
over = []
with h5py.File("/tmp/cell_szname.sdf", "r") as f:
    def walk(g, p=""):
        for k in g.keys():
            if len(k) > 15:
                over.append(p + "/" + k)
            if isinstance(g[k], h5py.Group):
                walk(g[k], p + "/" + k)
    walk(f["/"])
print("components >15 chars:", over if over else "NONE")
PY
rm -f /tmp/cell_szname.sdf
```
Expected: `components >15 chars: NONE`. (If `../cell_coadd.fits` is not present, skip this step — the unit tests already prove the behavior.)

---

## Task 3: Update the design-of-record for the corrected limit

**Files:**
- Modify: `docs/superpowers/specs/2026-06-08-ndf-name-shrinker-design.md`

The committed design spec still says "16". Update it so the design-of-record matches the corrected limit. (The older plan `docs/superpowers/plans/2026-06-08-ndf-name-shrinker.md` is a historical, already-executed artifact and is intentionally left unchanged.)

- [ ] **Step 1: Update the spec text**

In `docs/superpowers/specs/2026-06-08-ndf-name-shrinker-design.md`, update each "16" reference to the HDS name limit so it reads as 15 / `DAT__SZNAM`:

- `HDS limits every component name to 16 characters.` → `HDS limits every component name to 15 characters (Starlink dat_par.h DAT__SZNAM).`
- `- Components at or under 16 characters pass through unchanged (uppercased only),` → `- Components at or under the limit (DAT__SZNAM, 15) pass through unchanged (uppercased only),`
- In the example error-message block, `after 16-character shrinking; increase hash_size.` → `after shrinking to the 15-character HDS name limit; rename one of them or increase hash_size.`
- `  16-character limit.` (in the version-suffix paragraph) → `  HDS name-length limit (DAT__SZNAM, 15).`

If any exact string differs slightly from the file, match on the surrounding sentence and update the "16" reference to "15" / `DAT__SZNAM` consistently; the intent is that the spec no longer states the limit as 16.

- [ ] **Step 2: Confirm no "16" name-limit reference remains in the spec**

Run: `grep -n "16" docs/superpowers/specs/2026-06-08-ndf-name-shrinker-design.md`
Expected: no remaining reference to the name limit as 16. (Unrelated numbers, if any, are fine.)

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-06-08-ndf-name-shrinker-design.md
git commit -m "Update name-shrinker design spec for the 15-char DAT__SZNAM limit (DM-55183)"
```
