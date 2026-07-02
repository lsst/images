## Context

The project enforces 100% mypy coverage on all Python under `python/` and in `tests/`.
Two categories of annotation defect were found by the DM-54956 census:

**Category A — Wrong type on `tmp_path` (8 sites, 2 files)**
`pytest.MonkeyPatch` is the type of the `monkeypatch` fixture. `tmp_path` injects a
`pathlib.Path`. Using the wrong type means mypy silently accepts calls to
`MonkeyPatch` methods on what is actually a `Path` object — a latent bug.

| File | Occurrences |
|---|---|
| `test_serialization_io.py` | 6 functions (lines 86, 95, 130, 140, 151, 161) |
| `test_fuzz.py` | 2 functions (lines 91, 115) |

**Category B — Missing annotations (51 sites, 4 files)**
Sub-cases:
- `test_serialization_reader.py`: 1 fixture missing `-> VisitImage`; 15 test
  functions missing both `visit_image: VisitImage` and `tmp_path: pathlib.Path` on
  their parameters. Flagged by `TODO[DM-54956]` at line 66.
- `test_transforms.py`: 2 fixtures missing return types; 2 test functions missing
  fixture parameter types.
- `test_ndf_input_archive.py`: 23 test functions missing `-> None`.
- `test_ndf_output_archive.py`: 28 test functions missing `-> None`.

## Goals / Non-Goals

**Goals:**
- Every `tmp_path` parameter is annotated `pathlib.Path`.
- Every test function and fixture in the affected files is fully annotated.
- `mypy python/ tests/` reports no new errors after each file is changed.
- Resolved `TODO[DM-54956]` markers are removed.

**Non-Goals:**
- No changes to test logic or assertions.
- No changes to production code.
- This change does not attempt to annotate every test file in the suite — only the
  files identified by the census as deficient.

## Decisions

**Fix Category A and B in the same change.**
They are thematically identical (annotation completeness) and the affected files are
disjoint. Grouping them avoids a second round of mypy verification.

**Use `VisitImage` (not `Any`) for the `visit_image` fixture return type.**
The fixture in `test_serialization_reader.py` reads a JSON fixture and returns a
`VisitImage`. Using `Any` would defeat mypy's purpose; use the concrete type.
`VisitImage` is already in scope via `from lsst.images import ...` in that file
(or can be added).

**For `test_transforms.py` fixtures, use `Any` as a fallback for optional-dependency types.**
`legacy_camera_fixture` and `legacy_detector_wcs_fixture` return objects from
optional packages (`lsst.afw.cameraGeom`). If those types are unavailable, the
correct annotation is `Any` (matching the existing pattern used elsewhere in the
file for optional-dependency objects).

**Add `from pathlib import Path` where missing.**
Some affected files (`test_serialization_reader.py`, `test_ndf_input_archive.py`,
`test_ndf_output_archive.py`) may not currently import `pathlib.Path`. Add the import
if needed.

## Risks / Trade-offs

[Risk] `VisitImage` import not available at module level in `test_serialization_reader.py` →
Mitigation: check existing imports; add `from lsst.images import VisitImage` (or
use the existing `from lsst.images.serialization import ...` form) if needed.

[Risk] mypy may surface additional latent errors when annotations are added, beyond
the ones we already know about → Mitigation: run mypy per-file after each annotation
pass; resolve any newly visible errors before moving to the next file.
