# AGENTS.md — lsst-images

## Overview

Single Python package (`lsst.images`) in the Rubin Observatory / LSST Science Pipelines. Provides modern image types (`Image`, `MaskedImage`, `VisitImage`, `CellCoadd`) and a pluggable archive serialization system. Requires **Python ≥ 3.12**.

Source lives under `python/lsst/images/`; tests under `tests/`.

---

## Developer Commands

### Setup (once per session)
```bash
scons pkginfo version bin
```
Run this once at the start of a session. The LSST stack environment provides all dependencies (including optional ones like `piff`, `h5py`, `lsst.daf.butler`) via `PYTHONPATH`; pip/uv will miss those and should not be used. This command generates `python/lsst/images/pkginfo.py`, writes `python/lsst/images/version.py` from git tags, and wires up `bin/` — after which the package is fully importable in-place.

> **Do not use `pip install` or `uv pip install`** in an agent session. The stack environment is pre-configured; reinstalling via pip can shadow or break optional stack dependencies.

### Test
```bash
# Full suite (matches CI)
pytest -r a -v -n 3 --cov=lsst.images --cov-report=term

# Single file / class / method
pytest tests/test_image.py
pytest tests/test_image.py::ImageTestCase::test_basics
```
CI uses **3 workers** (`-n 3`); requires `pytest-xdist` for `-n`.

### Lint + format
```bash
ruff check --fix python/ tests/
ruff format python/ tests/
```
Or run all pre-commit hooks at once:
```bash
pre-commit run --all-files
```

### Type-check
```bash
mypy python/
```
CI enforces **100% fully-typed coverage** (`mypy-coverage` job). All new code in `lsst.images.*` must be fully annotated.

### Build docs
```bash
cd doc && package-docs clean && package-docs build -W -n
```

---

## Important Constraints

- **`python/lsst/images/version.py` is auto-generated** from git tags by `lsst-versions`. Never edit or commit it.
- **`__init__.py` files are excluded from ruff linting** (intentional; see `pyproject.toml [tool.ruff] exclude`).
- **Schema versioning is manual**: when an `ArchiveTree` Pydantic model changes shape, the developer must bump `SCHEMA_VERSION` (and possibly `MIN_READ_VERSION`) on the class. There is no automated enforcement.
- **`requirements.txt` must not pin `tickets/DM-*` branches** — the `do_not_merge` CI job blocks PRs that do.
- **No Makefile or task runner** — use the pytest/ruff/mypy/pre-commit commands above directly.

---

## Architecture Notes

### Public API
`lsst/images/__init__.py` star-imports from all submodules. Every public module declares `__all__` explicitly. All public modules use `from __future__ import annotations`.

### Subpackages
| Subpackage | Role |
|---|---|
| `lsst.images` | Core image/geometry types (`Image`, `MaskedImage`, `VisitImage`, `Box`, `Polygon`, `SkyProjection`, …) |
| `lsst.images.cells` | `CellCoadd`, `CellPointSpreadFunction`, `CoaddProvenance` |
| `lsst.images.psfs` | PSF types: `GaussianPointSpreadFunction`, `PiffWrapper`, `PSFExWrapper` |
| `lsst.images.fields` | Spatially-varying scalar fields (`ChebyshevField`, `SplineField`, …) |
| `lsst.images.cameras` | Detector/camera geometry |
| `lsst.images.serialization` | Abstract I/O: `ArchiveTree`, `read()`, `write()`, schema registry |
| `lsst.images.fits` / `.json` / `.ndf` | Format backends (FITS, JSON, Starlink NDF) |
| `lsst.images.formatters` | `GenericFormatter` for Butler integration |
| `lsst.images.cli` | Click admin CLI (`lsst-images-admin` / `python -m lsst.images`) |
| `lsst.images.tests` | Shared test helpers importable by downstream packages |

### Serialization
Every serializable class extends `ArchiveTree` (a `pydantic.BaseModel`) and declares:
- `SCHEMA_NAME`, `SCHEMA_VERSION` (semver), `MIN_READ_VERSION` (int), `PUBLIC_TYPE`

File format dispatched by extension: `.fits`, `.json`, `.sdf` (NDF requires `h5py`).

---

## Testing Quirks

### Optional-dependency skips
Tests auto-skip when optional packages are absent:
- `h5py` — NDF round-trip tests
- `piff` — Piff PSF tests
- `lsst.daf.butler` — Butler integration tests (`TemporaryButler` mixin)
- `lsst.afw.*`, `lsst.geom`, `lsst.cell_coadds` — legacy comparison tests (need full Rubin stack)

### Test helpers
- `lsst.images.tests._checks` — `assert_images_equal`, `compare_*_to_legacy`, etc.
- `lsst.images.tests._roundtrip` — `RoundtripFits`, `RoundtripJson`, `RoundtripNdf`, `TemporaryButler` mixins
- Converted test files use pytest free-functions and `@pytest.fixture`; legacy files still use `unittest.TestCase`

### Fixtures
- `tests/data/schema_v1/` — hand-committed reference JSON fixtures; `test_schema_v1_fixtures.py` auto-discovers them. Adding a new fixture file is sufficient — no generator changes needed.
- `tests/data/schema_v1/legacy/` — miniaturized real-data fixtures; regenerate via `lsst.images.tests._minify_for_fixtures.minify()` (requires real Rubin on-disk data).
- `TESTDATA_IMAGES_DIR` env var — optional path to real on-disk data for legacy-comparison tests.

---

## Style Conventions

- Line length: **110** (code), **79** (docstrings / doc max-line-length)
- Docstrings: **NumPy convention** (`numpydoc` validation runs in pre-commit)
- Imports sorted by ruff; `lsst`, `astshim`, `starlink` treated as first-party

#### Docstrings that exceed the 79-column limit
`ruff format` collapses closing `"""` onto the last line of text, but only when that line is already at the 79-column limit or below.  If a docstring would exceed column 79 as a single line, move one or more words to the next line and place the closing `"""` on a further new line — ruff will leave that form alone:

```python
# ✗ too long as a single line — ruff flags W505
def f():
    """Verify the mask round-trips through FITS without any data loss."""

# ✓ last word(s) moved to second line, closing quotes on their own line
def f():
    """Verify the mask round-trips through FITS without any data
    loss.
    """
```

### Test file style (pytest-converted files)
- **No comment-section headers** — do not add `# -----` banner/divider comments
  to group test functions. Function names and docstrings are the only
  navigation aid needed.
- **Imperative docstrings** — every `def test_*` function and `@pytest.fixture`
  must have a docstring whose first sentence opens with an imperative verb
  (`Verify`, `Test`, `Return`, `Assert`, …).  Declarative openings such as
  `"Repeated names get…"` or `"A freshly-written FITS carries…"` are not
  acceptable.  Note that ruff D401 catches some cases but not all; review
  manually.

---

## Git Workflow

- Every user-visible change needs a **towncrier news fragment** in `doc/changes/` named `<JIRA-TICKET>.<TYPE>.rst` (types: `feature`, `bugfix`, `api`, `perf`, `doc`, `removal`, `misc`).  These will be written by humans.
- `doc/lsst.images/CHANGES.rst` is auto-generated by towncrier; do not hand-edit it
- CI matrix: Python **3.12, 3.13, 3.14** — new code must be compatible with all three
- Agents work in a container sandbox with their own git clones on `sandbox/*` branches (usually `sandbox/DM-XXXXX`).
- A human outside the sandbox will push their `tickets/*` branch to these clones and pull the `sandbox/*` branches from them as needed.  A `tickets/*` branch should never be checked out within the sandbox, as this will prevent pushes from the human.
- When starting a new change, rebase the `/sandbox/*` branch on the `tickets/*` branch.
- After any OpenSpec step or other change you are asking a human to review, commit to the `/sandbox/*` branch; this is necessay to make your changes easily visible.
-
