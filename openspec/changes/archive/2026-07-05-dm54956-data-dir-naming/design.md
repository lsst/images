## Context

Test files in `tests/` use the module-level constant `DATA_DIR` for two semantically
distinct purposes:

1. **External data** — `os.environ.get("TESTDATA_IMAGES_DIR", None)`, an optional
   path to real on-disk Rubin data; tests that use it are skipped when the env var
   is unset. Eight files use this pattern.

2. **Local data** — `os.path.join(os.path.dirname(__file__), "data", "schema_v1")`,
   an always-present path to committed JSON/FITS fixtures in the repo. Four files
   use this pattern.

`test_visit_image.py` already uses the unambiguous names `EXTERNAL_DATA_DIR` and
`LOCAL_DATA_DIR` (with a `TODO[DM-54956]` requesting that other files follow suit).
`test_difference_image_extras.py` also uses `EXTERNAL_DATA_DIR` but has a docstring
typo (`EXXTERNAL_DATA_DIR`).

This is a rename-only change within `tests/`. No production code is affected.

## Goals / Non-Goals

**Goals:**
- Every test file uses `EXTERNAL_DATA_DIR` for the env-var path.
- Every test file uses `LOCAL_DATA_DIR` for the in-repo `tests/data/schema_v1/` path.
- All use-sites (guards, `os.path.join` calls, docstrings) are updated consistently.
- The `TODO[DM-54956]` marker in `test_visit_image.py` (line 79) is removed.
- The docstring typo in `test_difference_image_extras.py` is fixed.

**Non-Goals:**
- No changes to production code under `python/`.
- No changes to the directory layout of `tests/data/`.
- No changes to how tests are skipped or to any test logic.

## Decisions

**Rename `DATA_DIR` (external) → `EXTERNAL_DATA_DIR` in 8 files.**
Rationale: the name already used in `test_visit_image.py` and `test_difference_image_extras.py`;
adopting it everywhere makes `grep EXTERNAL_DATA_DIR` a reliable way to find all
env-var-gated tests.

**Rename `DATA_DIR` (local) → `LOCAL_DATA_DIR` in 4 files.**
Rationale: the name already used in `test_visit_image.py`; symmetric with
`EXTERNAL_DATA_DIR`. An alternative (`SCHEMA_V1_DATA_DIR`) would be more specific
but unnecessarily verbose for an intra-file constant.

**One commit per file** is not required; all renames may land in a single commit
per this change. Tests must pass before committing.

## Risks / Trade-offs

[Risk] A docstring or comment that references the old name `DATA_DIR` is missed →
Mitigation: run `grep -r 'DATA_DIR' tests/` after all edits and verify no stale
references remain (the only surviving occurrences should be in this design doc and
the to-do list).

[Risk] Line numbers in the census file (`/sandbox/src/dm54956-todos.md`) drift if
files are edited before this change lands → Mitigation: the census is a guide, not
a contract; authors should re-verify line numbers at implementation time.
