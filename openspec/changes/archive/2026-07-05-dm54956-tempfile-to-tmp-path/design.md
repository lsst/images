## Context

89 `tempfile` call-sites exist across 13 test files, falling into three patterns:

**Pattern A — `tempfile.TemporaryDirectory` in test functions (19 sites, 7 files)**
Used as `with tempfile.TemporaryDirectory() as tmp: path = os.path.join(tmp, "x.ext")`.
Straightforward to replace: add `tmp_path: pathlib.Path` parameter, use
`tmp_path / "x.ext"` directly.

**Pattern B — `tempfile.NamedTemporaryFile` without `delete_on_close` (18 sites, 2 files)**
Used as `with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp: h5py.File(tmp.name, "w")`.
The NTF handle stays open while h5py opens the same path — works on Linux, fragile on
Windows. Replace with `path = tmp_path / "test.sdf"` passed to `h5py.File(path, "w")`.

**Pattern C — `tempfile.NamedTemporaryFile(delete_on_close=False)` (49 sites, 2 files)**
Used as:
```python
with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
    tmp.close()
    write(obj, tmp.name)
```
The `delete_on_close=False` + immediate `close()` is a workaround to get a path
without a live file handle. Replace with `path = tmp_path / "test.sdf"` directly.

**Pattern D — `tempfile.mkdtemp()` with no cleanup (1 site)**
`test_serialization_backends.py:76` calls `mkdtemp()` and never cleans up. This is
the highest-priority fix. Replace with `tmp_path` fixture.

**Pattern E — `tempfile.TemporaryDirectory` in non-fixture helper functions (3 sites)**
`_write_archive()` in `test_fits_output_archive.py` and `make_hdu_list()` /
`_cutdown()` in `test_from_hdu_list.py` are module-level helpers (not pytest
fixtures) that create their own temp directories. They cannot directly receive
`tmp_path` from pytest.

## Goals / Non-Goals

**Goals:**
- All `tempfile` usages inside test functions are replaced with `tmp_path`.
- The `mkdtemp` resource leak is eliminated.
- Helper functions in Pattern E accept a `tmp_path: pathlib.Path` parameter that
  callers (which are test functions) supply from their own `tmp_path` fixture.
- `import tempfile` is removed from every file where it is no longer referenced.

**Non-Goals:**
- No changes to test logic or assertions.
- No changes to production code.
- No refactoring of helpers into fixtures (beyond adding the `tmp_path` parameter).

## Decisions

**Pattern E helpers: add `tmp_path` parameter, do not convert to fixtures.**
Converting `_write_archive`, `make_hdu_list`, and `_cutdown` to pytest fixtures
would require changing every call site to use indirect fixture references or
`request.getfixturevalue`, which is more invasive than simply adding a parameter.
Adding `tmp_path: pathlib.Path` keeps the helpers as plain functions while
eliminating the internal `TemporaryDirectory`.

**File-by-file approach preferred over one large PR.**
Each test file is independent. Converting them one at a time makes review easier and
reduces the risk of a mistake in one file blocking another.

**NDF test files are the largest chunk.**
`test_ndf_output_archive.py` (29 sites) and `test_ndf_input_archive.py` (20 sites)
share the identical Pattern C idiom throughout. Within each file the replacements are
mechanical and uniform; they can be done efficiently with a search-and-replace within
the file.

## Risks / Trade-offs

[Risk] `tmp_path` creates a unique directory per test, so if two test functions in
the same file previously shared a `TemporaryDirectory` (unlikely given the Pattern A
usage) paths might diverge → Mitigation: all identified usages create a new temp
directory per call; no sharing.

[Risk] NDF tests open files with h5py by `tmp.name` (a string); `tmp_path / "x.sdf"`
is a `pathlib.Path` object. h5py and `write()` both accept `pathlib.Path`, so this
is safe, but confirm for each call site. → Mitigation: verify by running the NDF
test subset after conversion.

[Risk] `test_from_hdu_list.py` helpers are called from multiple test functions;
adding `tmp_path` as a parameter requires updating every call site → Mitigation:
the census identified exactly 2 helpers and their call patterns; update all call
sites in the same commit.
