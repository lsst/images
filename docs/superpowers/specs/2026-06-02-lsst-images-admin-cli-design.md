# `lsst-images-admin` CLI design

Ticket: DM-55131

## Summary

Introduce a single `click`-based command, `lsst-images-admin`, that gathers the package's existing developer/test-data helpers under one entry point and adds a `convert` command for turning legacy `afw` files into the new `lsst.images` formats.
The command exposes four subcommands: `convert`, `inspect`, `minify`, and the existing `extract-test-data` group.
Two pieces of logic currently buried in the `minify` helper are promoted to reusable public APIs in the `serialization` layer so that `convert`, `inspect`, and `minify` all share them instead of duplicating them.

## Goals

- Provide one user-facing entry point, `lsst-images-admin`, installed via `[project.scripts]`.
- Add a `convert` subcommand: legacy `afw` file in, new-format `lsst.images` file out, with the output format chosen by the output file's extension.
- Add an `inspect` subcommand that reports a file's schema URL and format version, exercising the automatic read path.
- Surface the two existing helpers (`minify`, `extract-test-data`) under the same command without moving their implementation out of `tests/`.
- Refactor the suffix-to-reader logic and the schema-peeking logic out of `minify` into public, non-duplicated APIs.

## Non-goals

- Refactoring the butler `GenericFormatter` to consume the new suffix-to-backend helper (it keeps its own dispatch for now).
- Renaming `extract-test-data` to `extract-legacy-test-data` (reserved for a future ticket).
- A full structured/rich `inspect` output (dimensionality, per-extension detail such as PSF type); the first cut prints only schema URL and format version. See "Future work" for the intended direction.
- Supporting any top-level type beyond `VisitImage` and `CellCoadd` in `convert`.
- Reading legacy input in any format other than FITS; all legacy files are FITS.

## Command surface

```
lsst-images-admin
├── convert  INPUT OUTPUT [--type visit_image|cell_coadd]
│                         [--skymap PATH | --butler REPO]
│                         [--overwrite]
├── inspect  FILE
├── minify   INPUT OUTPUT [--schema-name NAME] [--overwrite]
└── extract-test-data            # existing click group, registered unchanged
    └── dp2  [...existing options...]
```

## Package layout

A new package owns the user-facing command surface:

```
python/lsst/images/cli/
  __init__.py     # __all__ = ("main",);  from ._main import main
  _main.py        # @click.group("lsst-images-admin"); registers all subcommands
  _convert.py     # the convert command
  _inspect.py     # the inspect command
  _minify.py      # thin click wrapper over tests._minify_for_fixtures.minify
  _common.py      # CLI-local helpers: lazy-import-with-hint, output-overwrite guard
```

Exposed in `pyproject.toml`:

```toml
dependencies = [ ..., "click >= 8" ]

[project.scripts]
lsst-images-admin = "lsst.images.cli:main"
```

`click` becomes a core dependency so the group and `--help` always load.
The heavy dependencies (`afw`, `cell_coadds`, `butler`, `h5py`) stay lazy: they are imported inside the command body and wrapped to raise a clear "install X" message only when the command that needs them actually runs.
This mirrors the existing `add_note` pattern in `tests/extract_legacy_test_data.py`.

`minify` and `extract-test-data` logic stays in `tests/` because it depends on test-only modules (`tests._creation`, `tests._data_ids`).
`cli/_main.py` imports and attaches their commands:

- `extract-test-data`: the existing `extract_test_data` group (with its `dp2` subcommand) is registered onto the root group under that name, unchanged.
- `minify`: `cli/_minify.py` is a thin click command that calls `tests._minify_for_fixtures.minify(in_path, out_path, schema_name=...)`.

## Refactors pulled out of `minify` (prerequisites)

These land in the `serialization` layer first; `minify` is then rewritten to consume them and its private copies are deleted.

### R1 — public reader/backend selection by suffix

A single public helper in the `serialization` package resolves a path's extension to its backend:

- `.fits` / `.fits.gz` → FITS backend
- `.sdf` / `.ndf` → NDF backend
- `.json` → JSON backend

The NDF backend is imported lazily so `h5py` stays optional.
The helper exposes the backend's `read` and `write` entry points so it serves both reading (`minify`, `inspect`) and writing (`convert`'s output side).

This replaces `minify._read_function` and the inline extension branching in `minify.minify`.
Placement is a `serialization`-level public function rather than an `InputArchive` classmethod, because it must span both read and write and must keep the NDF import lazy.
An unrecognised extension raises a clear error listing the supported extensions.

### R2 — per-backend basic-info accessor

An abstract classmethod on `InputArchive`, `get_basic_info(path) -> ArchiveInfo`, is implemented by `FitsInputArchive`, `NdfInputArchive`, and `JsonInputArchive`.
Each backend already knows where it *wrote* the top-level JSON tree and the format stamp, so this lives next to the writer rather than being duplicated in `minify`.

`ArchiveInfo` is a small frozen `pydantic` model (consistent with the rest of the package) carrying:

- `schema_url` — the canonical schema URL from the top-level tree.
- `schema_name` and `schema_version` — derived from `schema_url` (`.../schemas/{name}-{version}`).
- `format_version` — the container layout version (`FMTVER` for FITS, `FORMAT_VERSION` for NDF; `None` for JSON, which has no separate container version).

It reads only headers/metadata, not pixel data.
This deletes `minify._detect_schema_name`, `minify._peek_fits_top_json`, and `minify._peek_ndf_top_json`; `minify` calls `InputArchive.get_basic_info` (via the R1 backend) instead.

(`get_basic_info` is preferred over the narrower `get_schema_info` because `ArchiveInfo` already spans both schema and container-format fields and is the natural place to grow.)

## `convert`

```
lsst-images-admin convert INPUT OUTPUT
    [--type visit_image|cell_coadd]   # override auto-detection
    [--skymap PATH | --butler REPO]   # required only for cell coadds
    [--overwrite]
```

### Flow

`INPUT` is always a legacy FITS file, so convert reads it with the FITS legacy readers directly; R1 is used only to resolve the *output* backend.

1. Resolve the output backend from `OUTPUT`'s extension via R1 (`.fits` → FITS, `.sdf`/`.ndf` → NDF, `.json` → JSON).
2. Determine the legacy input type: use `--type` if given, otherwise auto-detect (see below).
3. Read the legacy FITS file:
   - `visit_image`: `VisitImage.read_legacy(INPUT)` with the default identity `plane_map`.
   - `cell_coadd`: `MultipleCellCoadd.read_fits(INPUT)`; obtain the skymap from `--skymap` (a pickled skymap) or `--butler` (looked up by the skymap name recorded in the coadd); then
     `CellCoadd.from_legacy(legacy, tract_info=skymap[tract], plane_map=get_legacy_deep_coadd_mask_planes())`.
4. Write the in-memory object with the resolved backend's `write(obj, OUTPUT)`.

### Legacy type auto-detection

`convert` reads *legacy* `afw` FITS files, which carry no `schema_url`/`FMTVER` stamp, so `get_basic_info` (R2) does not apply to convert input.
Detection instead inspects the legacy file's FITS headers with `astropy` (a core dependency), using the `HIERARCH LSST BUTLER DATASETTYPE` card: a dataset type ending in `visit_image` is a `VisitImage` (covering `visit_image`, `preliminary_visit_image`, and difference images, which are all `afw` exposures read through `VisitImage.read_legacy`), and one containing `coadd` is a `CellCoadd`.
When the header is absent or matches neither, `convert` errors and requires the `--type` option.
`--type` is always available as an explicit override.

Note that some legacy files written without butler provenance (such as the current `deep_coadd_cell_predetection.fits` test fixture) lack this header; converting those requires `--type`.

### Skymap source for cell coadds

A legacy coadd file records its skymap name and tract but not the tract geometry, which `CellCoadd.from_legacy` needs as `tract_info`.
`convert` accepts either:

- `--skymap PATH` — a pickled skymap file (the same `skyMap.pickle` that `extract-test-data` already produces); no butler dependency, or
- `--butler REPO` (together with `--collection`) — a butler repository, from which the skymap is resolved by the name recorded in the coadd within the given collection.

These options are required only when converting a cell coadd and are ignored for visit images.

## `inspect`

```
lsst-images-admin inspect FILE
```

The initial implementation prints `FILE`'s schema URL and format version, by resolving the backend with R1 and calling `get_basic_info` (R2).
This exercises the automatic read path without a full deserialize and without loading pixel data.
The output is structured so it can be extended later; the intended fuller form is described under "Future work".

## Error handling

- All three backend `write()` functions refuse to overwrite an existing file.
  `--overwrite` deletes the target first; without it, `convert`/`minify` fail early with a clear message rather than letting `write` raise from deep in the stack.
- Unrecognised output/input extension: error listing the supported extensions.
- Cell coadd convert with neither `--skymap` nor `--butler`: error explaining one is required.
- Missing heavy dependency (`afw`, `cell_coadds`, `butler`, `h5py`): the lazy import is caught and re-raised with the package/extra to install.
- Auto-detection failure (discriminating header absent): error instructing the user to pass `--type`.

## Future work

A fuller `inspect` should report the dataset's dimensionality and simple per-extension detail, such as the type of PSF held by a `VisitImage` or `CellCoadd`.
That information is not available from `get_basic_info` alone, so the straightforward path is to load the whole file and report from the in-memory object.

The longer-term direction avoids loading pixel data: read only the `pydantic` model (the JSON tree and its component descriptors) without the bulk arrays, and give the top-level types (`VisitImage`, `CellCoadd`, and similar) a `summarize()` method that returns a plain-text description for `inspect` to print.
This keeps inspection fast and low-memory.
The first cut deliberately does neither; it is scoped only to schema URL and format version so that the metadata-only read and the `summarize()` API can be designed properly in their own ticket.

## Testing

`tests/test_cli.py` using click's `CliRunner`:

- The root group and every subcommand's `--help` works with only core dependencies installed (no `afw`), guarding the lazy-import contract.
- `convert` of a visit image: gated on `afw` and `testdata_images`; convert a legacy file, read it back, and compare.
- `convert` of a cell coadd: gated on `cell_coadds` and `testdata_images`, using the extracted `skyMap.pickle`.
- Output-format-by-extension, `--overwrite` behaviour, error on an existing output file, and auto-detection picking the right type for both sample files.
- `inspect` on a sample file of each backend (FITS, NDF, JSON), asserting the reported schema URL and format version.

Unit coverage at the `serialization` level:

- R1: suffix → backend resolution, including the lazy NDF path and the unrecognised-extension error.
- R2: `get_basic_info` for each backend returns the expected `schema_url`, `schema_name`, `schema_version`, and `format_version`.
