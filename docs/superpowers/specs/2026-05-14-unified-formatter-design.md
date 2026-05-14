# Unified `lsst.images` butler formatter

**Date:** 2026-05-14
**Ticket:** DM-54817 (follow-up review comment)
**Status:** Design approved; implementation plan pending.

## Motivation

A review comment on the DM-54817 PR noted that the per-format butler formatters
(`lsst.images.fits.formatters`, `lsst.images.json.formatters`,
`lsst.images.ndf.formatters`) duplicate almost identical class hierarchies
that differ only in which backend `read`/`write` functions and which
input-archive class they call. The reviewer pointed to
`daf_butler/python/lsst/daf/butler/formatters/packages.py` as the precedent:
one formatter class that dispatches on a write-time `format` parameter and on
the file extension at read time.

Unifying the three formatters into one set of classes removes the duplication
and makes a butler config able to pick the on-disk format with a single
`format` write parameter rather than swapping the formatter class.

## Goals

1. Replace per-format formatter modules with a single
   `python/lsst/images/formatters.py` containing
   `GenericFormatter / ImageFormatter / MaskedImageFormatter /
   VisitImageFormatter / CellCoaddFormatter`.
2. Dispatch by `format` write parameter on write, by `uri.getExtension()` on
   read — same pattern as `PackagesFormatter`.
3. Preserve all current behavior: FITS compression recipes, NDF
   `MORE/FITS` header injection, JSON whole-tree provenance,
   component-level partial reads with bbox.
4. Keep the existing FITS-only formatter import path working as a deprecated
   shim so deployed test-repo configurations don't break in a single commit.

## Non-goals

- Adding a registry/plugin API for additional backends — three concrete
  formats is too few to justify the indirection. If a fourth format
  appears, that's when the registry pays off, not now.
- Renaming `FitsCompressionOptions` to something format-neutral — see
  "Future work" below.
- Touching `lsst.images.tests.Roundtrip{Fits,Json,Ndf}` — they exercise the
  format-specific `read`/`write` functions directly, not the formatter
  classes, so they remain.
- A news fragment is not added (per stakeholder direction).

## Architecture

### Single file: `python/lsst/images/formatters.py`

Contains the 5 formatter classes (`GenericFormatter`, `ImageFormatter`,
`MaskedImageFormatter`, `VisitImageFormatter`, `CellCoaddFormatter`) and the
`ComponentSentinel` enum, plus a private `_BACKENDS` table:

```python
_BACKENDS: dict[str, _Backend] = {
    ".fits": _Backend(read=fits.read, write=fits.write,
                      input_archive=FitsInputArchive,
                      pointer_model=FitsPointerModel),
    ".sdf":  _Backend(read=ndf.read,  write=ndf.write,
                      input_archive=NdfInputArchive,
                      pointer_model=NdfPointerModel),
    ".json": _Backend(read=json.read, write=json.write,
                      input_archive=None, pointer_model=None),
}
```

The `_Backend` rows for FITS and NDF carry an input-archive class for
component-level reads; the JSON row carries `None` because JSON does not
have a per-component archive concept.

The class hierarchy (`Generic → Image → MaskedImage → VisitImage / CellCoadd`)
is preserved. The hierarchy axis is "which set of components does the type
expose"; the new dispatch axis is "which on-disk format". They are orthogonal
and the formatters express only the type axis; the format axis is internal.

### Class attributes on `GenericFormatter`

Following the FormatterV2 contract and the `PackagesFormatter` precedent:

```python
default_extension     = ".fits"
supported_extensions  = frozenset({".fits", ".sdf", ".json"})
supported_write_parameters = frozenset({"format", "recipe"})
can_read_from_uri     = True
```

`recipe` remains in the union because it is supported for FITS output. It is
validated as FITS-only at write time (see below).

## Write path

```python
def get_write_extension(self) -> str:
    fmt = self.write_parameters.get("format", "fits")
    ext = "." + fmt
    if ext not in self.supported_extensions:
        raise RuntimeError(
            f"Requested format {fmt!r} is not supported; expected one of "
            f"{{fits, json, sdf}}."
        )
    return ext

def write_local_file(self, in_memory_dataset, uri):
    ext = self.get_write_extension()
    backend = _BACKENDS[ext]
    if ext != ".fits" and "recipe" in self.write_parameters:
        raise RuntimeError(
            "The 'recipe' write parameter is only valid for FITS output."
        )
    butler_info = ButlerInfo(
        dataset=self.dataset_ref.to_simple(),
        provenance=self.butler_provenance or DatasetProvenance(),
    )
    kwargs: dict[str, Any] = dict(butler_info=butler_info)
    if ext == ".fits":
        kwargs["update_header"] = self._update_header
        kwargs["compression_options"] = self._get_compression_options()
        kwargs["compression_seed"] = self._get_compression_seed()
    elif ext == ".sdf":
        kwargs["update_header"] = self._update_header  # NDF MORE/FITS
    backend.write(in_memory_dataset, uri.ospath, **kwargs)
```

- The `format` value uses the on-disk extension name (`fits`, `json`, `sdf`).
  `sdf` (not `ndf`) matches the file suffix and what `kappa`/`hdstrace`
  produce.
- The kwargs pruning is explicit. With three formats this is more legible
  than a clever generic plumbing layer.

Provenance handling is unchanged: `add_provenance` stashes
`self.butler_provenance`; `_update_header` (only called for FITS/NDF)
injects the existing `HIERARCH LSST BUTLER ...` cards. For JSON, provenance
already travels through the archive tree via `butler_info=`.

`_get_compression_options()` and `_get_compression_seed()` move verbatim from
`fits/formatters.py` to the unified `GenericFormatter`. They are only
consulted when `ext == ".fits"`.

## Read path

### Whole-object read

```python
def read_from_uri(self, uri, component=None, expected_size=-1):
    pytype = self.file_descriptor.storageClass.pytype
    ext = self._extension_from_uri(uri)
    backend = _BACKENDS[ext]
    if component is None:
        kwargs = self._collect_read_kwargs()  # bbox etc.
        return backend.read(pytype, uri, **kwargs).deserialized
    return self._read_component(component, uri, ext, backend)
```

`_extension_from_uri()` calls `uri.getExtension()` and matches it against
`supported_extensions`. We support exactly `.fits`, `.sdf`, and `.json`;
no compression-suffix normalization is performed (matching the
`PackagesFormatter` convention). Unknown or composite extensions
(`.fits.gz`, `.sdf.z`, etc.) raise
`RuntimeError(f"Cannot read {uri}: unsupported extension {got!r}.")`. If
gzipped FITS appears in real usage, the `_BACKENDS` table can be extended
later; not addressed here.

### Component-level read

The four subclasses (`ImageFormatter`, `MaskedImageFormatter`,
`VisitImageFormatter`, `CellCoaddFormatter`) own a single `read_component`
method each — they collapse to one per class because the per-format
`read_component` methods today differ only by archive type, and both
`FitsInputArchive` and `NdfInputArchive` already implement the same
`ArchiveTree` / `deserialize_pointer` protocol used by the body of those
methods.

Per-extension dispatch in `_read_component`:

| Extension | Mechanism |
|---|---|
| `.fits` | `FitsInputArchive(uri, partial=True).get_tree(pytype._get_archive_tree_type(FitsPointerModel))` then `self.read_component(component, tree, archive)`. |
| `.sdf`  | `NdfInputArchive(uri).get_tree(pytype._get_archive_tree_type(NdfPointerModel))` then same `read_component`. |
| `.json` | `json.read(pytype, uri, **kwargs).deserialized` then `getattr(obj, component)`. This path does *not* go through `read_component` / `ComponentSentinel`; component names that don't exist as attributes on the deserialized object raise `AttributeError`, which `_read_component` catches and re-raises as `NotImplementedError(f"Unrecognized component {component!r} for JSON read.")` so callers see the same exception type as for the FITS/NDF paths. |

The `pop_bbox_from_parameters` / `check_unhandled_parameters` helpers on
`ImageFormatter` are unchanged.

`ComponentSentinel.UNRECOGNIZED_COMPONENT` and `INVALID_COMPONENT_MODEL`
move into `formatters.py` and are imported from there by the four classes;
their semantics are unchanged.

## Backwards-compat

- **`python/lsst/images/fits/formatters.py`** — replaced with a thin
  shim. Each of the five classes becomes a one-line subclass of the unified
  counterpart that emits `DeprecationWarning` on first construction:

  ```python
  from .. import formatters as _unified

  __all__ = (
      "CellCoaddFormatter", "GenericFormatter", "ImageFormatter",
      "MaskedImageFormatter", "VisitImageFormatter",
  )

  def _warn(name: str) -> None:
      warnings.warn(
          f"lsst.images.fits.formatters.{name} is deprecated; "
          f"use lsst.images.formatters.{name}.",
          DeprecationWarning,
          stacklevel=3,
      )

  class GenericFormatter(_unified.GenericFormatter):
      def __init__(self, *a, **kw):
          _warn("GenericFormatter")
          super().__init__(*a, **kw)
  # ...four more, same shape
  ```

  Behavior is identical; deployed test-repo butler configs continue to work
  with a single deprecation warning per dataset write.

- **`python/lsst/images/json/formatters.py`** — deleted. Never used in
  deployed configurations.

- **`python/lsst/images/ndf/formatters.py`** — deleted. Never deployed.

## Testing

- **Move:** existing FITS-formatter tests to `tests/test_formatters.py` and
  parameterize over `(format, extension)` in
  `[("fits", ".fits"), ("sdf", ".sdf"), ("json", ".json")]`. The bbox /
  component-read cases run for `.fits` and `.sdf` only; the JSON
  component-read case asserts the whole-object fallback returns the
  equal attribute.
- **Add:** unit test for `get_write_extension()`: defaults to `.fits`;
  accepts `fits|json|sdf`; raises on unknown; raises on `recipe` with
  non-`fits` format.
- **Add:** unit test for read-side extension routing: `.fits`,
  `.fits.gz`, `.sdf`, `.json` route to the right backend; unknown
  extension raises with a clear message.
- **Add:** regression test for the deprecation shim — importing
  `lsst.images.fits.formatters.ImageFormatter` and instantiating it
  emits a `DeprecationWarning` and otherwise behaves identically to the
  unified class.
- **Skipping**: NDF-touching cases are decorated
  `@unittest.skipUnless(HAVE_H5PY, ...)` to match the convention
  established when h5py became optional.

## Risk surface

- **Merged `read_component` may inadvertently use format-specific archive
  methods.** Mitigated by parameterizing every component test path over
  both `FitsInputArchive` and `NdfInputArchive`.
- **`recipe` accidentally forwarded to non-FITS write functions.**
  Mitigated by the explicit kwargs prune and the explicit error when
  `recipe` is set with `format != fits`.
- **Existing butler configs pointing at the FITS-only path.** Mitigated by
  the deprecation shim; one warning per write, no behavior change.

## Future work (out of scope here, but flagged)

- **`FitsCompressionOptions` naming/concepts.** This type and the
  `recipe` write parameter are FITS-only abstractions, but with a
  unified formatter they appear in a class that may also produce
  `.sdf` or `.json` output. Renaming and reshaping
  `FitsCompressionOptions` to either (a) a format-neutral
  `CompressionOptions` superclass with FITS/HDF5 subclasses, or (b)
  keep it FITS-specific but move its visibility (so it doesn't read as
  the unified formatter's "compression options") would help readers
  understand which knobs apply to which format. The compression-recipe
  config dict structure should probably also generalize; the unified
  formatter today only checks the FITS recipe table.

  This is its own design discussion and should be a separate ticket. If
  the unified-formatter PR exposes confusion before that ticket lands, a
  short docstring note on the `recipe` write parameter (limiting it to
  FITS output) will suffice.

- **Registry/plugin API for additional backends.** Worth revisiting only
  when a fourth concrete format is on the roadmap.

- **`format=ndf` synonym.** If users find `format=sdf` unintuitive given
  the in-code module name `lsst.images.ndf`, a `format=ndf` alias can be
  added. Not added preemptively.
