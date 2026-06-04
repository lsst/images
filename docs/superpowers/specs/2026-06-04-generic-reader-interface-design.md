# Generic reader interface (`serialization.open`) — design

DM-55131

## Goal

Provide a user-friendly, path-based reader that opens an `lsst.images` file
once and lets the caller pull out individual components (or the whole
object) efficiently, sharing one open file handle and the archive's
pointer-dereference cache across calls:

```python
import lsst.images.serialization as ser

with ser.open("visit.fits") as reader:
    projection = reader.get_component("projection")
    obs_info = reader.get_component("obs_info")
```

`open()` is the incremental sibling of the existing `read()`: `read()`
returns the whole object in one call; `open()` hands back a reader for
pulling pieces. Neither requires the caller to know what an "archive" or a
"tree" is.

## Layering

Two layers, mirroring the existing `read()` (public) / `read_tree()` +
`InputArchive` (internal) split:

- **User-facing** — `serialization.open(path, cls=None, **backend_kwargs)`
  returns a `Reader[T]` context manager. Vocabulary is limited to
  components, the whole object, and identifying metadata. No archive/tree
  concepts appear in the signature or docstrings.
- **Internal** — `InputArchive.open_tree(path, tree_cls, ...)`, a classmethod
  context manager yielding `(archive, tree)`. This is the single
  open-and-load-tree primitive that `Reader`, the free `read()`/`read_tree()`,
  and `GenericFormatter` all build on. It replaces the three per-format
  `match`/`_get_archive_tree_type` sites that currently duplicate this logic.

## Public API

### `serialization.open`

```python
@overload
def open(path: ResourcePathExpression, cls: type[T], *, partial: bool = True, **backend_kwargs) -> AbstractContextManager[Reader[T]]: ...
@overload
def open(path: ResourcePathExpression, cls: None = None, *, partial: bool = True, **backend_kwargs) -> AbstractContextManager[Reader[Any]]: ...
```

- Dispatches to the FITS / NDF / JSON backend by file extension
  (`backend_for_path`).
- Reads `get_basic_info(path)` to obtain the file's `schema_name`, looks up
  the registered `ArchiveTree` subclass via `class_for_schema`. Raises
  `ArchiveReadError` when the schema is not registered.
- When `cls` is given, validates eagerly: if the schema's registered
  in-memory type (`public_type_for_schema`) is resolvable and is not a
  subclass of `cls`, raises before any deserialize. When the registered type
  cannot be statically resolved, validation falls through to the
  `isinstance` check in `Reader.read()`.
- `partial` and other `**backend_kwargs` (e.g. `page_size` for FITS) are
  forwarded to `InputArchive.open_tree`. Default `partial=True` matches the
  incremental intent of a reader; whole-object callers may pass
  `partial=False`.

### `Reader[T]`

A thin wrapper over `(archive, tree, info, expected_cls)`:

- `get_component(name: str, **kwargs) -> Any` — forwards to
  `tree.deserialize_component(name, archive, **kwargs)`. Raises
  `InvalidComponentError` for an unknown component and
  `InvalidParameterError` for unsupported `**kwargs` (existing behaviour of
  `deserialize_component`). Returns `Any`: components are heterogeneous.
- `read(**kwargs) -> T` — forwards to `tree.deserialize(archive, **kwargs)`,
  sets `_opaque_metadata` when the object supports it, and (when `cls` was
  given) asserts `isinstance(result, cls)`.
- `info: ArchiveInfo` — schema name/version/url and container format version,
  taken from the `get_basic_info` result used to resolve the class.
- `metadata` / `butler_info` — from the loaded tree (`tree.metadata`,
  `tree.butler_info`).

The reader is valid only inside its `with` block. After the context exits
the underlying file is closed; subsequent `get_component`/`read` calls raise
`RuntimeError("reader is closed")`.

No `list_components` method. The resolved Python class is available via the
existing `public_type_for_schema`; per-type component enumeration (e.g. on
`VisitImage` / `CellCoadd`) is deferred to future work.

### `read` gains `cls`

```python
@overload
def read(path: ResourcePathExpression, cls: type[T], **kwargs) -> ReadResult[T]: ...
@overload
def read(path: ResourcePathExpression, cls: None = None, **kwargs) -> ReadResult[Any]: ...
```

`read()` keeps its current shape — `get_basic_info` → `class_for_schema` →
`backend.read_tree(tree_cls, path, **kwargs)` — and `read_tree` now rests on
`open_tree` (below), so `read()` and `open()`/`Reader` share the single
`open_tree` primitive without `read()` physically calling the public
`open()`. (Routing `read()` through `open()` is rejected: `read(path,
**kwargs)` mixes open-time kwargs such as `page_size`/`partial` with
deserialize-time kwargs such as `bbox`, and only `read_tree` separates them;
`open()` does not deserialize and so cannot.) When `cls` is given, the
deserialized object is `isinstance`-checked against it and the static return
type is `ReadResult[T]` (no cast needed at the call site).

## Internal primitive

```python
class InputArchive:
    @classmethod
    @contextmanager
    def open_tree(cls, path, tree_cls, *, partial=True, **backend_kwargs) -> Iterator[tuple[Self, ArchiveTree]]:
        ...
```

`tree_cls` is the **un-parameterized** registry class. Each backend
parameterizes it with its own pointer model (`parameterize_tree`):

- **FITS** — `FitsInputArchive.open(path, page_size=..., partial=...)` →
  `archive.get_tree(parameterize_tree(tree_cls, PointerModel))`.
- **NDF** — `NdfInputArchive.open(path)` →
  `archive.get_tree(parameterize_tree(tree_cls, NdfPointerModel))`. `partial`
  is accepted and ignored (h5py reads lazily regardless).
- **JSON** — parse `tree_cls.model_validate_json(ResourcePath(path).read())`,
  build `JsonInputArchive(tree.indirect)`, yield. A no-resource context
  manager; `partial` is a no-op (fully in memory). FITS-only knobs such as
  `page_size` are not part of the JSON/NDF signatures.

## Rewiring

- **`read_tree()`** (free functions in `fits`/`json`/`ndf`) reduce to:
  `open_tree` → `tree.deserialize(...)` → set `_opaque_metadata` → wrap in
  `ReadResult`. The partial-defaulting heuristic (`partial=None` →
  `any(v is not None ...)`) stays in `read_tree`.
- **`read()`** keeps calling `backend.read_tree` (now on `open_tree`) and
  gains the `cls` parameter; it does not call the public `open()` (see the
  `read` gains `cls` section for why). Both `read()` and `open()` share the
  `open_tree` primitive.
- **`GenericFormatter.read_from_uri`** drops `_open_archive_and_tree` and the
  per-extension `match`, and uses the public `open()`:

  ```python
  def read_from_uri(self, uri, component=None, expected_size=-1):
      kwargs = self.file_descriptor.parameters or {}
      pytype = self.dataset_ref.datasetType.storageClass.pytype
      with ser.open(uri, cls=pytype, partial=bool(kwargs or component)) as reader:
          if component is None:
              return reader.read(**kwargs)
          return reader.get_component(component, **kwargs)
  ```

  This preserves current behaviour: `cls=pytype` performs the storage-class
  validation, `partial=bool(kwargs or component)` keeps today's memory
  trade-off, and `reader.read()` attaches `_opaque_metadata` exactly as the
  old path did.

## Error handling

- Unregistered schema → `ArchiveReadError` (matches `read()`).
- Unknown component → `InvalidComponentError`; bad component kwargs →
  `InvalidParameterError` (both from existing `deserialize_component`).
- `cls` mismatch → `TypeError` with a message naming the file's schema, the
  resolved type, and the requested `cls`. (Open question for review: fold
  this into the `ArchiveReadError` family instead, for catchability.)
- Use after context exit → `RuntimeError`.

## Code organisation

- New module `serialization/_reader.py`: `Reader`, `open`. Exported from
  `serialization/__init__` via `from ._reader import *`; `__all__` adds
  `Reader` and `open`. The public name deliberately shadows the builtin
  `open` only within the `serialization` namespace (`ser.open(...)`).
- `InputArchive.open_tree` added to the base class (abstract / `NotImplemented`)
  and implemented in each backend's `InputArchive`.
- `read`/`read_tree` updated in place.
- `formatters.py` simplified as above.

## Known limitations

- For JSON, `open()` parses the file twice (once in `get_basic_info` to
  resolve the schema, once in `open_tree`). This matches the existing
  `read()` behaviour and is acceptable: JSON is not the format for large
  pixel data. FITS/NDF read only a small header in `get_basic_info`.

## Testing

New `tests/test_serialization_reader.py`:

- For each backend (fits, json, ndf, NDF skipped without h5py), write a
  `VisitImage` fixture, then:
  - `open()` infers the type; `get_component("projection")`,
    `get_component("obs_info")`, and image-plane components return values
    equal to those from a full `read()`.
  - `reader.read()` returns an object equal to `read(path)`.
  - `reader.info` reports the expected `schema_name` / `schema_version`.
- Component caching: two `get_component` calls that dereference the same
  shared pointer (e.g. a PSF referenced twice) return the identical object.
- `open(path, cls=VisitImage)` succeeds; `open(path, cls=Mask)` on a
  visit-image file raises (eager schema check).
- `read(path, cls=Image)` returns the object; `read(path, cls=Mask)` on an
  image raises.
- Unknown component name raises `InvalidComponentError`.
- Using the reader after the `with` block raises `RuntimeError`.

Existing coverage:

- The `TemporaryButler` round-trip tests exercise the rewired
  `GenericFormatter.read_from_uri` (whole-object reads) for all three
  formats; confirm they still pass. Add a formatter component-read assertion
  if one is not already present.
- The static `cls` typing is exercised by typed usage in the new tests; CI
  mypy validates that no cast is required.

## Out of scope

- Per-type component enumeration / discovery methods.
- Writing through a reader (write stays via `write()` / the formatter).
- Eliminating the JSON double-parse.
