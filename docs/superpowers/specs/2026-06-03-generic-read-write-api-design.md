# Generic `read` / `write` API Design

Ticket: DM-55131 (continuation)
Date: 2026-06-03

## Problem

The package already has the infrastructure to dispatch by file extension and
to inspect a file's schema without reading its pixel data:

* `backend_for_path(path)` returns a `Backend` carrying the per-format
  `read` / `write` functions and the `InputArchive` subclass.
* `InputArchive.get_basic_info(path)` returns an `ArchiveInfo` with
  `schema_name` / `schema_version` parsed from the file's headers.
* Every `ArchiveTree` (serialization-model) subclass declares
  `SCHEMA_NAME` / `SCHEMA_VERSION` as ClassVars.

What is missing is a way to go from a file on disk to the right in-memory
class with no explicit type argument from the caller, so users can write:

```python
from lsst.images.serialization import read, write
result = read("file.fits")        # result.obj is e.g. a VisitImage
write(result.obj, "out.json")
```

## Goals

* Provide top-level `read(path, **kwargs)` and `write(obj, path, **kwargs)`
  functions that dispatch by file extension and infer the in-memory type
  from the file's schema name and version.
* Keep the existing per-backend `read(cls, path, ...)` and `write(obj, path,
  ...)` entry points unchanged: `read` / `write` here are pure dispatchers.
* Extend the `inspect` CLI to report the public Python class derived from
  the registered schema, using `lsst.utils.introspection.get_full_type_name`
  for the displayed name.

## Non-goals

* Hoisting `read` / `write` to the top of `lsst.images`.  For now they
  live alongside `backend_for_path` in `lsst.images.serialization`; we can
  re-export them at the top level once we are happy with the API.
* Adding new file-format backends.  This work is purely additive on top of
  the existing FITS / NDF / JSON backends.
* Changing the per-backend `read` / `write` signatures or behavior.

## Architecture overview

A new module `python/lsst/images/serialization/_io.py` adds three public
symbols:

* `read(path, **kwargs) -> ReadResult[Any]` -- generic reader.
* `write(obj, path, **kwargs) -> Any` -- generic writer.
* `class_for_schema(name) -> type[ArchiveTree] | None` -- registry
  lookup.

These are exported by `lsst.images.serialization` via `from ._io import *`,
matching the existing module convention.

The schema-name registry is a module-level
`dict[str, type[ArchiveTree]]` populated in
`ArchiveTree.__pydantic_init_subclass__`, the same hook that already
injects `$id` / `title` into each subclass's JSON Schema.  Every concrete
`ArchiveTree` subclass is registered, including nested ones such as
`ImageSerializationModel` and `ProjectionSerializationModel`; nested types
are sometimes legitimate top-level reads.

The registry is keyed by ``SCHEMA_NAME`` only.  A serialisation model
typically handles multiple on-disk schema versions (this is the basis of
the schema-migration design: a single `VisitImageSerializationModel`
reads ``visit_image-1.0.0``, ``visit_image-2.0.0``, ...).  Compatibility
is enforced when the selected tree's ``model_validate*`` runs, via
``min_read_version`` checks that already exist in `ArchiveTree`.

The "public in-memory class" associated with a registered tree is derived
lazily from `typing.get_type_hints(tree_cls.deserialize)["return"]`, with
`typing.get_origin(t) or t` to unwrap parameterised generics so
`Transform[Any, Any]` becomes `Transform` and `Projection[Any]` becomes
`Projection`.  The resolved class is cached on the tree class on first
access (e.g. as a private attribute) so `get_type_hints` does not run on
every call.  When the annotation is `Any` or cannot be resolved, the
helper returns `None`.

### `read()` flow

1. `backend = backend_for_path(path)`.
2. `info = backend.input_archive.get_basic_info(path)`.
3. `tree_cls = class_for_schema(info.schema_name)`.  Raises
   `ArchiveReadError` with a clear message if `None`.
4. `return backend.read_tree(tree_cls, path, **kwargs)`.  Each backend
   parameterises ``tree_cls`` over its pointer model and dispatches to
   ``tree.deserialize(archive)``; the model's ``model_validate*`` checks
   ``min_read_version`` against the on-disk ``schema_version``.

### `write()` flow

`write(obj, path, **kwargs)` is just
`backend_for_path(path).write(obj, path, **kwargs)`.  No registry lookup
is needed because the per-backend `write` already accepts any object with
a `serialize` method.

## Registry mechanics

Registration happens in `ArchiveTree.__pydantic_init_subclass__` after
the existing JSON-Schema injection block:

* Subclasses without `SCHEMA_NAME` (intermediate abstract bases) are
  skipped, mirroring the existing guard.
* Re-registering the same class for the same name (re-import in tests)
  is a no-op.
* Registering a *different* class for an existing name raises
  `RuntimeError` at import time.  This matches the spirit of the
  existing schema-versioning invariants.

`class_for_schema(name) -> type[ArchiveTree] | None` is the public
lookup; missing keys return `None` so callers (especially `inspect`)
can render their own messaging.

## Public API signatures

```python
# python/lsst/images/serialization/_io.py

def read(
    path: ResourcePathExpression,
    **kwargs: Any,
) -> ReadResult[Any]:
    """Read an archive whose in-memory type is inferred from the file's
    schema.  Dispatches to the FITS / NDF / JSON backend by extension,
    then looks up the class registered for the file's schema (name,
    version).  Extra keyword arguments are forwarded to the per-backend
    ``read``.
    """

def write(
    obj: Any,
    path: str,
    **kwargs: Any,
) -> Any:
    """Write ``obj`` to ``path``, dispatching to the backend chosen by
    the file extension.  Extra keyword arguments are forwarded to the
    per-backend ``write`` (e.g. ``compression_options`` for FITS).
    """

def class_for_schema(
    schema_name: str,
) -> type[ArchiveTree] | None:
    """Return the registered ``ArchiveTree`` subclass for a schema, or
    ``None`` if no class is registered for that name.
    """
```

`read` returns a `ReadResult` namedtuple to match the existing per-backend
`read` and avoid introducing a confusingly different shape.

A small internal helper `_public_type(tree_cls) -> type | None` holds the
`get_type_hints` + `get_origin` derivation and the per-class cache.  It is
not exported.

## Errors

`read()` raises:

* `ValueError` from `backend_for_path` when the extension is not
  recognised (unchanged behavior).
* `ArchiveReadError` when the file's schema name is not registered:
  `"No registered schema {name!r}; cannot determine in-memory type for
  {path!r}."`
* `ArchiveReadError` from `model_validate*` when the file's
  ``min_read_version`` exceeds the registered class's
  ``SCHEMA_VERSION`` major.
* Whatever the per-backend `read_tree` raises (`ArchiveReadError`,
  I/O errors, etc.) for downstream failures.

`write()` raises only what the per-backend `write` raises; mismatched
extensions and I/O errors flow through unchanged.

## `inspect` CLI extension

A new line is added to the existing `inspect` output:

```
path:           /some/file.fits
format:         fits
schema name:    visit_image
schema version: 1.0.0
schema URL:     https://images.lsst.io/schemas/visit_image-1.0.0
format version: 1
python class:   lsst.images.VisitImage
```

Logic in `_inspect.py`:

1. After `info = backend.input_archive.get_basic_info(file)`, look up
   `tree_cls = class_for_schema(info.schema_name)`.
2. If `tree_cls` is `None`, render:
   `python class:   <unregistered: {schema_name}>`.
3. Otherwise compute `public_cls = _public_type(tree_cls)`.  If that is
   also `None`, render the same `<unregistered: ...>` form; from the
   user's point of view "we could not tell you what Python class this
   becomes" is the same outcome regardless of cause.
4. Otherwise render
   `python class:   {get_full_type_name(public_cls)}`.

`lsst.utils.introspection.get_full_type_name` is imported lazily inside
`inspect`, matching the existing pattern of keeping `_inspect.py`
imports light.

## Testing

Three test files, each with one focus.

### `tests/test_serialization_registry.py` (new)

* `class_for_schema("visit_image")` returns
  `VisitImageSerializationModel`.
* Lookup of an unknown name returns `None`.
* Class-invariants assertion: every concrete `ArchiveTree` subclass
  currently in the codebase appears in the registry.
* Class-invariants assertion: every registered class has a concrete,
  resolvable `deserialize` return annotation (or is in a known-`Any`
  allow-list, likely empty given the current code).
* Duplicate registration of a different class with the same
  ``SCHEMA_NAME`` raises `RuntimeError`.

### `tests/test_serialization_io.py` (new)

* Parametrise over every file in `tests/data/schema_v1/` (and the
  `legacy/` subdir).  For each fixture, call `result = read(path)` and
  assert `isinstance(result.obj, _public_type(class_for_schema(...)))`.
  This exercises the read path for every schema in the v1 fixture suite
  without requiring an explicit return class, complementing the existing
  type-specific tests.
* Round-trip an in-memory object through `write(obj, path)` then
  `read(path)` for `.fits`, `.json`, and `.sdf` extensions.
* An unregistered schema causes `read()` to raise `ArchiveReadError`
  with the documented message, simulated by patching the registry or by
  writing a JSON file with a fabricated `schema_url`.
* An unsupported extension causes `read("foo.bar")` to raise the same
  `ValueError` that `backend_for_path` already raises (no swallowing).
* Backend-specific `**kwargs` flow through: e.g.
  `read(visit_image_fits_path, bbox=...)` produces the expected subset.

### `tests/test_cli.py` (extend)

* For at least one FITS fixture and one JSON fixture, the `inspect`
  output contains
  `python class:   lsst.images.<ExpectedType>`.
* An unregistered schema renders
  `python class:   <unregistered: {name}>`.

The class-invariants test in `test_serialization_registry.py` is what
catches future regressions: any new `ArchiveTree` subclass that forgets
a concrete return annotation or collides on ``SCHEMA_NAME`` fails CI
immediately.

## Open questions

None.  Decisions captured above:

* Generic API lives in `lsst.images.serialization` only; top-level
  re-export deferred.
* Registry is keyed by ``SCHEMA_NAME`` (one class per schema name) and
  contains every `ArchiveTree` subclass.  Schema-version compatibility
  is enforced by the model's existing ``min_read_version`` check on
  ``model_validate*``, which lets a single ``VisitImageSerializationModel``
  read multiple on-disk versions.
* In-memory class is derived from `deserialize`'s return annotation; no
  new ClassVars are added to serialization models.
* `read` returns `ReadResult` (matching the per-backend signature).
* `write` is a thin dispatcher with no registry interaction.
