# This file is part of lsst-images.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# Use of this source code is governed by a 3-clause BSD-style
# license that can be found in the LICENSE file.
"""Generic ``read`` / ``write`` dispatchers and the schema-name registry."""

from __future__ import annotations

__all__ = (
    "class_for_schema",
    "parameterize_tree",
    "public_type_for_schema",
    "read",
    "register_schema_class",
    "tree_class_for_info",
    "write",
)

import importlib
import importlib.metadata
from typing import TYPE_CHECKING, Any, overload

from lsst.resources import ResourcePathExpression

from ._backends import backend_for_path
from ._common import ArchiveReadError, ArchiveTree

if TYPE_CHECKING:
    from ._input_archive import ArchiveInfo

_REGISTRY: dict[str, type[ArchiveTree]] = {}
"""Map of ``SCHEMA_NAME`` to the registered ``ArchiveTree`` subclass.

The registry is keyed by name only.  Schema-version compatibility is
enforced when the selected tree's ``model_validate*`` runs, via
``min_read_version``.
"""

_SCHEMA_ENTRY_POINT_GROUP = "lsst.images.schemas"
"""Entry point group for third-party serialization-model providers."""

_BUILTIN_SCHEMA_PROVIDERS: dict[str, str] = {
    "cell_coadd": "lsst.images.cells._coadd:CellCoaddSerializationModel",
    "cell_psf": "lsst.images.cells._psf:CellPointSpreadFunctionSerializationModel",
    "coadd_provenance": "lsst.images.cells._provenance:CoaddProvenanceSerializationModel",
}
"""Schema providers owned by this package but not imported by ``lsst.images``.

These duplicate the package's own ``lsst.images.schemas`` entry points so
source-tree use via ``PYTHONPATH=python`` has the same lazy-import behavior as
an installed distribution with entry point metadata.

Schemas whose model classes are imported unconditionally by ``lsst.images`` do
not need built-in providers or entry points: their ``ArchiveTree`` subclass
hooks register them before this lazy path is needed.
"""


def class_for_schema(schema_name: str) -> type[ArchiveTree] | None:
    """Return the registered ``ArchiveTree`` subclass for ``schema_name``.

    If no class is already registered, this attempts schema-specific lazy
    imports from built-in providers and then from entry points in the
    ``lsst.images.schemas`` group before returning `None`.

    Parameters
    ----------
    schema_name
        Schema name (e.g. ``"visit_image"``).
    """
    if (cls := _REGISTRY.get(schema_name)) is not None:
        return cls
    _load_builtin_schema_provider(schema_name)
    if (cls := _REGISTRY.get(schema_name)) is not None:
        return cls
    _load_schema_entry_points(schema_name)
    return _REGISTRY.get(schema_name)


def register_schema_class(cls: type[ArchiveTree]) -> None:
    """Register ``cls`` under ``cls.SCHEMA_NAME``.

    No-op when the same class is registered again (re-import during
    tests).  Raises `RuntimeError` when a *different* class is
    registered under an existing name.

    Intended to be called from ``ArchiveTree.__pydantic_init_subclass__``;
    not part of the public API.
    """
    key = cls.SCHEMA_NAME
    existing = _REGISTRY.get(key)
    if existing is cls:
        return
    if existing is not None:
        raise RuntimeError(
            f"Schema {cls.SCHEMA_NAME!r} is already registered to "
            f"{existing.__qualname__}; refusing to replace it with "
            f"{cls.__qualname__}."
        )
    _REGISTRY[key] = cls


def _load_builtin_schema_provider(schema_name: str) -> None:
    """Import a package-local provider for ``schema_name``, if one exists."""
    provider = _BUILTIN_SCHEMA_PROVIDERS.get(schema_name)
    if provider is None:
        return
    try:
        obj = _load_provider_object(provider)
    except Exception as err:
        raise ArchiveReadError(
            f"Could not load built-in schema provider {provider!r} for schema {schema_name!r}: {err}"
        ) from err
    _register_provider_object(obj)
    if schema_name not in _REGISTRY:
        raise ArchiveReadError(
            f"Built-in schema provider {provider!r} did not register schema {schema_name!r}."
        )


def _load_schema_entry_points(schema_name: str) -> None:
    """Load entry points named ``schema_name`` from ``lsst.images.schemas``."""
    loaded: list[str] = []
    for entry_point in importlib.metadata.entry_points(
        group=_SCHEMA_ENTRY_POINT_GROUP,
        name=schema_name,
    ):
        loaded.append(entry_point.value)
        try:
            obj = entry_point.load()
        except Exception as err:
            raise ArchiveReadError(
                f"Could not load schema provider entry point {entry_point.value!r} "
                f"for schema {schema_name!r}: {err}"
            ) from err
        _register_provider_object(obj)
    if loaded and schema_name not in _REGISTRY:
        raise ArchiveReadError(
            f"Schema provider entry point(s) for {schema_name!r} did not register that schema: {loaded}."
        )


def _load_provider_object(provider: str) -> object:
    """Load ``module[:attribute[.nested]]`` provider specifications."""
    module_name, _, attr_path = provider.partition(":")
    obj: object = importlib.import_module(module_name)
    if attr_path:
        for attr in attr_path.split("."):
            obj = getattr(obj, attr)
    return obj


def _register_provider_object(obj: object) -> None:
    """Register ``obj`` if a provider returned an ``ArchiveTree`` subclass."""
    if isinstance(obj, type) and issubclass(obj, ArchiveTree):
        register_schema_class(obj)


def tree_class_for_info(info: ArchiveInfo, path: ResourcePathExpression) -> type[ArchiveTree]:
    """Return the registered `ArchiveTree` subclass for ``info``'s schema.

    Parameters
    ----------
    info
        Basic archive info whose ``schema_name`` selects the tree class.
    path
        Path being opened, used only for the error message.

    Raises
    ------
    ArchiveReadError
        If no class is registered for the schema.
    """
    tree_cls = class_for_schema(info.schema_name)
    if tree_cls is None:
        raise ArchiveReadError(f"No registered schema {info.schema_name!r}; cannot open {path!r}.")
    return tree_cls


def parameterize_tree(
    tree_cls: type[ArchiveTree],
    pointer_type: type[Any],
) -> type[ArchiveTree]:
    """Parameterise ``tree_cls`` over ``pointer_type`` if it is generic.

    Some `ArchiveTree` subclasses (e.g. ``SumFieldSerializationModel``)
    take no type parameters; their ``_get_archive_tree_type`` returns
    the class itself.  Match that behaviour here so per-backend
    ``open_tree`` implementations can call this uniformly.
    """
    if not getattr(tree_cls, "__parameters__", ()):
        return tree_cls
    return tree_cls[pointer_type]  # type: ignore[index]


def public_type_for_schema(schema_name: str) -> type | None:
    """Return the in-memory Python class produced when reading an archive
    whose top-level tree has schema name ``schema_name``.

    Looks the schema name up in the registry and returns the registered
    tree's ``PUBLIC_TYPE`` ClassVar (the type its ``deserialize`` produces).
    Returns `None` when nothing is registered for ``schema_name``.

    Parameters
    ----------
    schema_name
        Schema name (e.g. ``"visit_image"``).
    """
    tree_cls = class_for_schema(schema_name)
    if tree_cls is None:
        return None
    return getattr(tree_cls, "PUBLIC_TYPE", None)


@overload
def read[T](path: ResourcePathExpression, cls: type[T], **kwargs: Any) -> T: ...
@overload
def read(path: ResourcePathExpression, cls: None = ..., **kwargs: Any) -> Any: ...
def read(path: ResourcePathExpression, cls: type[Any] | None = None, **kwargs: Any) -> Any:
    """Read an archive whose in-memory type is inferred from its schema.

    Dispatches to the appropriate backend based on ``path``'s extension,
    resolves the registered in-memory type from the file's schema, and
    returns the fully deserialized object.
    Schema-version compatibility is enforced when the model validates the
    on-disk tree, via ``min_read_version``.

    This is the convenient way to read a whole object.  To read individual
    components, or to reach the metadata and butler info stored alongside the
    object, use `open` instead.

    Parameters
    ----------
    path
        File to read; convertible to `lsst.resources.ResourcePath`.
    cls
        Optional expected in-memory type.
        When given, the file's schema is checked against ``cls`` and the
        deserialized object is validated with ``isinstance`` (raising
        `TypeError` otherwise), and the static return type is ``T``.
    **kwargs
        Type-specific keyword arguments forwarded to the object's
        ``deserialize`` (e.g. ``bbox`` for an image subset read).
        Mis-targeted arguments surface as ``TypeError``.
        Backend-specific open options (e.g. ``page_size``) are not accepted
        here; use `open` for those.

    Returns
    -------
    object
        The deserialized object.

    Raises
    ------
    ValueError
        Raised by `backend_for_path` if the file extension is not recognized.
    ArchiveReadError
        Raised when the file's ``schema_name`` is not registered, or
        propagated from the model's ``min_read_version`` check on
        ``model_validate*``.
    TypeError
        Raised when ``cls`` is given and the file's schema or the
        deserialized object is not compatible with it.
    """
    # Imported here to break the _io <-> _reader import cycle: _reader imports
    # class_for_schema / public_type_for_schema from this module at load time.
    from ._reader import open as open_archive

    # A subset read (any deserialize kwarg with a value) reads incrementally;
    # a plain whole-object read may slurp the file up front.  This mirrors the
    # ``partial`` default the per-backend readers used.
    partial = any(value is not None for value in kwargs.values())
    with open_archive(path, cls, partial=partial) as reader:
        return reader.read(**kwargs)


def write(obj: Any, path: str, **kwargs: Any) -> Any:
    """Write ``obj`` to ``path``, dispatching by file extension.

    Forwards ``**kwargs`` to the per-backend ``write`` (e.g.
    ``compression_options`` for FITS).  No registry lookup is performed:
    the per-backend ``write`` already accepts any object with a
    ``serialize`` method.

    Parameters
    ----------
    obj
        Object to write; must implement ``serialize`` like the per-backend
        write functions expect.
    path
        Destination path.  The extension selects the backend.
    **kwargs
        Forwarded verbatim to the backend's ``write``.

    Returns
    -------
    Any
        Whatever the per-backend ``write`` returns (the serialised
        archive tree).
    """
    return backend_for_path(path).write(obj, path, **kwargs)
