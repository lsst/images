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

"""Python intermediate representation for zarr / xarray-CF / OME-NGFF content.

The IR is the source of truth for what gets written. ``ZarrOutputArchive``
populates a `ZarrDocument`; on context-manager exit, `to_zarr` materializes
it through a configured ``zarr.storage.Store``.

Reads invert that flow: ``ZarrInputArchive`` opens the store and calls
`ZarrDocument.from_zarr`, which builds the IR around **lazy** ``zarr.Array``
handles. No array bytes are read until a caller asks for them via
`ZarrArray.read`, which forwards slices straight to the underlying handle.
This keeps subset reads of remote files cheap: only the chunks intersecting
the requested slice are fetched.
"""

from __future__ import annotations

__all__ = (
    "CfFlagAttributes",
    "MaskPlaneEntry",
    "OmeMultiscale",
    "OmeOmeroChannel",
    "ZarrArray",
    "ZarrAttributes",
    "ZarrDocument",
    "ZarrGroup",
    "build_image_array_attrs",
)

from dataclasses import dataclass, field
from types import EllipsisType
from typing import Any, Self, cast

import numpy as np
import zarr
from zarr.abc.store import Store
from zarr.codecs import BloscCodec, BytesCodec

from ._common import LSST_NS, LSST_VERSION, OME_NS, OME_VERSION, ZarrCompressionOptions


@dataclass
class ZarrAttributes:
    """Namespaced attributes attached to a `ZarrGroup` or `ZarrArray`.

    Three namespaces:

    - ``lsst`` — LSST extensions (always emitted with a ``version`` key).
    - ``ome`` — OME-NGFF (emitted only when non-empty).
    - ``extra`` — flat top-level keys for CF / xarray conventions
      (``_ARRAY_DIMENSIONS``, ``flag_masks``, ``flag_meanings``,
      ``flag_descriptions``, ``units``, ``long_name``, …). These live at
      the top of ``zarr.json`` ``attributes`` so xarray and CF tooling
      see them without unwrapping a namespace.
    """

    lsst: dict[str, Any] = field(default_factory=dict)
    ome: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def dump(self) -> dict[str, Any]:
        """Return the raw mapping zarr-python writes to ``zarr.json``."""
        out: dict[str, Any] = dict(self.extra)
        # lsst is always present so readers can dispatch on lsst.archive_class.
        public_lsst = {k: v for k, v in self.lsst.items() if not k.startswith("__")}
        out[LSST_NS] = {"version": LSST_VERSION, **public_lsst}
        if self.ome:
            out[OME_NS] = {"version": OME_VERSION, **self.ome}
        return out

    @classmethod
    def load(cls, raw: dict[str, Any]) -> Self:
        """Construct from a raw attributes mapping read from zarr."""
        lsst = dict(raw.get(LSST_NS, {}))
        version = lsst.pop("version", None)
        if version is not None:
            # Stash the on-disk version under a private sentinel so the input
            # archive can validate without going back to the raw store.
            lsst["__version_remembered_at_load__"] = version
        ome = dict(raw.get(OME_NS, {}))
        ome.pop("version", None)
        extra = {k: v for k, v in raw.items() if k not in (LSST_NS, OME_NS)}
        return cls(lsst=lsst, ome=ome, extra=extra)


@dataclass
class ZarrArray:
    """An IR node holding either staged numpy data or a lazy zarr handle.

    Parameters
    ----------
    data
        Either a ``numpy.ndarray`` (when staged for write by the output
        archive) or a ``zarr.Array`` (when read by the input archive).
        The two forms never mix in a single instance.
    chunks
        Per-axis chunk shape. ``None`` lets `to_zarr` derive a fallback
        default for any IR node that reached the writer without explicit
        chunks (the output archive normally sets these via the
        `~lsst.images.zarr._layout.chunks_for` family of rules).
    shards
        Per-axis shard shape (zarr v3 native). ``None`` means the array
        is unsharded. Populated by `ZarrOutputArchive` via the
        `~lsst.images.zarr._layout.default_shards` rule for arrays large
        enough to benefit; tiny / single-chunk arrays stay ``None``.
    compression
        Codec configuration. ``None`` falls back to
        `ZarrCompressionOptions.default_for_dtype`.
    attributes
        Namespaced attributes for this array's ``zarr.json``.
    """

    data: np.ndarray | zarr.Array
    chunks: tuple[int, ...] | None = None
    shards: tuple[int, ...] | None = None
    compression: ZarrCompressionOptions | None = None
    attributes: ZarrAttributes = field(default_factory=ZarrAttributes)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.data.dtype)

    @classmethod
    def from_zarr(cls, zarr_array: zarr.Array) -> Self:
        """Wrap an open ``zarr.Array`` without reading its data."""
        attrs = ZarrAttributes.load(dict(zarr_array.attrs))
        # Mirror native zarr v3 ``dimension_names`` into the xarray v2-style
        # ``_ARRAY_DIMENSIONS`` attribute when only the v3 form is present,
        # so downstream consumers see both.
        dim_names = getattr(zarr_array.metadata, "dimension_names", None)
        if dim_names and "_ARRAY_DIMENSIONS" not in attrs.extra:
            attrs.extra["_ARRAY_DIMENSIONS"] = list(dim_names)
        return cls(
            data=zarr_array,
            chunks=tuple(zarr_array.chunks),
            attributes=attrs,
        )

    def read(self, *, slices: tuple[slice, ...] | EllipsisType = ...) -> np.ndarray:
        """Materialize this array (or a slice of it) into numpy.

        For a `ZarrArray` backed by a lazy handle, this is the only
        place that touches array bytes. ``slices`` is forwarded straight
        to the handle so only chunks intersecting the slice are fetched.
        """
        if isinstance(self.data, np.ndarray):
            return self.data if slices is ... else self.data[slices]
        result = self.data[...] if slices is ... else self.data[slices]
        return np.asarray(result)


@dataclass
class ZarrGroup:
    """A zarr group: nested groups, arrays, and namespaced attributes."""

    groups: dict[str, ZarrGroup] = field(default_factory=dict)
    arrays: dict[str, ZarrArray] = field(default_factory=dict)
    attributes: ZarrAttributes = field(default_factory=ZarrAttributes)

    def get(self, path: str) -> ZarrGroup | ZarrArray:
        """Return a child by absolute or relative zarr path."""
        if path in ("", "/"):
            return self
        parts = [p for p in path.strip("/").split("/") if p]
        cursor: ZarrGroup | ZarrArray = self
        for part in parts:
            if not isinstance(cursor, ZarrGroup):
                raise KeyError(path)
            if part in cursor.arrays:
                cursor = cursor.arrays[part]
            elif part in cursor.groups:
                cursor = cursor.groups[part]
            else:
                raise KeyError(path)
        return cursor

    def ensure_group(self, path: str) -> ZarrGroup:
        """Return or create a sub-group at ``path``."""
        if path in ("", "/"):
            return self
        parts = [p for p in path.strip("/").split("/") if p]
        cursor = self
        for part in parts:
            if part in cursor.arrays:
                raise KeyError(f"{part!r} already exists as an array.")
            if part not in cursor.groups:
                cursor.groups[part] = ZarrGroup()
            cursor = cursor.groups[part]
        return cursor


@dataclass
class ZarrDocument:
    """A complete zarr archive root."""

    root: ZarrGroup = field(default_factory=ZarrGroup)

    @classmethod
    def from_zarr(cls, store: Store) -> Self:
        """Open ``store`` and build a lazy IR view of its contents."""
        zarr_root = zarr.open_group(store=store, mode="r", zarr_format=3)
        return cls(root=_group_from_zarr(zarr_root))

    def to_zarr(self, store: Store) -> None:
        """Materialize this IR into ``store`` (which must be empty)."""
        zarr_root = zarr.create_group(
            store=store,
            zarr_format=3,
            overwrite=False,
            attributes=self.root.attributes.dump() or None,
        )
        _group_to_zarr(self.root, zarr_root)


def _group_from_zarr(zarr_group: zarr.Group) -> ZarrGroup:
    """Build a lazy `ZarrGroup` IR from an open ``zarr.Group``."""
    ir = ZarrGroup(attributes=ZarrAttributes.load(dict(zarr_group.attrs)))
    for name, child in zarr_group.members():
        if isinstance(child, zarr.Array):
            ir.arrays[name] = ZarrArray.from_zarr(child)
        else:
            ir.groups[name] = _group_from_zarr(child)
    return ir


def _group_to_zarr(ir: ZarrGroup, zarr_group: zarr.Group) -> None:
    """Write a `ZarrGroup` IR into an open ``zarr.Group``.

    The caller is responsible for creating ``zarr_group`` with its own
    attributes already populated (see `ZarrDocument.to_zarr`). This
    function only descends into children. Attributes are passed at
    create time rather than via a follow-up `update_attributes` so each
    ``zarr.json`` entry is written exactly once — required for append-
    only stores like ``ZipStore`` that emit duplicate-name warnings on
    rewrites.
    """
    for name, sub in ir.groups.items():
        sub_zarr = zarr_group.create_group(name, attributes=sub.attributes.dump() or None)
        _group_to_zarr(sub, sub_zarr)
    for name, array in ir.arrays.items():
        if not isinstance(array.data, np.ndarray):
            raise TypeError(
                f"Cannot write ZarrArray at {name!r}: data is a lazy zarr.Array, "
                "not numpy. Read it first or pass a fresh numpy array."
            )
        chunks = array.chunks or _default_chunks(array.data.shape)
        compression = array.compression or ZarrCompressionOptions.default_for_dtype(str(array.dtype))
        serializer, compressors = _build_codecs(compression)
        # Promote ``_ARRAY_DIMENSIONS`` from the CF-style attribute to the
        # native zarr v3 ``dimension_names`` metadata field; xarray's v3
        # backend reads from there, not from attributes, and refuses to
        # open the parent group if *any* array lacks the field. Arrays
        # without explicit names get distinct names derived from the array
        # name (``"<name>_<axis>"``): an anonymous ``None`` axis would be
        # shared with every other unnamed array, so xarray collapses them
        # onto one dimension and rejects the group when their sizes differ.
        dim_names = array.attributes.extra.get("_ARRAY_DIMENSIONS")
        if dim_names is None:
            dim_names = [f"{name}_{axis}" for axis in range(array.data.ndim)]
        else:
            dim_names = list(dim_names)
        zarr_array = zarr_group.create_array(
            name=name,
            shape=array.data.shape,
            chunks=chunks,
            dtype=array.data.dtype,
            shards=array.shards,
            serializer=serializer,
            compressors=compressors,
            dimension_names=dim_names,
            attributes=array.attributes.dump() or None,
        )
        zarr_array[:] = array.data


def _default_chunks(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return the default chunk shape: ``min(1024, dim)`` per axis."""
    return tuple(min(1024, dim) for dim in shape)


@dataclass
class OmeMultiscale:
    """OME-NGFF v0.5 multiscales metadata for a single-level image.

    The backend always writes one level whose ``path`` points at a
    sibling array (``image`` for typical archives).
    ``coordinate_transformations`` defaults to a unit ``scale`` so the
    OME block is well-formed even when the simplified affine is
    dropped by the residual validator.
    """

    name: str
    axes: tuple[str, ...]
    dataset_path: str = "image"
    coordinate_transformations: list[dict[str, Any]] | None = None

    @staticmethod
    def _axis_block(name: str) -> dict[str, Any]:
        if name == "c":
            return {"name": "c", "type": "channel"}
        if name == "t":
            return {"name": "t", "type": "time"}
        return {"name": name, "type": "space", "unit": "pixel"}

    def dump(self) -> dict[str, Any]:
        ndim = len(self.axes)
        ct = self.coordinate_transformations
        if ct is None:
            ct = [{"type": "scale", "scale": [1.0] * ndim}]
        return {
            "name": self.name,
            "axes": [self._axis_block(a) for a in self.axes],
            "datasets": [
                {
                    "path": self.dataset_path,
                    "coordinateTransformations": ct,
                }
            ],
        }


@dataclass
class OmeOmeroChannel:
    """OME ``omero/channels`` entry (used only when a channel axis exists)."""

    label: str
    color: str | None = None

    def dump(self) -> dict[str, Any]:
        out: dict[str, Any] = {"label": self.label}
        if self.color is not None:
            out["color"] = self.color
        return out


@dataclass
class MaskPlaneEntry:
    """One mask-plane definition."""

    name: str
    bit: int
    description: str = ""


@dataclass
class CfFlagAttributes:
    """CF-conventions flag metadata for a 2-D packed mask array.

    Emits ``flag_masks`` (list of bit values), ``flag_meanings``
    (single space-separated string per CF), and the LSST extension
    ``flag_descriptions`` (list of human-readable strings parallel to
    ``flag_meanings``).
    """

    planes: list[MaskPlaneEntry] = field(default_factory=list)

    def dump(self) -> dict[str, Any]:
        return {
            "flag_masks": [int(1 << p.bit) for p in self.planes],
            "flag_meanings": " ".join(p.name for p in self.planes),
            "flag_descriptions": [p.description for p in self.planes],
        }

    @classmethod
    def load(cls, raw: dict[str, Any]) -> Self:
        meanings = raw.get("flag_meanings", "").split()
        masks = [int(m) for m in raw.get("flag_masks", [])]
        descriptions = list(raw.get("flag_descriptions", [""] * len(meanings)))
        planes = []
        for name, mask, desc in zip(meanings, masks, descriptions, strict=False):
            # Recover bit position from the mask value (always a power of 2).
            bit = (mask & -mask).bit_length() - 1
            planes.append(MaskPlaneEntry(name=name, bit=bit, description=desc))
        return cls(planes=planes)


def build_image_array_attrs(
    *,
    axes: tuple[str, ...],
    units: str | None = None,
    long_name: str | None = None,
) -> dict[str, Any]:
    """Build the CF / xarray attribute block for an image array.

    Used for arrays of rank 2 or higher.
    """
    out: dict[str, Any] = {"_ARRAY_DIMENSIONS": list(axes)}
    if units is not None:
        out["units"] = units
    if long_name is not None:
        out["long_name"] = long_name
    return out


def _build_codecs(options: ZarrCompressionOptions) -> tuple[Any, list[Any]]:
    """Build a zarr v3 codec stack from `ZarrCompressionOptions`.

    Returns a ``(serializer, compressors)`` pair suitable for the
    ``serializer=`` and ``compressors=`` keyword arguments of
    `zarr.Group.create_array`.
    """
    if options.codec != "blosc":
        raise NotImplementedError(f"Unsupported codec {options.codec!r}.")
    serializer = BytesCodec()
    # ``cname`` and ``shuffle`` are typed as enum literals on BloscCodec;
    # at runtime any equivalent string is accepted, so cast through Any.
    compressors = [
        BloscCodec(
            cname=cast(Any, options.cname),
            clevel=options.clevel,
            shuffle=cast(Any, options.shuffle),
        )
    ]
    return serializer, compressors
