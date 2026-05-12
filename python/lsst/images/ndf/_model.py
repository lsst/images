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

"""Python intermediate representation for NDF/HDS content."""

from __future__ import annotations

__all__ = (
    "HdsExtension",
    "HdsPrimitive",
    "HdsStructure",
    "Ndf",
    "NdfArray",
    "NdfContainer",
    "NdfDocument",
    "NdfQuality",
    "NdfWcs",
)

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Self

import h5py
import numpy as np

from . import _hds


def _decode_ascii_attr(value: Any) -> str | None:
    """Decode an HDS ASCII attribute value from h5py."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("ascii")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("ascii")
    if isinstance(value, str):
        return value
    return str(value)


@dataclass
class HdsPrimitive:
    """An HDS primitive component.

    Parameters
    ----------
    data
        Numeric/logical array data, or an open HDF5 dataset when this
        primitive was read lazily from an existing file.
    char_lines
        Character data for ``_CHAR*N`` primitives. A scalar string is
        represented as a one-element list with ``char_scalar=True``.
    char_width
        Fixed HDS character width for ``char_lines``.
    is_char_scalar
        If `True`, write ``char_lines`` as a scalar ``_CHAR*N`` primitive.
        Otherwise write them as a one-dimensional character array.
    compression_options
        Optional h5py compression options for numeric array data.
    """

    data: np.ndarray | h5py.Dataset | None = None
    char_lines: list[str] | None = None
    char_width: int | None = None
    is_char_scalar: bool = False
    compression_options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def array(
        cls,
        data: np.ndarray | h5py.Dataset,
        *,
        compression_options: Mapping[str, Any] | None = None,
    ) -> Self:
        """Create a numeric/logical HDS primitive."""
        return cls(
            data=data if isinstance(data, h5py.Dataset) else np.asarray(data),
            compression_options=dict(compression_options) if compression_options else {},
        )

    @classmethod
    def char_array(cls, lines: Sequence[str], *, width: int = 80) -> Self:
        """Create a one-dimensional HDS ``_CHAR*N`` primitive."""
        return cls(char_lines=list(lines), char_width=width)

    @classmethod
    def char_scalar(cls, text: str, *, width: int | None = None) -> Self:
        """Create a scalar HDS ``_CHAR*N`` primitive."""
        encoded = text.encode("ascii")
        return cls(char_lines=[text], char_width=max(width or 0, len(encoded), 1), is_char_scalar=True)

    @classmethod
    def from_hdf5(cls, dataset: h5py.Dataset) -> Self:
        """Build an HDS primitive model from an open HDF5 dataset."""
        if dataset.dtype.kind == "S":
            if dataset.ndim == 0:
                raw = dataset[()]
                if isinstance(raw, np.bytes_):
                    raw = bytes(raw)
                assert isinstance(raw, bytes)
                return cls(
                    char_lines=[raw.decode("ascii").rstrip(" ")],
                    char_width=dataset.dtype.itemsize,
                    is_char_scalar=True,
                )
            return cls(
                char_lines=_hds.read_char_array(dataset),
                char_width=dataset.dtype.itemsize,
                is_char_scalar=False,
            )
        return cls(data=dataset)

    def read_array(self) -> np.ndarray:
        """Return this primitive as a numpy array."""
        if self.data is None:
            raise TypeError("Character HDS primitives cannot be read as numeric arrays.")
        if isinstance(self.data, h5py.Dataset):
            return _hds.read_array(self.data)
        return self.data

    def read_char_array(self) -> list[str]:
        """Return this primitive as stripped ASCII strings."""
        if self.char_lines is not None:
            return list(self.char_lines)
        if isinstance(self.data, h5py.Dataset) and self.data.dtype.kind == "S":
            if self.data.ndim == 0:
                raw = self.data[()]
                if isinstance(raw, np.bytes_):
                    raw = bytes(raw)
                assert isinstance(raw, bytes)
                return [raw.decode("ascii").rstrip(" ")]
            return _hds.read_char_array(self.data)
        raise TypeError("Numeric HDS primitives cannot be read as character arrays.")

    def write_to_hdf5(self, parent: h5py.Group, name: str) -> h5py.Dataset:
        """Write this primitive to an HDF5 group."""
        if name in parent:
            del parent[name]
        if self.char_lines is not None:
            width = self.char_width
            if width is None:
                width = max((len(line.encode("ascii")) for line in self.char_lines), default=1)
            if self.is_char_scalar:
                if len(self.char_lines) != 1:
                    raise ValueError("Scalar _CHAR*N primitives require exactly one string.")
                line = self.char_lines[0]
                encoded = line.encode("ascii")
                if len(encoded) > width:
                    raise ValueError(
                        f"Scalar _CHAR*N primitive {name!r} is {len(encoded)} bytes, "
                        f"longer than width={width}."
                    )
                return parent.create_dataset(name, data=np.bytes_(encoded.ljust(width)))
            return _hds.write_char_array(parent, name, self.char_lines, width=width)
        data = self.read_array()
        return _hds.write_array(
            parent,
            name,
            data,
            compression=self.compression_options.get("compression"),
            compression_opts=self.compression_options.get("compression_opts"),
        )


class HdsStructure:
    """An HDS structure component with named child components."""

    def __init__(
        self,
        hds_type: str,
        children: Mapping[str, HdsStructure | HdsPrimitive] | None = None,
    ) -> None:
        self.hds_type = hds_type
        self.children: dict[str, HdsStructure | HdsPrimitive] = dict(children) if children else {}

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> HdsStructure:
        """Build a structure model from an open HDF5 group."""
        hds_type = _decode_ascii_attr(group.attrs.get(_hds.ATTR_CLASS))
        if hds_type is None:
            hds_type = _decode_ascii_attr(group.attrs.get("HDSTYPE")) or "EXT"
        structure = _new_structure(hds_type)
        for name, child in group.items():
            if isinstance(child, h5py.Group):
                structure.children[name] = cls.from_hdf5(child)
            elif isinstance(child, h5py.Dataset):
                structure.children[name] = HdsPrimitive.from_hdf5(child)
        return structure

    def __contains__(self, path: str) -> bool:
        try:
            self.get(path)
        except KeyError:
            return False
        return True

    def __getitem__(self, path: str) -> HdsStructure | HdsPrimitive:
        return self.get(path)

    def items(self) -> Iterable[tuple[str, HdsStructure | HdsPrimitive]]:
        """Iterate over direct child components."""
        return self.children.items()

    def get(self, path: str) -> HdsStructure | HdsPrimitive:
        """Return a child component by relative or absolute path."""
        if path in ("", "/"):
            return self
        cursor: HdsStructure | HdsPrimitive = self
        for part in _split_path(path):
            if not isinstance(cursor, HdsStructure):
                raise KeyError(path)
            cursor = cursor.children[part]
        return cursor

    def get_structure(self, path: str) -> HdsStructure:
        """Return a child structure by relative or absolute path."""
        node = self.get(path)
        if not isinstance(node, HdsStructure):
            raise KeyError(f"{path!r} is an HDS primitive, not a structure.")
        return node

    def set(self, path: str, node: HdsStructure | HdsPrimitive) -> None:
        """Set a child component by relative or absolute path."""
        parts = _split_path(path)
        if not parts:
            raise ValueError("Cannot replace an HDS structure with itself.")
        parent = self.ensure_structure("/".join(parts[:-1]), "EXT")
        parent.children[parts[-1]] = node

    def delete(self, path: str) -> None:
        """Delete a child component if it exists."""
        parts = _split_path(path)
        if not parts:
            raise ValueError("Cannot delete an HDS structure root.")
        parent = self.get_structure("/".join(parts[:-1]))
        parent.children.pop(parts[-1], None)

    def ensure_structure(self, path: str, hds_type: str = "EXT") -> HdsStructure:
        """Return an existing structure or create it and its parents."""
        if path in ("", "/"):
            return self
        cursor: HdsStructure = self
        for index, part in enumerate(_split_path(path)):
            existing = cursor.children.get(part)
            if existing is None:
                child_type = hds_type if index == len(_split_path(path)) - 1 else "EXT"
                existing = _new_structure(child_type)
                cursor.children[part] = existing
            if not isinstance(existing, HdsStructure):
                raise KeyError(f"{part!r} already exists as an HDS primitive.")
            cursor = existing
        if cursor.hds_type != hds_type:
            cursor.hds_type = hds_type
        return cursor

    def write_to_hdf5(self, parent: h5py.Group, name: str | None = None) -> h5py.Group:
        """Write this structure to an HDF5 group."""
        if name is None:
            group = parent
            _clear_hdf5_group(group)
            _hds.set_ascii_attr(group, _hds.ATTR_CLASS, self.hds_type)
        else:
            if name in parent:
                del parent[name]
            group = _hds.create_structure(parent, name, self.hds_type)
        for child_name, child in self.children.items():
            if isinstance(child, HdsStructure):
                child.write_to_hdf5(group, child_name)
            else:
                child.write_to_hdf5(group, child_name)
        return group


class HdsExtension(HdsStructure):
    """A general-purpose HDS extension structure."""

    def __init__(self, children: Mapping[str, HdsStructure | HdsPrimitive] | None = None) -> None:
        super().__init__("EXT", children)


class NdfContainer(HdsStructure):
    """A top-level HDS container for multiple NDFs and shared metadata."""

    def __init__(self, children: Mapping[str, HdsStructure | HdsPrimitive] | None = None) -> None:
        super().__init__("EXT", children)

    def ensure_ndf(self, path: str) -> Ndf:
        """Return or create a child NDF at ``path``."""
        structure = self.ensure_structure(path, "NDF")
        if not isinstance(structure, Ndf):
            structure.__class__ = Ndf
        return structure


@dataclass
class NdfArray:
    """An NDF ARRAY component."""

    data: np.ndarray | h5py.Dataset
    origin: np.ndarray | Sequence[int] | None = None
    bad_pixel: bool | None = None
    compression_options: dict[str, Any] = field(default_factory=dict)

    def to_hds_structure(self) -> HdsStructure:
        """Convert this array component to an HDS ``ARRAY`` structure."""
        structure = HdsStructure("ARRAY")
        structure.children["DATA"] = HdsPrimitive.array(
            self.data,
            compression_options=self.compression_options,
        )
        if self.origin is not None:
            structure.children["ORIGIN"] = HdsPrimitive.array(np.asarray(self.origin))
        if self.bad_pixel is not None:
            structure.children["BAD_PIXEL"] = HdsPrimitive.array(np.array(self.bad_pixel, dtype=np.bool_))
        return structure

    @classmethod
    def from_hds_structure(cls, structure: HdsStructure) -> Self:
        """Build an `NdfArray` facade from an HDS ``ARRAY`` structure."""
        data = structure.children["DATA"]
        if not isinstance(data, HdsPrimitive):
            raise TypeError("ARRAY.DATA must be an HDS primitive.")
        origin: np.ndarray | None = None
        if (origin_node := structure.children.get("ORIGIN")) is not None:
            if not isinstance(origin_node, HdsPrimitive):
                raise TypeError("ARRAY.ORIGIN must be an HDS primitive.")
            origin = origin_node.read_array()
        bad_pixel: bool | None = None
        if (bad_pixel_node := structure.children.get("BAD_PIXEL")) is not None:
            if not isinstance(bad_pixel_node, HdsPrimitive):
                raise TypeError("ARRAY.BAD_PIXEL must be an HDS primitive.")
            bad_pixel = bool(np.asarray(bad_pixel_node.read_array()).reshape(-1)[0])
        array_data = data.data if data.data is not None else data.read_array()
        return cls(data=array_data, origin=origin, bad_pixel=bad_pixel)


@dataclass
class NdfQuality:
    """An NDF QUALITY component."""

    quality: NdfArray
    badbits: int = 1

    def to_hds_structure(self) -> HdsStructure:
        """Convert this quality component to an HDS ``QUALITY`` structure."""
        structure = HdsStructure("QUALITY")
        structure.children["BADBITS"] = HdsPrimitive.array(np.array(self.badbits, dtype=np.uint8))
        structure.children["QUALITY"] = self.quality.to_hds_structure()
        return structure


@dataclass
class NdfWcs:
    """An NDF WCS component represented by encoded AST channel records."""

    lines: list[str]
    width: int = _hds.NDF_AST_DATA_WIDTH

    def to_hds_structure(self) -> HdsStructure:
        """Convert this WCS component to an HDS ``WCS`` structure."""
        structure = HdsStructure("WCS")
        structure.children["DATA"] = HdsPrimitive.char_array(self.lines, width=self.width)
        return structure


class Ndf(HdsStructure):
    """An NDF structure with convenience accessors for standard components."""

    def __init__(self, children: Mapping[str, HdsStructure | HdsPrimitive] | None = None) -> None:
        super().__init__("NDF", children)

    def set_array_component(
        self,
        name: str,
        data: np.ndarray | h5py.Dataset,
        *,
        origin: Sequence[int] | np.ndarray | None = None,
        compression_options: Mapping[str, Any] | None = None,
    ) -> None:
        """Set an NDF array-like component such as ``DATA_ARRAY``."""
        self.children[name] = NdfArray(
            data,
            origin=origin,
            compression_options=dict(compression_options) if compression_options else {},
        ).to_hds_structure()

    def set_quality(self, quality: NdfQuality) -> None:
        """Set the NDF ``QUALITY`` component."""
        self.children["QUALITY"] = quality.to_hds_structure()

    def set_wcs(self, wcs: NdfWcs) -> None:
        """Set the NDF ``WCS`` component."""
        self.children["WCS"] = wcs.to_hds_structure()

    def set_units(self, units: str | None) -> None:
        """Set or remove the NDF ``UNITS`` component."""
        if units is None:
            self.children.pop("UNITS", None)
        else:
            self.children["UNITS"] = HdsPrimitive.char_scalar(units)

    def get_units(self) -> str | None:
        """Return the NDF ``UNITS`` component, if present."""
        node = self.children.get("UNITS")
        if node is None:
            return None
        if not isinstance(node, HdsPrimitive):
            raise TypeError("NDF.UNITS must be an HDS primitive.")
        lines = node.read_char_array()
        return lines[0] if lines else ""

    def ensure_lsst_extension(self, *, base_path: str = "MORE/LSST") -> HdsStructure:
        """Return or create the LSST extension structure for this NDF."""
        return self.ensure_structure(base_path, "EXT")


@dataclass
class NdfDocument:
    """A complete HDS root object containing one NDF or an NDF container."""

    root: Ndf | NdfContainer = field(default_factory=Ndf)
    root_name: str | None = None

    @classmethod
    def from_hdf5(cls, file: h5py.File) -> Self:
        """Read an NDF document model from an open HDF5 file."""
        root = HdsStructure.from_hdf5(file["/"])
        if isinstance(root, Ndf | NdfContainer):
            typed_root = root
        elif root.hds_type == "NDF":
            root.__class__ = Ndf
            typed_root = root
        else:
            root.__class__ = NdfContainer
            typed_root = root
        return cls(root=typed_root, root_name=_decode_ascii_attr(file["/"].attrs.get(_hds.ATTR_ROOT_NAME)))

    def write_to_hdf5(self, file: h5py.File) -> None:
        """Write this document to an open HDF5 file."""
        self.root.write_to_hdf5(file["/"])
        if self.root_name is not None:
            _hds.set_root_name(file, self.root_name, self.root.hds_type)

    def ensure_ndf(self, path: str = "/") -> Ndf:
        """Return or create an NDF at the requested absolute path."""
        if path in ("", "/"):
            if not isinstance(self.root, Ndf):
                raise TypeError("The document root is an NDF container, not an NDF.")
            return self.root
        if isinstance(self.root, NdfContainer):
            return self.root.ensure_ndf(path)
        structure = self.root.ensure_structure(path, "NDF")
        if not isinstance(structure, Ndf):
            structure.__class__ = Ndf
        return structure

    def get(self, path: str) -> HdsStructure | HdsPrimitive:
        """Return a component by absolute path."""
        return self.root.get(path)


def _new_structure(hds_type: str) -> HdsStructure:
    """Create a typed HDS structure model for an HDS type."""
    if hds_type == "NDF":
        return Ndf()
    if hds_type == "EXT":
        return HdsExtension()
    return HdsStructure(hds_type)


def _split_path(path: str) -> list[str]:
    """Split an HDS/HDF5 path into component names."""
    return [part for part in path.strip("/").split("/") if part]


def _clear_hdf5_group(group: h5py.Group) -> None:
    """Remove children and HDS root attributes before rewriting a group."""
    for name in list(group.keys()):
        del group[name]
    for name in (_hds.ATTR_CLASS, _hds.ATTR_ROOT_NAME, _hds.ATTR_STRUCTURE_DIMS, "HDSTYPE"):
        if name in group.attrs:
            del group.attrs[name]
