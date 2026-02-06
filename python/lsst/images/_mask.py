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

from __future__ import annotations

__all__ = ("Mask", "MaskPlane", "MaskPlaneBit", "MaskSchema")

import dataclasses
import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence, Set
from types import EllipsisType
from typing import Any

import astropy.io.fits
import numpy as np
import numpy.typing as npt
import pydantic

from . import fits
from ._geom import Box, Interval
from ._transforms import Projection, ProjectionSerializationModel
from .serialization import ArrayReferenceModel, InputArchive, OutputArchive, no_header_updates
from .utils import is_none


@dataclasses.dataclass(frozen=True)
class MaskPlane:
    """Name and description of a single plane in a mask array."""

    name: str
    """Unique name for the mask plane (`str`)."""

    description: str
    """Human-readable documentation for the mask plane (`str`)."""

    @classmethod
    def read_legacy(cls, header: astropy.io.fits.Header) -> dict[str, int]:
        """Read mask plane descriptions written by
        `lsst.afw.image.Mask.writeFits`.

        Parameters
        ----------
        header
            FITS header.

        Returns
        -------
        `dict` [`str`, `int`]
            A dictionary mapping mask plane name to integer bit index.
        """
        result: dict[str, int] = {}
        for card in list(header.cards):
            if card.keyword.startswith("MP_"):
                result[card.keyword.removeprefix("MP_")] = card.value
                del header[card.keyword]
        return result


@dataclasses.dataclass(frozen=True)
class MaskPlaneBit:
    """The nested array index and mask value associated with a single mask
    plane.
    """

    index: int
    """Index into the last dimension of the mask array where this plane's bit
    is stored.
    """

    mask: np.unsignedinteger
    """Bitmask that selects just this plane's bit from a mask array value
    (`numpy.unsignedinteger`).
    """

    @classmethod
    def compute(cls, overall_index: int, stride: int, mask_type: type[np.unsignedinteger]) -> MaskPlaneBit:
        """Construct a `MaskPlaneBit` from the overall index of a plane in a
        `MaskSchema` and the stride (number of bits per mask array element).
        """
        index, bit = divmod(overall_index, stride)
        return cls(index, mask_type(1 << bit))


class MaskSchema:
    """A schema for a bit-packed mask array.

    Parameters
    ----------
    planes
        Iterable of `MaskPlane` instances that define the schema.  `None`
        values may be included to reserve bits for future use.
    dtype
        The numpy data type of the mask arrays that use this schema.

    Notes
    -----
    A `MaskSchema` is a collection of mask planes, which each correspond to a
    single bit in a mask array.  Mask schemas are immutable and associated with
    a particular array data type, allowing them to safely precompute the index
    and bitmask for each plane.

    `MaskSchema` indexing is by integer (the overall index of a plane in the
    schema).  The `descriptions` attribute may be indexed by plane name to get
    the description for that plane, and the `bitmask` method can be used to
    obtain an array that can be used to select one or more planes by name in
    a mask array that uses this schema.
    """

    def __init__(self, planes: Iterable[MaskPlane | None], dtype: npt.DTypeLike = np.uint8):
        self._planes = tuple(planes)
        self._dtype = np.dtype(dtype)
        if not issubclass(self._dtype.type, np.unsignedinteger):
            raise TypeError("dtype for masks must be an unsigned integer.")
        self._descriptions = {plane.name: plane.description for plane in self._planes if plane is not None}
        stride = self._dtype.itemsize * 8
        self._mask_size = math.ceil(len(self._planes) / stride)
        self._bits: dict[str, MaskPlaneBit] = {
            plane.name: MaskPlaneBit.compute(n, stride, self._dtype.type)
            for n, plane in enumerate(self._planes)
            if plane is not None
        }

    def __iter__(self) -> Iterator[MaskPlane | None]:
        return iter(self._planes)

    def __len__(self) -> int:
        return len(self._planes)

    def __getitem__(self, i: int) -> MaskPlane | None:
        return self._planes[i]

    def __repr__(self) -> str:
        return f"MaskSchema({list(self._planes)}, dtype={self._dtype!r})"

    def __str__(self) -> str:
        return "\n".join(
            [
                f"{name} [{bit.index}@{hex(bit.mask)}]: {self._descriptions[name]}"
                for name, bit in self._bits.items()
            ]
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MaskSchema):
            return self._planes == other._planes
        return False

    @property
    def dtype(self) -> np.dtype:
        """The numpy data type of the mask arrays that use this schema."""
        return self._dtype

    @property
    def mask_size(self) -> int:
        """The number of elements in the last dimension of any mask array that
        uses this schema.
        """
        return self._mask_size

    @property
    def names(self) -> Set[str]:
        """The names of the mask planes, in bit order."""
        return self._bits.keys()

    @property
    def descriptions(self) -> Mapping[str, str]:
        """A mapping from plane name to description."""
        return self._descriptions

    def bit(self, plane: str) -> MaskPlaneBit:
        """Return the last array index and mask for the given mask plane."""
        return self._bits[plane]

    def bitmask(self, *planes: str) -> np.ndarray:
        """Return a 1-d mask array that represents the union (i.e. bitwise OR)
        of the planes with the given names.

        Parameters
        ----------
        *planes
            Mask plane names.

        Returns
        -------
        numpy.ndarray
            A 1-d array with shape ``(mask_size,)``.
        """
        result = np.zeros(self.mask_size, dtype=self._dtype)
        for plane in planes:
            bit = self._bits[plane]
            result[bit.index] |= bit.mask
        return result


class Mask:
    """A 2-d bitmask image backed by a 3-d byte array.

    Parameters
    ----------
    array_or_fill
        Array or fill value for the mask.  If a fill value, ``bbox`` or
        ``shape`` must be provided.
    schema
        Schema that defines the planes and their bit assignments.
    bbox
        Bounding box for the mask.  This sets the shape of the first two
        dimensions of the array.
    start
        Logical coordinates of the first pixel in the array, ordered ``y``,
        ``x`` (unless an `XY` instance is passed).  Ignored if
        ``bbox`` is provided.  Defaults to zeros.
    shape
        Leading dimensions of the array, ordered ``y``, ``x`` (unless an `XY`
        instance is passed).   Only needed if ``array_or_fill`` is not an
        array and ``bbox`` is not provided.  Like the bbox, this does not
        include the last dimension of the array.
    projection
        Projection that maps the pixel grid to the sky.

    Notes
    -----
    Indexing the `array` attribute of a `Mask` does not take into account its
    ``start`` offset, but accessing a subimage mask by indexing a `Mask` with
    a `Box` does, and the `bbox` of the subimage is set to match its location
    within the original mask.

    A mask's ``bbox`` corresponds to the leading dimensions of its backing
    `numpy.ndarray`, while the last dimension's size is always equal to the
    `~MaskSchema.mask_size` of its schema, since a schema can in general
    require multiple array elements to represent all of its planes.
    """

    def __init__(
        self,
        array_or_fill: np.ndarray | int = 0,
        /,
        *,
        schema: MaskSchema,
        bbox: Box | None = None,
        start: Sequence[int] | None = None,
        shape: Sequence[int] | None = None,
        projection: Projection | None = None,
    ):
        if shape is not None:
            shape = tuple(shape)
        if start is not None:
            start = tuple(start)
        if isinstance(array_or_fill, np.ndarray):
            array = np.array(array_or_fill, dtype=schema.dtype)
            if array.ndim != 3:
                raise ValueError("Mask array must be 3-d.")
            if bbox is None:
                bbox = Box.from_shape(array.shape[:-1], start=start)
            elif bbox.shape + (schema.mask_size,) != array.shape:
                raise ValueError(
                    f"Explicit bbox shape {bbox.shape} and schema of size {schema.mask_size} do not "
                    f"match array with shape {array.shape}."
                )
            if shape is not None and shape + (schema.mask_size,) != array.shape:
                raise ValueError(
                    f"Explicit shape {shape} and schema of size {schema.mask_size} do "
                    f"not match array with shape {array.shape}."
                )

        else:
            if bbox is None:
                if shape is None:
                    raise TypeError("No bbox, size, or array provided.")
                bbox = Box.from_shape(shape, start=start)
            array = np.full(bbox.shape + (schema.mask_size,), array_or_fill, dtype=schema.dtype)
        self._array = array
        self._bbox: Box = bbox
        self._schema: MaskSchema = schema
        self._projection = projection

    @property
    def array(self) -> np.ndarray:
        """The low-level array (`numpy.ndarray`).

        Assigning to this attribute modifies the existing array in place; the
        bounding box and underlying data pointer are never changed.
        """
        return self._array

    @array.setter
    def array(self, value: np.ndarray | int) -> None:
        self._array[:, :] = value

    @property
    def schema(self) -> MaskSchema:
        """Schema that defines the planes and their bit assignments
        (`MaskSchema`).
        """
        return self._schema

    @property
    def bbox(self) -> Box:
        """2-d bounding box of the mask (`Box`).

        This sets the shape of the first two dimensions of the array.
        """
        return self._bbox

    @property
    def projection(self) -> Projection[Any] | None:
        """The projection that maps this mask's pixel grid to the sky
        (`Projection` or `None`).

        Notes
        -----
        The pixel coordinates used by this projection account for the bounding
        box ``start``; they are not just array indices.
        """
        return self._projection

    def __getitem__(self, bbox: Box) -> Mask:
        return Mask(
            self.array[bbox.y.slice_within(self._bbox.y), bbox.x.slice_within(self._bbox.x), :],
            bbox=bbox,
            schema=self.schema,
        )

    def copy(
        self,
        *,
        schema: MaskSchema | EllipsisType = ...,
        projection: Projection | None | EllipsisType = ...,
        start: Sequence[int] | EllipsisType = ...,
    ) -> Mask:
        """Deep-copy the mask, with optional updates.

        Notes
        -----
        This can only be used to make changes to schema descriptions; plane
        names must remain the same (in the same order).
        """
        if schema is ...:
            schema = self._schema
        else:
            if list(schema.names) != list(self.schema.names):
                raise ValueError("Cannot create a mask view with a schema with different names.")
        if projection is ...:
            projection = self._projection
        if start is ...:
            start = self._bbox.start
        return Mask(self._array.copy(), bbox=self._bbox, schema=self._schema, projection=self._projection)

    def view(
        self,
        *,
        schema: MaskSchema | EllipsisType = ...,
        projection: Projection | None | EllipsisType = ...,
        start: Sequence[int] | EllipsisType = ...,
    ) -> Mask:
        """Make a view of the mask, with optional updates.

        Notes
        -----
        This can only be used to make changes to schema descriptions; plane
        names must remain the same (in the same order).
        """
        if schema is ...:
            schema = self._schema
        else:
            if list(schema.names) != list(self.schema.names):
                raise ValueError("Cannot create a mask view with a schema with different names.")
        if projection is ...:
            projection = self._projection
        if start is ...:
            start = self._bbox.start
        return Mask(self._array, start=start, schema=schema, projection=projection)

    def get(self, plane: str) -> np.ndarray:
        """Return a 2-d boolean array for the given mask plane.

        Parameters
        ----------
        plane
            Name of the mask plane.

        Returns
        -------
        numpy.ndarray
            A 2-d boolean array with the same shape as `bbox` that is `True`
            where the bit for ``plane`` is set and `False` elsewhere.
        """
        bit = self.schema.bit(plane)
        return (self._array[..., bit.index] & bit.mask).astype(bool)

    def set(self, plane: str, boolean_mask: np.ndarray | EllipsisType = ...) -> None:
        """Set a mask plane.

        Parameters
        ----------
        plane
            Name of the mask plane to set
        boolean_mask
            A 2-d boolean array with the same shape as `bbox` that is `True`
            where the bit for ``plane`` should be set and `False` where it
            should be left unchanged (*not* set to zero).  May be ``...`` to
            set the bit everywhere.
        """
        bit = self.schema.bit(plane)
        if boolean_mask is not ...:
            boolean_mask = boolean_mask.astype(bool)
        self._array[boolean_mask, bit.index] |= bit.mask

    def clear(self, plane: str, boolean_mask: np.ndarray | EllipsisType = ...) -> None:
        """Clear a mask plane.

        Parameters
        ----------
        plane
            Name of the mask plane to set
        boolean_mask
            A 2-d boolean array with the same shape as `bbox` that is `True`
            where the bit for ``plane`` should be cleared and `False` where it
            should be left unchanged.  May be ``...`` to clear the bit
            everywhere.
        """
        bit = self.schema.bit(plane)
        if boolean_mask is not ...:
            boolean_mask = boolean_mask.astype(bool)
        self._array[boolean_mask, bit.index] &= ~bit.mask

    def __str__(self) -> str:
        return f"Mask({self.bbox!s}, {list(self.schema.names)})"

    def __repr__(self) -> str:
        return f"Mask(..., bbox={self.bbox!r}, schema={self.schema!r})"

    def serialize[P: pydantic.BaseModel](
        self,
        archive: OutputArchive[P],
        *,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
        save_projection: bool = True,
        add_offset_wcs: str | None = "A",
    ) -> MaskSerializationModel[P]:
        """Serialize the mask to an output archive.

        Parameters
        ----------
        archive
            `~serialization.OutputArchive` instance to write to.
        update_header
            A callback that will be given the FITS header for the HDU
            containing this image in order to add keys to it.  This callback
            may be provided but will not be called if the output format is not
            FITS.
        save_projection
            If `True`, save the `Projection` attached to the mask, if there
            is one.
        add_offset_wcs
            A FITS WCS single-character suffix to use when adding a linear
            WCS that maps the FITS array to the logical pixel coordinates
            defined by ``bbox.start``.  Set to `None` to not write this WCS.

        Returns
        -------
        MaskSerializationModel
            A Pydantic model representation of the mask, holding references
            to data stored in the archive.
        """

        def _update_header(header: astropy.io.fits.Header) -> None:
            update_header(header)
            # TODO: save mask plane information.
            if self.projection is not None:
                fits_wcs = self.projection.as_fits_wcs(self.bbox)
                if fits_wcs:
                    header.update(fits_wcs.to_header(relax=True))
            if add_offset_wcs is not None:
                fits.add_offset_wcs(header, x=self.bbox.x.start, y=self.bbox.y.start, key=add_offset_wcs)

        ref = archive.add_array(self.array, update_header=_update_header)
        serialized_projection: ProjectionSerializationModel[P] | None = None
        if save_projection and self.projection is not None:
            serialized_projection = archive.serialize_direct("projection", self.projection.serialize)
        return MaskSerializationModel.model_construct(
            data=ref, start=list(self.bbox.start), planes=list(self.schema), projection=serialized_projection
        )

    @classmethod
    def deserialize(
        cls,
        model: MaskSerializationModel,
        archive: InputArchive[Any],
        *,
        bbox: Box | None = None,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> Mask:
        """Deserialize a mask from an input archive.

        Parameters
        ----------
        model
            A Pydantic model representation of the mask, holding references
            to data stored in the archive.
        archive
            `~serialization.InputArchive` instance to read from.
        bbox
            Bounding box of a subimage to read instead.
        strip_header
            A callable that strips out any FITS header cards added by the
            ``update_header`` argument in the corresponding call to
            `serialize`.
        """

        def _strip_header(header: astropy.io.fits.Header) -> None:
            strip_header(header)
            fits.strip_wcs_cards(header)
            # TODO: strip mask plane information.

        slices = bbox.slice_within(model.bbox) if bbox is not None else ...
        array = archive.get_array(model.data, strip_header=_strip_header, slices=slices)
        schema = MaskSchema(model.planes, dtype=array.dtype)
        projection = (
            Projection.deserialize(model.projection, archive) if model.projection is not None else None
        )
        return cls(
            array, schema=schema, start=model.start if bbox is None else bbox.start, projection=projection
        )

    @classmethod
    def read_legacy(cls, hdu: astropy.io.fits.ImageHDU | astropy.io.fits.CompImageHDU) -> Mask:
        """Read a FITS file written by `lsst.afw.image.Mask.writeFits`.

        Parameters
        ----------
        hdu
            An astropy image HDU.
        """
        dx: int = hdu.header.pop("LTV1")
        dy: int = hdu.header.pop("LTV2")
        start = (-dy, -dx)
        plane_dict = MaskPlane.read_legacy(hdu.header)
        schema = MaskSchema([MaskPlane(name, "") for name in plane_dict])
        array2d: np.ndarray = hdu.data
        mask = cls(0, schema=schema, start=start, shape=array2d.shape)
        for name, bit2d in plane_dict.items():
            bitmask2d = 1 << bit2d
            mask.set(name, array2d & bitmask2d)
        return mask


class MaskSerializationModel[P: pydantic.BaseModel](pydantic.BaseModel):
    """Pydantic model used to represent the serialized form of a `.Mask`."""

    data: ArrayReferenceModel = pydantic.Field(description="Reference to pixel data.")
    start: list[int] = pydantic.Field(
        description="Coordinate of the first pixels in the array, ordered (y, x)."
    )
    planes: list[MaskPlane | None] = pydantic.Field(description="Definitions of the bitplanes in the mask.")
    projection: ProjectionSerializationModel[P] | None = pydantic.Field(
        default=None,
        exclude_if=is_none,
        description="Projection that maps the logical pixel grid onto the sky.",
    )

    @property
    def bbox(self) -> Box:
        """The 2-d bounding box of the mask."""
        return Box(
            *[Interval.factory[begin : begin + size] for begin, size in zip(self.start, self.data.shape[:-1])]
        )
