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

__all__ = (
    "Mask",
    "MaskPlane",
    "MaskPlaneBit",
    "MaskSchema",
    "MaskSerializationModel",
    "get_legacy_deep_coadd_mask_planes",
    "get_legacy_difference_image_mask_planes",
    "get_legacy_non_cell_coadd_mask_planes",
    "get_legacy_visit_image_mask_planes",
)

import dataclasses
import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence, Set
from types import EllipsisType
from typing import TYPE_CHECKING, Any, ClassVar, cast

import astropy.io.fits
import astropy.wcs
import numpy as np
import numpy.typing as npt
import pydantic

from lsst.resources import ResourcePath, ResourcePathExpression

from . import fits
from ._generalized_image import GeneralizedImage
from ._geom import YX, Box, NoOverlapError
from ._transforms import Frame, SkyProjection, SkyProjectionSerializationModel
from .serialization import (
    ArchiveReadError,
    ArchiveTree,
    ArrayReferenceModel,
    InlineArrayModel,
    InputArchive,
    IntegerType,
    InvalidParameterError,
    MetadataValue,
    NumberType,
    OutputArchive,
    is_integer,
    no_header_updates,
)
from .utils import is_none

if TYPE_CHECKING:
    try:
        from lsst.afw.image import Mask as LegacyMask
    except ImportError:
        type LegacyMask = Any  # type: ignore[no-redef]


@dataclasses.dataclass(frozen=True)
class MaskPlane:
    """Name and description of a single plane in a mask array."""

    name: str
    """Unique name for the mask plane (`str`)."""

    description: str
    """Human-readable documentation for the mask plane (`str`)."""

    @classmethod
    def read_legacy(cls, header: astropy.io.fits.Header, *, strip: bool = True) -> dict[str, int]:
        """Read mask plane descriptions written by
        `lsst.afw.image.Mask.writeFits`.

        Parameters
        ----------
        header
            FITS header.
        strip
            If `True` (default), delete the ``MP_`` cards from the header after
            reading them, as appropriate when the mask is being reinterpreted
            for new code only.  If `False`, leave them in place so they can be
            propagated for backwards compatibility (re-indexed to the new
            schema by the caller).

        Returns
        -------
        `dict` [`str`, `int`]
            A dictionary mapping mask plane name to integer bit index.
        """
        result: dict[str, int] = {}
        for card in list(header.cards):
            if card.keyword.startswith("MP_"):
                result[card.keyword.removeprefix("MP_")] = card.value
                if strip:
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

    mask: np.integer
    """Bitmask that selects just this plane's bit from a mask array value
    (`numpy.integer`).
    """

    @classmethod
    def compute(cls, overall_index: int, stride: int, mask_type: type[np.integer]) -> MaskPlaneBit:
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

    If no mask planes are provided, a `None` placeholder is automatically
    added.
    """

    def __init__(self, planes: Iterable[MaskPlane | None], dtype: npt.DTypeLike = np.uint8) -> None:
        self._planes: tuple[MaskPlane | None, ...] = tuple(planes) or (None,)
        self._dtype = cast(np.dtype[np.integer], np.dtype(dtype))
        stride = self.bits_per_element(self._dtype)
        self._descriptions = {plane.name: plane.description for plane in self._planes if plane is not None}
        self._mask_size = math.ceil(len(self._planes) / stride)
        self._bits: dict[str, MaskPlaneBit] = {
            plane.name: MaskPlaneBit.compute(n, stride, self._dtype.type)
            for n, plane in enumerate(self._planes)
            if plane is not None
        }

    @staticmethod
    def bits_per_element(dtype: npt.DTypeLike) -> int:
        """Return the number of mask bits per array element for the given
        data type.
        """
        dtype = np.dtype(dtype)
        match dtype.kind:
            case "u":
                return dtype.itemsize * 8
            case "i":
                return dtype.itemsize * 8 - 1
            case _:
                raise TypeError(f"dtype for masks must be an integer; got {dtype} with kind={dtype.kind}.")

    def __iter__(self) -> Iterator[MaskPlane | None]:
        return iter(self._planes)

    def __len__(self) -> int:
        return len(self._planes)

    def __contains__(self, plane: str | MaskPlane) -> bool:
        return getattr(plane, "name", plane) in self.names

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
            return self._planes == other._planes and self._dtype == other._dtype
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

    def split(self, dtype: npt.DTypeLike) -> list[MaskSchema]:
        """Split the schema into an equivalent series of schemas that each
        have a `mask_size` of ``1``, dropping all `None` placeholders.

        Parameters
        ----------
        dtype
            Data type of the new mask pixels.

        Returns
        -------
        `list` [`MaskSchema`]
            A list of mask schemas that together include all planes in
            ``self`` and have `mask_size` equal to ``1``.  If there are no
            mask planes (only `None` placeholders) in ``self``, a single mask
            schema with a `None` placeholder is returned; otherwise `None`
            placeholders are returned.
        """
        dtype = np.dtype(dtype)
        planes: list[MaskPlane] = []
        schemas: list[MaskSchema] = []
        n_planes_per_schema = self.bits_per_element(dtype)
        for plane in self._planes:
            if plane is not None:
                planes.append(plane)
                if len(planes) == n_planes_per_schema:
                    schemas.append(MaskSchema(planes, dtype=dtype))
                    planes.clear()
        if planes:
            schemas.append(MaskSchema(planes, dtype=dtype))
        if not schemas:
            schemas.append(MaskSchema([None], dtype=dtype))
        return schemas

    def update_header(self, header: astropy.io.fits.Header) -> None:
        """Add a description of this mask schema to a FITS header."""
        for n, plane in enumerate(self):
            if plane is not None:
                bit = self.bit(plane.name)
                if bit.index != 0:
                    raise TypeError("Only mask schemas with mask_size==1 can be described in FITS.")
                header.set(f"MSKN{n:04d}", plane.name, f"Name for mask plane {n}.")
                header.set(f"MSKM{n:04d}", bit.mask, f"Bitmask for plane n={n}; always 1<<n.")
                # We don't add a comment to the description card, because it's
                # likely to overrun a single card and get the CONTINUE
                # treatment. That will cause Astropy to warn about the comment
                # being truncated and that's worse than just leaving it
                # unexplained; it's pretty obvious from context what it is.
                header.set(f"MSKD{n:04d}", plane.description)

    def strip_header(self, header: astropy.io.fits.Header) -> None:
        """Remove all header cards added by `update_header`."""
        for n, plane in enumerate(self):
            if plane is not None:
                header.remove(f"MSKN{n:04d}", ignore_missing=True)
                header.remove(f"MSKM{n:04d}", ignore_missing=True)
                header.remove(f"MSKD{n:04d}", ignore_missing=True)

    @classmethod
    def from_fits_header(cls, header: astropy.io.fits.Header, dtype: npt.DTypeLike = np.uint8) -> MaskSchema:
        """Reconstruct a schema from the ``MSKN``/``MSKD`` cards written by
        `update_header`.

        Parameters
        ----------
        header
            FITS header containing ``MSKN{n:04d}`` plane-name cards and
            ``MSKD{n:04d}`` description cards.
        dtype
            Data type of the mask arrays that will use this schema.  The cards
            describe a ``mask_size==1`` serialized form and do not record the
            in-memory dtype, so the caller must supply it; it defaults to the
            same ``uint8`` used by the `Mask` constructor.

        Returns
        -------
        `MaskSchema`
            Schema whose planes are ordered by their ``MSKN`` index, with
            `None` placeholders inserted for any gaps in that numbering.

        Raises
        ------
        ValueError
            Raised if the header contains no ``MSKN`` cards.
        """
        planes_by_index: dict[int, MaskPlane] = {}
        for card in header.cards:
            if card.keyword.startswith("MSKN"):
                n = int(card.keyword.removeprefix("MSKN"))
                planes_by_index[n] = MaskPlane(card.value, header.get(f"MSKD{n:04d}", ""))
        if not planes_by_index:
            raise ValueError("Header has no MSKN cards describing a mask schema.")
        planes = [planes_by_index.get(n) for n in range(max(planes_by_index) + 1)]
        return cls(planes, dtype=dtype)


class Mask(GeneralizedImage):
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
    yx0
        Logical coordinates of the first pixel in the array, ordered ``y``,
        ``x`` (unless an `XY` instance is passed).  Ignored if
        ``bbox`` is provided.  Defaults to zeros.
    shape
        Leading dimensions of the array, ordered ``y``, ``x`` (unless an `XY`
        instance is passed).   Only needed if ``array_or_fill`` is not an
        array and ``bbox`` is not provided.  Like the bbox, this does not
        include the last dimension of the array.
    sky_projection
        Projection that maps the pixel grid to the sky.
    metadata
        Arbitrary flexible metadata to associate with the mask.

    Notes
    -----
    Indexing the `array` attribute of a `Mask` does not take into account its
    ``yx0`` offset, but accessing a subimage mask by indexing a `Mask` with
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
        yx0: Sequence[int] | None = None,
        shape: Sequence[int] | None = None,
        sky_projection: SkyProjection | None = None,
        metadata: dict[str, MetadataValue] | None = None,
    ) -> None:
        super().__init__(metadata)
        if shape is not None:
            shape = tuple(shape)
        if isinstance(array_or_fill, np.ndarray):
            array = np.array(array_or_fill, dtype=schema.dtype, copy=None)
            if array.ndim != 3:
                raise ValueError("Mask array must be 3-d.")
            if bbox is None:
                bbox = Box.from_shape(array.shape[:-1], start=yx0)
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
                bbox = Box.from_shape(shape, start=yx0)
            array = np.full(bbox.shape + (schema.mask_size,), array_or_fill, dtype=schema.dtype)
        self._array = array
        self._bbox: Box = bbox
        self._schema: MaskSchema = schema
        self._sky_projection = sky_projection

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
    def sky_projection(self) -> SkyProjection[Any] | None:
        """The projection that maps this mask's pixel grid to the sky
        (`SkyProjection` | `None`).

        Notes
        -----
        The pixel coordinates used by this projection account for the bounding
        box ``start`` (i.e. ``yx0``); they are not just array indices.
        """
        return self._sky_projection

    def __getitem__(self, bbox: Box | EllipsisType) -> Mask:
        if bbox is ...:
            return self
        super().__getitem__(bbox)
        return self._transfer_metadata(
            Mask(
                self.array[bbox.y.slice_within(self._bbox.y), bbox.x.slice_within(self._bbox.x), :],
                bbox=bbox,
                schema=self.schema,
                sky_projection=self._sky_projection,
            ),
            bbox=bbox,
        )

    def __setitem__(self, bbox: Box | EllipsisType, value: Mask) -> None:
        subview = self[bbox]
        subview.clear()
        subview.update(value)

    def __str__(self) -> str:
        return f"Mask({self.bbox!s}, {list(self.schema.names)})"

    def __repr__(self) -> str:
        return f"Mask(..., bbox={self.bbox!r}, schema={self.schema!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mask):
            return NotImplemented
        return (
            self._bbox == other._bbox
            and self._schema == other._schema
            and np.array_equal(self._array, other._array, equal_nan=True)
        )

    def copy(self) -> Mask:
        """Deep-copy the mask and metadata."""
        return self._transfer_metadata(
            Mask(
                self._array.copy(), bbox=self._bbox, schema=self._schema, sky_projection=self._sky_projection
            ),
            copy=True,
        )

    def view(
        self,
        *,
        schema: MaskSchema | EllipsisType = ...,
        sky_projection: SkyProjection | None | EllipsisType = ...,
        yx0: Sequence[int] | EllipsisType = ...,
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
        if sky_projection is ...:
            sky_projection = self._sky_projection
        if yx0 is ...:
            yx0 = self._bbox.start
        return self._transfer_metadata(
            Mask(self._array, yx0=yx0, schema=schema, sky_projection=sky_projection)
        )

    def update(self, other: Mask) -> None:
        """Update ``self`` to include all common mask values set in ``other``.

        Notes
        -----
        This only operates on the intersection of the two mask bounding boxes
        and the mask planes that are present in both.  Mask bits are only set,
        not cleared (i.e. this uses ``|=`` updates, not ``=`` assignments).
        """
        lhs = self
        rhs = other
        if other.bbox != self.bbox:
            try:
                bbox = self.bbox.intersection(other.bbox)
            except NoOverlapError:
                return
            lhs = self[bbox]
            rhs = other[bbox]
        for name in self.schema.names & other.schema.names:
            lhs.set(name, rhs.get(name))

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

    def clear(self, plane: str | None = None, boolean_mask: np.ndarray | EllipsisType = ...) -> None:
        """Clear one or more mask planes.

        Parameters
        ----------
        plane
            Name of the mask plane to set.  If `None` all mask planes are
            cleared.
        boolean_mask
            A 2-d boolean array with the same shape as `bbox` that is `True`
            where the bit for ``plane`` should be cleared and `False` where it
            should be left unchanged.  May be ``...`` to clear the bit
            everywhere.
        """
        if boolean_mask is not ...:
            boolean_mask = boolean_mask.astype(bool)
        if plane is None:
            self._array[boolean_mask, :] = 0
        else:
            bit = self.schema.bit(plane)
            self._array[boolean_mask, bit.index] &= ~bit.mask

    def add_plane(self, name: str, description: str) -> Mask:
        """Return a new mask with one additional mask plane.

        This is a convenience wrapper around `add_planes` for the common case
        of adding a single plane.

        Parameters
        ----------
        name
            Unique name for the new mask plane.
        description
            Human-readable documentation for the new mask plane.

        Returns
        -------
        `Mask`
            A new mask whose schema includes the new plane; see `add_planes`
            for the reallocation and view semantics.

        Raises
        ------
        ValueError
            Raised if a plane named ``name`` already exists.
        """
        return self.add_planes([MaskPlane(name, description)])

    def add_planes(self, planes: Iterable[MaskPlane | None], *, drop: Iterable[str] = ()) -> Mask:
        """Return a new mask with planes added and/or dropped.

        Parameters
        ----------
        planes
            New mask planes to append, in order, after the planes retained
            from this mask.  `None` entries reserve unused bits (placeholders),
            exactly as in `MaskSchema`.
        drop
            Names of existing planes to remove from the schema.

        Returns
        -------
        `Mask`
            A new mask with the updated schema.  Retained planes keep their
            pixel values (copied by name); newly added planes start cleared.

        Raises
        ------
        ValueError
            Raised if a name in ``drop`` is not an existing plane, or if a
            plane in ``planes`` collides with a retained plane name.

        Notes
        -----
        Adding or dropping planes always reallocates the backing array and
        returns a new `Mask`; this mask is left unchanged and any views or
        subimages of it continue to refer to the original array with the
        original schema.  This is deliberate: there is no way to update the
        schema of an existing view, and a stale view must never set bits that
        its now-outdated schema regards as unused.  Dropping a plane compacts
        the schema, so planes after it are reassigned to lower bits and the
        pixel values are repacked by plane name to match.
        """
        drop_set = set(drop)
        if unknown := drop_set - set(self._schema.names):
            raise ValueError(f"Cannot drop mask planes that do not exist: {sorted(unknown)}.")
        retained = [plane for plane in self._schema if plane is None or plane.name not in drop_set]
        names = {plane.name for plane in retained if plane is not None}
        new_planes = list(planes)
        for plane in new_planes:
            if plane is None:
                continue
            if plane.name in names:
                raise ValueError(f"Mask plane {plane.name!r} already exists.")
            names.add(plane.name)
        new_schema = MaskSchema([*retained, *new_planes], dtype=self._schema.dtype)
        result = Mask(0, schema=new_schema, bbox=self._bbox, sky_projection=self._sky_projection)
        # The retained planes are exactly the names common to both schemas, and
        # ``result`` starts cleared and shares this mask's bbox, so ``update``
        # transfers their pixel values (and nothing else) by name.
        result.update(self)
        return self._transfer_metadata(result, copy=True)

    def serialize[P: pydantic.BaseModel](
        self,
        archive: OutputArchive[P],
        *,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
        save_projection: bool = True,
        add_offset_wcs: str | None = "A",
        tile_shape: tuple[int, ...] | None = None,
        options_name: str | None = None,
    ) -> MaskSerializationModel[P]:
        """Serialize the mask to an output archive.

        Parameters
        ----------
        archive
            Archive to write to.
        update_header
            A callback that will be given the FITS header for the HDU
            containing this mask in order to add keys to it.  This callback
            may be provided but will not be called if the output format is not
            FITS.  As multiple HDUs may be added, this function may be called
            multiple times.
        save_projection
            If `True`, save the `SkyProjection` attached to the image, if there
            is one.  This does not affect whether a FITS WCS corresponding to
            the projection is written (it always is, if available, and if
            ``add_offset_wcs`` is not ``" "``).
        add_offset_wcs
            A FITS WCS single-character suffix to use when adding a linear
            WCS that maps the FITS array to the logical pixel coordinates
            defined by ``bbox.start`` / ``yx0``.  Set to `None` to not write
            this WCS. If this is set to ``" "``, it will prevent the
            `SkyProjection` from being saved as a FITS WCS.
        tile_shape
            The recommended shape of each tile, if the archive will save
            the array in distinct tiles for faster subarray retrieval.
            This is a hint; archives are not required to use this value.
        options_name
            Use this name to look up archive options.
        """
        if _archive_prefers_native_mask_arrays(archive):
            # HDS presents array dimensions in Fortran order, which is the
            # reverse of the h5py dataset shape. Store the in-memory trailing
            # mask-byte axis first in HDF5 so Starlink tools see HDS axes
            # (x, y, byte), without changing the bit packing within a pixel.
            array_model = archive.add_array(
                np.moveaxis(self._array, -1, 0),
                update_header=update_header,
                tile_shape=tile_shape,
                options_name=options_name,
            )
            if not isinstance(array_model, ArrayReferenceModel):
                raise RuntimeError("Native mask arrays require reference array storage.")
            array_model.shape = list(self._array.shape)
            data: list[ArrayReferenceModel | InlineArrayModel] = [array_model]
        else:
            data = []
            for schema_2d in self.schema.split(np.int32):
                mask_2d = Mask(0, bbox=self.bbox, schema=schema_2d, sky_projection=self._sky_projection)
                mask_2d.update(self)
                data.append(
                    mask_2d._serialize_2d(
                        archive,
                        update_header=update_header,
                        add_offset_wcs=add_offset_wcs,
                        tile_shape=tile_shape,
                        options_name=options_name,
                    )
                )
        serialized_projection: SkyProjectionSerializationModel[P] | None = None
        if save_projection and self.sky_projection is not None:
            serialized_projection = archive.serialize_direct("sky_projection", self.sky_projection.serialize)
        serialized_dtype = NumberType.from_numpy(self.schema.dtype)
        assert is_integer(serialized_dtype), "Mask dtypes should always be integers."
        return MaskSerializationModel.model_construct(
            data=data,
            yx0=list(self.bbox.start),
            planes=list(self.schema),
            dtype=serialized_dtype,
            sky_projection=serialized_projection,
            metadata=self.metadata,
        )

    def _serialize_2d[P: pydantic.BaseModel](
        self,
        archive: OutputArchive[P],
        *,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
        add_offset_wcs: str | None = "A",
        tile_shape: tuple[int, ...] | None = None,
        options_name: str | None = None,
    ) -> ArrayReferenceModel | InlineArrayModel:
        def _update_header(header: astropy.io.fits.Header) -> None:
            update_header(header)
            self.schema.update_header(header)
            if self.sky_projection is not None and add_offset_wcs != " ":
                if self.fits_wcs:
                    header.update(self.fits_wcs.to_header(relax=True))
            if add_offset_wcs is not None:
                fits.add_offset_wcs(header, x=self.bbox.x.start, y=self.bbox.y.start, key=add_offset_wcs)

        assert self.array.shape[2] == 1, "Mask should be split before calling this method."
        return archive.add_array(
            self._array[:, :, 0],
            update_header=_update_header,
            tile_shape=tile_shape,
            options_name=options_name,
        )

    @staticmethod
    def _get_archive_tree_type[P: pydantic.BaseModel](
        pointer_type: type[P],
    ) -> type[MaskSerializationModel[P]]:
        """Return the serialization model type for this object for an archive
        type that uses the given pointer type.
        """
        return MaskSerializationModel[pointer_type]  # type: ignore

    _archive_default_name: ClassVar[str] = "mask"
    """The name this object should be serialized with when written as the
    top-level object.
    """

    @staticmethod
    def from_legacy(
        legacy: Any,
        plane_map: Mapping[str, MaskPlane] | None = None,
    ) -> Mask:
        """Convert from an `lsst.afw.image.Mask` instance.

        Parameters
        ----------
        legacy
            An `lsst.afw.image.Mask` instance.  This will not share pixel
            data with the new object.
        plane_map
            A mapping from legacy mask plane name to the new plane name and
            description.  If not provided, the right legacy mask plane will be
            guessed, but this can depend on which mask planes the legacy
            mask actually has set.
        """
        return Mask._from_legacy_array(
            legacy.array,
            legacy.getMaskPlaneDict(),
            yx0=YX(y=legacy.getY0(), x=legacy.getX0()),
            plane_map=plane_map,
        )

    def to_legacy(self, plane_map: Mapping[str, MaskPlane] | None = None) -> Any:
        """Convert to an `lsst.afw.image.Mask` instance.

        The pixel data will not be shared between the two objects.

        Parameters
        ----------
        plane_map
            A mapping from legacy mask plane name to the new plane name and
            description.
        """
        import lsst.afw.image
        import lsst.geom

        result = lsst.afw.image.Mask(self.bbox.to_legacy())
        if plane_map is None:
            plane_map = {plane.name: plane for plane in self.schema if plane is not None}
        for old_name, new_plane in plane_map.items():
            old_bit = result.addMaskPlane(old_name)
            old_bitmask = 1 << old_bit
            if old_bitmask == 2147483648:
                # afw uses int32 masks, but relies on overflow wrapping, which
                # numpy doesn't like.
                old_bitmask = -2147483648
            if new_plane in self.schema:
                result.array[self.get(new_plane.name)] |= old_bitmask
        return result

    @staticmethod
    def _from_legacy_array(
        array2d: np.ndarray,
        old_planes: Mapping[str, int],
        *,
        yx0: YX[int],
        plane_map: Mapping[str, MaskPlane] | None = None,
        sky_projection: SkyProjection | None = None,
    ) -> Mask:
        if plane_map is None:
            plane_map = _guess_legacy_plane_map(old_planes)
        planes: list[MaskPlane] = list(plane_map.values()) if plane_map is not None else []
        new_name_to_old_bitmask: dict[str, int] = {}
        for old_name, old_bit in old_planes.items():
            old_bitmask = 1 << old_bit
            if old_bitmask == 2147483648:
                # afw uses int32 masks, but relies on overflow wrapping, which
                # numpy doesn't like.
                old_bitmask = -2147483648
            if new_plane := plane_map.get(old_name):
                # Already added to 'planes' at initialization.
                new_name_to_old_bitmask[new_plane.name] = old_bitmask
            else:
                if n_orphaned := np.count_nonzero(array2d & old_bitmask):
                    raise RuntimeError(
                        f"Legacy mask plane {old_name!r} is not remapped, "
                        f"but {n_orphaned} pixels have this bit set."
                    )
        schema = MaskSchema(planes)
        mask = Mask(0, schema=schema, yx0=yx0, shape=array2d.shape, sky_projection=sky_projection)
        for new_name, old_bitmask in new_name_to_old_bitmask.items():
            mask.set(new_name, array2d & old_bitmask)
        return mask

    @staticmethod
    def read_legacy(
        uri: ResourcePathExpression,
        *,
        plane_map: Mapping[str, MaskPlane] | None = None,
        ext: str | int = 1,
        fits_wcs_frame: Frame | None = None,
    ) -> Mask:
        """Read a FITS file written by `lsst.afw.image.Mask.writeFits`.

        Parameters
        ----------
        uri
            URI or file name.
        plane_map
            A mapping from legacy mask plane name to the new plane name and
            description.  If not provided, the right legacy mask plane will be
            guessed, but this can depend on which mask planes the legacy
            mask actually has set.
        ext
            Name or index of the FITS HDU to read.
        fits_wcs_frame
            If not `None` and the HDU containing the mask has a FITS WCS,
            attach a `SkyProjection` to the returned mask by converting that
            WCS.
        """
        opaque_metadata = fits.FitsOpaqueMetadata()
        fs, fspath = ResourcePath(uri).to_fsspec()
        with fs.open(fspath) as stream, astropy.io.fits.open(stream) as hdu_list:
            opaque_metadata.extract_legacy_primary_header(hdu_list[0].header)
            result = Mask._read_legacy_hdu(
                hdu_list[ext], opaque_metadata, plane_map=plane_map, fits_wcs_frame=fits_wcs_frame
            )
            result._opaque_metadata = opaque_metadata
        return result

    @staticmethod
    def _read_legacy_hdu(
        hdu: astropy.io.fits.ImageHDU | astropy.io.fits.CompImageHDU | astropy.io.fits.BinTableHDU,
        opaque_metadata: fits.FitsOpaqueMetadata,
        plane_map: Mapping[str, MaskPlane] | None = None,
        fits_wcs_frame: Frame | None = None,
        strip_legacy_planes: bool = True,
    ) -> Mask:
        if isinstance(hdu, astropy.io.fits.BinTableHDU):
            hdu = astropy.io.fits.CompImageHDU(bintable=hdu)
        yx0 = fits.read_yx0(hdu.header)
        hdu.header.remove("LTV1", ignore_missing=True)
        hdu.header.remove("LTV2", ignore_missing=True)
        sky_projection: SkyProjection | None = None
        if fits_wcs_frame is not None:
            try:
                fits_wcs = astropy.wcs.WCS(hdu.header)
            except KeyError:
                pass
            else:
                sky_projection = SkyProjection.from_fits_wcs(
                    fits_wcs, pixel_frame=fits_wcs_frame, x0=yx0.x, y0=yx0.y
                )
        if any(card.keyword.startswith("MSKN") for card in hdu.header.cards):
            # New ``lsst.images`` form: plane definitions are self-describing
            # via MSKN/MSKM/MSKD cards, so no plane_map is needed.  The on-disk
            # array packs every plane into one element; ``set`` repacks each
            # plane into the (default uint8) in-memory layout by name.
            schema = MaskSchema.from_fits_header(hdu.header)
            mask = Mask(0, schema=schema, yx0=yx0, shape=hdu.data.shape, sky_projection=sky_projection)
            for n, plane in enumerate(schema):
                if plane is not None:
                    mask.set(plane.name, hdu.data & hdu.header.get(f"MSKM{n:04d}", 1 << n))
            schema.strip_header(hdu.header)
        else:
            # Legacy ``lsst.afw.image`` form: bit indices in MP_* cards are
            # mapped to new planes via ``plane_map``.
            old_planes = MaskPlane.read_legacy(hdu.header, strip=strip_legacy_planes)
            resolved_map = plane_map if plane_map is not None else _guess_legacy_plane_map(old_planes)
            mask = Mask._from_legacy_array(
                hdu.data, old_planes, yx0=yx0, plane_map=resolved_map, sky_projection=sky_projection
            )
            if not strip_legacy_planes:
                # Keep the MP_ cards for backwards compatibility, but re-index
                # them to the (reshuffled) positions of the new schema so a
                # legacy reader sees each plane at the bit it is actually
                # packed into on disk.
                _reindex_legacy_plane_cards(hdu.header, old_planes, resolved_map, mask.schema)
        fits.strip_wcs_cards(hdu.header)
        hdu.header.strip()
        hdu.header.remove("EXTTYPE", ignore_missing=True)
        hdu.header.remove("INHERIT", ignore_missing=True)
        # afw set BUNIT on masks because of limitations in how FITS
        # metadata is handled there.
        hdu.header.remove("BUNIT", ignore_missing=True)
        opaque_metadata.add_header(hdu.header)
        return mask


class MaskSerializationModel[P: pydantic.BaseModel](ArchiveTree):
    """Pydantic model used to represent the serialized form of a `.Mask`."""

    SCHEMA_NAME: ClassVar[str] = "mask"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = Mask

    data: list[ArrayReferenceModel | InlineArrayModel] = pydantic.Field(
        description="References to pixel data."
    )
    yx0: list[int] = pydantic.Field(
        description="Coordinate of the first pixels in the array, ordered (y, x)."
    )
    planes: list[MaskPlane | None] = pydantic.Field(description="Definitions of the bitplanes in the mask.")
    dtype: IntegerType = pydantic.Field(description="Data type of the in-memory mask.")
    sky_projection: SkyProjectionSerializationModel[P] | None = pydantic.Field(
        default=None,
        exclude_if=is_none,
        description="Projection that maps the logical pixel grid onto the sky.",
    )

    @property
    def bbox(self) -> Box:
        """The 2-d bounding box of the mask."""
        shape = self.data[0].shape
        if len(shape) == 3:
            shape = shape[:2]
        return Box.from_shape(shape, start=self.yx0)

    def deserialize(
        self,
        archive: InputArchive[Any],
        *,
        bbox: Box | None = None,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
        **kwargs: Any,
    ) -> Mask:
        """Deserialize a mask from an input archive.

        Parameters
        ----------
        archive
            Archive to read from.
        bbox
            Bounding box of a subimage to read instead.
        strip_header
            A callable that strips out any FITS header cards added by the
            ``update_header`` argument in the corresponding call to
            `Mask.serialize`.
        **kwargs
            Unsupported keyword arguments are accepted only to provide better
            error messages (raising `serialization.InvalidParameterError`).
        """
        if kwargs:
            raise InvalidParameterError(f"Unrecognized parameters for Mask: {set(kwargs.keys())}.")
        slices: tuple[slice, ...] | EllipsisType = ...
        if bbox is not None:
            slices = bbox.slice_within(self.bbox)
        else:
            bbox = self.bbox
        if not is_integer(self.dtype):
            raise ArchiveReadError(f"Mask array has a non-integer dtype: {self.dtype}.")
        schema = MaskSchema(self.planes, dtype=self.dtype.to_numpy())
        sky_projection = self.sky_projection.deserialize(archive) if self.sky_projection is not None else None
        if len(self.data) == 1 and tuple(self.data[0].shape) == tuple(self.bbox.shape) + (schema.mask_size,):
            storage_slices = slices if slices is ... else (slice(None),) + slices
            array = archive.get_array(self.data[0], strip_header=strip_header, slices=storage_slices)
            array = np.moveaxis(array, 0, -1)
            return Mask(array, schema=schema, bbox=bbox, sky_projection=sky_projection)._finish_deserialize(
                self
            )
        result = Mask(0, schema=schema, bbox=bbox, sky_projection=sky_projection)
        schemas_2d = schema.split(np.int32)
        if len(schemas_2d) != len(self.data):
            raise ArchiveReadError(
                f"Number of mask arrays ({len(self.data)}) does not match expectation ({len(schemas_2d)})."
            )
        for array_model, schema_2d in zip(self.data, schemas_2d):
            mask_2d = self._deserialize_2d(
                array_model, schema_2d, bbox.start, archive, strip_header=strip_header, slices=slices
            )
            result.update(mask_2d)
        return result._finish_deserialize(self)

    @staticmethod
    def _deserialize_2d(
        ref: ArrayReferenceModel | InlineArrayModel,
        schema_2d: MaskSchema,
        yx0: Sequence[int],
        archive: InputArchive[Any],
        *,
        slices: tuple[slice, ...] | EllipsisType = ...,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> Mask:
        def _strip_header(header: astropy.io.fits.Header) -> None:
            strip_header(header)
            schema_2d.strip_header(header)
            fits.strip_wcs_cards(header)

        array_2d = archive.get_array(ref, strip_header=_strip_header, slices=slices)
        return Mask(array_2d[:, :, np.newaxis], schema=schema_2d, yx0=yx0)

    def deserialize_component(self, component: str, archive: InputArchive[Any], **kwargs: Any) -> Any:
        if kwargs:
            raise InvalidParameterError(f"Unsupported parameters for Mask components: {set(kwargs.keys())}.")
        return super().deserialize_component(component, archive)


def _archive_prefers_native_mask_arrays(archive: OutputArchive[Any]) -> bool:
    """Return whether an archive wants masks in their native 3-D layout."""
    current: Any = archive
    while current is not None:
        if getattr(current, "_prefer_native_mask_arrays", False):
            return True
        current = getattr(current, "_parent", None)
    return False


def get_legacy_visit_image_mask_planes() -> dict[str, MaskPlane]:
    """Return a mapping from legacy mask plane name to `MaskPlane` instance
    for LSST visit images, c. DP2.
    """
    return {
        "BAD": MaskPlane("BAD", "Bad pixel in the instrument, including bad amplifiers."),
        "SAT": MaskPlane(
            "SATURATED", "Pixel was saturated or affected by saturation in a neighboring pixel."
        ),
        "INTRP": MaskPlane("INTERPOLATED", "Original pixel value was interpolated."),
        "CR": MaskPlane("COSMIC_RAY", "A cosmic ray affected this pixel."),
        "EDGE": MaskPlane(
            "DETECTION_EDGE",
            "Pixel was too close to the edge to be considered for detection, "
            "due to the finite size of the detection kernel.",
        ),
        "DETECTED": MaskPlane("DETECTED", "Pixel was part of a detected source."),
        "SUSPECT": MaskPlane("SUSPECT", "Pixel was close to the saturation level. "),
        "NO_DATA": MaskPlane("NO_DATA", "No data was available for this pixel."),
        "VIGNETTED": MaskPlane("VIGNETTED", "Pixel was vignetted by the optics."),
        "PARTLY_VIGNETTED": MaskPlane("PARTLY_VIGNETTED", "Pixel was partly vignetted by the optics."),
        "CROSSTALK": MaskPlane("CROSSTALK", "Pixel was affected by crosstalk and corrected accordingly."),
        "ITL_DIP": MaskPlane(
            "ITL_DIP", "Pixel was affected by a dark vertical trail from a bright source, on an ITL CCD."
        ),
        "NOT_DEBLENDED": MaskPlane(
            "NOT_DEBLENDED",
            "Pixel belonged to a detection that was not deblended, usually due to size limits.",
        ),
        "SPIKE": MaskPlane(
            "SPIKE", "Pixel is in the neighborhood of a diffraction spike from a bright star."
        ),
        "UNMASKEDNAN": MaskPlane("UNMASKED_NAN", "Pixel was found to be NaN unexpectedly."),
    }


def get_legacy_difference_image_mask_planes() -> dict[str, MaskPlane]:
    """Return a mapping from legacy mask plane name to `MaskPlane` instance
    for LSST difference images, c. DP2.
    """
    result = get_legacy_visit_image_mask_planes()
    result["DETECTED_NEGATIVE"] = MaskPlane(
        "DETECTED_NEGATIVE", "Pixel was part of a detected source with negative flux."
    )
    result["SAT_TEMPLATE"] = MaskPlane("SAT_TEMPLATE", "Template pixel was saturated.")
    result["HIGH_VARIANCE"] = MaskPlane("HIGH_VARIANCE", "TODO[DM-55036]")
    result["STREAK"] = MaskPlane(
        "STREAK", "An extended streak (probably an artificial satellite) affected this pixel."
    )
    return result


def get_legacy_deep_coadd_mask_planes() -> dict[str, MaskPlane]:
    """Return a mapping from legacy mask plane name to `MaskPlane` instance
    for LSST deep coadds, c. DP2.
    """
    return {
        "NO_DATA": MaskPlane("NO_DATA", "No data was available for this pixel."),
        "INTRP": MaskPlane("INTERPOLATED", "Pixel value is the result of interpolating nearby good pixels."),
        "CR": MaskPlane(
            "COSMIC_RAY",
            "A cosmic ray affected this pixel on at least one input image (and was interpolated).",
        ),
        "SAT": MaskPlane(
            "SATURATED",
            "More than 10% of the potential input visits had a saturated pixel at this location "
            "('potential' because saturated pixel values are not actually propagated to the coadd). "
            "SATURATED always implies REJECTED, and is often a reason for NO_DATA.",
        ),
        "EDGE": MaskPlane(
            "DETECTION_EDGE",
            "Pixel was too close to the edge of the patch to be considered for detection, "
            "due to the finite size of the detection kernel.",
        ),
        "CLIPPED": MaskPlane(
            "CLIPPED",
            "Region was identified as a probable artifact when comparing multiple single-visit warps. "
            "CLIPPED always implies REJECTED.",
        ),
        "REJECTED": MaskPlane(
            "REJECTED",
            "At least one input visit was left out of the coadd for this pixel due to masking. "
            "REJECTED always implies INEXACT_PSF.",
        ),
        "DETECTED": MaskPlane("DETECTED", "Pixel was part of a detected source."),
        "INEXACT_PSF": MaskPlane(
            "INEXACT_PSF",
            "The set of visits contributing to this pixel differs from the set of visits "
            "contributing to the PSF model for its cell.",
        ),
    }


def get_legacy_non_cell_coadd_mask_planes() -> dict[str, MaskPlane]:
    """Return a mapping from legacy mask plane name to `MaskPlane` instance
    for LSST non-cell coadds such as ``template_coadd`` in DP2, and all
    DP1 coadds.

    These coadds carry the visit-level planes propagated from their input
    warps in addition to the coadd-specific planes, and flag chip edges with
    ``SENSOR_EDGE`` (cell coadds use ``CELL_EDGE`` instead).
    """
    result = get_legacy_deep_coadd_mask_planes()
    result["BAD"] = MaskPlane("BAD", "Bad pixel in the instrument, including bad amplifiers.")
    result["SUSPECT"] = MaskPlane("SUSPECT", "Pixel was close to the saturation level.")
    result["CROSSTALK"] = MaskPlane("CROSSTALK", "Pixel was affected by crosstalk and corrected accordingly.")
    result["DETECTED_NEGATIVE"] = MaskPlane(
        "DETECTED_NEGATIVE", "Pixel was part of a detected source with negative flux."
    )
    result["NOT_DEBLENDED"] = MaskPlane(
        "NOT_DEBLENDED",
        "Pixel belonged to a detection that was not deblended, usually due to size limits.",
    )
    result["UNMASKEDNAN"] = MaskPlane("UNMASKED_NAN", "Pixel was found to be NaN unexpectedly.")
    result["SENSOR_EDGE"] = MaskPlane(
        "SENSOR_EDGE",
        "Pixel is near the edge of a contributing sensor/chip, so the coadd PSF is discontinuous there.",
    )
    return result


def _guess_legacy_plane_map(old_planes: Mapping[str, int]) -> dict[str, MaskPlane]:
    """Guess which of the ``get_legacy_*_plane_map`` created the given mask
    plane dictionary and call it.
    """
    if "SAT_TEMPLATE" in old_planes:
        return get_legacy_difference_image_mask_planes()
    if "INEXACT_PSF" in old_planes:
        # Both cell and non-cell coadds have INEXACT_PSF, but only non-cell
        # (assemble_coadd) coadds flag chip edges with SENSOR_EDGE; cell coadds
        # use CELL_EDGE.
        if "SENSOR_EDGE" in old_planes:
            return get_legacy_non_cell_coadd_mask_planes()
        return get_legacy_deep_coadd_mask_planes()
    return get_legacy_visit_image_mask_planes()


def _reindex_legacy_plane_cards(
    header: astropy.io.fits.Header,
    old_planes: Mapping[str, int],
    plane_map: Mapping[str, MaskPlane],
    schema: MaskSchema,
) -> None:
    """Rewrite retained legacy ``MP_`` cards in place to match a reshuffled
    schema.

    Parameters
    ----------
    header
        Header whose ``MP_`` cards are updated in place.
    old_planes
        Mapping from legacy mask plane name to its original (on-disk) bit
        index, as returned by `MaskPlane.read_legacy`.
    plane_map
        Mapping from legacy mask plane name to the `MaskPlane` it was remapped
        to in ``schema``.
    schema
        The reconstructed schema that defines the new bit positions.

    Notes
    -----
    Each ``MP_<legacy name>`` card is set to the index that its remapped plane
    occupies in ``schema`` (equivalently, the ``MSKN`` index written on
    serialization).  Cards for legacy planes that are not represented in the
    new schema are removed, since they no longer correspond to any stored bit.
    Legacy masks have at most 31 planes, so every plane maps to a single bit in
    one on-disk element and the index is unambiguous.
    """
    new_index = {plane.name: n for n, plane in enumerate(schema) if plane is not None}
    for old_name in old_planes:
        keyword = f"MP_{old_name}"
        new_plane = plane_map.get(old_name)
        if new_plane is not None and (index := new_index.get(new_plane.name)) is not None:
            header[keyword] = index
        else:
            del header[keyword]
