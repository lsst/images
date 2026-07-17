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

__all__ = ("ColorImage",)

import functools
from collections.abc import Sequence
from types import EllipsisType
from typing import Any, ClassVar, Literal

import numpy as np
import pydantic

from ._generalized_image import GeneralizedImage
from ._geom import Box
from ._image import Image, ImageSerializationModel
from ._transforms import SkyProjection, SkyProjectionSerializationModel
from .serialization import ArchiveTree, InputArchive, InvalidParameterError, MetadataValue, OutputArchive
from .utils import is_none


class ColorImage(GeneralizedImage):
    """An RGB image with an optional `SkyProjection`.

    Parameters
    ----------
    array
        Array or fill value for the image.  Must have three dimensions with
        the shape of the third dimension equal to three.
    bbox
        Bounding box for the image.
    yx0
        Logical coordinates of the first pixel in the array, ordered ``y``,
        ``x`` (unless an `XY` instance is passed).  Ignored if
        ``bbox`` is provided.  Defaults to zeros.
    sky_projection
        Projection that maps the pixel grid to the sky.
    metadata
        Arbitrary flexible metadata to associate with the image.
    """

    def __init__(
        self,
        array: np.ndarray[tuple[int, int, Literal[3]], np.dtype[Any]],
        /,
        *,
        bbox: Box | None = None,
        yx0: Sequence[int] | None = None,
        sky_projection: SkyProjection[Any] | None = None,
        metadata: dict[str, MetadataValue] | None = None,
    ) -> None:
        super().__init__(metadata)
        if bbox is None:
            bbox = Box.from_shape(array.shape[:2], start=yx0)
        elif bbox.shape + (3,) != array.shape:
            raise ValueError(
                f"Shape from bbox {bbox.shape + (3,)} does not match array with shape {array.shape}."
            )
        self._array = array
        self._red = Image(self._array[..., 0], bbox=bbox, sky_projection=sky_projection)
        self._green = Image(self._array[..., 1], bbox=bbox, sky_projection=sky_projection)
        self._blue = Image(self._array[..., 2], bbox=bbox, sky_projection=sky_projection)

    @staticmethod
    def from_channels(
        r: Image,
        g: Image,
        b: Image,
        *,
        sky_projection: SkyProjection[Any] | None = None,
        metadata: dict[str, MetadataValue] | None = None,
    ) -> ColorImage:
        """Construct from separate RGB images.

        All channels are assumed to have the same bounding box, sky_projection,
        and pixel type.

        Parameters
        ----------
        r
            Red channel image.
        g
            Green channel image.
        b
            Blue channel image.
        sky_projection
            Sky projection for the image, defaulting to that of ``r``.
        metadata
            Flexible metadata to associate with the image.
        """
        if sky_projection is None and r.sky_projection is not None:
            sky_projection = r.sky_projection
        return ColorImage(
            np.stack([r.array, g.array, b.array], axis=2),
            bbox=r.bbox,
            sky_projection=sky_projection,
            metadata=metadata,
        )

    @property
    def array(self) -> np.ndarray[tuple[int, int, Literal[3]], np.dtype[Any]]:
        """The 3-d array (`numpy.ndarray`)."""
        return self._array

    @property
    def red(self) -> Image:
        """A 2-d view of the red channel (`Image`)."""
        return self._red

    @property
    def green(self) -> Image:
        """A 2-d view of the green channel (`Image`)."""
        return self._green

    @property
    def blue(self) -> Image:
        """A 2-d view of the blue channel (`Image`)."""
        return self._blue

    @property
    def bbox(self) -> Box:
        """The 2-d bounding box of the image (`Box`)."""
        return self._red.bbox

    @property
    def sky_projection(self) -> SkyProjection[Any] | None:
        """The projection that maps the pixel grid to the sky
        (`SkyProjection` | `None`).
        """
        return self._red.sky_projection

    def __getitem__(self, bbox: Box | EllipsisType) -> ColorImage:
        bbox, indices = self._handle_getitem_args(bbox)
        return self._transfer_metadata(
            ColorImage(
                self.array[indices + (slice(None),)],
                bbox=bbox,
                sky_projection=self.sky_projection,
            ),
            bbox=bbox,
        )

    def __setitem__(self, bbox: Box | EllipsisType, value: ColorImage) -> None:
        self[bbox].array[...] = value.array

    def __str__(self) -> str:
        return f"ColorImage({self.bbox!s}, {self._array.dtype.type.__name__})"

    def __repr__(self) -> str:
        return f"ColorImage(..., bbox={self.bbox!r}, dtype={self._array.dtype!r})"

    def copy(self) -> ColorImage:
        """Deep-copy the image."""
        return self._transfer_metadata(
            ColorImage(self._array.copy(), bbox=self.bbox, sky_projection=self.sky_projection), copy=True
        )

    def serialize(self, archive: OutputArchive[Any]) -> ColorImageSerializationModel:
        """Serialize the masked image to an output archive.

        Parameters
        ----------
        archive
            Archive to write to.
        """
        r = archive.serialize_direct("red", functools.partial(self.red.serialize, save_projection=False))
        g = archive.serialize_direct("green", functools.partial(self.green.serialize, save_projection=False))
        b = archive.serialize_direct("blue", functools.partial(self.blue.serialize, save_projection=False))
        serialized_projection = (
            archive.serialize_direct("sky_projection", self.sky_projection.serialize)
            if self.sky_projection is not None
            else None
        )
        return ColorImageSerializationModel(
            red=r, green=g, blue=b, sky_projection=serialized_projection, metadata=self.metadata
        )

    @staticmethod
    def _get_archive_tree_type[P: pydantic.BaseModel](
        pointer_type: type[P],
    ) -> type[ColorImageSerializationModel[P]]:
        """Return the serialization model type for this object for an archive
        type that uses the given pointer type.
        """
        return ColorImageSerializationModel[pointer_type]  # type: ignore


class ColorImageSerializationModel[P: pydantic.BaseModel](ArchiveTree):
    """A Pydantic model used to represent a serialized `ColorImage`."""

    SCHEMA_NAME: ClassVar[str] = "color_image"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0.dev0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = ColorImage

    red: ImageSerializationModel[P] = pydantic.Field(description="The red channel.")
    green: ImageSerializationModel[P] = pydantic.Field(description="The green channel.")
    blue: ImageSerializationModel[P] = pydantic.Field(description="The blue channel")
    sky_projection: SkyProjectionSerializationModel[P] | None = pydantic.Field(
        default=None,
        exclude_if=is_none,
        description="Projection that maps the pixel grid to the sky.",
    )

    @property
    def bbox(self) -> Box:
        """The bounding box of the image."""
        return self.red.bbox

    def deserialize(
        self, archive: InputArchive[Any], *, bbox: Box | None = None, **kwargs: Any
    ) -> ColorImage:
        """Deserialize a image from an input archive.

        Parameters
        ----------
        archive
            Archive to read from.
        bbox
            Bounding box of a subimage to read instead.
        **kwargs
            Unsupported keyword arguments are accepted only to provide
            better error messages (raising
            `.serialization.InvalidParameterError`).
        """
        if kwargs:
            raise InvalidParameterError(f"Unrecognized parameters for ColoImage: {set(kwargs.keys())}.")
        r = self.red.deserialize(archive, bbox=bbox)
        g = self.green.deserialize(archive, bbox=bbox)
        b = self.blue.deserialize(archive, bbox=bbox)
        sky_projection = self.sky_projection.deserialize(archive) if self.sky_projection is not None else None
        return ColorImage.from_channels(r, g, b, sky_projection=sky_projection)._finish_deserialize(self)
