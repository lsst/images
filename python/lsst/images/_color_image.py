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
from typing import Any, Literal

import numpy as np
import pydantic

from ._generalized_image import GeneralizedImage
from ._geom import Box
from ._image import Image, ImageSerializationModel
from ._transforms import Projection, ProjectionSerializationModel
from .serialization import ArchiveTree, InputArchive, MetadataValue, OutputArchive
from .utils import is_none


class ColorImage(GeneralizedImage):
    """An RGB image with an optional `Projection`.

    Parameters
    ----------
    array
        Array or fill value for the image.  Must have three dimensions with
        the shape of the third dimension equal to three.
    bbox
        Bounding box for the image.
    start
        Logical coordinates of the first pixel in the array, ordered ``y``,
        ``x`` (unless an `XY` instance is passed).  Ignored if
        ``bbox`` is provided.  Defaults to zeros.
    projection
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
        start: Sequence[int] | None = None,
        projection: Projection[Any] | None = None,
        metadata: dict[str, MetadataValue] | None = None,
    ):
        super().__init__(metadata)
        if bbox is None:
            bbox = Box.from_shape(array.shape[:2], start=start)
        elif bbox.shape + (3,) != array.shape:
            raise ValueError(
                f"Shape from bbox {bbox.shape + (3,)} does not match array with shape {array.shape}."
            )
        self._array = array
        self._red = Image(self._array[..., 0], bbox=bbox, projection=projection)
        self._green = Image(self._array[..., 1], bbox=bbox, projection=projection)
        self._blue = Image(self._array[..., 2], bbox=bbox, projection=projection)

    @staticmethod
    def from_channels(
        r: Image,
        g: Image,
        b: Image,
        *,
        projection: Projection[Any] | None = None,
        metadata: dict[str, MetadataValue] | None = None,
    ) -> ColorImage:
        """Construct from separate RGB images.

        All channels are assumed to have the same bounding box, projection,
        and pixel type.
        """
        if projection is None and r.projection is not None:
            projection = r.projection
        return ColorImage(
            np.stack([r.array, g.array, b.array], axis=2),
            bbox=r.bbox,
            projection=projection,
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
    def projection(self) -> Projection[Any] | None:
        """The projection that maps the pixel grid to the sky
        (`Projection` | `None`).
        """
        return self._red.projection

    def __getitem__(self, bbox: Box | EllipsisType) -> ColorImage:
        super().__getitem__(bbox)
        if bbox is ...:
            return self
        return self._transfer_metadata(
            ColorImage(
                self.array[bbox.slice_within(self.bbox) + (slice(None),)],
                bbox=bbox,
                projection=self.projection,
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
            ColorImage(self._array.copy(), bbox=self.bbox, projection=self.projection), copy=True
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
            archive.serialize_direct("projection", self.projection.serialize)
            if self.projection is not None
            else None
        )
        return ColorImageSerializationModel(
            red=r, green=g, blue=b, projection=serialized_projection, metadata=self.metadata
        )

    @staticmethod
    def deserialize(
        model: ColorImageSerializationModel[Any], archive: InputArchive[Any], *, bbox: Box | None = None
    ) -> ColorImage:
        """Deserialize a image from an input archive.

        Parameters
        ----------
        model
            A Pydantic model representation of the image, holding references
            to data stored in the archive.
        archive
            Archive to read from.
        bbox
            Bounding box of a subimage to read instead.
        """
        r = Image.deserialize(model.red, archive, bbox=bbox)
        g = Image.deserialize(model.green, archive, bbox=bbox)
        b = Image.deserialize(model.blue, archive, bbox=bbox)
        projection = (
            Projection.deserialize(model.projection, archive) if model.projection is not None else None
        )
        return ColorImage.from_channels(r, g, b, projection=projection)._finish_deserialize(model)

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

    red: ImageSerializationModel[P] = pydantic.Field(description="The red channel.")
    green: ImageSerializationModel[P] = pydantic.Field(description="The green channel.")
    blue: ImageSerializationModel[P] = pydantic.Field(description="The blue channel")
    projection: ProjectionSerializationModel[P] | None = pydantic.Field(
        default=None,
        exclude_if=is_none,
        description="Projection that maps the pixel grid to the sky.",
    )

    @property
    def bbox(self) -> Box:
        """The bounding box of the image."""
        return self.red.bbox
