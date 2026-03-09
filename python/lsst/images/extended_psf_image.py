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

__all__ = ("ExtendedPsfFit", "ExtendedPsfImage", "ExtendedPsfImageSerializationModel")

import functools
from types import EllipsisType
from typing import Any

import numpy as np
from astropy.units import UnitBase
from pydantic import BaseModel, Field

from ._generalized_image import GeneralizedImage
from ._geom import Box
from ._image import Image, ImageSerializationModel
from .serialization import ArchiveTree, InputArchive, MetadataValue, OutputArchive


class ExtendedPsfFit(BaseModel):
    """Base class for ExtendedPsf fit results."""

    chi2: float
    reduced_chi2: float


class ExtendedPsfImage(GeneralizedImage):
    """A multi-plane image with data (image) and variance planes, and the
    results of a profile fit to the image.

    Parameters
    ----------
    image
        The main image plane.
    variance
        The per-pixel uncertainty of the main image as an image of variance
        values. Must have the same bounding box as ``image`` if provided, and
        its units must be the square of ``image.unit`` or `None`.
        Values default to ``1.0``. Any attached projection is replaced
        (possibly by `None`).
    fit
        The results of a profile fit to the image.
    metadata
        Arbitrary flexible metadata to associate with the image.
    """

    def __init__(
        self,
        image: Image,
        *,
        variance: Image | None = None,
        fit: ExtendedPsfFit | None = None,
        metadata: dict[str, MetadataValue] | None = None,
    ):
        super().__init__(metadata)
        if variance is None:
            variance = Image(
                1.0,
                dtype=np.float32,
                bbox=image.bbox,
                unit=None if image.unit is None else image.unit**2,
            )
        else:
            if image.bbox != variance.bbox:
                raise ValueError(f"Image ({image.bbox}) and variance ({variance.bbox}) bboxes do not agree.")
            if image.unit is None:
                if variance.unit is not None:
                    raise ValueError(f"Image has no units but variance does ({variance.unit}).")
            elif variance.unit is None:
                variance = variance.view(unit=image.unit**2)
            elif variance.unit != image.unit**2:
                raise ValueError(
                    f"Variance unit ({variance.unit}) should be the square of the image unit ({image.unit})."
                )
        if fit is None:
            fit = ExtendedPsfFit(chi2=np.nan, reduced_chi2=np.nan)
        self._image = image
        self._variance = variance
        self._fit = fit

    @property
    def image(self) -> Image:
        """The main image plane (`Image`)."""
        return self._image

    @property
    def variance(self) -> Image:
        """The variance plane (`Image`)."""
        return self._variance

    @property
    def bbox(self) -> Box:
        """The bounding box shared by both image planes (`Box`)."""
        return self._image.bbox

    @property
    def unit(self) -> UnitBase | None:
        """The units of the image plane (`astropy.units.Unit` | `None`)."""
        return self._image.unit

    @property
    def projection(self) -> None:
        """The projection that maps the pixel grid to the sky.

        ExtendedPsfImage does not support attached projections,
        so this always returns `None`.
        """
        return None

    @property
    def fit(self) -> ExtendedPsfFit:
        """The results of a profile fit to the image (`ExtendedPsfFit`)."""
        return self._fit

    def __getitem__(self, bbox: Box | EllipsisType) -> ExtendedPsfImage:
        super().__getitem__(bbox)
        if bbox is ...:
            return self
        return self._transfer_metadata(
            ExtendedPsfImage(
                self.image[bbox],
                variance=self.variance[bbox],
                fit=self.fit,
            ),
            bbox=bbox,
        )

    def __setitem__(self, bbox: Box | EllipsisType, value: ExtendedPsfImage) -> None:
        self._image[bbox] = value.image
        self._variance[bbox] = value.variance

    def __str__(self) -> str:
        return f"ExtendedPsfImage({self.image!s}, fit={self.fit!r})"

    def __repr__(self) -> str:
        return f"ExtendedPsfImage({self.image!r}, fit={self.fit!r})"

    def copy(self) -> ExtendedPsfImage:
        """Deep-copy the profile image and metadata."""
        return self._transfer_metadata(
            ExtendedPsfImage(
                image=self._image.copy(), variance=self._variance.copy(), fit=self._fit.model_copy()
            ),
            copy=True,
        )

    def serialize(self, archive: OutputArchive[Any]) -> ExtendedPsfImageSerializationModel:
        """Serialize the Extended PSF image to an output archive.

        Parameters
        ----------
        archive
            Archive to write to.
        """
        serialized_image = archive.serialize_direct(
            "image", functools.partial(self.image.serialize, save_projection=False)
        )
        serialized_variance = archive.serialize_direct(
            "variance", functools.partial(self.variance.serialize, save_projection=False)
        )
        serialized_fit = self.fit
        return ExtendedPsfImageSerializationModel(
            image=serialized_image,
            variance=serialized_variance,
            fit=serialized_fit,
            metadata=self.metadata,
        )

    @staticmethod
    def deserialize(
        model: ExtendedPsfImageSerializationModel[Any], archive: InputArchive[Any], *, bbox: Box | None = None
    ) -> ExtendedPsfImage:
        """Deserialize an image from an input archive.

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
        image = Image.deserialize(model.image, archive, bbox=bbox)
        variance = Image.deserialize(model.variance, archive, bbox=bbox)
        fit = model.fit
        return ExtendedPsfImage(image, variance=variance, fit=fit)._finish_deserialize(model)

    @staticmethod
    def _get_archive_tree_type[P: BaseModel](
        pointer_type: type[P],
    ) -> type[ExtendedPsfImageSerializationModel[P]]:
        """Return the serialization model type for this object for an archive
        type that uses the given pointer type.
        """
        return ExtendedPsfImageSerializationModel[pointer_type]  # type: ignore


class ExtendedPsfImageSerializationModel[P: BaseModel](ArchiveTree):
    """A Pydantic model used to represent a serialized `ExtendedPsfImage`."""

    image: ImageSerializationModel[P] = Field(description="The main data image.")
    variance: ImageSerializationModel[P] = Field(
        description="Per-pixel variance estimates for the main image."
    )
    fit: ExtendedPsfFit = Field(description="The results of an extended PSF fit to the image.")

    @property
    def bbox(self) -> Box:
        """The bounding box of the image."""
        return self.image.bbox
