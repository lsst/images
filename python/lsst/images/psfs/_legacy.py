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

__all__ = ("LegacyPointSpreadFunction", "PSFExSerializationModel", "PSFExWrapper")

from functools import cached_property
from typing import Any

import numpy as np
import pydantic

from .. import serialization
from .._geom import Bounds, Box, SerializableBounds
from .._image import Image
from ._base import PointSpreadFunction


class LegacyPointSpreadFunction(PointSpreadFunction):
    """A PSF model backed by a legacy `lsst.afw.detection.Psf` object.

    Parameters
    ----------
    impl
        An `lsst.afw.detection.Psf` instance.
    bounds
        The pixel-coordinate region where the model can safely be evaluated.

    Notes
    -----
    This wrapper is usable as-is on any `lsst.afw.detection.Psf` instance,
    but subclasses (e.g. `PSFExWrapper`) must be used for serialization.
    """

    def __init__(self, impl: Any, bounds: Bounds):
        self._impl = impl
        self._bounds = bounds

    @property
    def bounds(self) -> Bounds:
        return self._bounds

    @cached_property
    def kernel_bbox(self) -> Box:
        from lsst.geom import Box2I, Point2D

        biggest = Box2I()
        for y, x in self._bounds.boundary():
            biggest.include(self._impl.computeKernelBBox(Point2D(x, y)))
        return Box.from_legacy(biggest)

    def compute_kernel_image(self, *, x: float, y: float) -> Image:
        from lsst.geom import Point2D

        result = Image.from_legacy(self._impl.computeKernelImage(Point2D(x, y)))
        if result.bbox != self.kernel_bbox:
            # afw does not guarantee a consistent kernel_bbox, but we do now.
            padded = Image(0.0, bbox=self.kernel_bbox, dtype=np.float64)
            padded[self.kernel_bbox] = result[self.kernel_bbox]
            result = padded
        return result

    def compute_stellar_image(self, *, x: float, y: float) -> Image:
        from lsst.geom import Point2D

        return Image.from_legacy(self._impl.computeImage(Point2D(x, y)))

    def compute_stellar_bbox(self, *, x: float, y: float) -> Box:
        from lsst.geom import Point2D

        return Box.from_legacy(self._impl.computeImageBBox(Point2D(x, y)))

    @property
    def legacy_psf(self) -> Any:
        """The backing `lsst.afw.detection.Psf` object."""
        return self._impl

    @classmethod
    def from_legacy(cls, legacy_psf: Any, bounds: Bounds) -> LegacyPointSpreadFunction:
        from lsst.meas.extensions.psfex import PsfexPsf

        if isinstance(legacy_psf, PsfexPsf):
            return PSFExWrapper(legacy_psf, bounds)
        return cls(impl=legacy_psf, bounds=bounds)


class PSFExWrapper(LegacyPointSpreadFunction):
    """A specialization of LegacyPointSpreadFunction for the PSFEx backend."""

    def __init__(self, impl: Any, bounds: Bounds):
        from lsst.meas.extensions.psfex import PsfexPsf

        if not isinstance(impl, PsfexPsf):
            raise TypeError(f"{impl!r} is not a PSFEx object.")
        super().__init__(impl, bounds)

    def serialize(self, archive: serialization.OutputArchive[Any]) -> PSFExSerializationModel:
        """Serialize the PSF to an archive.

        This method is intended to be usable as the callback function passed to
        `.serialization.OutputArchive.serialize_direct` or
        `.serialization.OutputArchive.serialize_pointer`.
        """
        data = self._impl.getSerializationData()
        shape = tuple(reversed(data.size))
        dtype = np.dtype([("parameters", data.comp.dtype, shape[1:])])
        structured_array = np.empty(shape[:1], dtype=dtype)
        structured_array["parameters"] = data.comp.reshape(*shape)
        table_ref = archive.add_structured_array("parameters", structured_array)
        return PSFExSerializationModel(
            average_x=data.average_x,
            average_y=data.average_y,
            pixel_step=data.pixel_step,
            group=data.group,
            degree=data.degree,
            basis=data.basis,
            coeff=data.coeff,
            parameters=table_ref,
            context=data.context,
            bounds=self.bounds.serialize(),
        )

    @classmethod
    def deserialize(
        cls, model: PSFExSerializationModel, archive: serialization.InputArchive[Any]
    ) -> PSFExWrapper:
        """Deserialize the PSF from an archive.

        This method is intended to be usable as the callback function passed to
        `.serialization.InputArchive.deserialize_pointer`.
        """
        from lsst.meas.extensions.psfex import PsfexPsf, PsfexPsfSerializationData

        structured_array = archive.get_structured_array(model.parameters)
        parameters = structured_array["parameters"].astype(np.float32)
        data = PsfexPsfSerializationData()
        data.average_x = model.average_x
        data.average_y = model.average_y
        data.pixel_step = model.pixel_step
        data.group = model.group
        data.degree = model.degree
        data.basis = model.basis
        data.coeff = model.coeff
        data.size = list(reversed(parameters.shape))
        data.comp = parameters.flatten()
        data.context = model.context
        legacy_psf = PsfexPsf.fromSerializationData(data)
        return cls(legacy_psf, Bounds.deserialize(model.bounds))


class PSFExSerializationModel(pydantic.BaseModel):
    """Serialization model for PSFEx PSFs."""

    average_x: float = pydantic.Field(
        description="Average X position of the stars used to build this PSF model."
    )

    average_y: float = pydantic.Field(
        description="Average Y position of the stars used to build this PSF model."
    )

    pixel_step: float = pydantic.Field(
        description="Size of a model pixel, as a fraction or multiple of the native pixel size."
    )

    group: list[int] = pydantic.Field(
        default_factory=lambda: [0, 0],
        exclude_if=lambda v: v == [0, 0],
        description="Number of model groups in each dimension.",
    )

    degree: list[int] = pydantic.Field(description="Polynomial degree for each model group.")

    basis: list[float] = pydantic.Field(description="Basis function values.")

    coeff: list[float] = pydantic.Field(description="Polynomial coefficients.")

    parameters: serialization.TableModel = pydantic.Field(
        description="Reference to a table with the complete model parameters."
    )

    context: serialization.InlineArray = pydantic.Field(description="Internal PSFEx context array.")

    bounds: SerializableBounds = pydantic.Field(description="Validity range for this PSF model.")
