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
    "ConvolutionKernel",
    "ConvolutionKernelSerializationModel",
    "ImageBasisConvolutionKernel",
    "ImageBasisConvolutionKernelSerializationModel",
)

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import numpy as np
import pydantic

from ._geom import YX, Bounds, Box
from ._image import Image
from .fields import ChebyshevField, Field, FieldSerializationModel
from .serialization import (
    ArchiveTree,
    ArrayReferenceModel,
    InlineArrayModel,
    InputArchive,
    InvalidParameterError,
    OutputArchive,
)

if TYPE_CHECKING:
    try:
        from lsst.afw.math import LinearCombinationKernel as LegacyLinearCombinationKernel
    except ImportError:
        type LegacyLinearCombinationKernel = Any  # type: ignore[no-redef]


# This may become a union in the future.
type ConvolutionKernelSerializationModel = ImageBasisConvolutionKernelSerializationModel


class ConvolutionKernel(ABC):
    """An abstract base class for spatially-varying convolution kernels."""

    @property
    @abstractmethod
    def bounds(self) -> Bounds:
        """The region where this convolution kernel is valid
        (`~lsst.images.Bounds`).
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def kernel_bbox(self) -> Box:
        """Bounding box of all images returned by `compute_kernel_image`
        (`~lsst.images.Box`).
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_kernel_image(self, *, x: int, y: int) -> Image:
        """Evaluate the kernel at a point.

        Parameters
        ----------
        x
            Column position coordinate to evaluate at.
        y
            Row position coordinate to evaluate at.

        Returns
        -------
        Image
            An image of the kernel, centered on the center of the center pixel,
            which is defined to be ``(0, 0)`` by the image's origin.
        """
        raise NotImplementedError()

    @abstractmethod
    def serialize(self, archive: OutputArchive[Any]) -> ConvolutionKernelSerializationModel:
        """Serialize the kernel to an output archive.

        Parameters
        ----------
        archive
            Archive to write to.
        """
        raise NotImplementedError()


class ImageBasisConvolutionKernel(ConvolutionKernel):
    """A convolution kernel formed by a linear combination of images
    multiplied by `~lsst.images.fields.BaseField` instances.

    Parameters
    ----------
    basis
        A 3-d array holding the kernel images each basis function, with shape
        ``(n, height, width)``.
    spatial
        Iterable of `.fields.BaseField` of length ``basis.shape[0]``, holding
        the spatial variation of each basis kernel.
    center_y
        Center of the basis kernels in the x dimension.  Defaults to
        ``height//2``.
    center_x
        Center of the basis kernels in the x dimension.  Defaults to
        ``width//2``.
    """

    def __init__(
        self,
        basis: np.ndarray,
        spatial: Iterable[Field],
        center_y: int | None = None,
        center_x: int | None = None,
    ):
        self._spatial = tuple(spatial)
        bounds: Bounds | None = None
        for field in self._spatial:
            if field.unit is not None:
                raise ValueError("Kernel spatial fields should not have units.")
            if bounds is None:
                bounds = field.bounds
            else:
                bounds = bounds.intersection(field.bounds)
        if bounds is None:
            raise ValueError("Must have at least one basis function.")
        self._bounds = bounds
        self._basis = basis
        if self._basis.ndim != 3:
            raise ValueError(f"Basis array must be 3-d; shape={self._basis.shape}.")
        if len(self._spatial) != self._basis.shape[0]:
            raise ValueError(
                f"Number of spatial fields ({len(self._spatial)}) "
                f"does not match basis array shape ({self._basis.shape})."
            )
        if center_y is None:
            center_y = self._basis.shape[1] // 2
        if center_x is None:
            center_x = self._basis.shape[2] // 2
        self._kernel_bbox = Box.from_shape(self._basis.shape[1:], start=YX(y=-center_y, x=-center_x))

    @property
    def bounds(self) -> Bounds:
        return self._bounds

    @property
    def kernel_bbox(self) -> Box:
        return self._kernel_bbox

    @property
    def spatial(self) -> Sequence[Field]:
        """The spatial variation of each basis function
        (`~collections.abc.Sequence` [`~.fields.BaseField`]).
        """
        return self._spatial

    @property
    def basis(self) -> np.ndarray:
        """The kernel basis functions, as an array with shape ``(n, h, w)``
        (`numpy.ndarray`).
        """
        return self._basis

    def __len__(self) -> int:
        return len(self._spatial)

    def __iter__(self) -> Iterator[tuple[Image, Field]]:
        for field, array in zip(self._spatial, self._basis, strict=True):
            yield Image(array, bbox=self._kernel_bbox), field

    def compute_kernel_image(self, *, x: int, y: int) -> Image:
        # TODO[DM-54965]: simplify this once BaseField.__call__ behaves more
        # like a real ufunc and can handle scalars directly.
        x_array = np.array([x], dtype=np.float64)
        y_array = np.array([y], dtype=np.float64)
        weights = np.array(
            [spatial_field(x=x_array, y=y_array)[0] for spatial_field in self._spatial],
            dtype=np.float64,
        )
        return Image(np.tensordot(weights, self._basis, axes=(0, 0)), bbox=self._kernel_bbox)

    def serialize(self, archive: OutputArchive[Any]) -> ImageBasisConvolutionKernelSerializationModel:
        """Serialize the kernel to an output archive.

        Parameters
        ----------
        archive
            Archive to write to.
        """
        serialized_basis = archive.add_array(self._basis, name="basis")
        serialized_spatial = [archive.serialize_direct("spatial", f.serialize) for f in self._spatial]
        return ImageBasisConvolutionKernelSerializationModel(
            basis=serialized_basis,
            spatial=serialized_spatial,
            center_y=-self._kernel_bbox.y.min,
            center_x=-self._kernel_bbox.x.min,
        )

    @staticmethod
    def _get_archive_tree_type(
        pointer_type: type[Any],
    ) -> type[ImageBasisConvolutionKernelSerializationModel]:
        """Return the serialization model type for this object for an archive
        type that uses the given pointer type.
        """
        return ImageBasisConvolutionKernelSerializationModel

    @staticmethod
    def from_legacy(legacy_kernel: LegacyLinearCombinationKernel) -> ImageBasisConvolutionKernel:
        """Convert from a legacy `lsst.afw.math.LinearCombinationKernel`.

        Parameters
        ----------
        legacy_kernel
            The kernel to convert.  Must use Chebyshev polynomials for its
            spatial variation and `lsst.afw.math.FixedKernel` objects with a
            consistent shape and center for its basis functions.
        """
        from lsst.afw.math import FixedKernel as LegacyFixedKernel
        from lsst.afw.math import LinearCombinationKernel as LegacyLinearCombinationKernel

        if not isinstance(legacy_kernel, LegacyLinearCombinationKernel):
            raise TypeError(
                f"Cannot convert {type(legacy_kernel).__name__} instance to an ImageBasisConvolutionKernel."
            )
        dimensions = legacy_kernel.getDimensions()
        center = legacy_kernel.getCtr()
        basis = np.zeros((legacy_kernel.getNBasisKernels(), dimensions.y, dimensions.x), dtype=np.float64)
        for n, basis_kernel in enumerate(legacy_kernel.getKernelList()):
            if basis_kernel.getDimensions() != dimensions:
                raise ValueError("Cannot convert LinearCombinationKernel with different-size basis kernels.")
            if basis_kernel.getCtr() != center:
                raise ValueError(
                    "Cannot convert LinearCombinationKernel with differently-centered basis kernels."
                )
            if not isinstance(basis_kernel, LegacyFixedKernel):
                raise ValueError("Cannot convert LinearCombinationKernel with non-fixed basis kernels.")
            legacy_image_view = Image(basis[n, :, :], dtype=np.float64).to_legacy()
            basis_kernel.computeImage(legacy_image_view, doNormalize=False)
        spatial = [ChebyshevField.from_legacy_function2(f) for f in legacy_kernel.getSpatialFunctionList()]
        return ImageBasisConvolutionKernel(basis=basis, spatial=spatial, center_y=center.y, center_x=center.x)

    def to_legacy(self) -> LegacyLinearCombinationKernel:
        """Convert to a legacy `lsst.afw.math.LinearCombinationKernel`.

        This only works if all spatial variation is handled by
        `lsst.images.ChebyshevField`.
        """
        from lsst.afw.math import FixedKernel as LegacyFixedKernel
        from lsst.afw.math import LinearCombinationKernel as LegacyLinearCombinationKernel
        from lsst.geom import Point2I as LegacyPoint2I

        basis_kernels = []
        spatial_functions = []
        legacy_center = LegacyPoint2I(-self._kernel_bbox.x.min, -self._kernel_bbox.y.min)
        for image, field in self:
            legacy_image = image.to_legacy()
            legacy_image.setXY0(LegacyPoint2I())
            basis_kernel = LegacyFixedKernel(legacy_image)
            basis_kernel.setCtr(legacy_center)
            basis_kernels.append(basis_kernel)
            if not isinstance(field, ChebyshevField):
                raise ValueError("Only Chebyshev spatial variation can be converted.")
            spatial_functions.append(field.to_legacy_function2())
        result = LegacyLinearCombinationKernel(basis_kernels, spatial_functions)
        result.setCtr(legacy_center)
        return result


class ImageBasisConvolutionKernelSerializationModel(ArchiveTree):
    """The serialization model for `ImageBasisConvolutionKernel`."""

    SCHEMA_NAME: ClassVar[str] = "image_basis_convolution_kernel"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = ImageBasisConvolutionKernel

    basis: ArrayReferenceModel | InlineArrayModel = pydantic.Field(
        description="The basis images, with shape (n, h, w)."
    )
    spatial: list[FieldSerializationModel] = pydantic.Field(
        description="The spatial variation of each basis function."
    )
    center_y: int = pydantic.Field(description="Center row of the kernel in the basis images.")
    center_x: int = pydantic.Field(description="Center column of the kernel in the basis images.")

    kernel_type: Literal["IMAGE_BASIS"] = "IMAGE_BASIS"

    def deserialize(self, archive: InputArchive[Any], **kwargs: Any) -> ImageBasisConvolutionKernel:
        if kwargs:
            raise InvalidParameterError(f"Unrecognized parameters for ChebyshevField: {set(kwargs.keys())}.")
        basis = archive.get_array(self.basis)
        spatial = [f.deserialize(archive) for f in self.spatial]
        return ImageBasisConvolutionKernel(basis, spatial, center_y=self.center_y, center_x=self.center_x)
