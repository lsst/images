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

__all__ = ("InputArchive",)

from abc import ABC, abstractmethod
from collections.abc import Callable

import astropy.io.fits
import astropy.table
import astropy.units
import numpy as np
import pydantic

from .._coordinate_transform import CoordinateTransform
from .._geom import Box
from .._image import Image
from .._mask import Mask
from ._common import OpaqueArchiveMetadata, no_header_updates
from ._image import ImageModel
from ._mask import MaskModel
from ._tables import TableModel


class InputArchive[P: pydantic.BaseModel](ABC):
    """Abstract interface for reading from a file format.

    Notes
    -----
    An input archive instance is assumed to be paired with a Pydantic model
    that represents a JSON tree, with the archive used to deserialize data that
    is not native JSON from data that is (which may just be a reference to
    binary data stored elsewhere in the file).  The archive doesn't actually
    hold that model instance because we'd prefer to avoid making the input
    archive generic over the model type.  It is expected that most concrete
    archive implementations will provide a method to load the paired model from
    a file, but this is not part of the base class interface.
    """

    @abstractmethod
    def get_coordinate_transform(self, from_frame: str, to_frame: str = "sky") -> CoordinateTransform:
        """Return the coordinate transform that maps the two given frames.

        Parameters
        ----------
        from_frame
            Frame for coordinates passed into the transform.
        to_frame
            Frame for coordinates returned by the transform.

        Returns
        -------
        transform
            Coordinate transform

        Notes
        -----
        Implementations are expected to cache returned values, and may need to
        assemble composite transforms from serialized individual transforms as
        well, depending on how composite transforms are saved by the
        corresponding output archive.
        """
        raise NotImplementedError()

    # TODO: we probably need a way to get a coordinate transform from a P model
    # pointer, too.

    @abstractmethod
    def deserialize_pointer[U: pydantic.BaseModel, V](
        self, pointer: P, model_type: type[U], deserializer: Callable[[U, InputArchive[P]], V]
    ) -> V:
        """Deserialize an object that was saved by
        `OutputArchive.serialize_pointer`.

        Parameters
        ----------
        pointer
            JSON Pointer model to dereference.
        model_type
            Pydantic model type that the pointer should dereference to.
        deserializer
            Callable that takes an instance of ``model_type`` and an input
            archive, and returns the deserialized object.

        Returns
        -------
        deserialized
            The deserialized object.

        Notes
        -----
        Implementations are required to remember previously-deserialized
        objects and return them when the same pointer is passed in multiple
        times.

        There is no ``deserialize_direct`` (to pair with
        `OutputArchive.serialize_direct`) because the caller can just call a
        deserializer function directly on a sub-model of its Pydantic tree.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_image(
        self,
        ref: ImageModel,
        *,
        bbox: Box | None = None,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> Image:
        """Load an image from the archive.

        Parameters
        ----------
        ref
            A Pydantic model that references the image.
        bbox, optional
            A bounding box that specifies a subset of the original image to
            read.
        strip_header, optional
            A callable that strips out any FITS header cards added by the
            ``update_header`` argument in the corresponding call to
            `FitsOutputArchive.add_image`.

        Returns
        -------
        image
            The loaded image.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_mask(
        self,
        ref: MaskModel,
        *,
        bbox: Box | None = None,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> Mask:
        """Load a mask from the archive.

        Parameters
        ----------
        ref
            A Pydantic model that references the mask.
        bbox, optional
            A bounding box that specifies a subset of the original mask to
            read.
        strip_header, optional
            A callable that strips out any FITS header cards added by the
            ``update_header`` argument in the corresponding call to
            `FitsOutputArchive.add_mask`.

        Returns
        -------
        mask
            The loaded mask.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_table(
        self,
        ref: TableModel,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> astropy.table.Table:
        """Load a table from the archive.

        Parameters
        ----------
        ref
            A Pydantic model that references the table.
        strip_header, optional
            A callable that strips out any FITS header cards added by the
            ``update_header`` argument in the corresponding call to
            `FitsOutputArchive.add_table`.

        Returns
        -------
        table
            The loaded table.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_structured_array(
        self,
        ref: TableModel,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> np.ndarray:
        """Load a table from the archive as a structured array.

        Parameters
        ----------
        ref
            A Pydantic model that references the table.
        strip_header, optional
            A callable that strips out any FITS header cards added by the
            ``update_header`` argument in the corresponding call to
            `FitsOutputArchive.add_table`.

        Returns
        -------
        array
            The loaded table as a structured array.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_opaque_metadata(self) -> OpaqueArchiveMetadata:
        """Return opaque metadata loaded from the file that should be saved if
        another version of the object is saved to the same file format.

        Returns
        -------
        metadata
            Opaque metadata specific to this archive type that should be
            round-tripped if it is saved in the same format.
        """
        raise NotImplementedError()
