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

"""Abstract interfaces and helper classes for `OutputArchive` and
`InputArchive`, which abstract over different file formats.

These archive interfaces are designed with two specific implementations in
mind:

- FITS augmented with a JSON block in a special BINTABLE HDU (see the `fits`
  module for details), inspired by the now-defunct ASDF-in-FITS concept.

- ASDF (just hypothetical for now).

The base classes make some concessions to both FITS and ASDF in order to make
the representations in those formats conform to their respective expectations.

For ASDF, this is simple: we use ASDF schemas whenever possible to represent
primitive types, from units and times to multidimensional arrays. While the
archive interfaces use Pydantic, which maps to JSON, not YAML, the expectation
is that by encoding YAML tag information in the JSON Schema (which Pydantic
allows us to customize), it should be straightforward for an ASDF archive
implementation to have Pydantic dump to a Python `dict` (etc) tree, and then
convert that to tagged YAML by walking the tree along with its schema.

For FITS, the challenge is primarily to populate standard FITS header cards
when writing, despite the fact that FITS headers are generally too limiting to
be our preferred way of round-tripping any information.  To do this, the
archive interfaces accept `update_header` and `strip_header` callback arguments
that are only called by FITS implementations.

An implementation that writes HDF5 while embedding JSON should also be possible
with these interfaces, but is not something we've designed around. A more
natural HDF5 implementation might be possible by translating the JSON tree into
a binary HDF5 hierarchy as well, but this would be considerably more effort at
best.
"""

from __future__ import annotations

__all__ = (
    "InputArchive",
    "OpaqueArchiveMetadata",
    "OutputArchive",
)

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Iterable, Mapping
from typing import TYPE_CHECKING, Protocol, Self

import astropy.table
import astropy.units
import numpy as np
import pydantic

from ._coordinate_transform import CoordinateTransform
from ._geom import Box
from ._image import Image, ImageModel
from ._mask import Mask, MaskModel
from .tables import TableModel

if TYPE_CHECKING:
    import astropy.io.fits


class OpaqueArchiveMetadata(Protocol):
    """Interface for opaque archive metadata.

    In addition to implementing the methods defined here, all implementations
    must be pickleable.
    """

    def copy(self) -> Self | None:
        """Copy, reference, or discard metadata when its holding object is
        copied.
        """
        ...

    def subset(self, bbox: Box) -> Self | None:
        """Copy, reference, or discard metadata when a subset of its its
        holding object is extracted.
        """
        ...


def no_header_updates(header: astropy.io.fits.Header) -> None:
    """Do not make any modifications to the given FITS header."""


class OutputArchive[P: pydantic.BaseModel](ABC):
    """Abstract interface for writing to a file format.

    Notes
    -----
    An output archive instance is assumed to be paired with a Pydantic model
    that represents a JSON tree, with the archive used to serialize data that
    is not natively JSON into data that is (which may just be a reference to
    binary data stored elsewhere in the file).  The archive doesn't actually
    hold that model instance because we don't want to assume it can be built
    via default-initialization and assignment, and because we'd prefer to avoid
    making the output archive generic over the model type.  It is expected that
    most concrete archive implementations will accept the paired model in some
    sort of finalization method in order to write it into the file, but this is
    not part of the base class interface.
    """

    @abstractmethod
    def add_coordinate_transform(
        self, transform: CoordinateTransform, from_frame: str, to_frame: str = "sky"
    ) -> P:
        """Add a coordinate transform between two frames to the archive.

        Parameters
        ----------
        transform
            Mapping between the frames.
        from_frame
            Frame for the input coordinates to the transform.
        to_frame
            Frame for the output coordinates returned by the transform.

        Returns
        -------
        pointer
            JSON Pointer to the serialized coordinate transform.
        """
        # This interface assumes coordinate transforms are stored in in
        # JSON/YAML somewhere outside the user-controlled tree (i.e. a
        # centralized sibling tree controlled by the archive).  We can adjust
        # it as needed for consistency with ASDF/gwcs conventions.
        raise NotImplementedError()

    @abstractmethod
    def serialize_direct[T: pydantic.BaseModel](
        self, name: str, serializer: Callable[[OutputArchive], T]
    ) -> T:
        """Use a serializer function to save a nested object.

        Parameters
        ----------
        name
            Attribute of the paired Pydantic model that will be assigned the
            result of this call.  If it will not be assigned to a direct
            attribute, it may be a JSON Pointer path (relative to the paired
            Pydantic model) to the location where it will be added.
        serializer
            Callable that takes an `OutputArchive` and returns a Pydantic
            model.  This will be passed a new `OutputArchive` that
            automatically prepends ``{name}/`` (and any root path added by this
            archive) to names passed to it, so the ``serializer`` does not need
            to know where it appears in the overall tree.

        Returns
        -------
        serialized
            Result of the call to the serializer.
        """
        raise NotImplementedError()

    @abstractmethod
    def serialize_pointer[T: pydantic.BaseModel](
        self, name: str, serializer: Callable[[OutputArchive], T], key: Hashable
    ) -> T | P:
        """Use a serializer function to save a nested object that may be
        referenced in multiple locations in the same archive.

        Parameters
        ----------
        name
            Attribute of the paired Pydantic model that will be assigned the
            result of this call.  If it will not be assigned to a direct
            attribute, it may be a JSON Pointer path (relative to the paired
            Pydantic model) to the location where it will be added.
        serializer
            Callable that takes an `OutputArchive` and returns a Pydantic
            model.  This will be passed a new `OutputArchive` that
            automatically prepends ``{name}/`` (and any root path added by this
            archive) to names passed to it, so the ``serializer`` does not need
            to know where it appears in the overall tree.
        key
            A unique identifier for the in-memory object the serializer saves,
            e.g. a call to the built-in `id` function.

        Returns
        -------
        serialized_or_pointer
            Either the result of the call to the serializer, or a Pydantic
            model that can be considered a reference to it and added to a
            larger model in its place.
        """
        # Since Pydantic doesn't provide us a good way to "dereference" a JSON
        # Pointer (i.e. traversing the tree to extract the original model), it
        # is probably easier to implement an `InputArchive` for the case where
        # the `OutputArchive` opts to stuff all pointer serializations into a
        # standard location outside the user-controlled Pydantic model tree,
        # and always returned a JSON pointer to that standard location from
        # this function.
        raise NotImplementedError()

    @abstractmethod
    def add_image(
        self,
        name: str,
        image: Image,
        *,
        pixel_frame: str | None = None,
        wcs_frames: Iterable[str] = (),
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> ImageModel:
        """Add an image to the archive.

        Parameters
        ----------
        name
            Attribute of the paired Pydantic model that will be assigned the
            result of this call.  If it will not be assigned to a direct
            attribute, it may be a JSON Pointer path (relative to the paired
            Pydantic model) to the location where it will be added.
        image
            Image to save.
        pixel_frame
            String identifying the frame of this image's pixels.
        wcs_frames, optional
            Strings identifying additional frames whose transforms to
            ``pixel_frame`` should be saved along with the image (for archive
            formats such as FITS that associate coordinate transforms with
            images).  The special ``sky`` frame is included implicitly.
        update_header, optional
            A callback that will be given the FITS header for the HDU
            containing this image in order to add keys to it.  This call back
            may be provided but will not be called if the output format is not
            FITS.

        Returns
        -------
        image_model
            A Pydantic model that represents the image, including its bounding
            box, units (if any), and array data (via a reference to binary data
            stored elsewhere by the archive).

        Notes
        -----
        Frames and coordinate transforms may be ignored by archive
        implementations that do not have a standard way of associating them
        with individual images.  Images should generally be associated with
        frames and coordinate transforms manually in the paired Pydantic model
        as well, in a user-defined way.
        """
        # TODO: this method needs some way for the user to export FITS header
        # keys, which would be ignored for non-FITS archives; the goal is to
        # support writing standard FITS headers that belong with an image HDU
        # (rather than the primary HDU) while still letting the paired Pydantic
        # tree hold the same information (and for that to be all we read).
        raise NotImplementedError()

    @abstractmethod
    def add_mask(
        self,
        name: str,
        mask: Mask,
        *,
        pixel_frame: str | None = None,
        wcs_frames: Iterable[str] = (),
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> MaskModel:
        """Add a mask to the archive.

        Parameters
        ----------
        name
            Attribute of the paired Pydantic model that will be assigned the
            result of this call.  If it will not be assigned to a direct
            attribute, it may be a JSON Pointer path (relative to the paired
            Pydantic model) to the location where it will be added.
        mask
            Mask to save.
        pixel_frame
            String identifying the frame of this image's pixels.
        wcs_frames, optional
            Strings identifying additional frames whose transforms to
            ``pixel_frame`` should be saved along with the image (for archive
            formats such as FITS that associate coordinate transforms with
            images).  The special ``sky`` frame is included implicitly.
        update_header, optional
            A callback that will be given the FITS header for the HDU
            containing this mask in order to add keys to it.  This call back
            may be provided but will not be called if the output format is not
            FITS.

        Returns
        -------
        mask_model
            A Pydantic model that represents the mask, including its bounding
            box, schema, and array data (via a reference to binary data stored
            elsewhere by the archive).

        Notes
        -----
        Frames and coordinate transforms may be ignored by archive
        implementations that do not have a standard way of associating them
        with individual images.  Images should generally be associated with
        frames and coordinate transforms manually in the paired Pydantic model
        as well, in a user-defined way.
        """
        # TODO: this method needs FITS header export support (see comment on
        # add_image).
        # TODO: we should have a way of saving a mask schema once per file even
        # if there are multiple masks that use it.
        raise NotImplementedError()

    @abstractmethod
    def add_table(
        self,
        name: str,
        table: astropy.table.Table,
        *,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> TableModel:
        """Add a table to the archive.

        Parameters
        ----------
        name
            Attribute of the paired Pydantic model that will be assigned the
            result of this call.  If it will not be assigned to a direct
            attribute, it may be a JSON Pointer path (relative to the paired
            Pydantic model) to the location where it will be added.
        table
            Table to save.
        update_header, optional
            A callback that will be given the FITS header for the HDU
            containing this table in order to add keys to it.  This call back
            may be provided but will not be called if the output format is not
            FITS.

        Returns
        -------
        table_model
            A Pydantic model that represents the table.  Column definitions
            are included directly in the model while the actual data is
            stored elsewhere and referenced by the model.
        """
        # TODO: ASDF has schemas for tables and columns that we should probably
        # adopt [a subset of].  While that can reference external per-column
        # data (which would Just Work for a true ASDF archive), I'm not sure
        # there's a way to reference external data in a FITS binary table
        # column.  We could of course invent one, and since ASDF-in-FITS isn't
        # even referenced on the ASDF standard page our existing approach for
        # referencing FITS data in an image extension may be something only
        # we'll be using, too.
        raise NotImplementedError()

    @abstractmethod
    def add_structured_array(
        self,
        name: str,
        array: np.ndarray,
        *,
        units: Mapping[str, astropy.units.Unit] | None = None,
        descriptions: Mapping[str, str] | None = None,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> TableModel:
        """Add a table to the archive.

        Parameters
        ----------
        name
            Attribute of the paired Pydantic model that will be assigned the
            result of this call.  If it will not be assigned to a direct
            attribute, it may be a JSON Pointer path (relative to the paired
            Pydantic model) to the location where it will be added.
        array
            A structured numpy array.
        units, optional
            A mapping of units for columns.  Need not be complete.
        descriptions, optional
            A mapping of descriptions for columns.  Need not be complete.
        update_header, optional
            A callback that will be given the FITS header for the HDU
            containing this table in order to add keys to it.  This call back
            may be provided but will not be called if the output format is not
            FITS.

        Returns
        -------
        table_model
            A Pydantic model that represents the table.  Column definitions
            are included directly in the model while the actual data is
            stored elsewhere and referenced by the model.
        """
        # TODO: ASDF has schemas for tables and columns that we should probably
        # adopt [a subset of].  While that can reference external per-column
        # data (which would Just Work for a true ASDF archive), I'm not sure
        # there's a way to reference external data in a FITS binary table
        # column.  We could of course invent one, and since ASDF-in-FITS isn't
        # even referenced on the ASDF standard page our existing approach for
        # referencing FITS data in an image extension may be something only
        # we'll be using, too.
        raise NotImplementedError()


class NestedOutputArchive[P: pydantic.BaseModel](OutputArchive[P]):
    """A proxy output archive that joins a root path into all names before
    delegating back to its parent archive.

    This is intended to be used in the implementation of most
    `OutputArchive.serialize_direct` and `OutputArchive.serialize_pointer`
    implementations.

    Parameters
    ----------
    root
        Root of all JSON Pointer paths.  Should include a leading slash (as we
        always use absolute JSON Pointers) but no trailing slash.
    parent
        Parent output archive to delegate to.
    """

    def __init__(self, root: str, parent: OutputArchive):
        self._root = root
        self._parent = parent

    def add_coordinate_transform(
        self, transform: CoordinateTransform, from_frame: str, to_frame: str = "sky"
    ) -> P:
        return self._parent.add_coordinate_transform(transform, from_frame, to_frame)

    def serialize_direct[T: pydantic.BaseModel](
        self, name: str, serializer: Callable[[OutputArchive[P]], T]
    ) -> T:
        return self._parent.serialize_direct(self._join_path(name), serializer)

    def serialize_pointer[T: pydantic.BaseModel](
        self, name: str, serializer: Callable[[OutputArchive[P]], T], key: Hashable
    ) -> T | P:
        return self._parent.serialize_pointer(self._join_path(name), serializer, key)

    def add_image(
        self,
        name: str,
        image: Image,
        *,
        pixel_frame: str | None = None,
        wcs_frames: Iterable[str] = (),
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> ImageModel:
        return self._parent.add_image(
            self._join_path(name), image, pixel_frame=pixel_frame, update_header=update_header
        )

    def add_mask(
        self,
        name: str,
        mask: Mask,
        *,
        pixel_frame: str | None = None,
        wcs_frames: Iterable[str] = (),
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> MaskModel:
        return self._parent.add_mask(
            self._join_path(name), mask, pixel_frame=pixel_frame, update_header=update_header
        )

    def add_table(
        self,
        name: str,
        table: astropy.table.Table,
        *,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> TableModel:
        return self._parent.add_table(self._join_path(name), table, update_header=update_header)

    def add_structured_array(
        self,
        name: str,
        array: np.ndarray,
        *,
        units: Mapping[str, astropy.units.Unit] | None = None,
        descriptions: Mapping[str, str] | None = None,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> TableModel:
        return self._parent.add_structured_array(
            self._join_path(name), array, units=units, descriptions=descriptions, update_header=update_header
        )

    def _join_path(self, name: str) -> str:
        return f"{self._root}/{name}"


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
