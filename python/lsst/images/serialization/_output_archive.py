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
    "NestedOutputArchive",
    "OutputArchive",
)

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Mapping
from typing import TypeVar

import astropy.io.fits
import astropy.table
import astropy.units
import numpy as np
import pydantic

from .._image import Image
from .._mask import Mask
from ._common import no_header_updates
from ._image import ImageModel
from ._mask import MaskModel
from ._tables import TableModel

# This pre-python-3.12 declaration is needed by Sphinx (probably the
# autodoc-typehints plugin.
P = TypeVar("P", bound=pydantic.BaseModel)


class OutputArchive[P](ABC):
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
            Callable that takes an `~lsst.serialization.OutputArchive` and
            returns a Pydantic model.  This will be passed a new
            `~lsst.serialization.OutputArchive` that automatically prepends
            ``{name}/`` (and any root path added by this archive) to names
            passed to it, so the ``serializer`` does not need to know where it
            appears in the overall tree.

        Returns
        -------
        T
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
            Callable that takes an `~lsst.serialization.OutputArchive` and
            returns a Pydantic model.  This will be passed a new
            `~lsst.serialization.OutputArchive` that automatically prepends
            ``{name}/`` (and any root path added by this archive) to names
            passed to it, so the ``serializer`` does not need to know where it
            appears in the overall tree.
        key
            A unique identifier for the in-memory object the serializer saves,
            e.g. a call to the built-in `id` function.

        Returns
        -------
        T | P
            Either the result of the call to the serializer, or a Pydantic
            model that can be considered a reference to it and added to a
            larger model in its place.
        """
        # Since Pydantic doesn't provide us a good way to "dereference" a JSON
        # Pointer (i.e. traversing the tree to extract the original model), it
        # is probably easier to implement an `InputArchive` for the case where
        # the `~lsst.serialization.OutputArchive` opts to stuff all pointer
        # serializations into a standard location outside the user-controlled
        # Pydantic model tree, and always returned a JSON pointer to that
        # standard location from this function.
        raise NotImplementedError()

    @abstractmethod
    def add_image(
        self,
        name: str,
        image: Image,
        *,
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
        update_header
            A callback that will be given the FITS header for the HDU
            containing this image in order to add keys to it.  This call back
            may be provided but will not be called if the output format is not
            FITS.

        Returns
        -------
        ImageModel
            A Pydantic model that represents the image, including its bounding
            box, units (if any), and array data (via a reference to binary data
            stored elsewhere by the archive).
        """
        raise NotImplementedError()

    @abstractmethod
    def add_mask(
        self,
        name: str,
        mask: Mask,
        *,
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
        update_header
            A callback that will be given the FITS header for the HDU
            containing this mask in order to add keys to it.  This call back
            may be provided but will not be called if the output format is not
            FITS.

        Returns
        -------
        MaskModel
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
        update_header
            A callback that will be given the FITS header for the HDU
            containing this table in order to add keys to it.  This call back
            may be provided but will not be called if the output format is not
            FITS.

        Returns
        -------
        TableModel
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
        units
            A mapping of units for columns.  Need not be complete.
        descriptions
            A mapping of descriptions for columns.  Need not be complete.
        update_header
            A callback that will be given the FITS header for the HDU
            containing this table in order to add keys to it.  This call back
            may be provided but will not be called if the output format is not
            FITS.

        Returns
        -------
        TableModel
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
    `~lsst.serialization.OutputArchive.serialize_direct` and
    `~lsst.serialization.OutputArchive.serialize_pointer` implementations.

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
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> ImageModel:
        return self._parent.add_image(self._join_path(name), image, update_header=update_header)

    def add_mask(
        self,
        name: str,
        mask: Mask,
        *,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> MaskModel:
        return self._parent.add_mask(self._join_path(name), mask, update_header=update_header)

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
