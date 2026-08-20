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
    "Asinh",
    "AsinhZScale",
    "BaseDisplay",
    "BaseGrid",
    "EllipseColumns",
    "HTMGrid",
    "HealpixGrid",
    "LinearMinMax",
    "LinearZScale",
    "Moments",
    "SemiMajorMinor",
    "Shear",
    "SkyGrid",
    "Stretch",
)

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Hashable, Iterable, Mapping, Sequence
from types import EllipsisType
from typing import TYPE_CHECKING, Literal, final

import astropy.table
import astropy.units
import numpy as np

from ._color_image import ColorImage
from ._geom import XY, Box
from ._image import Image
from ._mask import MaskSchema
from ._masked_image import MaskedImage
from ._transforms import Projection

if TYPE_CHECKING:
    from lsst.daf.butler import Butler, DatasetRef


class BaseDisplay(ABC):
    """An abstract base class for display implementations.

    Parameters
    ----------
    butler
        An optional butler client that can be used to display images and
        overlay catalogs directly from `lsst.daf.butler.DatasetRef` instances,
        which may be faster or allow additional information to be displayed in
        certain cases.

    Notes
    -----
    Implementations are encouraged to add new methods and keyword arguments
    in addition to supporting those defined here, especially with respect
    to callbacks and interactivity.

    Implementations are also permitted to require an image to be provided at
    construction if they don't want to support a "no image yet" state, but
    should always allow that image to be replaced later.
    """

    def __init__(self, butler: Butler | None = None):
        self.butler = butler

    @property
    def autoshow(self) -> bool:
        """Whether this display updates automatically (`True`) or needs the
        `show` method to be called explicitly (`bool`).
        """
        # Implementations can override this value or make it controllable by
        # the user by adding a setter.
        return False

    @property
    @abstractmethod
    def mask_colors(self) -> Sequence[str]:
        """A sequence of HTML colors that will be used for mask planes by
        default (`~collections.abc.Sequeunce` [`str`]).

        These colors are guaranteed to be supported as explicit mask plane
        colors by this display implementation, but they may not be the only
        supported colors for this implementatation.  Implementations are also
        permitted to add a setter for this property.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def region_colors(self) -> Sequence[str]:
        """A sequence of HTML colors that will be used for regions by
        default (`~collections.abc.Sequeunce` [`str`]).

        These colors are guaranteed to be supported as explicit regions
        colors by this display implementation, but they may not be the only
        supported colors for this implementatation.  Implementations are also
        permitted to add a setter for this property.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def region_markers(self) -> Sequence[str]:
        """A sequence of markers that will be used for regions by default
        (`~collections.abc.Sequeunce` [`str`]).

        Implementations should aim to support as many as possible of "+", "x",
        and "o", and to use exactly those strings for markers with those
        shapes.

        These markers are guaranteed to be supported as explicit regions
        markers by this display implementation, but they may not be the only
        supported colors for this implementatation.  Implementations are also
        permitted to add a setter for this property.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def image(self) -> Image | MaskedImage | ColorImage:
        """The image currently associated with the display."""
        # This exists so if the user passes a DatasetRef to set_image, and
        # that does a client-side butler.get, the result goes somewhere the
        # user can get it from to avoid extra I/O.   In cases where the I/O
        # doesn't happen in the client this property would call butler.get the
        # first time it is accessed.
        #
        # We might want this to be annotated as returning `Any` so users don't
        # always have to cast what they get back to appease type-checkers.
        raise NotImplementedError()

    @property
    @abstractmethod
    def table(self) -> astropy.table.Table | None:
        """The table currently associated with the display."""
        # Rationale for this is the same as for the `image` property.
        raise NotImplementedError()

    @abstractmethod
    def reset(self) -> None:
        """Reset this display to its original post-construction state and
        remove all images and overlays.
        """
        raise NotImplementedError()

    def show(self) -> None:
        """Update the display to reflect all changes since the last `show`
        call.
        """
        pass

    @abstractmethod
    def set_image(
        self,
        image: Image | MaskedImage | ColorImage | DatasetRef,
        *,
        stretch: Stretch | None = None,
        mask_planes: Mapping[str | EllipsisType, str | None | EllipsisType] | None | EllipsisType = ...,
    ) -> None:
        """Set the image for the display.

        Parameters
        ----------
        image
            The generalized image to display. Implementations may also display
            additional information for `.MaskedImage` subclasses like
            `.VisitImage` or `.cells.CellCoadd`, but are not required to.
            If the image has a `.Projection`, this will be used to show sky
            coordinates and will enable table and marker displays in sky
            coordinates.  If a `.Projection` is attached but does not have a
            FITS approximation that is needed by the implementation, a warning
            should be emitted.

            If this is a `lsst.daf.butler.DatasetRef`, the display must have
            been constructed with a `~lsst.daf.butler.Butler` instance.
        stretch
            How to stretch the image for display. Must be `None` when
            ``image`` is a ``.ColorImage`` instance, as those are assumed to
            already have display-ready RGB values.
        mask_planes
            Whether and how to display a mask overlay:

            - If ``...``, colors for all mask planes will be selected
              automatically.
            - If `None`, there will be no mask overlay.
            - If a mapping, the keys are mask plane names and the values are
              valid HTML color strings (or ``...`` for automatic colors).
              If ``...`` is also included as a key, all mask planes that are
              not explicitly included in the mapping will use that value, with
              keys mapped to `None` skipped.  When ``...`` is not included as
              a key or is mapped to `None`, `None` values are the same as keys
              that were not included in the mapping.

            Must be ``...`` or `None` when the given image does not have a
            mask.
        """
        raise NotImplementedError()

    @abstractmethod
    def set_table(self, table: astropy.table.Table | DatasetRef | None = None) -> None:
        """Set the table associated with the display.

        Parameters
        ----------
        table
            The table to display.  For display implementations that don't
            support browsing a table, this may just make the table's columns
            available for use in `dot` calls.  If `None`, the table and all
            associated `dot` markers are removed (in the next call to `show`,
            if `autoshow` is `False`).

        Notes
        -----
        The base class interfaces assumes a display can be associated with at
        most one table.  Implementations that can have multiple tables should
        map these to different display instances (which may share state).
        """
        # TODO: we should duplicate all of the `dot` arguments here and call
        # `dot` automatically for convenience, but we should work out a way
        # for that to happen in the base class so it doesn't have to be
        # duplicated in every implementatation.
        raise NotImplementedError()

    @abstractmethod
    def dot[K: Hashable](
        self,
        *,
        key: K | None = None,
        x: str | np.ndarray | None = None,
        y: str | np.ndarray | None = None,
        ra: str | np.ndarray | astropy.units.Quantity | None = None,
        dec: str | np.ndarray | astropy.units.Quantity | None = None,
        # TODO: should there be another argument for specifying a compound
        # 'astropy.coordinate.SkyCoord' column?  I think that's a thing.
        color: str | EllipsisType = ...,
        marker: str | EllipsisType = ...,
        # TODO: how should we control marker size? Pixel/sky units? Display
        # units? Either?  Can it be an array?
        ellipse: EllipseColumns | None = None,
        id: str | np.ndarray | None = None,
        annotate: bool = False,
        where: str | np.ndarray | slice | EllipsisType = ...,
    ) -> K:
        """Draw a marker or region overlay on top of the image.

        Parameters
        ----------
        key
            A name or other identifier associated with this set of regions.
            If not provided, an opaque identifier is created and returned.
        x
            Center x coordinates in pixels, as a column name or array.
        y
            Center y coordinates in pixels, as a column name or array.
        ra
            Center right ascension coordinates in degrees, unless specified
            otherwise by table or `astropy.unit.Quantity` units, as a column
            name or array.  Only supported if the display's image has a
            projection.
        dec
            Center declination coordinates in degrees, unless specified
            otherwise by table or `astropy.unit.Quantity` units, as a column
            name or array.  Only supported if the display's image has a
            projection.
        color
            The color of the regions, as an HTML color.  ``...`` selects a
            color automatically from `region_colors`.
        marker
            The kind of marker to display.  ``...`` selects a marker
            automatically from `region_markers`.
        ellipse
            A triplet of columns that can be used to define an allipse.  If
            provided and ``marker`` is ``...``, no central marker will be
            added.
        id
            Unique identifiers (strings or integers) for each row, as a column
            name or array.
        annotate
            If `True`, ``id`` must be provided and those labels should be
            displayed along with each marker.
        where
            A selection to apply to all columns extracted by name from the
            table.  This may be:

            - a column name (a boolean mask that selects `True` values);
            - a column name with ``!`` prefixed (selects `False` values);
            - a boolean mask array of the same size as the table;
            - an integer array of indices into the table;
            - a slice;
            - ``...`` (for everything).

            This is *not* applied to arrays or quantity arrays passed in
            directly, but it may be provided even when all columsn are passed
            as arrays or quantity arrays, to link the selected rows with the
            markers in a GUI (for implementations that support this).

        Notes
        -----
        When any column argument is provided as a `str` column name, a table
        must have already been associated with the display and the resulting
        marker set will be associated with that table and deleted if the table
        is removed or dropped.  Implementations may also link the markers to
        those columns or the rows identified by ``where`` in other ways.
        """
        raise NotImplementedError()

    @abstractmethod
    def grid[K: Hashable](self, grid: BaseGrid, key: K | None = None) -> Hashable:
        """Add a grid overlay.

        Parameters
        ----------
        key
            A name or other identifier associated with this set of regions.
            If not provided, an opaque identifier is created and returned.

        Returns
        -------
        `~collections.abc.Hashable`
            A unique identifier for this overlay.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_selection(self) -> tuple[Hashable, int | str] | None:
        """Return the identifiers for the selected region in the GUI, if there
        is one.

        Returns
        -------
        `~collections.abc.Hashable`
            Key associated with the `dot` or `grid` overlay that is selected.
        `int` | `str`
            The table-row or grid-cell ID.
        """
        raise NotImplementedError()

    @abstractmethod
    def clear(self, key: Hashable | Iterable[Hashable] | EllipsisType) -> None:
        """Delete the overlay with the given key or keys."""
        raise NotImplementedError()

    def resolve_mask_overlay(
        self,
        schema: MaskSchema,
        planes: Mapping[str | EllipsisType, str | None | EllipsisType] | None | EllipsisType = ...,
    ) -> dict[str, str]:
        """Resolve a mask plane display specification to a particular mask
        schema and this display's `mask_colors`.

        Parameters
        ----------
        schema
            All mask planes to consider.
        planes
            Specification for how to display mask planes; see the
            ``mask_planes`` argument to `set_image`.

        Returns
        -------
        `dict` [`str`, `str`]
            A mapping from mask plane name to color that matches the
            specification.
        """
        raise NotImplementedError("TODO: implement in base class.")


# ---------------------------------------------------------------------------
# I'm not sure this is all of the important stretch algorithms, and even less
# sure I got their parameters right; the important thing for is that I'd like
# to enumerate them via classes.
#
# We may also want to have a way to construct these (e.g. static methods) from
# a display instance, just to cut down on how much users have to import.


@final
@dataclasses.dataclass
class LinearMinMax:
    """Linear stretch between given or from-data min/max bounds."""

    min: float | None = None
    max: float | None = None


@final
@dataclasses.dataclass
class LinearZScale:
    """Linear stretch with zscale bounds."""

    n_samples: int
    contrast: float


@final
@dataclasses.dataclass
class Asinh:
    """Hyperbolic arcsin stretch."""

    min: float
    range: float
    q: float


@final
@dataclasses.dataclass
class AsinhZScale:
    """Hyperbolic arcsin zscale stretch."""

    q: float
    pedestal: float


type Stretch = LinearMinMax | LinearZScale | Asinh | AsinhZScale


# --------------------------------------------------------------------------
# A dataclass union enumeration of the ways to specify an ellipse via table
# columns or arrays.


@final
class Moments:
    """An ellipse defined by second moments."""

    xx: str | np.ndarray | astropy.units.Quantity
    yy: str | np.ndarray | astropy.units.Quantity
    xy: str | np.ndarray | astropy.units.Quantity
    scale: float = 1.0
    frame: Literal["sky", "pixels", None] = None


@final
class SemiMajorMinor:
    """An ellipse defined by semimajor axis, semiminor axis, and position
    angle.
    """

    a: str | np.ndarray | astropy.units.Quantity
    b: str | np.ndarray | astropy.units.Quantity
    theta: str | np.ndarray | astropy.units.Quantity
    scale: float = 1.0
    frame: Literal["sky", "pixels", None] = None


@final
class Shear:
    """An ellipse defined by complex (reduced shear) ellipticity and the
    trace of the mements (xx + yy).
    """

    e1: str | np.ndarray | astropy.units.Quantity
    e2: str | np.ndarray | astropy.units.Quantity
    t: str | np.ndarray | astropy.units.Quantity
    scale: float = 1.0
    frame: Literal["sky", "pixels", None] = None


type EllipseColumns = Moments | SemiMajorMinor | Shear


# --------------------------------------------------------------------------
# An ABC for predefined and extension Grids we might want to draw.
#
# We probably want a way to save the users from having to import the
# predefined grids here, too, whether thats factory methods on BaseDisplay
# or kwargs that resolve to constructing those.


class BaseGrid(ABC):
    """An abstract interface for grids that can be overlaid on an image.

    Notes
    -----
    We expect display implementations to often detect certain concrete grid
    types (e.g. `HealpixGrid` and especially `SkyGrid`) to do their own
    drawing instead of calling `get_polygons`.
    """

    def get_polygons(self, box: Box, projection: Projection | None) -> dict[str, list[XY[float]]]:
        """Represent this grid as a mapping of pixel-coordinate polygons over
        an image's bounding box.

        Parameters
        ----------
        box
            The bounding box the polygons must cover.  The returned polygons
            may extend beyond this box.
        projection
            Projection that maps the image's pixel coordinates to sky
            coordinates.  If not provided, grids defined in sky coordinates
            will raise.

        Returns
        -------
        `dict`
            A dictionary mapping a grid-cell identifier to a list of polygon
            vertices in pixel coordinates.

        Notes
        -----
        Grids are mapped to polygons, not just lines, to make it possible for
        implementations to support selecting a grid cell and returning its
        identifier back to the display client.
        """
        raise NotImplementedError()


@final
@dataclasses.dataclass
class HealpixGrid(BaseGrid):
    """A HEALPix grid at a particular level."""

    level: int

    def get_polygons(self, box: Box, projection: Projection | None) -> dict[str, list[XY[float]]]:
        raise NotImplementedError("TODO")


@final
@dataclasses.dataclass
class HTMGrid(BaseGrid):
    """A Hierarchical Triangular Mesh grid at a particular level."""

    level: int

    def get_polygons(self, box: Box, projection: Projection | None) -> dict[str, list[XY[float]]]:
        raise NotImplementedError("TODO")


@final
@dataclasses.dataclass
class SkyGrid(BaseGrid):
    """A longitude/latitude grid."""

    spacing: float
    """Distance between grid lines in degrees."""

    kind: Literal["equatorial", "galactic", "ecliptic"] = "equatorial"


# Someday we could add SkyMap or focal plane grids, too.
