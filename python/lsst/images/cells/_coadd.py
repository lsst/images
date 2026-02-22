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

__all__ = ("CellCoadd", "CellCoaddSerializationModel")

import functools
from collections.abc import Mapping, Sequence
from types import EllipsisType
from typing import TYPE_CHECKING, Any, cast

import astropy.io.fits
import astropy.units
import astropy.wcs
import pydantic

from .._cell_grid import CellGrid, CellGridBounds, PatchDefinition
from .._geom import YX, Box
from .._image import Image, ImageSerializationModel
from .._mask import Mask, MaskPlane, MaskSchema, MaskSerializationModel
from .._masked_image import MaskedImage, MaskedImageSerializationModel
from .._transforms import Projection, ProjectionSerializationModel, TractFrame
from ..serialization import ArchiveReadError, InputArchive, OutputArchive
from ._provenance import CoaddProvenance, CoaddProvenanceSerializationModel
from ._psf import CellPointSpreadFunction, CellPointSpreadFunctionSerializationModel

if TYPE_CHECKING:
    try:
        from lsst.cell_coadds import MultipleCellCoadd
        from lsst.skymap import TractInfo
    except ImportError:
        type MultipleCellCoadd = Any  # type: ignore[no-redef]
        type TractInfo = Any  # type: ignore[no-redef]


class CellCoadd(MaskedImage):
    """A coadd comprised of cells on a regular grid.

    Parameters
    ----------
    image
        The main image plane.  If this has a `.Projection`, it will be used
        for all planes unless a ``projection`` is passed separately.
    mask
        A bitmask image that annotates the main image plane.  Must have the
        same bounding box as ``image`` if provided.  Any attached projection
        is replaced (possibly by `None`).
    variance
        The per-pixel uncertainty of the main image as an image of variance
        values.  Must have the same bounding box as ``image`` if provided, and
        its units must be the square of ``image.unit`` or `None`.
        Values default to ``1.0``.  Any attached projection is replaced
        (possibly by `None`).
    mask_fractions
        A mapping from an input-image mask plane name to an image of the
        weights sums of that plane.
    noise_realizations
        A sequence of images with Monte Carlo realizations of the noise in
        the coadd.
    mask_schema
        Schema for the mask plane.  Must be provided if and only if ``mask`` is
        not provided.
    projection
        Projection that maps the pixel grid to the sky.  Can only be `None` if
        a projection is already attached to ``image``.
    band
        Name of the band.
    psf
        Effective point-spread function for the coadd.  The missing cells
        reported by ``psf.bounds`` are assumed to apply to all image data for
        that cell as well (i.e. there is a PSF for a cell if and only if
        there is image data for that cell).
    patch
        Identifiers and geometry of the full patch, if the image is confined
        to a single patch.  When present, the cell grid of the PSF and
        provenance (if provideD) must be the full patch grid, even if its
        bounds select a subset of that area.
    provenance
        Information about the images that went into the coadd.
    """

    def __init__(
        self,
        image: Image,
        *,
        mask: Mask | None = None,
        variance: Image | None = None,
        mask_fractions: Mapping[str, Image] | None = None,
        noise_realizations: Sequence[Image] = (),
        mask_schema: MaskSchema | None = None,
        projection: Projection[TractFrame] | None = None,
        band: str | None = None,
        psf: CellPointSpreadFunction,
        patch: PatchDefinition | None = None,
        provenance: CoaddProvenance | None = None,
    ):
        super().__init__(
            image,
            mask=mask,
            variance=variance,
            mask_schema=mask_schema,
            projection=projection,
        )
        if self.image.unit is None:
            raise TypeError("The image component of a CellCoadd must have units.")
        if self.image.projection is None:
            raise TypeError("The projection component of a CellCoadd cannot be None.")
        if not isinstance(self.image.projection.pixel_frame, TractFrame):
            raise TypeError("The projection's pixel frame must be a TractFrame for CellCoadd.")
        self._mask_fractions = dict(mask_fractions) if mask_fractions is not None else {}
        self._noise_realizations = list(noise_realizations)
        self._band = band
        if psf.bounds.bbox != self.bbox:
            psf = psf[self.bbox]
        self._psf = psf
        self._patch = patch
        self._provenance = provenance
        if self._provenance and not self._patch:
            raise TypeError("A CellCoadd cannot carry provenance without a patch definition.")

    @property
    def skymap(self) -> str:
        """Name of the skymap (`str`)."""
        return self.projection.pixel_frame.skymap

    @property
    def tract(self) -> int:
        """ID of the tract (`int`)."""
        return self.projection.pixel_frame.tract

    @property
    def patch(self) -> PatchDefinition:
        """Identifiers and geometry of the full patch, if the image is confined
        to a single patch (`PatchDefinition`).
        """
        if self._patch is None:
            raise AttributeError("Coadd has no patch information.")
        return self._patch

    @property
    def band(self) -> str | None:
        """Name of the band (`str` or `None`)."""
        return self._band

    @property
    def mask_fractions(self) -> Mapping[str, Image]:
        """A mapping from an input-image mask plane name to an image of the
        weights sums of that plane
        (`~collections.abc.Mapping` [`str`, `.Image`]).
        """
        return self._mask_fractions

    @property
    def noise_realizations(self) -> Sequence[Image]:
        """A sequence of images with Monte Carlo realizations of the noise in
        the coadd (`~collections.abc.Sequence` [`.Image`]).
        """
        return self._noise_realizations

    @property
    def unit(self) -> astropy.units.UnitBase:
        """The units of the image plane (`astropy.units.Unit`)."""
        return cast(astropy.units.UnitBase, super().unit)

    @property
    def projection(self) -> Projection[TractFrame]:
        """The projection that maps the pixel grid to the sky
        (`.Projection` [`.TractFrame`]).
        """
        return cast(Projection[TractFrame], super().projection)

    @property
    def psf(self) -> CellPointSpreadFunction:
        """Effective point-spread function for the coadd
        (`CellPointSpreadFunction`).
        """
        return self._psf

    @property
    def bounds(self) -> CellGridBounds:
        """The grid of cells that overlap this coadd and a set of missing
        cells (`CellGridBounds`).
        """
        return self._psf.bounds

    @property
    def grid(self) -> CellGrid:
        """The grid of cells that overlap this coadd (`CellGrid`)."""
        return self._psf.bounds.grid

    @property
    def provenance(self) -> CoaddProvenance:
        """Information about the images that went into the coadd
        (`CoaddProvenance` or `None`).
        """
        if self._provenance is None:
            raise AttributeError("Coadd has no provenance information.")
        return self._provenance

    def __getitem__(self, bbox: Box | EllipsisType) -> CellCoadd:
        if bbox is ...:
            return self
        super().__getitem__(bbox)
        psf = self.psf[bbox]
        return self._transfer_metadata(
            CellCoadd(
                self.image[bbox],
                mask=self.mask[bbox],
                variance=self.variance[bbox],
                projection=self.projection,
                mask_fractions={k: v[bbox] for k, v in self._mask_fractions.items()},
                noise_realizations=[v[bbox] for v in self._noise_realizations],
                band=self.band,
                psf=psf,
                patch=self._patch,
                provenance=(
                    self._provenance.subset(psf.bounds.cell_indices())
                    if self._provenance is not None
                    else None
                ),
            ),
            bbox=bbox,
        )

    def __str__(self) -> str:
        return f"CellCoadd({self.bbox!s}, tract={self.tract})"

    def __repr__(self) -> str:
        return str(self)

    def copy(self) -> CellCoadd:
        """Deep-copy the coadd."""
        return self._transfer_metadata(
            CellCoadd(
                image=self._image.copy(),
                mask=self._mask.copy(),
                variance=self._variance.copy(),
                projection=self.projection,
                mask_fractions={k: v.copy() for k, v in self._mask_fractions.items()},
                noise_realizations=[v.copy() for v in self._noise_realizations],
                band=self.band,
                psf=self.psf,
                patch=self.patch,
                provenance=self.provenance,
            ),
            copy=True,
        )

    def serialize(self, archive: OutputArchive[Any]) -> CellCoaddSerializationModel:
        """Serialize the image to an output archive.

        Parameters
        ----------
        archive
            Archive to write to.
        """
        serialized_image = archive.serialize_direct(
            "image", functools.partial(self.image.serialize, save_projection=False)
        )
        serialized_mask = archive.serialize_direct(
            "mask", functools.partial(self.mask.serialize, save_projection=False)
        )
        serialized_variance = archive.serialize_direct(
            "variance", functools.partial(self.variance.serialize, save_projection=False)
        )
        serialized_projection = archive.serialize_direct("projection", self.projection.serialize)
        serialized_mask_fractions = {
            k: archive.serialize_direct(f"mask_fractions/{k}", v.serialize)
            for k, v in self.mask_fractions.items()
        }
        serialized_noise_realizations = [
            archive.serialize_direct(f"noise_realizations/{n}", v.serialize)
            for n, v in enumerate(self.noise_realizations)
        ]
        serialized_psf = archive.serialize_direct("psf", self.psf.serialize)
        serialized_provenance = (
            archive.serialize_direct("provenance", self._provenance.serialize)
            if self._provenance is not None
            else None
        )
        return CellCoaddSerializationModel(
            image=serialized_image,
            mask=serialized_mask,
            variance=serialized_variance,
            projection=serialized_projection,
            mask_fractions=serialized_mask_fractions,
            noise_realizations=serialized_noise_realizations,
            band=self._band,
            psf=serialized_psf,
            patch=self._patch,
            provenance=serialized_provenance,
            metadata=self.metadata,
        )

    # Type-checkers want the model argument to only require
    # MaskedImageSerializationModel[Any], and they'd be absolutely right if
    # this were a regular instance method. But whether Liskov substitution
    # applies to classmethods and staticmethods is sort of context-dependent,
    # and here we do not want it to.
    @staticmethod
    def deserialize(  # type: ignore[override]
        model: CellCoaddSerializationModel[Any],
        archive: InputArchive[Any],
        *,
        bbox: Box | None = None,
        provenance: bool = True,
    ) -> CellCoadd:
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
        provenance
            Whether to read and attach provenance information.
        """
        masked_image = MaskedImage.deserialize(model, archive, bbox=bbox)
        mask_fractions = {
            k.removeprefix("mask_fractions/"): Image.deserialize(v, archive)
            for k, v in model.mask_fractions.items()
        }
        noise_realizations = [Image.deserialize(v, archive) for v in model.noise_realizations]
        projection = Projection.deserialize(model.projection, archive)
        psf = CellPointSpreadFunction.deserialize(model.psf, archive, bbox=bbox)
        coadd_provenance: CoaddProvenance | None = None
        if model.provenance is not None and provenance:
            coadd_provenance = CoaddProvenance.deserialize(model.provenance, archive)
            if bbox is not None:
                coadd_provenance = coadd_provenance.subset(psf.bounds.cell_indices())
        return CellCoadd(
            masked_image.image,
            mask=masked_image.mask,
            variance=masked_image.variance,
            mask_fractions=mask_fractions,
            noise_realizations=noise_realizations,
            projection=projection,
            band=model.band,
            psf=psf,
            patch=model.patch,
            provenance=coadd_provenance,
        )._finish_deserialize(model)

    @staticmethod
    def _get_archive_tree_type[P: pydantic.BaseModel](
        pointer_type: type[P],
    ) -> type[CellCoaddSerializationModel[P]]:
        """Return the serialization model type for this object for an archive
        type that uses the given pointer type.
        """
        return CellCoaddSerializationModel[pointer_type]  # type: ignore

    # TODO: write_fits and read_fits inherited from MaskedImage, but that
    # write_fits doesn't have compression-option kwargs for all of the new
    # planes that CellCoadd adds.  This makes me lean towards dropping the
    # custom read_fits and write_fits in favor of the generic free functions
    # in the fits subpackage, even though those aren't ideal either.

    @staticmethod
    def from_legacy(  # type: ignore[override]
        legacy: MultipleCellCoadd,
        *,
        plane_map: Mapping[str, MaskPlane] | None = None,
        tract_info: TractInfo,
    ) -> CellCoadd:
        """Convert from an `lsst.cell_coadds.MultipleCellCoadd` instance.

        Parameters
        ----------
        legacy
            A `lsst.cell_coadds.MultipleCellCoadd` instance to convert.
        plane_map
            A mapping from legacy mask plane name to the new plane name and
            description.
        tract_info
            Information about the full tract.
        """
        from lsst.geom import Box2I

        legacy_bbox = Box2I()
        for single_cell in legacy.cells.values():
            legacy_bbox.include(single_cell.inner.bbox)
        legacy_stitched = legacy.stitch(legacy_bbox)
        unit = astropy.units.Unit(legacy.units.value)
        tract_bbox = Box.from_legacy(tract_info.getBBox())
        projection = Projection.from_legacy(
            legacy.wcs,
            TractFrame(
                skymap=legacy.identifiers.skymap,
                tract=legacy.identifiers.tract,
                bbox=tract_bbox,
            ),
            pixel_bounds=tract_bbox,
        )
        band = legacy.identifiers.band
        image = Image.from_legacy(legacy_stitched.image, unit=unit)
        mask = Mask.from_legacy(legacy_stitched.mask, plane_map=plane_map)
        variance = Image.from_legacy(legacy_stitched.variance, unit=unit**2)
        noise_realizations = [
            Image.from_legacy(noise_image) for noise_image in legacy_stitched.noise_realizations
        ]
        mask_fractions = (
            {"rejected": Image.from_legacy(legacy_stitched.mask_fractions)}
            if legacy_stitched.mask_fractions is not None
            else {}
        )
        psf = CellPointSpreadFunction.from_legacy(legacy_stitched.psf, image.bbox)
        patch_info = tract_info[legacy.identifiers.patch]
        patch = PatchDefinition(
            id=patch_info.getSequentialIndex(),
            index=YX(y=legacy.identifiers.patch.y, x=legacy.identifiers.patch.x),
            inner_bbox=Box.from_legacy(patch_info.getInnerBBox()),
            cells=CellGrid.from_legacy(legacy.grid),
        )
        provenance = CoaddProvenance.from_legacy(legacy)
        return CellCoadd(
            image=image,
            mask=mask,
            variance=variance,
            mask_fractions=mask_fractions,
            noise_realizations=noise_realizations,
            projection=projection,
            band=band,
            psf=psf,
            patch=patch,
            provenance=provenance,
        )


class CellCoaddSerializationModel[P: pydantic.BaseModel](MaskedImageSerializationModel[P]):
    """A Pydantic model used to represent a serialized `CellCoadd`."""

    # Inherited attributes are duplicated because that improves the docs
    # (some limitation in the sphinx/pydantic integration), and these are
    # important docs.

    image: ImageSerializationModel[P] = pydantic.Field(description="The main data image.")
    mask: MaskSerializationModel[P] = pydantic.Field(
        description="Bitmask that annotates the main image's pixels."
    )
    variance: ImageSerializationModel[P] = pydantic.Field(
        description="Per-pixel variance estimates for the main image."
    )
    projection: ProjectionSerializationModel[P] = pydantic.Field(
        description="Projection that maps the pixel grid to the sky.",
    )
    mask_fractions: dict[str, ImageSerializationModel[P]] = pydantic.Field(
        description=(
            "A mapping from an input-image mask plane name to an image of the weights sums of that plane."
        )
    )
    noise_realizations: list[ImageSerializationModel[P]] = pydantic.Field(
        description=(
            "A mapping from an input-image mask plane name to an image of the weights sums of that plane."
        )
    )
    band: str | None = pydantic.Field(description="Name of the band.")
    psf: CellPointSpreadFunctionSerializationModel = pydantic.Field(
        description="Effective point-spread function model for the coadd."
    )
    patch: PatchDefinition | None = pydantic.Field(description="Identifiers and geometry for the patch.")
    provenance: CoaddProvenanceSerializationModel | None = pydantic.Field(
        description="Information about the images that went into the coadd."
    )

    def deserialize_psf(self, archive: InputArchive[Any], bbox: Box | None = None) -> CellPointSpreadFunction:
        """Finish deserializing the PSF model."""
        return CellPointSpreadFunction.deserialize(self.psf, archive, bbox=bbox)

    def deserialize_provenance(self, archive: InputArchive[Any]) -> CoaddProvenance:
        """Finish deserializing the provenance information."""
        if self.provenance is not None:
            return CoaddProvenance.deserialize(self.provenance, archive)
        raise ArchiveReadError("No coadd provenance stored in this file.")
