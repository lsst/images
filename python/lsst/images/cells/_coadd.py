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
from typing import TYPE_CHECKING, Any, ClassVar, cast

import astropy.io.fits
import astropy.units
import astropy.wcs
import pydantic

from .._backgrounds import BackgroundMap, BackgroundMapSerializationModel
from .._cell_grid import CellGrid, CellGridBounds, PatchDefinition
from .._geom import YX, Box
from .._image import Image, ImageSerializationModel
from .._mask import Mask, MaskPlane, MaskSchema, MaskSerializationModel, get_legacy_deep_coadd_mask_planes
from .._masked_image import MaskedImage, MaskedImageSerializationModel
from .._transforms import Projection, ProjectionSerializationModel, TractFrame
from ..serialization import InputArchive, InvalidParameterError, OutputArchive
from ._aperture_corrections import CellApertureCorrectionMapSerializationModel, CellField
from ._provenance import CoaddProvenance, CoaddProvenanceSerializationModel
from ._psf import CellPointSpreadFunction, CellPointSpreadFunctionSerializationModel

if TYPE_CHECKING:
    try:
        from lsst.cell_coadds import MultipleCellCoadd as LegacyMultipleCellCoadd
        from lsst.skymap import TractInfo
    except ImportError:
        type LegacyMultipleCellCoadd = Any  # type: ignore[no-redef]
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
    aperture_corrections
        Aperture corrections for different photometry algorithms.
    patch
        Identifiers and geometry of the full patch, if the image is confined
        to a single patch.  When present, the cell grid of the PSF and
        provenance (if provideD) must be the full patch grid, even if its
        bounds select a subset of that area.
    provenance
        Information about the images that went into the coadd.
    backgrounds
        Background models associated with this image.
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
        aperture_corrections: Mapping[str, CellField] | None = None,
        patch: PatchDefinition | None = None,
        provenance: CoaddProvenance | None = None,
        backgrounds: BackgroundMap | None = None,
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
        self._aperture_corrections = dict(aperture_corrections) if aperture_corrections is not None else {}
        for ap_corr_name, ap_corr_field in self._aperture_corrections.items():
            if ap_corr_field.bounds.grid != self.grid:
                raise ValueError(
                    f"Grids for cell PSF and {ap_corr_name} aperture corrections are not consistent."
                )
        self._patch = patch
        self._provenance = provenance
        if self._provenance and not self._patch:
            raise TypeError("A CellCoadd cannot carry provenance without a patch definition.")
        self._backgrounds = backgrounds if backgrounds is not None else BackgroundMap()

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
    def aperture_corrections(self) -> Mapping[str, CellField]:
        """Aperture corrections for different photometry algorithms
        (`dict` [`str`, `CellField`]).
        """
        return self._aperture_corrections

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

    @property
    def backgrounds(self) -> BackgroundMap:
        """A mapping of backgrounds associated with this image
        (`.BackgroundMap`).
        """
        return self._backgrounds

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
                backgrounds=self._backgrounds,
                aperture_corrections=self._aperture_corrections.copy(),
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
                backgrounds=self._backgrounds.copy(),
                aperture_corrections=self._aperture_corrections.copy(),
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
            "image",
            functools.partial(self.image.serialize, save_projection=False, tile_shape=self.grid.cell_shape),
        )
        serialized_mask = archive.serialize_direct(
            "mask",
            functools.partial(self.mask.serialize, save_projection=False, tile_shape=self.grid.cell_shape),
        )
        serialized_variance = archive.serialize_direct(
            "variance",
            functools.partial(
                self.variance.serialize, save_projection=False, tile_shape=self.grid.cell_shape
            ),
        )
        serialized_projection = archive.serialize_direct("projection", self.projection.serialize)
        serialized_mask_fractions = {
            k: archive.serialize_direct(
                f"mask_fractions/{k}",
                functools.partial(
                    v.serialize,
                    save_projection=False,
                    tile_shape=self.grid.cell_shape,
                    options_name="mask_fractions",
                ),
            )
            for k, v in self.mask_fractions.items()
        }
        serialized_noise_realizations = [
            archive.serialize_direct(
                f"noise_realizations/{n}",
                functools.partial(
                    v.serialize, save_projection=False, tile_shape=self.grid.cell_shape, options_name="image"
                ),
            )
            for n, v in enumerate(self.noise_realizations)
        ]
        serialized_psf = archive.serialize_direct("psf", self.psf.serialize)
        serialized_aperture_corrections = archive.serialize_direct(
            "aperture_corrections",
            functools.partial(
                CellApertureCorrectionMapSerializationModel.serialize, self.aperture_corrections
            ),
        )
        serialized_provenance = (
            archive.serialize_direct("provenance", self._provenance.serialize)
            if self._provenance is not None
            else None
        )
        serialized_backgrounds = archive.serialize_direct("background", self._backgrounds.serialize)
        return CellCoaddSerializationModel(
            image=serialized_image,
            mask=serialized_mask,
            variance=serialized_variance,
            projection=serialized_projection,
            mask_fractions=serialized_mask_fractions,
            noise_realizations=serialized_noise_realizations,
            band=self._band,
            psf=serialized_psf,
            aperture_corrections=serialized_aperture_corrections,
            patch=self._patch,
            provenance=serialized_provenance,
            backgrounds=serialized_backgrounds,
            metadata=self.metadata,
        )

    @staticmethod
    def _get_archive_tree_type[P: pydantic.BaseModel](
        pointer_type: type[P],
    ) -> type[CellCoaddSerializationModel[P]]:
        """Return the serialization model type for this object for an archive
        type that uses the given pointer type.
        """
        return CellCoaddSerializationModel[pointer_type]  # type: ignore

    @staticmethod
    def from_legacy(  # type: ignore[override]
        legacy: LegacyMultipleCellCoadd,
        *,
        plane_map: Mapping[str, MaskPlane] | None = None,
        tract_info: TractInfo,
    ) -> CellCoadd:
        """Convert from a `lsst.cell_coadds.MultipleCellCoadd` instance.

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

        if plane_map is None:
            plane_map = get_legacy_deep_coadd_mask_planes()
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
        aperture_corrections = {
            ap_corr_name: CellField.from_legacy_aperture_correction(legacy_ap_corr, psf.bounds)
            for ap_corr_name, legacy_ap_corr in legacy_stitched.ap_corr_map.items()
        }
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
            aperture_corrections=aperture_corrections,
            patch=patch,
            provenance=provenance,
        )

    def to_legacy(
        self, copy: bool | None = None, plane_map: Mapping[str, MaskPlane] | None = None
    ) -> LegacyMultipleCellCoadd:
        """Convert to a `lsst.cell_coadds.MultipleCellCoadd` instance.

        Parameters
        ----------
        copy
            If `True`, always copy the image and variance pixel data.
            If `False`, return a view, and raise `TypeError` if the pixel data
            is read-only (this is not supported by afw).  If `None`, only copy
            if the pixel data is read-only.  Mask pixel data is always copied.
        plane_map
            A mapping from legacy mask plane name to the new plane name and
            description.
        """
        from frozendict import frozendict  # type: ignore[import-not-found]

        from lsst.cell_coadds import CellIdentifiers as LegacyCellIdentifiers
        from lsst.cell_coadds import CoaddUnits as LegacyCoaddUnits
        from lsst.cell_coadds import CommonComponents as LegacyCommonComponents
        from lsst.cell_coadds import MultipleCellCoadd as LegacyMultipleCellCoadd
        from lsst.cell_coadds import OwnedImagePlanes as LegacyOwnedImagePlanes
        from lsst.cell_coadds import PatchIdentifiers as LegacyPatchIdentifiers
        from lsst.cell_coadds import SingleCellCoadd as LegacySingleCellCoadd
        from lsst.skymap import Index2D as LegacyIndex2D

        if plane_map is None:
            plane_map = get_legacy_deep_coadd_mask_planes()
        if self.unit != astropy.units.nJy:
            raise ValueError("CellCoadd.to_legacy requires nJy pixel units.")
        legacy_grid = self.grid.to_legacy()
        visit_polygons = self.provenance.to_legacy_polygon_map()
        legacy_common = LegacyCommonComponents(
            units=LegacyCoaddUnits.nJy,
            wcs=self.projection.to_legacy(),
            band=self.band,
            identifiers=LegacyPatchIdentifiers(
                self.skymap,
                self.tract,
                LegacyIndex2D(x=self.patch.index.x, y=self.patch.index.y),
                band=self.band,
            ),
            visit_polygons=visit_polygons,
        )
        legacy_inputs = self.provenance.to_legacy_cell_coadd_inputs(visit_polygons.keys())
        cells: list[LegacySingleCellCoadd] = []
        for cell_index in self.bounds.cell_indices():
            cell_bbox = self.grid.bbox_of(cell_index)
            # Legacy type only has room for one mask_fractions plane.
            legacy_mask_fractions = (
                next(iter(self.mask_fractions.values()))[cell_bbox].to_legacy(copy=copy)
                if self.mask_fractions
                else None
            )
            legacy_planes = LegacyOwnedImagePlanes(
                image=self.image[cell_bbox].to_legacy(copy=copy),
                mask=self.mask[cell_bbox].to_legacy(plane_map),
                variance=self.variance[cell_bbox].to_legacy(copy=copy),
                mask_fractions=legacy_mask_fractions,
                noise_realizations=[n[cell_bbox].to_legacy(copy=copy) for n in self.noise_realizations],
            )
            legacy_aperture_correction_map = frozendict(
                {name: field.value_in_cell(cell_index) for name, field in self.aperture_corrections.items()}
            )
            cells.append(
                LegacySingleCellCoadd(
                    legacy_planes,
                    psf=self.psf[cell_index].to_legacy(copy=copy),
                    inner_bbox=cell_bbox.to_legacy(),
                    common=legacy_common,
                    inputs=legacy_inputs[cell_index.to_legacy()],
                    identifiers=LegacyCellIdentifiers(
                        self.skymap,
                        self.tract,
                        legacy_common.identifiers.patch,
                        band=self.band,
                        cell=cell_index.to_legacy(),
                    ),
                    aperture_correction_map=legacy_aperture_correction_map,
                )
            )
        return LegacyMultipleCellCoadd(
            cells,
            legacy_grid,
            outer_cell_size=self.grid.cell_shape.to_legacy_extent(),
            psf_image_size=self.psf.kernel_bbox.shape.to_legacy_extent(),
            common=legacy_common,
            inner_bbox=self.bbox.to_legacy(),
        )


class CellCoaddSerializationModel[P: pydantic.BaseModel](MaskedImageSerializationModel[P]):
    """A Pydantic model used to represent a serialized `CellCoadd`."""

    SCHEMA_NAME: ClassVar[str] = "cell_coadd"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = CellCoadd

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
    aperture_corrections: CellApertureCorrectionMapSerializationModel | None = pydantic.Field(
        None, description="Coadded aperture corrections for different photometry algorithms."
    )
    patch: PatchDefinition | None = pydantic.Field(description="Identifiers and geometry for the patch.")
    provenance: CoaddProvenanceSerializationModel | None = pydantic.Field(
        description="Information about the images that went into the coadd."
    )
    backgrounds: BackgroundMapSerializationModel = pydantic.Field(
        default_factory=BackgroundMapSerializationModel,
        description="Background models associated with this image.",
    )

    def deserialize(  # type: ignore[override]
        self,
        archive: InputArchive[Any],
        *,
        bbox: Box | None = None,
        provenance: bool = True,
        **kwargs: Any,
    ) -> CellCoadd:
        """Deserialize an image from an input archive.

        Parameters
        ----------
        archive
            Archive to read from.
        bbox
            Bounding box of a subimage to read instead.
        provenance
            Whether to read and attach provenance information.
        **kwargs
            Unsupported keyword arguments are accepted only to provide better
            error messages (raising `.serialization.InvalidParameterError`).
        """
        if kwargs:
            raise InvalidParameterError(f"Unrecognized parameters for CellCoadd: {set(kwargs.keys())}.")
        masked_image = super().deserialize(archive, bbox=bbox)
        mask_fractions = {
            k.removeprefix("mask_fractions/"): v.deserialize(archive) for k, v in self.mask_fractions.items()
        }
        noise_realizations = [v.deserialize(archive) for v in self.noise_realizations]
        projection = self.projection.deserialize(archive)
        psf = self.psf.deserialize(archive, bbox=bbox)
        aperture_corrections = (
            self.aperture_corrections.deserialize(archive) if self.aperture_corrections is not None else {}
        )
        coadd_provenance: CoaddProvenance | None = None
        if self.provenance is not None and provenance:
            coadd_provenance = self.provenance.deserialize(archive)
            if bbox is not None:
                coadd_provenance = coadd_provenance.subset(psf.bounds.cell_indices())
        backgrounds = self.backgrounds.deserialize(archive)
        return CellCoadd(
            masked_image.image,
            mask=masked_image.mask,
            variance=masked_image.variance,
            mask_fractions=mask_fractions,
            noise_realizations=noise_realizations,
            projection=projection,
            band=self.band,
            psf=psf,
            aperture_corrections=aperture_corrections,
            patch=self.patch,
            provenance=coadd_provenance,
            backgrounds=backgrounds,
        )._finish_deserialize(self)

    def deserialize_component(self, component: str, archive: InputArchive[Any], **kwargs: Any) -> Any:
        match component:
            case "mask_fractions":
                return {
                    name: image_model.deserialize(archive, **kwargs)
                    for name, image_model in self.mask_fractions.items()
                }
            case "noise_realizations":
                return [image_model.deserialize(archive, **kwargs) for image_model in self.noise_realizations]
            case "aperture_corrections" if self.aperture_corrections is None:
                # super() delegation handles the not-None case.
                return {}
        return super().deserialize_component(component, archive, **kwargs)
