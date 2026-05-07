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
    "GenericFormatter",
    "ImageFormatter",
    "MaskedImageFormatter",
    "VisitImageFormatter",
)

import enum
from typing import Any, ClassVar

import astropy.io.fits
from astro_metadata_translator import ObservationInfo

from lsst.daf.butler import DatasetProvenance, FormatterV2
from lsst.resources import ResourcePath

from .._geom import Box
from .._image import Image
from .._mask import Mask
from .._masked_image import MaskedImageSerializationModel
from .._transforms import Projection, ProjectionSerializationModel
from .._visit_image import VisitImageSerializationModel
from ..serialization import ButlerInfo
from ._common import NdfPointerModel
from ._input_archive import NdfInputArchive, read
from ._output_archive import write


class GenericFormatter(FormatterV2):
    """Butler interface to the NDF (HDS-on-HDF5) archive.

    Serialized types must meet all the requirements of the `read` and `write`
    functions.

    Notes
    -----
    This formatter just forwards all read parameters it receives as
    ``**kwargs`` to `.read` and hence the ``deserialize`` method of the type it
    is reading. This may or may not be appropriate.

    This formatter must be subclassed to add component support.

    Butler provenance is written to the FITS header of the NDF archive.
    """

    default_extension: ClassVar[str] = ".sdf"
    can_read_from_uri: ClassVar[bool] = True
    supported_write_parameters: ClassVar[frozenset[str]] = frozenset()

    butler_provenance: DatasetProvenance | None = None

    def read_from_uri(
        self,
        uri: ResourcePath,
        component: str | None = None,
        expected_size: int = -1,
    ) -> Any:
        pytype = self.dataset_ref.datasetType.storageClass.pytype
        kwargs = self.file_descriptor.parameters or {}
        return read(pytype, uri, **kwargs).deserialized

    def write_local_file(self, in_memory_dataset: Any, uri: ResourcePath) -> None:
        butler_info = ButlerInfo(
            dataset=self.dataset_ref.to_simple(),
            provenance=self.butler_provenance if self.butler_provenance is not None else DatasetProvenance(),
        )
        write(
            in_memory_dataset,
            uri.ospath,
            update_header=self._update_header,
            butler_info=butler_info,
        )

    def add_provenance(
        self,
        in_memory_dataset: Any,
        /,
        *,
        provenance: DatasetProvenance | None = None,
    ) -> Any:
        # Stash provenance for use during write_local_file. Mirrors the FITS
        # formatter's implementation.
        self.butler_provenance = provenance
        return in_memory_dataset

    def _update_header(self, header: astropy.io.fits.Header) -> None:
        # Inject butler provenance into the opaque-FITS primary header so it
        # round-trips through /MORE/FITS. Logic lifted from
        # lsst.images.fits.formatters.GenericFormatter._update_header (also
        # see lsst.obs.base.utils for the original).
        for key in list(header):
            if key.startswith("LSST BUTLER"):
                del header[key]
        if self.butler_provenance is not None:
            for key, value in self.butler_provenance.to_flat_dict(
                self.dataset_ref,
                prefix="HIERARCH LSST BUTLER",
                sep=" ",
                simple_types=True,
                max_inputs=3_000,
            ).items():
                header.set(key, value)


class ComponentSentinel(enum.Enum):
    """Special values returned by `ImageFormatter.read_component`."""

    UNRECOGNIZED_COMPONENT = enum.auto()
    """This formatter does not recognize the given component, but a subclass
    might.
    """

    INVALID_COMPONENT_MODEL = enum.auto()
    """This formatter recognizes the given component, but the expected
    attribute of the top-level `..serialization.ArchiveTree` did not exist
    or had the wrong type.
    """


class ImageFormatter(GenericFormatter):
    """Specialised butler interface to NDF serialization for image-like
    objects with ``projection`` and ``bbox`` components.

    Notes
    -----
    This formatter works by assuming the `..serialization.ArchiveTree` for the
    top-level object has a ``projection`` attribute (a
    `..ProjectionSerializationModel`) and a ``bbox`` property (a `..Box`).

    Subclasses can add support for additional components by overriding
    `read_component`, delegating to `super`, and handling the cases where it
    returns a `ComponentSentinel` instance.
    """

    def read_from_uri(
        self,
        uri: ResourcePath,
        component: str | None = None,
        expected_size: int = -1,
    ) -> Any:
        pytype: Any = self.file_descriptor.storageClass.pytype
        if component is None:
            result = read(pytype, uri, bbox=self.pop_bbox_from_parameters()).deserialized
        else:
            with NdfInputArchive.open(uri) as archive:
                tree = archive.get_tree(pytype._get_archive_tree_type(NdfPointerModel))
                result = self.read_component(component, tree, archive)
                if result is ComponentSentinel.UNRECOGNIZED_COMPONENT:
                    raise NotImplementedError(
                        f"Unrecognized component {component!r} for {type(self).__name__}."
                    )
                if result is ComponentSentinel.INVALID_COMPONENT_MODEL:
                    raise NotImplementedError(
                        f"Invalid serialization model for component {component!r} for {type(self).__name__}."
                    )
        self.check_unhandled_parameters()
        return result

    def pop_bbox_from_parameters(self) -> Box | None:
        parameters = self.file_descriptor.parameters or {}
        return parameters.pop("bbox", None)

    def check_unhandled_parameters(self) -> None:
        if self.file_descriptor.parameters:
            raise RuntimeError(f"Parameters {list(self.file_descriptor.parameters.keys())} not recognized.")

    def read_component(self, component: str, tree: Any, archive: NdfInputArchive) -> Any:
        match component:
            case "projection":
                if isinstance(p := getattr(tree, "projection", None), ProjectionSerializationModel):
                    return Projection.deserialize(p, archive)
                return ComponentSentinel.INVALID_COMPONENT_MODEL
            case "bbox":
                if isinstance(bbox := getattr(tree, "bbox", None), Box):
                    return bbox
                return ComponentSentinel.INVALID_COMPONENT_MODEL
            case "obs_info":
                if isinstance(oi := getattr(tree, "obs_info", None), ObservationInfo):
                    return oi
                return ComponentSentinel.INVALID_COMPONENT_MODEL
        return ComponentSentinel.UNRECOGNIZED_COMPONENT


class MaskedImageFormatter(ImageFormatter):
    """Butler interface to NDF serialization for `MaskedImage`."""

    def read_component(self, component: str, tree: Any, archive: NdfInputArchive) -> Any:
        match super().read_component(component, tree, archive):
            case ComponentSentinel():
                pass
            case handled:
                return handled
        if not isinstance(tree, MaskedImageSerializationModel):
            return ComponentSentinel.INVALID_COMPONENT_MODEL
        match component:
            case "image":
                return Image.deserialize(tree.image, archive, bbox=self.pop_bbox_from_parameters())
            case "mask":
                return Mask.deserialize(tree.mask, archive, bbox=self.pop_bbox_from_parameters())
            case "variance":
                return Image.deserialize(tree.variance, archive, bbox=self.pop_bbox_from_parameters())
        return ComponentSentinel.UNRECOGNIZED_COMPONENT


class VisitImageFormatter(MaskedImageFormatter):
    """Butler interface to NDF serialization for `VisitImage`."""

    def read_component(self, component: str, tree: Any, archive: NdfInputArchive) -> Any:
        match super().read_component(component, tree, archive):
            case ComponentSentinel():
                pass
            case handled:
                return handled
        if not isinstance(tree, VisitImageSerializationModel):
            return ComponentSentinel.INVALID_COMPONENT_MODEL
        match component:
            case "psf":
                return tree.deserialize_psf(archive)
            case "summary_stats":
                return tree.summary_stats
            case "aperture_corrections":
                return tree.aperture_corrections.deserialize(archive)
        return ComponentSentinel.UNRECOGNIZED_COMPONENT
