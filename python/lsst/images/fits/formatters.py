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

__all__ = ("GenericFormatter", "ImageFormatter", "MaskedImageFormatter", "VisitImageFormatter")

import enum
import hashlib
import json
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
from ..serialization import ButlerInfo, TableCellReferenceModel
from ._common import FitsCompressionOptions
from ._input_archive import FitsInputArchive, read
from ._output_archive import write


class GenericFormatter(FormatterV2):
    """The butler interface to FITS archive serialization.

    Serialized types must meet all the requirements of the `read` and `write`
    functions.

    Notes
    -----
    This formatter just forwards all read parameters it receives as
    ``**kwargs`` to `.read` and hence the ``deserialize`` method of the type it
    is reading.  This may or may not be appropriate.

    This formatter must be subclassed to add component support.

    The write parameter configuration for this formatter is designed to be
    identical to that for the legacy FITS formatters defined in
    `lsst.obs.base`.

    Butler provenance is written to both FITS headers and the archive tree.
    """

    default_extension: ClassVar[str] = ".fits"
    can_read_from_uri: ClassVar[bool] = True
    supported_write_parameters: ClassVar[frozenset[str]] = frozenset({"recipe"})

    butler_provenance: DatasetProvenance | None = None

    def read_from_uri(self, uri: ResourcePath, component: str | None = None, expected_size: int = -1) -> Any:
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
            compression_options=self._get_compression_options(),
            compression_seed=self._get_compression_seed(),
            butler_info=butler_info,
        )

    def add_provenance(
        self, in_memory_dataset: Any, /, *, provenance: DatasetProvenance | None = None
    ) -> Any:
        # Instead of attaching the provenance to the object we remember it on
        # the formatter, since a Formatter instance is only used once.
        self.butler_provenance = provenance
        return in_memory_dataset

    def _get_compression_seed(self) -> int:
        # Set the seed based on data ID (all logic here duplicated from
        # obs_base). We can't just use 'hash', since like 'set' that's not
        # deterministic. And we can't rely on a DimensionPacker because those
        # are only defined for certain combinations of dimensions. Doing an MD5
        # of the JSON feels like overkill but I don't really see anything much
        # simpler.
        hash_bytes = hashlib.md5(
            json.dumps(list(self.data_id.required_values)).encode(),
            usedforsecurity=False,
        ).digest()
        # And it *really* feels like overkill when we squash that into the [1,
        # 10000] range allowed by FITS.
        return 1 + int.from_bytes(hash_bytes) % 9999

    def _get_compression_options(self) -> dict[str, FitsCompressionOptions]:
        recipe = self.write_parameters.get("recipe", "default")
        try:
            config = self.write_recipes[recipe]
        except KeyError:
            if recipe == "default":
                # If there's no default recipe just use the software defaults.
                return {}
            raise RuntimeError(f"Invalid recipe for ImageFormatter: {recipe!r}.") from None
        return {k: FitsCompressionOptions.model_validate(v) for k, v in config.items()}

    def _update_header(self, header: astropy.io.fits.Header) -> None:
        # Logic here largely lifted from lsst.obs.base.utils, which we
        # can't use directly for dependency and maybe mapping-type
        # (PropertyList vs. astropy) reasons. We assume we can always add
        # long cards (astropy will CONTINUE them) but not comments
        # (astropy will truncate and warn on long cards).
        for key in list(header):
            if key.startswith("LSST BUTLER"):
                del header[key]
        if self.butler_provenance is not None:
            for key, value in self.butler_provenance.to_flat_dict(
                self.dataset_ref, prefix="HIERARCH LSST BUTLER", sep=" ", simple_types=True, max_inputs=3_000
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
    """The specialized butler interface to FITS archive serialization of
    image-like objects with ``projection`` and ``bbox`` components.

    Notes
    -----
    This formatter works by assuming the `..serialization.ArchiveTree` for the
    top-level object has a ``projection`` attribute (a
    `..ProjectionSerializationModel`) and a ``bbox`` property (a `..Box`).

    Subclasses can add support for additional components by overriding
    `read_component`, delegating to `super`, and handling the cases where it
    returns a `ComponentSentinel` instance.
    """

    def read_from_uri(self, uri: ResourcePath, component: str | None = None, expected_size: int = -1) -> Any:
        pytype: Any = self.file_descriptor.storageClass.pytype
        if component is None:
            result = read(pytype, uri, bbox=self.pop_bbox_from_parameters()).deserialized
        else:
            with FitsInputArchive.open(uri, partial=True) as archive:
                tree = archive.get_tree(pytype._get_archive_tree_type(TableCellReferenceModel))
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

    def read_component(
        self,
        component: str,
        tree: Any,
        archive: FitsInputArchive,
    ) -> Any:
        match component:
            case "projection":
                if isinstance(
                    serialized_projection := getattr(tree, "projection", None), ProjectionSerializationModel
                ):
                    return Projection.deserialize(serialized_projection, archive)
                else:
                    return ComponentSentinel.INVALID_COMPONENT_MODEL
            case "bbox":
                if isinstance(bbox := getattr(tree, "bbox", None), Box):
                    return bbox
                else:
                    return ComponentSentinel.INVALID_COMPONENT_MODEL
            case "obs_info":
                if isinstance(obs_info := getattr(tree, "obs_info", None), ObservationInfo):
                    return obs_info
                else:
                    return ComponentSentinel.INVALID_COMPONENT_MODEL
        return ComponentSentinel.UNRECOGNIZED_COMPONENT


class MaskedImageFormatter(ImageFormatter):
    """A specialized butler interface to FITS archive serialization of
    the `..MaskedImage` class.
    """

    def read_component(
        self,
        component: str,
        tree: Any,
        archive: FitsInputArchive,
    ) -> Any:
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
    """A specialized butler interface to FITS archive serialization of
    the `..VisitImage` class.
    """

    def read_component(
        self,
        component: str,
        tree: Any,
        archive: FitsInputArchive,
    ) -> Any:
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
        return ComponentSentinel.UNRECOGNIZED_COMPONENT
