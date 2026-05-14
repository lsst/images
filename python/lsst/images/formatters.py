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

"""Unified butler formatter for lsst.images.

This formatter dispatches on a write-time ``format`` parameter and on the
file extension at read time, replacing the three per-format
(`lsst.images.fits.formatters`, `lsst.images.json.formatters`,
`lsst.images.ndf.formatters`) hierarchies that previously duplicated almost
all of their logic.
"""

from __future__ import annotations

__all__ = (
    "CellCoaddFormatter",
    "ComponentSentinel",
    "GenericFormatter",
    "ImageFormatter",
    "MaskedImageFormatter",
    "VisitImageFormatter",
)

import enum
import hashlib
import json as _stdlib_json  # disambiguates from .json subpackage
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import astropy.io.fits
from astro_metadata_translator import ObservationInfo

from lsst.daf.butler import DatasetProvenance, FormatterV2
from lsst.resources import ResourcePath

from . import fits as _fits
from . import json as _json
from ._geom import Box
from ._masked_image import MaskedImageSerializationModel
from ._transforms import ProjectionSerializationModel
from ._visit_image import VisitImageSerializationModel
from .fits._common import FitsCompressionOptions
from .fits._common import PointerModel as _FitsPointerModel
from .fits._input_archive import FitsInputArchive as _FitsInputArchive
from .serialization import ButlerInfo

try:
    from . import ndf as _ndf
    from .ndf._common import NdfPointerModel as _NdfPointerModel
    from .ndf._input_archive import NdfInputArchive as _NdfInputArchive

    _HAVE_NDF = True
except ImportError:  # h5py is optional; see ndf/__init__.py
    _ndf = None  # type: ignore[assignment]
    _NdfPointerModel = None  # type: ignore[assignment]
    _NdfInputArchive = None  # type: ignore[assignment]
    _HAVE_NDF = False


@dataclass(frozen=True)
class _Backend:
    """One row of the extension-to-backend lookup table."""

    read: Callable[..., Any]
    write: Callable[..., Any]
    input_archive: type | None
    pointer_model: type | None


_BACKENDS: dict[str, _Backend] = {
    ".fits": _Backend(
        read=_fits.read,
        write=_fits.write,
        input_archive=_FitsInputArchive,
        pointer_model=_FitsPointerModel,
    ),
    ".json": _Backend(
        read=_json.read,
        write=_json.write,
        input_archive=None,
        pointer_model=None,
    ),
}
if _HAVE_NDF:
    _BACKENDS[".sdf"] = _Backend(
        read=_ndf.read,
        write=_ndf.write,
        input_archive=_NdfInputArchive,
        pointer_model=_NdfPointerModel,
    )


class GenericFormatter(FormatterV2):
    """Unified butler formatter for any lsst.images type.

    The on-disk format is selected by the ``format`` write parameter
    (``fits``, ``json``, ``sdf``) at write time and by the file
    extension at read time. The default format is taken from
    ``self.default_extension`` (``.fits`` for the base class).

    Notes
    -----
    Subclasses (`ImageFormatter` and below) add component-level read
    support. This base class forwards any read parameters straight to
    the underlying ``read`` function.
    """

    default_extension: ClassVar[str] = ".fits"
    supported_extensions: ClassVar[frozenset[str]] = frozenset({".fits", ".sdf", ".json"})
    supported_write_parameters: ClassVar[frozenset[str]] = frozenset({"format", "recipe"})
    can_read_from_uri: ClassVar[bool] = True

    butler_provenance: DatasetProvenance | None = None

    # --- Write parameter handling -------------------------------------------

    @property
    def write_parameters(self) -> dict[str, Any]:  # type: ignore[override]
        # Allow unit tests to inject a dict via `_write_parameters`. The
        # FormatterV2 base provides the property pulling from the file
        # descriptor; override only when our private attribute is set.
        params = getattr(self, "_write_parameters", None)
        if params is not None:
            return params
        return super().write_parameters  # type: ignore[misc]

    def get_write_extension(self) -> str:
        default_fmt = self.default_extension.lstrip(".")
        fmt = self.write_parameters.get("format", default_fmt)
        ext = "." + fmt
        if ext not in self.supported_extensions:
            raise RuntimeError(
                f"Requested format {fmt!r} is not supported; expected one of {{fits, json, sdf}}."
            )
        return ext

    def _validate_write_parameters(self) -> None:
        ext = self.get_write_extension()
        if ext != ".fits" and "recipe" in self.write_parameters:
            raise RuntimeError("The 'recipe' write parameter is only valid for FITS output.")

    # --- Write path ---------------------------------------------------------

    def write_local_file(self, in_memory_dataset: Any, uri: ResourcePath) -> None:
        self._validate_write_parameters()
        ext = self.get_write_extension()
        backend = _BACKENDS[ext]
        butler_info = ButlerInfo(
            dataset=self.dataset_ref.to_simple(),
            provenance=self.butler_provenance if self.butler_provenance is not None else DatasetProvenance(),
        )
        kwargs: dict[str, Any] = {"butler_info": butler_info}
        if ext == ".fits":
            kwargs["update_header"] = self._update_header
            kwargs["compression_options"] = self._get_compression_options()
            kwargs["compression_seed"] = self._get_compression_seed()
        elif ext == ".sdf":
            kwargs["update_header"] = self._update_header
        backend.write(in_memory_dataset, uri.ospath, **kwargs)

    def add_provenance(
        self,
        in_memory_dataset: Any,
        /,
        *,
        provenance: DatasetProvenance | None = None,
    ) -> Any:
        # A FormatterV2 instance is used once; stash provenance on self
        # rather than mutating the dataset.
        self.butler_provenance = provenance
        return in_memory_dataset

    # --- FITS-specific helpers (kept verbatim from fits/formatters.py) ----

    def _get_compression_seed(self) -> int:
        # Set the seed based on data ID (all logic here duplicated from
        # obs_base). We can't just use 'hash', since like 'set' that's not
        # deterministic. And we can't rely on a DimensionPacker because those
        # are only defined for certain combinations of dimensions. Doing an MD5
        # of the JSON feels like overkill but I don't really see anything much
        # simpler.
        hash_bytes = hashlib.md5(
            _stdlib_json.dumps(list(self.data_id.required_values)).encode(),
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
            raise RuntimeError(f"Invalid recipe for GenericFormatter: {recipe!r}.") from None
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
                self.dataset_ref,
                prefix="HIERARCH LSST BUTLER",
                sep=" ",
                simple_types=True,
                max_inputs=3_000,
            ).items():
                header.set(key, value)

    # --- Read path ---------------------------------------------------------

    def _extension_from_uri(self, uri: ResourcePath) -> str:
        ext = uri.getExtension()
        if ext not in self.supported_extensions:
            raise RuntimeError(f"Cannot read {uri}: unsupported extension {ext!r}.")
        return ext

    def read_from_uri(
        self,
        uri: ResourcePath,
        component: str | None = None,
        expected_size: int = -1,
    ) -> Any:
        pytype = self.dataset_ref.datasetType.storageClass.pytype
        ext = self._extension_from_uri(uri)
        backend = _BACKENDS[ext]
        kwargs = self.file_descriptor.parameters or {}
        return backend.read(pytype, uri, **kwargs).deserialized


class ComponentSentinel(enum.Enum):
    """Special return values from `ImageFormatter.read_component`."""

    UNRECOGNIZED_COMPONENT = enum.auto()
    """Subclasses might still recognise this component."""

    INVALID_COMPONENT_MODEL = enum.auto()
    """Component name is known but the model attribute is missing or
    has the wrong type.
    """


class ImageFormatter(GenericFormatter):
    """Adds component-level read support for image-like types.

    Subclasses override `read_component` to handle additional components
    (image/mask/variance for MaskedImage; psf/summary_stats/etc. for
    VisitImage).
    """

    def _storage_class_pytype_default(self) -> type:
        return self.file_descriptor.storageClass.pytype

    def _get_pytype(self) -> type:
        # Allow unit tests to inject a pytype without a real FileDescriptor.
        pytype = getattr(self, "_storage_class_pytype", None)
        if pytype is not None:
            return pytype
        return self._storage_class_pytype_default()

    def read_from_uri(
        self,
        uri: ResourcePath,
        component: str | None = None,
        expected_size: int = -1,
    ) -> Any:
        pytype = self._get_pytype()
        ext = self._extension_from_uri(uri)
        backend = _BACKENDS[ext]
        if component is None:
            result = backend.read(pytype, uri, bbox=self.pop_bbox_from_parameters()).deserialized
        else:
            result = self._read_component_from_uri(component, uri)
        self.check_unhandled_parameters()
        return result

    def _read_component_from_uri(self, component: str, uri: ResourcePath) -> Any:
        ext = self._extension_from_uri(uri)
        backend = _BACKENDS[ext]
        pytype = self._get_pytype()
        if ext == ".json":
            obj = backend.read(pytype, uri).deserialized
            try:
                return getattr(obj, component)
            except AttributeError as exc:
                raise NotImplementedError(f"Unrecognized component {component!r} for JSON read.") from exc
        # FITS/NDF archive path.
        archive_cls = backend.input_archive
        pointer_model = backend.pointer_model
        assert archive_cls is not None
        assert pointer_model is not None
        # FitsInputArchive uses partial=True for component reads; NDF
        # has no such kwarg.
        open_kwargs = {"partial": True} if ext == ".fits" else {}
        with archive_cls.open(uri, **open_kwargs) as archive:
            tree_type = pytype._get_archive_tree_type(pointer_model)
            tree = archive.get_tree(tree_type)
            result = self.read_component(component, tree, archive)
        if result is ComponentSentinel.UNRECOGNIZED_COMPONENT:
            raise NotImplementedError(f"Unrecognized component {component!r} for {type(self).__name__}.")
        if result is ComponentSentinel.INVALID_COMPONENT_MODEL:
            raise NotImplementedError(
                f"Invalid serialization model for component {component!r} for {type(self).__name__}."
            )
        return result

    def _get_parameters(self) -> dict[str, Any] | None:
        # Allow unit tests to inject parameters without a real FileDescriptor.
        if hasattr(self, "_parameters"):
            return self._parameters
        try:
            return self.file_descriptor.parameters
        except AttributeError:
            return None

    def pop_bbox_from_parameters(self) -> Box | None:
        parameters = self._get_parameters() or {}
        return parameters.pop("bbox", None)

    def check_unhandled_parameters(self) -> None:
        if self._get_parameters():
            raise RuntimeError(f"Parameters {list(self._get_parameters().keys())} not recognized.")  # type: ignore[union-attr]

    def read_component(self, component: str, tree: Any, archive: Any) -> Any:
        match component:
            case "projection":
                if isinstance(
                    p := getattr(tree, "projection", None),
                    ProjectionSerializationModel,
                ):
                    return p.deserialize(archive)
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
    """Adds image/mask/variance component support."""

    def read_component(self, component: str, tree: Any, archive: Any) -> Any:
        match super().read_component(component, tree, archive):
            case ComponentSentinel():
                pass
            case handled:
                return handled
        if not isinstance(tree, MaskedImageSerializationModel):
            return ComponentSentinel.INVALID_COMPONENT_MODEL
        match component:
            case "image":
                return tree.image.deserialize(archive, bbox=self.pop_bbox_from_parameters())
            case "mask":
                return tree.mask.deserialize(archive, bbox=self.pop_bbox_from_parameters())
            case "variance":
                return tree.variance.deserialize(archive, bbox=self.pop_bbox_from_parameters())
        return ComponentSentinel.UNRECOGNIZED_COMPONENT


class VisitImageFormatter(MaskedImageFormatter):
    """Adds psf/summary_stats/detector/aperture_corrections."""

    def read_component(self, component: str, tree: Any, archive: Any) -> Any:
        match super().read_component(component, tree, archive):
            case ComponentSentinel():
                pass
            case handled:
                return handled
        if not isinstance(tree, VisitImageSerializationModel):
            return ComponentSentinel.INVALID_COMPONENT_MODEL
        match component:
            case "psf":
                # The FITS path uses tree.psf.deserialize; the NDF tree
                # exposes deserialize_psf for the same effect.
                if hasattr(tree, "deserialize_psf"):
                    return tree.deserialize_psf(archive)
                return tree.psf.deserialize(archive)
            case "summary_stats":
                return tree.summary_stats
            case "detector":
                if getattr(tree, "detector", None) is not None:
                    return tree.detector.deserialize(archive)
                return ComponentSentinel.INVALID_COMPONENT_MODEL
            case "aperture_corrections":
                return tree.aperture_corrections.deserialize(archive)
        return ComponentSentinel.UNRECOGNIZED_COMPONENT


class CellCoaddFormatter(MaskedImageFormatter):
    """Adds CellCoadd-specific psf and provenance components."""

    def read_component(self, component: str, tree: Any, archive: Any) -> Any:
        from .cells import CellCoaddSerializationModel  # avoid cycles

        match super().read_component(component, tree, archive):
            case ComponentSentinel():
                pass
            case handled:
                return handled
        if not isinstance(tree, CellCoaddSerializationModel):
            return ComponentSentinel.INVALID_COMPONENT_MODEL
        match component:
            case "psf":
                bbox = self.pop_bbox_from_parameters()
                return tree.deserialize_psf(archive, bbox=bbox)
            case "provenance":
                return tree.deserialize_provenance(archive)
        return ComponentSentinel.UNRECOGNIZED_COMPONENT
