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

__all__ = ("NdfInputArchive", "read_starlink")

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import EllipsisType
from typing import Any, Self

import astropy.io.fits
import astropy.table
import astropy.units as u
import h5py
import numpy as np

from lsst.resources import ResourcePath, ResourcePathExpression

from .._geom import Box
from .._image import Image
from .._mask import Mask, MaskPlane, MaskSchema
from .._masked_image import MaskedImage
from .._transforms import FrameSet, SkyProjection
from .._transforms import _ast as astshim
from .._transforms._frames import GeneralFrame
from ..fits._common import FitsOpaqueMetadata
from ..serialization import (
    ArchiveInfo,
    ArchiveReadError,
    ArchiveTree,
    ArrayReferenceModel,
    InlineArrayModel,
    InputArchive,
    TableModel,
    no_header_updates,
    parameterize_tree,
    tree_class_for_info,
)
from ..serialization._common import _check_format_version
from . import _hds
from ._common import NdfPointerModel
from ._model import HdsPrimitive, NdfDocument

_LOG = logging.getLogger(__name__)

_NDF_FORMAT_VERSION = 1
"""Container layout version this release of `NdfInputArchive` understands."""


class NdfInputArchive(InputArchive[NdfPointerModel]):
    """Reads HDS-on-HDF5 NDF files written by `NdfOutputArchive`.

    Instances should only be constructed via the :meth:`open` context
    manager.

    Parameters
    ----------
    file
        Open `h5py.File` handle. Owned by the caller of :meth:`open`;
        the archive does not close it.
    """

    def __init__(self, file: h5py.File) -> None:
        self._file = file
        self._document = NdfDocument.from_hdf5(file)
        self._opaque_metadata = FitsOpaqueMetadata()
        self._deserialized_pointer_cache: dict[str, Any] = {}
        self._frame_set_cache: dict[str, FrameSet] = {}
        self._read_opaque_fits_metadata()
        self._check_format_version()

    @classmethod
    def get_basic_info(cls, path: ResourcePathExpression) -> ArchiveInfo:
        """Read the schema URL from the ``DATA_MODEL`` scalar and the
        ``FORMAT_VERSION`` primitive without deserializing pixel data.

        Both live at the fixed location ``/MORE/LSST`` (or ``/LSST``); we read
        only those and never search at arbitrary depth, so nested pointer
        trees cannot be mistaken for the top level.  Reading ``DATA_MODEL``
        directly avoids parsing the (potentially large) JSON tree.
        """
        ospath = ResourcePath(path).ospath
        schema_url: str | None = None
        format_version = 1

        with h5py.File(ospath, "r") as handle:
            for prefix in ("MORE/LSST", "LSST"):
                data_model = handle.get(f"{prefix}/DATA_MODEL")
                if not isinstance(data_model, h5py.Dataset):
                    continue
                schema_url = np.asarray(data_model).tobytes().decode("ascii").rstrip("\x00").strip()
                fmt_node = handle.get(f"{prefix}/FORMAT_VERSION")
                if fmt_node is not None:
                    format_version = int(np.asarray(fmt_node).item())
                break
        if not schema_url:
            raise ArchiveReadError(
                f"Could not read the schema of {path!r} from /MORE/LSST/DATA_MODEL or /LSST/DATA_MODEL."
            )
        return ArchiveInfo.from_schema_url(schema_url, format_version=format_version)

    @classmethod
    @contextmanager
    def open_tree(
        cls,
        path: ResourcePathExpression,
        *,
        partial: bool = True,
        **backend_kwargs: Any,
    ) -> Iterator[tuple[Self, ArchiveTree, ArchiveInfo]]:
        """Open the NDF file and yield ``(archive, tree, info)``.

        The schema is read from the open document's ``DATA_MODEL`` rather than
        a separate `get_basic_info` open.  Requires the symmetric LSST JSON
        tree; ``partial`` is accepted but not meaningful, since h5py reads
        lazily regardless.
        """
        with cls.open(path) as archive:
            if archive._get_main_json_path() is None:
                raise ArchiveReadError(
                    f"{path!r} has no LSST JSON tree; only the symmetric read path is supported."
                )
            info = archive.info
            tree_cls = tree_class_for_info(info, path)
            parameterized = parameterize_tree(tree_cls, NdfPointerModel)
            tree = archive.get_tree(parameterized)
            yield archive, tree, info

    @classmethod
    @contextmanager
    def open(cls, path: ResourcePathExpression) -> Iterator[Self]:
        """Open an NDF file for reading and yield an `NdfInputArchive`.

        Remote ResourcePaths are materialised locally first; fsspec-direct
        h5py reads are a deferred follow-up.
        """
        rp = ResourcePath(path)
        with rp.as_local() as local:
            with h5py.File(local.ospath, "r") as f:
                yield cls(f)

    def get_tree[T: ArchiveTree](self, model_type: type[T]) -> T:
        """Read and validate the main Pydantic tree at ``/MORE/LSST/JSON``."""
        json_path = self._get_main_json_path()
        if json_path is None:
            raise ArchiveReadError(
                "File has no /MORE/LSST/JSON tree; this is either a "
                "Starlink-only NDF (use ndf.read_starlink() for auto-detect) or "
                "the file was written by an unrelated tool."
            )
        json_text = _read_json_record(self._get_primitive(json_path), json_path)
        return model_type.model_validate_json(json_text)

    def deserialize_pointer[U: ArchiveTree, V](
        self,
        pointer: NdfPointerModel,
        model_type: type[U],
        deserializer: Callable[[U, InputArchive[NdfPointerModel]], V],
    ) -> V:
        # Cache by pointer.path so repeated dereferences reuse the same
        # deserialised result and don't re-run the deserializer.
        if (cached := self._deserialized_pointer_cache.get(pointer.path)) is not None:
            return cached
        if not self._has_model_path(pointer.path):
            raise ArchiveReadError(f"Pointer reference {pointer.path!r} not found in NDF file.")
        primitive = self._get_primitive(pointer.path)
        json_text = _read_json_record(primitive, pointer.path)
        model = model_type.model_validate_json(json_text)
        result = deserializer(model, self)
        self._deserialized_pointer_cache[pointer.path] = result
        if isinstance(result, FrameSet):
            self._frame_set_cache[pointer.path] = result
        return result

    def get_frame_set(self, pointer: NdfPointerModel) -> FrameSet:
        try:
            return self._frame_set_cache[pointer.path]
        except KeyError:
            raise AssertionError(
                f"Frame set at {pointer.path!r} must be deserialised via "
                f"deserialize_pointer before any dependent transform can be."
            ) from None

    def get_array(
        self,
        model: ArrayReferenceModel | InlineArrayModel,
        *,
        slices: tuple[slice, ...] | EllipsisType = ...,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> np.ndarray:
        if isinstance(model, InlineArrayModel):
            data: np.ndarray = np.array(model.data, dtype=model.datatype.to_numpy())
            return data if slices is ... else data[slices]
        if not isinstance(model.source, str) or not model.source.startswith("ndf:"):
            raise ArchiveReadError(
                f"NdfInputArchive cannot resolve array source {model.source!r}; "
                f"expected an 'ndf:<HDF5-path>' reference."
            )
        path = model.source[len("ndf:") :]
        if not self._has_model_path(path):
            raise ArchiveReadError(f"Array reference {path!r} not in file.")
        primitive = self._get_primitive(path)
        # h5py supports lazy slicing via dataset[slices].
        if isinstance(primitive.data, h5py.Dataset):
            return primitive.data[()] if slices is ... else primitive.data[slices]
        data = primitive.read_array()
        return data if slices is ... else data[slices]

    def get_table(
        self,
        model: TableModel,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> astropy.table.Table:
        result = astropy.table.Table(meta=model.meta)
        for column_model in model.columns:
            if isinstance(column_model.data, InlineArrayModel):
                data: Any = column_model.data.data
            else:
                data = self.get_array(column_model.data, strip_header=strip_header)
            result[column_model.name] = astropy.table.Column(
                data,
                name=column_model.name,
                dtype=column_model.data.datatype.to_numpy(),
                unit=column_model.unit,
                description=column_model.description,
                meta=column_model.meta,
            )
        return result

    def get_structured_array(
        self,
        model: TableModel,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> np.ndarray:
        return self.get_table(model, strip_header).as_array()

    def _read_opaque_fits_metadata(self) -> None:
        if not self._has_model_path("/MORE/FITS"):
            return
        cards = self._get_primitive("/MORE/FITS").read_char_array()
        # FITS Header.fromstring expects fixed-width 80-char cards
        # concatenated; pad each card defensively so readers tolerate
        # files written with shorter widths.
        header = astropy.io.fits.Header.fromstring("".join(c.ljust(80) for c in cards))
        self._opaque_metadata.add_header(header, name="", ver=1)

    def get_opaque_metadata(self) -> FitsOpaqueMetadata:
        return self._opaque_metadata

    @property
    def info(self) -> ArchiveInfo:
        """Schema/format info read from the open document's ``DATA_MODEL``."""
        schema_url: str | None = None
        format_version = 1
        for prefix in ("/MORE/LSST", "/LSST"):
            if self._has_model_path(f"{prefix}/DATA_MODEL"):
                lines = self._get_primitive(f"{prefix}/DATA_MODEL").read_char_array()
                schema_url = lines[0].strip() if lines else None
                if self._has_model_path(f"{prefix}/FORMAT_VERSION"):
                    format_version = int(self._get_primitive(f"{prefix}/FORMAT_VERSION").read_array().item())
                break
        if not schema_url:
            raise ArchiveReadError(
                "Could not read the schema from /MORE/LSST/DATA_MODEL or /LSST/DATA_MODEL."
            )
        return ArchiveInfo.from_schema_url(schema_url, format_version=format_version)

    def _get_main_json_path(self) -> str | None:
        """Return the path of the main LSST JSON tree, if present."""
        for path in ("/MORE/LSST/JSON", "/LSST/JSON"):
            if self._has_model_path(path):
                return path
        return None

    def _check_format_version(self) -> None:
        """Read FORMAT_VERSION from the NDF top-level structure and check it.

        Absence is treated as ``1`` (legacy default). DATA_MODEL is
        informational only on read; the JSON tree's ``schema_version`` /
        ``min_read_version`` drive data-model compatibility.
        """
        on_disk = 1
        for prefix in ("/MORE/LSST", "/LSST"):
            path = f"{prefix}/FORMAT_VERSION"
            if self._has_model_path(path):
                primitive = self._get_primitive(path)
                # We wrote the version as a 0-d int32 numpy array; .item()
                # unwraps to a Python int.
                on_disk = int(primitive.read_array().item())
                break
        _check_format_version("ndf", on_disk, _NDF_FORMAT_VERSION)

    def _has_model_path(self, path: str) -> bool:
        """Return `True` if a path exists in the NDF document model."""
        try:
            self._document.get(path)
        except KeyError:
            return False
        return True

    def _get_primitive(self, path: str) -> HdsPrimitive:
        """Return a primitive component from the NDF document model."""
        node = self._document.get(path)
        if not isinstance(node, HdsPrimitive):
            raise ArchiveReadError(f"NDF reference {path!r} is not a primitive dataset.")
        return node


def read_starlink[T: Any](cls: type[T], path: ResourcePathExpression) -> T:
    """Reconstruct an `~lsst.images.Image` or `~lsst.images.MaskedImage`
    from a schema-less Starlink NDF.

    Files written by this package carry a ``/MORE/LSST/JSON`` tree and are
    read through the generic `lsst.images.serialization.read` /
    `lsst.images.serialization.open`.  A Starlink-produced NDF has no such
    tree and therefore no schema, so it cannot go through that path; this
    function auto-detects a minimal recognised-component set
    (``DATA_ARRAY``, ``VARIANCE``, ``QUALITY``, ``MORE.FITS``) instead.
    ``WCS`` is reconstructed when possible; other components are
    logged-and-dropped.

    Parameters
    ----------
    cls
        Expected return type; `~lsst.images.Image` and
        `~lsst.images.MaskedImage` are the only types the auto-detect path
        can produce.
    path
        File path or `lsst.resources.ResourcePathExpression`.

    Returns
    -------
    object
        The deserialized ``cls`` instance.

    Raises
    ------
    ArchiveReadError
        If the file has an LSST JSON tree (use the generic ``read`` instead)
        or no recognised ``DATA_ARRAY`` component.
    """
    with NdfInputArchive.open(path) as archive:
        if archive._get_main_json_path() is not None:
            raise ArchiveReadError(
                f"{path!r} has an LSST JSON tree; read it with serialization.read()/open()."
            )
        return _read_auto_detect(cls, archive)


def _read_auto_detect[T: Any](cls: type[T], archive: NdfInputArchive) -> T:
    """Reconstruct an `Image` (or `MaskedImage`) from a Starlink NDF.

    Recognised components: ``DATA_ARRAY`` (in either simple or complex
    form), ``VARIANCE``, ``QUALITY``, ``MORE.FITS``. Other components
    (``WCS``, ``HISTORY``, ``AXIS``, ``LABEL``, custom ``MORE.*``,
    ``_LOGICAL`` primitives) are warned-and-dropped.
    """
    f = archive._file
    ndf_group = _locate_ndf_root(f)

    # DATA_ARRAY is required.
    if "DATA_ARRAY" not in ndf_group:
        raise ArchiveReadError(f"Auto-detect read of {f.filename!r}: no DATA_ARRAY component.")
    data_arr, bbox = _read_data_array_with_bbox(ndf_group["DATA_ARRAY"])

    # VARIANCE / QUALITY are optional.
    variance_arr: np.ndarray | None = None
    variance_bbox: Any | None = None
    if "VARIANCE" in ndf_group:
        variance_arr, variance_bbox = _read_data_array_with_bbox(ndf_group["VARIANCE"])
    quality_arr: np.ndarray | None = None
    quality_bbox: Any | None = None
    quality_badbits = 255
    if "QUALITY" in ndf_group and isinstance(ndf_group["QUALITY"], h5py.Group):
        q = ndf_group["QUALITY"]
        quality_badbits = _read_quality_badbits(q)
        if "QUALITY" in q and isinstance(q["QUALITY"], h5py.Dataset):
            quality_arr = _validate_quality_array(_hds.read_array(q["QUALITY"]))
            quality_bbox = _make_bbox(x_min=0, y_min=0, array=quality_arr)
        elif "QUALITY" in q and isinstance(q["QUALITY"], h5py.Group):
            quality_arr, quality_bbox = _read_data_array_with_bbox(q["QUALITY"])
            quality_arr = _validate_quality_array(quality_arr)

    sky_projection: SkyProjection | None = None
    if "WCS" in ndf_group:
        try:
            wcs_group = ndf_group["WCS"]
            if isinstance(wcs_group, h5py.Group) and "DATA" in wcs_group:
                wcs_lines = _hds.read_char_array(wcs_group["DATA"])
                wcs_text = _hds.decode_ndf_ast_data(wcs_lines)
                ast_obj = astshim.Object.fromString(wcs_text)
                if isinstance(ast_obj, astshim.FrameSet):
                    pixel_frame = GeneralFrame(unit=u.pix)
                    sky_projection = SkyProjection.from_ast_frame_set(
                        ast_obj,
                        pixel_frame,
                        pixel_bounds=bbox,
                    )
        except Exception:
            _LOG.warning(
                "Could not reconstruct Projection from WCS in %s; dropping.",
                f.filename,
                exc_info=True,
            )

    unit = _read_ndf_units(ndf_group)

    # Anything unrecognised: warn-and-drop.
    recognised = {
        "DATA_ARRAY",
        "VARIANCE",
        "QUALITY",
        "WCS",
        "MORE",
        "TITLE",
        "LABEL",
        "UNITS",
        "HISTORY",
        "AXIS",
    }
    for name in ndf_group:
        if name not in recognised:
            _LOG.warning(
                "Ignoring unrecognised NDF component %s/%s during auto-detect read.",
                ndf_group.name,
                name,
            )

    # Build the requested in-memory object. Any NDF can be read as an Image;
    # MaskedImage construction uses whatever VARIANCE/QUALITY are present and
    # lets the MaskedImage constructor provide defaults for missing planes.
    image = Image(data_arr, bbox=bbox, unit=unit, sky_projection=sky_projection)
    obj: Any
    if cls is Image:
        obj = image
    elif issubclass(cls, MaskedImage):
        if quality_arr is not None:
            schema = _make_quality_mask_schema(quality_badbits)
            mask = Mask(quality_arr[:, :, np.newaxis], schema=schema, bbox=quality_bbox)
        else:
            schema = MaskSchema([MaskPlane(name="BAD", description="Bad pixel.")])
            mask = None
        variance = Image(variance_arr, bbox=variance_bbox) if variance_arr is not None else None
        obj = cls(
            image=image,
            mask=mask,
            mask_schema=schema if mask is None else None,
            variance=variance,
        )
    else:
        raise ArchiveReadError(
            f"Auto-detect can produce Image or MaskedImage, but caller asked for {cls.__name__}."
        )
    obj._opaque_metadata = archive.get_opaque_metadata()
    return obj


def _read_ndf_units(ndf_group: h5py.Group) -> u.UnitBase | None:
    """Read the NDF UNITS component, if present."""
    if "UNITS" not in ndf_group or not isinstance(ndf_group["UNITS"], h5py.Dataset):
        return None
    dataset = ndf_group["UNITS"]
    if dataset.dtype.kind != "S":
        _LOG.warning("Ignoring non-character NDF UNITS component in %s.", ndf_group.name)
        return None
    if dataset.ndim == 0:
        raw = dataset[()]
        if isinstance(raw, np.bytes_):
            raw = bytes(raw)
        if not isinstance(raw, bytes):
            return None
        units_text = raw.decode("ascii").rstrip(" ")
    else:
        records = _hds.read_char_array(dataset)
        units_text = records[0] if records else ""
    if not units_text:
        return None
    for kwargs in ({"format": "fits"}, {}):
        try:
            return u.Unit(units_text, **kwargs)
        except ValueError:
            continue
    _LOG.warning("Could not parse NDF UNITS value %r in %s.", units_text, ndf_group.name)
    return None


def _read_quality_badbits(quality_group: h5py.Group) -> int:
    """Read the scalar NDF QUALITY.BADBITS value."""
    badbits = quality_group.get("BADBITS")
    if not isinstance(badbits, h5py.Dataset):
        return 255
    value = np.asarray(_hds.read_array(badbits)).reshape(-1)
    if value.size == 0:
        return 255
    return int(value[0])


def _validate_quality_array(quality: np.ndarray) -> np.ndarray:
    """Return an NDF QUALITY array as a `numpy.uint8` mask plane."""
    if quality.dtype != np.dtype(np.uint8):
        raise ArchiveReadError(f"NDF QUALITY array has dtype {quality.dtype}; expected uint8.")
    return quality


def _make_quality_mask_schema(badbits: int) -> MaskSchema:
    """Create a fallback `MaskSchema` for an unnamed 8-bit QUALITY array."""
    planes = []
    for bit in range(8):
        mask = 1 << bit
        description = f"NDF QUALITY bit {bit}."
        if badbits & mask:
            description += " Selected by BADBITS."
        planes.append(MaskPlane(name=f"MASK{bit}", description=description))
    return MaskSchema(planes, dtype=np.uint8)


def _locate_ndf_root(f: h5py.File) -> h5py.Group:
    """Return the group representing the top-level NDF.

    Most files have the NDF at the root group itself. A few wrap it
    in a single-child container at the root; we accept that shape
    too. Anything more elaborate raises.
    """
    root_class = f["/"].attrs.get(_hds.ATTR_CLASS)
    if isinstance(root_class, bytes):
        root_class = root_class.decode("ascii")
    if root_class == "NDF":
        return f["/"]
    # Maybe a one-level container.
    candidates = []
    for name, child in f["/"].items():
        if isinstance(child, h5py.Group):
            cls_attr = child.attrs.get(_hds.ATTR_CLASS)
            if isinstance(cls_attr, bytes):
                cls_attr = cls_attr.decode("ascii")
            if cls_attr == "NDF":
                candidates.append(name)
    if len(candidates) == 1:
        return f[candidates[0]]
    raise ArchiveReadError(
        f"Could not locate top-level NDF in {f.filename!r}; "
        f"expected the root group or a single NDF-typed child."
    )


def _read_data_array_with_bbox(
    obj: h5py.Group | h5py.Dataset,
) -> tuple[np.ndarray, Any]:
    """Read a DATA_ARRAY component in either simple or complex form.

    The complex form (what our writer always produces) is an HDS
    ARRAY structure (h5py group with CLASS="ARRAY") containing
    ``DATA`` and ``ORIGIN`` primitives. The simple form is a bare
    primitive dataset.

    Returns
    -------
    array, bbox : tuple
        ``array`` is the C-order numpy data (shape ``(height, width)``
        for 2D images). ``bbox`` is constructed from the ORIGIN if
        present, else from a default origin of (0, 0).
    """
    if isinstance(obj, h5py.Dataset):
        # Simple form.
        array = _hds.read_array(obj)
        bbox = _make_bbox(x_min=0, y_min=0, array=array)
        return array, bbox
    # Complex form: an HDS structure with DATA + ORIGIN.
    data = _hds.read_array(obj["DATA"])
    if "ORIGIN" in obj:
        origin = _hds.read_array(obj["ORIGIN"])
        bbox = _make_bbox(x_min=int(origin[0]), y_min=int(origin[1]), array=data)
    else:
        bbox = _make_bbox(x_min=0, y_min=0, array=data)
    return data, bbox


def _read_json_record(primitive: HdsPrimitive, path: str) -> str:
    """Read a JSON document stored as a single _CHAR*N record.

    Our writer always emits JSON trees as a single-element character
    array sized to the document. Joining multiple records would lose
    trailing whitespace inside JSON string values, since
    `read_char_array` strips trailing spaces per record.
    """
    records = primitive.read_char_array()
    if len(records) != 1:
        raise ArchiveReadError(f"Expected a single _CHAR*N record at {path!r}, got {len(records)}.")
    return records[0]


def _make_bbox(*, x_min: int, y_min: int, array: np.ndarray) -> Any:
    """Build an lsst.images.Box for a 2D image array.

    The array is C-order ``(height, width)``. NDF stores ``ORIGIN``
    in Fortran axis order ``(x_min, y_min)``.
    """
    if array.ndim != 2:
        raise ArchiveReadError(f"Auto-detect read only supports 2D arrays, got ndim={array.ndim}.")
    # Box.from_shape takes (height, width) and start=(y_start, x_start).
    return Box.from_shape(array.shape, start=(y_min, x_min))
