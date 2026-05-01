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

__all__ = ("NdfInputArchive", "read")

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Self

import astropy.io.fits
import astropy.table
import h5py
import numpy as np

from lsst.resources import ResourcePath, ResourcePathExpression

from .._transforms import FrameSet
from ..fits._common import FitsOpaqueMetadata
from ..serialization import (
    ArchiveReadError,
    ArchiveTree,
    ArrayReferenceModel,
    InlineArrayModel,
    InputArchive,
    ReadResult,
    TableModel,
    no_header_updates,
)
from . import _hds
from ._common import NdfPointerModel

if TYPE_CHECKING:
    pass


_LOG = logging.getLogger(__name__)


class NdfInputArchive(InputArchive[NdfPointerModel]):
    """Reads HDS-on-HDF5 NDF files written by `NdfOutputArchive`.

    Instances should only be constructed via the :meth:`open` context
    manager.

    Parameters
    ----------
    file
        Open ``h5py.File`` handle. Owned by the caller of :meth:`open`;
        the archive does not close it.
    """

    def __init__(self, file: h5py.File) -> None:
        self._file = file
        self._opaque_metadata = FitsOpaqueMetadata()
        # Hooks for Tasks 13–14. The opaque-FITS reader and the array
        # / pointer / frame-set caches will be populated then.
        self._deserialized_pointer_cache: dict[str, Any] = {}
        self._frame_set_cache: dict[str, FrameSet] = {}
        self._read_opaque_fits_metadata()

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
        if "/MORE/LSST/JSON" not in self._file:
            raise ArchiveReadError(
                "File has no /MORE/LSST/JSON tree; this is either a "
                "Starlink-only NDF (use ndf.read() with auto-detect) or "
                "the file was written by an unrelated tool."
            )
        lines = _hds.read_char_array(self._file["/MORE/LSST/JSON"])
        return model_type.model_validate_json("".join(lines))

    def deserialize_pointer[U: ArchiveTree, V](
        self,
        pointer: NdfPointerModel,
        model_type: type[U],
        deserializer: Callable[[U, InputArchive[NdfPointerModel]], V],
    ) -> V:
        # Cache by pointer.ref so repeated dereferences reuse the same
        # deserialised result and don't re-run the deserializer.
        if (cached := self._deserialized_pointer_cache.get(pointer.ref)) is not None:
            return cached
        if pointer.ref not in self._file:
            raise ArchiveReadError(f"Pointer reference {pointer.ref!r} not found in NDF file.")
        dataset = self._file[pointer.ref]
        if not isinstance(dataset, h5py.Dataset):
            raise ArchiveReadError(f"Pointer reference {pointer.ref!r} is not a primitive dataset.")
        lines = _hds.read_char_array(dataset)
        json_text = "".join(lines)
        model = model_type.model_validate_json(json_text)
        result = deserializer(model, self)
        self._deserialized_pointer_cache[pointer.ref] = result
        if isinstance(result, FrameSet):
            self._frame_set_cache[pointer.ref] = result
        return result

    def get_frame_set(self, ref: NdfPointerModel) -> FrameSet:
        try:
            return self._frame_set_cache[ref.ref]
        except KeyError:
            raise AssertionError(
                f"Frame set at {ref.ref!r} must be deserialised via "
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
        if path not in self._file:
            raise ArchiveReadError(f"Array reference {path!r} not in file.")
        dataset = self._file[path]
        if not isinstance(dataset, h5py.Dataset):
            raise ArchiveReadError(f"Array reference {path!r} is not a primitive dataset.")
        # h5py supports lazy slicing via dataset[slices].
        return dataset[()] if slices is ... else dataset[slices]

    def get_table(
        self,
        model: TableModel,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> astropy.table.Table:
        # Inline-only for v1, paralleling JsonInputArchive (Task 13 may
        # promote this if any inline-vs-reference distinction matters for
        # NDF). For now: same logic as JsonInputArchive.get_table.
        result = astropy.table.Table(meta=model.meta)
        for column_model in model.columns:
            if not isinstance(column_model.data, InlineArrayModel):
                raise ArchiveReadError("Only inline tables are supported in NDF archives in v1.")
            result[column_model.name] = astropy.table.Column(
                column_model.data.data,
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
        if "/MORE/FITS" not in self._file:
            return
        dataset = self._file["/MORE/FITS"]
        if not isinstance(dataset, h5py.Dataset):
            return
        cards = _hds.read_char_array(dataset)
        # FITS Header.fromstring expects fixed-width 80-char cards
        # concatenated; pad each card defensively so readers tolerate
        # files written with shorter widths.
        header = astropy.io.fits.Header.fromstring("".join(c.ljust(80) for c in cards))
        self._opaque_metadata.add_header(header, name="", ver=1)

    def get_opaque_metadata(self) -> FitsOpaqueMetadata:
        # The opaque-FITS reader is wired up in Task 14; for v1 this just
        # returns whatever has been accumulated in __init__ (currently empty).
        return self._opaque_metadata


def read[T: Any](cls: type[T], path: ResourcePathExpression, **kwargs: Any) -> ReadResult[T]:
    """Read an object from an NDF (HDS-on-HDF5) file.

    If the file has a ``/MORE/LSST/JSON`` tree it is used as the source
    of truth and ``cls.deserialize`` is invoked with the parsed model.
    Otherwise the reader falls back to auto-detection of a minimal
    recognised-component set (``DATA_ARRAY``, ``VARIANCE``, ``QUALITY``,
    ``MORE.FITS``); ``WCS`` is logged-and-dropped in v1.

    Parameters
    ----------
    cls
        Expected return type. ``Image`` and ``MaskedImage`` are the only
        types the auto-detect path can produce. The symmetric path
        accepts whatever the file's discriminator says.
    path
        File path or ``lsst.resources.ResourcePathExpression``.
    **kwargs
        Forwarded to ``cls.deserialize`` on the symmetric read path.

    Returns
    -------
    `~lsst.images.serialization.ReadResult` [T]
        Named tuple of (deserialized object, metadata, butler_info).
    """
    with NdfInputArchive.open(path) as archive:
        if "/MORE/LSST/JSON" in archive._file:
            tree_type = cls._get_archive_tree_type(NdfPointerModel)
            tree = archive.get_tree(tree_type)
            obj = cls.deserialize(tree, archive, **kwargs)
            obj._opaque_metadata = archive.get_opaque_metadata()
            return ReadResult(obj, tree.metadata, tree.butler_info)
        return _read_auto_detect(cls, archive)


def _read_auto_detect[T: Any](cls: type[T], archive: NdfInputArchive) -> ReadResult[T]:
    """Reconstruct an ``Image`` (or ``MaskedImage``) from a Starlink NDF.

    Recognised components: ``DATA_ARRAY`` (in either simple or complex
    form), ``VARIANCE``, ``QUALITY``, ``MORE.FITS``. Other components
    (``WCS``, ``HISTORY``, ``AXIS``, ``LABEL``, custom ``MORE.*``,
    ``_LOGICAL`` primitives) are warned-and-dropped.
    """
    from .._image import Image
    from .._mask import Mask, MaskPlane, MaskSchema
    from .._masked_image import MaskedImage

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
    if "QUALITY" in ndf_group and isinstance(ndf_group["QUALITY"], h5py.Group):
        q = ndf_group["QUALITY"]
        if "QUALITY" in q and isinstance(q["QUALITY"], h5py.Dataset):
            quality_arr = _hds.read_array(q["QUALITY"])
            quality_bbox = _make_bbox(0, 0, quality_arr)
        elif "QUALITY" in q and isinstance(q["QUALITY"], h5py.Group):
            quality_arr, quality_bbox = _read_data_array_with_bbox(q["QUALITY"])

    # WCS is dropped in v1 with a warning.  The write-side companion
    # (writing /WCS/DATA from the Projection's AST FrameSet) landed in
    # DM-54817; the read-side reconstruction is a separate follow-up ticket.
    if "WCS" in ndf_group:
        _LOG.warning(
            "Starlink WCS present in %s but auto-detect ingest does not yet "
            "build a Projection from it; dropping. Round-trip writes from "
            "lsst.images.ndf preserve WCS via the Pydantic tree.",
            f.filename,
        )

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
    image = Image(data_arr, bbox=bbox)
    obj: Any
    if cls is Image:
        obj = image
    elif issubclass(cls, MaskedImage):
        schema = MaskSchema([MaskPlane(name="BAD", description="Bad pixel.")])
        mask = (
            Mask(quality_arr[:, :, np.newaxis], schema=schema, bbox=quality_bbox)
            if quality_arr is not None
            else None
        )
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
    # Auto-detect path produces no archive-tree metadata or butler_info.
    return ReadResult(obj, {}, None)


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
        bbox = _make_bbox(0, 0, array)
        return array, bbox
    # Complex form: an HDS structure with DATA + ORIGIN.
    data = _hds.read_array(obj["DATA"])
    if "ORIGIN" in obj:
        origin = _hds.read_array(obj["ORIGIN"])
        bbox = _make_bbox(int(origin[0]), int(origin[1]), data)
    else:
        bbox = _make_bbox(0, 0, data)
    return data, bbox


def _make_bbox(x_min: int, y_min: int, array: np.ndarray) -> Any:
    """Build an lsst.images.Box for a 2D image array.

    The array is C-order ``(height, width)``. NDF stores ``ORIGIN``
    in Fortran axis order ``(x_min, y_min)``.
    """
    from .._geom import Box

    if array.ndim != 2:
        raise ArchiveReadError(f"Auto-detect read only supports 2D arrays, got ndim={array.ndim}.")
    height, width = array.shape
    # Box.from_shape takes (height, width) and start=(y_start, x_start).
    return Box.from_shape((height, width), start=(y_min, x_min))
