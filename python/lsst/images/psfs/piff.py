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

__all__ = ("PiffSerializationModel", "PiffWrapper")

import operator
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cached_property
from logging import getLogger
from typing import TYPE_CHECKING, Annotated, Any, Literal

import astropy.io.fits
import numpy as np
import pydantic

from .._geom import Box, Domain, SerializableDomain
from .._image import Image
from ..archive import ArchiveReadError, InputArchive, OutputArchive
from ..tables import TableModel
from ._base import PointSpreadFunction

if TYPE_CHECKING:
    import galsim.wcs
    import piff.config


_LOG = getLogger(__name__)


class PiffWrapper(PointSpreadFunction):
    """A PSF backed by the Piff library."""

    def __init__(self, impl: Any, domain: Domain, stamp_size: int):
        self._impl = impl
        self._domain = domain
        self._stamp_size = stamp_size

    @property
    def domain(self) -> Domain:
        return self._domain

    @cached_property
    def kernel_bbox(self) -> Box:
        r = self._stamp_size // 2
        return Box.factory[-r : r + 1, -r : r + 1]

    def compute_kernel_image(self, *, x: float, y: float) -> Image:
        if "colorValue" in self._impl.interp_property_names:
            raise NotImplementedError("Chromatic PSFs are not yet supported.")
        gs_image = self._impl.draw(x, y, stamp_size=self._stamp_size, center=True)
        r = self._stamp_size // 2
        result = Image(gs_image.array.copy(), start=(-r, -r))
        result.array /= np.sum(result.array)
        return result

    def compute_stellar_image(self, *, x: float, y: float) -> Image:
        if "colorValue" in self._impl.interp_property_names:
            raise NotImplementedError("Chromatic PSFs are not yet supported.")
        gs_image = self._impl.draw(x, y, stamp_size=self._stamp_size, center=None)
        r = self._stamp_size // 2
        result = Image(gs_image.array.copy(), start=(round(y) - r, round(x) - r))
        result.array /= np.sum(result.array)
        return result

    def compute_stellar_bbox(self, *, x: float, y: float) -> Box:
        r = self._stamp_size // 2
        xi = round(x)
        yi = round(y)
        return Box.factory[yi - r : yi + r + 1, xi - r : xi + r + 1]

    @property
    def piff_psf(self) -> Any:
        """The backing `piff.PSF` object.

        This is an internal object that must not be modified in place.
        """
        return self._impl

    @classmethod
    def from_legacy(cls, legacy_psf: Any, domain: Domain) -> PiffWrapper:
        return cls(impl=legacy_psf._piffResult, domain=domain, stamp_size=legacy_psf.width)

    def serialize(self, archive: OutputArchive[Any]) -> PiffSerializationModel:
        """Serialize the PSF to an archive.

        This method is intended to be usable as the callback function passed to
        `..archives.OutputArchive.serialize_direct` or
        `..archives.OutputArchive.serialize_pointer`.
        """
        from piff.config import LoggerWrapper

        writer = ArchivePiffWriter()
        with self._without_stars():
            self._impl._write(writer, "piff", LoggerWrapper(_LOG))
        piff_model = writer.serialize(archive)
        return PiffSerializationModel(
            piff=piff_model,
            stamp_size=self._stamp_size,
            domain=self._domain.serialize(),
        )

    @classmethod
    def deserialize(cls, model: PiffSerializationModel, archive: InputArchive[Any]) -> PiffWrapper:
        """Deserialize the PSF from an archive.

        This method is intended to be usable as the callback function passed to
        `..archives.InputArchive.deserialize_pointer`.
        """
        from piff import PSF
        from piff.config import LoggerWrapper

        reader = ArchivePiffReader(model.piff, archive)
        impl = PSF._read(reader, "piff", LoggerWrapper(_LOG))
        return cls(impl, domain=Domain.deserialize(model.domain), stamp_size=model.stamp_size)

    @contextmanager
    def _without_stars(self) -> Iterator[None]:
        """Temporarily drop the embedded list of stars used to fit the PSF.

        Notes
        -----
        By default Piff saves the list of stars (including postage stamps) used
        to fit the PSF, which makes the serialized form much larger.  But the
        upstream Piff serialization code recognizes the case where that
        ``stars`` attribute has been deleted and serializes everything else.

        Unfortunately, to date, Rubin's pickle-based Piff serialization instead
        just deleted the postage stamp image attributes from inside Piff
        ``stars`` list, which is not a state the Piff serialization code
        handles gracefully.  So for now we have to drop the full stars list
        during serialization if it is present.
        """
        if hasattr(self._impl, "stars"):
            stars = self._impl.stars
            try:
                del self._impl.stars
                yield
            finally:
                self._impl.stars = stars
        else:
            yield


# Piff serialization uses a lot of dictionaries and lists restricted to these
# basic types.
type PiffScalar = int | float | str | bool | None
type PiffDict = dict[str, PiffScalar | list[PiffScalar]]


class GalSimPixelScaleModel(pydantic.BaseModel):
    """Model used to serialize `galsim.wcs.PixelScale` instances."""

    scale: float
    wcs_type: Literal["pixel_scale"] = "pixel_scale"


# We expect this discriminated union to grow to include other trivial
# pixel-to-pixel transforms that get embedded in PSFs.  If we someday have to
# store Piff objects that embed more sophisticated PSFs, we'll hook them into
# the AST-based coordinate transform system instead, but as long as we're just
# talking about simple offsets and scalings, that's a lot of extra complexity
# for very little gain.
type GalSimLocalWcsModel = Annotated[GalSimPixelScaleModel, pydantic.Field(discriminator="wcs_type")]


class PiffTableModel(pydantic.BaseModel):
    """Model used to embed a reference to a binary-data table in a Piff
    serialization's JSON-like data.
    """

    metadata: PiffDict
    table: TableModel


class PiffObjectModel(pydantic.BaseModel):
    """General-purpose model used to serialize various Piff objects."""

    structs: dict[str, PiffDict] = pydantic.Field(default_factory=dict, exclude_if=operator.not_)
    tables: dict[str, PiffTableModel] = pydantic.Field(default_factory=dict, exclude_if=operator.not_)
    wcs: dict[str, GalSimLocalWcsModel] = pydantic.Field(default_factory=dict, exclude_if=operator.not_)
    objects: dict[str, PiffObjectModel] = pydantic.Field(default_factory=dict, exclude_if=operator.not_)


class PiffSerializationModel(pydantic.BaseModel):
    """Model that represents a serialized Piff PSF."""

    piff: PiffObjectModel = pydantic.Field(description="The Piff PSF object itself.")

    stamp_size: int = pydantic.Field(
        description="Width of the (square) images returned by this PSF's methods."
    )

    domain: SerializableDomain = pydantic.Field(
        description="The domain object that represents the PSF's validity region."
    )


class ArchivePiffWriter:
    """An adapter from the Piff serialization interface to the
    `..archives.OutputArchive` class.

    Notes
    -----
    Piff has its own simple serialization framework (contributed upstream by
    Rubin DM) that maps everything to dictionaries, structured numpy arrays,
    and a library of GalSim WCS objects, with the native implementation writing
    standalone FITS files.  That mostly maps nicely to the `lsst.images`
    archive system, but we don't get to leverage any Pydantic validation or
    JSON schema functionality since we only get opaque dictionaries from Piff.

    See `piff.FitsWriter` for most method documentation; this class is designed
    to mimic it exactly (the Piff authors prefer to just use duck-typing rather
    than ABCs or protocols for interface definition).
    """

    def __init__(self, base_name: str = ""):
        self._base_name = base_name
        self.structs: dict[str, PiffDict] = {}
        self.tables: dict[str, tuple[np.ndarray, PiffDict]] = {}
        self.wcs_models: dict[str, GalSimLocalWcsModel] = {}
        self.writers: dict[str, ArchivePiffWriter] = {}

    def write_struct(self, name: str, struct: PiffDict) -> None:
        self.structs[name] = {k: self._to_builtin(v) for k, v in struct.items()}

    def write_table(self, name: str, array: np.ndarray, metadata: PiffDict | None = None) -> None:
        self.tables[name] = (array, metadata or {})

    def write_wcs_map(
        self, name: str, wcs_map: dict[int, galsim.wcs.BaseWCS], pointing: galsim.CelestialCoord | None
    ) -> None:
        import galsim.wcs

        match wcs_map:
            case {0: galsim.wcs.PixelScale() as wcs} if pointing is None:
                self.wcs_models[name] = GalSimPixelScaleModel(scale=wcs.scale)
            case _:
                raise NotImplementedError("PSFs with complex embedded WCSs are not supported.")

    @contextmanager
    def nested(self, name: str) -> Iterator[ArchivePiffWriter]:
        nested = ArchivePiffWriter(self.get_full_name(name))
        yield nested
        self.writers[name] = nested

    def get_full_name(self, name: str) -> str:
        return f"{self._base_name}/{name}"

    def serialize(self, archive: OutputArchive[Any]) -> PiffObjectModel:
        """Serialize to an archive.

        This method is intended to be used as the callable passed to
        `..archives.OutputArchive.serialize_direct` and
        `..archives.OutputArchive.serialize_pointer`, after first passing this
        writer to a Piff object's ``write`` or ``_write`` method.
        """
        model = PiffObjectModel()
        for name, struct in self.structs.items():
            model.structs[name] = struct
        for name, (array, metadata) in self.tables.items():
            model.tables[name] = PiffTableModel(
                metadata=metadata,
                table=archive.add_structured_array(
                    name, array, update_header=lambda header: header.update(metadata)
                ),
            )
        for name, wcs_model in self.wcs_models.items():
            model.wcs[name] = wcs_model
        for name, writer in self.writers.items():
            model.objects[name] = archive.serialize_direct(name, writer.serialize)
        return model

    @staticmethod
    def _to_builtin(val: Any) -> PiffScalar:
        match val:
            case np.integer():
                return int(val)
            case np.floating():
                return float(val)
            case np.str_():
                return str(val)
        return val


class ArchivePiffReader:
    """An adapter from the Piff serialization interface to the
    `..archives.InputArchive` class.

    See `ArchivePiffWriter` for additional notes.
    """

    def __init__(self, object_model: PiffObjectModel, archive: InputArchive[Any], base_name: str = ""):
        self._model = object_model
        self._archive = archive
        self._base_name = base_name

    def read_struct(self, name: str) -> PiffDict | None:
        return self._model.structs.get(name)

    def read_table(self, name: str, metadata: PiffDict | None = None) -> np.ndarray | None:
        table_model = self._model.tables.get(name)
        if table_model is None:
            return None
        if metadata is not None:
            metadata.update(table_model.metadata)
        return self._archive.get_structured_array(
            table_model.table, strip_header=astropy.io.fits.Header.clear
        )

    def read_wcs_map(
        self, name: str, logger: piff.config.LoggerWrapper
    ) -> tuple[dict[int, galsim.wcs.BaseWCS] | None, galsim.CelestialCoord | None]:
        import galsim.wcs

        match self._model.wcs.get(name):
            case GalSimPixelScaleModel(scale=scale):
                return {0: galsim.wcs.PixelScale(scale)}, None
            case None:
                return None, None
            case unexpected:
                raise ArchiveReadError(
                    f"{self.get_full_name(name)} should be a WCS or WCS map, not {unexpected!r}."
                )

    @contextmanager
    def nested(self, name: str) -> Iterator[ArchivePiffReader]:
        nested_model = self._model.objects[name]
        yield ArchivePiffReader(nested_model, self._archive, self.get_full_name(name))

    def get_full_name(self, name: str) -> str:
        return f"{self._base_name}/{name}"
