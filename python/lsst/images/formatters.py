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

__all__ = ("GenericFormatter",)

import hashlib
import json as _stdlib_json  # disambiguates from .json subpackage
from collections.abc import Mapping
from typing import Any, ClassVar

import astropy.io.fits

from lsst.daf.butler import DatasetProvenance, FormatterV2
from lsst.resources import ResourcePath

from . import fits as _fits
from . import serialization as ser
from .serialization import ButlerInfo, write


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

    @classmethod
    def validate_write_recipes(cls, recipes: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        if not recipes:
            return recipes
        for name, recipe in recipes.items():
            try:
                _fits.FitsCompressionOptions.model_validate(recipe)
            except Exception as err:
                err.add_note(name)
                raise
        return recipes

    # --- Write path ---------------------------------------------------------

    def write_local_file(self, in_memory_dataset: Any, uri: ResourcePath) -> None:
        self._validate_write_parameters()
        ext = self.get_write_extension()
        butler_info = ButlerInfo(
            dataset=self.dataset_ref.to_simple(),
            provenance=self.butler_provenance if self.butler_provenance is not None else DatasetProvenance(),
        )
        kwargs: dict[str, Any] = {"butler_info": butler_info}
        if ext == ".fits":
            kwargs["update_header"] = self._update_header
            kwargs["compression_options"] = self._get_compression_options()
            kwargs["compression_seed"] = self._get_compression_seed()
        # The generic write() dispatches to the FITS / JSON / NDF backend by
        # the file extension, which get_write_extension has already set on uri.
        write(in_memory_dataset, uri.ospath, **kwargs)

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

    def _get_compression_options(self) -> dict[str, _fits.FitsCompressionOptions]:
        recipe = self.write_parameters.get("recipe", "default")
        try:
            config = self.write_recipes[recipe]
        except KeyError:
            if recipe == "default":
                # If there's no default recipe just use the software defaults.
                return {}
            raise RuntimeError(f"Invalid recipe for GenericFormatter: {recipe!r}.") from None
        return {k: _fits.FitsCompressionOptions.model_validate(v) for k, v in config.items()}

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

    def read_from_uri(
        self,
        uri: ResourcePath,
        component: str | None = None,
        expected_size: int = -1,
    ) -> Any:
        kwargs = self.file_descriptor.parameters or {}
        pytype: type[Any] = self.dataset_ref.datasetType.storageClass.pytype
        with ser.open(uri, cls=pytype, partial=bool(kwargs or component)) as reader:
            if component is None:
                return reader.read(**kwargs)
            return reader.get_component(component, **kwargs)
