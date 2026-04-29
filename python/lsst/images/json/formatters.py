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

__all__ = ("GenericFormatter",)

from typing import Any, ClassVar

from lsst.daf.butler import DatasetProvenance, FormatterV2
from lsst.resources import ResourcePath

from ..serialization import ButlerInfo
from ._input_archive import read
from ._output_archive import write


class GenericFormatter(FormatterV2):
    """The butler interface to JSON archive serialization.

    Serialized types must meet all the requirements of the `.read` and
    `.write` functions.

    Notes
    -----
    This formatter does not support any parameters or components.
    """

    default_extension: ClassVar[str] = ".json"
    can_read_from_uri: ClassVar[bool] = True
    unsupported_parameters: ClassVar[None] = None
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
        write(in_memory_dataset, uri.ospath, butler_info=butler_info)

    def add_provenance(
        self, in_memory_dataset: Any, /, *, provenance: DatasetProvenance | None = None
    ) -> Any:
        # Instead of attaching the provenance to the object we remember it on
        # the formatter, since a Formatter instance is only used once.
        self.butler_provenance = provenance
        return in_memory_dataset
