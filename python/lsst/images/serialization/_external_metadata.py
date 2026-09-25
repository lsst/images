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

__all__ = ("EmptyExternalMetadata", "ExternalMetadata", "ExternalMetadataValue")

from abc import abstractmethod
from collections.abc import Iterator, Mapping

type ExternalMetadataValue = bool | int | float | complex | str | None


class ExternalMetadata(Mapping[str, ExternalMetadataValue]):
    """Read-only access to metadata that was carried in from an external
    source, such as the primary header of a FITS file, and is not part of
    the data model.

    Notes
    -----
    Lookups and membership tests are case-insensitive.  Iteration yields
    each key once, in the form the underlying source reports it.  When a key
    is repeated in the source, which of its values ``[]`` returns is
    unspecified; use `get_all` to obtain every value.
    """

    @abstractmethod
    def get_all(self, key: str) -> tuple[ExternalMetadataValue, ...]:
        """Return every value for a key, in source order.

        Parameters
        ----------
        key
            Key to look up (case-insensitive).

        Returns
        -------
        values : `tuple`
            All values for ``key``.

        Raises
        ------
        KeyError
            Raised if ``key`` is not present.
        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"{type(self).__name__}({dict(self)!r})"


class EmptyExternalMetadata(ExternalMetadata):
    """An `ExternalMetadata` with no keys."""

    def __getitem__(self, key: str) -> ExternalMetadataValue:
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0

    def get_all(self, key: str) -> tuple[ExternalMetadataValue, ...]:
        # Docstring inherited.
        raise KeyError(key)
