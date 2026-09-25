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

__all__ = ("MetadataView", "NativeMetadata")

from collections import ChainMap
from collections.abc import Iterator, Mapping, MutableMapping
from typing import Any, NoReturn, cast

from .serialization import ExternalMetadata, ExternalMetadataValue, MetadataValue


class NativeMetadata(MutableMapping[str, MetadataValue]):
    """The flexible metadata that is part of an image's data model.

    Parameters
    ----------
    data
        Dictionary holding the metadata.  It is held by reference, so changes
        are visible to every object sharing it.
    external
        External metadata that new keys may not shadow.

    Notes
    -----
    Keys are case-sensitive.  Adding a key that case-insensitively matches a
    key of ``external`` raises `KeyError`; a key that is already present may
    always be updated.
    """

    def __init__(self, data: dict[str, MetadataValue], external: ExternalMetadata) -> None:
        self._data = data
        self._external = external

    def __getitem__(self, key: str) -> MetadataValue:
        return self._data[key]

    def __setitem__(self, key: str, value: MetadataValue) -> None:
        if key not in self._data and key in self._external:
            raise KeyError(
                f"Metadata key {key!r} would shadow a key in the external metadata "
                "(external keys are case-insensitive)."
            )
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"NativeMetadata({self._data!r})"


class MetadataView(ChainMap[str, MetadataValue | ExternalMetadataValue]):
    """Combined view of an image's native and external metadata.

    Parameters
    ----------
    native
        Metadata that is part of the image's data model.
    external
        Read-only metadata carried in from an external source.

    Notes
    -----
    Lookups check ``native`` first, matching case exactly, and then
    ``external``, ignoring case.  Writes and deletes go to ``native`` only,
    and adding a new key that case-insensitively matches an ``external`` key
    raises `KeyError`.  Iteration yields the exact-case union of the keys of
    both.

    When a key is repeated in the external source, which of its values
    ``[]`` returns is unspecified; use `get_all` to obtain every value.
    """

    def __init__(self, native: NativeMetadata, external: ExternalMetadata) -> None:
        # ChainMap only ever writes to its first mapping, so the read-only
        # external mapping is never mutated through it.
        super().__init__(
            cast(MutableMapping[str, MetadataValue | ExternalMetadataValue], native),
            cast(MutableMapping[str, MetadataValue | ExternalMetadataValue], external),
        )
        self._native = native
        self._external = external

    @property
    def native(self) -> NativeMetadata:
        """Metadata that is part of the image's data model and is saved with
        it (`NativeMetadata`).
        """
        return self._native

    @property
    def external(self) -> ExternalMetadata:
        """Read-only metadata carried in from an external source, such as
        the primary header of a FITS file (`.serialization.ExternalMetadata`).
        """
        return self._external

    def get_all(self, key: str) -> tuple[MetadataValue | ExternalMetadataValue, ...]:
        """Return every value for a key.

        Parameters
        ----------
        key
            Key to look up.

        Returns
        -------
        values : `tuple`
            A single-element tuple holding the native value if ``key`` is a
            native key, and otherwise every external value for ``key``, in
            source order.

        Raises
        ------
        KeyError
            Raised if ``key`` is not present.
        """
        if key in self._native:
            return (self._native[key],)
        return self._external.get_all(key)

    def copy(self) -> dict[str, MetadataValue]:  # type: ignore[override]
        """Return a plain `dict` copy of the native metadata.

        Returns
        -------
        metadata : `dict`
            A copy of the native metadata.
        """
        return dict(self._native)

    __copy__ = copy  # type: ignore[assignment]

    def __or__(self, other: Mapping[str, Any]) -> dict[str, Any]:  # type: ignore[override]
        return {**self, **other}

    def __ror__(self, other: Mapping[str, Any]) -> dict[str, Any]:  # type: ignore[override]
        return {**other, **self}

    def new_child(self, m: Any = None, **kwargs: Any) -> NoReturn:
        """Not supported.

        Parameters
        ----------
        m
            Ignored.
        **kwargs
            Ignored.

        Raises
        ------
        TypeError
            Always raised.
        """
        raise TypeError("MetadataView does not support new_child.")

    @property
    def parents(self) -> NoReturn:
        """Not supported; always raises `TypeError`."""
        raise TypeError("MetadataView does not support parents.")
