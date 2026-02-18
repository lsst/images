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

__all__ = ("RoundtripFits",)

import tempfile
import unittest
from contextlib import ExitStack
from typing import Any

import astropy.io.fits

from .. import fits


class RoundtripFits[T]:
    """A context manager for testing FITS-based serialization.

    Parameters
    ----------
    tc
        A test case object to used for internal checks.
    original
        The object to serialize.

    Notes
    -----
    When entered, this context manager writes the object and reads it back in
    to the ``result`` attribute.  When exited, any temporary files or
    directories are deleted, but the ``result`` attribute is still usable.
    In between the `inspect` and `get` methods can be used to perform other
    tests.
    """

    def __init__(self, tc: unittest.TestCase, original: T):
        self._original = original
        self._serialized: Any = None
        self._exit_stack = ExitStack()
        self._filename: str | None = None
        self.result: Any

    def __enter__(self) -> RoundtripFits[T]:
        self._exit_stack.__enter__()
        self._run_without_butler()
        return self

    def __exit__(self, *args: Any) -> bool | None:
        return self._exit_stack.__exit__(*args)

    @property
    def filename(self) -> str:
        """The name of the file the object was written to."""
        assert self._filename is not None, "Context manager must be entered first."
        return self._filename

    @property
    def serialized(self) -> Any:
        """The serialization model for this object
        (`.serialization.ArchiveTree`).
        """
        return self._serialized

    def inspect(self) -> astropy.io.fits.HDUList:
        """Open the FITS file with Astropy."""
        return self._exit_stack.enter_context(
            astropy.io.fits.open(self.filename, disable_image_compression=True)
        )

    def get(self, **kwargs: Any) -> Any:
        """Perform a partial read.

        Parameters
        ----------
        **kwargs
            Keyword arguments either passed directly to `.fits.read` or used
            as ``parameters`` for a `~lsst.daf.butler.Butler.get`.

        Return
        ------
        object
            Result of the partial read.
        """
        return fits.read(type(self._original), self.filename, **kwargs)

    def _run_without_butler(self) -> None:
        tmp = self._exit_stack.enter_context(
            tempfile.NamedTemporaryFile(suffix=".fits", delete_on_close=False, delete=True)
        )
        tmp.close()
        self._filename = tmp.name
        self._serialized = fits.write(self._original, tmp.name)
        self.result = fits.read(type(self._original), tmp.name)
