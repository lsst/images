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

__all__ = ("RoundtripFits", "TemporaryButler")

import tempfile
import unittest
from contextlib import ExitStack
from typing import Any, TypeVar

import astropy.io.fits

try:
    from lsst.daf.butler import Butler, DataCoordinate, DatasetRef, DatasetType

    HAVE_BUTLER = True
except ImportError:
    HAVE_BUTLER = False

from .. import fits

# We need an old-style TypeVar for Sphinx.
T = TypeVar("T")


class TemporaryButler:
    """Make a temporary butler repository.

    Parameters
    ----------
    run
        Name of a `~lsst.daf.butler.CollectionType.RUN` collection to
        register and use as the default run for the returned butler.
    **kwargs
        A mapping from a dataset type name to its storage class.  For each
        entry, a dataset type will be registered with empty dimensions, and a
        `~lsst.daf.butler.DatasetRef` will be created and added as an
        attribute of this class.

    Raises
    ------
    unittest.SkipTest
        Raised when the context manager is entered if `lsst.daf.butler` could
        not be imported.  This is typically handled by using this context
        manager within a `unittest.TestCase.subTest` context, which will skip
        just the butler-required tests in that context while allowing the rest
        of the test to continue.
    """

    def __init__(self, run: str = "test_run", **kwargs: str):
        self.run = run
        self._kwargs = kwargs
        self._exit_stack = ExitStack()

    def __enter__(self) -> TemporaryButler:
        if not HAVE_BUTLER:
            raise unittest.SkipTest("lsst.daf.butler could not be imported.")
        self._exit_stack.__enter__()
        root = self._exit_stack.enter_context(
            tempfile.TemporaryDirectory(ignore_cleanup_errors=True, delete=True)
        )
        butler_config = Butler.makeRepo(root)
        self.butler = self._exit_stack.enter_context(Butler.from_config(butler_config, run=self.run))
        empty_data_id = DataCoordinate.make_empty(self.butler.dimensions)
        for name, storage_class in self._kwargs.items():
            dataset_type = DatasetType(name, self.butler.dimensions.empty, storage_class)
            try:
                self.butler.registry.registerDatasetType(dataset_type)
            except KeyError as err:
                err.add_note(
                    "Storage class not configured in butler defaults.  "
                    "A newer version of daf_butler may be needed."
                )
                raise
            setattr(self, name, DatasetRef(dataset_type, empty_data_id, self.run))
        return self

    def __exit__(self, *args: Any) -> bool | None:
        return self._exit_stack.__exit__(*args)

    # Just for typing, since this class uses dynamic attributes.
    def __getattr__(self, name: str) -> DatasetRef:
        raise AttributeError(name)


class RoundtripFits[T]:
    """A context manager for testing FITS-based serialization.

    Parameters
    ----------
    tc
        A test case object to used for internal checks.
    original
        The object to serialize.
    storage_class
        A butler storage class name to use.  If not provided (or
        `lsst.daf.butler` cannot be imported), the roundtrip will just use
        a direct write to a temporary file.

    Notes
    -----
    When entered, this context manager writes the object and reads it back in
    to the ``result`` attribute.  When exited, any temporary files or
    directories are deleted, but the ``result`` attribute is still usable.
    In between the `inspect` and `get` methods can be used to perform other
    tests.
    """

    def __init__(self, tc: unittest.TestCase, original: T, storage_class: str | None = None):
        self._original = original
        self._storage_class = storage_class
        self._serialized: Any = None
        self._exit_stack = ExitStack()
        self._filename: str | None = None
        self.result: Any
        self.butler: Butler | None = None
        self.ref: DatasetRef | None = None

    def __enter__(self) -> RoundtripFits[T]:
        self._exit_stack.__enter__()
        if HAVE_BUTLER and self._storage_class is not None:
            self._run_with_butler()
        else:
            self._run_without_butler()
        return self

    def __exit__(self, *args: Any) -> bool | None:
        return self._exit_stack.__exit__(*args)

    @property
    def filename(self) -> str:
        """The name of the file the object was written to."""
        if self._filename is None:
            assert self.butler is not None and self.ref is not None
            self._filename = self.butler.getURI(self.ref).ospath
        return self._filename

    @property
    def serialized(self) -> Any:
        """The serialization model for this object
        (`.serialization.ArchiveTree`).
        """
        if self._serialized is None:
            # The butler code path doesn't give us a way to inspect the
            # serialized model, so we have to save it again directly to another
            # file (which we then discard).
            with tempfile.NamedTemporaryFile(suffix=".fits", delete_on_close=False, delete=True) as tmp:
                tmp.close()
                self._serialized = fits.write(self._original, tmp.name)
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
        if self.butler is None:
            return fits.read(type(self._original), self.filename, **kwargs)
        else:
            assert self.ref is not None, "butler and ref should be None or not together"
            return self.butler.get(self.ref, parameters=kwargs)

    def _run_with_butler(self) -> None:
        assert self._storage_class is not None, "Should not use butler if no storage class"
        butler_helper = self._exit_stack.enter_context(TemporaryButler(test_dataset=self._storage_class))
        self.butler = butler_helper.butler
        self.ref = self.butler.put(self._original, butler_helper.test_dataset)
        self.result = self.butler.get(self.ref)

    def _run_without_butler(self) -> None:
        tmp = self._exit_stack.enter_context(
            tempfile.NamedTemporaryFile(suffix=".fits", delete_on_close=False, delete=True)
        )
        tmp.close()
        self._filename = tmp.name
        self._serialized = fits.write(self._original, tmp.name)
        self.result = fits.read(type(self._original), tmp.name)
