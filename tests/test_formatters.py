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

import unittest

try:
    import h5py  # noqa: F401

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False


class BackendsTableTestCase(unittest.TestCase):
    """The private _BACKENDS table wires extension -> read/write/archive."""

    def test_table_keys(self):
        from lsst.images.formatters import _BACKENDS

        self.assertEqual(set(_BACKENDS), {".fits", ".sdf", ".json"})

    def test_fits_backend_wires_fits_read_write(self):
        from lsst.images import fits
        from lsst.images.fits._common import PointerModel
        from lsst.images.fits._input_archive import FitsInputArchive
        from lsst.images.formatters import _BACKENDS

        backend = _BACKENDS[".fits"]
        self.assertIs(backend.read, fits.read)
        self.assertIs(backend.write, fits.write)
        self.assertIs(backend.input_archive, FitsInputArchive)
        self.assertIs(backend.pointer_model, PointerModel)

    @unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
    def test_sdf_backend_wires_ndf_read_write(self):
        from lsst.images import ndf
        from lsst.images.formatters import _BACKENDS
        from lsst.images.ndf._common import NdfPointerModel
        from lsst.images.ndf._input_archive import NdfInputArchive

        backend = _BACKENDS[".sdf"]
        self.assertIs(backend.read, ndf.read)
        self.assertIs(backend.write, ndf.write)
        self.assertIs(backend.input_archive, NdfInputArchive)
        self.assertIs(backend.pointer_model, NdfPointerModel)

    def test_json_backend_wires_json_read_write_no_archive(self):
        from lsst.images import json as images_json
        from lsst.images.formatters import _BACKENDS

        backend = _BACKENDS[".json"]
        self.assertIs(backend.read, images_json.read)
        self.assertIs(backend.write, images_json.write)
        self.assertIsNone(backend.input_archive)
        self.assertIsNone(backend.pointer_model)


class GetWriteExtensionTestCase(unittest.TestCase):
    """`get_write_extension` reads the `format` write parameter."""

    def _make_formatter(self, write_parameters: dict[str, str] | None = None):
        from lsst.images.formatters import GenericFormatter

        formatter = GenericFormatter.__new__(GenericFormatter)
        # FormatterV2 exposes write_parameters as a property over the
        # file_descriptor. For unit-testing we monkey-patch a dict on
        # the instance via __dict__ to bypass the descriptor.
        object.__setattr__(formatter, "_write_parameters", write_parameters or {})
        return formatter

    def test_default_returns_fits(self):
        formatter = self._make_formatter()
        self.assertEqual(formatter.get_write_extension(), ".fits")

    def test_explicit_fits(self):
        formatter = self._make_formatter({"format": "fits"})
        self.assertEqual(formatter.get_write_extension(), ".fits")

    def test_explicit_json(self):
        formatter = self._make_formatter({"format": "json"})
        self.assertEqual(formatter.get_write_extension(), ".json")

    @unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
    def test_explicit_sdf(self):
        formatter = self._make_formatter({"format": "sdf"})
        self.assertEqual(formatter.get_write_extension(), ".sdf")

    def test_unknown_format_raises(self):
        formatter = self._make_formatter({"format": "pickle"})
        with self.assertRaisesRegex(RuntimeError, "is not supported"):
            formatter.get_write_extension()

    def test_recipe_with_non_fits_format_raises(self):
        # `recipe` is FITS-only; using it with format=json must error.
        formatter = self._make_formatter({"format": "json", "recipe": "default"})
        with self.assertRaisesRegex(RuntimeError, "only valid for FITS"):
            formatter._validate_write_parameters()
