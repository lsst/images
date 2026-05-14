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

from lsst.resources import ResourcePath

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


class ExtensionFromUriTestCase(unittest.TestCase):
    """`read_from_uri` routes based on `uri.getExtension()`."""

    def test_fits(self):
        from lsst.images.formatters import GenericFormatter

        formatter = GenericFormatter.__new__(GenericFormatter)
        uri = ResourcePath("/tmp/x.fits")
        self.assertEqual(formatter._extension_from_uri(uri), ".fits")

    @unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
    def test_sdf(self):
        from lsst.images.formatters import GenericFormatter

        formatter = GenericFormatter.__new__(GenericFormatter)
        uri = ResourcePath("/tmp/x.sdf")
        self.assertEqual(formatter._extension_from_uri(uri), ".sdf")

    def test_json(self):
        from lsst.images.formatters import GenericFormatter

        formatter = GenericFormatter.__new__(GenericFormatter)
        uri = ResourcePath("/tmp/x.json")
        self.assertEqual(formatter._extension_from_uri(uri), ".json")

    def test_unknown(self):
        from lsst.images.formatters import GenericFormatter

        formatter = GenericFormatter.__new__(GenericFormatter)
        uri = ResourcePath("/tmp/x.pickle")
        with self.assertRaisesRegex(RuntimeError, "unsupported extension"):
            formatter._extension_from_uri(uri)

    def test_compressed_fits_unsupported(self):
        # We don't claim to handle .fits.gz; getExtension returns
        # '.fits.gz' and the lookup misses.
        from lsst.images.formatters import GenericFormatter

        formatter = GenericFormatter.__new__(GenericFormatter)
        uri = ResourcePath("/tmp/x.fits.gz")
        with self.assertRaisesRegex(RuntimeError, "unsupported extension"):
            formatter._extension_from_uri(uri)


class ImageFormatterComponentReadTestCase(unittest.TestCase):
    """ImageFormatter routes component reads per extension."""

    def _make_image(self):
        import numpy as np

        from lsst.images import Box, Image

        return Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )

    def test_fits_bbox_component(self):
        import tempfile

        from lsst.images import Image, fits
        from lsst.images.formatters import ImageFormatter

        image = self._make_image()
        with tempfile.NamedTemporaryFile(suffix=".fits", delete_on_close=False) as tmp:
            tmp.close()
            fits.write(image, tmp.name)
            formatter = ImageFormatter.__new__(ImageFormatter)
            object.__setattr__(formatter, "_storage_class_pytype", Image)
            bbox = formatter._read_component_from_uri("bbox", ResourcePath(tmp.name))
            self.assertEqual(bbox, image.bbox)

    @unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
    def test_sdf_bbox_component(self):
        import tempfile

        from lsst.images import Image, ndf
        from lsst.images.formatters import ImageFormatter

        image = self._make_image()
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            ndf.write(image, tmp.name)
            formatter = ImageFormatter.__new__(ImageFormatter)
            object.__setattr__(formatter, "_storage_class_pytype", Image)
            bbox = formatter._read_component_from_uri("bbox", ResourcePath(tmp.name))
            self.assertEqual(bbox, image.bbox)

    def test_json_bbox_component_via_whole_object(self):
        import tempfile

        from lsst.images import Image
        from lsst.images import json as images_json
        from lsst.images.formatters import ImageFormatter

        image = self._make_image()
        with tempfile.NamedTemporaryFile(suffix=".json", delete_on_close=False) as tmp:
            tmp.close()
            images_json.write(image, tmp.name)
            formatter = ImageFormatter.__new__(ImageFormatter)
            object.__setattr__(formatter, "_storage_class_pytype", Image)
            bbox = formatter._read_component_from_uri("bbox", ResourcePath(tmp.name))
            self.assertEqual(bbox, image.bbox)

    def test_json_unknown_component_raises(self):
        import tempfile

        from lsst.images import Image
        from lsst.images import json as images_json
        from lsst.images.formatters import ImageFormatter

        image = self._make_image()
        with tempfile.NamedTemporaryFile(suffix=".json", delete_on_close=False) as tmp:
            tmp.close()
            images_json.write(image, tmp.name)
            formatter = ImageFormatter.__new__(ImageFormatter)
            object.__setattr__(formatter, "_storage_class_pytype", Image)
            with self.assertRaises(NotImplementedError):
                formatter._read_component_from_uri("nonexistent", ResourcePath(tmp.name))


class MaskedImageFormatterComponentReadTestCase(unittest.TestCase):
    """MaskedImageFormatter routes image/mask/variance per extension."""

    def _make_masked_image(self):
        import numpy as np

        from lsst.images import Image, MaskedImage, MaskPlane, MaskSchema

        rng = np.random.default_rng(11)
        return MaskedImage(
            Image(rng.normal(100.0, 8.0, size=(10, 12)), start=(0, 0)),
            mask_schema=MaskSchema([MaskPlane("BAD", "bad pixel")]),
        )

    def test_fits_image_component(self):
        import tempfile

        from lsst.images import MaskedImage, fits
        from lsst.images.formatters import MaskedImageFormatter

        mi = self._make_masked_image()
        with tempfile.NamedTemporaryFile(suffix=".fits", delete_on_close=False) as tmp:
            tmp.close()
            fits.write(mi, tmp.name)
            formatter = MaskedImageFormatter.__new__(MaskedImageFormatter)
            object.__setattr__(formatter, "_storage_class_pytype", MaskedImage)
            image = formatter._read_component_from_uri("image", ResourcePath(tmp.name))
            self.assertEqual(image.bbox, mi.image.bbox)

    @unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
    def test_sdf_mask_component(self):
        import tempfile

        from lsst.images import MaskedImage, ndf
        from lsst.images.formatters import MaskedImageFormatter

        mi = self._make_masked_image()
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            ndf.write(mi, tmp.name)
            formatter = MaskedImageFormatter.__new__(MaskedImageFormatter)
            object.__setattr__(formatter, "_storage_class_pytype", MaskedImage)
            mask = formatter._read_component_from_uri("mask", ResourcePath(tmp.name))
            self.assertEqual(mask.bbox, mi.mask.bbox)

    def test_json_variance_component_via_whole_object(self):
        import tempfile

        from lsst.images import MaskedImage
        from lsst.images import json as images_json
        from lsst.images.formatters import MaskedImageFormatter

        mi = self._make_masked_image()
        with tempfile.NamedTemporaryFile(suffix=".json", delete_on_close=False) as tmp:
            tmp.close()
            images_json.write(mi, tmp.name)
            formatter = MaskedImageFormatter.__new__(MaskedImageFormatter)
            object.__setattr__(formatter, "_storage_class_pytype", MaskedImage)
            variance = formatter._read_component_from_uri("variance", ResourcePath(tmp.name))
            self.assertEqual(variance.bbox, mi.variance.bbox)


class VisitImageFormatterComponentReadTestCase(unittest.TestCase):
    """VisitImageFormatter reads VisitImage-specific components."""

    def _make_visit_image(self):
        # Reuse the existing test helper from tests/test_visit_image.py.
        # Pytest places the tests directory on sys.path, so import the
        # sibling module by its bare name.
        from test_visit_image import VisitImageTestCase  # local import

        VisitImageTestCase.setUpClass()
        return VisitImageTestCase.visit_image

    def test_fits_summary_stats_component(self):
        import tempfile

        from lsst.images import VisitImage, fits
        from lsst.images.formatters import VisitImageFormatter

        vi = self._make_visit_image()
        with tempfile.NamedTemporaryFile(suffix=".fits", delete_on_close=False) as tmp:
            tmp.close()
            fits.write(vi, tmp.name)
            formatter = VisitImageFormatter.__new__(VisitImageFormatter)
            object.__setattr__(formatter, "_storage_class_pytype", VisitImage)
            summary = formatter._read_component_from_uri("summary_stats", ResourcePath(tmp.name))
            self.assertEqual(summary, vi.summary_stats)

    @unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
    def test_sdf_psf_component(self):
        import tempfile

        from lsst.images import VisitImage, ndf
        from lsst.images.formatters import VisitImageFormatter

        vi = self._make_visit_image()
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            ndf.write(vi, tmp.name)
            formatter = VisitImageFormatter.__new__(VisitImageFormatter)
            object.__setattr__(formatter, "_storage_class_pytype", VisitImage)
            psf = formatter._read_component_from_uri("psf", ResourcePath(tmp.name))
            self.assertEqual(type(psf), type(vi.psf))

    def test_json_aperture_corrections_via_whole_object(self):
        import tempfile

        from lsst.images import VisitImage
        from lsst.images import json as images_json
        from lsst.images.formatters import VisitImageFormatter

        vi = self._make_visit_image()
        with tempfile.NamedTemporaryFile(suffix=".json", delete_on_close=False) as tmp:
            tmp.close()
            images_json.write(vi, tmp.name)
            formatter = VisitImageFormatter.__new__(VisitImageFormatter)
            object.__setattr__(formatter, "_storage_class_pytype", VisitImage)
            ap = formatter._read_component_from_uri("aperture_corrections", ResourcePath(tmp.name))
            # ChebyshevField has no __eq__; compare keys and types.
            self.assertEqual(ap.keys(), vi.aperture_corrections.keys())
            for k, v in vi.aperture_corrections.items():
                self.assertEqual(type(ap[k]), type(v))
