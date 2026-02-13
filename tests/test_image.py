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

import os
import unittest

import numpy as np

from lsst.images import Box, Image

DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)


class ImageTestCase(unittest.TestCase):
    """Tests for the Image class."""

    @unittest.skipUnless(DATA_DIR is not None, "TESTDATA_IMAGES_DIR is not in the environment.")
    def test_legacy(self) -> None:
        """Test Image.read_legacy, Image.to_legacy, and Image.from_legacy."""
        assert DATA_DIR is not None, "Guaranteed by decorator."
        filename = os.path.join(DATA_DIR, "dp2", "legacy", "visit_image.fits")
        converted_on_read = Image.read_legacy(filename, preserve_quantization=True)
        try:
            from lsst.afw.image import MaskedImageFitsReader
        except ImportError:
            raise unittest.SkipTest("'lsst.afw.image' could not be imported.") from None
        reader = MaskedImageFitsReader(filename)
        self.assertEqual(converted_on_read.bbox, Box.from_legacy(reader.readBBox()))
        unconverted_afw = reader.readImage()
        np.testing.assert_array_equal(unconverted_afw.array, converted_on_read.array)
        # Converting back to afw will not share memory, because
        # preserve_quantization=True makes the array read-only and to_legacy
        # has to copy in that case.
        converted_back_to_afw_copy = converted_on_read.to_legacy()
        self.assertEqual(unconverted_afw.getBBox(), converted_back_to_afw_copy.getBBox())
        np.testing.assert_array_equal(unconverted_afw.array, converted_back_to_afw_copy.array)
        self.assertFalse(np.may_share_memory(unconverted_afw.array, converted_back_to_afw_copy.array))
        converted_in_memory = Image.from_legacy(unconverted_afw)
        self.assertEqual(converted_on_read.bbox, converted_in_memory.bbox)
        np.testing.assert_array_equal(converted_on_read.array, converted_in_memory.array)
        self.assertFalse(np.may_share_memory(converted_on_read.array, converted_in_memory.array))
        # Converting back to afw from the in-memory view will be another view.
        converted_back_to_afw_view = converted_in_memory.to_legacy()
        self.assertEqual(unconverted_afw.getBBox(), converted_back_to_afw_view.getBBox())
        np.testing.assert_array_equal(unconverted_afw.array, converted_back_to_afw_view.array)
        self.assertTrue(np.may_share_memory(unconverted_afw.array, converted_back_to_afw_view.array))


if __name__ == "__main__":
    unittest.main()
