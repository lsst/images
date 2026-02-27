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
from lsst.images.tests import assert_close, compare_image_to_legacy

DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)


class ImageTestCase(unittest.TestCase):
    """Tests for the Image class."""

    def test_basics(self):
        """Test basic constructor patterns."""
        image = Image(42, shape=(5, 5))
        assert_close(self, image.array, np.zeros([5, 5], dtype=np.int64) + 42)

        data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        image = Image(data)
        subset = image[Box.factory[:3, 1:3]]
        subset2 = image[:3, 1:3]
        self.assertEqual(subset2, subset)

        # Add an explicit bounding box and then slice it.
        image = Image(data, bbox=Box.factory[-2:1, 10:14])
        with self.assertRaises(IndexError):
            # Same slice no longer works because we have moved origin.
            image[:3, 1:3]
        subset = image[:0, 11:13]
        assert_close(self, subset.array, np.array([[2, 3], [6, 7]]))

    @unittest.skipUnless(DATA_DIR is not None, "TESTDATA_IMAGES_DIR is not in the environment.")
    def test_legacy(self) -> None:
        """Test Image.read_legacy, Image.to_legacy, and Image.from_legacy."""
        assert DATA_DIR is not None, "Guaranteed by decorator."
        filename = os.path.join(DATA_DIR, "dp2", "legacy", "visit_image.fits")
        image = Image.read_legacy(filename, preserve_quantization=True)
        try:
            from lsst.afw.image import MaskedImageFitsReader
        except ImportError:
            raise unittest.SkipTest("'lsst.afw.image' could not be imported.") from None
        reader = MaskedImageFitsReader(filename)
        legacy_image = reader.readImage()
        compare_image_to_legacy(self, image, legacy_image, expect_view=False)
        # Converting back to afw will not share memory, because
        # preserve_quantization=True makes the array read-only and to_legacy
        # has to copy in that case.
        compare_image_to_legacy(self, image, image.to_legacy(), expect_view=False)
        # Converting from afw will always share memory.
        image_view = Image.from_legacy(legacy_image)
        compare_image_to_legacy(self, image_view, legacy_image, expect_view=True)
        # Converting back to afw from the in-memory view will be another view.
        compare_image_to_legacy(self, image_view, image_view.to_legacy(), expect_view=True)


if __name__ == "__main__":
    unittest.main()
