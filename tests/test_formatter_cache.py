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
import unittest.mock

from lsst.images.serialization import open as real_ser_open
from lsst.images.serialization import read
from lsst.images.tests import TemporaryButler

try:
    # The formatter module requires lsst.daf.butler.
    from lsst.images.formatters import GenericFormatter, _TreeCache

    HAVE_BUTLER = True
except ImportError:
    HAVE_BUTLER = False

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "schema_v1")


def _count_ser_opens() -> unittest.mock._patch:
    """Patch lsst.images.serialization.open with a counting wrapper.

    The formatter resolves ``ser.open`` through the module object at call
    time, so patching the module attribute counts the opens it performs
    while delegating to the real implementation.
    """
    return unittest.mock.patch("lsst.images.serialization.open", side_effect=real_ser_open)


@unittest.skipUnless(HAVE_BUTLER, "lsst.daf.butler could not be imported.")
class FormatterComponentCacheTestCase(unittest.TestCase):
    """Component reads through the butler reuse the cached tree."""

    def setUp(self) -> None:
        self.visit_image = read(os.path.join(DATA_DIR, "visit_image.json"))
        self._reset_cache()
        self.addCleanup(self._reset_cache)

    @staticmethod
    def _reset_cache() -> None:
        GenericFormatter._tree_cache = _TreeCache()

    def test_free_component_reads_share_one_open(self) -> None:
        with TemporaryButler(visit_image="VisitImage") as helper:
            helper.butler.put(self.visit_image, helper.visit_image)
            self._reset_cache()
            with _count_ser_opens() as mocked:
                summary_stats = helper.butler.get(helper.visit_image.makeComponentRef("summary_stats"))
                self.assertEqual(mocked.call_count, 1)
                obs_info = helper.butler.get(helper.visit_image.makeComponentRef("obs_info"))
                sky_projection = helper.butler.get(helper.visit_image.makeComponentRef("sky_projection"))
                self.assertEqual(mocked.call_count, 1)
            self.assertEqual(summary_stats, self.visit_image.summary_stats)
            self.assertEqual(obs_info, self.visit_image.obs_info)
            self.assertIsNotNone(sky_projection)


if __name__ == "__main__":
    unittest.main()
