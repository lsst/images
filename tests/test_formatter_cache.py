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

import numpy as np
import pytest

from lsst.images import Box, VisitImage
from lsst.images.serialization import open as real_ser_open
from lsst.images.serialization import read
from lsst.images.tests import TemporaryButler

try:
    # The formatter module requires lsst.daf.butler.
    from lsst.images.formatters import GenericFormatter, _TreeCache

    HAVE_BUTLER = True
except ImportError:
    HAVE_BUTLER = False

skip_no_butler = pytest.mark.skipif(not HAVE_BUTLER, reason="lsst.daf.butler could not be imported.")

LOCAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "schema_v1")


def _count_ser_opens():
    """Return a patch that counts calls to lsst.images.serialization.open."""
    import unittest.mock

    return unittest.mock.patch("lsst.images.serialization.open", side_effect=real_ser_open)


def _reset_cache() -> None:
    """Reset the GenericFormatter tree cache to an empty state."""
    GenericFormatter._tree_cache.value = _TreeCache()


@pytest.fixture(scope="session")
def visit_image() -> VisitImage:
    """Return a test VisitImage."""
    return read(os.path.join(LOCAL_DATA_DIR, "visit_image.json"))


@skip_no_butler
def test_free_component_reads_share_one_open(visit_image: VisitImage) -> None:
    """Verify multiple free component reads share a single
    serialization.open call.
    """
    with TemporaryButler(visit_image="VisitImage") as helper:
        helper.butler.put(visit_image, helper.visit_image)
        _reset_cache()
        with _count_ser_opens() as mocked:
            summary_stats = helper.butler.get(helper.visit_image.makeComponentRef("summary_stats"))
            assert mocked.call_count == 1
            obs_info = helper.butler.get(helper.visit_image.makeComponentRef("obs_info"))
            sky_projection = helper.butler.get(helper.visit_image.makeComponentRef("sky_projection"))
            assert mocked.call_count == 1
        assert summary_stats == visit_image.summary_stats
        assert obs_info == visit_image.obs_info
        assert sky_projection is not None
    _reset_cache()


@skip_no_butler
def test_cached_components_are_independent(visit_image: VisitImage) -> None:
    """Verify repeated component reads return equal but independent objects."""
    with TemporaryButler(visit_image="VisitImage") as helper:
        helper.butler.put(visit_image, helper.visit_image)
        _reset_cache()
        ref = helper.visit_image.makeComponentRef("summary_stats")
        first = helper.butler.get(ref)
        second = helper.butler.get(ref)
        assert first == second
        assert first is not second
        # Mutating one result must not leak into later reads.
        first.zeroPoint = -100.0
        third = helper.butler.get(ref)
        assert third == second
        assert third != first
    _reset_cache()


@skip_no_butler
def test_pixel_component_falls_back_to_file(visit_image: VisitImage) -> None:
    """Verify pixel component reads bypass the cache and re-open the file."""
    with TemporaryButler(visit_image="VisitImage") as helper:
        helper.butler.put(visit_image, helper.visit_image)
        _reset_cache()
        with _count_ser_opens() as mocked:
            helper.butler.get(helper.visit_image.makeComponentRef("summary_stats"))
            assert mocked.call_count == 1
            image = helper.butler.get(helper.visit_image.makeComponentRef("image"))
            assert mocked.call_count == 2
        np.testing.assert_array_equal(image.array, visit_image.image.array)
    _reset_cache()


@skip_no_butler
def test_parameterized_read_bypasses_cache(visit_image: VisitImage) -> None:
    """Verify a parameterized (bbox) component read bypasses the cache."""
    with TemporaryButler(visit_image="VisitImage") as helper:
        helper.butler.put(visit_image, helper.visit_image)
        _reset_cache()
        bbox = visit_image.bbox
        cutout_box = Box.factory[bbox.y.start : bbox.y.start + 2, bbox.x.start : bbox.x.start + 2]
        with _count_ser_opens() as mocked:
            helper.butler.get(helper.visit_image.makeComponentRef("summary_stats"))
            assert mocked.call_count == 1
            cutout = helper.butler.get(
                helper.visit_image.makeComponentRef("image"), parameters={"bbox": cutout_box}
            )
            assert mocked.call_count == 2
        assert cutout.bbox == cutout_box
    _reset_cache()


@skip_no_butler
def test_full_read_populates_cache(visit_image: VisitImage) -> None:
    """Verify a full object read populates the cache for subsequent component
    reads.
    """
    with TemporaryButler(visit_image="VisitImage") as helper:
        helper.butler.put(visit_image, helper.visit_image)
        _reset_cache()
        helper.butler.get(helper.visit_image)
        with _count_ser_opens() as mocked:
            helper.butler.get(helper.visit_image.makeComponentRef("obs_info"))
            assert mocked.call_count == 0
    _reset_cache()


@skip_no_butler
def test_cache_invalidated_across_datasets(visit_image: VisitImage) -> None:
    """Verify the cache is invalidated when switching between dataset
    references.
    """
    with TemporaryButler(vi_a="VisitImage", vi_b="VisitImage") as helper:
        helper.butler.put(visit_image, helper.vi_a)
        helper.butler.put(visit_image, helper.vi_b)
        _reset_cache()
        with _count_ser_opens() as mocked:
            helper.butler.get(helper.vi_a.makeComponentRef("summary_stats"))
            assert mocked.call_count == 1
            helper.butler.get(helper.vi_b.makeComponentRef("summary_stats"))
            assert mocked.call_count == 2
            # vi_b evicted vi_a, so reading vi_a again reopens its file.
            helper.butler.get(helper.vi_a.makeComponentRef("obs_info"))
            assert mocked.call_count == 3
    _reset_cache()
