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

__all__ = ("reset_afw_mask_planes",)

import pytest

try:
    from lsst.afw.image import Mask
except ImportError:  # pragma: no cover

    @pytest.fixture
    def reset_afw_mask_planes() -> None:
        """Reset afw's process-global mask-plane registry.

        This variant of the fixture just skips tests, because afw could not
        be imported.
        """
        pytest.skip("lsst.afw.image could not be imported")
else:

    def _get_standard_afw_mask_planes() -> tuple[str, ...]:
        plane_dict = Mask(1, 1).getMaskPlaneDict()
        return tuple(sorted(plane_dict, key=plane_dict.get))

    _STANDARD_AFW_MASK_PLANES = _get_standard_afw_mask_planes()

    @pytest.fixture
    def reset_afw_mask_planes() -> None:
        """Reset afw's process-global mask-plane registry.

        `lsst.afw.image.Mask` has a fixed 32-plane registry that is global
        to the process.  We can run out of bits if we don't reset it for
        every test, even if no single Mask needs more than 32.
        """
        Mask.clearMaskPlaneDict()
        for name in _STANDARD_AFW_MASK_PLANES:
            Mask.addMaskPlane(name)
