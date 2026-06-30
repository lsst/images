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

__all__ = (
    "DP2_COADD_DATA_ID",
    "DP2_COADD_MISSING_CELL",
    "DP2_TEMPLATE_COADD_DATASETS",
    "DP2_VISIT_DETECTOR_DATA_ID",
)

import uuid
from typing import Any

DP2_VISIT_DETECTOR_DATA_ID: dict[str, Any] = {"instrument": "LSSTCam", "visit": 2025052000177, "detector": 85}
DP2_COADD_DATA_ID: dict[str, Any] = {"skymap": "lsst_cells_v2", "tract": 9813, "patch": 43, "band": "r"}
DP2_COADD_MISSING_CELL: dict[str, int] = {"i": 8, "j": 6}

DP2_TEMPLATE_COADD_DATASETS = {
    uuid.UUID("019d7854-1c93-7a5e-a594-c0ecbd16ad75"): {
        "band": "r",
        "skymap": "lsst_cells_v2",
        "tract": 9571,
        "patch": 97,
    },
    uuid.UUID("019d7854-34d9-781c-829c-6ba851899d5d"): {
        "band": "r",
        "skymap": "lsst_cells_v2",
        "tract": 9571,
        "patch": 98,
    },
    uuid.UUID("019d7854-339d-7767-8b5e-d81eaed45041"): {
        "band": "r",
        "skymap": "lsst_cells_v2",
        "tract": 9813,
        "patch": 2,
    },
    uuid.UUID("019d7854-12ca-7bd2-96f3-f036b9d6db97"): {
        "band": "r",
        "skymap": "lsst_cells_v2",
        "tract": 9813,
        "patch": 3,
    },
    uuid.UUID("019d7854-3110-784f-9fd8-13594bf3f230"): {
        "band": "r",
        "skymap": "lsst_cells_v2",
        "tract": 9813,
        "patch": 12,
    },
    uuid.UUID("019d7854-1036-7f97-838f-0c50719d0986"): {
        "band": "r",
        "skymap": "lsst_cells_v2",
        "tract": 9813,
        "patch": 13,
    },
    uuid.UUID("019d7854-2fcd-72a3-86c4-20f9dedd7f1c"): {
        "band": "r",
        "skymap": "lsst_cells_v2",
        "tract": 9813,
        "patch": 14,
    },
    uuid.UUID("019d7854-0db2-7f63-9487-1c888a64bcc1"): {
        "band": "r",
        "skymap": "lsst_cells_v2",
        "tract": 9813,
        "patch": 22,
    },
    uuid.UUID("019d7854-389e-7cd4-9464-a9e1e584c4ae"): {
        "band": "r",
        "skymap": "lsst_cells_v2",
        "tract": 9813,
        "patch": 23,
    },
}
