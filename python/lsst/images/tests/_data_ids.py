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
    "get_dp2_exposure_record",
)

import uuid
from typing import Any

import astropy.time

from lsst.daf.butler import DimensionRecord, DimensionUniverse, Timespan

DP2_VISIT_DETECTOR_DATA_ID: dict[str, Any] = {
    "instrument": "LSSTCam",
    "visit": 2025052000177,
    "detector": 85,
    "day_obs": 20250520,
    "physical_filter": "r_57",
    "band": "r",
}
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


def get_dp2_exposure_record(universe: DimensionUniverse) -> DimensionRecord:
    """Return the exposure record associated with the DP2 test data ID.

    Parameters
    ----------
    universe
        Dimension universe that defines the record's schema.
    """
    return universe["exposure"].RecordClass(
        instrument="LSSTCam",
        id=2025052000177,
        day_obs=20250520,
        group="2025-05-21T01:28:49.433",
        physical_filter="r_57",
        obs_id="MC_O_20250520_000177",
        exposure_time=30.001108646392822,
        dark_time=30.9427,
        observation_type="science",
        observation_reason="field_survey_science",
        seq_num=177,
        seq_start=177,
        seq_end=177,
        target_name="COSMOS",
        science_program="BLOCK-365",
        tracking_ra=150.1928872274704,
        tracking_dec=2.2413629369011385,
        sky_angle=57.7790074960578,
        azimuth=302.7758502576103,
        zenith_angle=50.1758964491342,
        has_simulated=False,
        can_see_sky=True,
        timespan=Timespan(
            begin=astropy.time.Time(2460817.0, -0.436745640356088, scale="tai", format="jd"),
            end=astropy.time.Time(2460817.0, -0.43638751157407407, scale="tai", format="jd"),
        ),
    )
