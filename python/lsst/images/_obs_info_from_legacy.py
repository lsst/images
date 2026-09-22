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

__all__ = ("obs_info_from_legacy",)

from typing import TYPE_CHECKING, Any

import astropy.units as u
from astro_metadata_translator import ObservationInfo, VisitInfoTranslator
from astropy.coordinates import AltAz

if TYPE_CHECKING:
    try:
        from lsst.afw.cameraGeom import Detector as LegacyDetector
        from lsst.afw.image import VisitInfo as LegacyVisitInfo
        from lsst.daf.butler import DimensionRecord
    except ImportError:
        type LegacyDetector = Any  # type: ignore[no-redef]
        type LegacyVisitInfo = Any  # type: ignore[no-redef]
        type DimensionRecord = Any  # type: ignore[no-redef]


def obs_info_from_legacy(
    visit_info: LegacyVisitInfo,
    exposure_record: DimensionRecord,
    detector: LegacyDetector,
    *,
    detector_exposure_id: int,
) -> ObservationInfo:
    """Reconstruct as much of an observation info struct as possible from
    what's available from a legacy `lsst.afw.image.Exposure`, an ``exposure``
    dimension record, and the camera geometry.

    Parameters
    ----------
    visit_info
        The legacy visit info struct.
    exposure_record
        The ``exposure`` `~lsst.daf.butler.DimensionRecord`.
    detector
        The legacy detector object; the source for detector identifiers.
    detector_exposure_id
        The combination detector + exposure ID.

    Returns
    -------
    obs_info : `~astro_metadata_translator.ObservationInfo`
        The reconstructed observation metadata.
    """
    from lsst.afw.image import setVisitInfoMetadata
    from lsst.daf.base import PropertyList

    # Start by exporting the VisitInfo to FITS header cards and translating
    # those. This populates a lot of fields correctly, a lot of fields not at
    # all, and some *incorrectly*, at least as compared to the original
    # ObservationInfo, because VisitInfoTranslator tries too hard to
    # reconstruct information that has been lost.
    pl = PropertyList()
    setVisitInfoMetadata(pl, visit_info)
    obs_info = ObservationInfo.from_header(pl, translator_class=VisitInfoTranslator, quiet=True)
    with obs_info.edit_copy() as obs_info:
        obs_info.detector_exposure_id = detector_exposure_id
        # Since the dimension record is created directly from the original
        # ObservationInfo, anything it actually holds is authoritative.
        obs_info.datetime_begin = exposure_record.timespan.begin
        obs_info.datetime_end = exposure_record.timespan.end
        obs_info.observing_day = exposure_record.day_obs
        obs_info.observation_id = exposure_record.obs_id
        obs_info.exposure_group = exposure_record.group
        obs_info.physical_filter = exposure_record.physical_filter
        # The record's exposure_time is the requested exposure time; the
        # VisitInfo translator's exposure_time is the actual shutter-open time.
        obs_info.exposure_time_requested = (
            None if exposure_record.exposure_time is None else exposure_record.exposure_time * u.s
        )
        obs_info.observation_counter = exposure_record.seq_num
        obs_info.group_counter_start = exposure_record.seq_start
        obs_info.group_counter_end = exposure_record.seq_end
        obs_info.can_see_sky = exposure_record.can_see_sky
        if (azimuth := exposure_record.azimuth) is not None and (
            zenith_angle := exposure_record.zenith_angle
        ) is not None:
            # The record's azimuth/zenith_angle were ingested from
            # obsInfo.altaz_begin (begin-of-exposure). The VisitInfo stores
            # mid-exposure az/alt, and neither stores altaz_end.
            obs_info.altaz_begin = AltAz(
                az=azimuth * u.deg,
                alt=(90.0 - zenith_angle) * u.deg,
                obstime=exposure_record.timespan.begin,
                location=obs_info.location,
            )
        obs_info.detector_num = detector.getId()
        obs_info.detector_unique_name = detector.getName()
        obs_info.detector_serial = detector.getSerial()
    return obs_info
