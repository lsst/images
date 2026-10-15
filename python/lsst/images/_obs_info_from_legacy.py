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

import logging
from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any

import astropy.units as u
from astro_metadata_translator import (
    FITS_FLAT_DICT_PREFIX,
    FITS_FLAT_DICT_SEP,
    ObservationInfo,
    VisitInfoTranslator,
)
from astropy.coordinates import AltAz

_LOG = logging.getLogger("lsst.images")

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
    exposure_record: DimensionRecord | None,
    detector: LegacyDetector,
    *,
    detector_exposure_id: int,
    telescope: str | None = None,
    physical_filter: str | None = None,
    header: MutableMapping[str, Any] | None = None,
) -> ObservationInfo:
    """Recover the observation info struct of a legacy
    `lsst.afw.image.Exposure`.

    If the legacy metadata carries an embedded
    `~astro_metadata_translator.ObservationInfo` (the flat ``OBS_INFO`` cards
    written by the raw formatter), that is returned as is.  Otherwise as much
    as possible is reconstructed from the `~lsst.afw.image.VisitInfo`, an
    ``exposure`` dimension record, and the camera geometry.

    Parameters
    ----------
    visit_info
        The legacy visit info struct.
    exposure_record
        The ``exposure`` `~lsst.daf.butler.DimensionRecord`, or `None` to
        convert without one.  Without a record, fields it would authoritatively
        supply are set to `None` and a warning is logged.
    detector
        The legacy detector object; the source for detector identifiers.
    detector_exposure_id
        The combination detector + exposure ID.
    telescope
        Value of the FITS ``TELESCOP`` card, if available from the legacy
        metadata.  `~lsst.afw.image.VisitInfo` carries no telescope
        information, so without this the telescope is reported as
        ``"Unknown"``.
    physical_filter
        Full name of the physical bandpass filter (the ``physicalLabel`` of
        the legacy exposure's `~lsst.afw.image.FilterLabel`), if available.
        Only used when there is no ``exposure_record``.
    header
        The legacy exposure's metadata (its primary FITS header as an
        `astropy.io.fits.Header` or `lsst.daf.base.PropertyList`), if
        available.  When it holds an embedded ``ObservationInfo`` the
        ``OBS_INFO`` cards are removed from it in place, so they do not leak
        into other metadata, and the other parameters are only used to check
        that the ``exposure_record`` describes the same exposure.

    Returns
    -------
    obs_info : `~astro_metadata_translator.ObservationInfo`
        The recovered observation metadata.

    Raises
    ------
    ValueError
        Raised if ``exposure_record`` is given and does not describe the same
        instrument and exposure as the embedded ``ObservationInfo`` or the
        ``visit_info``.
    """
    flat_dict_kwargs: dict[str, Any] = {"prefix": FITS_FLAT_DICT_PREFIX, "sep": FITS_FLAT_DICT_SEP}
    if header is not None and ObservationInfo.has_flat_dict(header, **flat_dict_kwargs):
        obs_info = ObservationInfo.from_flat_dict(header, **flat_dict_kwargs)
        ObservationInfo.strip_from_flat_dict(header, **flat_dict_kwargs)
        if exposure_record is not None:
            if obs_info.instrument != exposure_record.instrument:
                raise ValueError(
                    f"Instrument {obs_info.instrument!r} from embedded ObservationInfo does not match "
                    f"instrument {exposure_record.instrument!r} of the given exposure record."
                )
            if obs_info.exposure_id != exposure_record.id:
                raise ValueError(
                    f"Exposure ID {obs_info.exposure_id} from embedded ObservationInfo does not match "
                    f"exposure ID {exposure_record.id} of the given exposure record."
                )
        return obs_info

    from lsst.afw.image import setVisitInfoMetadata
    from lsst.daf.base import PropertyList

    if exposure_record is not None:
        if (label := visit_info.getInstrumentLabel()) and label != exposure_record.instrument:
            raise ValueError(
                f"Instrument label {label!r} from VisitInfo does not match instrument "
                f"{exposure_record.instrument!r} of the given exposure record."
            )
        if (visit_id := visit_info.getId()) != exposure_record.id:
            raise ValueError(
                f"Exposure ID {visit_id} from VisitInfo does not match exposure ID "
                f"{exposure_record.id} of the given exposure record."
            )
    else:
        _LOG.warning(
            "Converting without an exposure dimension record: ObservationInfo "
            "fields the record would supply are set to None."
        )

    # Start by exporting the VisitInfo to FITS header cards and translating
    # those. This populates a lot of fields correctly, a lot of fields not at
    # all, and some *incorrectly*, at least as compared to the original
    # ObservationInfo, because VisitInfoTranslator tries too hard to
    # reconstruct information that has been lost.
    pl = PropertyList()
    setVisitInfoMetadata(pl, visit_info)
    if telescope is not None:
        # TELESCOP is a standard FITS card that VisitInfo knows nothing about,
        # but VisitInfoTranslator uses it to populate the telescope field.
        pl.add("TELESCOP", telescope)
    obs_info = ObservationInfo.from_header(pl, translator_class=VisitInfoTranslator, quiet=True)
    with obs_info.edit_copy() as obs_info:
        obs_info.detector_exposure_id = detector_exposure_id
        if exposure_record is not None:
            # Since the dimension record is created directly from the original
            # ObservationInfo, anything it actually holds is authoritative.
            obs_info.datetime_begin = exposure_record.timespan.begin
            obs_info.datetime_end = exposure_record.timespan.end
            obs_info.observing_day = exposure_record.day_obs
            obs_info.observation_id = exposure_record.obs_id
            obs_info.exposure_group = exposure_record.group
            obs_info.physical_filter = exposure_record.physical_filter
            # The record's exposure_time is the requested exposure time;
            # the VisitInfo translator's exposure_time is the actual
            # shutter-open time.
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
        else:
            obs_info.datetime_begin = None
            obs_info.datetime_end = None
            obs_info.observing_day = None
            obs_info.observation_id = None
            obs_info.exposure_group = None
            obs_info.physical_filter = physical_filter
            obs_info.exposure_time_requested = None
            obs_info.observation_counter = None
            obs_info.group_counter_start = None
            obs_info.group_counter_end = None
            obs_info.can_see_sky = None
            obs_info.altaz_begin = None
        obs_info.detector_num = detector.getId()
        obs_info.detector_unique_name = detector.getName()
        obs_info.detector_serial = detector.getSerial()
    return obs_info
