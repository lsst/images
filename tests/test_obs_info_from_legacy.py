# This file is part of lsst-images.
#
# Developed for the LSST Data Management System.
#
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# Use of this source code is governed by a 3-clause BSD-style
# license that can be found in the LICENSE file.

"""Tests for ObservationInfo->VisitInfo->ObservationInfo round-trips.

Neither ObservationInfo nor VisitInfo live in `lsst.images`, but these tests
live here because the reconstruction logic they exercise lives in
`lsst.images.VisitImage.from_legacy` and `~lsst.images.VisitImage.read_legacy`.
"""

from __future__ import annotations

import dataclasses
import warnings
from functools import cached_property
from pathlib import Path
from typing import Any

import astropy.coordinates as coords
import astropy.io.fits
import astropy.units as u
import pytest
from astro_metadata_translator import ObservationInfo

from lsst.images import VisitImage
from lsst.images.cameras import Detector
from lsst.images.serialization import read_archive
from lsst.images.tests import (
    TemporaryButler,
    annotate_errors,
    assert_obs_metadata_fields_equal,
)

try:
    from lsst.afw.cameraGeom import Detector as LegacyDetector
    from lsst.afw.detection import GaussianPsf as LegacyGaussianPsf
    from lsst.afw.geom import makeCdMatrix, makeSkyWcs
    from lsst.afw.image import ExposureF as LegacyExposureF
    from lsst.afw.image import FilterLabel, stripVisitInfoKeywords
    from lsst.afw.image import VisitInfo as LegacyVisitInfo
    from lsst.daf.base import PropertyList
    from lsst.daf.butler import DimensionUniverse
    from lsst.geom import Point2D, SpherePoint, arcseconds, degrees
    from lsst.obs.base import MakeRawVisitInfoViaObsInfo, makeExposureRecordFromObsInfo

    HAVE_LEGACY = True
except ImportError:
    type LegacyVisitInfo = Any  # type: ignore[no-redef]
    HAVE_LEGACY = False

skip_no_legacy = pytest.mark.skipif(not HAVE_LEGACY, reason="afw etc could not be imported.")

DATA_DIR = Path(__file__).parent / "data" / "headers"
OBS_INFO_DIR = Path(__file__).parent / "data" / "obs_info"
DETECTORS_DIR = Path(__file__).parent / "data" / "detectors"


@dataclasses.dataclass(frozen=True)
class InstrumentCase:
    """Definition of a per-instrument test fixture and helpers for loading its
    data.
    """

    label: str
    test_data_filename: str
    obs_info_filename: str
    detector_filename: str

    def get_full_header(self) -> astropy.io.fits.Header:
        return astropy.io.fits.Header.fromtextfile(str(DATA_DIR / self.test_data_filename))

    def get_stripped_header(self) -> astropy.io.fits.Header:
        md = PropertyList.from_astropy_header(self.get_full_header())
        stripVisitInfoKeywords(md)
        for key in [  # extra cards stripped in ExposureFitsReader
            "MJD-OBS",
            "DATE-OBS",
            "DETNAME",
            "DETSER",
            "EXPID",
        ]:
            md.pop(key, ...)
        return md.to_astropy_header()

    @cached_property
    def true_obs_info(self) -> ObservationInfo:
        """The ground-truth ObservationInfo (translated from raw)."""
        return ObservationInfo.model_validate_json((OBS_INFO_DIR / self.obs_info_filename).read_text())

    def make_record(self) -> Any:
        """Build the `exposure` dimension record from the true
        `ObservationInfo`.
        """
        return makeExposureRecordFromObsInfo(self.true_obs_info, DimensionUniverse())

    @cached_property
    def legacy_detector(self) -> LegacyDetector:
        """The legacy `lsst.afw.cameraGeom.Detector` for this test case."""
        assert self.true_obs_info.detector_num is not None
        detector = read_archive(str(DETECTORS_DIR / self.detector_filename), Detector)
        assert detector.id == self.true_obs_info.detector_num
        return detector.to_legacy()

    def make_stripped_exposure(self) -> LegacyExposureF:
        """Build a test ``ExposureF`` with metadata content mimicking a legacy
        visit image that has been roundtripped through FITS at least once.
        """
        header = self.get_stripped_header()
        exposure = LegacyExposureF(16, 16)
        if self.true_obs_info.detector_exposure_id is not None:
            exposure.info.setId(self.true_obs_info.detector_exposure_id)
        exposure.setMetadata(PropertyList.from_astropy_header(header))
        visit_info = MakeRawVisitInfoViaObsInfo.observationInfo2visitInfo(self.true_obs_info)
        exposure.info.setVisitInfo(visit_info)
        wcs = makeSkyWcs(
            crpix=Point2D(8.0, 8.0),
            crval=SpherePoint(42.0 * degrees, -11.0 * degrees),  # arbitrary ICRS position; not asserted on
            cdMatrix=makeCdMatrix(scale=0.2 * arcseconds),
        )
        exposure.setWcs(wcs)
        exposure.setDetector(self.legacy_detector)
        exposure.setPsf(LegacyGaussianPsf(11, 11, 2.0))
        exposure.info.setFilter(
            FilterLabel(band="g", physical=self.true_obs_info.physical_filter or "UNKNOWN")
        )
        return exposure


# Fields that we can't recover from any of the inputs to obs_info_from_legacy,
# mapped to what we get instead.
UNRECOVERABLE_FIELDS = {
    "translator_name": "VisitInfo",
    "detector_name": None,
    "detector_group": None,
    "observing_day_offset": None,
    "altaz_end": None,
}


def assert_obs_info_equal(
    actual: ObservationInfo,
    expected: ObservationInfo,
    path: str,
    *,
    detector_serial: str | None,
    detector_unique_name: str,
) -> None:
    """Assert that two ObservationInfo structs are as equal as they can be
    after a from-legacy conversion.
    """
    for field in ObservationInfo.model_fields:
        if field == "warnings":
            continue
        expected_value = getattr(expected, field)
        if expected_value is None:
            continue
        actual_value = getattr(actual, field, None)
        if field in UNRECOVERABLE_FIELDS:
            if actual_value != UNRECOVERABLE_FIELDS[field]:
                raise AssertionError(
                    f"{field} via {path}: expected fallback "
                    f"{UNRECOVERABLE_FIELDS[field]!r}, got {actual_value!r}"
                )
            continue
        if field == "visit_id":
            if actual_value != actual.exposure_id:
                raise AssertionError(
                    f"visit_id via {path}: {actual_value!r} != exposure_id {actual.exposure_id!r}"
                )
            continue
        if field == "boresight_rotation_coord":
            if isinstance(expected_value, str) and isinstance(actual_value, str):
                if actual_value.casefold() != expected_value.casefold():
                    raise AssertionError(
                        f"{field} via {path}: {actual_value!r} != {expected_value!r} (ignoring case)"
                    )
                continue
        if field == "detector_serial":
            expected_value = detector_serial
        if field == "detector_unique_name":
            expected_value = detector_unique_name
        if actual_value is None:
            raise AssertionError(f"{field} was lost via {path}: expected {expected_value}")
        with annotate_errors(field):
            assert_obs_metadata_fields_equal(actual_value, expected_value, label=field)


def _point_to_astropy(point: Any) -> coords.SkyCoord:
    """Convert an `lsst.geom.SpherePoint` to an Astropy ICRS `SkyCoord`."""
    return coords.SkyCoord(
        ra=point.getRa().asRadians() * u.rad,
        dec=point.getDec().asRadians() * u.rad,
    )


def _point_to_altaz(point: Any) -> coords.AltAz:
    """Convert an `lsst.geom.SpherePoint` conventionally holding az/alt to
    an Astropy `AltAz` frame.
    """
    return coords.AltAz(
        az=coords.Longitude(point.getLongitude().asRadians(), u.rad),
        alt=coords.Latitude(point.getLatitude().asRadians(), u.rad),
    )


def assert_visit_info_equal(actual: LegacyVisitInfo, expected: LegacyVisitInfo) -> None:
    """Assert that two VisitInfo structs are equal after a roundtrip
    through `lsst.images.VisitImage`.
    """
    for name in (
        "id",
        "date",
        "ut1",
        "instrumentLabel",
        "scienceProgram",
        "observationType",
        "observationReason",
        "object",
        "observatory",
        "rotType",
        "weather",
        "exposureTime",
        "darkTime",
        "boresightAirmass",
        "focusZ",
        "era",
        "localEra",
        "boresightHourAngle",
        "boresightParAngle",
        "boresightRotAngle",
        "hasSimulatedContent",
        "isPersistable",
    ):
        with annotate_errors(name):
            assert_obs_metadata_fields_equal(getattr(actual, name), getattr(expected, name), label=name)
    with annotate_errors("boresightRaDec"):
        assert_obs_metadata_fields_equal(
            _point_to_astropy(actual.boresightRaDec),
            _point_to_astropy(expected.boresightRaDec),
            label="boresightRaDec",
        )
    with annotate_errors("boresightAzAlt"):
        assert_obs_metadata_fields_equal(
            _point_to_altaz(actual.boresightAzAlt),
            _point_to_altaz(expected.boresightAzAlt),
            label="boresightAzAlt",
        )


@skip_no_legacy
@pytest.mark.parametrize(
    "instrument_case",
    [
        InstrumentCase(
            label="DECam",
            test_data_filename="decam-c4d_150218_052850-N25.hdr",
            obs_info_filename="decam-c4d_150218_052850-N25.json",
            detector_filename="decam-c4d_150218_052850-N25.json",
        ),
        InstrumentCase(
            label="HSC",
            test_data_filename="hsc-HSCA90333426.hdr",
            obs_info_filename="hsc-HSCA90333426.json",
            detector_filename="hsc-HSCA90333426.json",
        ),
        InstrumentCase(
            label="LSSTCam",
            test_data_filename="lsstcam-MC_O_20250501_000292_R21_S11.hdr",
            obs_info_filename="lsstcam-MC_O_20250501_000292_R21_S11.json",
            detector_filename="lsstcam-MC_O_20250501_000292_R21_S11.json",
        ),
    ],
    ids=["DECam", "HSC", "LSSTCam"],
)
def test_from_and_read_legacy_preserve_obs_info(instrument_case: InstrumentCase) -> None:
    """ObservationInfo fields must survive from_legacy on a stripped header."""
    assert instrument_case.true_obs_info.instrument == instrument_case.label
    assert instrument_case.true_obs_info.exposure_id is not None
    exposure = instrument_case.make_stripped_exposure()
    exposure_record = instrument_case.make_record()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*filter label mismatch.*", category=UserWarning)
        result1 = VisitImage.from_legacy(exposure, exposure_record=exposure_record)
    assert_obs_info_equal(
        result1.obs_info,
        instrument_case.true_obs_info,
        "VisitImage.from_legacy on an in-memory stripped header",
        detector_serial=instrument_case.legacy_detector.getSerial(),
        detector_unique_name=instrument_case.legacy_detector.getName(),
    )
    roundtrip1 = result1.to_legacy()
    assert_visit_info_equal(roundtrip1.info.getVisitInfo(), exposure.info.getVisitInfo())
    with TemporaryButler(legacy="ExposureF") as helper:
        helper.butler.put(exposure, helper.legacy)
        visit_image_ref = helper.legacy.overrideStorageClass("VisitImage")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*filter label mismatch.*", category=UserWarning)
            try:
                result2 = helper.butler.get(visit_image_ref, parameters={"exposure_record": exposure_record})
            except Exception as err:
                pytest.fail(
                    f"butler.get as VisitImage raised {type(err).__name__}: {err} "
                    "-- every ObservationInfo field was lost."
                )
    roundtrip2 = result2.to_legacy()
    assert_visit_info_equal(roundtrip2.info.getVisitInfo(), exposure.info.getVisitInfo())
    assert_obs_info_equal(
        result2.obs_info,
        instrument_case.true_obs_info,
        "butler.get of an ExposureF dataset as VisitImage",
        detector_serial=instrument_case.legacy_detector.getSerial(),
        detector_unique_name=instrument_case.legacy_detector.getName(),
    )
    with TemporaryButler(legacy="ExposureF") as helper:
        helper.butler.put(exposure, helper.legacy)
        visit_image_ref = helper.legacy.overrideStorageClass("VisitImage")
        with pytest.raises(ValueError, match="exposure_record"):
            helper.butler.get(visit_image_ref)
