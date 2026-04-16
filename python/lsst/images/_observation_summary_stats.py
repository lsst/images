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

__all__ = ("ObservationSummaryStats",)

import math

import pydantic


def _default_corners() -> tuple[float, float, float, float]:
    return (math.nan, math.nan, math.nan, math.nan)


class ObservationSummaryStats(pydantic.BaseModel, ser_json_inf_nan="constants"):
    version: int = pydantic.Field(0, description="Version of the model.")

    psfSigma: float = pydantic.Field(math.nan, description="PSF determinant radius (pixels).")

    psfArea: float = pydantic.Field(math.nan, description="PSF effective area (pixels**2).")

    psfIxx: float = pydantic.Field(math.nan, description="PSF shape Ixx (pixels**2).")

    psfIyy: float = pydantic.Field(math.nan, description="PSF shape Iyy (pixels**2).")

    psfIxy: float = pydantic.Field(math.nan, description="PSF shape Ixy (pixels**2).")

    ra: float = pydantic.Field(math.nan, description="Bounding box center Right Ascension (degrees).")

    dec: float = pydantic.Field(math.nan, description="Bounding box center Declination (degrees).")

    pixelScale: float = pydantic.Field(math.nan, description="Measured detector pixel scale (arcsec/pixel).")

    zenithDistance: float = pydantic.Field(
        math.nan, description="Bounding box center zenith distance (degrees)."
    )

    expTime: float = pydantic.Field(math.nan, description="Exposure time of the exposure (seconds).")

    zeroPoint: float = pydantic.Field(math.nan, description="Mean zeropoint in detector (mag).")

    skyBg: float = pydantic.Field(math.nan, description="Average sky background (ADU).")

    skyNoise: float = pydantic.Field(math.nan, description="Average sky noise (ADU).")

    meanVar: float = pydantic.Field(math.nan, description="Mean variance of the weight plane (ADU**2).")

    raCorners: tuple[float, float, float, float] = pydantic.Field(
        default_factory=_default_corners, description="Right Ascension of bounding box corners (degrees)."
    )

    decCorners: tuple[float, float, float, float] = pydantic.Field(
        default_factory=_default_corners, description="Declination of bounding box corners (degrees)."
    )

    astromOffsetMean: float = pydantic.Field(math.nan, description="Astrometry match offset mean.")

    astromOffsetStd: float = pydantic.Field(math.nan, description="Astrometry match offset stddev.")

    nPsfStar: int = pydantic.Field(0, description="Number of stars used for psf model.")

    psfStarDeltaE1Median: float = pydantic.Field(
        math.nan, description="Psf stars median E1 residual (starE1 - psfE1)."
    )

    psfStarDeltaE2Median: float = pydantic.Field(
        math.nan, description="Psf stars median E2 residual (starE2 - psfE2)."
    )

    psfStarDeltaE1Scatter: float = pydantic.Field(
        math.nan, description="Psf stars MAD E1 scatter (starE1 - psfE1)."
    )

    psfStarDeltaE2Scatter: float = pydantic.Field(
        math.nan, description="Psf stars MAD E2 scatter (starE2 - psfE2)."
    )

    psfStarDeltaSizeMedian: float = pydantic.Field(
        math.nan, description="Psf stars median size residual (starSize - psfSize)."
    )

    psfStarDeltaSizeScatter: float = pydantic.Field(
        math.nan, description="Psf stars MAD size scatter (starSize - psfSize)."
    )

    psfStarScaledDeltaSizeScatter: float = pydantic.Field(
        math.nan, description="Psf stars MAD size scatter scaled by psfSize**2."
    )

    psfTraceRadiusDelta: float = pydantic.Field(
        math.nan,
        description=(
            "Delta (max - min) of the model psf trace radius values evaluated on a grid of "
            "unmasked pixels (pixels)."
        ),
    )

    psfApFluxDelta: float = pydantic.Field(
        math.nan,
        description=(
            "Delta (max - min) of the model psf aperture flux (with aperture radius of max(2, 3*psfSigma)) "
            "values evaluated on a grid of unmasked pixels."
        ),
    )

    psfApCorrSigmaScaledDelta: float = pydantic.Field(
        math.nan,
        description=(
            "Delta (max - min) of the psf flux aperture correction factors scaled (divided) by the "
            "psfSigma evaluated on a grid of unmasked pixels."
        ),
    )

    maxDistToNearestPsf: float = pydantic.Field(
        math.nan,
        description="Maximum distance of an unmasked pixel to its nearest model psf star (pixels).",
    )

    starEMedian: float = pydantic.Field(
        math.nan,
        description=(
            "Median ellipticity (sqrt(starE1**2.0 + starE2**2.0)) of the stars used in the PSF model."
        ),
    )

    starUnNormalizedEMedian: float = pydantic.Field(
        math.nan,
        description=(
            "Median un-normalized ellipticity (sqrt((starXX - starYY)**2.0 + "
            "(2.0*starXY)**2.0)) of the stars used in the PSF model."
        ),
    )

    effTime: float = pydantic.Field(
        math.nan,
        description="Effective exposure time calculated from psfSigma, skyBg, and zeroPoint (seconds).",
    )

    effTimePsfSigmaScale: float = pydantic.Field(
        math.nan, description="PSF scaling of the effective exposure time."
    )

    effTimeSkyBgScale: float = pydantic.Field(
        math.nan, description="Sky background scaling of the effective exposure time."
    )

    effTimeZeroPointScale: float = pydantic.Field(
        math.nan, description="Zeropoint scaling of the effective exposure time."
    )

    magLim: float = pydantic.Field(
        math.nan,
        description=(
            "Magnitude limit at fixed SNR (default SNR=5) calculated from psfSigma, skyBg,"
            " zeroPoint, and readNoise."
        ),
    )
