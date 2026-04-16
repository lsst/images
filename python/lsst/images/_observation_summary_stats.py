# This file is part of afw.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

__all__ = ("ObservationSummaryStats",)

import pydantic


def _default_corners() -> tuple[float, float, float, float]:
    return (float("nan"), float("nan"), float("nan"), float("nan"))


class ObservationSummaryStats(pydantic.BaseModel, ser_json_inf_nan="constants"):
    version: int = pydantic.Field(0, description="Version of the model.")

    psfSigma: float = pydantic.Field(float("nan"), description="PSF determinant radius (pixels).")

    psfArea: float = pydantic.Field(float("nan"), description="PSF effective area (pixels**2).")

    psfIxx: float = pydantic.Field(float("nan"), description="PSF shape Ixx (pixels**2).")

    psfIyy: float = pydantic.Field(float("nan"), description="PSF shape Iyy (pixels**2).")

    psfIxy: float = pydantic.Field(float("nan"), description="PSF shape Ixy (pixels**2).")

    ra: float = pydantic.Field(float("nan"), description="Bounding box center Right Ascension (degrees).")

    dec: float = pydantic.Field(float("nan"), description="Bounding box center Declination (degrees).")

    pixelScale: float = pydantic.Field(
        float("nan"), description="Measured detector pixel scale (arcsec/pixel)."
    )

    zenithDistance: float = pydantic.Field(
        float("nan"), description="Bounding box center zenith distance (degrees)."
    )

    expTime: float = pydantic.Field(float("nan"), description="Exposure time of the exposure (seconds).")

    zeroPoint: float = pydantic.Field(float("nan"), description="Mean zeropoint in detector (mag).")

    skyBg: float = pydantic.Field(float("nan"), description="Average sky background (ADU).")

    skyNoise: float = pydantic.Field(float("nan"), description="Average sky noise (ADU).")

    meanVar: float = pydantic.Field(float("nan"), description="Mean variance of the weight plane (ADU**2).")

    raCorners: tuple[float, float, float, float] = pydantic.Field(
        default_factory=_default_corners, description="Right Ascension of bounding box corners (degrees)."
    )

    decCorners: tuple[float, float, float, float] = pydantic.Field(
        default_factory=_default_corners, description="Declination of bounding box corners (degrees)."
    )

    astromOffsetMean: float = pydantic.Field(float("nan"), description="Astrometry match offset mean.")

    astromOffsetStd: float = pydantic.Field(float("nan"), description="Astrometry match offset stddev.")

    nPsfStar: int = pydantic.Field(0, description="Number of stars used for psf model.")

    psfStarDeltaE1Median: float = pydantic.Field(
        float("nan"), description="Psf stars median E1 residual (starE1 - psfE1)."
    )

    psfStarDeltaE2Median: float = pydantic.Field(
        float("nan"), description="Psf stars median E2 residual (starE2 - psfE2)."
    )

    psfStarDeltaE1Scatter: float = pydantic.Field(
        float("nan"), description="Psf stars MAD E1 scatter (starE1 - psfE1)."
    )

    psfStarDeltaE2Scatter: float = pydantic.Field(
        float("nan"), description="Psf stars MAD E2 scatter (starE2 - psfE2)."
    )

    psfStarDeltaSizeMedian: float = pydantic.Field(
        float("nan"), description="Psf stars median size residual (starSize - psfSize)."
    )

    psfStarDeltaSizeScatter: float = pydantic.Field(
        float("nan"), description="Psf stars MAD size scatter (starSize - psfSize)."
    )

    psfStarScaledDeltaSizeScatter: float = pydantic.Field(
        float("nan"), description="Psf stars MAD size scatter scaled by psfSize**2."
    )

    psfTraceRadiusDelta: float = pydantic.Field(
        float("nan"),
        description=(
            "Delta (max - min) of the model psf trace radius values evaluated on a grid of "
            "unmasked pixels (pixels)."
        ),
    )

    psfApFluxDelta: float = pydantic.Field(
        float("nan"),
        description=(
            "Delta (max - min) of the model psf aperture flux (with aperture radius of max(2, 3*psfSigma)) "
            "values evaluated on a grid of unmasked pixels."
        ),
    )

    psfApCorrSigmaScaledDelta: float = pydantic.Field(
        float("nan"),
        description=(
            "Delta (max - min) of the psf flux aperture correction factors scaled (divided) by the "
            "psfSigma evaluated on a grid of unmasked pixels."
        ),
    )

    maxDistToNearestPsf: float = pydantic.Field(
        float("nan"),
        description="Maximum distance of an unmasked pixel to its nearest model psf star (pixels).",
    )

    starEMedian: float = pydantic.Field(
        float("nan"),
        description=(
            "Median ellipticity (sqrt(starE1**2.0 + starE2**2.0)) of the stars used in the PSF model."
        ),
    )

    starUnNormalizedEMedian: float = pydantic.Field(
        float("nan"),
        description=(
            "Median un-normalized ellipticity (sqrt((starXX - starYY)**2.0 + "
            "(2.0*starXY)**2.0)) of the stars used in the PSF model."
        ),
    )

    effTime: float = pydantic.Field(
        float("nan"),
        description="Effective exposure time calculated from psfSigma, skyBg, and zeroPoint (seconds).",
    )

    effTimePsfSigmaScale: float = pydantic.Field(
        float("nan"), description="PSF scaling of the effective exposure time."
    )

    effTimeSkyBgScale: float = pydantic.Field(
        float("nan"), description="Sky background scaling of the effective exposure time."
    )

    effTimeZeroPointScale: float = pydantic.Field(
        float("nan"), description="Zeropoint scaling of the effective exposure time."
    )

    magLim: float = pydantic.Field(
        float("nan"),
        description=(
            "Magnitude limit at fixed SNR (default SNR=5) calculated from psfSigma, skyBg,"
            " zeroPoint, and readNoise."
        ),
    )
