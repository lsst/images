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

import dataclasses
import math
from typing import TYPE_CHECKING, Any, ClassVar, Self, final, get_origin

import pydantic

from lsst.images.describe import DescribableMixin, FieldRole, Report, ReportField
from lsst.images.serialization import ArchiveTree, InputArchive, InvalidParameterError, OutputArchive

if TYPE_CHECKING:
    try:
        from lsst.afw.image import ExposureSummaryStats as LegacyExposureSummaryStats
    except ImportError:
        type LegacyExposureSummaryStats = Any  # type: ignore[no-redef]


def _default_corners() -> tuple[float, float, float, float]:
    return (math.nan, math.nan, math.nan, math.nan)


def _is_empty(value: Any) -> bool:
    """Return whether a summary-statistic value is unset.

    A value counts as unset if it is NaN, or an empty sequence, or a sequence
    whose entries are all unset.  Such fields carry no information and can be
    dropped when converting to or from the legacy representation, allowing the
    two representations to define different sets of fields as long as the
    fields they do not share are empty.
    """
    if isinstance(value, (list, tuple)):
        return all(_is_empty(item) for item in value)
    return isinstance(value, float) and math.isnan(value)


@final
class ObservationSummaryStats(ArchiveTree, DescribableMixin):
    """Various statistics obtained from a single observation."""

    SCHEMA_NAME: ClassVar[str] = "observation_summary_stats"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0.dev0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type]  # Assigned after class construction.

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

    psfAdaptiveThresholdValue: float = pydantic.Field(
        math.nan,
        description="Threshold value used in the adaptive threshold detection pass for PSF modelling.",
    )

    psfAdaptiveIncludeThresholdMultiplier: float = pydantic.Field(
        math.nan,
        description="Threshold multiplier used in the adaptive threshold detection pass for PSF modelling.",
    )

    nShapeletsStar: int = pydantic.Field(
        0,
        description="Number of sources used in the shapelet decomposition.",
    )

    shapeletsOnlyIqScore: float = pydantic.Field(
        math.nan,
        description=(
            "The dimensionless image quality score as determined from the shapelets decomposition "
            "that includes power only from the non-atmospheric decomposition coefficients. The "
            "score spans the range [0.0, 1.0] with lower values indicating better image quality."
        ),
    )

    shapeletsIqScore: float = pydantic.Field(
        math.nan,
        description=(
            "The dimensionless image quality score as determined from the shapelets decomposition "
            "that includes power from the median centroid offset between those used in the decomposition "
            "and those of the centroid slot in addition to non-atmospheric decomposition coefficients. "
            "The score spans the range [0.0, 1.0] with lower values indicating better image quality."
        ),
    )

    shapeletsCoeffs: tuple[float, ...] = pydantic.Field(
        default_factory=tuple,
        description="Coefficients from the PSF star shapelet decomposition.",
    )

    centroidDiffShapeletsVsSlotMedian: float = pydantic.Field(
        math.nan,
        description=(
            "Median centroid difference (sqrt((slot_x - shapelet_x)**2 + (slot_y - shapelet_y)**2)) for "
            "sources used in the shapelet decomposition (pixels)."
        ),
    )

    shapeletsStarEMedian: float = pydantic.Field(
        math.nan,
        description=(
            "Median ellipticity (sqrt(starE1**2.0 + starE2**2.0)) of the sources used in the "
            "shapelet decomposition."
        ),
    )

    shapeletsStarUnNormalizedEMedian: float = pydantic.Field(
        math.nan,
        description=(
            "Median un-normalized ellipticity (sqrt((starXX - starYY)**2.0 + (2.0*starXY)**2.0)) "
            "of the sources used in the shapelet decomposition (pixel**2)."
        ),
    )

    refCatSourceDensity: float = pydantic.Field(
        math.nan,
        description=(
            "Source density for the detector region as computed from the loaded reference catalog "
            "(number per degrees**2)."
        ),
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

    starComa1Median: float = pydantic.Field(
        math.nan,
        description=(
            "Coma-like higher-order moment combination: median M30 + M12 of the stars used in the PSF model."
        ),
    )

    starComa2Median: float = pydantic.Field(
        math.nan,
        description=(
            "Coma-like higher-order moment combination: median M21 + M03 of the stars used in the PSF model."
        ),
    )

    starTrefoil1Median: float = pydantic.Field(
        math.nan,
        description=(
            "Trefoil-like higher-order moment combination: median M30 - 3*M12 "
            "of the stars used in the PSF model."
        ),
    )

    starTrefoil2Median: float = pydantic.Field(
        math.nan,
        description=(
            "Trefoil-like higher-order moment combination: median 3*M21 - M03 "
            "of the stars used in the PSF model."
        ),
    )

    starKurtosisMedian: float = pydantic.Field(
        math.nan,
        description=(
            "Kurtosis-like higher-order moment combination: median M40 + 2*M22 + M04 "
            "of the stars used in the PSF model."
        ),
    )

    starE41Median: float = pydantic.Field(
        math.nan,
        description=(
            "Fourth-order ellipticity-like higher-order moment combination: median M40 - M04 "
            "of the stars used in the PSF model."
        ),
    )

    starE42Median: float = pydantic.Field(
        math.nan,
        description=(
            "Fourth-order ellipticity-like higher-order moment combination: median 2*(M31 + M13) "
            "of the stars used in the PSF model."
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

    psfTE1e1: float = pydantic.Field(
        math.nan,
        description=(
            "Per-exposure TE1e1 ~ <de1 de1> of PSF residual ellipticity, averaged over theta "
            "[0,1] arcmin via treecorr KK correlation. Dimensionless; used to form the "
            "full-survey TE1 metric."
        ),
    )

    psfTE1e2: float = pydantic.Field(
        math.nan,
        description=(
            "Per-exposure TE1e2 ~ <de2 de2> of PSF residual ellipticity, averaged over theta "
            "[0,1] arcmin via treecorr KK correlation. Dimensionless; used to form the "
            "full-survey TE1 metric."
        ),
    )

    psfTE1ex: float = pydantic.Field(
        math.nan,
        description=(
            "Per-exposure TE1ex ~ <de1 de2> of PSF residual ellipticity, averaged over theta "
            "[0,1] arcmin via treecorr KK correlation. Dimensionless; used to form the "
            "full-survey TE1 metric."
        ),
    )

    psfTE2e1: float = pydantic.Field(
        math.nan,
        description=(
            "Per-exposure TE2e1 ~ <de1 de1> of PSF residual ellipticity, averaged over theta "
            "[5,100] arcmin via treecorr KK correlation. Dimensionless; used to form the "
            "full-survey TE2 metric."
        ),
    )

    psfTE2e2: float = pydantic.Field(
        math.nan,
        description=(
            "Per-exposure TE2e2 ~ <de2 de2> of PSF residual ellipticity, averaged over theta "
            "[5,100] arcmin via treecorr KK correlation. Dimensionless; used to form the "
            "full-survey TE2 metric."
        ),
    )

    psfTE2ex: float = pydantic.Field(
        math.nan,
        description=(
            "Per-exposure TE2ex ~ <de1 de2> of PSF residual ellipticity, averaged over theta "
            "[5,100] arcmin via treecorr KK correlation. Dimensionless; used to form the "
            "full-survey TE2 metric."
        ),
    )

    psfTE3e1: float = pydantic.Field(
        math.nan,
        description=(
            "Per-exposure median-over-CCDs of TE3e1 ~ <de1 de1> of PSF residual ellipticity, "
            "where each CCD uses theta within [0,5] arcmin bins. Dimensionless; downstream "
            "pipelines take the 85th percentile over images to evaluate TE3."
        ),
    )

    psfTE3e2: float = pydantic.Field(
        math.nan,
        description=(
            "Per-exposure median-over-CCDs of TE3e2 ~ <de2 de2> of PSF residual ellipticity, "
            "where each CCD uses theta within [0,5] arcmin bins. Dimensionless; downstream "
            "pipelines take the 85th percentile over images to evaluate TE3."
        ),
    )

    psfTE3ex: float = pydantic.Field(
        math.nan,
        description=(
            "Per-exposure median-over-CCDs of TE3ex ~ <de1 de2> of PSF residual ellipticity, "
            "where each CCD uses theta within [0,5] arcmin bins. Dimensionless; downstream "
            "pipelines take the 85th percentile over images to evaluate TE3."
        ),
    )

    psfTE4e1: float = pydantic.Field(
        math.nan,
        description=(
            "Per-exposure median-over-CCDs of TE4e1 ~ <de1 de1> of PSF residual ellipticity, "
            "where each CCD uses theta within [5,20] arcmin bins. Dimensionless; downstream "
            "pipelines take the 85th percentile over images to evaluate TE4."
        ),
    )

    psfTE4e2: float = pydantic.Field(
        math.nan,
        description=(
            "Per-exposure median-over-CCDs of TE4e2 ~ <de2 de2> of PSF residual ellipticity, "
            "where each CCD uses theta within [5,20] arcmin bins. Dimensionless; downstream "
            "pipelines take the 85th percentile over images to evaluate TE4."
        ),
    )

    psfTE4ex: float = pydantic.Field(
        math.nan,
        description=(
            "Per-exposure median-over-CCDs of TE4ex ~ <de1 de2> of PSF residual ellipticity, "
            "where each CCD uses theta within [5,20] arcmin bins. Dimensionless; downstream "
            "pipelines take the 85th percentile over images to evaluate TE4."
        ),
    )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ObservationSummaryStats):
            return NotImplemented
        for name in ObservationSummaryStats.model_fields:
            if name in ArchiveTree.model_fields:
                # Parent class fields, not summary statistics.
                continue
            a = getattr(self, name)
            b = getattr(other, name)
            if isinstance(a, tuple) and isinstance(b, tuple):
                if len(a) != len(b):
                    return False
                for ai, bi in zip(a, b):
                    if ai != bi and not (math.isnan(ai) and math.isnan(bi)):
                        return False
            elif a != b and not (math.isnan(a) and math.isnan(b)):
                return False
        return True

    def _describe(self, **kwargs: Any) -> Report:
        """Return a `Report` describing these summary statistics.

        Parameters
        ----------
        **kwargs
            Unused; accepted for interface compatibility.
        """
        fields = [
            ReportField(label=name, value=getattr(self, name), role=FieldRole.DERIVED)
            for name in (
                "psfSigma",
                "psfArea",
                "zeroPoint",
                "skyBg",
                "expTime",
                "ra",
                "dec",
                "pixelScale",
            )
        ]
        return Report(type_name="ObservationSummaryStats", fields=fields)

    def deserialize(self, archive: InputArchive[Any], **kwargs: Any) -> Self:
        """Extract this object from an archive.

        Parameters
        ----------
        archive
            Archive to read from.
        **kwargs
            Optional parameters. Not supported by this class.
        """
        if kwargs:
            raise InvalidParameterError(
                f"Unrecognized parameters for ObservationSummaryStats: {set(kwargs.keys())}."
            )
        # The model we want *is* the ArchiveTree.
        return self

    def serialize(self, archive: OutputArchive[Any]) -> Self:
        """Write this object to an archive.

        Parameters
        ----------
        archive
            Archive to write to.
        """
        # Copy so write-time overrides (metadata, butler_info) applied
        # by serialize_root do not mutate this object.
        return self.model_copy(deep=True)

    @classmethod
    def from_legacy(cls, exposure_summary_stats: LegacyExposureSummaryStats) -> Self:
        """Return an `ObservationSummaryStats` from a legacy
        `lsst.afw.image.ExposureSummaryStats`.

        Parameters
        ----------
        exposure_summary_stats
            Legacy exposure summary statistics to convert.

        Notes
        -----
        Legacy fields that are empty (NaN) are dropped, so a legacy struct that
        carries fields unknown to this class is accepted as long as those
        fields are empty.  A legacy field that holds a real value but is
        unknown here raises `ValueError`, since dropping it would lose data.
        """
        known_fields = set(cls.model_fields)
        kwargs: dict[str, Any] = {}
        for name, value in dataclasses.asdict(exposure_summary_stats).items():
            if _is_empty(value):
                continue
            # Strip version since it carries no information (it is always 0
            # in all our existing files) and this class uses explicit schema
            # versioning.
            if name == "version":
                continue
            if name not in known_fields:
                raise ValueError(
                    f"Legacy field {name!r} has a value ({value!r}) but is not known to "
                    f"ObservationSummaryStats."
                )
            kwargs[name] = value
        return cls.model_validate(kwargs)

    def to_legacy(self) -> LegacyExposureSummaryStats:
        """Convert to an `lsst.afw.image.ExposureSummaryStats` instance.

        Notes
        -----
        Empty (NaN) fields are not passed to the legacy struct, so fields
        defined here that are unknown to the installed version of
        `~lsst.afw.image.ExposureSummaryStats` are dropped when empty.  A field
        that holds a real value but is unknown to the legacy struct raises
        `ValueError`, since dropping it would lose data.
        """
        from lsst.afw.image import ExposureSummaryStats as LegacyExposureSummaryStats

        legacy_fields = {field.name for field in dataclasses.fields(LegacyExposureSummaryStats)}
        kwargs: dict[str, Any] = {}
        for name, info in ObservationSummaryStats.model_fields.items():
            if name in ArchiveTree.model_fields:
                # Parent class fields, not summary statistics.
                continue
            value = getattr(self, name)
            if _is_empty(value):
                continue
            if name not in legacy_fields:
                raise ValueError(
                    f"Field {name!r} has a value ({value!r}) but is not supported by this "
                    f"version of lsst.afw.image.ExposureSummaryStats."
                )
            # Doing this in general is hard, so we handle the fields that we
            # know about and raise if somebody adds a field with a new type
            # without updating this function.
            if info.annotation in (float, int):
                kwargs[name] = value
            elif get_origin(info.annotation) is tuple:
                kwargs[name] = list(value)
            else:
                raise NotImplementedError(f"Unsupported field type: {info.annotation}.")
        return LegacyExposureSummaryStats(**kwargs)


# Can not assign to itself in construction so assign now.
ObservationSummaryStats.PUBLIC_TYPE = ObservationSummaryStats
