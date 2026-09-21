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

from collections.abc import MutableMapping
from typing import TYPE_CHECKING, Any

from astro_metadata_translator import ObservationInfo, VisitInfoTranslator

if TYPE_CHECKING:
    try:
        from lsst.afw.cameraGeom import Detector as LegacyDetector
        from lsst.afw.image import Exposure as LegacyExposure
        from lsst.afw.image import FilterLabel as LegacyFilterLabel
        from lsst.afw.image import VisitInfo as LegacyVisitInfo
    except ImportError:
        type LegacyDetector = Any  # type: ignore[no-redef]
        type LegacyExposure = Any  # type: ignore[no-redef]
        type LegacyFilterLabel = Any  # type: ignore[no-redef]
        type LegacyVisitInfo = Any  # type: ignore[no-redef]


def obs_info_from_legacy(
    md: MutableMapping[str, Any],
    visit_info: LegacyVisitInfo | None = None,
    detector: LegacyDetector | None = None,
    filter_label: LegacyFilterLabel | None = None,
) -> ObservationInfo:
    """Reconstruct as much of an observation info struct as possible from
    what's available from a legacy `lsst.afw.image.Exposure`.

    Parameters
    ----------
    md
        FITS header metadata as a dict-like object.
    visit_info
        The legacy visit info struct; correct but incomplete.
    detector
        The legacy detector object.
    filter_label
        The legacy struct holding filter information.
    """
    # Try to get an ObservationInfo from the primary header as if
    # it's a raw header. Else fallback.
    try:
        obs_info = ObservationInfo.from_header(md, quiet=True)
    except ValueError:
        # Unknown translator; fall back to VisitInfo.
        if visit_info is not None:
            from lsst.afw.image import setVisitInfoMetadata
            from lsst.daf.base import PropertyList

            pl = PropertyList()
            setVisitInfoMetadata(pl, visit_info)
            # Merge so that we still have access to butler provenance.
            md.update(pl)

        # Try the given header looking for VisitInfo hints.
        # We get lots of warnings if nothing can be found. Currently
        # no way to disable those without capturing them.
        obs_info = ObservationInfo.from_header(md, translator_class=VisitInfoTranslator, quiet=True)
    return _update_obs_info_from_legacy(obs_info, detector, filter_label)


def _update_obs_info_from_legacy(
    obs_info: ObservationInfo,
    detector: LegacyDetector | None = None,
    filter_label: LegacyFilterLabel | None = None,
) -> ObservationInfo:
    """Return a copy of an observation info struct updated with detector and
    filter information.
    """
    extra_md: dict[str, str | int] = {}

    if filter_label is not None and filter_label.hasPhysicalLabel():
        extra_md["physical_filter"] = filter_label.physicalLabel

    # Fill in detector metadata, check for consistency.
    # ObsInfo detector name and group can not be derived from
    # the getName() information without knowing how the components
    # are separated.
    if detector is not None:
        detector_md = {
            "detector_num": detector.getId(),
            "detector_unique_name": detector.getName(),
        }
        extra_md.update(detector_md)

    obs_info_updates: dict[str, str | int] = {}
    for k, v in extra_md.items():
        current = getattr(obs_info, k)
        if current is None:
            obs_info_updates[k] = v
            continue
        if current != v:
            raise RuntimeError(
                f"ObservationInfo contains value for '{k}' that is inconsistent "
                f"with given legacy object: {v} != {current}"
            )

    if obs_info_updates:
        obs_info = obs_info.model_copy(update=obs_info_updates)
    return obs_info
