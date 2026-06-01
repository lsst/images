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

"""Generate v1 reference JSON fixtures for the schema-versioning tests.

Run from the repo root via:

    .pyenv/bin/python -m lsst.images.tests._make_schema_fixtures

This is a developer tool, not invoked from CI. It overwrites every
``<schema_name>.json`` file under ``tests/data/schema_v1/`` (excluding
the ``legacy/`` subdirectory) so it should only be run when intentionally
regenerating fixtures (e.g. after a schema_version bump).

Each builder constructs a minimal valid in-memory instance, runs it
through ``json.write`` with ``path=None`` (so we get the tree without
writing it), and dumps the resulting JSON dict to disk.

Some types require external test data (PSFEx, Piff, full Detector,
CellCoadd, CoaddProvenance) and their builders raise
``NotImplementedError`` to mark them as unsupported here. Tests that
operate on the fixtures must skip the missing names gracefully.
"""

from __future__ import annotations

__all__ = ("BUILDERS", "FIXTURE_DIR", "build_fixture", "main")

import json
import pathlib
from collections.abc import Callable
from typing import Any

import numpy as np

from .. import json as images_json
from .._cell_grid import CellGrid, CellGridBounds
from .._geom import Box

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
FIXTURE_DIR = _REPO_ROOT / "tests" / "data" / "schema_v1"
_DETECTOR_FIXTURE = _REPO_ROOT / "tests" / "data" / "detector.json"


def _serialize_to_dict(obj: Any) -> dict[str, Any]:
    """Run ``obj.serialize`` through a JSON archive and return a dict."""
    tree = images_json.write(obj)
    dumped = tree.model_dump(mode="json")
    if not isinstance(dumped, dict):
        raise TypeError(f"Expected dict from model_dump, got {type(dumped).__name__}.")
    return dumped


# -- Image / Mask / MaskedImage / VisitImage -------------------------------


def build_image() -> dict[str, Any]:
    """Build a minimal Image fixture."""
    from .. import Image

    image = Image(np.zeros((4, 4), dtype=np.float32))
    return _serialize_to_dict(image)


def build_mask() -> dict[str, Any]:
    """Build a minimal Mask fixture."""
    from .. import Mask, MaskPlane, MaskSchema

    mask = Mask(
        np.zeros((4, 4, 1), dtype=np.uint8),
        schema=MaskSchema([MaskPlane("BAD", "Bad pixel.")]),
    )
    return _serialize_to_dict(mask)


def build_masked_image() -> dict[str, Any]:
    """Build a minimal MaskedImage fixture."""
    from .. import Image, MaskedImage, MaskPlane, MaskSchema

    masked_image = MaskedImage(
        Image(np.zeros((4, 4), dtype=np.float32)),
        mask_schema=MaskSchema([MaskPlane("BAD", "Bad pixel.")]),
    )
    return _serialize_to_dict(masked_image)


def build_visit_image() -> dict[str, Any]:
    """Build a minimal VisitImage fixture."""
    import astropy.units as u
    import numpy as np
    from astro_metadata_translator import ObservationInfo

    from .. import (
        Box,
        DetectorFrame,
        Image,
        MaskPlane,
        MaskSchema,
        ObservationSummaryStats,
        VisitImage,
    )
    from ..cameras import Detector
    from ..json import read as read_json
    from ..psfs import GaussianPointSpreadFunction
    from ._creation import make_random_projection

    rng = np.random.default_rng(500)
    det_frame = DetectorFrame(instrument="Inst", visit=1234, detector=1, bbox=Box.factory[1:32, 1:32])
    mask_schema = MaskSchema([MaskPlane("BAD", "Bad pixel.")])
    obs_info = ObservationInfo(instrument="LSSTCam", detector_num=4, physical_filter="r1")
    summary_stats = ObservationSummaryStats(psfSigma=2.5, zeroPoint=31.4)
    psf = GaussianPointSpreadFunction(2.5, stamp_size=11, bounds=Box.factory[-5:6, -5:6])
    projection = make_random_projection(rng, det_frame, Box.factory[1:32, 1:32])
    detector, _, _ = read_json(Detector, str(_DETECTOR_FIXTURE))

    visit_image = VisitImage(
        Image(np.zeros((4, 4), dtype=np.float32), unit=u.nJy),
        psf=psf,
        mask_schema=mask_schema,
        projection=projection,
        obs_info=obs_info,
        summary_stats=summary_stats,
        detector=detector,
        band="r",
    )
    return _serialize_to_dict(visit_image)


def build_color_image() -> dict[str, Any]:
    """Build a minimal ColorImage fixture."""
    import numpy as np

    from .. import Box, ColorImage, TractFrame

    rng = np.random.default_rng(7)
    pixel_frame = TractFrame(skymap="test_skymap", tract=33, bbox=Box.factory[:8, :8])
    bbox = Box.factory[:4, :4]
    from ._creation import make_random_projection

    projection = make_random_projection(rng, pixel_frame, pixel_frame.bbox)
    array = rng.integers(low=0, high=255, size=bbox.shape + (3,), dtype=np.uint8)
    color_image = ColorImage(array, bbox=bbox, projection=projection)
    return _serialize_to_dict(color_image)


# -- PSFs ------------------------------------------------------------------


def build_gaussian_psf() -> dict[str, Any]:
    """Build a minimal GaussianPSF fixture."""
    from .. import Box
    from ..psfs import GaussianPointSpreadFunction

    psf = GaussianPointSpreadFunction(2.0, stamp_size=11, bounds=Box.factory[-5:6, -5:6])
    return _serialize_to_dict(psf)


def build_piff_psf() -> dict[str, Any]:
    """Piff PSF — requires real Piff data; not reproducible here."""
    raise NotImplementedError("Piff fixtures require external test data.")


def build_psfex_psf() -> dict[str, Any]:
    """PSFEx PSF — requires real PSFEx data; not reproducible here."""
    raise NotImplementedError("PSFEx fixtures require external test data.")


def build_cell_psf() -> dict[str, Any]:
    """Build a minimal CellPointSpreadFunction fixture."""
    import numpy as np

    from .. import YX, Box
    from ..cells._psf import CellPointSpreadFunction

    grid = CellGrid(bbox=Box.factory[0:8, 0:8], cell_shape=YX(y=4, x=4))
    bounds = CellGridBounds(grid=grid, bbox=grid.bbox)
    # 4-d array (cells_y, cells_x, kernel_y, kernel_x), kernel must be odd.
    array = np.zeros((2, 2, 5, 5), dtype=np.float32)
    array[..., 2, 2] = 1.0
    psf = CellPointSpreadFunction(array, bounds=bounds)
    return _serialize_to_dict(psf)


# -- Fields ----------------------------------------------------------------


def build_chebyshev_field() -> dict[str, Any]:
    """Build a minimal ChebyshevField fixture."""
    import numpy as np

    from ..fields import ChebyshevField

    field = ChebyshevField(Box.factory[6:32, -7:26], np.array([[0.5, -0.25], [0.40, 0.0]]))
    return _serialize_to_dict(field)


def build_spline_field() -> dict[str, Any]:
    """SplineField — serializes only via FITS (data: ArrayReferenceModel)."""
    raise NotImplementedError(
        "SplineField currently requires the FITS archive (its data field is "
        "ArrayReferenceModel rather than InlineArrayModel-or-ArrayReference)."
    )


def build_sum_field() -> dict[str, Any]:
    """Build a minimal SumField fixture."""
    import numpy as np

    from ..fields import ChebyshevField, SumField

    box = Box.factory[6:32, -7:26]
    a = ChebyshevField(box, np.array([[0.5, -0.25], [0.40, 0.0]]))
    b = ChebyshevField(box, np.array([[0.1, 0.0], [0.0, 0.0]]))
    field = SumField([a, b])
    return _serialize_to_dict(field)


def build_product_field() -> dict[str, Any]:
    """Build a minimal ProductField fixture."""
    import numpy as np

    from ..fields import ChebyshevField, ProductField

    box = Box.factory[6:32, -7:26]
    a = ChebyshevField(box, np.array([[0.5, -0.25], [0.40, 0.0]]))
    b = ChebyshevField(box, np.array([[0.1, 0.0], [0.0, 0.0]]))
    field = ProductField([a, b])
    return _serialize_to_dict(field)


# -- Backgrounds & aperture corrections ------------------------------------


def build_background_map() -> dict[str, Any]:
    """Build a minimal BackgroundMap fixture."""
    import numpy as np

    from .. import BackgroundMap
    from ..fields import ChebyshevField

    bg_map = BackgroundMap()
    bg_map.add(
        "standard",
        ChebyshevField(Box.factory[0:32, 0:32], np.array([[2.0]])),
        description="Background subtracted from the image.",
        is_subtracted=True,
    )
    return _serialize_to_dict(bg_map)


def build_aperture_correction_map() -> dict[str, Any]:
    """Build a minimal ApertureCorrectionMap fixture."""
    import numpy as np

    from ..aperture_corrections import (
        ApertureCorrectionMap,
        ApertureCorrectionMapSerializationModel,
    )
    from ..fields import ChebyshevField
    from ..json import JsonOutputArchive

    ap_corr: ApertureCorrectionMap = {
        "flux1": ChebyshevField(Box.factory[0:32, 0:32], np.array([0.75])),
        "flux2": ChebyshevField(Box.factory[0:32, 0:32], np.array([0.625])),
    }
    archive = JsonOutputArchive()
    tree = ApertureCorrectionMapSerializationModel.serialize(ap_corr, archive)
    archive.finish(tree)
    dumped = tree.model_dump(mode="json")
    if not isinstance(dumped, dict):
        raise TypeError(f"Expected dict, got {type(dumped).__name__}.")
    return dumped


# -- Detector / Cameras ----------------------------------------------------


def build_detector() -> dict[str, Any]:
    """Build a Detector fixture by reading the bundled detector.json."""
    from ..cameras import Detector
    from ..json import read as read_json

    detector, _, _ = read_json(Detector, str(_DETECTOR_FIXTURE))
    return _serialize_to_dict(detector)


def build_camera_frame_set() -> dict[str, Any]:
    """CameraFrameSet — requires AST representation from a legacy camera."""
    raise NotImplementedError("CameraFrameSet fixtures require external test data.")


# -- Transforms ------------------------------------------------------------


def build_transform() -> dict[str, Any]:
    """Build a minimal identity Transform fixture."""
    from .. import Box, DetectorFrame, Transform

    frame = DetectorFrame(instrument="Inst", visit=1234, detector=1, bbox=Box.factory[:5, :4])
    transform = Transform.identity(frame)
    return _serialize_to_dict(transform)


def build_projection() -> dict[str, Any]:
    """Build a minimal Projection fixture."""
    import numpy as np

    from .. import Box, DetectorFrame
    from ._creation import make_random_projection

    rng = np.random.default_rng(0)
    frame = DetectorFrame(instrument="Inst", visit=1234, detector=1, bbox=Box.factory[1:32, 1:32])
    projection = make_random_projection(rng, frame, Box.factory[1:32, 1:32])
    return _serialize_to_dict(projection)


# -- Cell coadd & provenance -----------------------------------------------


def build_cell_coadd() -> dict[str, Any]:
    """CellCoadd — requires lsst.cell_coadds and a real legacy file."""
    raise NotImplementedError("CellCoadd fixtures require external test data.")


def build_coadd_provenance() -> dict[str, Any]:
    """Build a minimal CoaddProvenance fixture with one input row."""
    from .._polygon import Polygon
    from ..cells._provenance import CoaddProvenance

    inputs = CoaddProvenance.make_empty_input_table(1)
    inputs["instrument"][0] = "LSSTCam"
    inputs["visit"][0] = 1
    inputs["detector"][0] = 1
    inputs["physical_filter"][0] = "r"
    inputs["day_obs"][0] = 20260101
    inputs["polygon"][0] = Polygon(
        x_vertices=[0.0, 1.0, 1.0, 0.0],
        y_vertices=[0.0, 0.0, 1.0, 1.0],
    )
    contributions = CoaddProvenance.make_empty_contribution_table(0)
    provenance = CoaddProvenance(inputs=inputs, contributions=contributions)
    return _serialize_to_dict(provenance)


# -- Builders registry -----------------------------------------------------

BUILDERS: dict[str, Callable[[], dict[str, Any]]] = {
    "image": build_image,
    "mask": build_mask,
    "masked_image": build_masked_image,
    "visit_image": build_visit_image,
    "color_image": build_color_image,
    "gaussian_psf": build_gaussian_psf,
    "piff_psf": build_piff_psf,
    "psfex_psf": build_psfex_psf,
    "cell_psf": build_cell_psf,
    "chebyshev_field": build_chebyshev_field,
    "spline_field": build_spline_field,
    "sum_field": build_sum_field,
    "product_field": build_product_field,
    "background_map": build_background_map,
    "aperture_correction_map": build_aperture_correction_map,
    "detector": build_detector,
    "camera_frame_set": build_camera_frame_set,
    "transform": build_transform,
    "projection": build_projection,
    "cell_coadd": build_cell_coadd,
    "coadd_provenance": build_coadd_provenance,
}


def build_fixture(name: str) -> dict[str, Any]:
    """Build a single fixture by name; raises ``KeyError`` if unknown."""
    return BUILDERS[name]()


def main() -> None:
    """Build and write all supported fixtures to ``FIXTURE_DIR``."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    skipped = []
    for name, builder in BUILDERS.items():
        try:
            tree = builder()
        except NotImplementedError as exc:
            skipped.append((name, str(exc)))
            continue
        out = FIXTURE_DIR / f"{name}.json"
        out.write_text(json.dumps(tree, indent=2, sort_keys=False) + "\n")
        print(f"wrote {out}")
    if skipped:
        print()
        print("Skipped (require external test data):")
        for name, reason in skipped:
            print(f"  {name}: {reason}")


if __name__ == "__main__":
    main()
