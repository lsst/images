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

__all__ = ()

import os
from typing import TYPE_CHECKING

import click
import numpy as np

if TYPE_CHECKING:
    from lsst.daf.butler import Butler, DatasetRef


from ._data_ids import DP2_COADD_DATA_ID, DP2_COADD_MISSING_CELL, DP2_VISIT_DETECTOR_DATA_ID


def extract_exposure(
    butler: Butler,
    output_path: str,
    dataset_ref: DatasetRef,
    shuffle: bool,
) -> None:
    """Load a subimage of a processed visit image from a butler repository
    and save it to testdata_images.
    """
    from lsst.afw.fits import (
        CompressionAlgorithm,
        CompressionOptions,
        DitherAlgorithm,
        QuantizationOptions,
        ScalingAlgorithm,
    )
    from lsst.geom import Box2I, Extent2I, Point2I

    exposure = butler.get(dataset_ref, parameters={"bbox": Box2I(Point2I(5, 4), Extent2I(256, 250))})
    if shuffle:
        indices = np.arange(exposure.image.array.size, dtype=int)
        rng = np.random.default_rng()
        rng.shuffle(indices)
        exposure.image.array[:, :] = exposure.image.array.flat[indices].reshape(250, 256)
        exposure.mask.array[:, :] = exposure.mask.array.flat[indices].reshape(250, 256)
        exposure.variance.array[:, :] = exposure.variance.array.flat[indices].reshape(250, 256)
    float_compression = CompressionOptions(
        algorithm=CompressionAlgorithm.RICE_1,
        tile_height=50,
        tile_width=64,
        quantization=QuantizationOptions(
            dither=DitherAlgorithm.SUBTRACTIVE_DITHER_2,
            scaling=ScalingAlgorithm.STDEV_MASKED,
            level=16,
            seed=747,
        ),
    )
    mask_compression = CompressionOptions(
        algorithm=CompressionAlgorithm.GZIP_2,
        tile_height=50,
        tile_width=64,
        quantization=None,
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    exposure.writeFits(
        output_path,
        imageOptions=float_compression,
        maskOptions=mask_compression,
        varianceOptions=float_compression,
    )


def extract_visit_image_background(
    butler: Butler,
    output_path: str,
    dataset_ref: DatasetRef,
) -> None:
    """Load the background model of a processed visit image from a butler
    repository and save it to testdata_images.
    """
    visit_image_background = butler.get(dataset_ref)
    visit_image_background.writeFits(output_path)


def extract_generic(
    butler: Butler,
    output_path: str,
    dataset_ref: DatasetRef,
) -> None:
    """Load a dataset and save it to testdata_images, assuming it has a
    writeFits method.
    """
    visit_summary = butler.get(dataset_ref)
    visit_summary.writeFits(output_path)


def extract_cell_coadd(
    butler: Butler,
    output_path: str,
    dataset_ref: DatasetRef,
    shuffle: bool,
) -> None:
    """Load a subimage of a cell coadd from a butler repository and save it
    to testdata_images.
    """
    from lsst.cell_coadds import MultipleCellCoadd

    full_cell_coadd = butler.get(dataset_ref)
    cell_coadd = MultipleCellCoadd(
        [
            full_cell_coadd.cells[x, y]
            for y in range(7, 11)
            for x in range(5, 8)
            if {"i": y, "j": x} != DP2_COADD_MISSING_CELL
        ],
        grid=full_cell_coadd.grid,
        outer_cell_size=full_cell_coadd.outer_cell_size,
        psf_image_size=full_cell_coadd.psf_image_size,
        common=full_cell_coadd.common,
    )
    if shuffle:
        rng = np.random.default_rng()
        for cell in cell_coadd.cells.values():
            indices = np.arange(cell.outer.image.array.size, dtype=int)
            rng.shuffle(indices)
            cell.outer.image.array[:, :] = cell.outer.image.array.flat[indices].reshape(150, 150)
            cell.outer.mask.array[:, :] = cell.outer.mask.array.flat[indices].reshape(150, 150)
            cell.outer.variance.array[:, :] = cell.outer.variance.array.flat[indices].reshape(150, 150)
            for n in cell.outer.noise_realizations:
                n.array[:, :] = n.array.flat[indices].reshape(150, 150)
            if cell.outer.mask_fractions is not None:
                cell.outer.mask_fractions.array[:, :] = cell.outer.mask_fractions.array.flat[indices].reshape(
                    150, 150
                )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cell_coadd.writeFits(output_path)


def extract_camera(butler: Butler, output_path: str, dataset_ref: DatasetRef) -> None:
    """Read camera geometry from a butler repository and save it to
    testdata_images.
    """
    camera = butler.get(dataset_ref)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    camera.writeFits(output_path)


def extract_skymap(butler: Butler, output_path: str, dataset_ref: DatasetRef) -> None:
    """Read a skymap definition from a butler repository and save it to
    testdata_images.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    (path,) = butler.retrieveArtifacts(
        [dataset_ref],
        destination=os.path.dirname(output_path),
        transfer="copy",
        preserve_path=False,
        overwrite=True,
    )
    if path.ospath != output_path:
        os.rename(path.ospath, output_path)


def find_dataset_or_raise(
    butler: Butler, dataset_type: str, *, collections: str | None = None, **kwargs
) -> DatasetRef:
    """Call `lsst.daf.butler.Butler.find_dataset` with the given arguments and
    raise `LookupError` if it returns `None`.
    """
    ref = butler.find_dataset(dataset_type, collections=collections, **kwargs)
    if ref is None:
        raise LookupError(f"Could not find dataset {dataset_type} with data ID {kwargs}.")
    return ref


@click.group("extract_test_data")
def extract_test_data() -> None:
    """Extract test fixtures from a Rubin data repository."""


@extract_test_data.command("dp2")
@click.option("-b", "--butler-repo", help="Path to the butler repository.")
@click.option("-d", "--testdata-dir", help="Path to the testdata_images directory.")
@click.option(
    "-c",
    "--collection",
    default="LSSTCam/runs/DRP/DP2",
    help="Collection to use for most data products.",
)
@click.option(
    "--visit-images/--no-visit-images",
    default=True,
    help="Whether to extract [preliminary_]visit_image datasets.",
)
@click.option(
    "--difference-images/--no-difference-images",
    default=True,
    help="Whether to extract difference_image datasets.",
)
@click.option(
    "--coadds/--no-coadds",
    default=True,
    help="Whether to extract coadd datasets.",
)
@click.option(
    "--camera/--no-camera",
    default=True,
    help="Whether to extract the camera.",
)
@click.option(
    "--skymap/--no-skymap",
    default=True,
    help="Whether to extract the skymap.",
)
def extract_dp2(
    butler_repo: str | None,
    testdata_dir: str | None,
    collection: str,
    *,
    visit_images: bool,
    difference_images: bool,
    coadds: bool,
    camera: bool,
    skymap: bool,
) -> None:
    """Extract test data from a butler repository."""
    try:
        # lsst.afw.image is imported only to confirm a full Rubin pipelines
        # environment is present: daf.butler and lsst.utils are available
        # from the pip-installed package and so cannot gate on their own.
        import lsst.afw.image  # noqa: F401
        from lsst.daf.butler import Butler
        from lsst.utils import getPackageDir
    except ImportError as err:
        err.add_note(
            "Updating the test data requires a full Rubin development enviroment with at least "
            "'afw', 'obs_base', 'meas_extensions_psfex', 'meas_extensions_piff' and 'cell_coadds' "
            "importable. This is not necessary for just running the tests."
        )
        raise
    if butler_repo is None:
        butler_repo = "dp2_prep"
    if testdata_dir is None:
        testdata_dir = getPackageDir("testdata_images")
    butler = Butler.from_config(butler_repo, collections=[collection])
    if visit_images:
        extract_exposure(
            butler,
            os.path.join(testdata_dir, "dp2", "legacy", "visit_image.fits"),
            find_dataset_or_raise(butler, "visit_image", **DP2_VISIT_DETECTOR_DATA_ID),
            shuffle=True,
        )
        extract_visit_image_background(
            butler,
            os.path.join(testdata_dir, "dp2", "legacy", "visit_image_background.fits"),
            find_dataset_or_raise(butler, "visit_image_background", **DP2_VISIT_DETECTOR_DATA_ID),
        )
        extract_generic(
            butler,
            os.path.join(testdata_dir, "dp2", "legacy", "visit_summary.fits"),
            find_dataset_or_raise(butler, "visit_summary", **DP2_VISIT_DETECTOR_DATA_ID),
        )
        extract_exposure(
            butler,
            os.path.join(testdata_dir, "dp2", "legacy", "preliminary_visit_image.fits"),
            find_dataset_or_raise(butler, "preliminary_visit_image", **DP2_VISIT_DETECTOR_DATA_ID),
            shuffle=True,
        )
        extract_visit_image_background(
            butler,
            os.path.join(testdata_dir, "dp2", "legacy", "preliminary_visit_image_background.fits"),
            find_dataset_or_raise(butler, "preliminary_visit_image_background", **DP2_VISIT_DETECTOR_DATA_ID),
        )
    if difference_images:
        extract_exposure(
            butler,
            os.path.join(testdata_dir, "dp2", "legacy", "difference_image.fits"),
            find_dataset_or_raise(butler, "difference_image", **DP2_VISIT_DETECTOR_DATA_ID),
            shuffle=True,
        )
        extract_exposure(
            butler,
            os.path.join(testdata_dir, "dp2", "legacy", "template_detector.fits"),
            find_dataset_or_raise(butler, "template_detector", **DP2_VISIT_DETECTOR_DATA_ID),
            shuffle=True,
        )
        extract_generic(
            butler,
            os.path.join(testdata_dir, "dp2", "legacy", "difference_kernel.fits"),
            find_dataset_or_raise(butler, "difference_kernel", **DP2_VISIT_DETECTOR_DATA_ID),
        )
    if coadds:
        extract_cell_coadd(
            butler,
            os.path.join(
                testdata_dir,
                "dp2",
                "legacy",
                "deep_coadd_cell_predetection.fits",
            ),
            find_dataset_or_raise(butler, "deep_coadd_cell_predetection", **DP2_COADD_DATA_ID),
            shuffle=True,
        )
    if skymap:
        extract_skymap(
            butler,
            os.path.join(
                testdata_dir,
                "dp2",
                "legacy",
                "skyMap.pickle",
            ),
            find_dataset_or_raise(butler, "skyMap", skymap=DP2_COADD_DATA_ID["skymap"]),
        )
    if camera:
        extract_camera(
            butler,
            os.path.join(
                testdata_dir,
                "dp2",
                "legacy",
                "camera.fits",
            ),
            find_dataset_or_raise(butler, "camera", instrument="LSSTCam"),
        )


if __name__ == "__main__":
    extract_test_data()
