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

import random
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, cast

import astropy.io.fits
import click
import fsspec
import numpy as np

from lsst.images import (
    BackgroundMap,
    Box,
    DifferenceImage,
    DifferenceImageTemplateInfo,
    VisitImage,
    get_legacy_difference_image_mask_planes,
)
from lsst.images.convolution_kernels import ConvolutionKernel
from lsst.images.tests import compare_masked_image_to_legacy, compare_visit_image_to_legacy

if TYPE_CHECKING:
    import tqdm

    from lsst.daf.butler import Butler, DataCoordinate


# These defaults match the rewrite tasks used for DP2 in pipe_tasks. That makes
# this a bit of a cyclic DRY violation, but not a serious problem.
VISIT_IMAGE_BACKGROUNDS: tuple[str, ...] = ("subtracted", "skyCorr")
DIFFERENCE_IMAGE_BACKGROUNDS: tuple[str, ...] = ()


def _check_kernel(kernel: ConvolutionKernel | None) -> None:
    """Sanity-check a DifferenceImage kernel."""
    assert isinstance(kernel, ConvolutionKernel), f"kernel has type {type(kernel)}"
    assert kernel.bounds is not None
    assert kernel.bounds.bbox is not None
    xy = kernel.bounds.bbox.meshgrid(3)
    for x, y in zip(xy.x.ravel(), xy.y.ravel(), strict=True):
        im = kernel.compute_kernel_image(x=int(x), y=int(y))
        assert im.array.size > 0
        assert np.isfinite(im.array).all(), "kernel has non-finite values"
        assert np.abs(im.array).max() > 0, "kernel is identically zero"


def _check_templates(templates: list[DifferenceImageTemplateInfo] | None, detector_bbox: Box) -> None:
    """Sanity-check the DifferenceImage template-info list."""
    assert isinstance(templates, list), f"templates has type {type(templates)}"
    assert len(templates) > 0, "no template info attached"
    for info in templates:
        assert detector_bbox.contains(info.bounds.bbox), "template bounds outside detector"
        if info.psf_shape_flag:
            continue
        assert info.psf_shape_xx * info.psf_shape_yy - info.psf_shape_xy**2 > 0, (
            "template PSF shape is not a valid ellipse"
        )


def _check_backgrounds(backgrounds: BackgroundMap, bbox: Box, *, expected: Sequence[str] = ()) -> None:
    """Sanity-check the backgrounds attached to an image.

    ``expected`` names must be present. Each attached background's field must
    cover the image bbox and evaluate to finite values over a coarse grid, and
    the map's ``subtracted`` attribute must point at a present background.
    """
    present = set(backgrounds)
    for name in expected:
        assert name in present, f"expected background {name!r} not attached"
    for name, bg in backgrounds.items():
        field = bg.field
        assert field.bounds.bbox.contains(bbox), f"background {name!r} does not cover the image bbox"
        grid = field.bounds.bbox.meshgrid(2)
        vals = field(x=grid.x, y=grid.y)
        assert np.isfinite(vals).all(), f"background {name!r} has non-finite values"
    if (subtracted := backgrounds.subtracted) is not None:
        assert subtracted.name in present, "subtracted background designation not in map"


@click.command("verify-rewrite")
@click.argument("repo")
@click.argument("dataset_type")
@click.argument("collection")
@click.option("--where", default="", help="Query string to constraint the comparison.")
@click.option("--old-prefix", default="legacy_", help="Prefix for the old dataset type names.")
@click.option("--new-prefix", default="", help="Prefix for the new dataset type names.")
@click.option(
    "--require-compressed/--no-require-compressed",
    default=True,
    help="Check that the new data product is lossy-compressed.",
)
@click.option("--check-kernel/--no-check-kernel", default=True, help="Sanity-check DifferenceImage.kernel.")
@click.option(
    "--check-templates/--no-check-templates", default=True, help="Sanity-check DifferenceImage.templates."
)
@click.option(
    "--check-backgrounds/--no-check-backgrounds", default=True, help="Sanity-check attached backgrounds."
)
def verify_rewrite(
    *,
    repo: str,
    dataset_type: str,
    collection: str,
    where: str,
    require_compressed: bool,
    old_prefix: str,
    new_prefix: str,
    check_kernel: bool,
    check_templates: bool,
    check_backgrounds: bool,
) -> None:  # numpydoc ignore=PR01
    """Compare rewritten images in COLLECTION against the originals in
    COLLECTION of REPO.
    """
    try:
        from lsst.afw.image import Exposure  # noqa: F401
        from lsst.daf.butler import Butler
    except ImportError as err:
        err.add_note("verify-rewrite requires a full Rubin development environment.")
        raise
    with Butler.from_config(repo, collections=[collection]) as butler:
        verifier = RewriteVerifier(butler, dataset_type, old_prefix=old_prefix, new_prefix=new_prefix)
        # Query for and compare datasets.
        for data_id in verifier.process(where):
            # Test visit_image vs. future_visit_image.
            verifier.compare_images(
                data_id,
                check_kernel=check_kernel,
                check_templates=check_templates,
                check_backgrounds=check_backgrounds,
            )
            if require_compressed:
                verifier.require_compressed(data_id)

    if verifier.n_problems:
        raise click.exceptions.Exit(1)


class RewriteVerifier:
    def __init__(
        self,
        butler: Butler,
        base_dataset_type: str,
        *,
        old_prefix: str,
        new_prefix: str,
    ) -> None:
        self.butler = butler
        self.base_dataset_type = base_dataset_type
        self.old_prefix = old_prefix
        self.new_prefix = new_prefix
        self._progress: tqdm.tqdm | None = None
        self._n_problems = 0

    def _report_problem(self) -> None:
        self._n_problems += 1

    @property
    def n_problems(self) -> int:
        return self._n_problems

    def process(self, where: str, *dimensions: str) -> Iterable[DataCoordinate]:
        import tqdm

        old_dataset_type = self.butler.get_dataset_type(f"{self.old_prefix}{self.base_dataset_type}")
        dimension_group = old_dataset_type.dimensions.union(self.butler.dimensions.conform(dimensions))
        with self.butler.query() as query:
            data_ids = list(
                tqdm.tqdm(
                    query.where(where).join_dataset_search(old_dataset_type.name).data_ids(dimension_group),
                    desc=f"querying for {old_dataset_type.name}",
                )
            )
            new_count = query.datasets(f"{self.new_prefix}{self.base_dataset_type}").where(where).count()
            assert len(data_ids) == new_count, f"Count mismatch: new ({new_count}) != old ({len(data_ids)})."
        random.shuffle(data_ids)
        self._progress = tqdm.tqdm(data_ids)
        yield from self._progress

    def compare_images(
        self,
        data_id: DataCoordinate,
        *,
        check_kernel: bool = True,
        check_templates: bool = True,
        check_backgrounds: bool = True,
    ) -> None:
        old = self.butler.get(f"{self.old_prefix}{self.base_dataset_type}", data_id)
        new = self.butler.get(f"{self.new_prefix}{self.base_dataset_type}", data_id)
        expected_backgrounds = None
        plane_map = None
        if isinstance(new, DifferenceImage):
            expected_backgrounds = DIFFERENCE_IMAGE_BACKGROUNDS
            plane_map = get_legacy_difference_image_mask_planes()
            if check_kernel:
                _check_kernel(new.kernel)
            if check_templates:
                _check_templates(new.templates, new.bbox)
        if isinstance(new, VisitImage):
            try:
                compare_visit_image_to_legacy(
                    new,
                    old,
                    expect_view=False,
                    plane_map=plane_map,
                    check_photometric_scaling=False,
                    instrument=cast(str, data_id["instrument"]),
                    visit=cast(int, data_id["visit"]),
                    detector=cast(int, data_id["detector"]),
                )
            except Exception as err:
                self.print_error(data_id, err)
                return
            if expected_backgrounds is None:
                expected_backgrounds = VISIT_IMAGE_BACKGROUNDS
        else:
            try:
                compare_masked_image_to_legacy(new, old, expect_view=False, plane_map=plane_map)
            except Exception as err:
                self.print_error(data_id, err)
                return
        if check_backgrounds and expected_backgrounds is not None:
            _check_backgrounds(new.backgrounds, new.bbox, expected=expected_backgrounds)

    def require_compressed(self, data_id: DataCoordinate, renamed: bool = False) -> None:
        ref = self.butler.find_dataset(f"{self.new_prefix}{self.base_dataset_type}", data_id)
        assert ref is not None, f"new dataset for {data_id} is missing"
        path = self.butler.getURI(ref)
        fs: fsspec.AbstractFileSystem
        fs, fp = path.to_fsspec()
        with fs.open(fp) as stream:
            with astropy.io.fits.open(stream, disable_image_compression=True) as hdu_list:
                if (image_zcmptype := hdu_list["IMAGE"].header.get("ZCMPTYPE")) != "RICE_1":
                    self._report_problem()
                    self.print(data_id, f"IMAGE HDU has ZCMPTYPE={image_zcmptype!r}")
                if (mask_zcmptype := hdu_list["MASK"].header.get("ZCMPTYPE")) != "GZIP_2":
                    self._report_problem()
                    self.print(data_id, f"MASK HDU has ZCMPTYPE={mask_zcmptype!r}")
                if (variance_zcmptype := hdu_list["VARIANCE"].header.get("ZCMPTYPE")) != "RICE_1":
                    self._report_problem()
                    self.print(data_id, f"VARIANCE HDU has ZCMPTYPE={variance_zcmptype!r}")

    def print(self, data_id: DataCoordinate, msg: str) -> None:
        if self._progress is not None:
            self._progress.write(f"{data_id}: {msg}")
        else:
            print(f"{data_id}: {msg}")

    def print_error(self, data_id: DataCoordinate, err: Exception) -> None:
        self._report_problem()
        message = str(err)
        notes = getattr(err, "__notes__", ())
        if notes:
            # The comparison utilities attach the failing component's name as
            # an exception note (a hierarchy from outer to inner when blocks
            # are nested, though they never are today).  Prefix the message
            # with that path so it reads as
            #   "<data-id>\n   <component>: <message>"
            # rather than an indented note trailing the message.
            component = " -> ".join(notes)
            block = f"{data_id}\n   {component}: {message}"
        else:
            block = f"{data_id}: {message}"
        if self._progress is not None:
            self._progress.write(block)
        else:
            print(block)


if __name__ == "__main__":
    verify_rewrite()
