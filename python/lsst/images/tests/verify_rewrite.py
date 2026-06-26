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

import difflib
import random
from collections.abc import Iterable, Set
from typing import TYPE_CHECKING, Any

import astropy.io.fits
import click
import fsspec
import numpy as np

if TYPE_CHECKING:
    import tqdm

    from lsst.daf.butler import Butler, DataCoordinate


@click.group("verify-rewrite")
def verify_rewrite() -> None:
    """Compare rewritten datasets against their originals in a butler repo."""


@verify_rewrite.command("stage4")
@click.argument("repo")
@click.argument("old_collection")
@click.argument("new_collection")
@click.option("--where", default="", help="Query string to constraint the comparison.")
@click.option("--old-prefix", default="", help="Prefix for the old dataset type names.")
@click.option("--new-prefix", default="future_", help="Prefix for the new dataset type names.")
def verify_stage4_rewrite(
    *, repo: str, old_collection: str, new_collection: str, where: str, old_prefix: str, new_prefix: str
) -> None:  # numpydoc ignore=PR01
    """Compare rewritten visit/difference images and downstream tables in
    NEW_COLLECTION against the originals in OLD_COLLECTION of REPO.
    """
    try:
        from lsst.afw.image import Exposure  # noqa: F401
        from lsst.daf.butler import Butler
    except ImportError as err:
        err.add_note("verify-rewrite requires a full Rubin development environment.")
        raise
    with Butler.from_config(repo) as butler:
        verifier = RewriteVerifier(
            butler,
            old_collection=old_collection,
            new_collection=new_collection,
            old_prefix=old_prefix,
            new_prefix=new_prefix,
        )
        # Query for and compare datasets with {visit, detector} data IDs.
        for data_id in verifier.process("visit_image", where, renamed=True):
            # Test visit_image vs. future_visit_image.
            verifier.compare_images("visit_image", data_id, renamed=True)
            verifier.require_compressed("visit_image", data_id, renamed=True)
            # Test difference_image_predetection, because it's downstream of
            # future_visit_image in the rewrite pipeline.
            verifier.compare_images("difference_image_predetection", data_id, renamed=False)
            # Test difference_image vs. future_visit_image.
            verifier.compare_images("difference_image", data_id, renamed=True)
            verifier.require_compressed("difference_image", data_id, renamed=True)
            # dia_source detector is downstream of difference_image; make sure
            # it's unchanged as a sanity check.
            verifier.compare_tables("dia_source_detector", data_id, ignore={"timeProcessedMjdTai"})
        # Query for and compare datasets with {visit, detector, tract} data
        # IDs that are downstream of difference_image and maybe visit_image.
        for data_id in verifier.process("object_forced_source_unstandardized", where):
            verifier.compare_tables("object_forced_source_unstandardized", data_id)
            verifier.compare_tables("dia_object_forced_source_unstandardized", data_id)


class RewriteVerifier:
    def __init__(
        self,
        butler: Butler,
        *,
        old_collection: str,
        new_collection: str,
        old_prefix: str,
        new_prefix: str,
    ) -> None:
        self.butler = butler
        self.old_collection = old_collection
        self.new_collection = new_collection
        self.old_prefix = old_prefix
        self.new_prefix = new_prefix
        self._progress: tqdm.tqdm | None = None

    def process(
        self, base_name: str, where: str, *dimensions: str, renamed: bool = False
    ) -> Iterable[DataCoordinate]:
        import tqdm

        base_dataset_type = self.butler.get_dataset_type(
            f"{self.new_prefix}{base_name}" if renamed else base_name
        )
        dimension_group = base_dataset_type.dimensions.union(self.butler.dimensions.conform(dimensions))
        with self.butler.query() as query:
            data_ids = list(
                tqdm.tqdm(
                    query.where(where)
                    .join_dataset_search(base_dataset_type.name, collections=[self.new_collection])
                    .data_ids(dimension_group),
                    desc=f"querying for {self.new_prefix}{base_name} in {self.new_collection}.",
                )
            )
        random.shuffle(data_ids)
        self._progress = tqdm.tqdm(data_ids)
        yield from self._progress

    def get_old(
        self, base_name: str, data_id: DataCoordinate, storage_class: str | None = None, renamed: bool = False
    ) -> Any | None:
        ref = self.butler.find_dataset(
            f"{self.old_prefix}{base_name}" if renamed else base_name,
            data_id,
            collections=[self.old_collection],
        )
        if ref is None:
            return None
        assert not ref.run.startswith(self.new_collection), f"Bad old run collection for {ref}."
        return self.butler.get(ref, storageClass=storage_class)

    def get_new(
        self, base_name: str, data_id: DataCoordinate, renamed: bool = False, storage_class: str | None = None
    ) -> Any | None:
        ref = self.butler.find_dataset(
            f"{self.new_prefix}{base_name}" if renamed else base_name,
            data_id,
            collections=[self.new_collection],
        )
        if ref is None or ref.run.startswith(self.old_collection):
            return None
        return self.butler.get(ref, storageClass=storage_class)

    def print(self, base_name: str, data_id: DataCoordinate, msg: str) -> None:
        if self._progress is not None:
            self._progress.write(f"{base_name}@{data_id}: {msg}")
        else:
            print(f"{base_name}@{data_id}: {msg}")

    def compare_images(self, base_name: str, data_id: DataCoordinate, renamed: bool = False) -> bool:
        old = self.get_old(base_name, data_id, renamed=renamed)
        new = self.get_new(base_name, data_id, renamed=renamed, storage_class="ExposureF")
        if old is None:
            if new is None:
                return False
            else:
                self.print(base_name, data_id, "old does not exist.")
                return True
        elif new is None:
            self.print(base_name, data_id, "new does not exist.")
            return True
        differences = []
        if not np.array_equal(old.image.array, new.image.array, equal_nan=True):
            differences.append("image")
        if not np.array_equal(old.mask.array, new.mask.array):
            differences.append("mask")
            self.print_mask_diff(base_name, data_id, old.mask, new.mask)
        if not np.array_equal(old.variance.array, new.variance.array):
            differences.append("variance")
        if differences:
            self.print(base_name, data_id, ", ".join(differences))
        return bool(differences)

    def compare_tables(self, base_name: str, data_id: DataCoordinate, ignore: Set[str] = frozenset()) -> bool:
        old = self.get_old(base_name, data_id, storage_class="ArrowAstropy")
        new = self.get_new(base_name, data_id, storage_class="ArrowAstropy")
        if old is None:
            if new is None:
                return False
            else:
                self.print(base_name, data_id, "old does not exist.")
                return True
        elif new is None:
            self.print(base_name, data_id, "new does not exist.")
            return True
        if len(old) != len(new):
            self.print(base_name, data_id, f"row counts differ ({len(old) != len(new)})")
            return True
        if old.colnames != new.colnames:
            self.print(base_name, data_id, "column names differ")
            return True
        differences = []
        for name in old.colnames:
            if name in ignore:
                continue
            try:
                if not np.array_equal(
                    old[name], new[name], equal_nan=issubclass(old[name].dtype.type, np.floating)
                ):
                    differences.append(name)
            except Exception as err:
                err.add_note(name)
                raise
        if differences:
            self.print(base_name, data_id, ", ".join(differences))
        return bool(differences)

    def require_compressed(self, base_name: str, data_id: DataCoordinate, renamed: bool = False) -> None:
        ref = self.butler.find_dataset(
            f"{self.new_prefix}{base_name}" if renamed else base_name,
            data_id,
            collections=[self.new_collection],
        )
        if ref is None:
            return
        path = self.butler.getURI(ref)
        fs: fsspec.AbstractFileSystem
        fs, fp = path.to_fsspec()
        with fs.open(fp) as stream:
            with astropy.io.fits.open(stream, disable_image_compression=True) as hdu_list:
                if (image_zcmptype := hdu_list["IMAGE"].header.get("ZCMPTYPE")) != "RICE_1":
                    self.print(base_name, data_id, f"IMAGE HDU has ZCMPTYPE={image_zcmptype!r}")
                if (mask_zcmptype := hdu_list["MASK"].header.get("ZCMPTYPE")) != "GZIP_2":
                    self.print(base_name, data_id, f"MASK HDU has ZCMPTYPE={mask_zcmptype!r}")
                if (variance_zcmptype := hdu_list["VARIANCE"].header.get("ZCMPTYPE")) != "RICE_1":
                    self.print(base_name, data_id, f"VARIANCE HDU has ZCMPTYPE={variance_zcmptype!r}")

    def print_log_diff(self, task_label: str, data_id: DataCoordinate) -> None:
        old = self.get_old(f"{task_label}_log", data_id)
        new = self.get_new(f"{task_label}_log", data_id)
        assert old is not None
        assert new is not None
        with open(f"{task_label}-{'_'.join(str(v) for v in data_id.required_values)}.diff", "w") as stream:
            for line in difflib.unified_diff([r.message for r in old], [r.message for r in new]):
                stream.write(line)
                stream.write("\n")

    def print_mask_diff(self, base_name: str, data_id: DataCoordinate, old_mask: Any, new_mask: Any) -> None:
        bitmasks: dict[str, int] = {
            name: old_mask.getPlaneBitMask(name) for name in old_mask.getMaskPlaneDict()
        }

        def interpret_mask_value(v: int) -> list[str]:
            return [name for name, bitmask in bitmasks.items() if (bitmask & v)]

        with open(f"{base_name}-{'_'.join(str(v) for v in data_id.required_values)}.mask", "w") as stream:
            for value, count in zip(*np.unique_counts(new_mask.array & ~old_mask.array)):
                if not value:
                    continue
                planes = interpret_mask_value(value)
                print(f"+ {{{', '.join(planes)}}}: {count}", file=stream)
            for value, count in zip(*np.unique_counts(old_mask.array & ~new_mask.array)):
                if not value:
                    continue
                planes = interpret_mask_value(value)
                print(f"- {{{', '.join(planes)}}}: {count}", file=stream)


if __name__ == "__main__":
    verify_rewrite()
