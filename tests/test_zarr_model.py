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

import unittest

import numpy as np

try:
    import zarr

    from lsst.images.zarr._common import LSST_NS, LSST_VERSION, OME_NS, OME_VERSION
    from lsst.images.zarr._model import (
        CfFlagAttributes,
        MaskPlaneEntry,
        OmeMultiscale,
        ZarrArray,
        ZarrAttributes,
        ZarrDocument,
        ZarrGroup,
        build_image_array_attrs,
    )

    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrAttributesTestCase(unittest.TestCase):
    """Tests for `ZarrAttributes` namespacing and round-tripping."""

    def test_dump_separates_namespaces(self) -> None:
        attrs = ZarrAttributes()
        attrs.lsst["archive_class"] = "MaskedImage"
        attrs.ome["multiscales"] = [{"name": "image"}]
        attrs.extra["_ARRAY_DIMENSIONS"] = ["y", "x"]
        attrs.extra["units"] = "adu"
        dumped = attrs.dump()
        self.assertEqual(dumped[LSST_NS]["archive_class"], "MaskedImage")
        self.assertEqual(dumped[LSST_NS]["version"], LSST_VERSION)
        self.assertEqual(dumped[OME_NS]["multiscales"], [{"name": "image"}])
        self.assertEqual(dumped[OME_NS]["version"], OME_VERSION)
        # CF / xarray attrs sit at the top level, not inside lsst: or ome:.
        self.assertEqual(dumped["_ARRAY_DIMENSIONS"], ["y", "x"])
        self.assertEqual(dumped["units"], "adu")

    def test_load_preserves_unknown_keys(self) -> None:
        # Forward compatibility: unknown lsst.* keys must survive a
        # load -> dump round-trip.
        raw = {
            LSST_NS: {
                "version": LSST_VERSION,
                "archive_class": "Image",
                "future_thing": {"x": 1},
            },
            OME_NS: {"version": OME_VERSION, "multiscales": []},
            "_ARRAY_DIMENSIONS": ["y", "x"],
            "units": "adu",
        }
        attrs = ZarrAttributes.load(raw)
        dumped = attrs.dump()
        self.assertEqual(dumped[LSST_NS]["future_thing"], {"x": 1})
        self.assertEqual(dumped["units"], "adu")


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrArrayTestCase(unittest.TestCase):
    """Tests for `ZarrArray` lazy backing and slice forwarding."""

    def test_lazy_data_after_from_zarr(self) -> None:
        store = zarr.storage.MemoryStore()
        root = zarr.create_group(store=store, zarr_format=3)
        zarr_array = root.create_array(name="image", shape=(8, 8), chunks=(4, 4), dtype="float32")
        zarr_array[:] = np.arange(64, dtype=np.float32).reshape(8, 8)

        ir_array = ZarrArray.from_zarr(zarr_array)
        # Lazy invariant: data is the zarr.Array handle, not numpy.
        self.assertIsInstance(ir_array.data, zarr.Array)
        self.assertNotIsInstance(ir_array.data, np.ndarray)
        self.assertEqual(ir_array.shape, (8, 8))
        self.assertEqual(str(ir_array.dtype), "float32")

    def test_subset_does_not_materialize_full_array(self) -> None:
        store = _CountingStore()
        root = zarr.create_group(store=store, zarr_format=3)
        zarr_array = root.create_array(name="image", shape=(16, 16), chunks=(4, 4), dtype="int32")
        zarr_array[:] = np.arange(256, dtype=np.int32).reshape(16, 16)
        store.reads = 0  # reset after the write phase

        ir_array = ZarrArray.from_zarr(zarr_array)
        # Reading shape / dtype must not fetch any chunk data.
        self.assertEqual(ir_array.shape, (16, 16))
        self.assertEqual(store.reads, 0)

        subset = ir_array.read(slices=(slice(0, 4), slice(0, 4)))
        self.assertEqual(subset.shape, (4, 4))
        np.testing.assert_array_equal(subset, np.arange(256).reshape(16, 16)[:4, :4])
        # A 4x4 subset aligned with chunks=(4, 4) intersects exactly one
        # data chunk; allow a small margin for incidental metadata reads,
        # but stay tight enough to catch a regression that fetches 2+ chunks.
        self.assertLessEqual(store.reads, 4)

    def test_staged_numpy_array_is_eager(self) -> None:
        data = np.arange(12, dtype=np.float64).reshape(3, 4)
        ir_array = ZarrArray(data=data)
        self.assertIs(ir_array.data, data)
        self.assertEqual(ir_array.shape, (3, 4))


class _CountingStore(zarr.storage.MemoryStore if HAVE_ZARR else object):
    """A MemoryStore that counts get() calls."""

    def __init__(self) -> None:
        super().__init__()
        self.reads = 0

    async def get(self, key, prototype, byte_range=None):  # type: ignore[override]
        self.reads += 1
        return await super().get(key, prototype, byte_range)


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrDocumentTestCase(unittest.TestCase):
    """Tests for `ZarrDocument` / `ZarrGroup` round-trip and tree walking."""

    def test_round_trip_through_memory_store(self) -> None:
        # Build a flat IR: image, variance, mask siblings at root.
        doc = ZarrDocument(root=ZarrGroup())
        doc.root.attributes.lsst["archive_class"] = "MaskedImage"
        doc.root.attributes.lsst["json"] = "lsst_json"

        image = ZarrArray(data=np.ones((4, 4), dtype="float32"))
        image.attributes.extra["_ARRAY_DIMENSIONS"] = ["y", "x"]
        doc.root.arrays["image"] = image

        mask = ZarrArray(data=np.zeros((4, 4), dtype="uint8"))
        mask.attributes.extra["_ARRAY_DIMENSIONS"] = ["y", "x"]
        mask.attributes.extra["flag_masks"] = [1, 2]
        mask.attributes.extra["flag_meanings"] = "BAD SAT"
        doc.root.arrays["mask"] = mask

        # Stub a 1-D uint8 'tree' array (JSON bytes).
        doc.root.arrays["lsst_json"] = ZarrArray(data=np.frombuffer(b"{}", dtype=np.uint8))

        store = zarr.storage.MemoryStore()
        doc.to_zarr(store)

        # Reload and verify lazy invariant on every array.
        recovered = ZarrDocument.from_zarr(store)
        self.assertIsInstance(recovered.root.arrays["image"].data, zarr.Array)
        self.assertIsInstance(recovered.root.arrays["mask"].data, zarr.Array)
        self.assertEqual(recovered.root.attributes.lsst["archive_class"], "MaskedImage")
        # CF flag attrs round-trip via the extra namespace.
        self.assertEqual(
            recovered.root.arrays["mask"].attributes.extra["flag_meanings"],
            "BAD SAT",
        )
        # xarray dims round-trip.
        self.assertEqual(
            recovered.root.arrays["image"].attributes.extra["_ARRAY_DIMENSIONS"],
            ["y", "x"],
        )
        # Subset reads still go through the lazy handle.
        np.testing.assert_array_equal(recovered.root.arrays["image"].read(), np.ones((4, 4), dtype="float32"))

    def test_get_walks_paths(self) -> None:
        doc = ZarrDocument(root=ZarrGroup())
        doc.root.arrays["image"] = ZarrArray(data=np.zeros((2, 2), dtype="float32"))
        red = doc.root.ensure_group("/red")
        red.arrays["image"] = ZarrArray(data=np.ones((2, 2), dtype="float32"))

        # Absolute and relative paths.
        self.assertIs(doc.root.get("/image"), doc.root.arrays["image"])
        self.assertIs(doc.root.get("image"), doc.root.arrays["image"])
        self.assertIs(doc.root.get("/red/image"), red.arrays["image"])
        self.assertIs(doc.root.get("/"), doc.root)

        with self.assertRaises(KeyError):
            doc.root.get("/missing")


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class OmeCfHelpersTestCase(unittest.TestCase):
    """Tests for the OME / CF attribute-shape helpers."""

    def test_multiscale_emits_expected_shape(self) -> None:
        m = OmeMultiscale(
            name="visitimage",
            axes=("y", "x"),
            dataset_path="image",
        )
        d = m.dump()
        self.assertEqual(d["name"], "visitimage")
        self.assertEqual(
            d["axes"],
            [
                {"name": "y", "type": "space", "unit": "pixel"},
                {"name": "x", "type": "space", "unit": "pixel"},
            ],
        )
        self.assertEqual(d["datasets"][0]["path"], "image")
        # Default coordinate transform is unit scale until a real one is set.
        self.assertEqual(
            d["datasets"][0]["coordinateTransformations"],
            [{"type": "scale", "scale": [1.0, 1.0]}],
        )

    def test_multiscale_with_affine(self) -> None:
        m = OmeMultiscale(
            name="image",
            axes=("y", "x"),
            dataset_path="image",
            coordinate_transformations=[
                {"type": "scale", "scale": [0.2, 0.2]},
                {
                    "type": "affine",
                    "affine": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                },
            ],
        )
        d = m.dump()
        self.assertEqual(len(d["datasets"][0]["coordinateTransformations"]), 2)
        self.assertEqual(d["datasets"][0]["coordinateTransformations"][0]["type"], "scale")

    def test_cf_flag_attributes(self) -> None:
        cf = CfFlagAttributes(
            planes=[
                MaskPlaneEntry(name="BAD", bit=0, description="Bad pixel."),
                MaskPlaneEntry(name="SAT", bit=1, description="Saturated."),
                MaskPlaneEntry(name="CR", bit=2, description="Cosmic ray."),
            ]
        )
        d = cf.dump()
        self.assertEqual(d["flag_masks"], [1, 2, 4])
        self.assertEqual(d["flag_meanings"], "BAD SAT CR")
        self.assertEqual(d["flag_descriptions"], ["Bad pixel.", "Saturated.", "Cosmic ray."])

    def test_image_array_attrs(self) -> None:
        attrs = build_image_array_attrs(axes=("y", "x"), units="adu", long_name="science image")
        self.assertEqual(attrs["_ARRAY_DIMENSIONS"], ["y", "x"])
        self.assertEqual(attrs["units"], "adu")
        self.assertEqual(attrs["long_name"], "science image")


if __name__ == "__main__":
    unittest.main()
