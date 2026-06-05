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

import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import astropy.io.fits
import click
import numpy as np
from click.testing import CliRunner

from lsst.images import Box, Image
from lsst.images import fits as images_fits
from lsst.images import json as images_json
from lsst.images.cli import main
from lsst.images.serialization import backend_for_path, read


class CliSkeletonTestCase(unittest.TestCase):
    """The root group loads and shows help with core deps only."""

    def test_group_help(self) -> None:
        result = CliRunner().invoke(main, ["--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("convert", result.output)
        self.assertIn("inspect", result.output)

    def test_python_m_entry_point(self) -> None:
        # ``python -m lsst.images`` must run the same CLI group.
        result = subprocess.run(
            [sys.executable, "-m", "lsst.images", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("convert", result.stdout)
        self.assertIn("inspect", result.stdout)


class InspectTestCase(unittest.TestCase):
    """inspect prints schema URL and format version."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.image = Image(np.zeros((4, 4), dtype=np.float32), bbox=Box.factory[0:4, 0:4])

    def test_inspect_fits(self) -> None:
        path = os.path.join(self.tmp, "x.fits")
        images_fits.write(self.image, path)
        result = CliRunner().invoke(main, ["inspect", path])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("https://images.lsst.io/schemas/image-1.0.0", result.output)
        self.assertIn("format version: 1", result.output)

    def test_inspect_json(self) -> None:
        path = os.path.join(self.tmp, "x.json")
        images_json.write(self.image, path)
        result = CliRunner().invoke(main, ["inspect", path])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("image-1.0.0", result.output)
        self.assertIn("n/a", result.output)  # no container format version for JSON

    def test_inspect_fits_python_class(self) -> None:
        path = os.path.join(self.tmp, "y.fits")
        images_fits.write(self.image, path)
        result = CliRunner().invoke(main, ["inspect", path])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("python class:", result.output)
        self.assertIn("lsst.images.Image", result.output)

    def test_inspect_json_python_class(self) -> None:
        path = os.path.join(self.tmp, "y.json")
        images_json.write(self.image, path)
        result = CliRunner().invoke(main, ["inspect", path])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("python class:", result.output)
        self.assertIn("lsst.images.Image", result.output)

    def test_inspect_unregistered_schema(self) -> None:
        path = os.path.join(self.tmp, "fake.json")
        with open(path, "w") as f:
            f.write(
                '{"schema_url": "https://images.lsst.io/schemas/no-such-schema-99.0.0",'
                ' "schema_version": "99.0.0", "min_read_version": 1, "indirect": []}'
            )
        result = CliRunner().invoke(main, ["inspect", path])
        # inspect should still succeed: it reports basic info regardless
        # of registry membership.
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("python class:", result.output)
        self.assertIn("<unregistered: no-such-schema>", result.output)

    def test_inspect_unknown_extension(self) -> None:
        path = os.path.join(self.tmp, "x.txt")
        with open(path, "w") as stream:
            stream.write("nope")
        result = CliRunner().invoke(main, ["inspect", path])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn(".fits", result.output)


class ReformatTestCase(unittest.TestCase):
    """reformat reads a file and writes it back in another container format."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.image = Image(np.arange(16, dtype=np.float32).reshape(4, 4), bbox=Box.factory[0:4, 0:4])

    def test_reformat_round_trip_json_fits_json(self) -> None:
        # JSON -> FITS -> JSON must preserve the object (NDF is optional, so
        # the round trip stays within the always-available backends).
        src = os.path.join(self.tmp, "in.json")
        mid = os.path.join(self.tmp, "mid.fits")
        out = os.path.join(self.tmp, "out.json")
        images_json.write(self.image, src)

        result = CliRunner().invoke(main, ["reformat", src, mid])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(backend_for_path(mid).input_archive.get_basic_info(mid).schema_name, "image")

        result = CliRunner().invoke(main, ["reformat", mid, out])
        self.assertEqual(result.exit_code, 0, result.output)

        np.testing.assert_array_equal(read(out, Image).array, self.image.array)

    def test_reformat_refuses_existing_output(self) -> None:
        src = os.path.join(self.tmp, "in.json")
        out = os.path.join(self.tmp, "out.fits")
        images_json.write(self.image, src)
        images_fits.write(self.image, out)
        result = CliRunner().invoke(main, ["reformat", src, out])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--overwrite", result.output)

    def test_reformat_unknown_output_extension(self) -> None:
        src = os.path.join(self.tmp, "in.json")
        images_json.write(self.image, src)
        result = CliRunner().invoke(main, ["reformat", src, os.path.join(self.tmp, "out.txt")])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn(".fits", result.output)


EXTERNAL_DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)


class ConvertDetectTestCase(unittest.TestCase):
    """Legacy type detection from the LSST BUTLER DATASETTYPE header.

    These build small FITS files carrying only the discriminating header so
    the test does not depend on a fixture happening to include it.
    """

    def _make(self, dataset_type: str | None) -> str:
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "x.fits")
        hdu = astropy.io.fits.PrimaryHDU()
        if dataset_type is not None:
            hdu.header["LSST BUTLER DATASETTYPE"] = dataset_type
        hdu.writeto(path)
        return path

    def test_detect_visit_image(self) -> None:
        from lsst.images.cli._convert import detect_legacy_type

        self.assertEqual(detect_legacy_type(self._make("visit_image")), "visit_image")
        self.assertEqual(detect_legacy_type(self._make("preliminary_visit_image")), "visit_image")

    def test_detect_cell_coadd(self) -> None:
        from lsst.images.cli._convert import detect_legacy_type

        self.assertEqual(detect_legacy_type(self._make("deep_coadd_cell_predetection")), "cell_coadd")

    def test_detect_indeterminate(self) -> None:
        from lsst.images.cli._convert import detect_legacy_type

        self.assertIsNone(detect_legacy_type(self._make(None)))
        self.assertIsNone(detect_legacy_type(self._make("camera")))

    @unittest.skipUnless(EXTERNAL_DATA_DIR is not None, "TESTDATA_IMAGES_DIR is not set.")
    def test_detect_visit_image_fixture(self) -> None:
        from lsst.images.cli._convert import detect_legacy_type

        path = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "visit_image.fits")
        self.assertEqual(detect_legacy_type(path), "visit_image")


class ConvertVisitImageTestCase(unittest.TestCase):
    """convert of a legacy visit image (needs afw + testdata)."""

    @unittest.skipUnless(EXTERNAL_DATA_DIR is not None, "TESTDATA_IMAGES_DIR is not set.")
    def test_convert_visit_image_to_json(self) -> None:
        try:
            import lsst.afw.image  # noqa: F401
        except ImportError:
            self.skipTest("afw not available.")
        tmp = tempfile.mkdtemp()
        src = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "visit_image.fits")
        out = os.path.join(tmp, "converted.json")
        result = CliRunner().invoke(main, ["convert", src, out])
        self.assertEqual(result.exit_code, 0, result.output)
        info = backend_for_path(out).input_archive.get_basic_info(out)
        self.assertEqual(info.schema_name, "visit_image")

    @unittest.skipUnless(EXTERNAL_DATA_DIR is not None, "TESTDATA_IMAGES_DIR is not set.")
    def test_convert_refuses_existing_output(self) -> None:
        try:
            import lsst.afw.image  # noqa: F401
        except ImportError:
            self.skipTest("afw not available.")
        tmp = tempfile.mkdtemp()
        src = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "visit_image.fits")
        out = os.path.join(tmp, "exists.json")
        with open(out, "w") as stream:
            stream.write("{}")
        result = CliRunner().invoke(main, ["convert", src, out])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--overwrite", result.output)


class ConvertCellCoaddTestCase(unittest.TestCase):
    """convert of a legacy cell coadd (needs cell_coadds + testdata)."""

    @unittest.skipUnless(EXTERNAL_DATA_DIR is not None, "TESTDATA_IMAGES_DIR is not set.")
    def test_convert_cell_coadd_to_json(self) -> None:
        try:
            import lsst.cell_coadds  # noqa: F401
        except ImportError:
            self.skipTest("cell_coadds not available.")
        tmp = tempfile.mkdtemp()
        legacy_dir = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy")
        src = os.path.join(legacy_dir, "deep_coadd_cell_predetection.fits")
        skymap = os.path.join(legacy_dir, "skyMap.pickle")
        out = os.path.join(tmp, "coadd.json")
        # This fixture has no LSST BUTLER DATASETTYPE header, so pass --type.
        result = CliRunner().invoke(main, ["convert", src, out, "--type", "cell_coadd", "--skymap", skymap])
        self.assertEqual(result.exit_code, 0, result.output)
        info = backend_for_path(out).input_archive.get_basic_info(out)
        self.assertEqual(info.schema_name, "cell_coadd")

    @unittest.skipUnless(EXTERNAL_DATA_DIR is not None, "TESTDATA_IMAGES_DIR is not set.")
    def test_convert_cell_coadd_requires_skymap(self) -> None:
        try:
            import lsst.cell_coadds  # noqa: F401
        except ImportError:
            self.skipTest("cell_coadds not available.")
        tmp = tempfile.mkdtemp()
        src = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "deep_coadd_cell_predetection.fits")
        out = os.path.join(tmp, "coadd.json")
        result = CliRunner().invoke(main, ["convert", src, out, "--type", "cell_coadd"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--skymap", result.output)


class ConvertPreserveQuantizationTestCase(unittest.TestCase):
    """--preserve-quantization defaults on, forwards for visit images, and
    is rejected only when set explicitly for other types.
    """

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def _make_input(self) -> str:
        path = os.path.join(self.tmp, "in.fits")
        astropy.io.fits.PrimaryHDU().writeto(path)
        return path

    def test_default_is_true(self) -> None:
        from lsst.images.cli._convert import convert

        option = next(p for p in convert.params if p.name == "preserve_quantization")
        self.assertIs(option.default, True)

    def test_explicit_flag_rejected_for_cell_coadd(self) -> None:
        src = self._make_input()
        out = os.path.join(self.tmp, "out.json")
        result = CliRunner().invoke(
            main, ["convert", src, out, "--type", "cell_coadd", "--preserve-quantization"]
        )
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("preserve-quantization", result.output)

    def test_default_does_not_reject_cell_coadd(self) -> None:
        # The on-by-default flag must not block cell-coadd conversion; any
        # failure here is about later steps (skymap / missing deps), never the
        # preserve-quantization gate.
        src = self._make_input()
        out = os.path.join(self.tmp, "out.json")
        result = CliRunner().invoke(main, ["convert", src, out, "--type", "cell_coadd"])
        self.assertNotIn("preserve-quantization", result.output)

    def test_forwarded_to_read_legacy(self) -> None:
        from lsst.images.cli._convert import _read_legacy

        with mock.patch("lsst.images.VisitImage.read_legacy") as read_legacy:
            _read_legacy("in.fits", "visit_image", None, None, None, True)
        read_legacy.assert_called_once_with("in.fits", preserve_quantization=True)


class CliRegistrationTestCase(unittest.TestCase):
    """minify, extract-test-data, and verify-rewrite register and load."""

    def test_subcommands_present(self) -> None:
        result = CliRunner().invoke(main, ["--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("minify", result.output)
        self.assertIn("reformat", result.output)
        self.assertIn("extract-test-data", result.output)
        self.assertIn("verify-rewrite", result.output)

    def test_minify_help(self) -> None:
        result = CliRunner().invoke(main, ["minify", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)

    def test_extract_test_data_help(self) -> None:
        result = CliRunner().invoke(main, ["extract-test-data", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)

    def test_verify_rewrite_help(self) -> None:
        # The group and its stage4 command load with core deps only; the
        # full Rubin environment is only required when stage4 actually runs.
        result = CliRunner().invoke(main, ["verify-rewrite", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("stage4", result.output)
        result = CliRunner().invoke(main, ["verify-rewrite", "stage4", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)

    def test_short_help_alias(self) -> None:
        # -h is an alias for --help on the group and every subcommand,
        # including the nested extract-test-data sub-group.
        for args in (
            ["-h"],
            ["convert", "-h"],
            ["inspect", "-h"],
            ["minify", "-h"],
            ["reformat", "-h"],
            ["extract-test-data", "-h"],
            ["extract-test-data", "dp2", "-h"],
            ["verify-rewrite", "-h"],
            ["verify-rewrite", "stage4", "-h"],
        ):
            with self.subTest(args=args):
                result = CliRunner().invoke(main, args)
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertIn("Usage:", result.output)


class ConvertSafetyTestCase(unittest.TestCase):
    """convert must not destroy data on identical paths or on failure."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def _make_input(self) -> str:
        path = os.path.join(self.tmp, "in.fits")
        astropy.io.fits.PrimaryHDU().writeto(path)
        return path

    def test_rejects_identical_paths(self) -> None:
        path = self._make_input()
        result = CliRunner().invoke(main, ["convert", path, path, "--type", "visit_image", "--overwrite"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("different", result.output)
        # The file is left untouched.
        self.assertTrue(os.path.exists(path))

    def test_preserves_existing_output_on_read_failure(self) -> None:
        src = self._make_input()
        out = os.path.join(self.tmp, "out.json")
        with open(out, "w") as stream:
            stream.write("ORIGINAL")
        # A read failure after the overwrite gate must leave OUTPUT intact.
        with mock.patch(
            "lsst.images.cli._convert._read_legacy",
            side_effect=click.ClickException("boom"),
        ):
            result = CliRunner().invoke(main, ["convert", src, out, "--type", "visit_image", "--overwrite"])
        self.assertNotEqual(result.exit_code, 0)
        with open(out) as stream:
            self.assertEqual(stream.read(), "ORIGINAL")
