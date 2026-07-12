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
from pathlib import Path
from unittest import mock

import astropy.io.fits
import click
import numpy as np
import pytest
from click.testing import CliRunner

from lsst.images import Box, Image
from lsst.images import fits as images_fits
from lsst.images import json as images_json
from lsst.images.cli import main
from lsst.images.serialization import backend_for_path, read_archive


@pytest.fixture(scope="session")
def external_data_dir() -> str:
    """Return the external test-data directory path, skipping if unset."""
    if (result := os.environ.get("TESTDATA_IMAGES_DIR")) is None:
        pytest.skip("TESTDATA_IMAGES_DIR is not set.")
    return result


def _make_cli_input(tmp_path: Path) -> str:
    """Return the path to a minimal FITS file written under tmp_path."""
    path = str(tmp_path / "in.fits")
    astropy.io.fits.PrimaryHDU().writeto(path)
    return path


def _make_detect_file(tmp_path: Path, dataset_type: str | None) -> str:
    """Return a path to a FITS file with LSST BUTLER DATASETTYPE set to
    dataset_type.
    """
    name = dataset_type.replace(" ", "_") if dataset_type is not None else "none"
    path = str(tmp_path / f"detect_{name}.fits")
    hdu = astropy.io.fits.PrimaryHDU()
    with images_fits.suppress_fits_card_warnings():
        if dataset_type is not None:
            hdu.header["LSST BUTLER DATASETTYPE"] = dataset_type
        hdu.writeto(path)
    return path


def test_group_help() -> None:
    """Test that the root CLI group loads and lists core subcommands."""
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0, result.output
    assert "convert" in result.output
    assert "inspect" in result.output


def test_python_m_entry_point() -> None:
    """Test that python -m lsst.images runs the same CLI group."""
    result = subprocess.run(
        [sys.executable, "-m", "lsst.images", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "convert" in result.stdout
    assert "inspect" in result.stdout


def test_inspect_fits(tmp_path: Path) -> None:
    """Test 'inspect' on a FITS file."""
    path = str(tmp_path / "x.fits")
    image = Image(np.zeros((4, 4), dtype=np.float32), bbox=Box.factory[0:4, 0:4])
    images_fits.write(image, path)
    result = CliRunner().invoke(main, ["inspect", path])
    assert result.exit_code == 0, result.output
    assert "https://images.lsst.io/schemas/image-1.0.0" in result.output
    assert "format version: 1" in result.output
    assert "python class:" in result.output
    assert "lsst.images.Image" in result.output


def test_inspect_json(tmp_path: Path) -> None:
    """Test 'inspect' on a JSON file."""
    path = str(tmp_path / "x.json")
    image = Image(np.zeros((4, 4), dtype=np.float32), bbox=Box.factory[0:4, 0:4])
    images_json.write(image, path)
    result = CliRunner().invoke(main, ["inspect", path])
    assert result.exit_code == 0, result.output
    assert "image-1.0.0" in result.output
    assert "n/a" in result.output
    assert "python class:" in result.output
    assert "lsst.images.Image" in result.output


def test_inspect_unregistered_schema(tmp_path: Path) -> None:
    """Test that 'inspect' succeeds and reports an unregistered schema name."""
    path = str(tmp_path / "fake.json")
    with open(path, "w") as f:
        f.write(
            '{"schema_url": "https://images.lsst.io/schemas/no-such-schema-99.0.0",'
            ' "schema_version": "99.0.0", "min_read_version": 1, "indirect": []}'
        )
    result = CliRunner().invoke(main, ["inspect", path])
    assert result.exit_code == 0, result.output
    assert "python class:" in result.output
    assert "<unregistered: no-such-schema>" in result.output


def test_inspect_unknown_extension(tmp_path: Path) -> None:
    """Test that 'inspect' fails with a non-zero exit code for an unsupported
    file extension.
    """
    path = str(tmp_path / "x.txt")
    with open(path, "w") as stream:
        stream.write("nope")
    result = CliRunner().invoke(main, ["inspect", path])
    assert result.exit_code != 0
    assert ".fits" in result.output


def test_reformat_round_trip_json_fits_json(tmp_path: Path) -> None:
    """Test that reformat JSON→FITS→JSON preserves the image data."""
    image = Image(np.arange(16, dtype=np.float32).reshape(4, 4), bbox=Box.factory[0:4, 0:4])
    src = str(tmp_path / "in.json")
    mid = str(tmp_path / "mid.fits")
    out = str(tmp_path / "out.json")
    images_json.write(image, src)

    result = CliRunner().invoke(main, ["reformat", src, mid])
    assert result.exit_code == 0, result.output
    assert backend_for_path(mid).input_archive.get_basic_info(mid).schema_name == "image"

    result = CliRunner().invoke(main, ["reformat", mid, out])
    assert result.exit_code == 0, result.output

    np.testing.assert_array_equal(read_archive(out, Image).array, image.array)


def test_reformat_refuses_existing_output(tmp_path: Path) -> None:
    """Test that reformat refuses to overwrite an existing output file without
    --overwrite.
    """
    image = Image(np.arange(16, dtype=np.float32).reshape(4, 4), bbox=Box.factory[0:4, 0:4])
    src = str(tmp_path / "in.json")
    out = str(tmp_path / "out.fits")
    images_json.write(image, src)
    images_fits.write(image, out)
    result = CliRunner().invoke(main, ["reformat", src, out])
    assert result.exit_code != 0
    assert "--overwrite" in result.output


def test_reformat_unknown_output_extension(tmp_path: Path) -> None:
    """Test that reformat fails for an unsupported output file extension."""
    image = Image(np.arange(16, dtype=np.float32).reshape(4, 4), bbox=Box.factory[0:4, 0:4])
    src = str(tmp_path / "in.json")
    images_json.write(image, src)
    result = CliRunner().invoke(main, ["reformat", src, str(tmp_path / "out.txt")])
    assert result.exit_code != 0
    assert ".fits" in result.output


def test_detect_visit_image(tmp_path: Path) -> None:
    """Test that detect_legacy_type identifies visit_image and
    preliminary_visit_image.
    """
    from lsst.images.cli._convert import detect_legacy_type

    assert detect_legacy_type(_make_detect_file(tmp_path, "visit_image")) == "visit_image"
    assert detect_legacy_type(_make_detect_file(tmp_path, "preliminary_visit_image")) == "visit_image"


def test_detect_cell_coadd(tmp_path: Path) -> None:
    """Test that detect_legacy_type identifies deep_coadd_cell_predetection as
    cell_coadd.
    """
    from lsst.images.cli._convert import detect_legacy_type

    assert detect_legacy_type(_make_detect_file(tmp_path, "deep_coadd_cell_predetection")) == "cell_coadd"


def test_detect_indeterminate(tmp_path: Path) -> None:
    """Test that detect_legacy_type returns None for unknown or absent dataset-
    type headers.
    """
    from lsst.images.cli._convert import detect_legacy_type

    assert detect_legacy_type(_make_detect_file(tmp_path, None)) is None
    assert detect_legacy_type(_make_detect_file(tmp_path, "camera")) is None


def test_detect_visit_image_fixture(tmp_path: Path, external_data_dir: str) -> None:
    """Test that detect_legacy_type detects a real legacy visit-image fixture
    file.
    """
    from lsst.images.cli._convert import detect_legacy_type

    path = os.path.join(external_data_dir, "dp2", "legacy", "visit_image.fits")
    assert detect_legacy_type(path) == "visit_image"


def test_convert_visit_image_to_json(tmp_path: Path, external_data_dir: str) -> None:
    """Test that convert produces a valid visit_image JSON file from a legacy
    FITS fixture.
    """
    pytest.importorskip("lsst.afw.image")
    src = os.path.join(external_data_dir, "dp2", "legacy", "visit_image.fits")
    out = str(tmp_path / "converted.json")
    result = CliRunner().invoke(main, ["convert", src, out])
    assert result.exit_code == 0, result.output
    info = backend_for_path(out).input_archive.get_basic_info(out)
    assert info.schema_name == "visit_image"


def test_convert_refuses_existing_output(tmp_path: Path, external_data_dir: str) -> None:
    """Test that convert refuses to overwrite an existing output file without
    --overwrite.
    """
    pytest.importorskip("lsst.afw.image")
    src = os.path.join(external_data_dir, "dp2", "legacy", "visit_image.fits")
    out = str(tmp_path / "exists.json")
    with open(out, "w") as stream:
        stream.write("{}")
    result = CliRunner().invoke(main, ["convert", src, out])
    assert result.exit_code != 0
    assert "--overwrite" in result.output


def test_convert_cell_coadd_to_json(tmp_path: Path, external_data_dir: str) -> None:
    """Test that convert produces a valid cell_coadd JSON file from a legacy
    FITS MultipleCellCoadd.
    """
    pytest.importorskip("lsst.cell_coadds")
    legacy_dir = os.path.join(external_data_dir, "dp2", "legacy")
    src = os.path.join(legacy_dir, "deep_coadd_cell_predetection.fits")
    skymap = os.path.join(legacy_dir, "skyMap.pickle")
    out = str(tmp_path / "coadd.json")
    result = CliRunner().invoke(main, ["convert", src, out, "--type", "cell_coadd", "--skymap", skymap])
    assert result.exit_code == 0, result.output
    info = backend_for_path(out).input_archive.get_basic_info(out)
    assert info.schema_name == "cell_coadd"


def test_convert_cell_coadd_requires_skymap(tmp_path: Path, external_data_dir: str) -> None:
    """Test that convert fails with a helpful message when --skymap is missing
    for cell_coadd.
    """
    pytest.importorskip("lsst.cell_coadds")
    src = os.path.join(external_data_dir, "dp2", "legacy", "deep_coadd_cell_predetection.fits")
    out = str(tmp_path / "coadd.json")
    result = CliRunner().invoke(main, ["convert", src, out, "--type", "cell_coadd"])
    assert result.exit_code != 0
    assert "--skymap" in result.output


def test_preserve_quantization_default_is_true() -> None:
    """Test that the --preserve-quantization option defaults to True."""
    from lsst.images.cli._convert import convert

    option = next(p for p in convert.params if p.name == "preserve_quantization")
    assert option.default is True


def test_preserve_quantization_explicit_flag_rejected_for_cell_coadd(tmp_path: Path) -> None:
    """Test that explicitly passing --preserve-quantization is rejected for
    cell_coadd conversions.
    """
    src = _make_cli_input(tmp_path)
    out = str(tmp_path / "out.json")
    result = CliRunner().invoke(
        main, ["convert", src, out, "--type", "cell_coadd", "--preserve-quantization"]
    )
    assert result.exit_code != 0
    assert "preserve-quantization" in result.output


def test_preserve_quantization_default_does_not_reject_cell_coadd(tmp_path: Path) -> None:
    """Test that the --preserve-quantization option default doesn't get in the
    way of cell-coadd conversion.
    """
    src = _make_cli_input(tmp_path)
    out = str(tmp_path / "out.json")
    result = CliRunner().invoke(main, ["convert", src, out, "--type", "cell_coadd"])
    assert "preserve-quantization" not in result.output


def test_preserve_quantization_forwarded_to_read_legacy() -> None:
    """Test that _read_legacy forwards preserve_quantization=True to
    VisitImage.read_legacy.
    """
    from lsst.images.cli._convert import _read_legacy

    with mock.patch("lsst.images.VisitImage.read_legacy") as read_legacy:
        _read_legacy("in.fits", "visit_image", None, None, None, True)
    read_legacy.assert_called_once_with("in.fits", preserve_quantization=True)


def test_rejects_identical_paths(tmp_path: Path) -> None:
    """Test that 'convert' rejects identical src and dst paths even with
    --overwrite.
    """
    path = _make_cli_input(tmp_path)
    result = CliRunner().invoke(main, ["convert", path, path, "--type", "visit_image", "--overwrite"])
    assert result.exit_code != 0
    assert "different" in result.output
    assert os.path.exists(path)


def test_preserves_existing_output_on_read_failure(tmp_path: Path) -> None:
    """Test that 'convert' leaves the existing output file intact when
    read_legacy raises.
    """
    src = _make_cli_input(tmp_path)
    out = str(tmp_path / "out.json")
    with open(out, "w") as stream:
        stream.write("ORIGINAL")
    with mock.patch(
        "lsst.images.cli._convert._read_legacy",
        side_effect=click.ClickException("boom"),
    ):
        result = CliRunner().invoke(main, ["convert", src, out, "--type", "visit_image", "--overwrite"])
    assert result.exit_code != 0
    with open(out) as stream:
        assert stream.read() == "ORIGINAL"


def test_subcommands_present() -> None:
    """Test that the minify, reformat, extract-test-data, verify-rewrite, and
    fuzz-masked-image are listed by --help.
    """
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0, result.output
    assert "minify" in result.output
    assert "reformat" in result.output
    assert "extract-test-data" in result.output
    assert "verify-rewrite" in result.output
    assert "fuzz-masked-image" in result.output


def test_minify_help() -> None:
    """Verify minify --help exits cleanly."""
    result = CliRunner().invoke(main, ["minify", "--help"])
    assert result.exit_code == 0, result.output


def test_extract_test_data_help() -> None:
    """Verify extract-test-data --help exits cleanly."""
    result = CliRunner().invoke(main, ["extract-test-data", "--help"])
    assert result.exit_code == 0, result.output


def test_verify_rewrite_help() -> None:
    """Verify verify-rewrite and its stage4 subcommand load with core deps
    only.
    """
    result = CliRunner().invoke(main, ["verify-rewrite", "--help"])
    assert result.exit_code == 0, result.output
    assert "stage4" in result.output
    result = CliRunner().invoke(main, ["verify-rewrite", "stage4", "--help"])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    "args",
    [
        ["-h"],
        ["convert", "-h"],
        ["inspect", "-h"],
        ["minify", "-h"],
        ["reformat", "-h"],
        ["extract-test-data", "-h"],
        ["extract-test-data", "dp2", "-h"],
        ["verify-rewrite", "-h"],
        ["verify-rewrite", "stage4", "-h"],
        ["fuzz-masked-image", "-h"],
        ["schemas", "-h"],
        ["schemas", "write", "-h"],
        ["schemas", "check", "-h"],
    ],
    ids=[
        "root",
        "convert",
        "inspect",
        "minify",
        "reformat",
        "extract-test-data",
        "extract-test-data-dp2",
        "verify-rewrite",
        "verify-rewrite-stage4",
        "fuzz-masked-image",
        "schemas",
        "schemas-write",
        "schemas-check",
    ],
)
def test_short_help_alias(args: list[str]) -> None:
    """Test that -h is an alias for --help on the group and every
    subcommand.
    """
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output


def test_schemas_write_and_check(tmp_path: Path) -> None:
    """Verify schemas write populates a directory that schemas check
    accepts.
    """
    runner = CliRunner()
    result = runner.invoke(main, ["schemas", "write", "--dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert list(tmp_path.glob("image/image-*.json"))
    result = runner.invoke(main, ["schemas", "check", "--dir", str(tmp_path)])
    assert result.exit_code == 0, result.output


def test_schemas_check_fails_when_stale(tmp_path: Path) -> None:
    """Verify schemas check exits nonzero and names the fix when stale."""
    runner = CliRunner()
    runner.invoke(main, ["schemas", "write", "--dir", str(tmp_path)])
    (path,) = tmp_path.glob("image/image-*.json")
    path.unlink()
    result = runner.invoke(main, ["schemas", "check", "--dir", str(tmp_path)])
    assert result.exit_code != 0
    assert "schemas write" in result.output
