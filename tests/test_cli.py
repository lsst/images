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

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest import mock

import astropy.io.fits
import click
import click.testing
import numpy as np
import pytest
from cli_schema_doubles import CliFixtureDouble
from click.testing import CliRunner

from lsst.images import Box, Image
from lsst.images import fits as images_fits
from lsst.images import json as images_json
from lsst.images.cli import main
from lsst.images.serialization import backend_for_path, read_archive
from lsst.images.tests import current_fixture_path

FIXTURE_DIR = Path(__file__).parent / "data" / "schemas"


def _copy_fixture_tree(tmp_path: Path, *, source: Path = FIXTURE_DIR, name: str = "schemas") -> Path:
    """Copy a fixture tree so a mutating CLI test cannot touch its source."""
    destination = tmp_path / name
    shutil.copytree(source, destination)
    return destination


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
        ["fixtures", "-h"],
        ["fixtures", "check", "-h"],
        ["fixtures", "refresh", "-h"],
        ["fixtures", "freeze", "-h"],
        ["fixtures", "coverage", "-h"],
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
        "fixtures",
        "fixtures-check",
        "fixtures-refresh",
        "fixtures-freeze",
        "fixtures-coverage",
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


def test_schemas_write_package_option(tmp_path: Path) -> None:
    """Verify --package freezes only schemas defined under that package."""
    runner = CliRunner()
    result = runner.invoke(
        main, ["schemas", "write", "--dir", str(tmp_path), "--package", "lsst.images.cells"]
    )
    assert result.exit_code == 0, result.output
    names = sorted(p.name for p in tmp_path.rglob("*.json"))
    assert any(n.startswith("cell_coadd-") for n in names)
    assert not any(n.startswith("image-") for n in names)


def test_schemas_check_fails_when_stale(tmp_path: Path) -> None:
    """Verify schemas check exits nonzero and names the fix when stale."""
    runner = CliRunner()
    runner.invoke(main, ["schemas", "write", "--dir", str(tmp_path)])
    (path,) = tmp_path.glob("image/image-*.json")
    path.unlink()
    result = runner.invoke(main, ["schemas", "check", "--dir", str(tmp_path)])
    assert result.exit_code != 0
    assert "schemas write" in result.output


def test_cli_describe_visit_image() -> None:
    """The describe command renders a deserialized VisitImage."""
    path = current_fixture_path(FIXTURE_DIR, "visit_image")
    result = CliRunner().invoke(main, ["describe", str(path)])
    assert result.exit_code == 0, result.output
    assert "VisitImage" in result.output


def test_cli_describe_coadd_provenance() -> None:
    """The describe command renders a deserialized CoaddProvenance.

    Provenance is describable in its own right, not only as part of a coadd,
    so the command must not fall back to the default object repr for it.
    """
    path = current_fixture_path(FIXTURE_DIR, "coadd_provenance")
    result = CliRunner().invoke(main, ["describe", str(path)])
    assert result.exit_code == 0, result.output
    assert "CoaddProvenance" in result.output
    assert "input images" in result.output


def test_fixtures_check_reports_a_clean_tree() -> None:
    """Verify 'fixtures check' exits zero on the committed tree."""
    runner = click.testing.CliRunner()
    result = runner.invoke(
        main,
        [
            "fixtures",
            "check",
            "--dir",
            str(Path(__file__).parent / "data" / "schemas"),
            "--schema-dir",
            str(Path(__file__).parent.parent / "schemas"),
            "--exempt",
            "psfex_psf",
        ],
    )
    assert result.exit_code == 0, result.output


def test_fixtures_coverage_reports_positions_for_a_composite() -> None:
    """Verify 'fixtures coverage --schema' reports a composite's positions.

    Asserted on structure rather than on any particular gap, so adding a model
    or widening a fixture cannot turn this into a failure.
    """
    result = click.testing.CliRunner().invoke(
        main, ["fixtures", "coverage", "--dir", str(FIXTURE_DIR), "--schema", "cell_coadd"]
    )
    assert result.exit_code == 0, result.output
    assert result.output.startswith("cell_coadd ")
    assert "holds  .psf [cell_psf]" in result.output


def test_fixtures_coverage_emits_parseable_json() -> None:
    """Verify 'fixtures coverage --format json' emits a JSON object."""
    result = click.testing.CliRunner().invoke(
        main,
        ["fixtures", "coverage", "--dir", str(FIXTURE_DIR), "--schema", "cell_coadd", "--format", "json"],
    )
    assert result.exit_code == 0, result.output
    parsed = json.loads(result.output)
    (entry,) = parsed.values()
    assert {"sources", "expressed", "absent", "positions"} == set(entry)
    assert any(position["path"] == ".psf" for position in entry["positions"])


def test_fixtures_coverage_reports_an_unknown_schema_cleanly() -> None:
    """Verify a schema name that matches nothing says so and exits zero."""
    result = click.testing.CliRunner().invoke(
        main, ["fixtures", "coverage", "--dir", str(FIXTURE_DIR), "--schema", "no_such_schema"]
    )
    assert result.exit_code == 0, result.output
    assert "no schema matches 'no_such_schema'" in result.output


def test_fixtures_check_requires_a_schema_directory() -> None:
    """Verify pairing cannot be accidentally omitted from the CLI check."""
    result = click.testing.CliRunner().invoke(main, ["fixtures", "check"])
    assert result.exit_code != 0
    assert "Missing option '--schema-dir'" in result.output


def test_fixtures_check_reports_a_problem(tmp_path: Path) -> None:
    """Verify 'fixtures check' reports why a malformed fixture fails.

    The empty object fails model validation, which must be reported as a
    validation failure on this exact file, not merely as some problem
    somewhere; a fixture tree with no ``image`` file at all would already
    report that file as missing, so the assertion has to name the failure
    mode to tell the two apart.
    """
    name = current_fixture_path(FIXTURE_DIR, "image").name
    directory = tmp_path / "image"
    directory.mkdir(parents=True)
    (directory / name).write_text("{}\n")
    runner = click.testing.CliRunner()
    result = runner.invoke(
        main,
        [
            "fixtures",
            "check",
            "--dir",
            str(tmp_path),
            "--schema-dir",
            str(Path(__file__).parent.parent / "schemas"),
        ],
    )
    assert result.exit_code != 0
    assert f"{name}: does not validate" in result.output


def test_fixtures_refresh_reports_no_change(tmp_path: Path) -> None:
    """Verify 'fixtures refresh' would be a no-op on the committed tree.

    The command is intentionally mutating, so run it on a fresh copy. If a
    committed development fixture is dirty, every test run must copy and
    detect that same dirty input rather than repairing the repository on the
    first run and passing on the second.
    """
    directory = _copy_fixture_tree(tmp_path)
    runner = click.testing.CliRunner()
    result = runner.invoke(main, ["fixtures", "refresh", "--dir", str(directory)])
    assert result.exit_code == 0, result.output
    assert "already up to date" in result.output


def test_fixtures_freeze_reports_nothing_to_freeze(tmp_path: Path) -> None:
    """Verify 'fixtures freeze' would be a no-op on the committed tree.

    Run the destructive command on a copy so a newly finalized schema makes
    this test fail on every run without moving or deleting the developer's
    fixture.
    """
    directory = _copy_fixture_tree(tmp_path)
    runner = click.testing.CliRunner()
    result = runner.invoke(main, ["fixtures", "freeze", "--dir", str(directory)])
    assert result.exit_code == 0, result.output
    assert "nothing to freeze" in result.output


def test_fixtures_refresh_detection_is_repeatable_without_mutating_source(tmp_path: Path) -> None:
    """Verify a dirty fixture is detected repeatedly without being repaired.

    Dirty here means valid but not canonically formatted, which is what
    refresh regenerates.  The fixture belongs to `CliFixtureDouble` because
    refresh acts only on a schema still in development.
    """
    name = CliFixtureDouble.SCHEMA_NAME
    source = tmp_path / "dirty-source"
    (source / name).mkdir(parents=True)
    fixture = source / name / f"{name}-1.0.0.dev.json"
    dirty_text = json.dumps({"schema_version": "1.0.0.dev0", "min_read_version": 1, "value": "x"})
    fixture.write_text(dirty_text)

    for run in range(2):
        directory = _copy_fixture_tree(tmp_path, source=source, name=f"run-{run}")
        result = click.testing.CliRunner().invoke(
            main,
            ["fixtures", "refresh", "--dir", str(directory), "--package", CliFixtureDouble.__module__],
        )
        assert result.exit_code == 0, result.output
        assert "already up to date" not in result.output
        assert "wrote" in result.output
        assert fixture.read_text() == dirty_text


def test_fixtures_refresh_reports_finalized_conflict_cleanly(tmp_path: Path) -> None:
    """Verify 'fixtures refresh' turns a finalized-fixture conflict into a
    clean error instead of a raw traceback.
    """
    source = current_fixture_path(FIXTURE_DIR, "image")
    name = source.name
    data = json.loads(source.read_text())
    directory = tmp_path / "image"
    directory.mkdir(parents=True)
    # Valid but not canonically formatted, so it differs from the text
    # refresh would regenerate for this already-finalized version.
    (directory / name).write_text(json.dumps(data))
    runner = click.testing.CliRunner()
    result = runner.invoke(main, ["fixtures", "refresh", "--dir", str(tmp_path)])
    assert result.exit_code != 0
    assert name in result.output
    assert "bump SCHEMA_VERSION" in result.output
    assert "Traceback" not in result.output


def test_fixtures_freeze_reports_existing_target_cleanly(tmp_path: Path) -> None:
    """Verify 'fixtures freeze' turns a pre-existing target conflict into a
    clean error instead of a raw traceback.
    """
    target = current_fixture_path(FIXTURE_DIR, "image")
    target_name = target.name
    dev_name = target_name.removesuffix(".json") + ".dev.json"
    text = target.read_text()
    directory = tmp_path / "image"
    directory.mkdir(parents=True)
    (directory / dev_name).write_text(text)
    (directory / target_name).write_text(text)
    runner = click.testing.CliRunner()
    result = runner.invoke(main, ["fixtures", "freeze", "--dir", str(tmp_path)])
    assert result.exit_code != 0
    assert target_name in result.output
    assert "already exists" in result.output
    assert "Traceback" not in result.output


def test_fixtures_freeze_reports_a_validation_failure_cleanly(tmp_path: Path) -> None:
    """Verify 'fixtures freeze' turns a fixture that no longer validates into
    a clean error instead of a raw traceback.

    freeze_schema_fixtures reads each fixture through its live model
    (read_fixture_tree) as it freezes it; a fixture that fails that read
    raises pydantic.ValidationError or ArchiveReadError, neither of which is
    SchemaFixtureError, so the CLI must catch them too rather than let a
    traceback escape.
    """
    directory = tmp_path / "image"
    directory.mkdir(parents=True)
    (directory / "image-1.0.0.dev.json").write_text("{}\n")
    runner = click.testing.CliRunner()
    result = runner.invoke(main, ["fixtures", "freeze", "--dir", str(tmp_path)])
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    # A raw pydantic.ValidationError escaping uncaught propagates out of
    # CliRunner.invoke() as result.exception with no "Error: ..." line ever
    # written to result.output; only a caught-and-reraised
    # click.ClickException produces that line and a clean SystemExit.
    assert "Error:" in result.output
    assert isinstance(result.exception, SystemExit)


def test_fixtures_refresh_reports_a_validation_failure_cleanly(tmp_path: Path) -> None:
    """Verify 'fixtures refresh' turns a fixture that no longer validates
    into a clean error instead of a raw traceback.

    refresh_schema_fixtures reads a development fixture through
    read_fixture_tree to canonicalize it; a fixture that fails that read
    raises pydantic.ValidationError or ArchiveReadError, neither of which is
    SchemaFixtureError, so the CLI must catch them too rather than let a
    traceback escape.  Refresh acts only on a schema still in development, so
    this uses `CliFixtureDouble` rather than a real schema that would be
    finalized eventually, and is finalized during every release.
    """
    name = CliFixtureDouble.SCHEMA_NAME
    directory = tmp_path / name
    directory.mkdir(parents=True)
    (directory / f"{name}-1.0.0.dev.json").write_text("{}\n")
    runner = click.testing.CliRunner()
    result = runner.invoke(
        main, ["fixtures", "refresh", "--dir", str(tmp_path), "--package", CliFixtureDouble.__module__]
    )
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    # A raw pydantic.ValidationError escaping uncaught propagates out of
    # CliRunner.invoke() as result.exception with no "Error: ..." line ever
    # written to result.output; only a caught-and-reraised
    # click.ClickException produces that line and a clean SystemExit.
    assert "Error:" in result.output
    assert isinstance(result.exception, SystemExit)
