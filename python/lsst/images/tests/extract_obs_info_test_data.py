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

"""Extract the ObservationInfo conversion test fixtures from various CI
test data packaes.

Run from a development checkout with::

    lsst-images-admin extract-obs-info-test-data

For this script to run, the following packages must be set up:

 - ``obs_lsst``
 - ``obs_decam``
 - ``obs_subaru``
 - ``testdata_ci_hsc``
 - ``testdata_ci_lsstcam_m49``
 - ``ap_verify_ci_hits2015``
"""

from __future__ import annotations

__all__ = ()

import dataclasses
from pathlib import Path
from typing import TYPE_CHECKING

import click
from astro_metadata_translator import ObservationInfo

if TYPE_CHECKING:
    import astropy.io.fits

_REPO_ROOT = Path(__file__).parents[4]
_HEADERS_DIR = _REPO_ROOT / "tests" / "data" / "headers"
_OBS_INFO_DIR = _REPO_ROOT / "tests" / "data" / "obs_info"
_DETECTORS_DIR = _REPO_ROOT / "tests" / "data" / "detectors"


@dataclasses.dataclass(frozen=True)
class FixtureCase:
    """The provenance and generation inputs for one fixture case."""

    package: str
    """EUPS test data package containing the raw file."""

    raw_path: str
    """Raw FITS file path within the test data package."""

    original_filename: str
    """Original observation filename passed to
    ``ObservationInfo.from_header``.
    """
    instrument_class: str
    """Full type name of the Instrument class used to load camera geometry."""

    extensions: tuple[int, ...]
    """Non-primary HDU indexes whose headers are merged (INHERIT-aware) into
    the primary header..
    """

    @property
    def full_raw_path(self) -> Path:
        from lsst.utils import getPackageDir

        return Path(getPackageDir(self.package)) / self.raw_path


_CASES = {
    "decam-c4d_150218_052850-N25": FixtureCase(
        package="ap_verify_ci_hits2015",
        raw_path="raw/c4d_150218_052850_ori.fits.fz",
        original_filename="c4d_150218_052850_ori.fits.fz",
        instrument_class="lsst.obs.decam.DarkEnergyCamera",
        extensions=(1,),  # the N25 CCD.
    ),
    "hsc-HSCA90333426": FixtureCase(
        package="testdata_ci_hsc",
        raw_path="raw/HSCA90333426.fits",
        original_filename="HSCA90333426.fits",
        instrument_class="lsst.obs.subaru.HyperSuprimeCam",
        extensions=(1,),  # the file's only (detector) extension.
    ),
    "lsstcam-MC_O_20250501_000292_R21_S11": FixtureCase(
        package="testdata_ci_lsstcam_m49",
        raw_path="LSSTCam/raw/all/raw/20250501/MC_O_20250501_000292/MC_O_20250501_000292_R21_S11.fits",
        original_filename="MC_O_20250501_000292_R21_S11.fits",
        instrument_class="lsst.obs.lsst.LsstCam",
        extensions=(17,),  # the REB_COND table; segment images are shared
        # across the group's detectors and the detector identity lives in the
        # primary header (via the filename).
    ),
}


def extract_header(raw_path: Path, extensions: tuple[int, ...], output_path: Path) -> astropy.io.fits.Header:
    """Write the pipeline-processed FITS header of one raw file.

    The primary header and the named extension headers are read with
    ``lsst.afw.fits.readMetadata``, merged INHERIT-aware as
    ``astro_metadata_translator.merge_headers`` does during raw ingest, and
    then corrected with ``astro_metadata_translator.fix_header`` for this
    observation's known per-header fixes.

    Parameters
    ----------
    raw_path : `pathlib.Path`
        The raw FITS file to read.
    extensions : `tuple` [ `int` ]
        Non-primary HDU indexes whose headers are merged into the primary
        header.
    output_path : `pathlib.Path`
        The ``.hdr`` text file to write.

    Returns
    -------
    header : `astropy.io.fits.Header`
        The processed header.
    """
    import astropy.io.fits
    from astro_metadata_translator import MetadataTranslator, fix_header, merge_headers

    from lsst.afw.fits import readMetadata

    def to_astropy(ext: int) -> astropy.io.fits.Header:
        header = astropy.io.fits.Header()
        for key, value in readMetadata(str(raw_path), ext).toOrderedDict().items():
            if value is None or isinstance(value, (bool, int, float, str)):
                header[key] = value
            else:
                header[key] = str(value)
        return header

    headers = [to_astropy(0)]
    if extensions:
        headers.extend(to_astropy(ext) for ext in extensions)
    else:
        # Read the first extension too, as RawIngestTask.extractMetadata
        # does when the primary header alone can not identify a translator.
        try:
            MetadataTranslator.determine_translator(headers[0], filename=str(raw_path))
        except ValueError:
            headers.append(to_astropy(1))
    merged = merge_headers(headers, mode="overwrite")
    # merge_headers preserves the type of the headers it is given.
    assert isinstance(merged, astropy.io.fits.Header)
    header = merged
    translator_class = MetadataTranslator.determine_translator(header, filename=str(raw_path))
    fix_header(header, translator_class=translator_class, filename=str(raw_path))
    # fix_header records when the corrections were applied; that is fixture
    # build bookkeeping, not observation metadata, and would make
    # regeneration non-deterministic.
    fix_date = "HIERARCH ASTRO METADATA FIX DATE"
    if fix_date in header:
        del header[fix_date]
    header.totextfile(str(output_path), endcard=True, overwrite=True)
    # Strip card padding and end with a newline, exactly as the repo's
    # trailing-whitespace and EOF-fixer pre-commit hooks would, so
    # regeneration does not show phantom churn against the committed file.
    text = output_path.read_text()
    output_path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n")
    return header


def extract_obs_info(
    header: astropy.io.fits.Header,
    original_filename: str,
    output_path: Path,
) -> ObservationInfo:
    """Write the ``ObservationInfo`` JSON wire form of one header.

    Parameters
    ----------
    header : `astropy.io.fits.Header`
        The processed header to translate.
    original_filename : `str`
        The original observation filename, passed to
        ``ObservationInfo.from_header`` so correction YAMLs apply.
    output_path : `pathlib.Path`
        The JSON file to write the serialized `ObservationInfo` to.

    Returns
    -------
    obs_info : `~astro_metadata_translator.ObservationInfo`
        The translated observation metadata.
    """
    obs_info = ObservationInfo.from_header(header, filename=original_filename, quiet=True)
    output_path.write_text(obs_info.model_dump_json(indent=2) + "\n")
    return obs_info


def extract_detector(
    instrument_class_name: str,
    instrument: str,
    detector_id: int,
    output_path: Path,
) -> None:
    """Write the serialized detector archive of one fixture case.

    The detector is converted from the real instrument camera's detector to
    an `lsst.images.cameras.Detector` archive: a small, diffable JSON file,
    unlike afw camera FITS files, which drag along the camera's entire
    shared transform graph.

    Parameters
    ----------
    instrument_class_name : `str`
        Fully-qualified instrument class name, e.g. ``lsst.obs.lsst.LsstCam``.
    instrument : `str`
        Instrument name to record on the detector.
    detector_id : `int`
        Identifier of the detector to take from the instrument camera.
    output_path : `pathlib.Path`
        The JSON archive file to write.
    """
    from lsst.utils.introspection import get_instance_of

    from ..cameras import Detector
    from ..serialization import write_archive

    instrument_obj = get_instance_of(instrument_class_name)
    legacy_detector = instrument_obj.getCamera()[detector_id]
    detector = Detector.from_legacy(legacy_detector, instrument=instrument)
    write_archive(detector, output_path)
    # End with a newline, as the repo's EOF-fixer pre-commit hook requires,
    # so regeneration does not show phantom churn against the committed file.
    with open(output_path, "a") as fh:
        fh.write("\n")


@click.command("extract_obs_info_test_data")
@click.option(
    "--testdata-dir",
    type=Path,
    default=None,
    help="Directory containing unpacked test-data packages, used when they"
    " are not set up in the current EUPS environment.",
)
@click.option(
    "-H",
    "--headers-dir",
    type=Path,
    default=_HEADERS_DIR,
    show_default=True,
    help="Directory to write the .hdr header fixtures to.",
)
@click.option(
    "-o",
    "--output-dir",
    type=Path,
    default=_OBS_INFO_DIR,
    show_default=True,
    help="Directory to write the ObservationInfo JSON fixtures to.",
)
@click.option(
    "-d",
    "--detectors-dir",
    type=Path,
    default=_DETECTORS_DIR,
    show_default=True,
    help="Directory to write the Detector JSON archives to.",
)
def extract_obs_info_test_data(
    testdata_dir: Path | None,
    headers_dir: Path,
    output_dir: Path,
    detectors_dir: Path,
) -> None:  # numpydoc ignore=PR01
    """Regenerate legacy-conversion fixtures from raw test data."""
    try:
        # The LSSTCam fixture's metadata translator is an obs_lsst plugin,
        # which astro_metadata_translator does not load on its own; importing
        # the package registers it.
        import lsst.obs.decam
        import lsst.obs.lsst
        import lsst.obs.subaru  # noqa: F401
    except ImportError as err:
        err.add_note(
            "Regenerating these fixtures requires a full Rubin development"
            " environment with at least 'obs_decam', 'obs_lsst' and"
            " 'obs_subaru' importable, and the fixture cases' test-data"
            " packages available. This is not necessary for just running the"
            " tests."
        )
        raise
    headers_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    detectors_dir.mkdir(parents=True, exist_ok=True)
    for stem, case in _CASES.items():
        header_path = headers_dir / (stem + ".hdr")
        header = extract_header(case.full_raw_path, case.extensions, header_path)
        click.echo(f"Wrote {header_path}")
        obs_info_path = output_dir / (stem + ".json")
        obs_info = extract_obs_info(header, case.original_filename, obs_info_path)
        click.echo(f"Wrote {obs_info_path}")
        detector_path = detectors_dir / (stem + ".json")
        assert obs_info.detector_num is not None
        extract_detector(
            case.instrument_class, obs_info.instrument or "", obs_info.detector_num, detector_path
        )
        click.echo(f"Wrote {detector_path}")
