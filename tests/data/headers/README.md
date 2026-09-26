# Raw FITS headers for VisitInfo-ObservationInfo Conversion Tests

The files in this directory are used by `../../test_obs_info_from_legacy.py`.
They are text-only FITS headers extracted from raw FITS files in various test data packages:

`decam-c4d_150218_052850-N25.hdr`:
`ap_verify_ci_hits2015/raw/c4d_150218_052850_ori.fits.fz`

`hsc-HSCA90333426.hdr`:
`testdata_ci_hsc/raw/HSCA90333426.fits`

`lsstcam-MC_O_20250501_000292_R21_S11.hdr`:
`testdata_ci_lsstcam_m49/LSSTCam/raw/all/raw/20250501/MC_O_20250501_000292/MC_O_20250501_000292_R21_S11.fits`

## Recipe

The contents of these file is what the appropriate raw formatters'
`readMetadata()` returns (which includes `astro_metadata_translator.fix_header`
updates), i.e.:

```
formatter = FormatterClass(
    FileDescriptor(Location(None, path), StorageClass()), ref=ref
)
formatter._reader_path = path
md = formatter.readMetadata().toOrderedDict()
header = astropy.io.fits.Header()
for key, value in md.items():
    header[key] = (
        value if isinstance(value, (bool, int, float, str)) else str(value)
    )
header.totextfile(out_path, endcard=True, overwrite=True)
```

This recipe was emitted by an AI agent that actually did the work, and has not
been independently verified.
