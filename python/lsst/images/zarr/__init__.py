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

"""Zarr v3 archive backend for `lsst.images`.

This module reads and writes Zarr v3 archives whose root layout is
xarray/CF-shaped (``image``, ``variance``, ``mask`` as siblings sharing
``(y, x)`` dimensions, CF ``flag_masks`` / ``flag_meanings`` /
``flag_descriptions`` on the mask) with OME-NGFF v0.5 multiscales
metadata as a discoverability layer pointing at the same ``image``
array. The same bytes are visible to ``xarray``, GDAL's Zarr driver,
and OME-Zarr tooling like ``napari`` and ``ome-zarr-py``.

Supported types
---------------

Every image type that already serializes to FITS / JSON / NDF:
`~lsst.images.Image`, `~lsst.images.Mask`, `~lsst.images.MaskedImage`,
`~lsst.images.VisitImage`, `~lsst.images.ColorImage`, plus any object
reachable through the `~lsst.images.serialization.OutputArchive`
interface.

On-disk layout
--------------

A `~lsst.images.MaskedImage` archive contains:

- ``image``, ``variance``, ``mask`` arrays at the root, shaped
  ``(Y, X)`` with shared chunk sizes.
- ``lsst_json`` — 1-D ``uint8`` zarr array containing UTF-8 JSON of
  the Pydantic archive tree (the round-trip authority). The same name
  convention is used by the FITS backend's ``JSON`` HDU and the NDF
  backend's ``/MORE/LSST/JSON`` path. WCS information (including
  full SIP / PolyMap distortion coefficients) lives inside this JSON
  as part of the projection sub-tree.

The mask is a 2-D unsigned integer (``uint8`` for ≤8 planes, up to
``uint64`` for 64 planes; >64 raises). Each pixel's bits encode the
applicable mask planes.

For `~lsst.images.ColorImage`, the three channels are written as flat 2-D arrays
at ``red``, ``green``, ``blue``.

For ``CellCoadd``, ``image`` / ``variance`` / ``mask`` are siblings
(cell-aligned chunks driven by the ``tile_shape`` hint each array is
serialized with), and ``psf`` is a 4-D ``(Cy, Cx, Py, Px)`` array with
single-cell chunks ``(1, 1, Py, Px)``.

WCS handling
------------

The full WCS (frames, mappings, polynomial distortions) round-trips
through the JSON tree at ``lsst_json``. The layout layer also emits
an OME-NGFF v0.5 affine ``coordinateTransformations`` block on the
root group as a discoverability aid for OME tooling. Before emitting,
residuals are sampled on an 11×11 grid; if the worst pixel-equivalent
error exceeds 1.0 pixel, the affine block is dropped and
``lsst.wcs_simplified_dropped: true`` is recorded with the observed
maximum. The OME block is informational only — readers always
reconstruct the projection from the JSON tree.

Cloud-friendly defaults
-----------------------

- Default chunk geometry is tile-aligned: ``min(256, dim)`` per
  axis for plain images, the per-array ``tile_shape`` hint (the cell
  shape) for ``CellCoadd``, single-cell for ``CellCoadd``'s 4-D PSF.
  The per-axis cap is configurable via the `DEFAULT_CHUNK_AXIS_LIMIT`
  constant.
- Bulk pixel arrays (``image``, ``variance``, ``mask``, and
  ``CellCoadd``'s ``psf``) are sharded by default to keep object
  counts on S3 / GCS low. The shard size is chosen by a byte-budget
  rule (~16 MiB by default; tunable via the
  ``LSST_IMAGES_ZARR_TARGET_SHARD_BYTES`` environment variable).
  Tiny single-chunk arrays (``lsst_json``, ``wcs_ast``, FITS
  opaque-metadata blocks) stay unsharded.
- Subset reads via ``slices=`` to
  `~lsst.images.serialization.InputArchive.get_array` exploit zarr's
  chunk index: only chunks intersecting the slice are fetched, even
  from remote stores.
- Both ``DirectoryStore`` and ``ZipStore`` are supported; the choice
  is driven by URI shape (``*.zarr.zip`` → ``ZipStore``, otherwise
  directory). Remote URIs (``s3://``, ``gs://``, ``http(s)://``) go
  through `lsst.resources.ResourcePath` and ``fsspec``.

Round-trip with FITS
--------------------

When an object that originated from a FITS read carries a
`~lsst.images.fits.FitsOpaqueMetadata`, the primary-HDU header is
preserved at ``/lsst/opaque_metadata/fits/primary``. Reading the
zarr back attaches an equivalent ``FitsOpaqueMetadata`` to the
deserialized object so a subsequent FITS write reproduces the
original cards.

Optional install
----------------

This backend requires ``zarr >= 3.0``. Install via the ``[zarr]``
extra::

    pip install lsst-images[zarr]

The top-level ``import lsst.images.zarr`` raises a clear
`ImportError` with this guidance if ``zarr`` is not installed.
"""

try:
    import zarr  # noqa: F401
except ImportError as e:
    raise ImportError(
        "lsst.images.zarr requires the optional 'zarr' package (>=3.0). "
        "Install it directly or via 'pip install lsst-images[zarr]'."
    ) from e

from ._common import *
from ._input_archive import *
from ._output_archive import *
