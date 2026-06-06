Zarr I/O
========

This is an *experimental* serialization backend to demonstrate that the package can be used to write files in formats other than FITS.

A Zarr v3 serialization backend whose on-disk layout is xarray/CF-shaped at the root (``image`` / ``variance`` / ``mask`` as siblings sharing ``(y, x)`` dimensions, CF ``flag_masks`` / ``flag_meanings`` on the mask) with OME-NGFF v0.5 multiscales metadata as a discoverability layer pointing at the same ``image`` array.
The same bytes are visible to ``xarray``, GDAL's Zarr driver, and OME-Zarr tooling like ``napari`` and ``ome-zarr-py``.

Default chunking is tile-aligned — 256 pixels per spatial axis for plain images, ``cell_shape`` for ``CellCoadd`` — and bulk pixel arrays are sharded with a ~16 MiB byte budget so a typical archive is a small handful of objects rather than thousands of chunk files.
Subset reads via ``slices=`` only fetch the chunks they need, including on remote stores accessed through ``lsst.resources.ResourcePath`` and ``fsspec``.

This backend requires the optional ``zarr >= 3.0`` package. Install via the ``[zarr]`` extra::

    pip install lsst-images[zarr]

Standards alignment
-------------------

The on-disk container is `Zarr v3 <https://zarr.dev>`_.
On top of that we layer four community standards so the same bytes are usable by tools that don't know anything about LSST:

* `xarray / CF-conventions <https://cfconventions.org>`_ — every array carries an ``_ARRAY_DIMENSIONS`` attribute and a v3 ``dimension_names`` metadata field.  The mask carries CF ``flag_masks`` / ``flag_meanings`` / ``flag_descriptions`` so any CF-aware tool can interpret the bit assignments.
* `OME-NGFF v0.5 <https://ngff.openmicroscopy.org>`_ — the root group carries a ``multiscales`` block whose only ``dataset.path`` points back at the same ``image`` array.  This makes the same archive openable by OME-Zarr tooling without any byte duplication.
* `Geo-Zarr <https://github.com/zarr-developers/geozarr-spec>`_ shape compatibility — sibling arrays sharing ``(y, x)`` dimensions with CF flag attributes is the same convention ``rasterio`` and ``GDAL``'s Zarr driver expect for raster + mask layers.
* `LSST archive tree <#data-model>`_ — a Pydantic JSON document at ``/lsst_json`` carries the full LSST-specific metadata (WCS, PSF, detector, butler info, …) that the community standards have no place for.  Same convention as the FITS backend's ``JSON`` HDU and the NDF backend's ``/MORE/LSST/JSON`` path.

Data model
----------

Every archive contains the following pieces:

``/lsst_json`` (1-D ``uint8``)
    UTF-8 encoded JSON of the Pydantic archive tree (see `~lsst.images.serialization.ArchiveTree`).
    The round-trip authority — every array reference, projection, PSF, mask schema, butler provenance, etc. lives here.
    SIP polynomials and other ``PolyMap``-based distortions round-trip byte-exact through the chain of `Mapping <https://starlink-pyast.readthedocs.io/en/latest/Mapping.html>`_ models embedded in this JSON.
    Stored as a single chunk because it is always read whole.

Root attributes (``zarr.json`` ``attributes``)
    Three namespaces:

    * ``lsst.*`` — backend-specific keys: ``archive_class``, ``json``, ``opaque_metadata_format``, ``cell_grid``, ``wcs_simplified_dropped``.
    * ``ome.*`` — OME-NGFF v0.5 ``multiscales`` block (and ``omero/channels`` when a channel axis exists).
    * top-level — CF / xarray attributes that aren't tied to a specific axis.

``/lsst/opaque_metadata/fits/primary`` (2-D ``(N, 80) uint8``)
    Present only when an object originated from a FITS read.
    Holds the primary HDU's card stream verbatim — ``Header.tostring()`` reshaped one row per card.
    ``COMMENT``, ``HISTORY``, ``HIERARCH``, and ``CONTINUE`` cards survive byte-for-byte.

Per-array data
    The ``image`` / ``variance`` / ``mask`` arrays at the root, plus any class-specific extras.
    Mask is a 2-D unsigned integer (``uint8`` for ≤8 planes, ``uint64`` for 17–64 planes; >64 raises) with CF ``flag_masks`` / ``flag_meanings`` / ``flag_descriptions``.

Chunking and sharding
---------------------

Chunks
    The default chunk shape per top-level array is ``min(256, dim)`` per axis for plain image arrays (``DEFAULT_CHUNK_AXIS_LIMIT`` in `lsst.images.zarr`).
    For `~lsst.images.cells.CellCoadd`, ``image`` / ``variance`` / ``mask`` chunks are aligned to ``cell_shape`` so a single-cell read is one chunk per array; the 4-D ``psf`` array is chunked ``(1, 1, Py, Px)`` so a single-cell PSF read is also one chunk.
    Sibling arrays (``variance`` / ``mask``) inherit the ``image`` array's chunk shape unless the caller passes an explicit override to `~lsst.images.zarr.write`.

Sharding
    Bulk pixel arrays (``image`` / ``variance`` / ``mask`` / ``CellCoadd``'s ``psf``) are sharded by default so a remote archive on S3 / GCS is a small number of objects rather than thousands of chunk files.
    The shard shape is chosen by a byte-budget rule that grows axes whose chunk does not already cover the full extent until each shard is close to ``LSST_IMAGES_ZARR_TARGET_SHARD_BYTES`` of uncompressed data; the default budget is 16 MiB.
    Shard axes are always integer multiples of the corresponding chunk axes, capped at the array extent.
    Tiny single-chunk arrays (``lsst_json``, ``wcs_ast``, the FITS opaque-metadata block, per-PSF parameter arrays whose chunks already cover the whole array) are left unsharded — sharding them would only add a layer of indirection.
    Sharding can be disabled or overridden per-array by passing ``shards={"image": None, ...}`` to `~lsst.images.zarr.write`.

Stores
    The store implementation is selected from the URI shape: a path ending in ``.zarr.zip`` (or any ``.zip``) opens a ``ZipStore``, a remote URI (``s3://``, ``gs://``, ``http(s)://``) opens a ``FsspecStore`` via `lsst.resources.ResourcePath`, and anything else opens a ``LocalStore`` directory.
    Two caveats worth knowing about:

    * Writing a ``ZipStore`` directly to a remote URI is not yet supported — write to a local ``.zarr.zip`` and upload, or write to a remote directory store. Reading a remote ``.zarr.zip`` works (the file is fetched to a local cache first via ``ResourcePath.as_local``, then opened).
    * After a directory or fsspec write, consolidated metadata is emitted so a single read fetches the whole hierarchy's ``zarr.json`` contents — a significant latency win on remote stores. ``ZipStore`` does not support consolidation; zip writes succeed without consolidated metadata, and reads of zip archives walk the hierarchy normally.

Example layouts
---------------

`~lsst.images.VisitImage`
^^^^^^^^^^^^^^^^^^^^^^^^^

The most common case — a single detector exposure with a projection, PSF, and detector geometry::

    visit.zarr/
    ├── zarr.json                                ← root attrs (lsst.archive_class="VisitImage",
    │                                              ome.multiscales, data_model, version, …)
    ├── image/                                   ← (Y, X) float32, dim_names=["y", "x"]
    ├── variance/                                ← (Y, X) float64
    ├── mask/                                    ← (Y, X) packed wide-int with CF flag attrs
    ├── lsst_json/                               ← 1-D uint8, the LSST archive tree
    ├── psf/                                     ← (PSF parameters as one or more arrays)
    └── lsst/opaque_metadata/fits/primary/       ← (N, 80) uint8 (when read from a FITS file)

The ``lsst_json`` tree carries the projection, PSF type, detector reference, observation summary stats, photometric scaling, aperture-correction map, and any background fields.
For the WCS specifically, the projection's ``pixel_to_sky`` mapping is decomposed into a chain of Frames and Mappings (including any ``PolyMap`` for SIP distortion); reading is byte-exact.

`~lsst.images.cells.CellCoadd`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A coadd composed of a regular grid of cells, each with its own PSF::

    coadd.zarr/
    ├── zarr.json                                ← lsst.archive_class="CellCoadd",
    │                                              lsst.cell_grid={bbox, cell_shape},
    │                                              ome.multiscales pointing at /image
    ├── image/                                   ← (Y, X) float32, chunks = cell_shape
    ├── variance/                                ← (Y, X) float64, chunks = cell_shape
    ├── mask/                                    ← (Y, X) packed wide-int, chunks = cell_shape
    ├── psf/                                     ← (Cy, Cx, Py, Px) float32,
    │                                              chunks=(1, 1, Py, Px) — one chunk per cell
    ├── lsst_json/
    └── lsst/opaque_metadata/fits/primary/

The ``image`` / ``variance`` / ``mask`` chunks are aligned to the cell grid so reading a single cell is one chunk per array.
The ``psf`` array's chunking is per-cell so a single-cell PSF read is also one chunk.

`~lsst.images.ColorImage`
^^^^^^^^^^^^^^^^^^^^^^^^^

A 3-channel display image::

    color.zarr/
    ├── zarr.json                                ← lsst.archive_class="ColorImage"
    │                                              (no root-level ome.multiscales)
    ├── red/                                     ← (Y, X) uint8, dim_names=["y", "x"]
    ├── green/                                   ← (Y, X) uint8
    ├── blue/                                    ← (Y, X) uint8
    └── lsst_json/

Channels are flat top-level arrays rather than a stacked ``(3, Y, X)`` array, so xarray sees them as three independent 2-D variables and there is no byte duplication for the OME view.

WCS handling
------------

The full WCS — including SIP polynomials and any other ``PolyMap``-based distortion — round-trips through the JSON tree at ``lsst_json`` as a chain of `~lsst.images.FrameSet` / Mapping models.
The layout layer also asks AST's `linearapprox <https://starlink-pyast.readthedocs.io/en/latest/Mapping.html#starlink.Ast.Mapping.linearapprox>`_ for an affine approximation over the image footprint at one-pixel accuracy.
If AST returns one, the OME ``coordinateTransformations`` block on the root multiscale is populated with the resulting ``[scale, affine]`` pair.
If AST cannot fit a linear approximation within tolerance, the block is dropped and ``lsst.wcs_simplified_dropped: true`` is set on the root attrs.
The OME block is always informational — readers reconstruct the projection from the JSON tree, never from the OME block.

Tooling that can read these files
---------------------------------

The standards-aligned root layout means tools that don't know about LSST can still open the file in some useful capacity:

`xarray <https://docs.xarray.dev>`_
    ``xr.open_zarr(path)`` returns a ``Dataset`` with one ``DataArray`` per zarr array sharing ``(y, x)`` dimensions, CF flag attributes on the mask variable, and any per-array ``units`` / ``long_name``.
    The Pydantic JSON tree at ``/lsst_json`` shows up as a 1-D ``uint8`` variable; xarray ignores it for analysis, you decode it manually if you need the LSST metadata.

`napari-ome-zarr <https://github.com/ome/napari-ome-zarr>`_ and `ome-zarr-py <https://ome-zarr.readthedocs.io>`_
    Browse and visualize the science image through the OME-NGFF multiscales block.
    Sees the ``image`` array as the only level of a single multiscale; ignores everything else.

`GDAL <https://gdal.org/drivers/raster/zarr.html>`_'s Zarr driver and `rasterio <https://rasterio.readthedocs.io>`_
    Opens individual top-level arrays as raster bands.
    Reads CF attributes including the mask's ``flag_masks`` / ``flag_meanings``.

`zarr-python <https://zarr.readthedocs.io>`_
    Direct array access at any path, including from S3 / GCS / HTTP via fsspec.
    Subset reads via ``arr[y0:y1, x0:x1]`` only fetch chunks intersecting the slice.

`napari <https://napari.org>`_ via the OME-Zarr plugin
    Same OME view as ``napari-ome-zarr``.

`neuroglancer <https://github.com/google/neuroglancer>`_
    Native OME-NGFF support; will display the science image with the affine ``coordinateTransformations`` block when present.

`ngff-zarr <https://ngff-zarr.readthedocs.io>`_ (with the ``validate`` extra)
    Validates the OME-NGFF v0.5 metadata block against the bundled OME-NGFF JSON schemas via ``ngff_zarr.validate``.

Round-trip with FITS
--------------------

When an object that originated from a FITS read carries a `~lsst.images.fits.FitsOpaqueMetadata`, the primary-HDU header is preserved at ``/lsst/opaque_metadata/fits/primary`` as a 2-D ``(N, 80)`` byte array.
Reading the zarr back attaches an equivalent ``FitsOpaqueMetadata`` to the deserialized object so a subsequent FITS write reproduces the original cards.
This means an ``LSSTCam`` raw read in via FITS, written to zarr, read back, and written again to FITS will round-trip the full primary header — including ``COMMENT``, ``HISTORY``, ``HIERARCH``, and ``CONTINUE`` cards — byte-for-byte.

API reference
-------------

.. automodapi:: lsst.images.zarr
   :no-inheritance-diagram:
   :include-all-objects:
   :inherited-members:
