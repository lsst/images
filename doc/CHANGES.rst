Changes
=======

lsst-images v30.0.9 (2026-07-15)
--------------------------------

New Features
""""""""""""

- Added the TEx PSF residual-ellipticity correlation metrics (``psfTE1e1`` through ``psfTE4ex``, covering TE1-TE4 and their e1/e2/ex components) to ``ObservationSummaryStats``, synchronized from ``lsst.afw.image.ExposureSummaryStats``. (`DM-46582 <https://rubinobs.atlassian.net/browse/DM-46582>`_)
- * Butler metadata-component reads now cache the serialization tree of the most recently read dataset, so requesting several metadata components of one dataset (for example ``sky_projection`` then ``obs_info``) opens the file only once.
    Components that need pixel, table, or archive-pointer data still fall back to reading the file, decided at runtime by probing the tree with the new ``lsst.images.serialization.DetachedArchive``.
  * The butler formatter now understands a special ``"components"`` component that returns several components from a single ``get`` as a `dict` keyed by component name.
    Pass a ``components`` parameter listing the components to read; if it is omitted, all components are returned.
  * Added a ``lsst-images-admin fuzz-masked-image`` subcommand that shuffles the image, mask, and variance pixels of a ``MaskedImage`` within tiles so a file can be released as public test data.
    It reads any supported format, writes the format chosen by the output extension, and applies lossy RICE compression to the science planes for FITS output.
  * Added ``lsst.images.Mask.add_planes`` (and the single-plane convenience ``add_plane``) to derive a new ``lsst.images.Mask`` with mask planes added and/or dropped; it always reallocates the backing array, so views of the original mask are unaffected.
    ``lsst.images.MaskedImage.mask`` is now settable, so a grown mask can be assigned back to a masked image.
  * Added ``lsst.images.Image.from_hdu_list`` and ``lsst.images.MaskedImage.from_hdu_list`` to reconstruct an ``Image`` or ``MaskedImage`` from the cut-down FITS HDU lists written by ``dax_images_cutout`` (an ``lsst.images`` file with its JSON-tree and index HDUs dropped), so they can be re-serialized as normal ``lsst.images`` files.
    Reconstruction from a legacy (``afw``-written) mask keeps the ``MP_*`` mask-plane cards (re-indexed to the reshuffled schema) for legacy tooling; the normal ``lsst.images`` reader strips them on read since the serialized schema is authoritative. (`DM-55210 <https://rubinobs.atlassian.net/browse/DM-55210>`_)
- Added difference kernel (``kernel``) and provenance (``templates``) components to ``DifferenceImage``. (`DM-55220 <https://rubinobs.atlassian.net/browse/DM-55220>`_)
- Added a ``lsst-images-admin diagram`` subcommand that renders the composition layout of a serialization model (e.g. ``visit-image`` or ``cell-coadd``) as Mermaid, Graphviz ``dot``, or an ASCII ``tree``.
  By default it labels nodes with the public class names (``Image``, ``SkyProjection``, ...) and shows only model composition; pass ``--attributes`` to include scalar fields, ``--serialization-names`` to use the raw model names, ``--collapse``/``--expand``/``--expand-leaves`` to tune which helper models are drawn, and ``--hide-field``/``--hide-type`` to clip fields or whole types.
  With ``--from-file`` it diagrams the concrete structure of a serialized file (reading only the reference tree, not pixels), collapsing unions such as the PSF to the type actually stored. (`DM-55281 <https://rubinobs.atlassian.net/browse/DM-55281>`_)
- The generic ``read_archive`` and ``open_archive`` APIs now accept a seekable binary stream in addition to a path, so in-memory data (e.g., a VO cutout service response) can be read without writing it to disk first: ``read_archive(io.BytesIO(data))``.
  The format is identified from the stream's leading bytes, with an optional ``format`` argument to override; compressed streams are rejected with an error telling the caller to decompress first.
  Paths with a ``.gz`` or ``.zst`` compression suffix are decompressed transparently through a temporary file for all supported formats.
  Reading zstd-compressed files requires Python 3.14 or the ``zstandard`` package. (`DM-55385 <https://rubinobs.atlassian.net/browse/DM-55385>`_)
- * Added support for broadcasting, array-like inputs, and ``XY`` / ``YX`` arguments to ufunc-like methods, including ``Bounds.contains``, ``Transform.apply*``, and ``BaseField.__call__``.
  * Added a ``MaskSchema.interpret`` method for identifying which mask planes are set on a given pixel. (`DM-55422 <https://rubinobs.atlassian.net/browse/DM-55422>`_)


API Changes
"""""""""""

- The generic serialization entry points have been renamed: ``lsst.images.serialization.read`` is now ``read_archive``, ``write`` is now ``write_archive``, and ``open`` is now ``open_archive``.
  The old names have been removed outright; no deprecation aliases are provided. (`DM-55421 <https://rubinobs.atlassian.net/browse/DM-55421>`_)
- * Changed ``CellCoadd.to_legacy`` API to return an ``lsst.afw.image.ExposureF`` to make it consistent with the other imaging classes.
    This allows for a consistent experience for the users where they do not have to check whether they have a ``VisitImage`` or ``CellCoadd`` to work out what the return type is going to be.
    The corollary of this change is that there is now ``CellCoadd.to_legacy_cell_coadd``.
    ``CellCoadd.to_legacy_exposure`` has been removed with no deprecation period.

  * The ``lsst.images.Box`` constructor now raises `TypeError` if its arguments are not `~lsst.images.Interval` instances, instead of silently constructing a broken box. (`DM-55488 <https://rubinobs.atlassian.net/browse/DM-55488>`_)


Bug Fixes
"""""""""

- * Unified the logic for determination of multiple versions of a single extension, fixing overwrites in the NDF writer.
  * Added a name shrinker for NDF output to ensure that files written out can be read by the Starlink software tools (which limits structure and component names to 15 characters). (`DM-55183 <https://rubinobs.atlassian.net/browse/DM-55183>`_)
- * Fixed read of bounding box cutouts for ``VisitImage`` and ``DifferenceImage``.

  * Each HDU written to a ``lsst.images`` FITS file now includes a ``DATE`` header card recording the UTC time the header was created, as required by the FITS standard. (`DM-55210 <https://rubinobs.atlassian.net/browse/DM-55210>`_)
- * Fixed reading of ``.fits.gz`` files, which were dispatched to the FITS backend but handed to it still compressed.
  * Fixed reading of ``cell_aperture_correction_map`` archives in a process that has not otherwise imported ``lsst.images.cells``: the schema was missing from the built-in provider table and the ``lsst.images.schemas`` entry points, so the schema-driven lazy import could not find it. (`DM-55385 <https://rubinobs.atlassian.net/browse/DM-55385>`_)
- * Fixed ``lsst.images.SkyProjection.to_legacy`` failing with ``No frame with domain PIXELS found`` for projections created by ``from_fits_wcs``, including projections read back from serialized files such as image cutouts.
    ``lsst.afw.geom.SkyWcs`` requires the pixel frame's AST domain to be ``PIXELS``, while this package follows the AST/NDF convention of ``PIXEL``; the domain is now renamed in the converted copy. (`DM-55488 <https://rubinobs.atlassian.net/browse/DM-55488>`_)


Miscellaneous Changes of Minor Interest
"""""""""""""""""""""""""""""""""""""""

- Improved the performance of the FITS reader when accessing remote objects by ensuring that we only open the file once (previously it was opened once to find the relevant schema and then again to do the full read) across all backends and for FITS specifically we have increased the block size used by ``fsspec``. (`DM-55217 <https://rubinobs.atlassian.net/browse/DM-55217>`_)
- Made generalized image ``__getitem__(...)`` calls return a view, not ``self``. (`DM-55422 <https://rubinobs.atlassian.net/browse/DM-55422>`_)


lsst-images v30.0.8 (2026-06-09)
--------------------------------

New Features
""""""""""""

- Added the ``CellCoadd`` class and format, as well as associated PSF and provenance types. (`DM-54225 <https://rubinobs.atlassian.net/browse/DM-54225>`_)
- Added psf star shapelet decomposition parameters and metrics to the ``ObservationSummaryStats`` class. (`DM-54482 <https://rubinobs.atlassian.net/browse/DM-54482>`_)
- Added a ``GeneralFrame`` type so we can represent transforms and projections where there are no identifiers (or none worth standardizing) for the pixel coordinate system. (`DM-54555 <https://rubinobs.atlassian.net/browse/DM-54555>`_)
- Added ``VisitImage.to_legacy`` for conversion from ``VisitImage`` to ``lsst.afw.image.Exposure``. (`DM-54556 <https://rubinobs.atlassian.net/browse/DM-54556>`_)
- Added schema versioning (``schema_version``, ``min_read_version``, computed ``schema_url``) to every top-level Pydantic serialization model, and integer-major container-format stamps (``FMTVER`` for FITS, ``FORMAT_VERSION`` for NDF, plus ``DATAMODL`` / ``DATA_MODEL`` schema-URL keywords).
  Files written before this change continue to read; absent stamps are interpreted as the v1 defaults. (`DM-54557 <https://rubinobs.atlassian.net/browse/DM-54557>`_)
- Added the ``Detector`` and ``Amplifiers`` classes, as well as a ``detector`` component for ``VisitImage``. (`DM-54558 <https://rubinobs.atlassian.net/browse/DM-54558>`_)
- Reworked the ``Polygon`` class into a two-type hierarchy (``Region`` is the new base class) that implements the ``Bounds`` protocol, and added a ``bounds`` attribute to ``VisitImage``. (`DM-54559 <https://rubinobs.atlassian.net/browse/DM-54559>`_)
- Added the ``fields`` subpackage, with 2-d Chebyshev, spline, product, and sum maps.

  These are intended as successors to the ``lsst.afw.math.BoundedField`` and
  ``lsst.afw.math.BackgroundList`` types. (`DM-54571 <https://rubinobs.atlassian.net/browse/DM-54571>`_)
- Added aperture corrections to ``VisitImage``. (`DM-54572 <https://rubinobs.atlassian.net/browse/DM-54572>`_)
- Added ``ObservationSummaryStats`` as a class analogous to the legacy ``afw`` ``ExposureSummaryStats`` class.
  This class is attached to the ``VisitImage.summary_stats`` property. (`DM-54575 <https://rubinobs.atlassian.net/browse/DM-54575>`_)
- Added higher-order moment-based metrics (``coma1[2]``, ``trefoil1[2]``, ``kurtosis``, ``e4_1``, and ``e4_2``) to the ``ObservationSummaryStats`` class. (`DM-54675 <https://rubinobs.atlassian.net/browse/DM-54675>`_)
- Added a ``json`` subpackage with support for pure JSON I/O (for small objects). (`DM-54780 <https://rubinobs.atlassian.net/browse/DM-54780>`_)
- * Added an experimental HDF5 file writer.
    The initial version supports Masked Image and Visit Image and uses the `Starlink NDF data model <https://ui.adsabs.harvard.edu/abs/2015A%26C....12..146J%2F/abstract>`_ layered on `top of HDF5 <https://ui.adsabs.harvard.edu/abs/2015A%26C....12..221J/abstract>`_.
    The files created by this writer are compatible with all the Starlink software with the caveat that the Starlink software does not understand more than 8 mask bits and the Starlink software does not understand the ``.MORE.LSST`` extensions.
    The Starlink software will understand FITS headers and our full WCS.
  * Unified the Butler formatters such that they now write the file format configured by the Butler datastore (defaulting to FITS) and will determine how to read the file based on the file extension. (`DM-54817 <https://rubinobs.atlassian.net/browse/DM-54817>`_)
- Added column for reference catalog source density (in number per degrees**2) to the ``ObservationSummaryStats`` class. (`DM-54866 <https://rubinobs.atlassian.net/browse/DM-54866>`_)
- Added a mapping of background models to ``VisitImage`` and ``CellCoadd``. (`DM-54910 <https://rubinobs.atlassian.net/browse/DM-54910>`_)
- Added a ``photometric_scaling`` component to ``VisitImage``. (`DM-54912 <https://rubinobs.atlassian.net/browse/DM-54912>`_)
- * Added the ``DifferenceImage`` class.
  * Added ``VisitImage.to_legacy`` and associated helpers and fixes.

    Includes a small modifications to the PIFF PSF serialization schema. (`DM-55036 <https://rubinobs.atlassian.net/browse/DM-55036>`_)
- * Added aperture corrections and legacy conversions to ``CellCoadd``.
  * Added support for FITS compression recipes in the butler formatter. (`DM-55129 <https://rubinobs.atlassian.net/browse/DM-55129>`_)
- * Added the ``lsst-images-admin`` command-line tool (also runnable as ``python -m lsst.images``) with ``convert`` (legacy FITS to a new format, for visit images, difference images, and cell coadds, with a ``--preserve-quantization`` option that is on by default), ``inspect`` (schema URL, container format version, and registered Python class), ``reformat`` (rewrite a file in a different container format, such as FITS to NDF), ``minify``, ``extract-test-data``, and ``verify-rewrite`` subcommands.
  * Added ``lsst.images.serialization.backend_for_path`` and ``InputArchive.get_basic_info`` as public APIs for resolving a backend by file suffix and reading basic archive information.
  * Added ``lsst.images.serialization.read`` and ``lsst.images.serialization.write``, which dispatch by file suffix; ``read`` infers the in-memory type from the file's schema and returns the deserialized object directly.
    Also added ``class_for_schema`` and ``public_type_for_schema`` for looking up the registered ``ArchiveTree`` subclass and the in-memory Python type for a ``schema_name``.
    ``open`` and ``read`` accept an optional ``cls`` argument that validates the deserialized type and narrows the static return type.
  * Added ``lsst.images.serialization.open``, a context-manager reader for efficiently pulling an individual component (``reader.get_component("projection")``), the whole object (``reader.read()``), or the metadata and butler info stored alongside it, layered on the new ``InputArchive.open_tree`` primitive.
  * Added ``GeneralizedImage.read`` (a classmethod) and ``GeneralizedImage.write``, inherited by ``Image``, ``Mask``, ``MaskedImage``, ``ColorImage``, ``VisitImage``, and ``DifferenceImage``, as discoverable thin wrappers over the generic ``read`` / ``write`` functions. (`DM-55131 <https://rubinobs.atlassian.net/browse/DM-55131>`_)


API Changes
"""""""""""

- * Removed the ``obs_info`` component from ``Image``, ``Mask``, and ``MaskedImage``, in favor of defining it directly on ``VisitImage``.
  * Fully unified the butler formatters into ``lsst.images.formatters.GenericFormatter`` and deleted the old ones. (`DM-54976 <https://rubinobs.atlassian.net/browse/DM-54976>`_)


Bug Fixes
"""""""""

- Bug fixes uncovered while strengthening round-trip tests:

  * ``CellPointSpreadFunctionSerializationModel.array`` now accepts ``InlineArrayModel`` alongside ``ArrayReferenceModel``, so a ``CellCoadd`` whose PSF kernel gets inlined (e.g. via the JSON archive) round-trips through Pydantic again.
  * The starlink-pyast ``Object`` wrapper now has a content-based ``__eq__`` (mirroring the structural equality already provided by ``astshim.Object``), so two ``FrameSet``\ s with identical content but different wrapper instances compare equal.
  * ``assert_psfs_equal`` no longer self-intersects ``psf1.bounds.bbox`` when picking default evaluation points, and compares ``bounds.contains`` symmetrically rather than relying on ``compute_kernel_image`` to raise — robust for ``CellPointSpreadFunction``, whose evaluation does not always raise on out-of-domain points. (`DM-55089 <https://rubinobs.atlassian.net/browse/DM-55089>`_)


Miscellaneous Changes of Minor Interest
"""""""""""""""""""""""""""""""""""""""

- Added structural equality and round-trip test helpers:

  * ``AmplifierCalibrations``, ``Detector``, ``CameraFrameSet``, and the ``BaseField``
    subclasses (``ChebyshevField``, ``ProductField``, ``SplineField``, ``SumField``) now define a content-based ``__eq__``, so two instances that round-trip identically compare equal.
  * New ``assert_visit_images_equal`` and ``assert_cell_coadds_equal`` helpers in ``lsst.images.tests`` cover every type-specific attribute (PSF, detector, filter, summary stats, backgrounds, polygon bounds, cell grid, missing cells, patch/tract, band, ...) so a single call now drives a full round-trip fidelity check.
  * ``TemporaryButler`` accepts a ``format=`` keyword that overlays a per-storage-class formatter binding so ``GenericFormatter`` writes the requested backend instead of falling back to its ``.fits`` default — used by the round-trip helpers to ensure butler-path artifacts match the extension the test asserts against. (`DM-55089 <https://rubinobs.atlassian.net/browse/DM-55089>`_)
- Reworked how ``ObservationInfo`` is shared and serialized in ``MaskedImage`` and its subclasses. (`DM-54555 <https://rubinobs.atlassian.net/browse/DM-54555>`_)
- Switched from 1-based to 0-based bit indexing in Mask FITS headers. (`DM-54617 <https://rubinobs.atlassian.net/browse/DM-54617>`_)
- Fixed models to support JSON Schema creation and validation. (`DM-54987 <https://rubinobs.atlassian.net/browse/DM-54987>`_)


An API Removal or Deprecation
"""""""""""""""""""""""""""""

- * Removed the per-backend ``read`` and ``read_tree`` functions (``lsst.images.fits.read``, ``lsst.images.json.read``, ``lsst.images.ndf.read``) and the ``read_fits`` / ``write_fits`` methods on ``Image``, ``Mask``, and ``MaskedImage``, along with the ``ReadResult`` return type.
    Use the ``read`` / ``write`` methods on the image classes (or the generic ``lsst.images.serialization.read`` / ``write`` with the optional ``cls`` argument) for whole-object access, and ``lsst.images.serialization.open`` for components, metadata, and butler info.
    Schema-less Starlink NDFs are now read with ``lsst.images.ndf.read_starlink``. (`DM-55131 <https://rubinobs.atlassian.net/browse/DM-55131>`_)


lsst-images v30.0.6 (2026-04-07)
--------------------------------

New Features
""""""""""""

- Implemented ``to_legacy`` conversion for ``Transform`` and ``Projection``. (`DM-54551 <https://rubinobs.atlassian.net/browse/DM-54551>`_)
- Added the ``ColorImage`` class and format for RGB images. (`DM-54220 <https://rubinobs.atlassian.net/browse/DM-54220>`_)
- Added a flexible metadata dictionary and optional butler provenance to all top-level serialization models and generalized images. (`DM-54285 <https://rubinobs.atlassian.net/browse/DM-54285>`_)
- Added more slicing support to all generalized images, with ``.local`` and ``.absolute`` proxy properties to make the indexing conventions clearer. (`DM-54292 <https://rubinobs.atlassian.net/browse/DM-54292>`_)
- Added ``GaussianPointSpreadFunction`` PSF class. (`DM-54472 <https://rubinobs.atlassian.net/browse/DM-54472>`_)


Miscellaneous Changes of Minor Interest
"""""""""""""""""""""""""""""""""""""""

- Improved the test coverage of the image classes.
  This uncovered some minor bugs that have also been fixed. (`DM-54472 <https://rubinobs.atlassian.net/browse/DM-54472>`_)

lsst-images v30.0.4 (2026-03-02)
--------------------------------

First public release of package.

New Features
""""""""""""

- Added FITS tile compression support and import-read support for ``lsst.afw.image.MaskedImage``. (`DM-53698 <https://rubinobs.atlassian.net/browse/DM-53698>`_)
- Added support for ``ObservationInfo`` to be attached to images.
  ``VisitImage`` will construct this object from legacy images. (`DM-54279 <https://rubinobs.atlassian.net/browse/DM-54279>`_)
