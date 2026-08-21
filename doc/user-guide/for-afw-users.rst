.. py:currentmodule:: lsst.images

.. _guide-for-afw-users:

For `lsst.afw` Users
====================

The `lsst.images` package is heavily inspired by (and is intended to partially supersede) the `lsst.afw` and `lsst.geom` packages.
Most of the types in `lsst.images` have a direct counterpart in `lsst.afw`, with bidirectional conversions between them (generally called ``to_legacy`` or ``from_legacy``, sometimes with additional suffixes).
Despite these conceptual similarities, the interfaces are often quite different in detail, generally because this is an opportunity to make interface improvements that are now difficult to make in `lsst.afw` or `lsst.geom`.

Geometry
--------

``lsst.geom.Point*``, `lsst.geom.Extent*`, and Coordinate Ordering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

There are no true point or extent types in `lsst.images`.
The philosophy is instead that we will usually want to operate on pairs of *arrays* of points, and hence the focus is on vectorized, ``ufunc``-like interfaces that take ``x`` and ``y`` arguments.

Because of the ubiquity of both ``(x, y)`` ordering and ``(y, x)`` ordering in Astropy, NumPy, and other libraries we interoperate with, `lsst.images` does *not* impose a uniform consistent order for such pairs across all interfaces.
Instead, pairs of ``x`` and ``y`` arguments are almost always keyword-only, and functions that return coordinate pairs (or pairs of coordinate arrays) use the `XY` or `YX` named tuples, which should generally be unpacked via their ``x`` and ``y`` attributes but are still formally `tuple` objects, for cases (e.g. shapes of arrays) where a `tuple` is needed.

.. note::

   `XY` and `YX` are tuples, so their ``+`` and ``*`` operators correspond to the `collections.abc.Sequence` definitions (concatenation and duplication).
   They are *not* point-like types with point-like operators, even though they are the closest thing `lsst.images` has to a point type.

**Conversions**

From `lsst.images` to ``lsst.geom``:

- `XY.to_legacy_int_extent`
- `XY.to_legacy_int_point`
- `XY.to_legacy_float_extent`
- `XY.to_legacy_float_point`
- `YX.to_legacy_int_extent`
- `YX.to_legacy_int_point`
- `YX.to_legacy_float_extent`
- `YX.to_legacy_float_point`

Conversions from ``lsst.geom`` points and extents are not provided, because most `lsst.images` signatures accept ``x`` and ``y`` kwargs rather than `XY` or `YX`.


Intervals, Boxes, and Polygons
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. rst-class:: table-sm

.. list-table::
   :header-rows: 1
   :widths: 20 20 40

   * - ``lsst.geom`` / ``lsst.afw.geom`` type
     - `lsst.images` equivalent
     - Notes and caveats
   * - `lsst.geom.Point2I`, `lsst.geom.Point2D`, `lsst.geom.Extent2I`, `lsst.geom.Extent2D`
     - `XY` / `YX` named tuples or keyword-only ``x=``, ``y=`` arguments
     - There are no dedicated point or extent types in `lsst.images`.
       Functions take keyword-only ``x`` and ``y`` arguments.
       Return values use `XY` (x-first) or `YX` (y-first) named tuples.
       Conversions: `XY.to_legacy_int_point`, `XY.to_legacy_float_point`,
       `YX.to_legacy_int_point`, etc.
   * - `lsst.geom.IntervalI`
     - `Interval`
     - ``begin`` / ``end`` (half-exclusive) renamed to ``start`` / ``stop``
       (Python convention). ``min`` / ``max`` (inclusive pixel-center bounds) unchanged. As in `lsst.geom.Interval`, `Interval.min` and `Interval.max` are the inclusive bounds, which are integers that correspond to the *centers* of the outermost pixels included in the interval; this means the interval size is actually ``1 + max - min``, and floating point coordinates between ``min - 0.5`` and ``max + 0.5`` are actually included in the interval.
       Convert with `Interval.to_legacy` / `Interval.from_legacy`.
   * - `lsst.geom.Box2I`
     - `Box`
     - `Box` is **immutable**.
       `Box.min`, `Box.max`, `Box.start`, and `Box.stop` all return `YX` tuples.
       As a `Box` is ultimately just a pair of ``y`` and ``x`` `Interval` objects, all of the `Interval` nomenclature changes and bounds definitions apply to `Box` as well.
       Convert with `Box.to_legacy` / `Box.from_legacy`.
   * - `lsst.geom.Box2D`, `lsst.geom.IntervalD`
     - `Polygon`
     - There is no dedicated floating-point box class; use `Polygon` or the more general
       `Region` (arbitrary polygon sets with holes).
       Convert with `Polygon.to_legacy` / `Polygon.from_legacy`.
   * - `lsst.geom.Angle`, `lsst.sphgeom.Angle`
     - ``astropy.units.Quantity``
     - Use Astropy angle quantities with explicit units.
   * - `lsst.geom.SpherePoint`, `lsst.sphgeom.LonLat`
     - ``astropy.coordinates.SkyCoord``
     - Use Astropy sky coordinates.
   * - `lsst.afw.geom.Polygon`
     - `Polygon` / `Region`
     - `Region` is the more general base class representing arbitrary sets of polygons
       (with holes) in a Euclidean (e.g. pixel) coordinate system.
       Convert with `Polygon.to_legacy` / `Polygon.from_legacy`.

Note: `Box` and `Region` (and by extension, `Polygon`) all satisfy the `Bounds` `~typing.Protocol`, allowing them to be attached to various objects (e.g. `~psfs.PointSpreadFunction`, `Transform`) to specify the region where those objects are valid.
The `Bounds` system is not yet fully implemented in `lsst.images`, but the goal is to provide consistent control options (e.g. raise, warn, extrapolate) for handling out-of-bounds positions across the library.


Coordinate Systems and Transforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`SkyProjection` and `Transform` differ from their `lsst.afw.geom` counterparts in that
they can identify the frames they transform between
(e.g. the pixels of a particular ``{visit, detector}`` and the ICRS sky),
via an object that satisfies the `Frame` `~typing.Protocol`.
This additional information needs to be provided when creating an `lsst.images`
type from an `lsst.afw.geom` one (e.g. via `SkyProjection.from_legacy`).

.. rst-class:: table-sm

.. list-table::
   :header-rows: 1
   :widths: 20 20 40

   * - ``lsst.geom`` / ``lsst.afw.geom`` type
     - `lsst.images` equivalent
     - Notes and caveats

   * - `lsst.afw.geom.SkyWcs`
     - `SkyProjection`
     - `SkyProjection` can identify the frames it transforms between.
       Convert with `SkyProjection.to_legacy` / `SkyProjection.from_legacy`.
       Both types can be (but are not necessarily!) representable as FITS WCS, and are capable of carrying around their own FITS WCS approximation.
   * - `lsst.afw.geom.TransformPoint2ToPoint2` (and related instantiations)
     - `Transform`
     - Convert with `Transform.to_legacy` / `Transform.from_legacy`.
       (Mostly used to represent camera geometry coordinate transforms)
   * - `lsst.afw.cameraGeom.TransformMap`
     - `cameras.CameraFrameSet`
     - Convert with `cameras.CameraFrameSet.from_legacy`.

General-Purpose Images
----------------------

All image-like objects in `lsst.images` inherit from `GeneralizedImage`, which allows any number of image planes that correspond to a single (optional) `SkyProjection` and bounding `Box`.

Pixel indexing conventions in the two libraries are the same: the center of the lower-left pixel of most images is ``(0, 0)`` (and always a pair of integers).

.. rubric:: Types

.. rst-class:: table-sm

.. list-table::
   :header-rows: 1
   :widths: 20 20 40

   * - ``lsst.afw.image`` type
     - `lsst.images` equivalent
     - Notes and caveats
   * - `lsst.afw.image.Image`
     - `Image`
     - `Image` can also hold a `SkyProjection`, flexible metadata, and
       ``astropy.units`` units.
       Convert with `Image.to_legacy` / `Image.from_legacy`.
   * - `lsst.afw.image.Mask`
     - `Mask`
     - `Mask` can also hold a `SkyProjection` and flexible metadata. Its backing array is 3-d `numpy.uint8` array with shape ``(height, width, N)``, where ``N`` can change depending on the number of mask planes (which is fully dynamic). Mask planes are fully dynamic; there is no global state.
       A separate `MaskSchema` manages shared plane definitions.
       Use `Mask.get`, `Mask.set`, and `Mask.clear` for bitwise operations in most cases.
       Convert with `Mask.to_legacy` / `Mask.from_legacy`.
   * - `lsst.afw.image.MaskedImage`
     - `MaskedImage`
     - `MaskedImage` can also hold a `SkyProjection`, flexible metadata, and units.
       Convert with `MaskedImage.to_legacy` / `MaskedImage.from_legacy`.
   * - `lsst.afw.image.Exposure`
     - One of `VisitImage`, `cells.CellCoadd`, or `DifferenceImage`
     - See below for details on the three specific `lsst.images` types.


.. rubric:: Attributes and Methods

.. rst-class:: table-sm

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - ``lsst.afw.image`` method/attribute
     - `lsst.images` equivalent
     - Notes and caveats
   * - ``.image`` / ``.getImage()``
     - ``.image``
     - Returns an `Image` object. Access pixel data as a ``numpy.ndarray`` via
       ``.image.array`` (or ``.array`` directly on a plain `Image`).
   * - ``.mask`` / ``.getMask()``
     - ``.mask``
     - Returns a `Mask` object.
   * - ``.variance`` / ``.getVariance()``
     - ``.variance``
     - Returns an `Image` whose ``.array`` is the variance plane.
   * - ``.wcs`` / ``.getWcs()``
     - ``.sky_projection``
     - Returns a `SkyProjection` instead of ``lsst.afw.geom.SkyWcs``.
       `SkyProjection` can identify the frames it transforms between.
       Convert with `SkyProjection.to_legacy` / `SkyProjection.from_legacy`.
   * - ``xy0`` / ``.getXY0()`` / ``.setXY0()``
     - ``.yx0``
     - **Axis order is reversed**: ``xy0`` → ``yx0``, returning a `YX` named tuple
       ``(y, x)`` consistent with NumPy ``(row, col)`` convention.
   * - ``.getBBox()``
     - ``.bbox``
     - Returns a `Box` instead of ``lsst.geom.Box2I``.
       `Box` is **immutable**.
       Convert with `Box.to_legacy` / `Box.from_legacy`.
   * - ``.getDimensions()``
     - ``.bbox.shape``
     - Returns a `YX` named tuple ``(height, width)`` consistent with NumPy convention.
   * - ``.getWidth()`` / ``.getHeight()``
     - ``.bbox.shape.x`` / ``.bbox.shape.y``
     - Access the ``x`` (width) or ``y`` (height) attribute on the `YX` shape tuple.
   * - ``lsst.afw.image.PARENT``
     - Default / ``.absolute``
     - The offset-aware PARENT coordinate system is used by default everywhere in
       `lsst.images`. Access it explicitly via the `GeneralizedImage.absolute`
       slicing proxy. As in `lsst.afw.image`, underlying `numpy.ndarray` view attributes do not know about
       this offset, and instead operate in what is called the `lsst.afw.image.LOCAL`
       coordinate system in `lsst.afw`.
   * - ``lsst.afw.image.LOCAL``
     - ``.local``
     - The origin-zero LOCAL coordinate system is accessed via the
       `GeneralizedImage.local` slicing proxy.
   * - Slice with `lsst.geom.Box2I` in PARENT coords
     - ``.absolute[box]``
     - Use the `GeneralizedImage.absolute` proxy to slice in absolute (offset-aware)
       pixel coordinates. This is also the default indexing mode.
   * - Slice with `lsst.geom.Box2I` in LOCAL coords
     - ``.local[box]``
     - Use the `GeneralizedImage.local` proxy to slice in local (origin-zero) pixel
       coordinates.
   * - ``.getCutout()``
     - ``.bbox_from_sky_circle()``
     - Returns a `Box`, which can then be used to slice the image. For example:

       .. code-block:: python

          from astropy.coordinates import SkyCoord
          from astropy import units as u
          center = SkyCoord(ra=ra, dec=dec, unit="deg")
          box = image.bbox_from_sky_circle(center, 60 * u.arcsec)
          cutout = image[box]



Single-Visit Images
-------------------

When used to represent a calibrated single-visit, single-detector image, `lsst.afw.image.Exposure` corresponds to `VisitImage` or `DifferenceImage`, which are subclasses of `MaskedImage`.

**Conversions**

- `VisitImage.to_legacy`
- `VisitImage.from_legacy`
- `DifferenceImage.to_legacy`
- `DifferenceImage.from_legacy`

Most `lsst.afw.image.Exposure` components have `VisitImage`
or `DifferenceImage` counterparts:

.. rubric:: Attributes and Methods

.. rst-class:: table-sm

.. list-table::
   :header-rows: 1
   :widths: 20 20 40

   * - ``lsst.afw.image`` method/attribute
     - `lsst.images` equivalent
     - Notes and caveats
   * - ``.psf`` / ``.getPsf()``
     - ``.psf``
     - Returns a `psfs.PointSpreadFunction` instead of ``lsst.afw.detection.Psf``.
       A position must always be supplied when evaluating the PSF.
   * - ``.photoCalib`` / ``.getPhotoCalib()``
     - ``.photometric_scaling``
     - Returns a `fields.BaseField` (a spatially variable scalar field) instead of
       ``lsst.afw.image.PhotoCalib``.
       Convert with `fields.field_from_legacy_photo_calib` and
       `fields.BaseField.to_legacy_photo_calib`.
   * - ``.visitInfo`` / ``.getVisitInfo()``
     - ``.obs_info``
     - Returns an ``astro_metadata_translator.ObservationInfo`` instead of
       ``lsst.afw.image.VisitInfo``.
   * - ``.summaryStats`` / ``.getSummaryStats()``
     - ``.summary_stats``
     - Returns an `ObservationSummaryStats` instead of
       ``lsst.afw.image.ExposureSummaryStats``.
   * - ``.detector`` / ``.getDetector()``
     - ``.detector``
     - Returns a `cameras.Detector` instead of ``lsst.afw.cameraGeom.Detector``.
       `cameras.Detector` is mutable (no builder pattern needed).
       Convert with `cameras.Detector.to_legacy` / `cameras.Detector.from_legacy`.
   * - ``.apCorrMap`` / ``.getApCorrMap()``
     - ``.aperture_corrections``
     - Returns a ``dict`` of `fields.BaseField` objects instead of
       ``lsst.afw.image.ApCorrMap``.
   * - ``.validPolygon`` / ``.getValidPolygon()``
     - ``.bounds``
     - Returns any `Bounds` implementation (typically `Polygon`) instead of
       ``lsst.afw.geom.Polygon``.
       Convert with `Polygon.to_legacy` / `Polygon.from_legacy`.
   * - Backgrounds (saved separately from ``ExposureF``)
     - ``.backgrounds``
     - `VisitImage` can hold one or more background models as `fields.BaseField` objects
       directly on the image object.




Coadd Images
------------

Coadded images can be represented outside of `lsst.images` by any of the following three types:

- `lsst.afw.image.Exposure`: traditional coadds (including templates) with `lsst.meas.extensions.CoaddPsf` that are evaluated by warping and coadding per-visit PSFs on-the-fly, as well as post-detection deep cell-based coadds (for compatibility with most coadd measurement tasks).

- `lsst.cell_coadds.MultipleCellCoadd`: the immediate result of building a cell-based coadd.

- `lsst.cell_coadds.StitchedCoadd`: an intermediate object that keeps all of the extra information in a cell-based coadd while having traditional full-patch arrays for the image planes and mask, but does not have any I/O support.

The `cells.CellCoadd` most closely resembles `~lsst.cell_coadds.StitchedCoadd`; it inherits from `MaskedImage` and hence has full-array image and mask planes, but its PSF model, bounds, and provenance data structures are explicitly cell-based.
It can fully represent a `~lsst.cell_coadds.MultipleCellCoadd` or `~lsst.cell_coadds.StitchedCoadd` when the skymap has no cell overlap regions, and will also typically hold the additional mask information (e.g. the ``DETECTED`` plane) and background offset held by downstream `lsst.afw.image.Exposure` datasets.

Because image subtraction templates are now Rubin's only traditional coadd data product, but the spatial variation of those coadd PSFs is not used by the image subtraction pipeline (the difference kernel is fit directly to the pixels of both images), we plan to convert these to the `~cells.CellCoadd` data structure by approximating their PSFs as cell-based, i.e. evaluating the `~lsst.meas.algorithms.CoaddPsf` model at the centers of cells.
This is roughly equivalent to a procedure that builds templates as "edgy" cell coadds, in which visit-cell combinations that do not wholly overlap a cell are nevertheless included in the coadd (which is what we may do in the future, when `lsst.images` types are used as direct pipeline outputs).

The `~cells.CellCoadd` type has counterparts for only some of the components of `lsst.afw.image.Exposure`:

.. rubric:: Types

.. rst-class:: table-sm

.. list-table::
   :header-rows: 1
   :widths: 20 20 40

   * - `lsst.afw` / ``lsst.cell_coadds`` type
     - `cells.CellCoadd` equivalent
     - Notes and caveats
   * - `lsst.afw.image.Exposure` (traditional coadd including templates) or
       `lsst.cell_coadds.StitchedCoadd`
     - `cells.CellCoadd`
     - `~cells.CellCoadd` most closely resembles ``StitchedCoadd``: full-array image and
       mask planes, but an explicitly cell-based PSF, bounds, and provenance data structures.
       Convert with `cells.CellCoadd.to_legacy`.
   * - `lsst.cell_coadds.MultipleCellCoadd`
     - `cells.CellCoadd`
     - Convert with `cells.CellCoadd.from_legacy` (from ``MultipleCellCoadd``) and
       `cells.CellCoadd.to_legacy_cell_coadd` (back to ``MultipleCellCoadd``).
   * - `lsst.meas.algorithms.CoaddPsf` (on coadd ``Exposure``)
     - `cells.CellPointSpreadFunction`
     - Cell-based PSF model.
       Convert with `cells.CellPointSpreadFunction.to_legacy` /
       `cells.CellPointSpreadFunction.from_legacy`.

.. rubric:: Attributes and Methods

.. rst-class:: table-sm

.. list-table::
   :header-rows: 1
   :widths: 20 20 40

   * - ``lsst.afw`` / ``lsst.cell_coadds`` attributes
     - `cells.CellCoadd` equivalent
     - Notes and caveats
   * - ``.psf`` / ``.getPsf()``
     - ``.psf``
     - Returns a `~cells.CellPointSpreadFunction` instead of ``lsst.afw.detection.Psf``.
   * - ``.apCorrMap`` / ``.getApCorrMap()``
     - ``.aperture_corrections``
     - Returns a ``dict`` of `cells.CellField` objects instead of `lsst.afw.image.ApCorrMap`.
   * - ``validPolygon`` / per-cell validity (``MultipleCellCoadd``)
     - ``.bounds``
     - Returns a `~cells.CellGridBounds` that records which cells have data, rather
       than a single polygon.

Cell coadds can also store one or more background models (`~cells.CellCoadd.backgrounds`), which have to be saved separately from `lsst.afw.image.Exposure`.


BoundedFields and Backgrounds
-----------------------------

`lsst.afw.math.BoundedField` (used directly for aperture corrections and indirectly by `lsst.afw.image.PhotoCalib`) corresponds directly to the `fields.BaseField` base class, whose subclasses are closed to the ``fields.Field`` type-union (i.e. no external implementations are permitted; this greatly simplifies serialization).
All `fields.BaseField` objects can be associated with units (via `astropy.units`).

The `fields.field_from_legacy` and `fields.field_from_legacy_background` free functions can be used to convert from `lsst.afw` when the exact type is unknown.

.. rst-class:: table-sm

.. list-table::
   :header-rows: 1
   :widths: 20 20 40

   * - ``lsst.afw`` type
     - `lsst.images` type
     - Notes and caveats
   * - `lsst.afw.math.BoundedField`
     - `fields.BaseField`
     - All `fields.BaseField` objects can carry ``astropy.units`` units.
       No external implementations are permitted (closed ``fields.Field`` type-union),
       which simplifies serialization.
       Use `fields.field_from_legacy` when the exact subtype is unknown.
   * - `lsst.afw.math.ChebyshevBoundedField`
     - `fields.ChebyshevField`
     - Also: `fields.ChebyshevField.to_legacy_function2` /
       `fields.ChebyshevField.from_legacy_function2` for
       ``lsst.afw.math.Chebyshev1Function2``.
   * - `lsst.afw.math.ProductBoundedField`
     - `fields.ProductField`
     - Convert with `fields.BaseField.to_legacy` / `fields.field_from_legacy`.
   * - `lsst.afw.math.BackgroundMI` (interpolated)
     - `fields.SplineField`
     - Use `fields.field_from_legacy_background` to convert from
       ``BackgroundMI`` or ``BackgroundList``.
   * - `lsst.afw.math.BackgroundMI` (approximated)
     - `fields.ChebyshevField`
     - Use `fields.field_from_legacy_background`.
   * - `lsst.afw.math.BackgroundList`
     - `fields.SumField`
     - The `lsst.afw.math.BackgroundMI` and `lsst.afw.math.BackgroundList` types
       are *also* mapped to the `fields.BaseField` hierarchy in `lsst.images`,
       since those are also just calculated images.
       Use `fields.field_from_legacy_background`.
   * - `lsst.afw.image.PhotoCalib`
     - `fields.BaseField`
     - Convert with `fields.field_from_legacy_photo_calib` and
       `fields.BaseField.to_legacy_photo_calib`.
   * - `lsst.cell_coadds.StitchedApertureCorrection`
     - `cells.CellField`
     - `cells.CellField` is a true `fields.BaseField`,
       but it is not directly serializable; the containing ``dict`` is serialized as a unit.
       (instead, a `dict` with `~cells.CellField` values is serialized all at once),
       and is hence not a member of the ``fields.Field`` type-union.
       Convert with `cells.CellField.to_legacy_aperture_correction` /
       `cells.CellField.from_legacy_aperture_correction`.
       `~lsst.cell_coadds.StitchedApertureCorrection` is not a true `~lsst.afw.math.BoundedField`
       (it acts like one just enough to be used to apply aperture corrections);



PSF Models
----------

The `lsst.afw.detection.Psf` class corresponds directly to `psfs.PointSpreadFunction`.
There is no concept of "average position" in `~psfs.PointSpreadFunction`, so a position must always be used to evaluate the PSF model.
The `~psfs.PointSpreadFunction` interface also currently lacks a way to represent wavelength-dependent PSF models, as we do not want to rush the in-code definition of the independent spectral-dimension variable.

.. rst-class:: table-sm

.. list-table::
   :header-rows: 1
   :widths: 20 20 40

   * - ``lsst.afw`` PSF type
     - `lsst.images` PSF type
     - Notes and caveats
   * - `lsst.afw.detection.Psf`
     - `psfs.PointSpreadFunction`
     - A position must always be provided when evaluating the PSF.
       There is no "average position" concept.
       Wavelength-dependent PSF models are not yet supported.
   * - `lsst.meas.extensions.piff.PiffPsf`
     - `psfs.PiffWrapper`
     - Convert with `psfs.PiffWrapper.to_legacy` / `psfs.PiffWrapper.from_legacy`.
   * - `lsst.meas.extensions.psfex.PsfExPsf`
     - `psfs.PSFExWrapper`
     - Wraps the legacy type; cannot be used when ``lsst.meas.extensions.psfex``
       cannot be imported.
       Convert with `psfs.LegacyPointSpreadFunction.to_legacy` /
       `psfs.LegacyPointSpreadFunction.from_legacy`. (inherited by `psfs.PSFExWrapper`)
   * - `lsst.afw.detection.SingleGaussianPsf`
     - `psfs.GaussianPointSpreadFunction`
     - Native implementation; no legacy wrapper needed.
   * - `lsst.cell_coadds.StitchedPsf`
     - `cells.CellPointSpreadFunction`
     - Convert with `cells.CellPointSpreadFunction.to_legacy` /
       `cells.CellPointSpreadFunction.from_legacy`.

Camera Geometry
---------------

The `lsst.afw.cameraGeom.Amplifier` class corresponds directly to `cameras.Amplifier`.

.. rst-class:: table-sm

.. list-table::
   :header-rows: 1
   :widths: 20 20 40

   * - ``lsst.afw.cameraGeom`` type
     - `lsst.images` type
     - Notes and caveats
   * - `lsst.afw.cameraGeom.Detector`
     - `cameras.Detector`
     - `cameras.Detector` is mutable; no builder pattern needed.
       Convert with `cameras.Detector.to_legacy` / `cameras.Detector.from_legacy`.
   * - `lsst.afw.cameraGeom.Amplifier`
     - `cameras.Amplifier`
     - Assembly-stage bounding boxes are factored into optional sub-objects.
       Electronic parameters are superseded by calibration datasets for Rubin data.
       Convert with `cameras.Amplifier.to_legacy_builder` /
       `cameras.Amplifier.from_legacy`.
       Also: `cameras.AmplifierRawGeometry.from_legacy_amplifier` and
       `cameras.AmplifierCalibrations.from_legacy_amplifier`.
   * - `lsst.afw.cameraGeom.DetectorType`
     - `cameras.DetectorType`
     - Now a `~enum.StrEnum` (values: ``SCIENCE``, ``FOCUS``, ``GUIDER``, ``WAVEFRONT``).
       Convert with `cameras.DetectorType.to_legacy` / `cameras.DetectorType.from_legacy`.
   * - `lsst.afw.cameraGeom.ReadoutCorner`
     - `cameras.ReadoutCorner`
     - Now a `~enum.StrEnum` (values: ``LL``, ``LR``, ``UR``, ``UL``).
       Lives on `cameras.Amplifier` via its raw geometry.
       Convert with `cameras.ReadoutCorner.to_legacy` / `cameras.ReadoutCorner.from_legacy`.
   * - `lsst.afw.cameraGeom.Orientation`
     - `cameras.Orientation`
     - Now a pydantic ``BaseModel``. Rotation angles (``yaw``, ``pitch``, ``roll``) are
       ``astropy.units.Quantity`` instead of ``lsst.geom.Angle``.
       The pixel reference point default changed from ``(-0.5, -0.5)`` (lower-left corner)
       to ``(0.5, 0.5)`` (center of the lower-left pixel).
       Convert with `cameras.Orientation.to_legacy` / `cameras.Orientation.from_legacy`.
