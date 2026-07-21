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

`lsst.geom.IntervalI` corresponds directly to `Interval`.
As in `lsst.geom.Interval`, `Interval.min` and `Interval.max` are the inclusive bounds, which are integers that correspond to the *centers* of the outermost pixels included in the interval; this means the interval size is actually ``1 + max - min``, and floating point coordinates between ``min - 0.5`` and ``max + 0.5`` are actually included in the interval.
The half-exclusive bounds have been renamed from ``begin`` (inclusive) and ``end`` (exclusive) in `lsst.geom` to ``start`` and ``stop`` in `lsst.images` (``begin`` and ``end`` is the standard nomenclature in C++, while ``start`` and ``stop`` are the standard names in Python).

`lsst.geom.Box2I` corresponds directly to `Box` (but the latter is immutable).
As a `Box` is ultimately just a pair of ``y`` and ``x`` `Interval` objects, all of the `Interval` nomenclature changes and bounds definitions apply to `Box` as well.
Note that `Box.min`, `Box.max`, `Box.start`, and `Box.stop` all return `YX` tuples, for consistency with `Box.shape`.

`lsst.geom.IntervalD` and `lsst.geom.Box2D` do not have direct counterparts in `lsst.images`.
2-d floating-point boxes are represented as `Polygon` objects; the expectation is that - unlike an integer-coordinate `Box` - there is nothing special about a floating-point rectangle that necessitates a dedicated class.

`lsst.geom.Angle`, `lsst.sphgeom.Angle`, `lsst.geom.SpherePoint`, and `lsst.sphgeom.LonLat` do not have direct counterparts in `lsst.images` itself, but the `astropy.units.Quantity` and `astropy.coordinates.SkyCoord` types are generally used in the same roles.

`lsst.afw.geom.Polygon` corresponds to `Polygon` and its more general `Region` base class, which can represent arbitrary sets of polygons (with holes) in a Euclidean (e.g. pixel) coordinate system.

`Box` and `Region` (and by extension, `Polygon`) all satisfy the `Bounds` `~typing.Protocol`, allowing them to be attached to various objects (e.g. `~psfs.PointSpreadFunction`, `Transform`) to specify the region where those objects are valid.
The `Bounds` system is not yet fully implemented in `lsst.images`, but the goal is to provide consistent control options (e.g. raise, warn, extrapolate) for handling out-of-bounds positions across the library.

**Conversions**

From `lsst.images` to ``lsst.[afw.]geom``:

- `Interval.to_legacy`
- `Box.to_legacy`
- `Polygon.to_legacy`

From ``lsst.[afw.]geom`` to `lsst.images`:

- `Interval.from_legacy`
- `Box.from_legacy`
- `Polygon.from_legacy`

Coordinate Systems and Transforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`lsst.afw.geom.SkyWcs` corresponds directly to `SkyProjection`.
Both types can be (but are not necessarily!) representable as FITS WCS, and are capable of carrying around their own FITS WCS approximation.

`lsst.afw.geom.TransformPoint2ToPoint2` and other instantiations of the same underlying C++ template (which are used to represent camera geometry coordinate transforms, mostly) correspond directly to `Transform`.

`SkyProjection` and `Transform` differ from their `lsst.afw.geom` counterparts in that they can identify the frames they transform between (e.g. the pixels of a particular ``{visit, detector}`` and the ICRS sky), via an object that satisfies the `Frame` `~typing.Protocol`.
This additional information needs to be provided when creating an `lsst.images` type from an `lsst.afw.geom` one (e.g. via `SkyProjection.from_legacy`).

`lsst.afw.cameraGeom.TransformMap` corresponds directly to `CameraFrameSet`.

**Conversions**

From `lsst.images` to ``lsst.afw``:

- `Transform.to_legacy`
- `SkyProjection.to_legacy`

From ``lsst.afw`` to `lsst.images`:

- `Transform.from_legacy`
- `SkyProjection.from_legacy`
- `CameraFrameSet.from_legacy` (`lsst.afw.cameraGeom.Camera`)


General-Purpose Images
----------------------

All image-like objects in `lsst.images` inherit from `GeneralizedImage`, which allows any number of image planes that correspond to a single (optional) `SkyProjection` and bounding `Box`.

Pixel indexing conventions in the two libraries are the same: the center of the lower-left pixel of most images is ``(0, 0)`` (and always a pair of integers).

The offset often referred to as ``xy0`` in `lsst.afw` is generally called ``yx0`` on `GeneralizedImage` subclasses, since the order has been swapped.

The `lsst.afw.image.PARENT` coordinate system that is aware of this offset is used by default on all `lsst.images` types, and can be used more explicitly via the `GeneralizedImage.absolute` slicing proxy.

As in `lsst.afw.image`, underlying `numpy.ndarray` view attributes do not know about this offset, and instead operate in what is called the `lsst.afw.image.LOCAL` coordinate system in `lsst.afw`.
The types in `lsst.images` can be sliced in this coordinate system via the `GeneralizedImage.local` slicing proxy.

`lsst.afw.image.Image` corresponds directly to `Image`, but the latter can also hold a `SkyProjection`, flexible metadata, and units (via `astropy.units`).

`lsst.afw.image.Mask` corresponds directly to `Mask`, but the latter can also hold a `SkyProjection` and flexible metadata, and its backing array is 3-d `numpy.uint8` array with shape ``(height, width, N)``, where ``N`` can change depending on the number of mask planes (which is fully dynamic).
This means that a "mask pixel" is actually a shape ``(N,)`` `numpy.uint8` array, but (thanks to automatic broadcasting) the usual bitwise operations still work.
The `Mask.get`, `Mask.set`, and `Mask.clear` convenience methods can be used instead of direct bitwise array operations in most cases.
The planes of different `Mask` objects are not necessarily the same (as is enforced by global state in `lsst.afw.image.Mask`); instead, a separate `MaskSchema` object is used to manage shared mask plane definitions.

`lsst.afw.image.MaskedImage` corresponds directly to `MaskedImage`, but the latter can also hold a `SkyProjection`, flexible metadata, and units.

**Conversions**

From `lsst.images` to ``lsst.afw.image``:

- `Image.to_legacy`
- `Mask.to_legacy`
- `MaskedImage.to_legacy`

From ``lsst.afw.image`` to `lsst.images`:

- `Image.from_legacy`
- `Mask.from_legacy`
- `MaskedImage.from_legacy`

Single-Visit `lsst.afw.image.Exposure` Objects
----------------------------------------------

When used to represent a calibrated single-visit, single-detector image, `lsst.afw.image.Exposure` corresponds to `VisitImage`, which is a subclass of `MaskedImage`.

Most `lsst.afw.image.Exposure` components have `VisitImage` counterparts:

- ``wcs`` (`lsst.afw.geom.SkyWcs`) -> `VisitImage.sky_projection` (`SkyProjection`)
- ``psf`` (`lsst.afw.detection.Psf`) -> `VisitImage.psf` (`psfs.PointSpreadFunction`)
- ``validPolygon`` (`lsst.afw.geom.Polygon`) -> `VisitImage.bounds` (any `Bounds` implementation)
- ``visitInfo`` (`lsst.afw.image.VisitInfo`) -> `VisitImage.obs_info` (`astro_metadata_translator.ObservationInfo`)
- ``summaryStats`` (`lsst.afw.image.ExposureSummaryStats`) -> `VisitImage.summary_stats` (`ObservationSummaryStats`)
- ``detector`` (`lsst.afw.cameraGeom.Detector`) -> `VisitImage.detector` (`cameras.Detector`)
- ``apCorrMap`` (`lsst.afw.image.ApCorrMap`) -> `VisitImage.aperture_corrections` (`dict` of `fields.BaseField`)
- ``photoCalib`` (`lsst.afw.image.PhotoCalib`) -> `VisitImage.photometric_scaling` (`fields.BaseField`)

`VisitImage` can also hold one or more background models (`VisitImage.backgrounds`), which have to be saved separately from `lsst.afw.image.Exposure`.

**Conversions**

- `VisitImage.to_legacy`
- `VisitImage.from_legacy`

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

- ``wcs`` (`lsst.afw.geom.SkyWcs`) -> `~cells.CellCoadd.sky_projection` (`SkyProjection`)
- ``psf`` (`lsst.afw.detection.Psf`) -> `~cells.CellCoadd.psf` (`~cells.CellPointSpreadFunction`)
- ``apCorrMap`` (`lsst.afw.image.ApCorrMap`) -> `~cells.CellCoadd.aperture_corrections` (`dict` of `~cells.CellField`)

Cell coadds also have a `~cells.CellCoadd.bounds` attribute (`~cells.CellGridBounds`) that plays a similar role to the `VisitImage.bounds` attribute in that it represents the area where pixels and other objects (e.g. PSFs) are valid, by recording which cells do and do not have data.

Cell coadds can also store one or more background models (`~cells.CellCoadd.backgrounds`), which have to be saved separately from `lsst.afw.image.Exposure`.

**Conversions**

- `cells.CellCoadd.to_legacy` (to `lsst.afw.image.Exposure`)
- `cells.CellCoadd.to_legacy_cell_coadd` (to `lsst.cell_coadds.MultipleCellCoadd`)

- `cells.CellCoadd.from_legacy` (from `lsst.cell_coadds.MultipleCellCoadd`)

BoundedFields and Backgrounds
-----------------------------

`lsst.afw.math.BoundedField` (used directly for aperture corrections and indirectly by `lsst.afw.image.PhotoCalib`) corresponds directly to the `fields.BaseField` base class, whose subclasses are closed to the ``fields.Field`` type-union (i.e. no external implementations are permitted; this greatly simplifies serialization).
All `fields.BaseField` objects can be associated with units (via `astropy.units`).

The `lsst.afw.math.BackgroundMI` and `lsst.afw.math.BackgroundList` types are *also* mapped to the `fields.BaseField` hierarchy in `lsst.images`, since those are also essentially just calculated images.

Implementations include:

- `lsst.afw.math.ChebyshevBoundedField` -> `fields.ChebyshevField`
- `lsst.afw.math.ProductBoundedField` -> `fields.ProductField`
- `lsst.afw.math.BackgroundMI` (interpolate) -> `fields.SplineField`
- `lsst.afw.math.BackgroundMI` (approximate) -> `fields.ChebyshevField`
- `lsst.afw.math.BackgroundList` -> `fields.SumField`.
- `lsst.cell_coadds.StitchedApertureCorrection` -> `cells.CellField`

The last of these has some caveats:

- `~lsst.cell_coadds.StitchedApertureCorrection` is not a true `~lsst.afw.math.BoundedField` (it acts like one just enough to be used to apply aperture corrections);
- `cells.CellField` is a true `fields.BaseField`, but it is not directly serializable at present (instead, a `dict` with `~cells.CellField` values is serialized all at once), and is hence not a member of the ``fields.Field`` type-union.

The `fields.field_from_legacy` and `fields.field_from_legacy_background` free functions can be used to convert from `lsst.afw` when the exact type is unknown.

**Conversions**

- `fields.BaseField.to_legacy` (to `lsst.afw.math.BoundedField`)
- `cells.CellField.to_legacy_aperture_correction` (to `lsst.cell_coadds.StitchedApertureCorrection`)
- `fields.BaseField.to_legacy_photo_calib` (to `lsst.afw.image.PhotoCalib`)
- `fields.ChebyshevField.to_legacy_function2` (to `lsst.afw.math.Chebyshev1Function2`)

- `fields.field_from_legacy` (from `lsst.afw.math.BoundedField`)
- `cells.CellField.from_legacy_aperture_correction` (from `lsst.cell_coadds.StitchedApertureCorrection`)
- `fields.field_from_legacy_photo_calib` (from `lsst.afw.image.PhotoCalib`)
- `fields.field_from_legacy_background` (from `lsst.afw.math.BackgroundMI` or lsst.afw.math.BackgroundList`)
- `fields.ChebyshevField.from_legacy_function2` (from `lsst.afw.math.Chebyshev1Function2`)

PSF Models
----------

The `lsst.afw.detection.Psf` class corresponds directly to `psfs.PointSpreadFunction`.
There is no concept of "average position" in `~psfs.PointSpreadFunction`, so a position must always be used to evaluate the PSF model.
The `~psfs.PointSpreadFunction` interface also currently lacks a way to represent wavelength-dependent PSF models, as we do not want to rush the in-code definition of the independent spectral-dimension variable.

Concrete PSF implementations include:

- `lsst.meas.extensions.piff.PiffPsf` -> `psfs.PiffWrapper`
- `lsst.meas.extensions.psfex.PsfExPsf` -> `psfs.PSFExWrapper` (this actually just wraps `lsst.meas.extensions.psfex.PsfExPsf` and cannot be used when that cannot be imported)
- `lsst.afw.detection.SingleGaussianPsf` -> `psfs.GaussianPointSpreadFunction`
- `lsst.cell_coadds.StitchedPsf` -> `cells.CellPointSpreadFunction`

**Conversions**

- `psfs.PiffWrapper.to_legacy`
- `psfs.LegacyPointSpreadFunction.to_legacy` (inherited by `psfs.PSFExWrapper`)
- `cells.CellPointSpreadFunction.to_legacy`

- `psfs.LegacyPointSpreadFunction.from_legacy` (inherited by `psfs.PSFExWrapper`)
- `psfs.PiffWrapper.from_legacy`
- `cells.CellPointSpreadFunction.from_legacy`

Camera Geometry
---------------

The `lsst.afw.cameraGeom.Detector` class corresponds directly to `cameras.Detector`, but the latter is mutable and hence has no need for the various builders of the former.

The `lsst.afw.cameraGeom.Amplifier` class corresponds directly to `cameras.Amplifier`, but the latter factors out into optional sub-objects the various section bounding boxes that are modified during assembly and the suite of electronic parameters are superseded (at least for Rubin Observatory data) by calibration datasets.

**Conversions**

- `cameras.DetectorType.to_legacy`
- `cameras.ReadoutCorner.to_legacy`
- `cameras.Orientation.to_legacy`
- `cameras.Amplifier.to_legacy_builder`
- `cameras.Detector.to_legacy`

- `cameras.DetectorType.from_legacy`
- `cameras.Orientation.from_legacy`
- `cameras.ReadoutCorner.from_legacy`
- `cameras.Amplifier.from_legacy`
- `cameras.AmplifierRawGeometry.from_legacy_amplifier`
- `cameras.AmplifierCalibrations.from_legacy_amplifier`
- `cameras.Detector.from_legacy`
