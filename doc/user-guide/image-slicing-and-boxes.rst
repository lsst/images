.. py:currentmodule:: lsst.images

.. _image-slicing-and-boxes:

Image Slicing and Boxes
=======================

Absolute and Local coordinates
------------------------------

All `GeneralizedImage` instances in `lsst.images` (`Image`, `Mask`, `MaskedImage`, etc.) have a `bounding box <GeneralizedImage.bbox>` whose minimum coordinate values are not necessarily zero, allowing the image to be located in some larger pixel grid.
This is most often used to allow a subimage view to use the same pixel coordinates as its parent.
Since LSST coadds are built on large *tract* coordinate grids that are subdivided into smaller *patches*, most patch-level images will have a bounding box that does not start at ``(0, 0)``, representing full-tract coordinates instead of single-patch coordinates.
This nonzero origin is often referred to as ``xy0`` or ``yx0`` (depending on how the coordinates are ordered), and the `GeneralizedImage.yx0` property is available as an alias for the image's bounding box's `Box.min`.

We call the pixel coordinate system that starts at `~GeneralizedImage.yx0` *absolute* coordinates, and the coordinate system that starts at ``(0, 0)`` the image's *local* coordinates.

The big advantage of absolute coordinates is that it allows all of the objects typically attached to an image - `SkyProjection`, `psfs.PointSpreadFunction`, etc. - to continue to work without modification on a subimage, because they operate in absolute coordinates as well.
If these objects operated in local coordinates, they'd all have to be sliced whenever the image they are attached to is sliced.

The main disadvantage of absolute coordinates is that the `numpy.ndarray` objects that actually hold pixel values have no knowledge of `~GeneralizedImage.yx0`, and hence any slicing of underlying arrays (`Image.array`) has to operate in local coordinates.

Absolute-coordinate subimages
-----------------------------

Subimages are usually obtained by slicing a `GeneralizedImage` with a `Box` defined in absolute coordinates::

   subimage = parent[box]

While there is no constraint on user-created `Box` instances, all `Box` instances created or returned by `lsst.images` interfaces are in absolute coordinates.

When a `Box` isn't already available, the `GeneralizedImage.absolute` property provides a proxy object that can be sliced directly::

   subimage = parent.absolute[y_a:y_b, x_a:x_b]

Because `~GeneralizedImage.yx0` can go negative (and hence absolute coordinates can go negative, too), negative indices are just regular pixel coordinates (not a way to index backwards from the end, as in `list` or `numpy.ndarray` slicing).
Leaving out an endpoint can still be used to refer to the image's bounding box, however::

   subimage = parent.absolute[y_a:, :x_b]

is equivalent to::

   subimage = parent.absolute[y_a:parent.bbox.y.stop, parent.bbox.x.start:x_b]

Note that `~GeneralizedImage.absolute` is not a method or a real object you can use on its own; it's a special type that allows Python's slicing syntax to be used on its parent (similar to how ``DataFrame.loc`` works in `Pandas <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html>`__).

Absolute indexing that reduces the dimensionality of the result (e.g. ``parent.absolute[y, :]``) does not work; there are no 1-d or 0-d image types that could be returned.

Local-coordinate subimages
--------------------------

When slices are more naturally expressed without the `~GeneralizedImage.yx0` offset, the `GeneralizedImage.local` slicing proxy can be used::

   subimage = parent.local[i_a:i_b, j_a:j_b]

Negative indexes operate just as they do in `list` or `numpy.ndarray` indexing here (since it's impossible to have a "real" negative local pixel coordinate); this cuts off a 3-pixel border from around the image::

   subimage = parent.local[3:-3, 3:-3]

Like `GeneralizedImage.absolute`, `~GeneralizedImage.local` is not a method or a real object you can use on its own.

Local indexing that reduces dimensionality (e.g. ``parent.local[i, :]``) also doesn't work, but of course a slice that reduces the dimensionality of an array view (which always uses local coordinates) is completely fine::

   row_array = parent.array[i, :]

This only works on some `GeneralizedImage` types like `Image` and `Mask`, however; there is no single array that backs a `MaskedImage`, for example (instead you'd have to separately slice its `~MaskedImage.image`, `~MaskedImage.mask`, and `~MaskedImage.variance` arrays).

Using slices to create Intervals and Boxes
------------------------------------------

The `Box` and `Interval` classes should usually be *constructed* using slice syntax::

   new_interval = Interval.factory[a:b]
   new_box = Box.factory[y_a:y_b, x_a:x_b]

Because they are intended for creating absolute-coordinate intervals and boxes, these treat negative indexes as regular coordinate values, and missing upper bounds are not allowed (missing lower bounds are defaulted to zero).

Once you have a `Box` or `Interval`, you can also use their `~Box.absolute` and `~Box.local` slicing proxies (or, for `Interval`: `~Interval.absolute`, `~Interval.local`) to produce related boxes and intervals::

   absolute_subset_interval = parent_interval[a:]
   local_subset_bbox = parent_box[y_a:-2, :x_b]

As shown, empty bounds work naturally for these, and negative indexes operate in reverse from the end for local slicing only.

When a `GeneralizedImage` is available, slicing it directly is generally preferred.
The ability to perform similar operations on boxes alone is extremely useful for working out the appropriate bounding box to use when *reading* a subimage from a larger file.
