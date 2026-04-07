lsst-images v30.0.6 (2026-04-07)
================================

New Features
------------

- Implemented ``to_legacy`` conversion for ``Transform`` and ``Projection``. (`DM-54551 <https://rubinobs.atlassian.net/browse/DM-54551>`_)
- Added the ``ColorImage`` class and format for RGB images. (`DM-54220 <https://rubinobs.atlassian.net/browse/DM-54220>`_)
- Added a flexible metadata dictionary and optional butler provenance to all top-level serialization models and generalized images. (`DM-54285 <https://rubinobs.atlassian.net/browse/DM-54285>`_)
- Added more slicing support to all generalized images, with ``.local`` and ``.absolute`` proxy properties to make the indexing conventions clearer. (`DM-54292 <https://rubinobs.atlassian.net/browse/DM-54292>`_)
- Added ``GaussianPointSpreadFunction`` PSF class. (`DM-54472 <https://rubinobs.atlassian.net/browse/DM-54472>`_)


Miscellaneous Changes of Minor Interest
---------------------------------------

- Improved the test coverage of the image classes.
  This uncovered some minor bugs that have also been fixed. (`DM-54472 <https://rubinobs.atlassian.net/browse/DM-54472>`_)

lsst-images v30.0.4 (2026-03-02)
================================

First public release of package.

New Features
------------

- Added FITS tile compression support and import-read support for ``lsst.afw.image.MaskedImage``. (`DM-53698 <https://rubinobs.atlassian.net/browse/DM-53698>`_)
- Added support for ``ObservationInfo`` to be attached to images.
  ``VisitImage`` will construct this object from legacy images. (`DM-54279 <https://rubinobs.atlassian.net/browse/DM-54279>`_)
