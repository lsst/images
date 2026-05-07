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


"""Archive implementations that write HDS-on-HDF5 files compatible with the
Starlink N-Dimensional Data Format (NDF) data model.

Files written by this archive are valid NDF files readable by
applications and libraries from the
`Starlink Software Collection <https://starlink.eao.hawaii.edu>`_
(KAPPA, ``hdstrace``, CUPID, etc.) although the LSST data model adds
extensions that are not known to the Starlink tooling.

The NDF data model is described in
`Jenness et al (2015) <https://doi.org/10.1016/j.ascom.2014.11.001>`_.

For a `lsst.images.MaskedImage` the data and variance arrays map to the
standard NDF equivalents.
Any FITS headers are stored in the standard ``.MORE.FITS`` extension and
the WCS is stored as a serialized AST FrameSet, modified to include a ``GRID``
frame in addition to a ``PIXEL`` frame, in the ``.WCS`` component.
Starlink tools only allow 8-bit masks so the mask is stored directly as a
``QUALITY`` component when only 8 bits are needed but if more bits are
required the full mask is stored in the ``.MORE.LSST.MASK`` extension as
a 3-D unsigned byte NDF where the third dimension matches the internal
representation of the 3-D mask. The ``QUALITY`` component includes a collapsed
version of the full mask to enable Starlink software to apply a coarse mask.
In the future the collapsing is intending to be more granular to allow mask
planes to be combined based on their related concepts.
Since the NDF data model has a subset that is compatible with the
`~lsst.images.MaskedImage` data model, an NDF can be read as a
`~lsst.images.MaskedImage` with full data, variance, mask, WCS, and FITS
header support, although of course any extensions will be ignored.

For more complex image types such as `lsst.images.VisitImage` the LSST
components are stored in the ``.MORE.LSST`` extension, including the ``JSON``
component containing the Pydantic models. Data arrays are represented as
NDFs in the ``.MORE.LSST`` extension and referenced by path from the JSON.
Tabular data is not supported by the NDF data model and is not currently
representable. It is expected that they would become individual columns
in an extension, all with the same length.

These files use HDF5 format and can be read by any HDF5 tooling such as
``h5dump`` or the Python ``h5py``, although the data model includes
group attributes to allow it to be read by the Starlink libraries
that add additional semantic meaning to data structures.
The HDS-on-HDF5 format is described in
`Jenness (2015) <https://doi.org/10.1016/j.ascom.2015.02.003>`_.

HDF5 files are not generally usable for remote access of components through
cloud storage. There is no facility to store byte offsets to components in
any of the data structures.
"""

from ._common import *
from ._input_archive import *
from ._output_archive import *
