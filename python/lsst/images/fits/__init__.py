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

"""Archive implementations for the FITS file format.

The archives this package define a FITS-based meta format with the following
layout:

- A no-data primary HDU with the special header cards ``INDXADDR``,
  ``INDXSIZE``, ``JSONADDR``, and ``JSONSIZE``, which provide the offsets to
  and sizes of two special HDUs at the end of the file (see below).  The
  primary header may also hold arbitrary cards exported by the top-level type
  being serialized or propagated as opaque metadata from a previous read.

- Any number of "normal" image, compressed-image, and binary table HDUs.  These
  have unique ``EXTNAME`` values that are the all-caps variants of a JSON
  Pointer (IETF RFC 6901) path in the special JSON HDU (see below), with no
  ``EXTVER`` or ``EXTLEVEL``.

- A special binary table HDU holding JSON data.  This binary table has a single
  variable-length array byte column (i.e. ``TFORM='PB'``) that holds UTF-8 JSON
  data.  There is always at least one row, which holds the JSON representation
  of the top-level object being serialized.  Additional rows may be present to
  hold additional JSON blocks that are logically nested within the main one,
  but have been moved outside it to keep the main block more compact (the main
  JSON block will have pointers back to these).

- A special binary table HDU that acts as an index into all others, by holding
  byte offsets and sizes for all preceding HDUs along with their ``EXTNAME``,
  ``XTENSION``, and ``ZIMAGE`` header values.

When images and tables are saved to a `FitsOutputArchive`, "normal" HDUs are
added to hold their binary data, and a small Pydantic model is returned
with a reference to that HDU for inclusion in the JSON tree.
"""

from ._common import *
from ._input_archive import *
from ._output_archive import *
