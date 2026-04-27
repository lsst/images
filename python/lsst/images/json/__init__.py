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


"""Archive implementations for simple JSON files.

The archives in this package write to and read from JSON by embedding all
array and table data into the JSON tree as inline arrays.  While this
technically allows them to support arbitrary archive-serializable object, it
can be extremely inefficient for large arrays and tables.

The outermost object in the stored form is just the
`.serialization.ArchiveTree` that corresponds to the top-level in-memory
object being saved.
"""

from ._input_archive import *
from ._output_archive import *
