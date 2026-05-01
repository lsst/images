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
Starlink NDF data model.

Files written by this archive are valid NDF files readable by Starlink tools
(KAPPA, ``hdstrace``, etc.). The HDS-on-HDF5 format is described in
arxiv:1502.04029; the NDF data model in arxiv:1410.7513.
"""

from ._common import *
from ._input_archive import *
from ._output_archive import *
