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
"""Entry point for ``python -m lsst.images``.

The same command is also installed as the ``lsst-images-admin`` console
script (see ``[project.scripts]`` in ``pyproject.toml``).
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    main()
