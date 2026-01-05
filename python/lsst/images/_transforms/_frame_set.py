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

from __future__ import annotations

__all__ = ("FrameLookupError", "FrameSet")

from abc import ABC, abstractmethod

from . import _frames  # use this import style to facilitate pattern matching
from ._transform import Transform


class FrameLookupError(LookupError):
    """Exception raised when a frame cannot be found in a `FrameSet`."""


class FrameSet(ABC):
    """A container or factory for `Transform` objects that relates frames.

    Notes
    -----
    `FrameSet` supposes ``in`` (``__contains__``) tests on individual `Frame`
    objects to test whether they are known to the frame set, and indexing
    (``__getitem__``) of **pairs** of frames to return a `Transform` that maps
    the first to the second.
    """

    @abstractmethod
    def __contains__(self, frame: _frames.Frame) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def __getitem__[I: _frames.Frame, O: _frames.Frame](self, key: tuple[I, O]) -> Transform[I, O]:
        raise NotImplementedError()

    def get[I: _frames.Frame, O: _frames.Frame](self, in_frame: I, out_frame: O) -> Transform[I, O] | None:
        """Return the `Transform` that maps the two frames, or `None` if at
        least one is not known to the `FrameSet`.
        """
        try:
            return self[in_frame, out_frame]
        except FrameLookupError:
            return None
