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

__all__ = (
    "ComponentQuality",
    "ComponentQualityState",
    "CompositeQuality",
    "FlaggedComponentError",
)

import enum
from collections.abc import Set
from types import EllipsisType

import pydantic

from lsst.pipe.base import ExceptionInfo  # TODO[DM-56192]: update dependencies accordingly


class ComponentQualityState(enum.IntEnum):
    """States that a data product component might be in.

    Notes
    -----
    Quality state is defined relative to a particular data product; if that
    data product can never have a better version of a particular component,
    that's `COMPLETE`, even if that whole data product is in some sense
    "preliminary" compared to a similar one that appears later in a pipeline.
    """

    BOOTSTRAP_GUESS = 0
    """This component was not actually fit to data; it's an initial guess
    derived from some combination of configuration and metadata.
    """

    PRELIMINARY = 50
    """This component was derived from data, but the full procedure for
    fitting it did not complete and this preliminary result is the best
    available.
    """

    COMPLETE = 100
    """This component was fit successfully to data using the full procedure
    for this data product.
    """


class ComponentQuality(pydantic.BaseModel):
    state: ComponentQualityState = ComponentQualityState.COMPLETE
    failed_cuts: set[str] = pydantic.Field(default_factory=set)


class CompositeQuality(pydantic.BaseModel):
    components: dict[str, ComponentQuality] = pydantic.Field(default_factory=dict)
    """Quality information that affects the usability of components.

    The set of component names is defined by the type this quality object is
    attached.
    """

    exceptions: list[ExceptionInfo] = pydantic.Field(default_factory=list)
    """Exceptions raised during the processing that produced this object.
    """

    def check_access(
        self,
        component: str,
        *,
        min_state: int = ComponentQualityState.COMPLETE,
        failed_cuts_allowed: Set[str] | EllipsisType = frozenset(),
    ) -> list[str]:
        """Check whether the given component can be accessed given its quality
        and the user's quality allowances.

        Parameters
        ----------
        component
            Name of the component.
        min_state
            Minimum state for the component, inclusive.  See
            `ComponentQualityState` for bounds.
        failed_cuts_allowed
            The names of quality cuts this component may have failed while
            still being considered good enough for this request.  Pass ``...``
            to accept all failed cuts.

        Returns
        -------
        `list` [`str`]
            A list of string messages indicating unacceptable problems with
            the component.  Empty if access is allowed.
        """
        result: list[str] = []
        if (component_quality := self.components.get(component)) is None:
            return result
        if component_quality.state < min_state:
            result.append(
                f"Quality state for component {component!r} is {component_quality.state}, "
                f"but minimum is {min_state}."
            )
        if failed_cuts_allowed is not ...:
            for disallowed_failed_cut in sorted(component_quality.failed_cuts - failed_cuts_allowed):
                result.append(f"Component {component!r} failed quality cut {disallowed_failed_cut!r}.")
        return result


class FlaggedComponentError(RuntimeError):
    """An exception raised when a component with quality flags is accessed
    without an explicit acceptance of those flags.
    """
