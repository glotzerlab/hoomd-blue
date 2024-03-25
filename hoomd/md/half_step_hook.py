# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Perform user defined computations during the half-step of a \
`hoomd.md.Integrator`.

HalfStepHook can be subclassed to define custom operations at the middle of
each integration step. Examples of use cases include evaluating collective
variables or biasing the simulation.
"""

from hoomd.md import _md


class HalfStepHook(_md.HalfStepHook):
    """HalfStepHook base class.

    HalfStepHook provides an interface to perform computations during the
    half-step of a hoomd.md.Integrator.
    """

    def update(self, timestep):
        """Called during the half-step of a `hoomd.md.Integrator`.

        This method should provide the implementation of any computation that
        the user wants to execute at each timestep in the middle of the
        integration routine.
        """
        raise TypeError(
            "Use a hoomd.md.HalfStepHook derived class implementing the "
            "corresponding update method.")
