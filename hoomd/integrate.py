# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement BaseIntegrator."""

from hoomd.operation import Operation


class BaseIntegrator(Operation):
    """Defines the base for all HOOMD-blue integrators.

    An integrator in HOOMD-blue is the primary operation that drives a
    simulation state forward. In `hoomd.hpmc`, integrators perform particle
    based Monte Carlo moves. In `hoomd.md`, the `hoomd.md.Integrator` class
    organizes the forces, equations of motion, and other factors of the given
    simulation.
    """

    def _attach(self):
        self._simulation._cpp_sys.setIntegrator(self._cpp_obj)
        super()._attach()

        # The integrator has changed, update the number of DOF in all groups
        self._simulation.state.update_group_dof()
