# coding: utf-8

# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

# Maintainer: joaander / All Developers are free to add commands for new
# features


from hoomd.md import _md
from hoomd.parameterdicts import ParameterDict
from hoomd.typeconverter import OnlyFrom
from hoomd.integrate import _BaseIntegrator
from hoomd.syncedlist import SyncedList
from hoomd.md.methods import _Method
from hoomd.md.force import _Force
from hoomd.md.constrain import _ConstraintForce


def preprocess_aniso(value):  # noqa: D103
    if value is True:
        return "true"
    elif value is False:
        return "false"
    else:
        return value


def set_synced_list(old_list, new_list):  # noqa: D103
    old_list.clear()
    old_list.extend(new_list)


class _DynamicIntegrator(_BaseIntegrator):
    def __init__(self, forces, constraints, methods):
        forces = [] if forces is None else forces
        constraints = [] if constraints is None else constraints
        methods = [] if methods is None else methods
        self._forces = SyncedList(lambda x: isinstance(x, _Force),
                                  to_synced_list=lambda x: x._cpp_obj,
                                  iterable=forces)

        self._constraints = SyncedList(lambda x: isinstance(x,
                                                            _ConstraintForce),
                                       to_synced_list=lambda x: x._cpp_obj,
                                       iterable=constraints)

        self._methods = SyncedList(lambda x: isinstance(x, _Method),
                                   to_synced_list=lambda x: x._cpp_obj,
                                   iterable=methods)

    def _attach(self):
        self.forces.attach(self._simulation, self._cpp_obj.forces)
        self.constraints.attach(self._simulation, self._cpp_obj.constraints)
        self.methods.attach(self._simulation, self._cpp_obj.methods)
        super()._attach()

    @property
    def forces(self):
        return self._forces

    @forces.setter
    def forces(self, value):
        set_synced_list(self._forces, value)

    @property
    def constraints(self):
        return self._constraints

    @constraints.setter
    def constraints(self, value):
        set_synced_list(self._constraints, value)

    @property
    def methods(self):
        return self._methods

    @methods.setter
    def methods(self, value):
        set_synced_list(self._methods, value)


class Integrator(_DynamicIntegrator):
    R""" Enables a variety of standard integration methods.

    Args:
        dt (float): Integrator time step size (in time units).

        methods (Sequence[hoomd.md.methods._Method]): Sequence of integration
            methods. Each integration method can be applied to only a specific
            subset of particles, default [].

        forces (Sequence[hoomd.md.force._Force]): Sequence of forces applied to 
            the particles in the system. All the forces are summed together, 
            default [].

        aniso (str or bool): Whether to integrate rotational degrees of freedom 
            (bool), default 'auto' (autodetect).

        constraints (Sequence[hoomd.md.constrain._ConstraintForce]): Sequence of
            constraint forces applied to the particles in the system,
            default [].


    The following classes can be used as elements in `methods`

    - `hoomd.md.methods.Brownian`
    - `hoomd.md.methods.Langevin`
    - `hoomd.md.methods.NVE`
    - `hoomd.md.methods.NVT`
    - `hoomd.md.methods.npt`
    - `hoomd.md.methods.nph`

    The classes of following modules can be used as elements in `forces`

    - `hoomd.md.angle`
    - `hoomd.md.bond`
    - `hoomd.md.charge`
    - `hoomd.md.dihedral`
    - `hoomd.md.external`
    - `hoomd.md.force`
    - `hoomd.md.improper`
    - `hoomd.md.pair`
    - `hoomd.md.wall`
    - `hoomd.md.special_pair`
    - `hoomd.dem.pair`

    The classes of the following module can be used as elements in `constraints`

    - `hoomd.md.constrain`

    Examples::

        integrator = hoomd.md.Integrator(dt=0.001, methods=[nve], forces=[lj])
        sim.operations.integrator = integrator


    Attributes:
        dt (float): Integrator time step size (in time units).

        methods (List[hoomd.md.methods._Method]): List of integration methods.
            Each integration method can be applied to only a specific subset of
            particles.

        forces (List[hoomd.md.force._Force]): List of forces applied to
            the particles in the system. All the forces are summed together.

        aniso (str): Whether rotational degrees of freedom are integrated.

        constraints (List[hoomd.md.constrain._ConstraintForce]): List of
            constraint forces applied to the particles in the system.
    """

    def __init__(self, dt, aniso='auto', forces=[], constraints=[],
                 methods=[]):

        super().__init__(forces, constraints, methods)

        self._param_dict = ParameterDict(
            dt=float(dt),
            aniso=OnlyFrom(['true', 'false', 'auto'],
                           preprocess=preprocess_aniso),
            _defaults=dict(aniso="auto")
        )
        if aniso is not None:
            self.aniso = aniso

    def attach(self, simulation):  # noqa: D102
        # initialize the reflected c++ class
        self._cpp_obj = _md.IntegratorTwoStep(simulation.state._cpp_sys_def,
                                              self.dt)
        # Call attach from DynamicIntegrator which attaches forces,
        # constraint_forces, and methods, and calls super().attach() itself.
        super().attach(simulation)
