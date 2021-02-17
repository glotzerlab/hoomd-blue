# coding: utf-8

# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

# Maintainer: joaander / All Developers are free to add commands for new
# features

import itertools
from functools import partial

from hoomd.md import _md
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyFrom
from hoomd.integrate import BaseIntegrator
from hoomd.data import syncedlist
from hoomd.md.methods import _Method
from hoomd.md.force import Force
from hoomd.md.constrain import ConstraintForce


def _preprocess_aniso(value):
    if value is True:
        return "true"
    elif value is False:
        return "false"
    else:
        return value


def _set_synced_list(old_list, new_list):
    old_list.clear()
    old_list.extend(new_list)


class _DynamicIntegrator(BaseIntegrator):
    def __init__(self, forces, constraints, methods):
        forces = [] if forces is None else forces
        constraints = [] if constraints is None else constraints
        methods = [] if methods is None else methods
        self._forces = syncedlist.SyncedList(
            Force, syncedlist._PartialGetAttr('_cpp_obj'), iterable=forces)

        self._constraints = syncedlist.SyncedList(
            ConstraintForce,
            syncedlist._PartialGetAttr('_cpp_obj'),
            iterable=constraints)

        self._methods = syncedlist.SyncedList(
            _Method, syncedlist._PartialGetAttr('_cpp_obj'), iterable=methods)

    def _attach(self):
        self.forces._sync(self._simulation, self._cpp_obj.forces)
        self.constraints._sync(self._simulation, self._cpp_obj.constraints)
        self.methods._sync(self._simulation, self._cpp_obj.methods)
        super()._attach()

    @property
    def forces(self):
        return self._forces

    @forces.setter
    def forces(self, value):
        _set_synced_list(self._forces, value)

    @property
    def constraints(self):
        return self._constraints

    @constraints.setter
    def constraints(self, value):
        _set_synced_list(self._constraints, value)

    @property
    def methods(self):
        return self._methods

    @methods.setter
    def methods(self, value):
        _set_synced_list(self._methods, value)

    @property
    def _children(self):
        children = list(self.forces)
        children.extend(self.constraints)
        children.extend(self.methods)

        for child in itertools.chain(self.forces, self.constraints,
                                     self.methods):
            children.extend(child._children)

        return children


class Integrator(_DynamicIntegrator):
    R""" Enables a variety of standard integration methods.

    Args:
        dt (float): Integrator time step size (in time units).

        methods (Sequence[hoomd.md.methods._Method]): Sequence of integration
            methods. Each integration method can be applied to only a specific
            subset of particles. The intersection of the subsets must be null.
            The default value of ``None`` initializes an empty list.

        forces (Sequence[hoomd.md.force.Force]): Sequence of forces applied to
            the particles in the system. All the forces are summed together.
            The default value of ``None`` initializes an empty list.

        aniso (str or bool): Whether to integrate rotational degrees of freedom
            (bool), default 'auto' (autodetect if there is anisotropic factor
            from any defined active or constraint forces).

        constraints (Sequence[hoomd.md.constrain.ConstraintForce]): Sequence of
            constraint forces applied to the particles in the system.
            The default value of ``None`` initializes an empty list.


    The following classes can be used as elements in `methods`

    - `hoomd.md.methods.Brownian`
    - `hoomd.md.methods.Langevin`
    - `hoomd.md.methods.NVE`
    - `hoomd.md.methods.NVT`
    - `hoomd.md.methods.NPT`

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

    The classes of the following module can be used as elements in `constraints`

    - `hoomd.md.constrain`

    Examples::

        nlist = hoomd.md.nlist.Cell()
        lj = hoomd.md.pair.LJ(nlist=nlist)
        lj.params.default = dict(epsilon=1.0, sigma=1.0)
        lj.r_cut[('A', 'A')] = 2**(1/6)
        nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
        integrator = hoomd.md.Integrator(dt=0.001, methods=[nve], forces=[lj])
        sim.operations.integrator = integrator


    Attributes:
        dt (float): Integrator time step size (in time units).

        methods (List[hoomd.md.methods._Method]): List of integration methods.
            Each integration method can be applied to only a specific subset of
            particles.

        forces (List[hoomd.md.force.Force]): List of forces applied to
            the particles in the system. All the forces are summed together.

        aniso (str): Whether rotational degrees of freedom are integrated.

        constraints (List[hoomd.md.constrain.ConstraintForce]): List of
            constraint forces applied to the particles in the system.
    """

    def __init__(self, dt, aniso='auto', forces=None, constraints=None,
                 methods=None):

        super().__init__(forces, constraints, methods)

        self._param_dict = ParameterDict(
            dt=float(dt),
            aniso=OnlyFrom(['true', 'false', 'auto'],
                           preprocess=_preprocess_aniso),
            _defaults=dict(aniso="auto")
            )
        if aniso is not None:
            self.aniso = aniso

    def _attach(self):
        # initialize the reflected c++ class
        self._cpp_obj = _md.IntegratorTwoStep(
            self._simulation.state._cpp_sys_def, self.dt)
        # Call attach from DynamicIntegrator which attaches forces,
        # constraint_forces, and methods, and calls super()._attach() itself.
        super()._attach()
