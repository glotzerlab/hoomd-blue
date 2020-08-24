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

    def attach(self, simulation):  # noqa: D102
        self.forces.attach(simulation, self._cpp_obj.forces)
        self.constraints.attach(simulation, self._cpp_obj.constraints)
        self.methods.attach(simulation, self._cpp_obj.methods)
        super().attach(simulation)

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
        dt (float): Each time step of the simulation `hoomd.Simulation.run`?
            will advance the real time of the system forward by *dt* (in time units).
        methods (list of `hoomd.md.methods`?): Integration method `hoomd.md.methods`. It performs a time step integration ?line-breaking
            technique to move the system forward. At each time step, all of the specified forces are evaluated and used in moving the system forward to the next step.
            Each integration method can be applied to only a specific group of particles enabling advanced simulation techniques.?how method argument works, multiple component or not
            ,default None ?what if no methods
        forces (list?): Forces that be applied to the particles in the system. `hoomd.md.pair`, `hoomd.md.force`, `hoomd.md.dihedral`, `hoomd.md.angle`, `hoomd.md.bond`
            ,default None ?what if no forces
        aniso (bool): Whether to integrate rotational degrees of freedom (bool),
            default None (autodetect).
        constraints (list?): Constrains a given set of particles to a given surface, to have some relative orientation, or impose some other type of constraint from `hoomd.md.constrain`, default None. 
    

    The following commands can be used to specify the integration methods used
    by ``methods`` argument.

    - `hoomd.md.methods.Brownian`
    - `hoomd.md.methods.Langevin`
    - `hoomd.md.methods.NVE`
    - `hoomd.md.methods.NVT`
    - `hoomd.md.methods.npt`
    - `hoomd.md.methods.nph`

    There can only be one integration method active at a time. 
    ?If there are more than one ``integrate.mode_*`` commands in a hoomd script, only the most
    recent before a given ```hoomd.run``` will take effect.

    Examples::
        nlist = hoomd.md.nlist.Cell()
        lj = hoomd.md.pair.LJ(nlist=nlist)
        lj.params.default = dict(epsilon=1.0, sigma=1.0)
        lj.r_cut[('A', 'A')] = 2**(1/6)
        nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
        integrator = hoomd.md.Integrator(dt=0.001, methods=[nve], forces=[lj])
    """

    def __init__(self, dt, aniso=None, forces=None, constraints=None,
                 methods=None):

        super().__init__(forces, constraints, methods)

        self._param_dict = ParameterDict(
            dt=float(dt),
            aniso=OnlyFrom(['true', 'false', 'auto'],
                           preprocess=preprocess_aniso),
            _defaults=dict(aniso="auto")
        )
        if aniso is not None:
            self.aniso = aniso

    def attach(self, simulation):
        # initialize the reflected c++ class
        self._cpp_obj = _md.IntegratorTwoStep(simulation.state._cpp_sys_def,
                                              self.dt)
        # Call attach from DynamicIntegrator which attaches forces,
        # constraint_forces, and methods, and calls super().attach() itself.
        super().attach(simulation)
