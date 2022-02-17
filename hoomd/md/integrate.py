# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement MD Integrator."""

import itertools

import hoomd
from hoomd.md import _md
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyTypes
from hoomd.integrate import BaseIntegrator
from hoomd.data import syncedlist
from hoomd.md.methods import Method
from hoomd.md.force import Force
from hoomd.md.constrain import Constraint, Rigid


def _set_synced_list(old_list, new_list):
    old_list.clear()
    old_list.extend(new_list)


class _DynamicIntegrator(BaseIntegrator):

    def __init__(self, forces, constraints, methods, rigid):
        forces = [] if forces is None else forces
        constraints = [] if constraints is None else constraints
        methods = [] if methods is None else methods
        self._forces = syncedlist.SyncedList(
            Force, syncedlist._PartialGetAttr('_cpp_obj'), iterable=forces)

        self._constraints = syncedlist.SyncedList(
            OnlyTypes(Constraint, disallow_types=(Rigid,)),
            syncedlist._PartialGetAttr('_cpp_obj'),
            iterable=constraints)

        self._methods = syncedlist.SyncedList(
            Method, syncedlist._PartialGetAttr('_cpp_obj'), iterable=methods)

        param_dict = ParameterDict(rigid=OnlyTypes(Rigid, allow_none=True))
        if rigid is not None and rigid._added:
            raise ValueError("Rigid object can only belong to one integrator.")
        param_dict["rigid"] = rigid
        self._param_dict.update(param_dict)

    def _attach(self):
        self._forces._sync(self._simulation, self._cpp_obj.forces)
        self._constraints._sync(self._simulation, self._cpp_obj.constraints)
        self._methods._sync(self._simulation, self._cpp_obj.methods)
        if self.rigid is not None:
            self.rigid._attach()
        super()._attach()

    def _detach(self):
        self._forces._unsync()
        self._methods._unsync()
        self._constraints._unsync()
        if self.rigid is not None:
            self.rigid._detach()
        super()._detach()

    def _remove(self):
        if self.rigid is not None:
            self.rigid._remove()
        super()._remove()

    def _add(self, simulation):
        super()._add(simulation)
        if self.rigid is not None:
            self.rigid._add(simulation)

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

    def _setattr_param(self, attr, value):
        if attr == "rigid":
            self._set_rigid(value)
            return
        super()._setattr_param(attr, value)

    def _set_rigid(self, new_rigid):
        """Handles the adding and detaching of potential Rigid objects."""
        # this generally only happens when attaching and we can ignore it since
        # we attach the rigid body in _attach.
        if new_rigid is self.rigid:
            return

        old_rigid = self.rigid

        if new_rigid is not None and new_rigid._added:
            raise ValueError("Cannot add Rigid object to multiple integrators.")

        if old_rigid is not None:
            if self._attached:
                old_rigid._detach()
            if self._added:
                old_rigid._remove()

        if new_rigid is None:
            self._param_dict["rigid"] = None
            return

        if self._added:
            new_rigid._add(self._simulation)
        if self._attached:
            new_rigid._attach()
        self._param_dict["rigid"] = new_rigid


class Integrator(_DynamicIntegrator):
    """Enables a variety of standard integration methods.

    Args:
        dt (float): Integrator time step size :math:`[\\mathrm{time}]`.

        methods (Sequence[hoomd.md.methods.Method]): Sequence of integration
            methods. Each integration method can be applied to only a specific
            subset of particles. The intersection of the subsets must be null.
            The default value of ``None`` initializes an empty list.

        forces (Sequence[hoomd.md.force.Force]): Sequence of forces applied to
            the particles in the system. All the forces are summed together.
            The default value of ``None`` initializes an empty list.

        integrate_rotational_dof (bool): When True, integrate rotational degrees
            of freedom.

        constraints (Sequence[hoomd.md.constrain.Constraint]): Sequence of
            constraint forces applied to the particles in the system.
            The default value of ``None`` initializes an empty list. Rigid body
            objects (i.e. `hoomd.md.constrain.Rigid`) are not allowed in the
            list.

        rigid (hoomd.md.constrain.Rigid): A rigid bodies object defining the
            rigid bodies in the simulation.


    Classes of the following modules can be used as elements in `methods`:

    - `hoomd.md.methods`
    - `hoomd.md.methods.rattle`

    The classes of following modules can be used as elements in `forces`

    - `hoomd.md.angle`
    - `hoomd.md.bond`
    - `hoomd.md.long_range.pppm`
    - `hoomd.md.dihedral`
    - `hoomd.md.external.field`
    - `hoomd.md.force`
    - `hoomd.md.improper`
    - `hoomd.md.pair`
    - `hoomd.md.external.wall`
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
        dt (float): Integrator time step size :math:`[\\mathrm{time}]`.

        methods (list[hoomd.md.methods.Method]): List of integration methods.
            Each integration method can be applied to only a specific subset of
            particles.

        forces (list[hoomd.md.force.Force]): List of forces applied to
            the particles in the system. All the forces are summed together.

        integrate_rotational_dof (bool): When True, integrate rotational degrees
            of freedom.

        constraints (list[hoomd.md.constrain.Constraint]): List of
            constraint forces applied to the particles in the system.

        rigid (hoomd.md.constrain.Rigid): The rigid body definition for the
            simulation associated with the integrator.
    """

    def __init__(self,
                 dt,
                 integrate_rotational_dof=False,
                 forces=None,
                 constraints=None,
                 methods=None,
                 rigid=None):

        super().__init__(forces, constraints, methods, rigid)

        self._param_dict.update(
            ParameterDict(
                dt=float(dt),
                integrate_rotational_dof=bool(integrate_rotational_dof)))

    def _attach(self):
        # initialize the reflected c++ class
        self._cpp_obj = _md.IntegratorTwoStep(
            self._simulation.state._cpp_sys_def, self.dt)
        # Call attach from DynamicIntegrator which attaches forces,
        # constraint_forces, and methods, and calls super()._attach() itself.
        super()._attach()

    def __setattr__(self, attr, value):
        """Hande group DOF update when setting integrate_rotational_dof."""
        super().__setattr__(attr, value)
        if (attr == 'integrate_rotational_dof' and self._simulation is not None
                and self._simulation.state is not None):
            self._simulation.state.update_group_dof()

    @hoomd.logging.log(requires_run=True)
    def linear_momentum(self):
        """tuple(float,float,float): The linear momentum vector of the system \
            :math:`[\\mathrm{mass} \\cdot \\mathrm{velocity}]`.

        .. math::

            \\vec{p} = \\sum_{i=0}^\\mathrm{N_particles} m_i \\vec{v}_i
        """
        v = self._cpp_obj.computeLinearMomentum()
        return (v.x, v.y, v.z)
