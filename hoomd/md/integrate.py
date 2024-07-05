# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement MD Integrator."""

import hoomd
from hoomd.md import _md
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyTypes
from hoomd.operation import Integrator as BaseIntegrator
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
        if rigid is not None and rigid._attached:
            raise ValueError("Rigid object can only belong to one integrator.")
        param_dict["rigid"] = rigid
        self._param_dict.update(param_dict)

    def _attach_hook(self):
        self._forces._sync(self._simulation, self._cpp_obj.forces)
        self._constraints._sync(self._simulation, self._cpp_obj.constraints)
        self._methods._sync(self._simulation, self._cpp_obj.methods)
        if self.rigid is not None:
            self.rigid._attach(self._simulation)
        super()._attach_hook()

    def _post_attach_hook(self):
        self.validate_groups()

    def _detach_hook(self):
        self._forces._unsync()
        self._methods._unsync()
        self._constraints._unsync()
        if self.rigid is not None:
            self.rigid._detach()

    def validate_groups(self):
        """Verify groups.

        Groups may change after attaching.
        Users can call `validate_groups` to verify the groups after changing
        them.
        """
        self._cpp_obj.validate_groups()

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

        if new_rigid is not None and new_rigid._attached:
            raise ValueError("Cannot add Rigid object to multiple integrators.")

        if old_rigid is not None:
            if self._attached:
                old_rigid._detach()

        if new_rigid is None:
            self._param_dict["rigid"] = None
            return

        if self._attached:
            new_rigid._attach(self._simulation)
        self._param_dict["rigid"] = new_rigid


@hoomd.logging.modify_namespace(("md", "Integrator"))
class Integrator(_DynamicIntegrator):
    r"""Molecular dynamics time integration.

    Args:
        dt (float): Integrator time step size :math:`[\mathrm{time}]`.

        methods (Sequence[hoomd.md.methods.Method]): Sequence of integration
          methods. The default value of ``None`` initializes an empty list.

        forces (Sequence[hoomd.md.force.Force]): Sequence of forces applied to
          the particles in the system. The default value of ``None`` initializes
          an empty list.

        integrate_rotational_dof (bool): When True, integrate rotational degrees
          of freedom.

        constraints (Sequence[hoomd.md.constrain.Constraint]): Sequence of
          constraint forces applied to the particles in the system.
          The default value of ``None`` initializes an empty list. Rigid body
          objects (i.e. `hoomd.md.constrain.Rigid`) are not allowed in the
          list.

        rigid (hoomd.md.constrain.Rigid): An object defining the rigid bodies in
          the simulation.

        half_step_hook (hoomd.md.HalfStepHook): Enables the user to perform
            arbitrary computations during the half-step of the integration.

    `Integrator` is the top level class that orchestrates the time integration
    step in molecular dynamics simulations. The integration `methods` define
    the equations of motion to integrate under the influence of the given
    `forces` and `constraints`.

    Each method applies the given equations of motion to a subset of particles
    in the simulation state. See the documentation for each method for details
    on what equations of motion it solves. The intersection of the subsets must
    be null.

    `Integrator` computes the net force, torque, energy, and virial on each
    particle as a sum of those applied by `hoomd.md.force.Force` objects in the
    `forces` and `constraints` lists:

    .. math::

        \vec{F}_{\mathrm{net},i} &= \sum_{f \in \mathrm{forces}} \vec{F}_i^f \\
        \vec{\tau}_{\mathrm{net},i} &=
        \sum_{f \in \mathrm{forces}} \vec{\tau}_i^f \\
        U_{\mathrm{net},i} &= \sum_{f \in \mathrm{forces}} U_i^f \\
        W_{\mathrm{net},i} &= \sum_{f \in \mathrm{forces}} W_i^f \\

    `Integrator` also computes the net additional energy and virial

    .. math::

        U_{\mathrm{net},\mathrm{additional}} &= \sum_{f \in \mathrm{forces}}
        U_\mathrm{additional}^f \\
        W_{\mathrm{net},\mathrm{additional}} &= \sum_{f \in \mathrm{forces}}
        W_\mathrm{additional}^f \\

    See `md.force.Force` for definitions of these terms. Constraints are a
    special type of force used to enforce specific constraints on the system
    state, such as distances between particles with
    `hoomd.md.constrain.Distance`. `Integrator` handles rigid bodies as a
    special case, as it only integrates the degrees of freedom of each body's
    center of mass. See `hoomd.md.constrain.Rigid` for details.

    .. rubric:: Degrees of freedom

    `Integrator` always integrates the translational degrees of freedom.
    It *optionally* integrates one or more rotational degrees of freedom
    for a given particle *i* when all the following conditions are met:

    * The intergration method supports rotational degrees of freedom.
    * `integrate_rotational_dof` is ``True``.
    * The moment of inertia is non-zero :math:`I^d_i > 0`.

    Each particle may have zero, one, two, or three rotational degrees of
    freedom.

    Note:
        By default, `integrate_rotational_dof` is ``False``. `gsd` and
        `hoomd.Snapshot` also set particle moments of inertia to 0 by default.

    .. rubric:: Classes

    Classes of the following modules can be used as elements in `methods`:

    - `hoomd.md.methods`
    - `hoomd.md.methods.rattle`

    The classes of following modules can be used as elements in `forces`:

    - `hoomd.md.angle`
    - `hoomd.md.bond`
    - `hoomd.md.dihedral`
    - `hoomd.md.external`
    - `hoomd.md.improper`
    - `hoomd.md.long_range`
    - `hoomd.md.pair`
    - `hoomd.md.special_pair`
    - `hoomd.md.many_body`.

    The classes of the following module can be used as elements in
    `constraints`:

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
        dt (float): Integrator time step size :math:`[\mathrm{time}]`.

        methods (list[hoomd.md.methods.Method]): List of integration methods.

        forces (list[hoomd.md.force.Force]): List of forces applied to
            the particles in the system.

        integrate_rotational_dof (bool): When True, integrate rotational degrees
            of freedom.

        constraints (list[hoomd.md.constrain.Constraint]): List of
            constraint forces applied to the particles in the system.

        rigid (hoomd.md.constrain.Rigid): The rigid body definition for the
            simulation associated with the integrator.

        half_step_hook (hoomd.md.HalfStepHook): User defined implementation to
            perform computations during the half-step of the integration.
    """

    def __init__(self,
                 dt,
                 integrate_rotational_dof=False,
                 forces=None,
                 constraints=None,
                 methods=None,
                 rigid=None,
                 half_step_hook=None):

        super().__init__(forces, constraints, methods, rigid)

        self._param_dict.update(
            ParameterDict(
                dt=float(dt),
                integrate_rotational_dof=bool(integrate_rotational_dof),
                half_step_hook=OnlyTypes(hoomd.md.HalfStepHook,
                                         allow_none=True)))

        self.half_step_hook = half_step_hook

    def _attach_hook(self):
        # initialize the reflected c++ class
        self._cpp_obj = _md.IntegratorTwoStep(
            self._simulation.state._cpp_sys_def, self.dt)
        # Call attach from DynamicIntegrator which attaches forces,
        # constraint_forces, and methods, and calls super()._attach() itself.
        super()._attach_hook()

    def __setattr__(self, attr, value):
        """Hande group DOF update when setting integrate_rotational_dof."""
        super().__setattr__(attr, value)
        if (attr == 'integrate_rotational_dof' and self._simulation is not None
                and self._simulation.state is not None):
            self._simulation.state.update_group_dof()

    @hoomd.logging.log(category="sequence", requires_run=True)
    def linear_momentum(self):
        """tuple(float,float,float): The linear momentum vector of the system \
            :math:`[\\mathrm{mass} \\cdot \\mathrm{velocity}]`.

        .. math::

            \\vec{p} = \\sum_{i=0}^\\mathrm{N_particles-1} m_i \\vec{v}_i
        """
        v = self._cpp_obj.computeLinearMomentum()
        return (v.x, v.y, v.z)
