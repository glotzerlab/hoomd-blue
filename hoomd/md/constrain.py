# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Constraints.

Constraint force classes apply forces and the resulting virial to particles that
enforce specific constraints on the positions of the particles. The constraint
is satisfied at all times, so there is no potential energy associated with the
constraint.

Each constraint removes a number of degrees of freedom from the system.
`hoomd.md.compute.ThermodynamicQuantities` accounts for these lost degrees of
freedom when computing kinetic temperature and pressure. See
`hoomd.State.update_group_dof` for details on when the degrees of freedom for a
group are calculated.

Warning:
    Do not apply multiple constraint class instances to the same particle. Each
    instance solves for its constraints independently.
"""

from hoomd.md import _md
from hoomd.data.parameterdicts import ParameterDict, TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.data.typeconverter import OnlyIf, to_type_converter
from hoomd.md.force import Force
import hoomd


class Constraint(Force):
    """Constraint force base class.

    `Constraint` is the base class for all constraint forces.

    Warning:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.
    """

    # Module where the C++ class is defined. Reassign this when developing an
    # external plugin.
    _ext_module = _md

    def _attach_hook(self):
        """Create the c++ mirror class."""
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_cls = getattr(_md, self._cpp_class_name)
        else:
            cpp_cls = getattr(_md, self._cpp_class_name + "GPU")

        self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def)


class Distance(Constraint):
    """Constrain pairwise particle distances.

    Args:
        tolerance (float): Relative tolerance for constraint violation warnings.

    `Distance` applies forces between particles that constrain the distances
    between particles to specific values. The algorithm implemented is described
    in:

    1. M. Yoneya, H. J. C. Berendsen, and K. Hirasawa, "A Non-Iterative
       Matrix Method for Constraint Molecular Dynamics Simulations," Molecular
       Simulation, vol. 13, no. 6, pp. 395--405, 1994.
    2. M. Yoneya, "A Generalized Non-iterative Matrix Method for Constraint
       Molecular Dynamics Simulations," Journal of Computational Physics,
       vol. 172, no. 1, pp. 188--197, Sep. 2001.

    Each distance constraint takes the form:

    .. math::

        \\chi_{ij}(r) = \\mathrm{minimum\\_image}(\\vec{r}_j - \\vec{r}_i)^2
            - d_{ij}^2 = 0

    Where :math:`i` and :math:`j` are the the particle tags in the
    ``constraint_group`` and :math:`d_{ij}` is the constraint distance as given
    by the `system state <hoomd.State>`_.

    The method sets the second derivative of the Lagrange multipliers with
    respect to time to zero, such that both the distance constraints and their
    time derivatives are conserved within the accuracy of the Velocity Verlet
    scheme (:math:`O(\\delta t^2)`. It solves the corresponding linear system of
    equations to determine the force. The constraints are satisfied at :math:`t
    + 2 \\delta t`, so the scheme is self-correcting and avoids drifts.

    Add an instance of `Distance` to the integrator constraints list
    `hoomd.md.Integrator.constraints` to apply the force during the simulation.

    Warning:
        In MPI simulations, it is an error when molecules defined by constraints
        extend over more than half the local domain size because all particles
        connected through constraints will be communicated between ranks as
        ghost particles.

    Note:
        `tolerance` sets the tolerance to detect constraint violations and
        issue a warning message. It does not influence the computation of the
        constraint force.

    Attributes:
        tolerance (float): Relative tolerance for constraint violation warnings.
    """

    _cpp_class_name = "ForceDistanceConstraint"

    def __init__(self, tolerance=1e-3):
        self._param_dict.update(ParameterDict(tolerance=float(tolerance)))


class Rigid(Constraint):
    r"""Constrain particles in rigid bodies.

    .. rubric:: Overview

    Rigid bodies are defined by a single central particle and a number of
    constituent particles. All of these are particles in the simulation state
    and can interact with other particles via forces. The mass and moment of
    inertia of the central particle set the full mass and moment of inertia of
    the rigid body (constituent particle mass is ignored).

    The central particle is at the center of mass of the rigid body and the
    particle's orientation quaternion defines the rotation from the body space
    into the simulation box. Body space refers to a rigid body viewed in a
    particular reference frame. In body space, the center of mass of the body is
    at :math:`(0,0,0)` and the moment of inertia is diagonal.

    .. rubric:: Constituent particles

    Select one or more particle types in the simulation to use as the central
    particles. For each rigid body particle type, set the constituent particle
    type, position and orientations in body coordinates (see `body`). Then,
    `Rigid` takes control of the constituent particles and sets their position
    and orientation in the simulation box relative to the position and
    orientation of the central particle:

    .. math::

        \vec{r}_c &= \vec{r}_b
                    + \mathbf{q}_b \vec{r}_{c,\mathrm{body}} \mathbf{q}_b^* \\
        \mathbf{q}_c &= \mathbf{q}_b \mathbf{q}_{c,\mathrm{body}}

    where :math:`\vec{r}_c` and :math:`\mathbf{q}_c` are the position and
    orientation of a constituent particle in the simulation box,
    :math:`\vec{r}_{c,\mathrm{body}}` and :math:`\mathbf{q}_{c,\mathrm{body}}`
    are the position and orientation of that particle in body coordinates, and
    :math:`\vec{r}_b` and :math:`\mathbf{q}_b` are the position and orientation
    of the central particle of that rigid body. `Rigid` also sets the
    constituent particle image consistent with the image of the central particle
    and the location of the constituent particle wrapped back into the
    box.

    Warning:
        `Rigid` **overwrites** the constituent particle type ids, positions and
        orientations. To change the position and orientation of a body, set the
        desired position and orientation of the central particle and call
        `run(0) <Simulation.run>` to trigger `Rigid` to update the particles.

    In the simulation state, the ``body`` particle property defines the particle
    tag of the central particle: ``b = body[c]``. In setting the ``body`` array,
    set central particles to their tag :math:`b_i = t_i`, constituent particles
    to their central particle's tag :math:`b_i = t_{center}`, and free particles
    to -1: :math:`b_i = -1`. Free particles are particles that are not part of
    a rigid body.

    The central particle of a rigid body must have a tag smaller than all of its
    constituent particles. Constituent particles follow in monotonically
    increasing tag order, corresponding to the order they are defined in the
    argument to `Rigid` initialization. The central and constituent particles do
    not need to be contiguous.

    Tip:
        To create constituent particles, initialize a simulation state
        containing only central particles (but both rigid body and constituent
        particle types). Then call `create_bodies` to add all the constituent
        particles to the state.

    .. rubric:: Net force and torque

    `Rigid` transfers forces, energies, and torques from constituent particles
    to the central particle and adds them to those from the interaction on the
    central particle itself. The integration methods use these forces and
    torques to integrate the equations of motion of the central particles
    (representing the whole rigid body) forward in time.

    .. math::

        \vec{F}_b' &= \vec{F}_b + \sum_c \vec{F}_c \\
        \vec{U}_b' &= U_b + \sum_c U_c \\
        \vec{\tau}_b' &= \vec{\tau}_b + \sum_c \vec{\tau}_c +
            (\mathbf{q}_b \vec{r}_{c,\mathrm{body}} \mathbf{q}_b^*)
            \times \vec{F}_c

    `Rigid` also computes the corrected virial accounting for the effective
    constraint force (see `Glaser 2020
    <https://dx.doi.org/10.1016/j.commatsci.2019.109430>`_).

    Note:
        Include ``'body'`` in the `Neighborlist exclusions
        <hoomd.md.nlist.NeighborList.exclusions>` to avoid calculating
        inter-body forces that will sum to 0. This is *required* in many cases
        where nearby particles lead to numerical errors from extremely large
        forces.

    .. rubric:: Integrating bodies

    Set the ``rigid`` attribute of `hoomd.md.Integrator` to an instance of
    `Rigid` to apply rigid body constraints and apply an integration method (or
    methods) to the central and non-rigid particles (leave the constituent
    particles out - `Rigid` will set their position and orientation). Most
    integration methods support the integration of rotational degrees of
    freedom.

    Example::

        rigid_centers_and_free_filter = hoomd.filter.Rigid(
            ("center", "free"))
        langevin = hoomd.md.methods.Langevin(
            filter=rigid_centers_and_free_filter, kT=1.0)

    .. rubric:: Thermodynamic quantities of bodies

    `hoomd.md.compute.ThermodynamicQuantities` computes thermodynamic quantities
    (temperature, kinetic energy, etc.) over the central and non-rigid particles
    in the system, ignoring the consitutent particles. The body central
    particles contribute translational and rotational energies to the total.

    .. rubric:: Continuing simulations with rigid bodies.

    To continue a simulation, write the simulation state to a file with
    `hoomd.write.GSD` and initialize the new `Simulation <hoomd.Simulation>`
    using `create_state_from_gsd <Simulation.create_state_from_gsd>`. GSD stores
    all the particle data fields needed to reconstruct the state of the system,
    including the body, angular momentum, and orientation of the body. Set the
    same local body space environment to `body` as in the earlier simulation -
    GSD does not store this information.

    Example::

        rigid = constrain.Rigid()
        rigid.body['A'] = {
            "constituent_types": ['A_const', 'A_const'],
            "positions": [(0,0,1),(0,0,-1)],
            "orientations": [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
            }
        rigid.body['B'] = {
            "constituent_types": ['B_const', 'B_const'],
            "positions": [(0,0,.5),(0,0,-.5)],
            "orientations": [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
            }

        # Can set rigid body definition to be None explicitly.
        rigid.body["A"] = None

    Warning:
        `Rigid` will significantly slow simulation performance when frequently
        changing rigid body definitions or adding/removing particles from the
        simulation.

    .. py:attribute:: body

        `body` is a mapping from the central particle type to a body definition
        represented as a dictionary. The mapping respects ``None`` as meaning
        that the type is not a rigid body center. All types are set to ``None``
        by default. The keys for the body definition are:

        - ``constituent_types`` (`list` [`str`]): List of types of constituent
          particles.

        - ``positions`` (`list` [`tuple` [`float`, `float`, `float`]]): List of
          relative positions of constituent particles.

        - ``orientations`` (`list` [`tuple` [`float`, `float`, `float`,
          `float`]]): List of orientations (as quaternions) of constituent
          particles.

        Of these, `Rigid` uses ``constituent_types``, ``positions`` and
        ``orientations`` to set the constituent particle type ids, positions and
        orientations every time step. `create_bodies` uses all these parameters
        to populate those particle properties when creating constituent
        particles.

        Type: `TypeParameter` [``particle_type``, `dict`]
    """

    _cpp_class_name = "ForceComposite"

    def __init__(self):
        body = TypeParameter(
            "body", "particle_types",
            TypeParameterDict(OnlyIf(to_type_converter({
                'constituent_types': [str],
                'positions': [(float,) * 3],
                'orientations': [(float,) * 4],
            }),
                                     allow_none=True),
                              len_keys=1))
        self._add_typeparam(body)
        self.body.default = None

    def create_bodies(self, state, charges=None):
        r"""Create rigid bodies from central particles in state.

        Args:
            state (hoomd.State): The state in which to create rigid bodies.
            charges (dict[str, list[float]]): (optional) The charges for each of
                the constituent particles, defaults to ``None``. If ``None``,
                all charges are zero. The keys should be the central particles.

        `create_bodies` removes any existing constituent particles and adds new
        ones based on the body definitions in `body`. It overwrites all existing
        particle ``body`` tags in the state.
        """
        if self._attached:
            raise RuntimeError(
                "Cannot call create_bodies after running simulation.")
        super()._attach(state._simulation)
        self._cpp_obj.createRigidBodies({} if charges is None else charges)
        # Restore previous state
        self._detach()

    def _attach_hook(self):
        super()._attach_hook()
        # Need to ensure body tags and molecule sizes are correct and that the
        # positions and orientations are accurate before integration.
        self._cpp_obj.validateRigidBodies()
        self._cpp_obj.updateCompositeParticles(0)
