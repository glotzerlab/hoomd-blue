# Copyright (c) 2009-2022 The Regents of the University of Michigan.
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
import hoomd
from hoomd.operation import _HOOMDBaseObject


class Constraint(_HOOMDBaseObject):
    """Constraint force base class.

    Note:
        :py:class:`Constraint` is the base class for all constraint forces.
        Users should not instantiate this class directly.
    """

    def _attach(self):
        """Create the c++ mirror class."""
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_cls = getattr(_md, self._cpp_class_name)
        else:
            cpp_cls = getattr(_md, self._cpp_class_name + "GPU")

        self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def)

        super()._attach()


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
    ``constraint_group`` and :math:`d_{ij}` is the constraint distance.
    Define any number of constraint groups in the system state.

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
    orientation quaternion defines the rotation from the body space into the
    simulation box. Body space refers to a rigid body viewed in a particular
    reference frame. In body space, the center of mass of the body is at
    :math:`(0,0,0)` and the moment of inertia is diagonal. You specify the
    constituent particles to `Rigid` for each type of body in body coordinates.
    Then, `Rigid` takes control of those particles, and sets their position and
    orientation in the simulation box relative to the position and orientation
    of the central particle:

    .. math::

        \vec{r}_c &= \vec{r}_b
                    + \mathbf{q}_b \vec{r}_{c,\mathrm{body}} \mathbf{q}_b^* \\
        \mathbf{q}_c &= \mathbf{q}_b \mathbf{q}_{c,\mathrm{body}}

    where :math:`\vec{r}_c` and :math:`\mathbf{q}_c` are the position and
    orientation of a constituent particle in the simulation box,
    :math:`\vec{r}_{c,\mathrm{body}}` and :math:`\mathbf{q}_{c,\mathrm{body}}`
    are the position and orientation of that particle in body coordinates, and
    :math:`\vec{r}_b` and :math:`\mathbf{q}_b` are the position and orientation
    of the central particle of that rigid body. In the simulation state, the
    ``body`` particle property defines the particle tag of the central particle:
    ``b = body[c]``. In setting the ``body`` array, central particles should be set
    to their tag :math:`b_i = t_i`, constituent particles to their central particle's tag
    :math:`b_i = t_{center}`, and free particles :math:`b_i = -1`

    `Rigid` transfers forces, energies, and torques from constituent particles
    to the central particle and adds them to those from the interaction on the
    central particle itself. The molecular integration methods use these forces
    and torques to integrate the equations of motion of the central particles
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

    .. rubric:: Defining bodies

    `Rigid` accepts one local body definition per body type. The type of a body
    is the particle type of the central particle in that body. In this way, each
    particle of type *R* in the system configuration defines a body of type *R*.

    As a convenience, you do not need to create placeholder entries for all of
    the constituent particles in your initial configuration. You can specify
    only the positions and orientations of all the central particles, then call
    `create_bodies` to create all constituent particles.

    Warning:
        Place constituent particle placeholders in the simulation state when
        there are bonds between particles. `create_bodies` changes particle
        tags.

    In the simulation state, the central particle of a rigid body must have a
    lower tag than all of its constituent particles. Constituent particles
    follow in monotonically increasing tag order, corresponding to the order
    they are defined in the argument to `Rigid` initialization. The central and
    constituent particles do not need to be contiguous. Additionally, you must
    set the ``body`` field for each of the particles in the rigid body to the
    tag of the central particle (for both the central and constituent
    particles). Set ``body`` to -1 for particles that do not belong to a rigid
    body (i.e. free bodies).

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

    To continue a simulation, use `hoomd.write.GSD` to write the system state to
    a file. GSD stores all of the particle data fields needed to reconstruct the
    state of the system, including the body tag, angular momentum, and
    orientation of the body. Continuing from a gsd file is equivalent to
    manually placing constituent particles. You must specify the same local body
    space environment to `body` as you did in the earlier simulation.

    To set constituent particle types and coordinates for a rigid body use the
    `body` attribute.

    .. caution::
        The constituent particle type(s) must exist in the simulation state.

    Example::

        rigid = constrain.Rigid()
        rigid.body['A'] = {
            "constituent_types": ['A_const', 'A_const'],
            "positions": [(0,0,1),(0,0,-1)],
            "orientations": [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
            "charges": [0.0, 0.0],
            "diameters": [1.0, 1.0]
            }
        rigid.body['B'] = {
            "constituent_types": ['B_const', 'B_const'],
            "positions": [(0,0,.5),(0,0,-.5)],
            "orientations": [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
            "charges": [0.0, 1.0],
            "diameters": [1.5, 1.0]
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
        - ``charge`` (`list` [`float`]): List of charges of constituent
          particles.
        - ``diameters`` (`list` [`float`]): List of diameters of constituent
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
                'charges': [float],
                'diameters': [float]
            }),
                                     allow_none=True),
                              len_keys=1))
        self._add_typeparam(body)
        self.body.default = None

    def create_bodies(self, state):
        r"""Create rigid bodies from central particles in state.

        Args:
            state (hoomd.State): The state in which to create rigid bodies.

        `create_bodies` removes any existing constituent particles and adds new
        ones based on the body definitions in `body`. It overwrites all existing
        particle ``body`` tags in the state.
        """
        if self._attached:
            raise RuntimeError(
                "Cannot call create_bodies after running simulation.")
        # Attach and store information for detaching after calling
        # createRigidBodies
        old_sim = None
        if self._added:
            old_sim = self._simulation
        self._add(state._simulation)
        super()._attach()

        self._cpp_obj.createRigidBodies()

        # Restore previous state
        self._detach()
        if old_sim is not None:
            self._simulation = old_sim
        else:
            self._remove()

    def _attach(self):
        super()._attach()
        # Need to ensure body tags and molecule sizes are correct and that the
        # positions and orientations are accurate before integration.
        self._cpp_obj.validateRigidBodies()
        self._cpp_obj.updateCompositeParticles(0)
