# Copyright (c) 2009-2021 The Regents of the University of Michigan This file is
# part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander / All Developers are free to add commands for new
# features

r""" Constraints.

Constraint forces can constrain particles to be a set distance from each other,
to have some relative orientation, or impose other types of constraint.

As with other force commands in hoomd, multiple constrain commands can be issued
to specify multiple constraints, which are additively applied.

The `Rigid` class is special in that only one is allowed in a system and is set
to an `hoomd.md.Integator` object separately in the `rigid` attribute.

Warning:
    Constraints will be invalidated if two separate constraint commands
    apply to the same particle.

The degrees of freedom removed from the system by constraints are correctly
taken into account when computing the temperature.
"""

from hoomd.md import _md
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.data.typeconverter import OnlyIf, to_type_converter
import hoomd
from hoomd.operation import _HOOMDBaseObject


class Constraint(_HOOMDBaseObject):
    """A constraint force that acts on the system."""
    def _attach(self):
        """Create the c++ mirror class."""
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_cls = getattr(_md, self._cpp_class_name)
        else:
            cpp_cls = getattr(_md, self._cpp_class_name + "GPU")

        self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def)

        super()._attach()


class distance(Constraint):
    R"""Constrain pairwise particle distances.

    :py:class:`distance` specifies that forces will be applied to all particles
    pairs for which constraints have been defined.

    The constraint algorithm implemented is described in:

     * [1] M. Yoneya, H. J. C. Berendsen, and K. Hirasawa, "A Non-Iterative
     Matrix Method for Constraint Molecular Dynamics Simulations," Mol. Simul.,
     vol. 13, no. 6, pp. 395--405, 1994.
     * [2] M. Yoneya, "A Generalized Non-iterative Matrix Method for Constraint
     Molecular Dynamics Simulations," J. Comput. Phys., vol. 172, no. 1, pp.
     188--197, Sep. 2001.

    In brief, the second derivative of the Lagrange multipliers with respect to
    time is set to zero, such that both the distance constraints and their time
    derivatives are conserved within the accuracy of the Velocity Verlet scheme,
    i.e. within :math:`\Delta t^2`. The corresponding linear system of equations
    is solved. Because constraints are satisfied at :math:`t + 2 \Delta t`, the
    scheme is self-correcting and drifts are avoided.

    Warning:
        In MPI simulations, all particles connected through constraints will be
        communicated between processors as ghost particles. Therefore, it is an
        error when molecules defined by constraints extend over more than half
        the local domain size.

    .. caution::
        constrain.distance() does not currently interoperate with
        integrate.brownian() or integrate.langevin()

    Example::

        constrain.distance()

    """

    def __init__(self):

        # initialize the base class
        Constraint.__init__(self)

        # create the c++ mirror class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self.cpp_force = _md.ForceDistanceConstraint(
                hoomd.context.current.system_definition
            )
        else:
            self.cpp_force = _md.ForceDistanceConstraintGPU(
                hoomd.context.current.system_definition
            )

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

    def set_params(self, rel_tol=None):
        R"""Set parameters for constraint computation.

        Args:
            rel_tol (float): The relative tolerance with which constraint
                violations are detected (**optional**).

        Example::

            dist = constrain.distance()
            dist.set_params(rel_tol=0.0001)
        """
        if rel_tol is not None:
            self.cpp_force.setRelativeTolerance(float(rel_tol))


class Rigid(Constraint):
    R"""Constrain particles in rigid bodies.

    .. rubric:: Overview

    Rigid bodies are defined by a single central particle and a number of
    constituent particles. All of these are particles in the HOOMD system
    configuration and can interact with other particles via force computes. The
    mass and moment of inertia of the central particle set the full mass and
    moment of inertia of the rigid body (constituent particle mass is ignored).

    The central particle is at the center of mass of the rigid body and the
    orientation quaternion defines the rotation from the body space into the
    simulation box. In body space, the center of mass of the body is at 0,0,0
    and the moment of inertia is diagonal. You specify the constituent particles
    to `Rigid` for each type of body in body coordinates. Then,
    `Rigid` takes control of those particles, and sets their position
    and orientation in the simulation box relative to the position and
    orientation of the central particle. `Rigid` also transfers forces
    and torques from constituent particles to the central particle. Then, MD
    integrators can use these forces and torques to integrate the equations of
    motion of the central particles (representing the whole rigid body) forward
    in time.

    .. rubric:: Defining bodies

    `Rigid` accepts one local body definition per body type. The
    type of a body is the particle type of the central particle in that body.
    In this way, each particle of type *R* in the system configuration defines
    a body of type *R*.

    As a convenience, you do not need to create placeholder entries for all of
    the constituent particles in your initial configuration. You only need to
    specify the positions and orientations of all the central particles. When
    you call `~.create_bodies`, it will create all constituent particles.

    Warning:
        Automatic creation of constituent particles can change particle tags. If
        bonds have been defined between particles in the initial configuration,
        or bonds connect to constituent particles, rigid bodies should be
        created manually.

    When you create the constituent particles manually (i.e. in an input file
    or with snapshots), the central particle of a rigid body must have a lower
    tag than all of its constituent particles. Constituent particles follow in
    monotonically increasing tag order, corresponding to the order they were
    defined in the argument to `Rigid` initialization. The order of central and
    contiguous particles need **not** to be contiguous. Additionally, you must
    set the ``body`` field for each of the particles in the rigid body to the
    tag of the central particle (for both the central and constituent
    particles). Set ``body`` to -1 for particles that do not belong to a rigid
    body.

    .. rubric:: Integrating bodies

    Most integrators in HOOMD support the integration of rotational degrees of
    freedom. When there are rigid bodies present in the system, do not apply
    integrators to the constituent particles, only the central and non-rigid
    particles.

    Example::

        rigid_centers_and_free_filter = hoomd.filter.Rigid(
            ("center", "free"))
        langevin = hoomd.md.methods.Langevin(
            filter=rigid_centers_and_free_filter, kT=1.0)


    .. rubric:: Thermodynamic quantities of bodies

    HOOMD computes thermodynamic quantities (temperature, kinetic energy,
    etc...) appropriately when there are rigid bodies present in the system.
    When it does so, it ignores all constituent particles and computes the
    translational and rotational energies of the central particles, which
    represent the whole body.

    .. rubric:: Restarting simulations with rigid bodies.

    To restart, use `hoomd.write.GSD` to write restart files. GSD
    stores all of the particle data fields needed to reconstruct the state of
    the system, including the body tag, rotational momentum, and orientation of
    the body. Restarting from a gsd file is equivalent to manual constituent
    particle creation. You still need to specify the same local body space
    environment to `Rigid` as you did in the earlier simulation.

    Set constituent particle types and coordinates for a rigid body.

    Args:
        type_name (str): The type of the central particle
        types (list): List of types of constituent particles
        positions (list): List of relative positions of constituent
            particles
        orientations (list): List of orientations of constituent particles
            (**optional**)
        charge (list): List of charges of constituent particles
            (**optional**)
        diameters (list): List of diameters of constituent particles
            (**optional**)

    .. caution::
        The constituent particle type must exist.

    Example::

        rigid = constrain.Rigid()
        rigid.body['A'] = {
            "types": ['A_const', 'A_const'],
            "positions": [(0,0,1),(0,0,-1)],
            "orientations": [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
            "charges": [0.0, 0.0],
            "diameters": [1.0, 1.0]
            }
        rigid.body['B'] = {
            "types": ['B_const', 'B_const'],
            "positions": [(0,0,.5),(0,0,-.5)],
            "orientations": [(1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)],
            "charges": [0.0, 1.0],
            "diameters": [1.5, 1.0]
            }

        # Can set rigid body definition to be None explicitly.
        rigid.body["A"] = None

    """

    _cpp_class_name = "ForceComposite"

    def __init__(self):
        body = TypeParameter(
            "body", "particle_types", TypeParameterDict(
                OnlyIf(to_type_converter(
                    {'constituent_types': [str],
                     'positions': [(float,) * 3],
                     'orientations': [(float,) * 4],
                     'charges': [float],
                     'diameters': [float]
                     }),
                    allow_none=True),
                len_keys=1
            )
        )
        self._add_typeparam(body)
        self.body.default = None

    def create_bodies(self, state):
        R"""Create rigid bodies from central particles in state.

        Args:
            state (hoomd.State): the state to add rigid bodies too.
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
