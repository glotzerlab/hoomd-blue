# Copyright (c) 2009-2021 The Regents of the University of Michigan This file is
# part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander / All Developers are free to add commands for new
# features

R""" Constraints.

Constraint forces constrain a given set of particle to a given surface, to have
some relative orientation, or impose some other type of constraint.

As with other force commands in hoomd, multiple constrain commands can be issued
to specify multiple constraints, which are additively applied.

Warning: Constraints will be invalidated if two separate constraint commands
apply to the same particle.

The degrees of freedom removed from the system by constraints are correctly
taken into account when computing the temperature.
"""

from hoomd import _hoomd
from hoomd.md import _md
from hoomd.md.force import ConstraintForce
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
        ConstraintForce.__init__(self)

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
    :py:class:`Rigid` takes control of those particles, and sets their position
    and orientation in the simulation box relative to the position and
    orientation of the central particle. :py:class:`Rigid` also transfers forces
    and torques from constituent particles to the central particle. Then, MD
    integrators can use these forces and torques to integrate the equations of
    motion of the central particles (representing the whole rigid body) forward
    in time.

    .. rubric:: Defining bodies

    :py:class:`Rigid` accepts one local body environment per body type. The
    type of a body is the particle type of the central particle in that body.
    In this way, each particle of type *R* in the system configuration defines
    a body of type *R*.

    As a convenience, you do not need to create placeholder entries for all of
    the constituent particles in your initial configuration. You only need to
    specify the positions and orientations of all the central particles. When
    you call :py:meth:`create_bodies()`, it will create all constituent
    particles that do not exist. (those that already exist e.g. in a restart
    file are left unchanged).

    .. danger:: Automatic creation of constituent particles can change particle
    tags. If bonds have been defined between particles in the initial
    configuration, or bonds connect to constituent particles, rigid bodies
    should be created manually.

    When you create the constituent particles manually (i.e. in an input file
    or with snapshots), the central particle of a rigid body must have a lower
    tag than all of its constituent particles. Constituent particles follow in
    monotonically increasing tag order, corresponding to the order they were
    defined in the argument to `Rigid` initialization. The order of central and
    contiguous particles need **not** to be contiguous. Additionally, you must
    set the ``body`` field for each of the particles in the rigid body to the
    tag of the central particle (for both the central and constituent
    particles). Set ``body`` to -1 for particles that do not belong to a rigid
    body. After setting an initial configuration that contains properly defined
    bodies and all their constituent particles, call :py:meth:`validate_bodies`
    to verify that the bodies are defined and prepare the constraint.

    You must call either :py:meth:`create_bodies` or :py:meth:`validate_bodies`
    prior to starting a simulation :py:meth:`hoomd.Simulation.run`.

    .. rubric:: Integrating bodies

    Most integrators in HOOMD support the integration of rotational degrees of
    freedom. When there are rigid bodies present in the system, do not apply
    integrators to the constituent particles, only the central and non-rigid
    particles.

    Example::

        rigid = hoomd.group.rigid_center()
        hoomd.md.integrate.langevin(group=rigid, kT=1.0)


    .. rubric:: Thermodynamic quantities of bodies

    HOOMD computes thermodynamic quantities (temperature, kinetic energy,
    etc...) appropriately when there are rigid bodies present in the system.
    When it does so, it ignores all constituent particles and computes the
    translational and rotational energies of the central particles, which
    represent the whole body. :py:meth:`hoomd.logging.log` can log the
    translational and rotational energy terms separately.

    .. rubric:: Restarting simulations with rigid bodies.

    To restart, use :py:class:`hoomd.write.GSD` to write restart files. GSD
    stores all of the particle data fields needed to reconstruct the state of
    the system, including the body tag, rotational momentum, and orientation of
    the body. Restarting from a gsd file is equivalent to manual constituent
    particle creation. You still need to specify the same local body space
    environment to :py:class:`Rigid` as you did in the earlier simulation.

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
        The constituent particle type must be exist.
        If it does not exist, it can be created on the fly using
        ``system.particles.types.add('A_const')``.

    Example::

        rigid = constrain.Rigid()
        rigid.set_param(
            'A',
            types = ['A_const', 'A_const'],
            positions = [(0,0,1),(0,0,-1)]
            )
        rigid.set_param(
            'B',
            types = ['B_const', 'B_const'],
            positions = [(0,0,.5),(0,0,-.5)]
            )

    """

    _cpp_class_name = "ForceComposite"
    def __init__(self):
        pass

    def set_params(
            self,
            type_name,
            types,
            positions,
            orientations=None,
            charges=None,
            diameters=None,
            ):
        # get a list of types from the particle data
        ntypes = (
            self._simulation.state._cpp_sys_def.getParticleData().getNTypes()
        )
        type_list = []
        for i in range(0, ntypes):
            type_list.append(
                self._simulation.state._cpp_sys_def.getParticleData()
                .getNameByType(i)
            )

        if type_name not in type_list:
            self._simulation.device._cpp_msg.error(
                "Type " "{}" " not found.\n".format(type_name)
            )
            raise RuntimeError(
                "Error setting up parameters for constrain.rigid()"
            )

        type_id = type_list.index(type_name)

        if not isinstance(types, list):
            self._simulation.device._cpp_msg.error(
                "Expecting list of particle types.\n"
            )
            raise RuntimeError(
                "Error setting up parameters for constrain.rigid()"
            )

        type_vec = _hoomd.std_vector_uint()
        for t in types:
            if t not in type_list:
                self._simulation.device._cpp_msg.error(
                    "Type " "{}" " not found.\n".format(t)
                )
                raise RuntimeError(
                    "Error setting up parameters for constrain.rigid()"
                )
            constituent_type_id = type_list.index(t)

            type_vec.append(constituent_type_id)

        pos_vec = _hoomd.std_vector_scalar3()
        positions_list = list(positions)
        for p in positions_list:
            p = tuple(p)
            if len(p) != 3:
                self._simulation.device._cpp_msg.error(
                    "Particle position is not a coordinate triple.\n"
                )
                raise RuntimeError(
                    "Error setting up parameters for constrain.rigid()"
                )
            pos_vec.append(_hoomd.make_scalar3(p[0], p[1], p[2]))

        orientation_vec = _hoomd.std_vector_scalar4()
        if orientations is not None:
            orientations_list = list(orientations)
            for o in orientations_list:
                o = tuple(o)
                if len(o) != 4:
                    self._simulation.device._cpp_msg.error(
                        "Particle orientation is not a 4-tuple.\n"
                    )
                    raise RuntimeError(
                        "Error setting up parameters for constrain.rigid()"
                    )
                orientation_vec.append(
                    _hoomd.make_scalar4(o[0], o[1], o[2], o[3])
                )
        else:
            for p in positions:
                orientation_vec.append(_hoomd.make_scalar4(1, 0, 0, 0))

        charge_vec = _hoomd.std_vector_scalar()
        if charges is not None:
            charges_list = list(charges)
            for c in charges_list:
                charge_vec.append(float(c))
        else:
            for p in positions:
                charge_vec.append(0.0)

        diameter_vec = _hoomd.std_vector_scalar()
        if diameters is not None:
            diameters_list = list(diameters)
            for d in diameters_list:
                diameter_vec.append(float(d))
        else:
            for p in positions:
                diameter_vec.append(1.0)

        # set parameters in C++ force
        self._cpp_obj.setParam(
            type_id,
            type_vec,
            pos_vec,
            orientation_vec,
            charge_vec,
            diameter_vec,
        )

    def create_bodies(self, create=True):
        R"""Create copies of rigid bodies.

        Args:
            create (bool): When True, create rigid bodies, otherwise validate
                existing ones.
        """
        self._cpp_obj.validateRigidBodies(create)

    def validate_bodies(self):
        R"""Validate that bodies are well defined and prepare for the
        simulation run.
        """
        self._cpp_obj.validateRigidBodies(False)

