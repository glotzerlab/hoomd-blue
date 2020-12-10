# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

# Maintainer: joaander / All Developers are free to add commands for new
# features

R""" Apply forces to particles.
"""

import hoomd
from hoomd import _hoomd
from hoomd.md import _md
from hoomd.operation import _HOOMDBaseObject
from hoomd.logging import log
from hoomd.data.typeparam import TypeParameter
from hoomd.data.typeconverter import OnlyType
from hoomd.data.parameterdicts import ParameterDict, TypeParameterDict
from hoomd.filter import ParticleFilter


def ellip_preprocessing(constraint):
    if constraint is not None:
        if constraint.__class__.__name__ == "constraint_ellipsoid":
            return act_force
        else:
            raise RuntimeError(
                "Active force constraint is not accepted (currently only "
                "accepts ellipsoids)"
            )
    else:
        return None


class _force:
    pass


class Force(_HOOMDBaseObject):
    """Defines a force in HOOMD-blue.

    Pair, angle, bond, and other forces are subclasses of this class.

    Note:
        :py:class:`Force` is the base class for all loggable forces.
        Users should not instantiate this class directly.

    Initializes some loggable quantities.
    """

    def _attach(self):
        super()._attach()

    @log
    def energy(self):
        """float: Sum of the energy of the whole system."""
        if self._attached:
            self._cpp_obj.compute(self._simulation.timestep)
            return self._cpp_obj.calcEnergySum()
        else:
            return None

    @log(flag="particle")
    def energies(self):
        """(*N_particles*, ) `numpy.ndarray` of ``numpy.float64``: The energies
        for all particles."""
        if self._attached:
            self._cpp_obj.compute(self._simulation.timestep)
            return self._cpp_obj.getEnergies()
        else:
            return None

    @log(flag="particle")
    def forces(self):
        """(*N_particles*, 3) `numpy.ndarray` of ``numpy.float64``: The forces
        for all particles."""
        if self._attached:
            self._cpp_obj.compute(self._simulation.timestep)
            return self._cpp_obj.getForces()
        else:
            return None

    @log(flag="particle")
    def torques(self):
        """(*N_particles*, 3) `numpy.ndarray` of ``numpy.float64``: The torque
        for all particles."""
        if self._attached:
            self._cpp_obj.compute(self._simulation.timestep)
            return self._cpp_obj.getTorques()
        else:
            return None

    @log(flag="particle")
    def virials(self):
        """(*N_particles*, ) `numpy.ndarray` of ``numpy.float64``: The virial
        for all particles."""
        if self._attached:
            self._cpp_obj.compute(self._simulation.timestep)
            return self._cpp_obj.getVirials()
        else:
            return None


class constant(Force):
    R"""Constant force.

    Args:
        fvec (tuple): force vector (in force units)
        tvec (tuple): torque vector (in torque units)
        fx (float): x component of force, retained for backwards compatibility
        fy (float): y component of force, retained for backwards compatibility
        fz (float): z component of force, retained for backwards compatibility
        group (``hoomd.group``): Group for which the force will be set.
        callback (`callable`): A python callback invoked every time the forces
            are computed

    :py:class:`constant` specifies that a constant force should be added to
    every particle in the simulation or optionally to all particles in a group.

    Note:
        Forces are kept constant during the simulation. If a callback should
        re-compute particle forces every time step, it needs to overwrite the
        old forces of **all** particles with new values.

    Note:
        Per-particle forces take precedence over a particle group, which takes
        precedence over constant forces for all particles.

    Examples::

        force.constant(fx=1.0, fy=0.5, fz=0.25)
        const = force.constant(fvec=(0.4,1.0,0.5))
        const = force.constant(fvec=(0.4,1.0,0.5),group=fluid)
        const = force.constant(fvec=(0.4,1.0,0.5), tvec=(0,0,1) ,group=fluid)

        def updateForces(timestep):
            global const
            const.setForce(tag=1, fvec=(1.0*timestep,2.0*timestep,3.0*timestep))
        const = force.constant(callback=updateForces)
    """

    def __init__(
        self,
        fx=None,
        fy=None,
        fz=None,
        fvec=None,
        tvec=None,
        group=None,
        callback=None,
    ):

        if (fx is not None) and (fy is not None) and (fz is not None):
            self.fvec = (fx, fy, fz)
        elif fvec is not None:
            self.fvec = fvec
        else:
            self.fvec = (0, 0, 0)

        if tvec is not None:
            self.tvec = tvec
        else:
            self.tvec = (0, 0, 0)

        if (self.fvec == (0, 0, 0)) and (
            self.tvec == (0, 0, 0) and callback is None
        ):
            hoomd.context.current.device.cpp_msg.warning(
                "The constant force specified has no non-zero components\n"
            )

        # initialize the base class
        Force.__init__(self)

        # create the c++ mirror class
        if group is not None:
            self.cppForce = _hoomd.ConstForceCompute(
                hoomd.context.current.system_definition,
                group.cpp_group,
                self.fvec[0],
                self.fvec[1],
                self.fvec[2],
                self.tvec[0],
                self.tvec[1],
                self.tvec[2],
            )
        else:
            self.cppForce = _hoomd.ConstForceCompute(
                hoomd.context.current.system_definition,
                self.fvec[0],
                self.fvec[1],
                self.fvec[2],
                self.tvec[0],
                self.tvec[1],
                self.tvec[2],
            )

        if callback is not None:
            self.cppForce.setCallback(callback)

        hoomd.context.current.system.addCompute(self.cppForce, self.force_name)

    R""" Change the value of the constant force.

    Args:
        fx (float) New x-component of the force (in force units)
        fy (float) New y-component of the force (in force units)
        fz (float) New z-component of the force (in force units)
        fvec (tuple) New force vector
        tvec (tuple) New torque vector
        group Group for which the force will be set
        tag (int) Particle tag for which the force will be set
            .. versionadded:: 2.3

     Using setForce() requires that you saved the created constant force in a
     variable. i.e.

     Examples:
        const = force.constant(fx=0.4, fy=1.0, fz=0.5)

        const.setForce(fx=0.2, fy=0.1, fz=-0.5)
        const.setForce(fx=0.2, fy=0.1, fz=-0.5, group=fluid)
        const.setForce(fvec=(0.2,0.1,-0.5), tvec=(0,0,1), group=fluid)
    """

    def setForce(
        self,
        fx=None,
        fy=None,
        fz=None,
        fvec=None,
        tvec=None,
        group=None,
        tag=None,
    ):

        if (fx is not None) and (fy is not None) and (fx is not None):
            self.fvec = (fx, fy, fz)
        elif fvec is not None:
            self.fvec = fvec
        else:
            self.fvec = (0, 0, 0)

        if tvec is not None:
            self.tvec = tvec
        else:
            self.tvec = (0, 0, 0)

        if (fvec == (0, 0, 0)) and (tvec == (0, 0, 0)):
            hoomd.context.current.device.cpp_msg.warning(
                "You are setting the constant force to have no non-zero "
                "components\n"
            )

        self.check_initialization()
        if group is not None:
            self.cppForce.setGroupForce(
                group.cpp_group,
                self.fvec[0],
                self.fvec[1],
                self.fvec[2],
                self.tvec[0],
                self.tvec[1],
                self.tvec[2],
            )
        elif tag is not None:
            self.cppForce.setParticleForce(
                tag,
                self.fvec[0],
                self.fvec[1],
                self.fvec[2],
                self.tvec[0],
                self.tvec[1],
                self.tvec[2],
            )
        else:
            self.cppForce.setForce(
                self.fvec[0],
                self.fvec[1],
                self.fvec[2],
                self.tvec[0],
                self.tvec[1],
                self.tvec[2],
            )

    R""" Set a python callback to be called before the force is evaluated

    Args:
        callback (`callable`) The callback function

     Examples:
        const = force.constant(fx=0.4, fy=1.0, fz=0.5)

        def updateForces(timestep):
            global const
            const.setForce(tag=1, fvec=(1.0*timestep,2.0*timestep,3.0*timestep))

        const.set_callback(updateForces)
        run(100)

        # Reset the callback
        const.set_callback(None)
    """

    def set_callback(self, callback=None):
        self.cppForce.setCallback(callback)

    # there are no coeffs to update in the constant force compute
    def update_coeffs(self):
        pass


class Active(Force):
    R"""Active force.

    Attributes:
        filter (:py:mod:`hoomd.filter`): Subset of particles on which to apply
            active forces.
        seed (int): required user-specified seed number for random number
            generator.
        rotation_diff (float): rotational diffusion constant, :math:`D_r`, for
            all particles in the group.
        active_force (tuple): active force vector in reference to the
            orientation of a particle. It is defined per particle type and stays
            constant during the simulation.
        active_torque (tuple): active torque vector in reference to the
            orientation of a particle. It is defined per particle type and stays
            constant during the simulation.

    :py:class:`Active` specifies that an active force should be added to all
    particles.  Obeys :math:`\delta {\bf r}_i = \delta t v_0 \hat{p}_i`, where
    :math:`v_0` is the active velocity. In 2D :math:`\hat{p}_i = (\cos \theta_i,
    \sin \theta_i)` is the active force vector for particle :math:`i` and the
    diffusion of the active force vector follows :math:`\delta \theta / \delta t
    = \sqrt{2 D_r / \delta t} \Gamma`, where :math:`D_r` is the rotational
    diffusion constant, and the gamma function is a unit-variance random
    variable, whose components are uncorrelated in time, space, and between
    particles.  In 3D, :math:`\hat{p}_i` is a unit vector in 3D space, and
    diffusion follows :math:`\delta \hat{p}_i / \delta t = \sqrt{2 D_r / \delta
    t} \Gamma (\hat{p}_i (\cos \theta - 1) + \hat{p}_r \sin \theta)`, where
    :math:`\hat{p}_r` is an uncorrelated random unit vector. The persistence
    length of an active particle's path is :math:`v_0 / D_r`.  The rotational
    diffusion is applied to the orientation vector/quaternion of each particle.
    This implies that both the active force and the active torque vectors in the
    particle frame stay constant during the simulation. Hence, the active forces
    in the system frame are composed of the forces in particle frame and the
    current orientation of the particle.

    Examples::


        all = filter.All()
        active = hoomd.md.force.Active(
            filter=hoomd.filter.All(), seed=1, rotation_diff=0.01
            )
        active.active_force['A','B'] = (1,0,0)
        active.active_torque['A','B'] = (0,0,0)
    """
    def __init__(self, filter, seed, rotation_diff=0.1):
        # store metadata
        param_dict = ParameterDict(
            filter=ParticleFilter,
            seed=int(seed),
            rotation_diff=float(rotation_diff),
            constraint=OnlyType(
                ConstraintForce, allow_none=True, preprocess=ellip_preprocessing
            ),
        )
        param_dict.update(
            dict(
                constraint=None,
                rotation_diff=rotation_diff,
                seed=seed,
                filter=filter,
            )
        )
        # set defaults
        self._param_dict.update(param_dict)

        active_force = TypeParameter(
            "active_force",
            type_kind="particle_types",
            param_dict=TypeParameterDict((1, 0, 0), len_keys=1),
        )
        active_torque = TypeParameter(
            "active_torque",
            type_kind="particle_types",
            param_dict=TypeParameterDict((0, 0, 0), len_keys=1),
        )

        self._extend_typeparam([active_force, active_torque])

    def _attach(self):
        # initialize the reflected c++ class
        if isinstance(self._simulation.device, hoomd.device.CPU):
            my_class = _md.ActiveForceCompute
        else:
            my_class = _md.ActiveForceComputeGPU

        self._cpp_obj = my_class(
            self._simulation.state._cpp_sys_def,
            self._simulation.state._get_group(self.filter),
            self.seed,
            self.rotation_diff,
            _hoomd.make_scalar3(0, 0, 0),
            0,
            0,
            0,
        )

        # Attach param_dict and typeparam_dict
        super()._attach()


class dipole(Force):
    R"""Treat particles as dipoles in an electric field.

    Args:
        field_x (float): x-component of the field (units?)
        field_y (float): y-component of the field (units?)
        field_z (float): z-component of the field (units?)
        p (float): magnitude of the particles' dipole moment in the local z
            direction

    Examples::

        force.external_field_dipole(
            field_x=0.0, field_y=1.0 ,field_z=0.5, p=1.0
            )
        const_ext_f_dipole = force.external_field_dipole(
            field_x=0.0, field_y=1.0 ,field_z=0.5, p=1.0
            )
    """

    def __init__(self, field_x, field_y, field_z, p):

        # initialize the base class
        Force.__init__(self)

        # create the c++ mirror class
        self.cppForce = _md.ConstExternalFieldDipoleForceCompute(
            hoomd.context.current.system_definition,
            field_x,
            field_y,
            field_z,
            p,
        )

        hoomd.context.current.system.addCompute(self.cppForce, self.force_name)

        # store metadata
        self.field_x = field_x
        self.field_y = field_y
        self.field_z = field_z

    def set_params(field_x, field_y, field_z, p):
        R"""Change the constant field and dipole moment.

        Args:
            field_x (float): x-component of the field (units?)
            field_y (float): y-component of the field (units?)
            field_z (float): z-component of the field (units?)
            p (float): magnitude of the particles' dipole moment in the local z
                direction

        Examples::

            const_ext_f_dipole = force.external_field_dipole(
                field_x=0.0, field_y=1.0 ,field_z=0.5, p=1.0
                )
            const_ext_f_dipole.setParams(
                field_x=0.1, field_y=0.1, field_z=0.0, p=1.0
                )

        """
        self.check_initialization()

        self.cppForce.setParams(field_x, field_y, field_z, p)

    # there are no coeffs to update in the constant
    # ExternalFieldDipoleForceCompute
    def update_coeffs(self):
        pass


class ConstraintForce(Force):
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
    def _attach(self):
        """Create the c++ mirror class."""
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cpp_cls = getattr(_md, self._cpp_class_name)
        else:
            cpp_cls = getattr(_md, self._cpp_class_name + "GPU")

        # TODO remove string argument
        self._cpp_obj = cpp_cls(self._simulation.state._cpp_sys_def, "")

        super()._attach()

    def update_coeffs(self):
        pass
        # does nothing: this is for derived classes to implement


# set default counter
ConstraintForce.cur_id = 0


class distance(ConstraintForce):
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
        force.distance() does not currently interoperate with
        integrate.brownian() or integrate.langevin()

    Example::

        force.distance()

    """

    def __init__(self):

        # initialize the base class
        ConstraintForce.__init__(self)

        # create the c++ mirror class
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
            self.cpp_force = _md.ForceDistanceConstraint(
                self._simulation.state._cpp_sys_def
            )
        else:
            self.cpp_force = _md.ForceDistanceConstraintGPU(
                self._simulation.state._cpp_sys_def
            )

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name)

    def set_params(self, rel_tol=None):
        R"""Set parameters for constraint computation.

        Args:
            rel_tol (float): The relative tolerance with which constraint
                violations are detected (**optional**).

        Example::

            dist = force.distance()
            dist.set_params(rel_tol=0.0001)
        """
        if rel_tol is not None:
            self.cpp_force.setRelativeTolerance(float(rel_tol))


class rigid(ConstraintForce):
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
    to :py:class:`rigid` for each type of body in body coordinates. Then,
    :py:class:`rigid` takes control of those particles, and sets their position
    and orientation in the simulation box relative to the position and
    orientation of the central particle. :py:class:`rigid` also transfers forces
    and torques from constituent particles to the central particle. Then, MD
    integrators can use these forces and torques to integrate the equations of
    motion of the central particles (representing the whole rigid body) forward
    in time.

    .. rubric:: Defining bodies

    :py:class:`rigid` accepts one local body environment per body type. The
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
    defined in the argument to :py:meth:`set_param()`. The order of central and
    contiguous particles need **not** to be contiguous. Additionally, you must
    set the ``body`` field for each of the particles in the rigid body to the
    tag of the central particle (for both the central and constituent
    particles). Set ``body`` to -1 for particles that do not belong to a rigid
    body. After setting an initial configuration that contains properly defined
    bodies and all their constituent particles, call :py:meth:`validate_bodies`
    to verify that the bodies are defined and prepare the constraint.

    You must call either :py:meth:`create_bodies` or :py:meth:`validate_bodies`
    prior to starting a simulation ```hoomd.run```.

    .. rubric:: Integrating bodies

    Most integrators in HOOMD support the integration of rotational degrees of
    freedom. When there are rigid bodies present in the system, do not apply
    integrators to the constituent particles, only the central and non-rigid
    particles.

    Example::

        rigid = hoomd.group.rigid_center()
        hoomd.md.integrate.langevin(group=rigid, kT=1.0, seed=42)

    .. rubric:: Thermodynamic quantities of bodies

    HOOMD computes thermodynamic quantities (temperature, kinetic energy,
    etc...) appropriately when there are rigid bodies present in the system.
    When it does so, it ignores all constituent particles and computes the
    translational and rotational energies of the central particles, which
    represent the whole body. ``hoomd.analyze.log`` can log the translational
    and rotational energy terms separately.

    .. rubric:: Restarting simulations with rigid bodies.

    To restart, use :py:class:`hoomd.dump.GSD` to write restart files. GSD
    stores all of the particle data fields needed to reconstruct the state of
    the system, including the body tag, rotational momentum, and orientation of
    the body. Restarting from a gsd file is equivalent to manual constituent
    particle creation. You still need to specify the same local body space
    environment to :py:class:`rigid` as you did in the earlier simulation.

    """
    _cpp_class_name = "ForceComposite"
    def set_param(
        self,
        type_name,
        types,
        positions,
        orientations=None,
        charges=None,
        diameters=None,
    ):
        R"""Set constituent particle types and coordinates for a rigid body.

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

            rigid = force.rigid()
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
        # get a list of types from the particle data
        ntypes = (
            self._simulation.state._cpp_sys_def.getParticleData().getNTypes()
        )
        type_list = []
        for i in range(0, ntypes):
            type_list.append(
                self._simulation.state._cpp_sys_def.getParticleData().getNameByType(
                    i
                )
            )

        if type_name not in type_list:
            hoomd.context.current.device.cpp_msg.error(
                "Type " "{}" " not found.\n".format(type_name)
            )
            raise RuntimeError(
                "Error setting up parameters for force.rigid()"
            )

        type_id = type_list.index(type_name)

        if not isinstance(types, list):
            hoomd.context.current.device.cpp_msg.error(
                "Expecting list of particle types.\n"
            )
            raise RuntimeError(
                "Error setting up parameters for force.rigid()"
            )

        type_vec = _hoomd.std_vector_uint()
        for t in types:
            if t not in type_list:
                hoomd.context.current.device.cpp_msg.error(
                    "Type " "{}" " not found.\n".format(t)
                )
                raise RuntimeError(
                    "Error setting up parameters for force.rigid()"
                )
            constituent_type_id = type_list.index(t)

            type_vec.append(constituent_type_id)

        pos_vec = _hoomd.std_vector_scalar3()
        positions_list = list(positions)
        for p in positions_list:
            p = tuple(p)
            if len(p) != 3:
                hoomd.context.current.device.cpp_msg.error(
                    "Particle position is not a coordinate triple.\n"
                )
                raise RuntimeError(
                    "Error setting up parameters for force.rigid()"
                )
            pos_vec.append(_hoomd.make_scalar3(p[0], p[1], p[2]))

        orientation_vec = _hoomd.std_vector_scalar4()
        if orientations is not None:
            orientations_list = list(orientations)
            for o in orientations_list:
                o = tuple(o)
                if len(o) != 4:
                    hoomd.context.current.device.cpp_msg.error(
                        "Particle orientation is not a 4-tuple.\n"
                    )
                    raise RuntimeError(
                        "Error setting up parameters for force.rigid()"
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
        self.cpp_force.setParam(
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
        self.cpp_force.validateRigidBodies(create)

    def validate_bodies(self):
        R"""Validate that bodies are well defined and prepare for the
        simulation run.
        """
        self.cpp_force.validateRigidBodies(False)

    ## \internal
    # \brief updates force coefficients
    def update_coeffs(self):
        # validate copies of rigid bodies
        self.create_bodies(False)

