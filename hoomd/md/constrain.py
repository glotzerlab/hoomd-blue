# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander / All Developers are free to add commands for new features

R""" Constraints.

Constraint forces constrain a given set of particle to a given surface, to have some relative orientation,
or impose some other type of constraint. For example, a group of particles can be constrained to the surface of a
sphere with :py:class:`sphere`.

As with other force commands in hoomd, multiple constrain commands can be issued to specify multiple
constraints, which are additively applied.

Warning:
    Constraints will be invalidated if two separate constraint commands apply to the same particle.

The degrees of freedom removed from the system by constraints are correctly taken into account when computing the
temperature for thermostatting and logging.
"""

from hoomd import _hoomd
from hoomd.md import _md
from hoomd.md import force;
import hoomd;

## \internal
# \brief Base class for constraint forces
#
# A constraint_force in hoomd reflects a ForceConstraint in c++. It is responsible
# for all high-level management that happens behind the scenes for hoomd
# writers. 1) The instance of the c++ constraint force itself is tracked and added to the
# System 2) methods are provided for disabling the force from being added to the
# net force on each particle
class _constraint_force(hoomd.meta._metadata):
    ## \internal
    # \brief Constructs the constraint force
    #
    # \param name name of the constraint force instance
    #
    # Initializes the cpp_force to None.
    # If specified, assigns a name to the instance
    # Assigns a name to the force in force_name;
    def __init__(self):
        # check if initialization has occurred
        if not hoomd.init.is_initialized():
            hoomd.context.msg.error("Cannot create force before initialization\n");
            raise RuntimeError('Error creating constraint force');

        self.cpp_force = None;

        # increment the id counter
        id = _constraint_force.cur_id;
        _constraint_force.cur_id += 1;

        self.force_name = "constraint_force%d" % (id);
        self.enabled = True;

        self.composite = False;
        hoomd.context.current.constraint_forces.append(self);

        # create force data iterator
        self.forces = hoomd.data.force_data(self);

        # base class constructor
        hoomd.meta._metadata.__init__(self)

    ## \var enabled
    # \internal
    # \brief True if the force is enabled

    ## \var composite
    # \internal
    # \brief True if this is a composite body force

    ## \var cpp_force
    # \internal
    # \brief Stores the C++ side ForceCompute managed by this class

    ## \var force_name
    # \internal
    # \brief The Force's name as it is assigned to the System

    ## \internal
    # \brief Checks that proper initialization has completed
    def check_initialization(self):
        # check that we have been initialized properly
        if self.cpp_force is None:
            hoomd.context.msg.error('Bug in hoomd: cpp_force not set, please report\n');
            raise RuntimeError();

    def disable(self):
        R""" Disable the force.

        Example::

            force.disable()


        Executing the disable command removes the force from the simulation.
        Any :py:func:`hoomd.run()` command executed after disabling a force will not calculate or
        use the force during the simulation. A disabled force can be re-enabled
        with :py:meth:`enable()`
        """
        hoomd.util.print_status_line();
        self.check_initialization();

        # check if we are already disabled
        if not self.enabled:
            hoomd.context.msg.warning("Ignoring command to disable a force that is already disabled");
            return;

        self.enabled = False;

        # remove the compute from the system
        hoomd.context.current.system.removeCompute(self.force_name);
        hoomd.context.current.constraint_forces.remove(self)

    def enable(self):
        R""" Enable the force.

        Example::

            force.enable()

        See :py:meth:`disable()`.
        """
        hoomd.util.print_status_line();
        self.check_initialization();

        # check if we are already disabled
        if self.enabled:
            hoomd.context.msg.warning("Ignoring command to enable a force that is already enabled");
            return;

        # add the compute back to the system
        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);
        hoomd.context.current.constraint_forces.append(self)

        self.enabled = True;

    ## \internal
    # \brief updates force coefficients
    def update_coeffs(self):
        pass
        # does nothing: this is for derived classes to implement


    ## \internal
    # \brief Get metadata
    def get_metadata(self):
        data = hoomd.meta._metadata.get_metadata(self)
        data['enabled'] = self.enabled
        if self.name != "":
            data['name'] = self.name
        return data

# set default counter
_constraint_force.cur_id = 0;

class sphere(_constraint_force):
    R""" Constrain particles to the surface of a sphere.

    Args:
        group (:py:mod:`hoomd.group`): Group on which to apply the constraint.
        P (tuple): (x,y,z) tuple indicating the position of the center of the sphere (in distance units).
        r (float): Radius of the sphere (in distance units).

    :py:class:`sphere` specifies that forces will be applied to all particles in the given group to constrain
    them to a sphere. Currently does not work with Brownian or Langevin dynamics (:py:class:`hoomd.md.integrate.brownian`
    and :py:class:`hoomd.md.integrate.langevin`).

    Example::

        constrain.sphere(group=groupA, P=(0,10,2), r=10)

    """
    def __init__(self, group, P, r):
        hoomd.util.print_status_line();

        # initialize the base class
        _constraint_force.__init__(self);

        # create the c++ mirror class
        P = _hoomd.make_scalar3(P[0], P[1], P[2]);
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _md.ConstraintSphere(hoomd.context.current.system_definition, group.cpp_group, P, r);
        else:
            self.cpp_force = _md.ConstraintSphereGPU(hoomd.context.current.system_definition, group.cpp_group, P, r);

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        # store metadata
        self.group = group
        self.P = P
        self.r = r
        self.metadata_fields = ['group','P', 'r']

class distance(_constraint_force):
    R""" Constrain pairwise particle distances.

    :py:class:`distance` specifies that forces will be applied to all particles pairs for
    which constraints have been defined.

    The constraint algorithm implemented is described in:

     * [1] M. Yoneya, H. J. C. Berendsen, and K. Hirasawa, "A Non-Iterative Matrix Method for Constraint Molecular Dynamics Simulations," Mol. Simul., vol. 13, no. 6, pp. 395--405, 1994.
     * [2] M. Yoneya, "A Generalized Non-iterative Matrix Method for Constraint Molecular Dynamics Simulations," J. Comput. Phys., vol. 172, no. 1, pp. 188--197, Sep. 2001.

    In brief, the second derivative of the Lagrange multipliers with respect to time is set to zero, such
    that both the distance constraints and their time derivatives are conserved within the accuracy of the Velocity
    Verlet scheme, i.e. within :math:`\Delta t^2`. The corresponding linear system of equations is solved.
    Because constraints are satisfied at :math:`t + 2 \Delta t`, the scheme is self-correcting and drifts are avoided.

    Warning:
        In MPI simulations, all particles connected through constraints will be communicated between processors as ghost particles.
        Therefore, it is an error when molecules defined by constraints extend over more than half the local domain size.

    .. caution::
        constrain.distance() does not currently interoperate with integrate.brownian() or integrate.langevin()

    Example::

        constrain.distance()

    """
    def __init__(self):
        hoomd.util.print_status_line();

        # initialize the base class
        _constraint_force.__init__(self);

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _md.ForceDistanceConstraint(hoomd.context.current.system_definition);
        else:
            self.cpp_force = _md.ForceDistanceConstraintGPU(hoomd.context.current.system_definition);

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

    def set_params(self,rel_tol=None):
        R""" Set parameters for constraint computation.

        Args:
            rel_tol (float): The relative tolerance with which constraint violations are detected (**optional**).

        Example::

            dist = constrain.distance()
            dist.set_params(rel_tol=0.0001)
        """
        if rel_tol is not None:
            self.cpp_force.setRelativeTolerance(float(rel_tol))

class rigid(_constraint_force):
    R""" Constrain particles in rigid bodies.

    .. rubric:: Overview

    Rigid bodies are defined by a single central particle and a number of
    constituent particles. All of these are particles in the HOOMD system configuration and can interact with
    other particles via force computes. The mass and moment of inertia of the central particle set the full
    mass and moment of inertia of the rigid body (constituent particle mass is ignored).

    The central particle is at the center of mass of the rigid body and the orientation quaternion defines the rotation
    from the body space into the simulation box. In body space, the center of mass of the body is at 0,0,0 and the
    moment of inertia is diagonal. You specify the constituent particles to :py:class:`rigid` for each type of body
    in body coordinates. Then, :py:class:`rigid` takes control of those particles, and sets their position and
    orientation in the simulation box relative to the position and orientation of the central particle.
    :py:class:`rigid` also transfers forces and torques from constituent particles to the central
    particle. Then, MD integrators can use these forces and torques to integrate the equations of motion of the
    central particles (representing the whole rigid body) forward in time.

    .. rubric:: Defining bodies

    :py:class:`rigid` accepts one local body environment per body type. The type of a body is the particle type
    of the central particle in that body. In this way, each particle of type *R* in the system configuration defines
    a body of type *R*.

    As a convenience, you do not need to create placeholder entries for all of the constituent particles in your
    initial configuration. You only need to specify the positions and orientations of all the central particles.
    When you call :py:meth:`create_bodies()`, it will create all constituent particles that do not exist. (those
    that already exist e.g. in a restart file are left unchanged).

    Example that creates rigid rods::

        # Place the type R central particles
        uc = hoomd.lattice.unitcell(N = 1,
                                    a1 = [10.8, 0,   0],
                                    a2 = [0,    1.2, 0],
                                    a3 = [0,    0,   1.2],
                                    dimensions = 3,
                                    position = [[0,0,0]],
                                    type_name = ['R'],
                                    mass = [1.0],
                                    moment_inertia = [[0,
                                                       1/12*1.0*8**2,
                                                       1/12*1.0*8**2]],
                                    orientation = [[1, 0, 0, 0]]);
        system = hoomd.init.create_lattice(unitcell=uc, n=[2,18,18]);

        # Add constituent particles of type A and create the rods
        system.particles.types.add('A');
        rigid = hoomd.md.constrain.rigid();
        rigid.set_param('R',
                        types=['A']*8,
                        positions=[(-4,0,0),(-3,0,0),(-2,0,0),(-1,0,0),
                                   (1,0,0),(2,0,0),(3,0,0),(4,0,0)]);

        rigid.create_bodies()

    .. danger:: Automatic creation of constituent particles can change particle tags. If bonds have been defined between
        particles in the initial configuration, or bonds connect to constituent particles, rigid bodies should be
        created manually.

    When you create the constituent particles manually (i.e. in an input file or with snapshots), the central particle
    of a rigid body must have a lower tag than all of its constituent particles. Constituent particles follow in
    monotonically increasing tag order, corresponding to the order they were defined in the argument to
    :py:meth:`set_param()`. The order of central and contiguous particles need **not** to be contiguous.
    Additionally, you must set the ``body`` field for each of the particles in the rigid body to the tag of
    the central particle (for both the central and constituent particles). Set ``body`` to -1 for particles that do
    not belong to a rigid body. After setting an initial configuration that contains properly defined bodies and
    all their constituent particles, call :py:meth:`validate_bodies` to verify that the bodies are defined
    and prepare the constraint.

    You must call either :py:meth:`create_bodies` or :py:meth:`validate_bodies` prior to starting a simulation
    :py:func:`hoomd.run`.

    .. rubric:: Integrating bodies

    Most integrators in HOOMD support the integration of rotational degrees of freedom. When there are rigid bodies
    present in the system, do not apply integrators to the constituent particles, only the central and non-rigid
    particles.

    Example::

        rigid = hoomd.group.rigid_center();
        hoomd.md.integrate.langevin(group=rigid, kT=1.0, seed=42);

    .. rubric:: Thermodynamic quantities of bodies

    HOOMD computes thermodynamic quantities (temperature, kinetic energy, etc...) appropriately when there are rigid
    bodies present in the system. When it does so, it ignores all constituent particles and computes the translational
    and rotational energies of the central particles, which represent the whole body. :py:class:`hoomd.analyze.log`
    can log the translational and rotational energy terms separately.

    .. rubric:: Restarting simulations with rigid bodies.

    To restart, use :py:class:`hoomd.dump.gsd` to write restart files. GSD stores all of the particle data fields
    needed to reconstruct the state of the system, including the body tag, rotational momentum, and orientation
    of the body. Restarting from a gsd file is equivalent to manual constituent particle creation. You still need to
    specify the same local body space environment to :py:class:`rigid` as you did in the earlier simulation.

    """
    def __init__(self):
        hoomd.util.print_status_line();

        # initialize the base class
        _constraint_force.__init__(self);

        self.composite = True

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _md.ForceComposite(hoomd.context.current.system_definition);
        else:
            self.cpp_force = _md.ForceCompositeGPU(hoomd.context.current.system_definition);

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

    def set_param(self,type_name, types, positions, orientations=None, charges=None, diameters=None):
        R""" Set constituent particle types and coordinates for a rigid body.

        Args:
            type_name (str): The type of the central particle
            types (list): List of types of constituent particles
            positions (list): List of relative positions of constituent particles
            orientations (list): List of orientations of constituent particles (**optional**)
            charge (list): List of charges of constituent particles (**optional**)
            diameters (list): List of diameters of constituent particles (**optional**)

        .. caution::
            The constituent particle type must be exist.
            If it does not exist, it can be created on the fly using
            ``system.particles.types.add('A_const')`` (see :py:mod:`hoomd.data`).

        Example::

            rigid = constrain.rigid()
            rigid.set_param('A', types = ['A_const', 'A_const'], positions = [(0,0,1),(0,0,-1)])
            rigid.set_param('B', types = ['B_const', 'B_const'], positions = [(0,0,.5),(0,0,-.5)])

        """
        # get a list of types from the particle data
        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i));

        if type_name not in type_list:
            hoomd.context.msg.error('Type ''{}'' not found.\n'.format(type_name))
            raise RuntimeError('Error setting up parameters for constrain.rigid()')

        type_id = type_list.index(type_name)

        if not isinstance(types, list):
            hoomd.context.msg.error('Expecting list of particle types.\n')
            raise RuntimeError('Error setting up parameters for constrain.rigid()')

        type_vec = _hoomd.std_vector_uint()
        for t in types:
            if t not in type_list:
                hoomd.context.msg.error('Type ''{}'' not found.\n'.format(t))
                raise RuntimeError('Error setting up parameters for constrain.rigid()')
            constituent_type_id = type_list.index(t)

            type_vec.append(constituent_type_id)

        pos_vec = _hoomd.std_vector_scalar3()
        positions_list = list(positions)
        for p in positions_list:
            p = tuple(p)
            if len(p) != 3:
                hoomd.context.msg.error('Particle position is not a coordinate triple.\n')
                raise RuntimeError('Error setting up parameters for constrain.rigid()')
            pos_vec.append(_hoomd.make_scalar3(p[0],p[1],p[2]))

        orientation_vec = _hoomd.std_vector_scalar4()
        if orientations is not None:
            orientations_list = list(orientations)
            for o in orientations_list:
                o = tuple(o)
                if len(o) != 4:
                    hoomd.context.msg.error('Particle orientation is not a 4-tuple.\n')
                    raise RuntimeError('Error setting up parameters for constrain.rigid()')
                orientation_vec.append(_hoomd.make_scalar4(o[0], o[1], o[2], o[3]))
        else:
            for p in positions:
                orientation_vec.append(_hoomd.make_scalar4(1,0,0,0))

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
        self.cpp_force.setParam(type_id, type_vec, pos_vec, orientation_vec, charge_vec, diameter_vec)

    def create_bodies(self, create=True):
        R""" Create copies of rigid bodies.

        Args:
            create (bool): When True, create rigid bodies, otherwise validate existing ones.
        """
        self.cpp_force.validateRigidBodies(create)

    def validate_bodies(self):
        R""" Validate that bodies are well defined and prepare for the simulation run.
        """
        self.cpp_force.validateRigidBodies(False)

    ## \internal
    # \brief updates force coefficients
    def update_coeffs(self):
        # validate copies of rigid bodies
        self.create_bodies(False)

class oneD(_constraint_force):
    R""" Constrain particles to move along a specific direction only

    Args:
        group (:py:mod:`hoomd.group`): Group on which to apply the constraint.
        constraint_vector (list): [x,y,z] list indicating the direction that the particles are restricted to

    :py:class:`oneD` specifies that forces will be applied to all particles in the given group to constrain
    them to only move along a given vector.

    Example::

        constrain.oneD(group=groupA, constraint_vector=[1,0,0])

    .. versionadded:: 2.1
    """
    def __init__(self, group, constraint_vector=[0,0,1]):

        if (constraint_vector[0]**2 + constraint_vector[1]**2 + constraint_vector[2]**2) < 1e-10:
            raise RuntimeError("The one dimension constraint vector is zero");

        constraint_vector = _hoomd.make_scalar3(constraint_vector[0], constraint_vector[1], constraint_vector[2]);

        hoomd.util.print_status_line();

        # initialize the base class
        _constraint_force.__init__(self);

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _md.OneDConstraint(hoomd.context.current.system_definition, group.cpp_group, constraint_vector);
        else:
            self.cpp_force = _md.OneDConstraintGPU(hoomd.context.current.system_definition, group.cpp_group, constraint_vector);

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        # store metadata
        self.group = group
        self.constraint_vector = constraint_vector
        self.metadata_fields = ['group','constraint_vector']



