# -- start license --
# Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
# (HOOMD-blue) Open Source Software License Copyright 2009-2016 The Regents of
# the University of Michigan All rights reserved.

# HOOMD-blue may contain modifications ("Contributions") provided, and to which
# copyright is held, by various Contributors who have granted The Regents of the
# University of Michigan the right to modify and/or distribute such Contributions.

# You may redistribute, use, and create derivate works of HOOMD-blue, in source
# and binary forms, provided you abide by the following conditions:

# * Redistributions of source code must retain the above copyright notice, this
# list of conditions, and the following disclaimer both in the code and
# prominently in any materials provided with the distribution.

# * Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions, and the following disclaimer in the documentation and/or
# other materials provided with the distribution.

# * All publications and presentations based on HOOMD-blue, including any reports
# or published results obtained, in whole or in part, with HOOMD-blue, will
# acknowledge its use according to the terms posted at the time of submission on:
# http://codeblue.umich.edu/hoomd-blue/citations.html

# * Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
# http://codeblue.umich.edu/hoomd-blue/

# * Apart from the above required attributions, neither the name of the copyright
# holder nor the names of HOOMD-blue's contributors may be used to endorse or
# promote products derived from this software without specific prior written
# permission.

# Disclaimer

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
# WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -- end license --

# Maintainer: joaander / All Developers are free to add commands for new features

## \package hoomd.constrain
# \brief Commands that create constraint forces on particles
#
# Constraint forces %constrain a given set of particle to a given surface, to have some relative orientation,
# or impose some other type of constraint. For example, a group of particles can be constrained to the surface of a
# sphere with constrain.sphere.
#
# As with other force commands in hoomd_script, multiple constrain commands can be issued to specify multiple
# constraints, which are additively applied. Note, however, that not all constraints specified in this manner will
# be valid if two separate constrain commands operate on the same particles.
#
# The degrees of freedom removed from the system by constraints are correctly taken into account when computing the
# temperature for thermostatting and logging.
#

from hoomd import _hoomd
from hoomd.md import _md
from hoomd.md import force;
import hoomd;

## \internal
# \brief Base class for constraint forces
#
# A constraint_force in hoomd_script reflects a ForceConstraint in c++. It is responsible
# for all high-level management that happens behind the scenes for hoomd_script
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
        # check if initialization has occured
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
            hoomd.context.msg.error('Bug in hoomd_script: cpp_force not set, please report\n');
            raise RuntimeError();


    ## Disables the force
    #
    # \b Examples:
    # \code
    # force.disable()
    # \endcode
    #
    # Executing the disable command will remove the force from the simulation.
    # Any run() command executed after disabling a force will not calculate or
    # use the force during the simulation. A disabled force can be re-enabled
    # with enable()
    #
    # To use this command, you must have saved the force in a variable, as
    # shown in this example:
    # \code
    # force = constrain.some_force()
    # # ... later in the script
    # force.disable()
    # \endcode
    def disable(self):
        hoomd.util.print_status_line();
        self.check_initialization();

        # check if we are already disabled
        if not self.enabled:
            hoomd.context.msg.warning("Ignoring command to disable a force that is already disabled");
            return;

        self.enabled = False;

        # remove the compute from the system
        hoomd.context.current.system.removeCompute(self.force_name);

    ## Benchmarks the force computation
    # \param n Number of iterations to average the benchmark over
    #
    # \b Examples:
    # \code
    # t = force.benchmark(n = 100)
    # \endcode
    #
    # The value returned by benchmark() is the average time to perform the force
    # computation, in milliseconds. The benchmark is performed by taking the current
    # positions of all particles in the simulation and repeatedly calculating the forces
    # on them. Thus, you can benchmark different situations as you need to by simply
    # running a simulation to achieve the desired state before running benchmark().
    #
    # \note
    # There is, however, one subtle side effect. If the benchmark() command is run
    # directly after the particle data is initialized with an init command, then the
    # results of the benchmark will not be typical of the time needed during the actual
    # simulation. Particles are not reordered to improve cache performance until at least
    # one time step is performed. Executing run(1) before the benchmark will solve this problem.
    #
    # To use this command, you must have saved the force in a variable, as
    # shown in this example:
    # \code
    # force = pair.some_force()
    # # ... later in the script
    # t = force.benchmark(n = 100)
    # \endcode
    def benchmark(self, n):
        self.check_initialization();

        # run the benchmark
        return self.cpp_force.benchmark(int(n))

    ## Enables the force
    #
    # \b Examples:
    # \code
    # force.enable()
    # \endcode
    #
    # See disable() for a detailed description.
    def enable(self):
        hoomd.util.print_status_line();
        self.check_initialization();

        # check if we are already disabled
        if self.enabled:
            hoomd.context.msg.warning("Ignoring command to enable a force that is already enabled");
            return;

        # add the compute back to the system
        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        self.enabled = True;

    ## \internal
    # \brief updates force coefficients
    def update_coeffs(self):
        pass
        # does nothing: this is for derived classes to implement


    ## \internal
    # \brief Get metadata
    def get_metadata(self):
        data = meta._metadata.get_metadata(self)
        data['enabled'] = self.enabled
        if self.name is not "":
            data['name'] = self.name
        return data

# set default counter
_constraint_force.cur_id = 0;

## Constrain particles to the surface of a sphere
#
# The command constrain.sphere specifies that forces will be applied to all particles in the given group to constrain
# them to a sphere. Currently does not work with Brownian or Langevin dynamics (integrate.brownian and
# integrate.langevin).
# \MPI_SUPPORTED
class sphere(_constraint_force):
    ## Specify the %sphere constraint %force
    #
    # \param group Group on which to apply the constraint
    # \param P (x,y,z) tuple indicating the position of the center of the sphere (in distance units)
    # \param r Radius of the sphere (in distance units)
    #
    # \b Examples:
    # \code
    # constrain.sphere(group=groupA, P=(0,10,2), r=10)
    # \endcode
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

## Constrain pairwise particle distances
#
# The command constrain.distance specifies that forces will be applied to all particles pairs for
# which constraints have been defined
#
# The constraint algorithm implemented is described in
#
# [1] M. Yoneya, H. J. C. Berendsen, and K. Hirasawa, "A Non-Iterative Matrix Method for Constraint Molecular Dynamics Simulations," Mol. Simul., vol. 13, no. 6, pp. 395--405, 1994.
# and
# [2] M. Yoneya, "A Generalized Non-iterative Matrix Method for Constraint Molecular Dynamics Simulations," J. Comput. Phys., vol. 172, no. 1, pp. 188--197, Sep. 2001.
#
# In brief, the second derivative of the Lagrange multipliers with resepect to time is set to zero, such
# that both the distance constraints and their time derivatives are conserved within the accuracy of the Velocity
# Verlet scheme, i.e. within \f$ \Delta t^2 \f$. The corresponding linear system of equations is solved.
# Because constraints are satisfied at \f$ t + 2 \Delta t \f$, the scheme is self-correcting and drifts are avoided.
#
# \note In MPI simulations, all particles connected through constraints will be communicated between processors as ghost particles.
# Therefore, if molecules defined by constraints extend over more than half the local domain size, an error is raised.
#
# \warning constrain.distance() does not currently interoperate with integrate.brownian() or integrate.langevin()
#
# \sa hoomd.data.system_data
#
# \MPI_SUPPORTED
class distance(_constraint_force):
    ## Specify the pairwise %distance constraint %force
    #
    # \b Examples:
    # \code
    # constrain.distance()
    # \endcode
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

    ## Set parameters for constraint computation
    #
    # \param rel_tol The relative tolerance with which constraint violations are detected (**optional**)
    # \b Examples:
    # \code
    # dist = constrain.distance()
    # dist.set_params(rel_tol=0.0001)
    def set_params(self,rel_tol=None):
        if rel_tol is not None:
            self.cpp_force.setRelativeTolerance(float(rel_tol))

## Constrain rigid bodies
#
# \MPI_SUPPORTED
class rigid(_constraint_force):
    ## Specify the pairwise %distance constraint %force
    #
    # \b Examples:
    # \code
    # constrain.distance()
    # \endcode
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

        self.create_rigid_bodies = False

    ## Set constituent particle types and coordinates for a rigid body
    #
    # Note: a mirror data structure for bodies in python would be nice OR as a proxy
    def set_param(self,type_name, types, positions, orientations=None):
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
                hoomd.context.msg.error('Type ''{}'' not found.\n'.format(type_name))
                raise RuntimeError('Error setting up parameters for constrain.rigid()')
            constituent_type_id = type_list.index(t)

            type_vec.append(constituent_type_id)

        if not isinstance(positions, list):
            hoomd.context.msg.error('Expecting list of particle positions.\n')
            raise RuntimeError('Error setting up parameters for constrain.rigid()')

        pos_vec = _hoomd.std_vector_scalar3()
        for p in positions:
            if not isinstance(p, tuple) or len(p) != 3:
                hoomd.context.msg.error('Particle position is not a coordinate triple.\n')
                raise RuntimeError('Error setting up parameters for constrain.rigid()')
            pos_vec.append(_hoomd.make_scalar3(p[0],p[1],p[2]))

        orientation_vec = _hoomd.std_vector_scalar4()
        if orientations is not None:
            if not isinstance(orientations, list):
                hoomd.context.msg.error('Expecting list of particle orientations.\n')
                raise RuntimeError('Error setting up parameters for constrain.rigid()')

            for o in orientations:
                if not isinstance(o, tuple()) or len(o) != 4:
                    hoomd.context.msg.error('Particle orientation is not a 4-tuple.\n')
                    raise RuntimeError('Error setting up parameters for constrain.rigid()')
                orientation_vec.append(_hoomd.make_scalar4(o[0], o[1], o[2], o[3]))
        else:
            for p in positions:
                orientation_vec.append(_hoomd.make_scalar4(1,0,0,0))

        # set parameters in C++ force
        self.cpp_force.setParam(type_id, type_vec, pos_vec, orientation_vec)

    ## Set a flag whether to automatically create copies of rigid bodies
    # \param create If true, constituent particles will be created next time run() is called
    def set_auto_create(self,create):
        self.create_rigid_bodies = bool(create)

    ## \internal
    # \brief updates force coefficients
    def update_coeffs(self):
        # validate copies of rigid bodies
        self.cpp_force.validateRigidBodies(self.create_rigid_bodies)
