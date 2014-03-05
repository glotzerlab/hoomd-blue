# -- start license --
# Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
# (HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
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

# Maintainer: joaander

import hoomd
from hoomd_script import globals
from hoomd_script import util

## \package hoomd_script.data
# \brief Access particles, bonds, and other state information inside scripts
#
# Code in the data package provides high-level access to all of the particle, bond and other %data that define the
# current state of the system. By writing python code that modifies this %data, any conceivable initialization of the
# system can be achieved without needing to invoke external tools or generate xml files. Data can be read and additional
# analysis performed during or after simulation runs as well. Basically, the user's imagination is the limit to what can
# be done with the %data.
#
# The only thing to be aware of is that accessing the %data in this way can slow a simulation significantly if performed
# too often. As a general guideline, consider writing a high performance C++ / GPU  plugin (\ref sec_build_plugin)
# if particle %data needs to accessed more often than once every few thousand time steps.
#
# If modifications need to be done on more than just a few particles, e.g.
# setting new positions for all particles, or updating the velocities, etc., \b snapshots can be used.
# \ref data_snapshot store the entire system state in a single (currently opaque) object and can
# be used to re-initialize the system system.restore_snapshot().
#
# <h2>Documentation by example</h2>
#
# For most of the cases below, it is assumed that the result of the initialization command was saved at the beginning
# of the script, like so:
# \code
# system = init.read_xml(filename="input.xml")
# \endcode
#
# <hr>
# <h3>Getting/setting the box</h3>
# You can access the dimensions of the simulation box like so:
# \code
# >>> print system.box
# Box: Lx=17.3646569289 Ly=17.3646569289 Lz=17.3646569289 xy=0.0 xz=0.0 yz=0.0
# \endcode
# and can change it like so:
# \code
# >>> system.box = data.boxdim(Lx=10, Ly=20, Lz=30, xy=1.0, xz=0.1, yz=2.0)
# >>> print system.box
# Box: Lx=10 Ly=20 Lz=30 xy=1.0 xz=0.1 yz=2.0
# \endcode
# \b All particles must \b always remain inside the box. If a box is set in this way such that a particle ends up outside of the box, expect
# errors to be thrown or for hoomd to just crash. The dimensionality of the system cannot change after initialization.
# <hr>
# <h3>Particle properties</h3>
# For a list of all particle properties that can be read and/or set, see the particle_data_proxy. The examples
# here only demonstrate changing a few of them.
#
# With the result of an init command saved in the variable \c system (see above), \c system.particles is a window
# into all of the particles in the system. It behaves like standard python list in many ways.
# - Its length (the number of particles in the system) can be queried
# \code
# >>> len(system.particles)
# 64000
# \endcode
# - A short summary can be printed of the list
# \code
# >>> print system.particles
# Particle Data for 64000 particles of 1 type(s)
# \endcode
# - The list of all particle types in the simulation can be accessed
# \code
# >>> print system.particles.types
# ['A']
# \endcode
# - Individual particles can be accessed at random.
# \code
# >>> i = 4
# >>> p = system.particles[i]
# \endcode
# - Various properties can be accessed of any particle
# \code
# >>> p.tag
# 4
# >>> p.position
# (27.296911239624023, -3.5986068248748779, 10.364067077636719)
# >>> p.velocity
# (-0.60267972946166992, 2.6205904483795166, -1.7868227958679199)
# >>> p.mass
# 1.0
# >>> p.diameter
# 1.0
# >>> p.type
# 'A'
# \endcode
# (note that p can be replaced with system.particles.[i] above and the results are the same)
# - Particle properties can be set in the same way:
# \code
# >>> p.position = (1,2,3)
# >>> p.position
# (1.0, 2.0, 3.0)
# \endcode
# - Finally, all particles can be easily looped over
# \code
# for p in system.particles:
#     p.velocity = (0,0,0)
# \endcode
#
# Performance is decent, but not great. The for loop above that sets all velocities to 0 takes 0.86 seconds to execute
# on a 2.93 GHz core2 iMac. The interface has been designed to be flexible and easy to use for the widest variety of
# initialization tasks, not efficiency.
# For doing modifications that operate on the whole system data efficiently, snapshots have been
# designed. Their usage is described below.
#
# There is a second way to access the particle data. Any defined group can be used in exactly the same way as
# \c system.particles above, only the particles accessed will be those just belonging to the group. For a specific
# example, the following will set the velocity of all particles of type A to 0.
# \code
# groupA = group.type(name="a-particles", type='A')
# for p in groupA:
#     p.velocity = (0,0,0)
# \endcode
#
# <hr>
# <h3>Rigid Body Data</h3>
# Rigid Body data can be accessed via the body_data_proxy.  Here are examples
#
#\code
#
# >>> b = system.bodies[0]
# >>> print b
#num_particles    : 5
#mass             : 5.0
# COM              : (0.33264800906181335, -2.495814800262451, -1.2669427394866943)
# velocity         : (0.0, 0.0, 0.0)
# orientation      : (0.9244732856750488, -0.3788720965385437, -0.029276784509420395, 0.0307924821972847)
# angular_momentum (space frame) : (0.0, 0.0, 0.0)
# moment_inertia: (10.000000953674316, 10.0, 0.0)
# particle_tags    : [0, 1, 2, 3, 4]
# particle_disp    : [[-3.725290298461914e-09, -4.172325134277344e-07, 2.0], [-2.421438694000244e-08, -2.086162567138672e-07, 0.9999998211860657], [-2.6206091519043184e-08, -2.073889504572435e-09, -3.361484459674102e-07], [-5.029141902923584e-08, 2.682209014892578e-07, -1.0000004768371582], [-3.3527612686157227e-08, -2.980232238769531e-07, -2.0]]
# >>> print b.COM
# (0.33264800906181335, -2.495814800262451, -1.2669427394866943)
# >>> b.particle_disp = [[0,0,0], [0,0,0], [0,0,0.0], [0,0,0], [0,0,0]]
#
#\endcode
#
# <hr>
# <h3>Bond Data</h3>
# Bonds may be added at any time in the job script.
# \code
# >>> system.bonds.add("bondA", 0, 1)
# >>> system.bonds.add("bondA", 1, 2)
# >>> system.bonds.add("bondA", 2, 3)
# >>> system.bonds.add("bondA", 3, 4)
# \endcode
#
# Individual bonds may be accessed by index.
# \code
# >>> bnd = system.bonds[0]
# >>> print bnd
# tag          : 0
# typeid       : 0
# a            : 0
# b            : 1
# type         : bondA
# >>> print bnd.type
# bondA
# >>> print bnd.a
# 0
# >>> print bnd.b
#1
# \endcode
# \note The order in which bonds appear by index is not static and may change at any time!
#
# Bonds may be deleted by index.
# \code
# >>> del system.bonds[0]
# >>> print system.bonds[0]
# tag          : 3
# typeid       : 0
# a            : 3
# b            : 4
# type         : bondA
# \endcode
# \note Regarding the previous note: see how the last bond added is now at index 0. No guarantee is made about how the
# order of bonds by index will or will not change, so do not write any job scripts which assume a given ordering.
#
# To access bonds in an index-independent manner, use their tags. For example, to delete all bonds which connect to
# particle 2, first loop through the bonds and build a list of bond tags that match the criteria.
# \code
# tags = []
# for b in system.bonds:
#     if b.a == 2 or b.b == 2:
#         tags.append(b.tag)
# \endcode
# Then remove each of the bonds by their unique tag.
# \code
# for t in tags:
#     system.bonds.remove(t)
# \endcode
#
# <hr>
# <h3>Angle, Dihedral, and Improper Data</h3>
# Angles, Dihedrals, and Impropers may be added at any time in the job script.
# \code
# >>> system.angles.add("angleA", 0, 1, 2)
# >>> system.dihedrals.add("dihedralA", 1, 2, 3, 4)
# >>> system.impropers.add("dihedralA", 2, 3, 4, 5)
# \endcode
#
# Individual angles, dihedrals, and impropers may be accessed, deleted by index or removed by tag with the same syntax
# as described for bonds, just replace \em bonds with \em angles, \em dihedrals, or, \em impropers and access the
# appropriate number of tag elements (a,b,c for angles) (a,b,c,d for dihedrals/impropers).
# <hr>
#
# <h3>Forces</h3>
# Forces can be accessed in a similar way.
# \code
# >>> lj = pair.lj(r_cut=3.0)
# >>> lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
# >>> print lj.forces[0]
# tag         : 0
# force       : (-0.077489577233791351, -0.029512746259570122, -0.13215918838977814)
# virial      : -0.0931386947632
# energy      : -0.0469368174672
# >>> f0 = lj.forces[0]
# >>> print f0.force
# (-0.077489577233791351, -0.029512746259570122, -0.13215918838977814)
# >>> print f0.virial
# -0.0931386947632
# >>> print f0.energy
# -0.0469368174672
# \endcode
#
# In this manner, forces due to the lj %pair %force, bonds, and any other %force commands in hoomd can be accessed
# independently from one another. See force_data_proxy for a definition of each parameter accessed.
#
# <hr>
# <h3>Proxy references</h3>
#
# For advanced code using the particle data access from python, it is important to understand that the hoomd_script
# particles, forces, bonds, et cetera, are accessed as proxies. This means that after
# \code
# p = system.particles[i]
# \endcode
# is executed, \a p \b doesn't store the position, velocity, ... of particle \a i. Instead, it just stores \a i and
# provides an interface to get/set the properties on demand. This has some side effects. They aren't necessarily
# bad side effects, just some to be aware of.
# - First, it means that \a p (or any other proxy reference) always references the current state of the particle.
# As an example, note how the position of particle p moves after the run() command.
# \code
# >>> p.position
# (-21.317455291748047, -23.883811950683594, -22.159387588500977)
# >>> run(1000)
# ** starting run **
# ** run complete **
# >>> p.position
# (-19.774742126464844, -23.564577102661133, -21.418502807617188)
# \endcode
# - Second, it means that copies of the proxy reference cannot be changed independently.
# \code
# p.position
# >>> a = p
# >>> a.position
# (-19.774742126464844, -23.564577102661133, -21.418502807617188)
# >>> p.position = (0,0,0)
# >>> a.position
# (0.0, 0.0, 0.0)
# \endcode
#
# If you need to store some particle properties at one time in the simulation and access them again later, you will need
# to make copies of the actual property values themselves and not of the proxy references.
#
# \section data_snapshot Snapshots
# <hr>
# <h3>Snapshots</h3>
#
# A snaphot of the current system state is obtained using system_data.take_snapshot(). It contains information
# about the simulation box, particles, bonds, angles, dihedrals, impropers, walls and rigid bodies.
# Once taken, it is not updated anymore (as opposed to the particle %data proxies, which always
# return the current state). Instead, it can be used to restart the simulation
# using system.restore_snapshot().
#
# In future releases it will be possible to modify or %analyze the contents of a snapshot.
#
# Example for taking a snapshot:
# \code
# snapshot = system.take_snapshot(all=True)
# \endcode

## Define box dimensions
#
# Simulation boxes in hoomd are specified by six parameters, *Lx*, *Ly*, *Lz*, *xy*, *xz* and *yz*. For full details,
# see \ref page_box. A boxdim provides a way to specify all six parameters for a given box and perform some common
# operations with them. Modifying a boxdim does not modify the underlying simulation box in hoomd. A boxdim can be passed
# to an initialization method or to assigned to a saved system.
#
# boxdim parameters may be accessed directly.
# ~~~~
# b = data.boxdim(L=20);
# b.xy = 1.0;
# b.yz = 0.5;
# b.Lz = 40;
# ~~~~
#
# **Two dimensional systems**
#
# 2D simulations in hoomd are embedded in 3D boxes with short heights in the z direction. To create a 2D box,
# set dimensions=2 when creating the boxdim. This will force Lz=1 and xz=yz=0. init commands that support 2D boxes
# will pass the dimensionality along to the system. When you assign a new boxdim to an already initialized system,
# the dimensionality flag is ignored. Changing the number of dimensions during a simulation run is not supported.
#
# In 2D boxes, "volume" refers to area.
#
class boxdim:
    ## Initialize a boxdim object
    #
    # \param Lx box extent in the x direction (distance units)
    # \param Ly box extent in the y direction (distance units)
    # \param Lz box extent in the z direction (distance units)
    # \param xy tilt factor xy (dimensionless)
    # \param xz tilt factor xz (dimensionless)
    # \param yz tilt factor yz (dimensionless)
    # \param dimensions Number of dimensions in the box (2 or 3).
    # \param L shorthand for specifying Lx=Ly=Lz=L (distance units)
    # \param volume Scale the given box dimensions up to the this volume (area if dimensions=2)
    #
    def __init__(self, Lx=1.0, Ly=1.0, Lz=1.0, xy=0.0, xz=0.0, yz=0.0, dimensions=3, L=None, volume=None):
        if L is not None:
            Lx = L;
            Ly = L;
            Lz = L;

        if dimensions == 2:
            Lz = 1.0;
            xz = yz = 0.0;

        self.Lx = Lx;
        self.Ly = Ly;
        self.Lz = Lz;
        self.xy = xy;
        self.xz = xz;
        self.yz = yz;
        self.dimensions = dimensions;

        if volume is not None:
            self.set_volume(volume);

    ## Scale box dimensions
    #
    # \param sx scale factor in the x direction
    # \param sy scale factor in the y direction
    # \param sz scale factor in the z direction
    #
    # Scales the box by the given scale factors. Tilt factors are not modified.
    #
    def scale(self, sx, sy, sz):
        self.Lx = self.Lx * sx;
        self.Ly = self.Ly * sy;
        self.Lz = self.Lz * sz;

    ## Set the box volume
    #
    # \param volume new box volume (area if dimensions=2)
    #
    # setVolume() scales the box to the given volume (or area).
    #
    def set_volume(self, volume):
        cur_vol = self.get_volume();

        if self.dimensions == 3:
            s = (volume / cur_vol)**(1.0/3.0)
            self.scale(s, s, s);
        else:
            s = (volume / cur_vol)**(1.0/2.0)
            self.scale(s, s, 1.0);

    ## Get the box volume
    #
    # Returns the box volume (area in 2D).
    #
    def get_volume(self):
        b = self._getBoxDim();
        return b.getVolume(self.dimensions == 2);

    ## \internal
    # \brief Get a C++ boxdim
    def _getBoxDim(self):
        b = hoomd.BoxDim(self.Lx, self.Ly, self.Lz);
        b.setTiltFactors(self.xy, self.xz, self.yz);
        return b

    def __str__(self):
        return 'Box: Lx=' + str(self.Lx) + ' Ly=' + str(self.Ly) + ' Lz=' + str(self.Lz) + ' xy=' + str(self.xy) + \
                    ' xz='+ str(self.xz) + ' yz=' + str(self.yz);
##
# \brief Access system data
#
# system_data provides access to the different data structures that define the current state of the simulation.
# This documentation is intentionally left sparse, see hoomd_script.data for a full explanation of how to use
# system_data, documented by example.
#
class system_data:
    ## \internal
    # \brief create a system_data
    #
    # \param sysdef SystemDefinition to connect
    def __init__(self, sysdef):
        self.sysdef = sysdef;
        self.particles = particle_data(sysdef.getParticleData());
        self.bonds = bond_data(sysdef.getBondData());
        self.angles = angle_data(sysdef.getAngleData());
        self.dihedrals = dihedral_data(sysdef.getDihedralData());
        self.impropers = dihedral_data(sysdef.getImproperData());
        self.bodies = body_data(sysdef.getRigidData());

    ## Take a snapshot of the current system data
    #
    # This functions returns a snapshot object. It contains the current
    # partial or complete simulation state. With appropriate options
    # it is possible to select which data properties should be included
    # in the snapshot.
    #
    # \param particles If true, particle data is included in the snapshot
    # \param bonds If true, bond data is included in the snapshot
    # \param angles If true, angle data is included in the snapshot
    # \param dihedrals If true, dihedral data is included in the snapshot
    # \param impropers If true, dihedral data is included in the snapshot
    # \param rigid_bodies If true, rigid body data is included in the snapshot
    # \param walls If true, wall data is included in the snapshot
    # \param integrators If true, integrator data is included the snapshot
    # \param all If true, the entire system state is saved in the snapshot
    #
    # Specific options (such as \b particles=True) take precedence over \b all=True.
    #
    # \returns the snapshot object.
    #
    # \code
    # snapshot = system.take_snapshot()
    # snapshot = system.take_snapshot(particles=true)
    # snapshot = system.take_snapshot(bonds=true)
    # \endcode
    #
    # \MPI_SUPPORTED
    def take_snapshot(self,particles=None,bonds=None,angles=None,dihedrals=None, impropers=None, rigid_bodies=None, walls=None, integrators=None, all=None ):
        util.print_status_line();

        if all is True:
            if particles is None:
                particles=True
            if bonds is None:
                bonds=True
            if angles is None:
                angles=True
            if dihedrals is None:
                dihedrals=True
            if impropers is None:
                impropers=True
            if rigid_bodies is None:
                rigid_bodies=True
            if walls is None:
                walls=True
            if integrators is None:
                integrators=True

        if particles is None and not all:
            particles = False
        if bonds is None and not all:
            bonds = False
        if angles is None and not all:
            angles = False
        if dihedrals is None and not all:
            dihedrals = False
        if impropers is None and not all:
            impropers = False
        if rigid_bodies is None and not all:
            rigid_bodies = False
        if walls is None and not all:
            walls = False
        if integrators is None and not all:
            integrators = False

        if not (particles or bonds or angles or dihedrals or impropers or rigid_bodies or walls or integrators):
            globals.msg.warning("No options specified. Ignoring request to create an empty snapshot.\n")
            return None

        # take the snapshot
        cpp_snapshot = self.sysdef.takeSnapshot(particles,bonds,angles,dihedrals,impropers,rigid_bodies,walls,integrators)

        return cpp_snapshot

    ## Re-initializes the system from a snapshot
    #
    # \param snapshot The snapshot to initialize the system from
    #
    # Snapshots temporarily store system %data. Snapshots contain the complete simulation state in a
    # single object. They can be used to restart a simulation.
    #
    # Example use cases in which a simulation may be restarted from a snapshot include python-script-level
    # \b Monte-Carlo schemes, where the system state is stored after a move has been accepted (according to
    # some criterium), and where the system is re-initialized from that same state in the case
    # when a move is not accepted.
    #
    # Example for the procedure of taking a snapshot and re-initializing from it:
    # \code
    # system = init.read_xml("some_file.xml")
    #
    # ... run a simulation ...
    #
    # snapshot = system.take_snapshot(all=True)
    # ...
    # system.restore_snapshot(snapshot)
    # \endcode
    #
    # \sa hoomd_script.data
    # \MPI_SUPPORTED
    def restore_snapshot(self, snapshot):
        util.print_status_line();

        self.sysdef.initializeFromSnapshot(snapshot);

    ## \var sysdef
    # \internal
    # \brief SystemDefinition to which this instance is connected

    ## \internal
    # \brief Translate attribute accesses into the low level API function calls
    def __setattr__(self, name, value):
        if name == "box":
            if not isinstance(value, boxdim):
                raise TypeError('box must be a data.boxdim object');
            self.sysdef.getParticleData().setGlobalBox(value._getBoxDim());

        # otherwise, consider this an internal attribute to be set in the normal way
        self.__dict__[name] = value;

    ## \internal
    # \brief Translate attribute accesses into the low level API function calls
    def __getattr__(self, name):
        if name == "box":
            b = self.sysdef.getParticleData().getGlobalBox();
            L = b.getL();
            return boxdim(Lx=L.x, Ly=L.y, Lz=L.z, xy=b.getTiltFactorXY(), xz=b.getTiltFactorXZ(), yz=b.getTiltFactorYZ());

        # if we get here, we haven't found any names that match, post an error
        raise AttributeError;

## \internal
# \brief Access particle data
#
# particle_data provides access to the per-particle data of all particles in the system.
# This documentation is intentionally left sparse, see hoomd_script.data for a full explanation of how to use
# particle_data, documented by example.
#
class particle_data:
    ## \internal
    # \brief particle_data iterator
    class particle_data_iterator:
        def __init__(self, data):
            self.data = data;
            self.index = 0;
        def __iter__(self):
            return self;
        def __next__(self):
            if self.index == len(self.data):
                raise StopIteration;

            result = self.data[self.index];
            self.index += 1;
            return result;

        # support python2
        next = __next__;

    ## \internal
    # \brief create a particle_data
    #
    # \param pdata ParticleData to connect
    def __init__(self, pdata):
        self.pdata = pdata;

        ntypes = globals.system_definition.getParticleData().getNTypes();
        self.types = [];
        for i in range(0,ntypes):
            self.types.append(globals.system_definition.getParticleData().getNameByType(i));

    ## \var pdata
    # \internal
    # \brief ParticleData to which this instance is connected

    ## \internal
    # \brief Get a particle_proxy reference to the particle with tag \a tag
    # \param tag Particle tag to access
    def __getitem__(self, tag):
        if tag >= len(self) or tag < 0:
            raise IndexError;
        return particle_data_proxy(self.pdata, tag);

    ## \internal
    # \brief Set a particle's properties
    # \param tag Particle tag to set
    # \param p Value containing properties to set
    def __setitem__(self, tag, p):
        raise RuntimeError('__setitem__ not implemented');

    ## \internal
    # \brief Get the number of particles
    def __len__(self):
        return self.pdata.getNGlobal();

    ## \internal
    # \brief Get an informal string representing the object
    def __str__(self):
        result = "Particle Data for %d particles of %d type(s)" % (self.pdata.getNGlobal(), self.pdata.getNTypes());
        return result

    ## \internal
    # \brief Return an interator
    def __iter__(self):
        return particle_data.particle_data_iterator(self);

## Access a single particle via a proxy
#
# particle_data_proxy provides access to all of the properties of a single particle in the system.
# This documentation is intentionally left sparse, see hoomd_script.data for a full explanation of how to use
# particle_data_proxy, documented by example.
#
# The following attributes are read only:
# - \c tag          : An integer indexing the particle in the system. Tags run from 0 to N-1;
# - \c acceleration : A 3-tuple of floats   (x, y, z) Note that acceleration is a calculated quantity and cannot be set. (in acceleration units)
# - \c typeid       : An integer defining the type id
#
# The following attributes can be both read and set
# - \c position     : A 3-tuple of floats   (x, y, z) (in distance units)
# - \c image        : A 3-tuple of integers (x, y, z)
# - \c velocity     : A 3-tuple of floats   (x, y, z) (in velocity units)
# - \c charge       : A single float
# - \c mass         : A single float (in mass units)
# - \c diameter     : A single float (in distance units)
# - \c type         : A string naming the type
# - \c body         : Rigid body id integer (-1 for free particles)
# - \c orientation  : Orientation of anisotropic particle (quaternion)
# - \c net_force    : Net force on particle (x, y, z) (in force units)
# - \c net_energy   : Net contribution of particle to the potential energy (in energy units)
# - \c net_torque   : Net torque on the particle (x, y, z) (in torque units)
#
# In the current version of the API, only already defined type names can be used. A future improvement will allow
# dynamic creation of new type names from within the python API.
#
class particle_data_proxy:
    ## \internal
    # \brief create a particle_data_proxy
    #
    # \param pdata ParticleData to which this proxy belongs
    # \param tag Tag of this particle in \a pdata
    def __init__(self, pdata, tag):
        self.pdata = pdata;
        self.tag = tag;

    ## \internal
    # \brief Get an informal string representing the object
    def __str__(self):
        result = "";
        result += "tag         : " + str(self.tag) + "\n"
        result += "position    : " + str(self.position) + "\n";
        result += "image       : " + str(self.image) + "\n";
        result += "velocity    : " + str(self.velocity) + "\n";
        result += "acceleration: " + str(self.acceleration) + "\n";
        result += "charge      : " + str(self.charge) + "\n";
        result += "mass        : " + str(self.mass) + "\n";
        result += "diameter    : " + str(self.diameter) + "\n";
        result += "type        : " + str(self.type) + "\n";
        result += "typeid      : " + str(self.typeid) + "\n";
        result += "body        : " + str(self.body) + "\n";
        result += "orientation : " + str(self.orientation) + "\n";
        result += "net_force   : " + str(self.net_force) + "\n";
        result += "net_energy  : " + str(self.net_energy) + "\n";
        result += "net_torque  : " + str(self.net_torque) + "\n";
        return result;

    ## \internal
    # \brief Translate attribute accesses into the low level API function calls
    def __getattr__(self, name):
        if name == "position":
            pos = self.pdata.getPosition(self.tag);
            return (pos.x, pos.y, pos.z);
        if name == "velocity":
            vel = self.pdata.getVelocity(self.tag);
            return (vel.x, vel.y, vel.z);
        if name == "acceleration":
            accel = self.pdata.getAcceleration(self.tag);
            return (accel.x, accel.y, accel.z);
        if name == "image":
            image = self.pdata.getImage(self.tag);
            return (image.x, image.y, image.z);
        if name == "charge":
            return self.pdata.getCharge(self.tag);
        if name == "mass":
            return self.pdata.getMass(self.tag);
        if name == "diameter":
            return self.pdata.getDiameter(self.tag);
        if name == "typeid":
            return self.pdata.getType(self.tag);
        if name == "body":
            return self.pdata.getBody(self.tag);
        if name == "type":
            typeid = self.pdata.getType(self.tag);
            return self.pdata.getNameByType(typeid);
        if name == "orientation":
            o = self.pdata.getOrientation(self.tag);
            return (o.x, o.y, o.z, o.w);
        if name == "net_force":
            f = self.pdata.getPNetForce(self.tag);
            return (f.x, f.y, f.z);
        if name == "net_energy":
            f = self.pdata.getPNetForce(self.tag);
            return f.w;
        if name == "net_torque":
            f = self.pdata.getNetTorque(self.tag);
            return (f.x, f.y, f.z);

        # if we get here, we haven't found any names that match, post an error
        raise AttributeError;

    ## \internal
    # \brief Translate attribute accesses into the low level API function calls
    def __setattr__(self, name, value):
        if name == "position":
            v = hoomd.Scalar3();
            v.x = float(value[0]);
            v.y = float(value[1]);
            v.z = float(value[2]);
            self.pdata.setPosition(self.tag, v, True);
            return;
        if name == "velocity":
            v = hoomd.Scalar3();
            v.x = float(value[0]);
            v.y = float(value[1]);
            v.z = float(value[2]);
            self.pdata.setVelocity(self.tag, v);
            return;
        if name == "image":
            v = hoomd.int3();
            v.x = int(value[0]);
            v.y = int(value[1]);
            v.z = int(value[2]);
            self.pdata.setImage(self.tag, v);
            return;
        if name == "charge":
            self.pdata.setCharge(self.tag, float(value));
            return;
        if name == "mass":
            self.pdata.setMass(self.tag, float(value));
            return;
        if name == "diameter":
            self.pdata.setDiameter(self.tag, value);
            return;
        if name == "body":
            self.pdata.setBody(self.tag, value);
            return;
        if name == "type":
            typeid = self.pdata.getTypeByName(value);
            self.pdata.setType(self.tag, typeid);
            return;
        if name == "typeid":
            raise AttributeError;
        if name == "acceleration":
            raise AttributeError;
        if name == "orientation":
            o = hoomd.Scalar4();
            o.x = float(value[0]);
            o.y = float(value[1]);
            o.z = float(value[2]);
            o.w = float(value[3]);
            self.pdata.setOrientation(self.tag, o);
            return;
        if name == "net_force":
            raise AttributeError;
        if name == "net_energy":
            raise AttributeError;

        # otherwise, consider this an internal attribute to be set in the normal way
        self.__dict__[name] = value;

## \internal
# Access force data
#
# force_data provides access to the per-particle data of all forces in the system.
# This documentation is intentionally left sparse, see hoomd_script.data for a full explanation of how to use
# force_data, documented by example.
#
class force_data:
    ## \internal
    # \brief force_data iterator
    class force_data_iterator:
        def __init__(self, data):
            self.data = data;
            self.index = 0;
        def __iter__(self):
            return self;
        def __next__(self):
            if self.index == len(self.data):
                raise StopIteration;

            result = self.data[self.index];
            self.index += 1;
            return result;

        # support python2
        next = __next__;

    ## \internal
    # \brief create a force_data
    #
    # \param pdata ParticleData to connect
    def __init__(self, force):
        self.force = force;

    ## \var force
    # \internal
    # \brief ForceCompute to which this instance is connected

    ## \internal
    # \brief Get a force_proxy reference to the particle with tag \a tag
    # \param tag Particle tag to access
    def __getitem__(self, tag):
        if tag >= len(self) or tag < 0:
            raise IndexError;
        return force_data_proxy(self.force, tag);

    ## \internal
    # \brief Set a particle's properties
    # \param tag Particle tag to set
    # \param p Value containing properties to set
    def __setitem__(self, tag, p):
        raise RuntimeError('__setitem__ not implemented');

    ## \internal
    # \brief Get the number of particles
    def __len__(self):
        return globals.system_definition.getParticleData().getNGlobal();

    ## \internal
    # \brief Get an informal string representing the object
    def __str__(self):
        result = "Force Data for %d particles" % (len(self));
        return result

    ## \internal
    # \brief Return an interator
    def __iter__(self):
        return force_data.force_data_iterator(self);

## Access the %force on a single particle via a proxy
#
# force_data_proxy provides access to the current %force, virial, and energy of a single particle due to a single
# %force computations.
#
# This documentation is intentionally left sparse, see hoomd_script.data for a full explanation of how to use
# force_data_proxy, documented by example.
#
# The following attributes are read only:
# - \c %force         : A 3-tuple of floats (x, y, z) listing the current %force on the particle
# - \c virial         : A float containing the contribution of this particle to the total virial
# - \c energy         : A float containing the contribution of this particle to the total potential energy
# - \c torque         : A 3-tuple of floats (x, y, z) listing the current torque on the particle
#
class force_data_proxy:
    ## \internal
    # \brief create a force_data_proxy
    #
    # \param force ForceCompute to which this proxy belongs
    # \param tag Tag of this particle in \a force
    def __init__(self, force, tag):
        self.fdata = force;
        self.tag = tag;

    ## \internal
    # \brief Get an informal string representing the object
    def __str__(self):
        result = "";
        result += "tag         : " + str(self.tag) + "\n"
        result += "force       : " + str(self.force) + "\n";
        result += "virial      : " + str(self.virial) + "\n";
        result += "energy      : " + str(self.energy) + "\n";
        result += "torque      : " + str(self.torque) + "\n";
        return result;

    ## \internal
    # \brief Translate attribute accesses into the low level API function calls
    def __getattr__(self, name):
        if name == "force":
            f = self.fdata.cpp_force.getForce(self.tag);
            return (f.x, f.y, f.z);
        if name == "virial":
            return (self.fdata.cpp_force.getVirial(self.tag,0),
                    self.fdata.cpp_force.getVirial(self.tag,1),
                    self.fdata.cpp_force.getVirial(self.tag,2),
                    self.fdata.cpp_force.getVirial(self.tag,3),
                    self.fdata.cpp_force.getVirial(self.tag,4),
                    self.fdata.cpp_force.getVirial(self.tag,5));
        if name == "energy":
            energy = self.fdata.cpp_force.getEnergy(self.tag);
            return energy;
        if name == "torque":
            f = self.fdata.cpp_force.getTorque(self.tag);
            return (f.x, f.y, f.z)

        # if we get here, we haven't found any names that match, post an error
        raise AttributeError;

## \internal
# \brief Access bond data
#
# bond_data provides access to the bonds in the system.
# This documentation is intentionally left sparse, see hoomd_script.data for a full explanation of how to use
# bond_data, documented by example.
#
class bond_data:
    ## \internal
    # \brief bond_data iterator
    class bond_data_iterator:
        def __init__(self, data):
            self.data = data;
            self.tag = 0;
        def __iter__(self):
            return self;
        def __next__(self):
            if self.tag == len(self.data):
                raise StopIteration;

            result = self.data[self.tag];
            self.tag += 1;
            return result;

        # support python2
        next = __next__;

    ## \internal
    # \brief create a bond_data
    #
    # \param bdata BondData to connect
    def __init__(self, bdata):
        self.bdata = bdata;

    ## \internal
    # \brief Add a new bond
    # \param type Type name of the bond to add
    # \param a Tag of the first particle in the bond
    # \param b Tag of the second particle in the bond
    # \returns Unique tag identifying this bond
    def add(self, type, a, b):
        typeid = self.bdata.getTypeByName(type);
        return self.bdata.addBondedGroup(hoomd.Bond(typeid, a, b));

    ## \internal
    # \brief Remove a bond by tag
    # \param tag Unique tag of the bond to remove
    def remove(self, tag):
        self.bdata.removeBondedGroup(tag);

    ## \var bdata
    # \internal
    # \brief BondData to which this instance is connected

    ## \internal
    # \brief Get a bond_proxy reference to the bond with id \a id
    # \param id Bond id to access
    def __getitem__(self, tag):
        if tag >= len(self) or tag < 0:
            raise IndexError;
        return bond_data_proxy(self.bdata, tag);

    ## \internal
    # \brief Set a bond's properties
    # \param id Bond id to set
    # \param b Value containing properties to set
    def __setitem__(self, id, b):
        raise RuntimeError('Cannot change bonds once they are created');

    ## \internal
    # \brief Delete a bond by id
    # \param id Bond id to delete
    def __delitem__(self, id):
        if id >= len(self) or id < 0:
            raise IndexError;
        tag = self.bdata.getNthTag(id);
        self.bdata.removeBond(tag);

    ## \internal
    # \brief Get the number of bonds
    def __len__(self):
        return self.bdata.getNGlobal();

    ## \internal
    # \brief Get an informal string representing the object
    def __str__(self):
        result = "Bond Data for %d bonds of %d typeid(s)" % (self.bdata.getNGlobal(), self.bdata.getNBondTypes());
        return result

    ## \internal
    # \brief Return an interator
    def __iter__(self):
        return bond_data.bond_data_iterator(self);

## Access a single bond via a proxy
#
# bond_data_proxy provides access to all of the properties of a single bond in the system.
# This documentation is intentionally left sparse, see hoomd_script.data for a full explanation of how to use
# bond_data_proxy, documented by example.
#
# The following attributes are read only:
# - \c tag          : A unique integer attached to each bond (not in any particular range). A bond's tag remans fixed
#                     during its lifetime. (Tags previously used by removed bonds may be recycled).
# - \c typeid       : An integer indexing the bond type of the bond.
# - \c a            : An integer indexing the A particle in the bond. Particle tags run from 0 to N-1;
# - \c b            : An integer indexing the B particle in the bond. Particle tags run from 0 to N-1;
# - \c type         : A string naming the type
#
# In the current version of the API, only already defined type names can be used. A future improvement will allow
# dynamic creation of new type names from within the python API.
# \MPI_SUPPORTED
class bond_data_proxy:
    ## \internal
    # \brief create a bond_data_proxy
    #
    # \param bdata BondData to which this proxy belongs
    # \param id index of this bond in \a bdata (at time of proxy creation)
    def __init__(self, bdata, tag):
        self.bdata = bdata;
        self.tag = tag

    ## \internal
    # \brief Get an informal string representing the object
    def __str__(self):
        result = "";
        result += "typeid       : " + str(self.typeid) + "\n";
        result += "a            : " + str(self.a) + "\n"
        result += "b            : " + str(self.b) + "\n"
        result += "type         : " + str(self.type) + "\n";
        return result;

    ## \internal
    # \brief Translate attribute accesses into the low level API function calls
    def __getattr__(self, name):
        if name == "a":
            bond = self.bdata.getGroupByTag(self.tag);
            return bond.a;
        if name == "b":
            bond = self.bdata.getGroupByTag(self.tag);
            return bond.b;
        if name == "typeid":
            bond = self.bdata.getGroupByTag(self.tag);
            return bond.type;
        if name == "type":
            bond = self.bdata.getGroupByTag(self.tag);
            typeid = bond.type;
            return self.bdata.getNameByType(typeid);

        # if we get here, we haven't found any names that match, post an error
        raise AttributeError;

    ## \internal
    # \brief Translate attribute accesses into the low level API function calls
    def __setattr__(self, name, value):
        if name == "a":
            raise AttributeError;
        if name == "b":
            raise AttributeError;
        if name == "type":
            raise AttributeError;
        if name == "typeid":
            raise AttributeError;

        # otherwise, consider this an internal attribute to be set in the normal way
        self.__dict__[name] = value;

## \internal
# \brief Access angle data
#
# angle_data provides access to the angles in the system.
# This documentation is intentionally left sparse, see hoomd_script.data for a full explanation of how to use
# angle_data, documented by example.
#
class angle_data:
    ## \internal
    # \brief angle_data iterator
    class angle_data_iterator:
        def __init__(self, data):
            self.data = data;
            self.index = 0;
        def __iter__(self):
            return self;
        def __next__(self):
            if self.index == len(self.data):
                raise StopIteration;

            result = self.data[self.index];
            self.index += 1;
            return result;

        # support python2
        next = __next__;

    ## \internal
    # \brief create a angle_data
    #
    # \param bdata AngleData to connect
    def __init__(self, adata):
        self.adata = adata;

    ## \internal
    # \brief Add a new angle
    # \param type Type name of the angle to add
    # \param a Tag of the first particle in the angle
    # \param b Tag of the second particle in the angle
    # \param c Tag of the thrid particle in the angle
    # \returns Unique tag identifying this bond
    def add(self, type, a, b, c):
        typeid = self.adata.getTypeByName(type);
        return self.adata.addBondedGroup(hoomd.Angle(typeid, a, b, c));

    ## \internal
    # \brief Remove an angle by tag
    # \param tag Unique tag of the angle to remove
    def remove(self, tag):
        self.adata.removeBondedGroup(tag);

    ## \var adata
    # \internal
    # \brief AngleData to which this instance is connected

    ## \internal
    # \brief Get anm angle_proxy reference to the bond with id \a id
    # \param id Angle id to access
    def __getitem__(self, id):
        if id >= len(self) or id < 0:
            raise IndexError;
        return angle_data_proxy(self.adata, id);

    ## \internal
    # \brief Set an angle's properties
    # \param id Angle id to set
    # \param b Value containing properties to set
    def __setitem__(self, id, b):
        raise RuntimeError('Cannot change angles once they are created');

    ## \internal
    # \brief Delete an angle by id
    # \param id Angle id to delete
    def __delitem__(self, id):
        if id >= len(self) or id < 0:
            raise IndexError;

        # Get the tag of the bond to delete
        tag = self.adata.getNthTag(id);
        self.adata.removeBondedGroup(tag);

    ## \internal
    # \brief Get the number of angles
    def __len__(self):
        return self.adata.getNGlobal();

    ## \internal
    # \brief Get an informal string representing the object
    def __str__(self):
        result = "Angle Data for %d angles of %d typeid(s)" % (self.adata.getNGlobal(), self.adata.getNTypes());
        return result;

    ## \internal
    # \brief Return an interator
    def __iter__(self):
        return angle_data.angle_data_iterator(self);

## Access a single angle via a proxy
#
# angle_data_proxy provides access to all of the properties of a single angle in the system.
# This documentation is intentionally left sparse, see hoomd_script.data for a full explanation of how to use
# angle_data_proxy, documented by example.
#
# The following attributes are read only:
# - \c tag          : A unique integer attached to each angle (not in any particular range). A angle's tag remans fixed
#                     during its lifetime. (Tags previously used by removed angles may be recycled).
# - \c typeid       : An integer indexing the angle's type.
# - \c a            : An integer indexing the A particle in the angle. Particle tags run from 0 to N-1;
# - \c b            : An integer indexing the B particle in the angle. Particle tags run from 0 to N-1;
# - \c c            : An integer indexing the C particle in the angle. Particle tags run from 0 to N-1;
# - \c type         : A string naming the type
#
# In the current version of the API, only already defined type names can be used. A future improvement will allow
# dynamic creation of new type names from within the python API.
# \MPI_SUPPORTED
class angle_data_proxy:
    ## \internal
    # \brief create a angle_data_proxy
    #
    # \param adata AngleData to which this proxy belongs
    # \param id index of this angle in \a adata (at time of proxy creation)
    def __init__(self, adata, id):
        self.adata = adata;
        self.tag = self.adata.getNthTag(id);

    ## \internal
    # \brief Get an informal string representing the object
    def __str__(self):
        result = "";
        result += "tag          : " + str(self.tag) + "\n";
        result += "typeid       : " + str(self.typeid) + "\n";
        result += "a            : " + str(self.a) + "\n"
        result += "b            : " + str(self.b) + "\n"
        result += "c            : " + str(self.c) + "\n"
        result += "type         : " + str(self.type) + "\n";
        return result;

    ## \internal
    # \brief Translate attribute accesses into the low level API function calls
    def __getattr__(self, name):
        if name == "a":
            angle = self.adata.getGroupByTag(self.tag);
            return angle.a;
        if name == "b":
            angle = self.adata.getGroupByTag(self.tag);
            return angle.b;
        if name == "c":
            angle = self.adata.getGroupByTag(self.tag);
            return angle.c;
        if name == "typeid":
            angle = self.adata.getGroupByTag(self.tag);
            return angle.type;
        if name == "type":
            angle = self.adata.getGroupByTag(self.tag);
            typeid = angle.type;
            return self.adata.getNameByType(typeid);

        # if we get here, we haven't found any names that match, post an error
        raise AttributeError;

    ## \internal
    # \brief Translate attribute accesses into the low level API function calls
    def __setattr__(self, name, value):
        if name == "a":
            raise AttributeError;
        if name == "b":
            raise AttributeError;
        if name == "c":
            raise AttributeError;
        if name == "type":
            raise AttributeError;
        if name == "typeid":
            raise AttributeError;

        # otherwise, consider this an internal attribute to be set in the normal way
        self.__dict__[name] = value;

## \internal
# \brief Access dihedral data
#
# dihedral_data provides access to the dihedrals in the system.
# This documentation is intentionally left sparse, see hoomd_script.data for a full explanation of how to use
# dihedral_data, documented by example.
#
class dihedral_data:
    ## \internal
    # \brief dihedral_data iterator
    class dihedral_data_iterator:
        def __init__(self, data):
            self.data = data;
            self.index = 0;
        def __iter__(self):
            return self;
        def __next__(self):
            if self.index == len(self.data):
                raise StopIteration;

            result = self.data[self.index];
            self.index += 1;
            return result;

        # support python2
        next = __next__;

    ## \internal
    # \brief create a dihedral_data
    #
    # \param bdata DihedralData to connect
    def __init__(self, ddata):
        self.ddata = ddata;

    ## \internal
    # \brief Add a new dihedral
    # \param type Type name of the dihedral to add
    # \param a Tag of the first particle in the dihedral
    # \param b Tag of the second particle in the dihedral
    # \param c Tag of the thrid particle in the dihedral
    # \param d Tag of the fourth particle in the dihedral
    # \returns Unique tag identifying this bond
    def add(self, type, a, b, c, d):
        typeid = self.ddata.getTypeByName(type);
        return self.ddata.addBondedGroup(hoomd.Dihedral(typeid, a, b, c, d));

    ## \internal
    # \brief Remove an dihedral by tag
    # \param tag Unique tag of the dihedral to remove
    def remove(self, tag):
        self.ddata.removeBondedGroup(tag);

    ## \var ddata
    # \internal
    # \brief DihedralData to which this instance is connected

    ## \internal
    # \brief Get anm dihedral_proxy reference to the dihedral with id \a id
    # \param id Dihedral id to access
    def __getitem__(self, id):
        if id >= len(self) or id < 0:
            raise IndexError;
        return dihedral_data_proxy(self.ddata, id);

    ## \internal
    # \brief Set an dihedral's properties
    # \param id dihedral id to set
    # \param b Value containing properties to set
    def __setitem__(self, id, b):
        raise RuntimeError('Cannot change angles once they are created');

    ## \internal
    # \brief Delete an dihedral by id
    # \param id Dihedral id to delete
    def __delitem__(self, id):
        if id >= len(self) or id < 0:
            raise IndexError;

        # Get the tag of the bond to delete
        tag = self.ddata.getNthTag(id);
        self.ddata.removeBondedGroup(tag);

    ## \internal
    # \brief Get the number of angles
    def __len__(self):
        return self.ddata.getNGlobal();

    ## \internal
    # \brief Get an informal string representing the object
    def __str__(self):
        result = "Dihedral Data for %d angles of %d typeid(s)" % (self.ddata.getNGlobal(), self.ddata.getNTypes());
        return result;

    ## \internal
    # \brief Return an interator
    def __iter__(self):
        return dihedral_data.dihedral_data_iterator(self);

## Access a single dihedral via a proxy
#
# dihedral_data_proxy provides access to all of the properties of a single dihedral in the system.
# This documentation is intentionally left sparse, see hoomd_script.data for a full explanation of how to use
# dihedral_data_proxy, documented by example.
#
# The following attributes are read only:
# - \c tag          : A unique integer attached to each dihedral (not in any particular range). A dihedral's tag remans fixed
#                     during its lifetime. (Tags previously used by removed dihedral may be recycled).
# - \c typeid       : An integer indexing the dihedral's type.
# - \c a            : An integer indexing the A particle in the angle. Particle tags run from 0 to N-1;
# - \c b            : An integer indexing the B particle in the angle. Particle tags run from 0 to N-1;
# - \c c            : An integer indexing the C particle in the angle. Particle tags run from 0 to N-1;
# - \c d            : An integer indexing the D particle in the dihedral. Particle tags run from 0 to N-1;
# - \c type         : A string naming the type
#
# In the current version of the API, only already defined type names can be used. A future improvement will allow
# dynamic creation of new type names from within the python API.
# \MPI_NOT_SUPPORTED
class dihedral_data_proxy:
    ## \internal
    # \brief create a dihedral_data_proxy
    #
    # \param ddata DihedralData to which this proxy belongs
    # \param id index of this dihedral in \a ddata (at time of proxy creation)
    def __init__(self, ddata, id):
        self.ddata = ddata;
        self.tag = self.ddata.getNthTag(id);

    ## \internal
    # \brief Get an informal string representing the object
    def __str__(self):
        result = "";
        result += "tag          : " + str(self.tag) + "\n";
        result += "typeid       : " + str(self.typeid) + "\n";
        result += "a            : " + str(self.a) + "\n"
        result += "b            : " + str(self.b) + "\n"
        result += "c            : " + str(self.c) + "\n"
        result += "d            : " + str(self.d) + "\n"
        result += "type         : " + str(self.type) + "\n";
        return result;

    ## \internal
    # \brief Translate attribute accesses into the low level API function calls
    def __getattr__(self, name):
        if name == "a":
            dihedral = self.ddata.getGroupByTag(self.tag);
            return dihedral.a;
        if name == "b":
            dihedral = self.ddata.getGroupByTag(self.tag);
            return dihedral.b;
        if name == "c":
            dihedral = self.ddata.getGroupByTag(self.tag);
            return dihedral.c;
        if name == "d":
            dihedral = self.ddata.getGroupByTag(self.tag);
            return dihedral.d;
        if name == "typeid":
            dihedral = self.ddata.getGroupByTag(self.tag);
            return dihedral.type;
        if name == "type":
            dihedral = self.ddata.getGroupByTag(self.tag);
            typeid = dihedral.type;
            return self.ddata.getNameByType(typeid);

        # if we get here, we haven't found any names that match, post an error
        raise AttributeError;

    ## \internal
    # \brief Translate attribute accesses into the low level API function calls
    def __setattr__(self, name, value):
        if name == "a":
            raise AttributeError;
        if name == "b":
            raise AttributeError;
        if name == "c":
            raise AttributeError;
        if name == "d":
            raise AttributeError;
        if name == "type":
            raise AttributeError;
        if name == "typeid":
            raise AttributeError;

        # otherwise, consider this an internal attribute to be set in the normal way
        self.__dict__[name] = value;

## \internal
# \brief Access body data
#
# body_data provides access to the per-body data of all bodies in the system.
# This documentation is intentionally left sparse, see hoomd_script.data for a full explanation of how to use
# body_data, documented by example.
#
class body_data:
    ## \internal
    # \brief bond_data iterator
    class body_data_iterator:
        def __init__(self, data):
            self.data = data;
            self.index = 0;
        def __iter__(self):
            return self;
        def __next__(self):
            if self.index == len(self.data):
                raise StopIteration;

            result = self.data[self.index];
            self.index += 1;
            return result;

        # support python2
        next = __next__;

    ## \internal
    # \brief create a body_data
    #
    # \param bdata BodyData to connect
    def __init__(self, bdata):
        self.bdata = bdata;

    # \brief updates the v and x positions of a rigid body
    # \note the second arguement is dt, but the value should not matter as long as not zero
    def updateRV(self):
        self.bdata.setRV(True);

    ## \var bdata
    # \internal
    # \brief BodyData to which this instance is connected

    ## \internal
    # \brief Get a body_proxy reference to the body with body index \a tag
    # \param tag Body tag to access
    def __getitem__(self, tag):
        if tag >= len(self) or tag < 0:
            raise IndexError;
        return body_data_proxy(self.bdata, tag);

    ## \internal
    # \brief Set a body's properties
    # \param tag Body tag to set
    # \param p Value containing properties to set
    def __setitem__(self, tag, p):
        raise RuntimeError('__setitem__ not implemented');

    ## \internal
    # \brief Get the number of bodies
    def __len__(self):
        return self.bdata.getNumBodies();

    ## \internal
    # \brief Get an informal string representing the object
    def __str__(self):
        result = "Body Data for %d bodies" % (self.bdata.getNumBodies());
        return result

    ## \internal
    # \brief Return an interator
    def __iter__(self):
        return body_data.body_data_iterator(self);

## Access a single body via a proxy
#
# body_data_proxy provides access to all of the properties of a single bond in the system.
# This documentation is intentionally left sparse, see hoomd_script.data for a full explanation of how to use
# body_data_proxy, documented by example.
#
# The following attributes are read only:
# - \c num_particles : The number of particles (or interaction sites) composing the body
# - \c particle_tags : the tags of the particles (or interaction sites) composing the body
# - \c net_force     : Net force acting on the body (x, y, z) (in force units)
# - \c net_torque    : Net torque acting on the body (x, y, z) (in units of force * distance)
#
# The following attributes can be both read and set
# - \c mass          : The mass of the body
# - \c COM           : The Center of Mass position of the body
# - \c velocity      : The velocity vector of the center of mass of the body
# - \c orientation   : The orientation of the body (quaternion)
# - \c angular_momentum : The angular momentum of the body in the space frame
# - \c moment_inertia : the principle components of the moment of inertia
# - \c particle_disp : the displacements of the particles (or interaction sites) of the body relative to the COM in the body frame.
# \MPI_NOT_SUPPORTED
class body_data_proxy:
    ## \internal
    # \brief create a body_data_proxy
    #
    # \param bdata RigidData to which this proxy belongs
    # \param tag tag of this body in \a bdata
    def __init__(self, bdata, tag):

        # Error out in MPI simulations
        if (hoomd.is_MPI_available()):
            if globals.system_definition.getParticleData().getDomainDecomposition():
                globals.msg.error("Rigid bodies are not supported in multi-processor simulations.\n\n")
                raise RuntimeError("Error accessing body data.")

        self.bdata = bdata;
        self.tag = tag;

    ## \internal
    # \brief Get an informal string representing the object
    def __str__(self):
        result = "";
        result += "num_particles    : " + str(self.num_particles) + "\n"
        result += "mass             : " + str(self.mass) + "\n"
        result += "COM              : " + str(self.COM) + "\n"
        result += "velocity         : " + str(self.velocity) + "\n"
        result += "orientation      : " + str(self.orientation) + "\n"
        result += "angular_momentum (space frame) : " + str(self.angular_momentum) + "\n"
        result += "moment_inertia: " + str(self.moment_inertia) + "\n"
        result += "particle_tags    : " + str(self.particle_tags) + "\n"
        result += "particle_disp    : " + str(self.particle_disp) + "\n"
        result += "net_force        : " + str(self.net_force) + "\n"
        result += "net_torque       : " + str(self.net_torque) + "\n"

        return result;

    ## \internal
    # \brief Translate attribute accesses into the low level API function calls
    def __getattr__(self, name):
        if name == "COM":
            COM = self.bdata.getBodyCOM(self.tag);
            return (COM.x, COM.y, COM.z);
        if name == "velocity":
            velocity = self.bdata.getBodyVel(self.tag);
            return (velocity.x, velocity.y, velocity.z);
        if name == "orientation":
            orientation = self.bdata.getBodyOrientation(self.tag);
            return (orientation.x, orientation.y, orientation.z, orientation.w);
        if name == "angular_momentum":
            angular_momentum = self.bdata.getBodyAngMom(self.tag);
            return (angular_momentum.x, angular_momentum.y, angular_momentum.z);
        if name == "num_particles":
            num_particles = self.bdata.getBodyNSize(self.tag);
            return num_particles;
        if name == "mass":
            mass = self.bdata.getMass(self.tag);
            return mass;
        if name == "moment_inertia":
            moment_inertia = self.bdata.getBodyMomInertia(self.tag);
            return (moment_inertia.x, moment_inertia.y, moment_inertia.z);
        if name == "particle_tags":
            particle_tags = [];
            for i in range(0, self.num_particles):
               particle_tags.append(self.bdata.getParticleTag(self.tag, i));
            return particle_tags;
        if name == "particle_disp":
            particle_disp = [];
            for i in range(0, self.num_particles):
               disp = self.bdata.getParticleDisp(self.tag, i);
               particle_disp.append([disp.x, disp.y, disp.z]);
            return particle_disp;
        if name == "net_force":
            f = self.bdata.getBodyNetForce(self.tag);
            return (f.x, f.y, f.z);
        if name == "net_torque":
            t = self.bdata.getBodyNetTorque(self.tag);
            return (t.x, t.y, t.z);

        # if we get here, we haven't found any names that match, post an error
        raise AttributeError;

    ## \internal
    # \brief Translate attribute accesses into the low level API function calls
    def __setattr__(self, name, value):
        if name == "COM":
            p = hoomd.Scalar3();
            p.x = float(value[0]);
            p.y = float(value[1]);
            p.z = float(value[2]);
            self.bdata.setBodyCOM(self.tag, p);
            return;
        if name == "velocity":
            v = hoomd.Scalar3();
            v.x = float(value[0]);
            v.y = float(value[1]);
            v.z = float(value[2]);
            self.bdata.setBodyVel(self.tag, v);
            return;
        if name == "mass":
            self.bdata.setMass(self.tag, value);
            return;
        if name == "orientation":
            q = hoomd.Scalar4();
            q.x = float(value[0]);
            q.y = float(value[1]);
            q.z = float(value[2]);
            q.w = float(value[3]);
            self.bdata.setBodyOrientation(self.tag, q);
            return;
        if name == "angular_momentum":
            p = hoomd.Scalar3();
            p.x = float(value[0]);
            p.y = float(value[1]);
            p.z = float(value[2]);
            self.bdata.setAngMom(self.tag, p);
            return;
        if name == "moment_inertia":
            p = hoomd.Scalar3();
            p.x = float(value[0]);
            p.y = float(value[1]);
            p.z = float(value[2]);
            self.bdata.setBodyMomInertia(self.tag, p);
            return;
        if name == "particle_disp":
            p = hoomd.Scalar3();
            for i in range(0, self.num_particles):
                p.x = float(value[i][0]);
                p.y = float(value[i][1]);
                p.z = float(value[i][2]);
                self.bdata.setParticleDisp(self.tag, i, p);
            return;
        if name == "net_force":
            raise AttributeError;
        if name == "net_torque":
            raise AttributeError;

        # otherwise, consider this an internal attribute to be set in the normal way
        self.__dict__[name] = value;
