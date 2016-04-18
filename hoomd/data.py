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

# Maintainer: joaander

from hoomd import _hoomd
import hoomd

## \package hoomd.data
# \brief Access particles, bonds, and other state information inside scripts
#
# Code in the data package provides high-level access to all of the particle, bond and other %data that define the
# current state of the system. You can use python code to directly read and modify this data, allowing you to analyze
# simulation results while the simulation runs, or to create custom initial configurations with python code.
#
# There are two ways to access the data.
#
# 1. Snapshots record the system configuration at one instant in time. You can store this state to analyze the data,
#    restore it at a future point in time, or to modify it and reload it. Use snapshots for initializing simulations,
#    or when you need to access or modify the entire simulation state.
# 2. Data proxies directly access the current simulation state. Use data proxies if you need to only touch a few
#    particles or bonds at a a time.
#
# \section data_snapshot Snapshots
# <hr>
#
# <h3>Relevant methods:</h3>
#
# * system_data.take_snapshot() captures a snapshot of the current system state. A snapshot is a copy of the simulation
# state. As the simulation continues to progress, data in a captured snapshot will remain constant.
# * system_data.restore_snapshot() replaces the current system state with the state stored in a snapshot.
# * data.make_snapshot() creates an empty snapshot that you can populate with custom data.
# * init.read_snapshot() initializes a simulation from a snapshot.
#
# \code
# snapshot = system.take_snapshot()
# system.restore_snapshot(snapshot)
# snapshot = data.make_snapshot(N=100, particle_types=['A', 'B'], box=data.boxdim(L=10))
# # ... populate snapshot with data ...
# init.read_snapshot(snapshot)
# \endcode
#
# <hr>
# <h3>Snapshot and MPI</h3>
# In MPI simulations, the snapshot is only valid on rank 0. make_snapshot, read_snapshot, and take_snapshot, restore_snapshot are
# collective calls, and need to be called on all ranks. But only rank 0 can access data in the snapshot.
# \code
# snapshot = system.take_snapshot(all=True)
# if comm.get_rank() == 0:
#     s = init.create_random(N=100, phi_p=0.05);numpy.mean(snapshot.particles.velocity))
#     snapshot.particles.position[0] = [1,2,3];
#
# system.restore_snapshot(snapshot);
# snapshot = data.make_snapshot(N=10, box=data.boxdim(L=10))
# if comm.get_rank() == 0:
#     snapshot.particles.position[:] = ....
# init.read_snapshot(snapshot)
# \endcode
#
# <hr>
# <h3>Simulation box</h3>
# You can access the simulation box from a snapshot:
# \code
# >>> print(snapshot.box)
# Box: Lx=17.3646569289 Ly=17.3646569289 Lz=17.3646569289 xy=0.0 xz=0.0 yz=0.0 dimensions=3
# \endcode
# and can change it:
# \code
# >>> snapsot.box = data.boxdim(Lx=10, Ly=20, Lz=30, xy=1.0, xz=0.1, yz=2.0)
# >>> print(snapshot.box)
# Box: Lx=10 Ly=20 Lz=30 xy=1.0 xz=0.1 yz=2.0 dimensions=3
# \endcode
# \b All particles must be inside the box before using the snapshot to initialize a simulation or restoring it.
# The dimensionality of the system (2D/3D) cannot change after initialization.
#
# <h3>Particle properties</h3>
#
# Particle properties are present in `snapshot.particles`. Each property is stored in a numpy array that directly
# accesses the memory of the snapshot. References to these arrays will become invalid when the snapshot itself is
# garbage collected.
#
# - `N` is the number of particles in the particle data snapshot
# \code
# >>> print(snapshot.particles.N)
# 64000
# \endcode
# - Change the number of particles in the snapshot with resize. Existing particle properties are
#   preserved after the resize. Any newly created particles will have default values. After resizing,
#   existing references to the numpy arrays will be invalid, access them again
#   from `snapshot.particles.*`
# \code
# >>> snapshot.particles.resize(1000);
# \endcode
# - The list of all particle types in the simulation can be accessed and modified
# \code
# >>> print(snapshot.particles.types)
# ['A', 'B', 'C']
# >>> snapshot.particles.types = ['1', '2', '3', '4'];
# \endcode
# - Individual particles properties are stored in numpy arrays. Vector quantities are stored in Nx3 arrays of floats
#   (or doubles) and scalar quantities are stored in N length 1D arrays.
# \code
# >>> print(snapshot.particles.position[10])
# [ 1.2398  -10.2687  100.6324]
# \endcode
# - Various properties can be accessed of any particle, and the numpy arrays can be sliced or passed whole to other
#   routines.
# \code
# >>> print(snapshot.particles.typeid[10])
# 2
# >>> print(snapshot.particles.velocity[10])
# (-0.60267972946166992, 2.6205904483795166, -1.7868227958679199)
# >>> print(snapshot.particles.mass[10])
# 1.0
# >>> print(snapshot.particles.diameter[10])
# 1.0
# \endcode
# - Particle properties can be set in the same way. This modifies the data in the snapshot, not the
#   current simulation state.
# \code
# >>> snapshot.particles.position[10] = [1,2,3]
# >>> print(snapshot.particles.position[10])
# [ 1.  2.  3.]
# \endcode
# - Snapshots store particle types as integers that index into the type name array:
# \code
# >>> print(snapshot.particles.typeid)
# [ 0.  1.  2.  0.  1.  2.  0.  1.  2.  0.]
# >>> snapshot.particles.types = ['A', 'B', 'C'];
# >>> snapshot.particles.typeid[0] = 2;   # C
# >>> snapshot.particles.typeid[1] = 0;   # A
# >>> snapshot.particles.typeid[2] = 1;   # B
# \endcode
#
# For a list of all particle properties in the snapshot see SnapshotParticleData.
#
# <h3>Bonds</h3>
#
# Bonds are stored in `snapshot.bonds`. system_data.take_snapshot() does not record the bonds by default, you need to
# request them with the argument `bonds=True`.
#
# - `N` is the number of bonds in the bond data snapshot
# \code
# >>> print(snapshot.bonds.N)
# 100
# \endcode
# - Change the number of bonds in the snapshot with resize. Existing bonds are
#   preserved after the resize. Any newly created bonds will be initialized to 0. After resizing,
#   existing references to the numpy arrays will be invalid, access them again
#   from `snapshot.bonds.*`
# \code
# >>> snapshot.bonds.resize(1000);
# \endcode
# - Bonds are stored in an Nx2 numpy array `group`. The first axis accesses the bond `i`. The second axis `j` goes over
#   the individual particles in the bond. The value of each element is the tag of the particle participating in the
#   bond.
# \code
# >>> print(snapshot.bonds.group)
# [[0 1]
# [1 2]
# [3 4]
# [4 5]]
# >>> snapshot.bonds.group[0] = [10,11]
# \endcode
# - Snapshots store bond types as integers that index into the type name array:
# \code
# >>> print(snapshot.bonds.typeid)
# [ 0.  1.  2.  0.  1.  2.  0.  1.  2.  0.]
# >>> snapshot.bonds.types = ['A', 'B', 'C'];
# >>> snapshot.bonds.typeid[0] = 2;   # C
# >>> snapshot.bonds.typeid[1] = 0;   # A
# >>> snapshot.bonds.typeid[2] = 1;   # B
# \endcode
#
# <h3>Angles, dihedrals and impropers</h3>
#
# Angles, dihedrals, and impropers are stored similar to bonds. The only difference is that the group array is sized
# appropriately to store the number needed for each type of bond.
#
# * `snapshot.angles.group` is Nx3
# * `snapshot.dihedrals.group` is Nx4
# * `snapshot.impropers.group` is Nx4
#
# <h3>Constraints</h3>
#
# Pairwise distance constraints are added and removed like bonds. They are defined between two particles.
# The only difference is that instead of a type, constraints take a distance as parameter.
#
# - `N` is the number of constraints in the constraint data snapshot
# \code
# >>> print(snapshot.constraints.N)
# 99
# \endcode
# - Change the number of constraints in the snapshot with resize. Existing constraints are
#   preserved after the resize. Any newly created constraints will be initialized to 0. After resizing,
#   existing references to the numpy arrays will be invalid, access them again
#   from `snapshot.constraints.*`
# \code
# >>> snapshot.constraints.resize(1000);
# \endcode
# - Bonds are stored in an Nx2 numpy array `group`. The first axis accesses the constraint `i`. The second axis `j` goes over
#   the individual particles in the constraint. The value of each element is the tag of the particle participating in the
#   constraint.
# \code
# >>> print(snapshot.constraints.group)
# [[4 5]
# [6 7]
# [6 8]
# [7 8]]
# >>> snapshot.constraints.group[0] = [10,11]
# \endcode
# - Snapshots store constraint distances as floats
# \code
# >>> print(snapshot.constraints.value)
# [ 1.5 2.3 1.0 0.1 ]
# \endcode
#
# \section data_proxy Proxy access
#
# For most of the cases below, it is assumed that the result of the initialization command was saved at the beginning
# of the script, like so:
# \code
# system = init.read_xml(filename="input.xml")
# \endcode
#
# <hr>
# <h3>Simulation box</h3>
# You can access the simulation box like so:
# \code
# >>> print(system.box)
# Box: Lx=17.3646569289 Ly=17.3646569289 Lz=17.3646569289 xy=0.0 xz=0.0 yz=0.0
# \endcode
# and can change it like so:
# \code
# >>> system.box = data.boxdim(Lx=10, Ly=20, Lz=30, xy=1.0, xz=0.1, yz=2.0)
# >>> print(system.box)
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
# >>> print(system.particles)
# Particle Data for 64000 particles of 1 type(s)
# \endcode
# - The list of all particle types in the simulation can be accessed
# \code
# >>> print(system.particles.types)
# ['A']
# >>> print system.particles.types
# Particle types: ['A']
# \endcode
# - Particle types can be added between subsequent run() commands:
# \code
# >>> system.particles.types.add('newType')
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
# >>> p.tag
# 4
# \endcode
# (note that p can be replaced with system.particles[i] above and the results are the same)
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
# For doing modifications that operate on the whole system data efficiently, snapshots can be used.
# Their usage is described below.
#
# Particles may be added at any time in the job script, and a unique tag is returned.
# \code
# >>> system.particles.add('A')
# >>> t = system.particles.add('B')
# \endcode
#
# Particles may be deleted by index.
# \code
# >>> del system.particles[0]
# >>> print(system.particles[0])
# tag         : 1
# position    : (23.846603393554688, -27.558368682861328, -20.501256942749023)
# image       : (0, 0, 0)
# velocity    : (0.0, 0.0, 0.0)
# acceleration: (0.0, 0.0, 0.0)
# charge      : 0.0
# mass        : 1.0
# diameter    : 1.0
# type        : A
# typeid      : 0
# body        : 4294967295
# orientation : (1.0, 0.0, 0.0, 0.0)
# net_force   : (0.0, 0.0, 0.0)
# net_energy  : 0.0
# net_torque  : (0.0, 0.0, 0.0)
# \endcode
# \note The particle with tag 1 is now at index 0. No guarantee is made about how the
# order of particles by index will or will not change, so do not write any job scripts which assume a given ordering.
#
# To access particles in an index-independent manner, use their tags. For example, to remove all particles
# of type 'A', do
# \code
# tags = []
# for p in system.particles:
#     if p.type == 'A'
#         tags.append(p.tag)
# \endcode
# Then remove each of the bonds by their unique tag.
# \code
# for t in tags:
#     system.particles.remove(t)
# \endcode
# Particles can also be accessed through their unique tag:
# \code
# t = system.particles.add('A')
# p = system.particles.get(t)
# \endcode
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
# >>> print(bnd)
# tag          : 0
# typeid       : 0
# a            : 0
# b            : 1
# type         : bondA
# >>> print(bnd.type)
# bondA
# >>> print(bnd.a)
# 0
# >>> print(bnd.b)
#1
# \endcode
# \note The order in which bonds appear by index is not static and may change at any time!
#
# Bonds may be deleted by index.
# \code
# >>> del system.bonds[0]
# >>> print(system.bonds[0])
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
# Bonds can also be accessed through their unique tag:
# \code
# t = system.bonds.add('polymer',0,1)
# p = system.bonds.get(t)
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
# <hr>
# <h3>Constraints</h3>
# Constraints may be added and removed from within the job script.
#
# To add a constraint of length 1.5 between particles 0 and 1:
# \code
# >>> t = system.constraints.add(0, 1, 1.5)
# \endcode
#
# To remove it again:
# \code
# >>> system.contraints.remove(t)
# \endcode
#
# <hr>
# <h3>Forces</h3>
# Forces can be accessed in a similar way.
# \code
# >>> lj = pair.lj(r_cut=3.0)
# >>> lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
# >>> print(lj.forces[0])
# tag         : 0
# force       : (-0.077489577233791351, -0.029512746259570122, -0.13215918838977814)
# virial      : -0.0931386947632
# energy      : -0.0469368174672
# >>> f0 = lj.forces[0]
# >>> print(f0.force)
# (-0.077489577233791351, -0.029512746259570122, -0.13215918838977814)
# >>> print(f0.virial)
# -0.093138694763n
# >>> print(f0.energy)
# -0.0469368174672
# \endcode
#
# In this manner, forces due to the lj %pair %force, bonds, and any other %force commands in hoomd can be accessed
# independently from one another. See force_data_proxy for a definition of each parameter accessed.
#
# <hr>
# <h3>Proxy references</h3>
#
# For advanced code using the particle data access from python, it is important to understand that the hoomd
# particles, forces, bonds, et cetera, are accessed as proxies. This means that after
# \code
# p = system.particles[i]
# \endcode
# is executed, \a p \b does \b not store the position, velocity, ... of particle \a i. Instead, it just stores \a i and
# provides an interface to get/set the properties on demand. This has some side effects you need to be aware of.
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

## Define box dimensions
#
# Simulation boxes in hoomd are specified by six parameters, *Lx*, *Ly*, *Lz*, *xy*, *xz* and *yz*. For full details,
# see \ref page_box. A boxdim provides a way to specify all six parameters for a given box and perform some common
# operations with them. Modifying a boxdim does not modify the underlying simulation box in hoomd. A boxdim can be passed
# to an initialization method or to assigned to a saved sysdef variable (`system.box = new_box`) to set the simulation
# box.
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
# In 2D boxes, *volume* is in units of area.
#
# **Shorthand notation**
#
# data.boxdim accepts the keyword argument *L=x* as shorthand notation for `Lx=x, Ly=x, Lz=x` in 3D
# and `Lx=x, Ly=z, Lz=1` in 2D. If you specify both `L=` and `Lx,Ly, or Lz`, then the value for `L` will override
# the others.
#
# **Examples:**
#
# There are many ways to define boxes.
#
# * Cubic box with given volume: `data.boxdim(volume=V)`
# * Triclinic box in 2D with given area: `data.boxdim(xy=1.0, dimensions=2, volume=A)`
# * Rectangular box in 2D with given area and aspect ratio: `data.boxdim(Lx=1, Ly=aspect, dimensions=2, volume=A)`
# * Cubic box with given length: `data.boxdim(L=10)`
# * Fully define all box parameters: `data.boxdim(Lx=10, Ly=20, Lz=30, xy=1.0, xz=0.5, yz=0.1)`
#
# system = init.read_xml('init.xml')
# system.box = system.box.scale(s=2)
# ~~~
class boxdim(hoomd.meta._metadata):
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

        # base class constructor
        hoomd.meta._metadata.__init__(self)

    ## Scale box dimensions
    #
    # \param sx scale factor in the x direction
    # \param sy scale factor in the y direction
    # \param sz scale factor in the z direction
    #
    # Scales the box by the given scale factors. Tilt factors are not modified.
    #
    # \returns a reference to itself
    def scale(self, sx=1.0, sy=1.0, sz=1.0, s=None):
        if s is not None:
            sx = s;
            sy = s;
            sz = s;

        self.Lx = self.Lx * sx;
        self.Ly = self.Ly * sy;
        self.Lz = self.Lz * sz;
        return self

    ## Set the box volume
    #
    # \param volume new box volume (area if dimensions=2)
    #
    # setVolume() scales the box to the given volume (or area).
    #
    # \returns a reference to itself
    def set_volume(self, volume):
        cur_vol = self.get_volume();

        if self.dimensions == 3:
            s = (volume / cur_vol)**(1.0/3.0)
            self.scale(s, s, s);
        else:
            s = (volume / cur_vol)**(1.0/2.0)
            self.scale(s, s, 1.0);
        return self

    ## Get the box volume
    #
    # Returns the box volume (area in 2D).
    #
    def get_volume(self):
        b = self._getBoxDim();
        return b.getVolume(self.dimensions == 2);

    ## Get a lattice vector
    #
    # \param i (=0,1,2) direction of lattice vector
    #
    # \returns a lattice vector (3-tuple) along direction \a i
    #
    def get_lattice_vector(self,i):
        b = self._getBoxDim();
        v = b.getLatticeVector(int(i))
        return (v.x, v.y, v.z)

    ## Wrap a vector using the periodic boundary conditions
    #
    # \param v The vector to wrap
    # \param img A vector of integer image flags that will be updated (optional)
    #
    # \returns the wrapped vector and the image flags
    #
    def wrap(self,v, img=(0,0,0)):
        u = _hoomd.make_scalar3(v[0],v[1],v[2])
        i = _hoomd.make_int3(int(img[0]),int(img[1]),int(img[2]))
        c = _hoomd.make_char3(0,0,0)
        self._getBoxDim().wrap(u,i,c)
        img = (i.x,i.y,i.z)
        return (u.x, u.y, u.z),img

    ## Apply the minimum image convention to a vector using periodic boundary conditions
    #
    # \param v The vector to apply minimum image to
    #
    # \returns the minimum image
    #
    def min_image(self,v):
        u = _hoomd.make_scalar3(v[0],v[1],v[2])
        u = self._getBoxDim().minImage(u)
        return (u.x, u.y, u.z)

    ## Scale a vector to fractional coordinates
    #
    # \param v The vector to convert to fractional coordinates
    #
    # make_fraction() takes a vector in a box and computes a vector where all components are
    # between 0 and 1.
    #
    # \returns the scaled vector
    def make_fraction(self,v):
        u = _hoomd.make_scalar3(v[0],v[1],v[2])
        w = _hoomd.make_scalar3(0,0,0)

        u = self._getBoxDim().makeFraction(u,w)
        return (u.x, u.y, u.z)

    ## \internal
    # \brief Get a C++ boxdim
    def _getBoxDim(self):
        b = _hoomd.BoxDim(self.Lx, self.Ly, self.Lz);
        b.setTiltFactors(self.xy, self.xz, self.yz);
        return b

    def __str__(self):
        return 'Box: Lx=' + str(self.Lx) + ' Ly=' + str(self.Ly) + ' Lz=' + str(self.Lz) + ' xy=' + str(self.xy) + \
                    ' xz='+ str(self.xz) + ' yz=' + str(self.yz) + ' dimensions=' + str(self.dimensions);

    ## \internal
    # \brief Get a dictionary representation of the box dimensions
    def get_metadata(self):
        data = hoomd.meta._metadata.get_metadata(self)
        data['d'] = self.dimensions
        data['Lx'] = self.Lx
        data['Ly'] = self.Ly
        data['Lz'] = self.Lz
        data['xy'] = self.xy
        data['xz'] = self.xz
        data['yz'] = self.yz
        data['V'] = self.get_volume()
        return data

##
# \brief Access system data
#
# system_data provides access to the different data structures that define the current state of the simulation.
# This documentation is intentionally left sparse, see hoomd.data for a full explanation of how to use
# system_data, documented by example.
#
class system_data(hoomd.meta._metadata):
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
        self.constraints = constraint_data(sysdef.getConstraintData());

        # base class constructor
        hoomd.meta._metadata.__init__(self)

    ## Take a snapshot of the current system data
    #
    # This functions returns a snapshot object. It contains the current
    # partial or complete simulation state. With appropriate options
    # it is possible to select which data properties should be included
    # in the snapshot.
    #
    # \param particles If true, particle data is included in the snapshot
    # \param bonds If true, bond, angle, dihedral, improper and constraint data is included
    # \param integrators If true, integrator data is included the snapshot
    # \param all If true, the entire system state is saved in the snapshot
    # \param dtype Datatype for the snapshot numpy arrays. Must be either 'float' or 'double'.
    #
    # \returns the snapshot object.
    #
    # \code
    # snapshot = system.take_snapshot()
    # snapshot = system.take_snapshot()
    # snapshot = system.take_snapshot(bonds=true)
    # \endcode
    #
    # \MPI_SUPPORTED
    def take_snapshot(self,
                      particles=True,
                      bonds=False,
                      integrators=False,
                      all=False,
                      dtype='float'):
        hoomd.util.print_status_line();

        if all is True:
                particles=True
                bonds=True
                integrators=True

        # take the snapshot
        if dtype == 'float':
            cpp_snapshot = self.sysdef.takeSnapshot_float(particles,bonds,bonds,bonds,bonds,bonds,integrators)
        elif dtype == 'double':
            cpp_snapshot = self.sysdef.takeSnapshot_double(particles,bonds,bonds,bonds,bonds,bonds,integrators)
        else:
            raise ValueError("dtype must be float or double");

        return cpp_snapshot

    ## Replicates the system along the three spatial dimensions
    #
    # \param nx Number of times to replicate the system along the x-direction
    # \param ny Number of times to replicate the system along the y-direction
    # \param nz Number of times to replicate the system along the z-direction
    #
    # This method explictly replicates particles along all three spatial directions, as
    # opposed to replication implied by periodic boundary conditions.
    # The box is resized and the number of particles is updated so that the new box
    # holds the specified number of replicas of the old box along all directions.
    # Particle coordinates are updated accordingly to fit into the new box. All velocities and
    # other particle properties are replicated as well. Also bonded groups between particles
    # are replicated.
    #
    # Example usage:
    # \code
    # system = init.read_xml("some_file.xml")
    # system.replicate(nx=2,ny=2,nz=2)
    # \endcode
    #
    # \note It is a limitation that in MPI simulations the dimensions of the processor grid
    # are not updated upon replication. For example, if an initially cubic box is replicated along only one
    # spatial direction, this could lead to decreased performance if the processor grid was
    # optimal for the original box dimensions, but not for the new ones.
    #
    # \MPI_SUPPORTED
    def replicate(self, nx=1, ny=1, nz=1):
        hoomd.util.print_status_line()

        nx = int(nx)
        ny = int(ny)
        nz = int(nz)

        if nx == ny == nz == 1:
            hoomd.context.msg.warning("All replication factors == 1. Not replicating system.\n")
            return

        if nx <= 0 or ny <= 0 or nz <= 0:
            hoomd.context.msg.error("Cannot replicate by zero or by a negative value along any direction.")
            raise RuntimeError("nx, ny, nz need to be positive integers")

        # Take a snapshot
        hoomd.util.quiet_status()
        cpp_snapshot = self.take_snapshot(all=True)
        hoomd.util.unquiet_status()

        if hoomd.comm.get_rank() == 0:
            # replicate
            cpp_snapshot.replicate(nx, ny, nz)

        # restore from snapshot
        hoomd.util.quiet_status()
        self.restore_snapshot(cpp_snapshot)
        hoomd.util.unquiet_status()

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
    # \warning restore_snapshot() may invalidate force coefficients, neighborlist r_cut values, and other per type quantities if called within a callback
    #          during a run(). You can restore a snapshot during a run only if the snapshot is of a previous state of the currently running system.
    #          Otherwise, you need to use restore_snapshot() between run() commands to ensure that all per type coefficients are updated properly.
    #
    # \sa hoomd.data
    # \MPI_SUPPORTED
    def restore_snapshot(self, snapshot):
        hoomd.util.print_status_line();

        self.sysdef.initializeFromSnapshot(snapshot);

    ## \var sysdef
    # \internal
    # \brief SystemDefinition to which this instance is connected

    ## \internal
    # \brief Get particle metadata
    def get_metadata(self):
        data = hoomd.meta._metadata.get_metadata(self)
        data['box'] = self.box
        data['particles'] = self.particles
        data['number_density'] = len(self.particles)/self.box.get_volume()

        data['bonds'] = self.bonds
        data['angles'] = self.angles
        data['dihedrals'] = self.dihedrals
        data['impropers'] = self.impropers
        data['constraints'] = self.constraints

        data['timestep'] = hoomd.context.current.system.getCurrentTimeStep()
        return data

    ## Get the system box
    @property
    def box(self):
        b = self.sysdef.getParticleData().getGlobalBox();
        L = b.getL();
        return boxdim(Lx=L.x, Ly=L.y, Lz=L.z, xy=b.getTiltFactorXY(), xz=b.getTiltFactorXZ(), yz=b.getTiltFactorYZ(), dimensions=self.sysdef.getNDimensions());

    ## Set the system box
    # \param value The new boundaries (a data.boxdim object)
    @box.setter
    def box(self, value):
        if not isinstance(value, boxdim):
            raise TypeError('box must be a data.boxdim object');
        self.sysdef.getParticleData().setGlobalBox(value._getBoxDim());


## \internal
# \brief Access the list of types
#
# pdata_types_proxy provides access to the type names and the possibility to add types to the simulation
# This documentation is intentionally left sparse, see hoomd.data for a full explanation of how to use
# particle_data, documented by example.
#
class pdata_types_proxy(object):
    ## \internal
    # \brief particle_data iterator
    class pdata_types_iterator(object):
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
    # \brief create a pdata_types_proxy
    #
    # \param pdata ParticleData to connect
    def __init__(self, pdata):
        self.pdata = pdata;

    ## \var pdata
    # \internal
    # \brief ParticleData to which this instance is connected

    ## \internal
    # \brief Get a the name of a type
    # \param type_idx Type index
    def __getitem__(self, type_idx):
        ntypes = self.pdata.getNTypes();
        if type_idx >= ntypes or type_idx < 0:
            raise IndexError;
        return self.pdata.getNameByType(type_idx);

    ## \internal
    # \brief Set the name of a type
    # \param type_idx Particle tag to set
    # \param name New type name
    def __setitem__(self, type_idx, name):
        ntypes = self.pdata.getNTypes();
        if type_idx >= ntypes or type_idx < 0:
            raise IndexError;
        self.pdata.setTypeName(type_idx, name);

    ## \internal
    # \brief Get the number of types
    def __len__(self):
        return self.pdata.getNTypes();

    ## \internal
    # \brief Get an informal string representing the object
    def __str__(self):
        ntypes = self.pdata.getNTypes();
        result = "Particle types: ["
        for i in range(0,ntypes):
            result += "'" + self.pdata.getNameByType(i) + "'"
            if (i != ntypes-1):
                result += ", "
            else:
                result += "]"

        return result

    ## \internal
    # \brief Return an interator
    def __iter__(self):
        return pdata_types_proxy.pdata_types_iterator(self);

    ## \internal
    # \brief Add a new particle type
    # \param name Name of type to add
    # \returns Index of newly added type
    def add(self, name):
        # check that type does not yet exist
        ntypes = self.pdata.getNTypes();
        for i in range(0,ntypes):
            if self.pdata.getNameByType(i) == name:
                hoomd.context.msg.warning("Type '"+name+"' already defined.\n");
                return i

        typeid = self.pdata.addType(name);
        return typeid


## \internal
# \brief Access particle data
#
# particle_data provides access to the per-particle data of all particles in the system.
# This documentation is intentionally left sparse, see hoomd.data for a full explanation of how to use
# particle_data, documented by example.
#
class particle_data(hoomd.meta._metadata):
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

        self.types = pdata_types_proxy(hoomd.context.current.system_definition.getParticleData())

        # base class constructor
        hoomd.meta._metadata.__init__(self)

    ## \var pdata
    # \internal
    # \brief ParticleData to which this instance is connected

    ## \internal
    # \brief Get a particle_proxy reference to the particle with contiguous id \a id
    # \param id Contiguous particle id to access
    def __getitem__(self, id):
        if id >= len(self) or id < 0:
            raise IndexError;
        tag = self.pdata.getNthTag(id);
        return particle_data_proxy(self.pdata, tag);

    ## \internal
    # \brief Get a particle_proxy reference to the particle with tag \a tag
    # \param tag Particle tag to access
    def get(self, tag):
        if tag > self.pdata.getMaximumTag() or tag < 0:
            raise IndexError;
        return particle_data_proxy(self.pdata, tag);

    ## \internal
    # \brief Set a particle's properties
    # \param tag Particle tag to set
    # \param p Value containing properties to set
    def __setitem__(self, tag, p):
        raise RuntimeError('__setitem__ not implemented');

    ## \internal
    # \brief Add a new particle
    # \param type Type name of the particle to add
    # \returns Unique tag identifying this bond
    def add(self, type):
        typeid = self.pdata.getTypeByName(type);
        return self.pdata.addParticle(typeid);

    ## \internal
    # \brief Remove a bond by tag
    # \param tag Unique tag of the bond to remove
    def remove(self, tag):
        self.pdata.removeParticle(tag);

    ## \internal
    # \brief Delete a particle by id
    # \param id Bond id to delete
    def __delitem__(self, id):
        if id >= len(self) or id < 0:
            raise IndexError;
        tag = self.pdata.getNthTag(id);
        self.pdata.removeParticle(tag);

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
    # \brief Return an iterator
    def __iter__(self):
        return particle_data.particle_data_iterator(self);

    ## \internal
    # \brief Return metadata for this particle_data instance
    def get_metadata(self):
        data = hoomd.meta._metadata.get_metadata(self)
        data['N'] = len(self)
        data['types'] = list(self.types);
        return data

## Access a single particle via a proxy
#
# particle_data_proxy provides access to all of the properties of a single particle in the system.
# This documentation is intentionally left sparse, see hoomd.data for a full explanation of how to use
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
class particle_data_proxy(object):
    ## \internal
    # \brief create a particle_data_proxy
    #
    # \param pdata ParticleData to which this proxy belongs
    # \param tag Tag this particle in \a pdata
    def __init__(self, pdata, tag):
        self.pdata = pdata;
        self.tag = tag

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
        result += "mom. inertia: " + str(self.moment_inertia) + "\n";
        result += "angular_momentum: " + str(self.angular_momentum) + "\n";
        result += "net_force   : " + str(self.net_force) + "\n";
        result += "net_energy  : " + str(self.net_energy) + "\n";
        result += "net_torque  : " + str(self.net_torque) + "\n";
        return result;

    @property
    def position(self):
        pos = self.pdata.getPosition(self.tag);
        return (pos.x, pos.y, pos.z);

    @position.setter
    def position(self, value):
        v = _hoomd.Scalar3();
        v.x = float(value[0]);
        v.y = float(value[1]);
        v.z = float(value[2]);
        self.pdata.setPosition(self.tag, v, True);

    @property
    def velocity(self):
        vel = self.pdata.getVelocity(self.tag);
        return (vel.x, vel.y, vel.z);

    @velocity.setter
    def velocity(self, value):
        v = _hoomd.Scalar3();
        v.x = float(value[0]);
        v.y = float(value[1]);
        v.z = float(value[2]);
        self.pdata.setVelocity(self.tag, v);

    @property
    def acceleration(self):
        accel = self.pdata.getAcceleration(self.tag);
        return (accel.x, accel.y, accel.z);

    @property
    def image(self):
        image = self.pdata.getImage(self.tag);
        return (image.x, image.y, image.z);

    @image.setter
    def image(self, value):
        v = _hoomd.int3();
        v.x = int(value[0]);
        v.y = int(value[1]);
        v.z = int(value[2]);
        self.pdata.setImage(self.tag, v);

    @property
    def charge(self):
        return self.pdata.getCharge(self.tag);

    @charge.setter
    def charge(self, value):
        self.pdata.setCharge(self.tag, float(value));

    @property
    def mass(self):
        return self.pdata.getMass(self.tag);

    @mass.setter
    def mass(self, value):
        self.pdata.setMass(self.tag, float(value));

    @property
    def diameter(self):
        return self.pdata.getDiameter(self.tag);

    @diameter.setter
    def diameter(self, value):
        self.pdata.setDiameter(self.tag, value);

    @property
    def typeid(self):
        return self.pdata.getType(self.tag);

    @property
    def body(self):
        return self.pdata.getBody(self.tag);

    @body.setter
    def body(self, value):
        self.pdata.setBody(self.tag, value);

    @property
    def type(self):
        typeid = self.pdata.getType(self.tag);
        return self.pdata.getNameByType(typeid);

    @type.setter
    def type(self, value):
        typeid = self.pdata.getTypeByName(value);
        self.pdata.setType(self.tag, typeid);

    @property
    def orientation(self):
        o = self.pdata.getOrientation(self.tag);
        return (o.x, o.y, o.z, o.w);

    @orientation.setter
    def orientation(self, value):
        o = _hoomd.Scalar4();
        o.x = float(value[0]);
        o.y = float(value[1]);
        o.z = float(value[2]);
        o.w = float(value[3]);
        self.pdata.setOrientation(self.tag, o);

    @property
    def angular_momentum(self):
        a = self.pdata.getAngularMomentum(self.tag);
        return (a.x, a.y, a.z, a.w);

    @angular_momentum.setter
    def angular_momentum(self, value):
        a = _hoomd.Scalar4();
        a.x = float(value[0]);
        a.y = float(value[1]);
        a.z = float(value[2]);
        a.w = float(value[3]);
        self.pdata.setAngularMomentum(self.tag, a);

    @property
    def moment_inertia(self):
        m = self.pdata.getMomentsOfInertia(self.tag)
        return (m.x, m.y, m.z);

    @moment_inertia.setter
    def moment_inertia(self, value):
        m = _hoomd.Scalar3();
        m.x = float(value[0]);
        m.y = float(value[1]);
        m.z = float(value[2]);
        self.pdata.setMomentsOfInertia(self.tag, m);

    @property
    def net_force(self):
        f = self.pdata.getPNetForce(self.tag);
        return (f.x, f.y, f.z);

    @property
    def net_energy(self):
        f = self.pdata.getPNetForce(self.tag);
        return f.w;

    @property
    def net_torque(self):
        f = self.pdata.getNetTorque(self.tag);
        return (f.x, f.y, f.z);

## \internal
# Access force data
#
# force_data provides access to the per-particle data of all forces in the system.
# This documentation is intentionally left sparse, see hoomd.data for a full explanation of how to use
# force_data, documented by example.
#
class force_data(object):
    ## \internal
    # \brief force_data iterator
    class force_data_iterator(object):
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
        return hoomd.context.current.system_definition.getParticleData().getNGlobal();

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
# This documentation is intentionally left sparse, see hoomd.data for a full explanation of how to use
# force_data_proxy, documented by example.
#
# The following attributes are read only:
# - \c %force         : A 3-tuple of floats (x, y, z) listing the current %force on the particle
# - \c virial         : A float containing the contribution of this particle to the total virial
# - \c energy         : A float containing the contribution of this particle to the total potential energy
# - \c torque         : A 3-tuple of floats (x, y, z) listing the current torque on the particle
#
class force_data_proxy(object):
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


    @property
    def force(self):
        f = self.fdata.cpp_force.getForce(self.tag);
        return (f.x, f.y, f.z);

    @property
    def virial(self):
        return (self.fdata.cpp_force.getVirial(self.tag,0),
                self.fdata.cpp_force.getVirial(self.tag,1),
                self.fdata.cpp_force.getVirial(self.tag,2),
                self.fdata.cpp_force.getVirial(self.tag,3),
                self.fdata.cpp_force.getVirial(self.tag,4),
                self.fdata.cpp_force.getVirial(self.tag,5));

    @property
    def energy(self):
        energy = self.fdata.cpp_force.getEnergy(self.tag);
        return energy;

    @property
    def torque(self):
        f = self.fdata.cpp_force.getTorque(self.tag);
        return (f.x, f.y, f.z)

## \internal
# \brief Access bond data
#
# bond_data provides access to the bonds in the system.
# This documentation is intentionally left sparse, see hoomd.data for a full explanation of how to use
# bond_data, documented by example.
#
class bond_data(hoomd.meta._metadata):
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

        # base class constructor
        hoomd.meta._metadata.__init__(self)

    ## \internal
    # \brief Add a new bond
    # \param type Type name of the bond to add
    # \param a Tag of the first particle in the bond
    # \param b Tag of the second particle in the bond
    # \returns Unique tag identifying this bond
    def add(self, type, a, b):
        typeid = self.bdata.getTypeByName(type);
        return self.bdata.addBondedGroup(_hoomd.Bond(typeid, int(a), int(b)));

    ## \internal
    # \brief Remove a bond by tag
    # \param tag Unique tag of the bond to remove
    def remove(self, tag):
        self.bdata.removeBondedGroup(tag);

    ## \var bdata
    # \internal
    # \brief BondData to which this instance is connected

    ## \internal
    # \brief Get a bond_data_proxy reference to the bond with contiguous id \a id
    # \param id Bond id to access
    def __getitem__(self, id):
        if id >= len(self) or id < 0:
            raise IndexError;
        tag = self.bdata.getNthTag(id);
        return bond_data_proxy(self.bdata, tag);

    ## \internal
    # \brief Get a bond_data_proxy reference to the bond with tag \a tag
    # \param tag Bond tag to access
    def get(self, tag):
        if tag > self.bdata.getMaximumTag() or tag < 0:
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
        self.bdata.removeBondedGroup(tag);

    ## \internal
    # \brief Get the number of bonds
    def __len__(self):
        return self.bdata.getNGlobal();

    ## \internal
    # \brief Get an informal string representing the object
    def __str__(self):
        result = "Bond Data for %d bonds of %d typeid(s)" % (self.bdata.getNGlobal(), self.bdata.getNTypes());
        return result

    ## \internal
    # \brief Return an interator
    def __iter__(self):
        return bond_data.bond_data_iterator(self);

    ## \internal
    # \brief Return metadata for this bond_data instance
    def get_metadata(self):
        data = hoomd.meta._metadata.get_metadata(self)
        data['N'] = len(self)
        data['types'] = [self.bdata.getNameByType(i) for i in range(self.bdata.getNTypes())];
        return data

## Access a single bond via a proxy
#
# bond_data_proxy provides access to all of the properties of a single bond in the system.
# This documentation is intentionally left sparse, see hoomd.data for a full explanation of how to use
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
class bond_data_proxy(object):
    ## \internal
    # \brief create a bond_data_proxy
    #
    # \param bdata BondData to which this proxy belongs
    # \param tag Tag of this bond in \a bdata
    def __init__(self, bdata, tag):
        self.bdata = bdata;
        self.tag = tag;

    ## \internal
    # \brief Get an informal string representing the object
    def __str__(self):
        result = "";
        result += "typeid       : " + str(self.typeid) + "\n";
        result += "a            : " + str(self.a) + "\n"
        result += "b            : " + str(self.b) + "\n"
        result += "type         : " + str(self.type) + "\n";
        return result;

    @property
    def a(self):
        bond = self.bdata.getGroupByTag(self.tag);
        return bond.a;

    @property
    def b(self):
        bond = self.bdata.getGroupByTag(self.tag);
        return bond.b;

    @property
    def typeid(self):
        bond = self.bdata.getGroupByTag(self.tag);
        return bond.type;

    @property
    def type(self):
        bond = self.bdata.getGroupByTag(self.tag);
        typeid = bond.type;
        return self.bdata.getNameByType(typeid);

## \internal
# \brief Access constraint data
#
# constraint_data provides access to the constraints in the system.
# This documentation is intentionally left sparse, see hoomd.data for a full explanation of how to use
# bond_data, documented by example.
#
class constraint_data(hoomd.meta._metadata):
    ## \internal
    # \brief bond_data iterator
    class constraint_data_iterator:
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
    # \brief create a constraint_data
    #
    # \param bdata ConstraintData to connect
    def __init__(self, cdata):
        self.cdata = cdata;

        # base class constructor
        hoomd.meta._metadata.__init__(self)

    ## \internal
    # \brief Add a new distance constraint
    # \param a Tag of the first particle in the bond
    # \param b Tag of the second particle in the bond
    # \param d Distance of the constraint to add
    # \returns Unique tag identifying this bond
    def add(self, a, b, d):
        return self.cdata.addBondedGroup(_hoomd.Constraint(float(d), int(a), int(b)));

    ## \internal
    # \brief Remove a bond by tag
    # \param tag Unique tag of the bond to remove
    def remove(self, tag):
        self.cdata.removeBondedGroup(tag);

    ## \var cdata
    # \internal
    # \brief ConstraintData to which this instance is connected

    ## \internal
    # \brief Get a constraint_data_proxy reference to the bond with contiguous id \a id
    # \param id Constraint id to access
    def __getitem__(self, id):
        if id >= len(self) or id < 0:
            raise IndexError;
        tag = self.cdata.getNthTag(id);
        return constraint_data_proxy(self.cdata, tag);

    ## \internal
    # \brief Get a constriant_data_proxy reference to the bond with tag \a tag
    # \param tag Bond tag to access
    def get(self, tag):
        if tag > self.cdata.getMaximumTag() or tag < 0:
            raise IndexError;
        return constraint_data_proxy(self.cdata, tag);

    ## \internal
    # \brief Set a constraint's properties
    # \param id constraint id to set
    # \param b Value containing properties to set
    def __setitem__(self, id, b):
        raise RuntimeError('Cannot change constraints once they are created');

    ## \internal
    # \brief Delete a constraint by id
    # \param id Constraint id to delete
    def __delitem__(self, id):
        if id >= len(self) or id < 0:
            raise IndexError;
        tag = self.cdata.getNthTag(id);
        self.cdata.removeBondedGroup(tag);

    ## \internal
    # \brief Get the number of bonds
    def __len__(self):
        return self.cdata.getNGlobal();

    ## \internal
    # \brief Get an informal string representing the object
    def __str__(self):
        result = "Constraint Data for %d constraints" % (self.cdata.getNGlobal());
        return result

    ## \internal
    # \brief Return an interator
    def __iter__(self):
        return constraint_data.constraint_data_iterator(self);

    ## \internal
    # \brief Return metadata for this bond_data instance
    def get_metadata(self):
        data = hoomd.meta._metadata.get_metadata(self)
        data['N'] = len(self)
        return data

## Access a single constraint via a proxy
#
# constraint_data_proxy provides access to all of the properties of a single constraint in the system.
# This documentation is intentionally left sparse, see hoomd.data for a full explanation of how to use
# constraint_data_proxy, documented by example.
#
# The following attributes are read only:
# - \c tag          : A unique integer attached to each constraint (not in any particular range). A constraint's tag remans fixed
#                     during its lifetime. (Tags previously used by removed constraint may be recycled).
# - \c d            : A float indicating the constraint distance
# - \c a            : An integer indexing the A particle in the constraint. Particle tags run from 0 to N-1;
# - \c b            : An integer indexing the B particle in the constraint. Particle tags run from 0 to N-1;
# - \c type         : A string naming the type
#
# In the current version of the API, only already defined type names can be used. A future improvement will allow
# dynamic creation of new type names from within the python API.
# \MPI_SUPPORTED
class constraint_data_proxy(object):
    ## \internal
    # \brief create a constraint_data_proxy
    #
    # \param cdata ConstraintData to which this proxy belongs
    # \param tag Tag of this constraint in \a cdata
    def __init__(self, cdata, tag):
        self.cdata = cdata;
        self.tag = tag;

    ## \internal
    # \brief Get an informal string representing the object
    def __str__(self):
        result = "";
        result += "a            : " + str(self.a) + "\n"
        result += "b            : " + str(self.b) + "\n"
        result += "d            : " + str(self.d) + "\n";
        return result;

    @property
    def a(self):
        constraint = self.cdata.getGroupByTag(self.tag);
        return constraint.a;

    @property
    def b(self):
        constraint = self.cdata.getGroupByTag(self.tag);
        return constraint.b;

    @property
    def d(self):
        constraint = self.cdata.getGroupByTag(self.tag);
        return constraint.d;

## \internal
# \brief Access angle data
#
# angle_data provides access to the angles in the system.
# This documentation is intentionally left sparse, see hoomd.data for a full explanation of how to use
# angle_data, documented by example.
#
class angle_data(hoomd.meta._metadata):
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

        # base class constructor
        hoomd.meta._metadata.__init__(self)

    ## \internal
    # \brief Add a new angle
    # \param type Type name of the angle to add
    # \param a Tag of the first particle in the angle
    # \param b Tag of the second particle in the angle
    # \param c Tag of the thrid particle in the angle
    # \returns Unique tag identifying this bond
    def add(self, type, a, b, c):
        typeid = self.adata.getTypeByName(type);
        return self.adata.addBondedGroup(_hoomd.Angle(typeid, int(a), int(b), int(c)));

    ## \internal
    # \brief Remove an angle by tag
    # \param tag Unique tag of the angle to remove
    def remove(self, tag):
        self.adata.removeBondedGroup(tag);

    ## \var adata
    # \internal
    # \brief AngleData to which this instance is connected

    ## \internal
    # \brief Get an angle_data_proxy reference to the angle with contiguous id \a id
    # \param id Angle id to access
    def __getitem__(self, id):
        if id >= len(self) or id < 0:
            raise IndexError;
        tag = self.adata.getNthTag(id);
        return angle_data_proxy(self.adata, tag);

    ## \internal
    # \brief Get a angle_data_proxy reference to the angle with tag \a tag
    # \param tag Angle tag to access
    def get(self, tag):
        if tag > self.adata.getMaximumTag() or tag < 0:
            raise IndexError;
        return angle_data_proxy(self.adata, tag);

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

    ## \internal
    # \brief Return metadata for this angle_data instance
    def get_metadata(self):
        data = hoomd.meta._metadata.get_metadata(self)
        data['N'] = len(self)
        data['types'] = [self.adata.getNameByType(i) for i in range(self.adata.getNTypes())];
        return data

## Access a single angle via a proxy
#
# angle_data_proxy provides access to all of the properties of a single angle in the system.
# This documentation is intentionally left sparse, see hoomd.data for a full explanation of how to use
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
class angle_data_proxy(object):
    ## \internal
    # \brief create a angle_data_proxy
    #
    # \param adata AngleData to which this proxy belongs
    # \param tag Tag of this angle in \a adata
    def __init__(self, adata, tag):
        self.adata = adata;
        self.tag = tag;

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

    @property
    def a(self):
        angle = self.adata.getGroupByTag(self.tag);
        return angle.a;

    @property
    def b(self):
        angle = self.adata.getGroupByTag(self.tag);
        return angle.b;

    @property
    def c(self):
        angle = self.adata.getGroupByTag(self.tag);
        return angle.c;

    @property
    def typeid(self):
        angle = self.adata.getGroupByTag(self.tag);
        return angle.type;

    @property
    def type(self):
        angle = self.adata.getGroupByTag(self.tag);
        typeid = angle.type;
        return self.adata.getNameByType(typeid);

## \internal
# \brief Access dihedral data
#
# dihedral_data provides access to the dihedrals in the system.
# This documentation is intentionally left sparse, see hoomd.data for a full explanation of how to use
# dihedral_data, documented by example.
#
class dihedral_data(hoomd.meta._metadata):
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

        # base class constructor
        hoomd.meta._metadata.__init__(self)

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
        return self.ddata.addBondedGroup(_hoomd.Dihedral(typeid, int(a), int(b), int(c), int(d)));

    ## \internal
    # \brief Remove an dihedral by tag
    # \param tag Unique tag of the dihedral to remove
    def remove(self, tag):
        self.ddata.removeBondedGroup(tag);

    ## \var ddata
    # \internal
    # \brief DihedralData to which this instance is connected

    ## \internal
    # \brief Get an dihedral_data_proxy reference to the dihedral with contiguous id \a id
    # \param id Dihedral id to access
    def __getitem__(self, id):
        if id >= len(self) or id < 0:
            raise IndexError;
        tag = self.ddata.getNthTag(id);
        return dihedral_data_proxy(self.ddata, tag);

    ## \internal
    # \brief Get a dihedral_data_proxy reference to the dihedral with tag \a tag
    # \param tag Dihedral tag to access
    def get(self, tag):
        if tag > self.ddata.getMaximumTag() or tag < 0:
            raise IndexError;
        return dihedral_data_proxy(self.ddata, tag);

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

    ## \internal
    # \brief Return metadata for this dihedral_data instance
    def get_metadata(self):
        data = hoomd.meta._metadata.get_metadata(self)
        data['N'] = len(self)
        data['types'] = [self.ddata.getNameByType(i) for i in range(self.ddata.getNTypes())];
        return data

## Access a single dihedral via a proxy
#
# dihedral_data_proxy provides access to all of the properties of a single dihedral in the system.
# This documentation is intentionally left sparse, see hoomd.data for a full explanation of how to use
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
# \MPI_SUPPORTED
class dihedral_data_proxy(object):
    ## \internal
    # \brief create a dihedral_data_proxy
    #
    # \param ddata DihedralData to which this proxy belongs
    # \param tag Tag of this dihedral in \a ddata
    def __init__(self, ddata, tag):
        self.ddata = ddata;
        self.tag = tag;

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

    @property
    def a(self):
        dihedral = self.ddata.getGroupByTag(self.tag);
        return dihedral.a;
    @property
    def b(self):
        dihedral = self.ddata.getGroupByTag(self.tag);
        return dihedral.b;

    @property
    def c(self):
        dihedral = self.ddata.getGroupByTag(self.tag);
        return dihedral.c;

    @property
    def d(self):
        dihedral = self.ddata.getGroupByTag(self.tag);
        return dihedral.d;

    @property
    def typeid(self):
        dihedral = self.ddata.getGroupByTag(self.tag);
        return dihedral.type;

    @property
    def type(self):
        dihedral = self.ddata.getGroupByTag(self.tag);
        typeid = dihedral.type;
        return self.ddata.getNameByType(typeid);

## \internal
# \brief Get data.boxdim from a SnapshotSystemData
def get_snapshot_box(snapshot):
    b = snapshot._global_box;
    L = b.getL();
    return boxdim(Lx=L.x, Ly=L.y, Lz=L.z, xy=b.getTiltFactorXY(), xz=b.getTiltFactorXZ(), yz=b.getTiltFactorYZ(), dimensions=snapshot._dimensions);

## \internal
# \brief Set data.boxdim to a SnapshotSystemData
def set_snapshot_box(snapshot, box):
    snapshot._global_box = box._getBoxDim();
    snapshot._dimensions = box.dimensions;

# Inject a box property into SnapshotSystemData that provides and accepts boxdim objects
_hoomd.SnapshotSystemData_float.box = property(get_snapshot_box, set_snapshot_box);
_hoomd.SnapshotSystemData_double.box = property(get_snapshot_box, set_snapshot_box);

## Make an empty snapshot
#
# \param N Number of particles to create
# \param box a data.boxdim object that defines the simulation box
# \param particle_types List of particle type names (must not be zero length)
# \param bond_types List of bond type names (may be zero length)
# \param angle_types List of angle type names (may be zero length)
# \param dihedral_types List of Dihedral type names (may be zero length)
# \param improper_types List of improper type names (may be zero length)
# \param dtype Data type for the real valued numpy arrays in the snapshot. Must be either 'float' or 'double'.
#
# \b Examples:
# \code
# snapshot = data.make_snapshot(N=1000, box=data.boxdim(L=10))
# snapshot = data.make_snapshot(N=64000, box=data.boxdim(L=1, dimensions=2, volume=1000), particle_types=['A', 'B'])
# snapshot = data.make_snapshot(N=64000, box=data.boxdim(L=20), bond_types=['polymer'], dihedral_types=['dihedralA', 'dihedralB'], improper_types=['improperA', 'improperB', 'improperC'])
# ... set properties in snapshot ...
# init.read_snapshot(snapshot);
# \endcode
#
# make_snapshot() creates particles with <b>default properties</b>. You must set reasonable values for particle
# properties before initializing the system with init.read_snapshot().
#
# The default properties are:
# - position 0,0,0
# - velocity 0,0,0
# - image 0,0,0
# - orientation 1,0,0,0
# - typeid 0
# - charge 0
# - mass 1.0
# - diameter 1.0
#
# make_snapshot() creates the particle, bond, angle, dihedral, and improper types with the names specified. Use these
# type names later in the job script to refer to particles (i.e. in lj.set_params).
#
# \sa hoomd.init.read_snapshot()
def make_snapshot(N, box, particle_types=['A'], bond_types=[], angle_types=[], dihedral_types=[], improper_types=[], dtype='float'):
    if dtype == 'float':
        snapshot = _hoomd.SnapshotSystemData_float();
    elif dtype == 'double':
        snapshot = _hoomd.SnapshotSystemData_double();
    else:
        raise ValueError("dtype must be either float or double");

    snapshot.box = box;
    if hoomd.comm.get_rank() == 0:
        snapshot.particles.resize(N);

    snapshot.particles.types = particle_types;
    snapshot.bonds.types = bond_types;
    snapshot.angles.types = angle_types;
    snapshot.dihedrals.types = dihedral_types;
    snapshot.impropers.types = improper_types;

    return snapshot;

## Read a snapshot from a GSD file
#
# \param filename GSD file to read the snapshot from
# \param frame Frame to read from the GSD file
#
# gsd_snapshot() opens the given GSD file and reads a snapshot from it.
#
def gsd_snapshot(filename, frame=0):
    reader = _hoomd.GSDReader(hoomd.context.exec_conf, filename, frame);
    return reader.getSnapshot();

## \class SnapshotParticleData
# \brief Snapshot that stores particle properties
#
# Users should not create SnapshotParticleData directly. Use data.make_snapshot() or system_data.take_snapshot()
# to get a snapshot of the system.
class SnapshotParticleData:
    # dummy class just to make doxygen happy

    def __init__(self):
        # doxygen even needs to see these variables to generate documentation for them
        self.N = None;
        self.position = None;
        self.velocity = None;
        self.acceleration = None;
        self.typeid = None;
        self.mass = None;
        self.charge = None;
        self.diameter = None;
        self.image = None;
        self.body = None;
        self.types = None;
        self.orientation = None;
        self.moment_inertia = None;
        self.angmom = None;

    ## \property N
    # Number of particles in the snapshot

    ## \property types
    # List of string type names (assignable)

    ## \property position
    # Nx3 numpy array containing the position of each particle (float or double)

    ## \property orientation
    # Nx4 numpy array containing the orientation quaternion of each particle (float or double)

    ## \property velocity
    # Nx3 numpy array containing the velocity of each particle (float or double)

    ## \property acceleration
    # Nx3 numpy array containing the acceleration of each particle (float or double)

    ## \property typeid
    # N length numpy array containing the type id of each particle (32-bit unsigned int)

    ## \property mass
    # N length numpy array containing the mass of each particle (float or double)

    ## \property charge
    # N length numpy array containing the charge of each particle (float or double)

    ## \property diameter
    # N length numpy array containing the diameter of each particle (float or double)

    ## \property image
    # Nx3 numpy array containing the image of each particle (32-bit int)

    ## \property body
    # N length numpy array containing the body of each particle (32-bit unsigned int)

    ## \property moment_inertia
    # Nx3 length numpy array containing the principal moments of inertia of each particle (float or double)

    ## \property angmom
    # Nx4 length numpy array containing the angular momentum quaternion of each particle (float or double)

    ## Resize the snapshot to hold N particles
    #
    # \param N new size of the snapshot
    #
    # resize() changes the size of the arrays in the snapshot to hold \a N particles. Existing particle properties are
    # preserved after the resize. Any newly created particles will have default values. After resizing,
    # existing references to the numpy arrays will be invalid, access them again
    # from `snapshot.particles.*`
    def resize(self, N):
        pass
