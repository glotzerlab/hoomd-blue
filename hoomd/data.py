# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander

R""" Access system configuration data.

Code in the data package provides high-level access to all of the particle, bond and other data that define the
current state of the system. You can use python code to directly read and modify this data, allowing you to analyze
simulation results while the simulation runs, or to create custom initial configurations with python code.

There are two ways to access the data.

1. Snapshots record the system configuration at one instant in time. You can store this state to analyze the data,
   restore it at a future point in time, or to modify it and reload it. Use snapshots for initializing simulations,
   or when you need to access or modify the entire simulation state.
2. Data proxies directly access the current simulation state. Use data proxies if you need to only touch a few
   particles or bonds at a a time.

.. rubric:: Snapshots

Relevant methods:

* :py:meth:`hoomd.data.system_data.take_snapshot()` captures a snapshot of the current system state. A snapshot is a
  copy of the simulation state. As the simulation continues to progress, data in a captured snapshot will remain
  constant.
* :py:meth:`hoomd.data.system_data.restore_snapshot()` replaces the current system state with the state stored in
  a snapshot.
* :py:meth:`hoomd.data.make_snapshot()` creates an empty snapshot that you can populate with custom data.
* :py:func:`hoomd.init.read_snapshot()` initializes a simulation from a snapshot.

Examples::

    snapshot = system.take_snapshot()
    system.restore_snapshot(snapshot)
    snapshot = data.make_snapshot(N=100, particle_types=['A', 'B'], box=data.boxdim(L=10))
    # ... populate snapshot with data ...
    init.read_snapshot(snapshot)

.. rubric:: Snapshot and MPI

In MPI simulations, the snapshot is only valid on rank 0 by default. make_snapshot, read_snapshot, and take_snapshot,
restore_snapshot are collective calls, and need to be called on all ranks. But only rank 0 can access data
in the snapshot::

    snapshot = system.take_snapshot(all=True)
    if comm.get_rank() == 0:
        s = init.create_random(N=100, phi_p=0.05);numpy.mean(snapshot.particles.velocity))
        snapshot.particles.position[0] = [1,2,3];

    system.restore_snapshot(snapshot);
    snapshot = data.make_snapshot(N=10, box=data.boxdim(L=10))
    if comm.get_rank() == 0:
        snapshot.particles.position[:] = ....
    init.read_snapshot(snapshot)

You can explicitly broadcast the information contained in the snapshot to all other ranks, using **broadcast**.

    snapshot = system.take_snapshot(all=True)
    snapshot.broadcast() # broadcast from rank 0 to all other ranks using MPI
    snapshot.broadcast_all() # broadcast from partition 0 to all other ranks and partitions using MPI

.. rubric:: Simulation box

You can access the simulation box from a snapshot::

    >>> print(snapshot.box)
    Box: Lx=17.3646569289 Ly=17.3646569289 Lz=17.3646569289 xy=0.0 xz=0.0 yz=0.0 dimensions=3

and can change it::

    >>> snapshot.box = data.boxdim(Lx=10, Ly=20, Lz=30, xy=1.0, xz=0.1, yz=2.0)
    >>> print(snapshot.box)
    Box: Lx=10 Ly=20 Lz=30 xy=1.0 xz=0.1 yz=2.0 dimensions=3

*All* particles must be inside the box before using the snapshot to initialize a simulation or restoring it.
The dimensionality of the system (2D/3D) cannot change after initialization.

.. rubric:: Particle properties

Particle properties are present in `snapshot.particles`. Each property is stored in a numpy array that directly
accesses the memory of the snapshot. References to these arrays will become invalid when the snapshot itself is
garbage collected.

* `N` is the number of particles in the particle data snapshot::

    >>> print(snapshot.particles.N)
    64000

* Change the number of particles in the snapshot with resize. Existing particle properties are
  preserved after the resize. Any newly created particles will have default values. After resizing,
  existing references to the numpy arrays will be invalid, access them again
  from `snapshot.particles.*`::

    >>> snapshot.particles.resize(1000);

* The list of all particle types in the simulation can be accessed and modified::

    >>> print(snapshot.particles.types)
    ['A', 'B', 'C']
    >>> snapshot.particles.types = ['1', '2', '3', '4'];

* Individual particles properties are stored in numpy arrays. Vector quantities are stored in Nx3 arrays of floats
  (or doubles) and scalar quantities are stored in N length 1D arrays::

    >>> print(snapshot.particles.position[10])
    [ 1.2398  -10.2687  100.6324]

* Various properties can be accessed of any particle, and the numpy arrays can be sliced or passed whole to other
  routines::

    >>> print(snapshot.particles.typeid[10])
    2
    >>> print(snapshot.particles.velocity[10])
    (-0.60267972946166992, 2.6205904483795166, -1.7868227958679199)
    >>> print(snapshot.particles.mass[10])
    1.0
    >>> print(snapshot.particles.diameter[10])
    1.0

* Particle properties can be set in the same way. This modifies the data in the snapshot, not the
  current simulation state::

    >>> snapshot.particles.position[10] = [1,2,3]
    >>> print(snapshot.particles.position[10])
    [ 1.  2.  3.]

* Snapshots store particle types as integers that index into the type name array::

    >>> print(snapshot.particles.typeid)
    [ 0.  1.  2.  0.  1.  2.  0.  1.  2.  0.]
    >>> snapshot.particles.types = ['A', 'B', 'C'];
    >>> snapshot.particles.typeid[0] = 2;   # C
    >>> snapshot.particles.typeid[1] = 0;   # A
    >>> snapshot.particles.typeid[2] = 1;   # B

For a list of all particle properties in the snapshot see :py:class:`hoomd.data.SnapshotParticleData`.

.. rubric:: Bonds

Bonds are stored in `snapshot.bonds`. :py:meth:`hoomd.data.system_data.take_snapshot()` does not record the bonds
by default, you need to request them with the argument `bonds=True`.

* `N` is the number of bonds in the bond data snapshot::

    >>> print(snapshot.bonds.N)
    100

* Change the number of bonds in the snapshot with resize. Existing bonds are
  preserved after the resize. Any newly created bonds will be initialized to 0. After resizing,
  existing references to the numpy arrays will be invalid, access them again
  from `snapshot.bonds.*`::

    >>> snapshot.bonds.resize(1000);

* Bonds are stored in an Nx2 numpy array `group`. The first axis accesses the bond `i`. The second axis `j` goes over
  the individual particles in the bond. The value of each element is the tag of the particle participating in the
  bond::

    >>> print(snapshot.bonds.group)
    [[0 1]
    [1 2]
    [3 4]
    [4 5]]
    >>> snapshot.bonds.group[0] = [10,11]

* Snapshots store bond types as integers that index into the type name array::

    >>> print(snapshot.bonds.typeid)
    [ 0.  1.  2.  0.  1.  2.  0.  1.  2.  0.]
    >>> snapshot.bonds.types = ['A', 'B', 'C'];
    >>> snapshot.bonds.typeid[0] = 2;   # C
    >>> snapshot.bonds.typeid[1] = 0;   # A
    >>> snapshot.bonds.typeid[2] = 1;   # B


.. rubric:: Angles, dihedrals and impropers

Angles, dihedrals, and impropers are stored similar to bonds. The only difference is that the group array is sized
appropriately to store the number needed for each type of bond.

* `snapshot.angles.group` is Nx3
* `snapshot.dihedrals.group` is Nx4
* `snapshot.impropers.group` is Nx4

.. rubric:: Special pairs

Special pairs are exactly handled like bonds. The snapshot entry is called **pairs**.

.. rubric:: Constraints

Pairwise distance constraints are added and removed like bonds. They are defined between two particles.
The only difference is that instead of a type, constraints take a distance as parameter.

* `N` is the number of constraints in the constraint data snapshot::

    >>> print(snapshot.constraints.N)
    99

* Change the number of constraints in the snapshot with resize. Existing constraints are
  preserved after the resize. Any newly created constraints will be initialized to 0. After resizing,
  existing references to the numpy arrays will be invalid, access them again
  from `snapshot.constraints.*`::

    >>> snapshot.constraints.resize(1000);

* Bonds are stored in an Nx2 numpy array `group`. The first axis accesses the constraint `i`. The second axis `j` goes over
  the individual particles in the constraint. The value of each element is the tag of the particle participating in the
  constraint::

    >>> print(snapshot.constraints.group)
    [[4 5]
    [6 7]
    [6 8]
    [7 8]]
    >>> snapshot.constraints.group[0] = [10,11]

* Snapshots store constraint distances as floats::

    >>> print(snapshot.constraints.value)
    [ 1.5 2.3 1.0 0.1 ]

.. rubric:: data_proxy Proxy access

For most of the cases below, it is assumed that the result of the initialization command was saved at the beginning
of the script::

    system = init.read_xml(filename="input.xml")

Warning:
    The performance of the proxy access is very slow. Use snapshots to access the whole system configuration
    efficiently.

.. rubric:: Simulation box

You can access the simulation box::

    >>> print(system.box)
    Box: Lx=17.3646569289 Ly=17.3646569289 Lz=17.3646569289 xy=0.0 xz=0.0 yz=0.0

and can change it::

    >>> system.box = data.boxdim(Lx=10, Ly=20, Lz=30, xy=1.0, xz=0.1, yz=2.0)
    >>> print(system.box)
    Box: Lx=10 Ly=20 Lz=30 xy=1.0 xz=0.1 yz=2.0

**All** particles must **always** remain inside the box. If a box is set in this way such that a particle ends up outside of the box, expect
errors to be thrown or for hoomd to just crash. The dimensionality of the system cannot change after initialization.

.. rubric:: Particle properties

For a list of all particle properties that can be read and/or set, see :py:class:`hoomd.data.particle_data_proxy`.
The examples here only demonstrate changing a few of them.

``system.particles`` is a window into all of the particles in the system.
It behaves like standard python list in many ways.

* Its length (the number of particles in the system) can be queried::

    >>> len(system.particles)
    64000

* A short summary can be printed of the list::

    >>> print(system.particles)
    Particle Data for 64000 particles of 1 type(s)

* The list of all particle types in the simulation can be accessed::

    >>> print(system.particles.types)
    ['A']
    >>> print system.particles.types
    Particle types: ['A']

* Particle types can be added between :py:func:`hoomd.run()` commands::

    >>> system.particles.types.add('newType')

* Individual particles can be accessed at random::

    >>> i = 4
    >>> p = system.particles[i]

* Various properties can be accessed of any particle (note that p can be replaced with system.particles[i]
  and the results are the same)::

    >>> p.tag
    4
    >>> p.position
    (27.296911239624023, -3.5986068248748779, 10.364067077636719)
    >>> p.velocity
    (-0.60267972946166992, 2.6205904483795166, -1.7868227958679199)
    >>> p.mass
    1.0
    >>> p.diameter
    1.0
    >>> p.type
    'A'
    >>> p.tag
    4

* Particle properties can be set in the same way::

    >>> p.position = (1,2,3)
    >>> p.position
    (1.0, 2.0, 3.0)

* Finally, all particles can be easily looped over::

    for p in system.particles:
        p.velocity = (0,0,0)

Particles may be added at any time in the job script, and a unique tag is returned::

    >>> system.particles.add('A')
    >>> t = system.particles.add('B')

Particles may be deleted by index::

    >>> del system.particles[0]
    >>> print(system.particles[0])
    tag         : 1
    position    : (23.846603393554688, -27.558368682861328, -20.501256942749023)
    image       : (0, 0, 0)
    velocity    : (0.0, 0.0, 0.0)
    acceleration: (0.0, 0.0, 0.0)
    charge      : 0.0
    mass        : 1.0
    diameter    : 1.0
    type        : A
    typeid      : 0
    body        : 4294967295
    orientation : (1.0, 0.0, 0.0, 0.0)
    net_force   : (0.0, 0.0, 0.0)
    net_energy  : 0.0
    net_torque  : (0.0, 0.0, 0.0)

Note:
    The particle with tag 1 is now at index 0. No guarantee is made about how the
    order of particles by index will or will not change, so do not write any job scripts which assume
    a given ordering.

To access particles in an index-independent manner, use their tags. For example, to remove all particles
of type 'A', do::

    tags = []
    for p in system.particles:
        if p.type == 'A'
            tags.append(p.tag)

Then remove each of the particles by their unique tag::

    for t in tags:
        system.particles.remove(t)

Particles can also be accessed through their unique tag::

    t = system.particles.add('A')
    p = system.particles.get(t)

Any defined group can be used in exactly the same way as ``system.particles`` above, only the particles accessed
will be those just belonging to the group. For a specific example, the following will set the velocity of all
particles of type A to 0::

    groupA = group.type(name="a-particles", type='A')
    for p in groupA:
        p.velocity = (0,0,0)

.. rubric:: Bond Data

Bonds may be added at any time in the job script::

    >>> system.bonds.add("bondA", 0, 1)
    >>> system.bonds.add("bondA", 1, 2)
    >>> system.bonds.add("bondA", 2, 3)
    >>> system.bonds.add("bondA", 3, 4)

Individual bonds may be accessed by index::

    >>> bnd = system.bonds[0]
    >>> print(bnd)
    tag          : 0
    typeid       : 0
    a            : 0
    b            : 1
    type         : bondA
    >>> print(bnd.type)
    bondA
    >>> print(bnd.a)
    0
    >>> print(bnd.b)
    1

Warning:
    The order in which bonds appear by index is not static and may change at any time!

Bonds may be deleted by index::

    >>> del system.bonds[0]
    >>> print(system.bonds[0])
    tag          : 3
    typeid       : 0
    a            : 3
    b            : 4
    type         : bondA

To access bonds in an index-independent manner, use their tags. For example, to delete all bonds which connect to
particle 2, first loop through the bonds and build a list of bond tags that match the criteria::

    tags = []
    for b in system.bonds:
        if b.a == 2 or b.b == 2:
            tags.append(b.tag)

Then remove each of the bonds by their unique tag::

    for t in tags:
        system.bonds.remove(t)

Bonds can also be accessed through their unique tag::

    t = system.bonds.add('polymer',0,1)
    p = system.bonds.get(t)

.. rubric:: Angle, Dihedral, and Improper Data

Angles, Dihedrals, and Impropers may be added at any time in the job script::

    >>> system.angles.add("angleA", 0, 1, 2)
    >>> system.dihedrals.add("dihedralA", 1, 2, 3, 4)
    >>> system.impropers.add("dihedralA", 2, 3, 4, 5)

Individual angles, dihedrals, and impropers may be accessed, deleted by index or removed by tag with the same syntax
as described for bonds, just replace *bonds* with *angles*, *dihedrals*, or, *impropers* and access the
appropriate number of tag elements (a,b,c for angles) (a,b,c,d for dihedrals/impropers).

.. rubric:: Constraints

Constraints may be added and removed from within the job script.

To add a constraint of length 1.5 between particles 0 and 1::

    >>> t = system.constraints.add(0, 1, 1.5)

To remove it again::

    >>> system.constraints.remove(t)

.. rubric:: Forces

Forces can be accessed in a similar way::

    >>> lj = pair.lj(r_cut=3.0)
    >>> lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
    >>> print(lj.forces[0])
    tag         : 0
    force       : (-0.077489577233791351, -0.029512746259570122, -0.13215918838977814)
    virial      : -0.0931386947632
    energy      : -0.0469368174672
    >>> f0 = lj.forces[0]
    >>> print(f0.force)
    (-0.077489577233791351, -0.029512746259570122, -0.13215918838977814)
    >>> print(f0.virial)
    -0.093138694763n
    >>> print(f0.energy)
    -0.0469368174672

In this manner, forces due to the lj pair force, bonds, and any other force commands in hoomd can be accessed
independently from one another. See :py:class:`hoomd.data.force_data_proxy` for a definition of each data field.

.. Proxy references

For advanced code using the particle data access from python, it is important to understand that the hoomd
particles, forces, bonds, et cetera, are accessed as proxies. This means that after::

    p = system.particles[i]

is executed, *p* **does not** store the position, velocity, ... of particle *i*. Instead, it stores *i* and
provides an interface to get/set the properties on demand. This has some side effects you need to be aware of.

* First, it means that *p* (or any other proxy reference) always references the current state of the particle.
  As an example, note how the position of particle p moves after the run() command::

    >>> p.position
    (-21.317455291748047, -23.883811950683594, -22.159387588500977)
    >>> run(1000)
    ** starting run **
    ** run complete **
    >>> p.position
    (-19.774742126464844, -23.564577102661133, -21.418502807617188)

* Second, it means that copies of the proxy reference cannot be changed independently::

    p.position
    >>> a = p
    >>> a.position
    (-19.774742126464844, -23.564577102661133, -21.418502807617188)
    >>> p.position = (0,0,0)
    >>> a.position
    (0.0, 0.0, 0.0)

"""

from hoomd import _hoomd
import hoomd

class boxdim(hoomd.meta._metadata):
    R""" Define box dimensions.

    Args:
        Lx (float): box extent in the x direction (distance units)
        Ly (float): box extent in the y direction (distance units)
        Lz (float): box extent in the z direction (distance units)
        xy (float): tilt factor xy (dimensionless)
        xz (float): tilt factor xz (dimensionless)
        yz (float): tilt factor yz (dimensionless)
        dimensions (int): Number of dimensions in the box (2 or 3).
        L (float): shorthand for specifying Lx=Ly=Lz=L (distance units)
        volume (float): Scale the given box dimensions up to the this volume (area if dimensions=2)

    Simulation boxes in hoomd are specified by six parameters, *Lx*, *Ly*, *Lz*, *xy*, *xz* and *yz*. For full details,
    see :ref:`boxdim`. A boxdim provides a way to specify all six parameters for a given box and perform some common
    operations with them. Modifying a boxdim does not modify the underlying simulation box in hoomd. A boxdim can be passed
    to an initialization method or to assigned to a saved sysdef variable (``system.box = new_box``) to set the simulation
    box.

    Access attributes directly::

        b = data.boxdim(L=20)
        b.xy = 1.0
        b.yz = 0.5
        b.Lz = 40


    .. rubric:: Two dimensional systems

    2D simulations in hoomd are embedded in 3D boxes with short heights in the z direction. To create a 2D box,
    set dimensions=2 when creating the boxdim. This will force Lz=1 and xz=yz=0. init commands that support 2D boxes
    will pass the dimensionality along to the system. When you assign a new boxdim to an already initialized system,
    the dimensionality flag is ignored. Changing the number of dimensions during a simulation run is not supported.

    In 2D boxes, *volume* is in units of area.

    .. rubric:: Shorthand notation

    data.boxdim accepts the keyword argument ``L=x`` as shorthand notation for ``Lx=x, Ly=x, Lz=x`` in 3D
    and ``Lx=x, Ly=x, Lz=1`` in 2D. If you specify both ``L`` and ``Lx``, ``Ly``, or ``Lz``, then the value for ``L`` will override
    the others.

    Examples:

    * Cubic box with given volume: ``data.boxdim(volume=V)``
    * Triclinic box in 2D with given area: ``data.boxdim(xy=1.0, dimensions=2, volume=A)``
    * Rectangular box in 2D with given area and aspect ratio: ``data.boxdim(Lx=1, Ly=aspect, dimensions=2, volume=A)``
    * Cubic box with given length: ``data.boxdim(L=10)``
    * Fully define all box parameters: ``data.boxdim(Lx=10, Ly=20, Lz=30, xy=1.0, xz=0.5, yz=0.1)``
    """
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

    def scale(self, sx=1.0, sy=1.0, sz=1.0, s=None):
        R""" Scale box dimensions.

        Args:
            sx (float): scale factor in the x direction
            sy (float): scale factor in the y direction
            sz (float): scale factor in the z direction
            s (float): Shorthand for sx=s, sy=x, sz=s

        Scales the box by the given scale factors. Tilt factors are not modified.

        Returns:
            A reference to the modified box.
        """
        if s is not None:
            sx = s;
            sy = s;
            sz = s;

        self.Lx = self.Lx * sx;
        self.Ly = self.Ly * sy;
        self.Lz = self.Lz * sz;
        return self

    def set_volume(self, volume):
        R""" Set the box volume.

        Args:
            volume (float): new box volume (area if dimensions=2)

        Scale the box to the given volume (or area).

        Returns:
            A reference to the modified box.
        """

        cur_vol = self.get_volume();

        if self.dimensions == 3:
            s = (volume / cur_vol)**(1.0/3.0)
            self.scale(s, s, s);
        else:
            s = (volume / cur_vol)**(1.0/2.0)
            self.scale(s, s, 1.0);
        return self

    def get_volume(self):
        R""" Get the box volume.

        Returns:
            The box volume (area in 2D).
        """
        b = self._getBoxDim();
        return b.getVolume(self.dimensions == 2);

    def get_lattice_vector(self,i):
        R""" Get a lattice vector.

        Args:
            i (int): (=0,1,2) direction of lattice vector

        Returns:
            The lattice vector (3-tuple) along direction *i*.
        """

        b = self._getBoxDim();
        v = b.getLatticeVector(int(i))
        return (v.x, v.y, v.z)

    def wrap(self,v, img=(0,0,0)):
        R""" Wrap a vector using the periodic boundary conditions.

        Args:
            v (tuple): The vector to wrap
            img (tuple): A vector of integer image flags that will be updated (optional)

        Returns:
            The wrapped vector and the image flags in a tuple.
        """
        u = _hoomd.make_scalar3(float(v[0]),float(v[1]),float(v[2]))
        i = _hoomd.make_int3(int(img[0]),int(img[1]),int(img[2]))
        c = _hoomd.make_char3(0,0,0)
        self._getBoxDim().wrap(u,i,c)
        img = (i.x,i.y,i.z)
        return (u.x, u.y, u.z),img

    def min_image(self,v):
        R""" Apply the minimum image convention to a vector using periodic boundary conditions.

        Args:
            v (tuple): The vector to apply minimum image to

        Returns:
            The minimum image as a tuple.

        """
        u = _hoomd.make_scalar3(v[0],v[1],v[2])
        u = self._getBoxDim().minImage(u)
        return (u.x, u.y, u.z)

    def make_fraction(self,v):
        R""" Scale a vector to fractional coordinates.

        Args:
            v (tuple): The vector to convert to fractional coordinates

        make_fraction() takes a vector in a box and computes a vector where all components are
        between 0 and 1.

        Returns:
            The scaled vector.
        """
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

class system_data(hoomd.meta._metadata):
    R""" Access system data

    system_data provides access to the different data structures that define the current state of the simulation.
    See :py:mod:`hoomd.data` for a full explanation of how to use by example.

    Attributes:
        box (:py:class:`hoomd.data.boxdim`)
        particles (:py:class:`hoomd.data.particle_data_proxy`)
        bonds (:py:class:`hoomd.data.bond_data_proxy`)
        angles (:py:class:`hoomd.data.angle_data_proxy`)
        dihedrals (:py:class:`hoomd.data.dihedral_data_proxy`)
        impropers (:py:class:`hoomd.data.dihedral_data_proxy`)
        constraint (:py:class:`hoomd.data.constraint_data_proxy`)
        pairs (:py:class:`hoomd.data.bond_data_proxy`)
            .. versionadded:: 2.1
    """

    def __init__(self, sysdef):
        self.sysdef = sysdef;
        self.particles = particle_data(sysdef.getParticleData());
        self.bonds = bond_data(sysdef.getBondData());
        self.angles = angle_data(sysdef.getAngleData());
        self.dihedrals = dihedral_data(sysdef.getDihedralData());
        self.impropers = dihedral_data(sysdef.getImproperData());
        self.constraints = constraint_data(sysdef.getConstraintData());
        self.pairs = bond_data(sysdef.getPairData());

        # base class constructor
        hoomd.meta._metadata.__init__(self)

    def take_snapshot(self,
                      particles=True,
                      bonds=False,
                      pairs=False,
                      integrators=False,
                      all=False,
                      dtype='float'):
        R""" Take a snapshot of the current system data.

        Args:
            particles (bool): When True, particle data is included in the snapshot.
            bonds (bool): When true, bond, angle, dihedral, improper and constraint data is included.
            pairs (bool): When true, special pair data is included
                .. versionadded:: 2.1

            integrators (bool): When true, integrator data is included the snapshot.
            all (bool): When true, the entire system state is saved in the snapshot.
            dtype (str): Datatype for the snapshot numpy arrays. Must be either 'float' or 'double'.

        Returns:
            The snapshot object.

        This functions returns a snapshot object. It contains the current.
        partial or complete simulation state. With appropriate options
        it is possible to select which data properties should be included
        in the snapshot

        Examples::

            snapshot = system.take_snapshot()
            snapshot = system.take_snapshot()
            snapshot = system.take_snapshot(bonds=true)

        """
        hoomd.util.print_status_line();

        if all is True:
                particles=True
                bonds=True
                pairs=True
                integrators=True

        # take the snapshot
        if dtype == 'float':
            cpp_snapshot = self.sysdef.takeSnapshot_float(particles,bonds,bonds,bonds,bonds,bonds,integrators,pairs)
        elif dtype == 'double':
            cpp_snapshot = self.sysdef.takeSnapshot_double(particles,bonds,bonds,bonds,bonds,bonds,integrators,pairs)
        else:
            raise ValueError("dtype must be float or double");

        return cpp_snapshot

    def replicate(self, nx=1, ny=1, nz=1):
        R""" Replicates the system along the three spatial dimensions.

        Args:
            nx (int): Number of times to replicate the system along the x-direction
            ny (int): Number of times to replicate the system along the y-direction
            nz (int): Number of times to replicate the system along the z-direction

        This method replicates particles along all three spatial directions, as
        opposed to replication implied by periodic boundary conditions.
        The box is resized and the number of particles is updated so that the new box
        holds the specified number of replicas of the old box along all directions.
        Particle coordinates are updated accordingly to fit into the new box. All velocities and
        other particle properties are replicated as well. Also bonded groups between particles
        are replicated.

        Examples::

            system = init.read_xml("some_file.xml")
            system.replicate(nx=2,ny=2,nz=2)


        Note:
            The dimensions of the processor grid are not updated upon replication. For example, if an initially
            cubic box is replicated along only one spatial direction, this could lead to decreased performance
            if the processor grid was optimal for the original box dimensions, but not for the new ones.

        """
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

    def restore_snapshot(self, snapshot):
        R""" Re-initializes the system from a snapshot.

        Args:
            snapshot:. The snapshot to initialize the system from.

        Snapshots temporarily store system data. Snapshots contain the complete simulation state in a
        single object. They can be used to restart a simulation.

        Example use cases in which a simulation may be restarted from a snapshot include python-script-level
        Monte-Carlo schemes, where the system state is stored after a move has been accepted (according to
        some criterion), and where the system is re-initialized from that same state in the case
        when a move is not accepted.

        Example::

            system = init.read_xml("some_file.xml")

            ... run a simulation ...

            snapshot = system.take_snapshot(all=True)
            ...
            system.restore_snapshot(snapshot)

        Warning:
                restore_snapshot() may invalidate force coefficients, neighborlist r_cut values, and other per type
                quantities if called within a callback during a run(). You can restore a snapshot during a run only
                if the snapshot is of a previous state of the currently running system. Otherwise, you need to use
                restore_snapshot() between run() commands to ensure that all per type coefficients are updated properly.

        """
        hoomd.util.print_status_line();

        if hoomd.comm.get_rank() == 0:
            if snapshot.has_particle_data and len(snapshot.particles.types) != self.sysdef.getParticleData().getNTypes():
                raise RuntimeError("Number of particle types must remain the same")
            if snapshot.has_bond_data and len(snapshot.bonds.types) != self.sysdef.getBondData().getNTypes():
                raise RuntimeError("Number of bond types must remain the same")
            if snapshot.has_angle_data and len(snapshot.angles.types) != self.sysdef.getAngleData().getNTypes():
                raise RuntimeError("Number of angle types must remain the same")
            if snapshot.has_dihedral_data and len(snapshot.dihedrals.types) != self.sysdef.getDihedralData().getNTypes():
                raise RuntimeError("Number of dihedral types must remain the same")
            if snapshot.has_improper_data and len(snapshot.impropers.types) != self.sysdef.getImproperData().getNTypes():
                raise RuntimeError("Number of dihedral types must remain the same")
            if snapshot.has_pair_data and len(snapshot.pairs.types) != self.sysdef.getPairData().getNTypes():
                raise RuntimeError("Number of pair types must remain the same")

        self.sysdef.initializeFromSnapshot(snapshot);

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
        data['pairs'] = self.pairs

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
    # \brief Return an iterator
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

class particle_data_proxy(object):
    R""" Access a single particle via a proxy.

    particle_data_proxy provides access to all of the properties of a single particle in the system.
    See :py:mod:`hoomd.data` for examples.

    Attributes:
        tag (int): A unique name for the particle in the system. Tags run from 0 to N-1.
        acceleration (tuple): A 3-tuple of floats (x, y, z). Acceleration is a calculated quantity and cannot be set. (in acceleration units)
        typeid (int): The type id of the particle.
        position (tuple): (x, y, z) (float, in distance units).
        image (tuple): (x, y, z) (int).
        velocity (tuple): (x, y, z) (float, in velocity units).
        charge (float): Particle charge.
        mass (float): (in mass units).
        diameter (float): (in distance units).
        type (str): Particle type name.
        body (int): Body id. -1 for free particles, 0 or larger for rigid bodies, and -2 or lesser for floppy bodies.
        orientation (tuple) : (w,x,y,z) (float, quaternion).
        net_force (tuple): Net force on particle (x, y, z) (float, in force units).
        net_energy (float): Net contribution of particle to the potential energy (in energy units).
        net_torque (tuple): Net torque on the particle (x, y, z) (float, in torque units).
        net_virial (tuple): Net virial for the particle (xx,yy,zz, xy, xz, yz)
    """

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
        result += "net_virial  : " + str(self.net_virial) + "\n";
        return result;

    @property
    def position(self):
        pos = self.pdata.getPosition(self.tag);
        return (pos.x, pos.y, pos.z);

    @position.setter
    def position(self, value):
        if len(value) != 3:
            raise ValueError("The input value/vector should be exactly length 3.")
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
        if len(value) != 3:
            raise ValueError("The input value/vector should be exactly length 3.")
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
        if len(value) != 3:
            raise ValueError("The input value/vector should be exactly length 3.")
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
        self.pdata.setDiameter(self.tag, float(value));

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
        if len(value) != 4:
            raise ValueError("The input value/vector should be exactly length 4.")
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
        if len(value) != 4:
            raise ValueError("The input value/vector should be exactly length 4.")
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
        if len(value) != 3:
            raise ValueError("The input value/vector should be exactly length 3.")
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
    def net_virial(self):
        v = (self.pdata.getPNetVirial(self.tag,0),
             self.pdata.getPNetVirial(self.tag,1),
             self.pdata.getPNetVirial(self.tag,2),
             self.pdata.getPNetVirial(self.tag,3),
             self.pdata.getPNetVirial(self.tag,4),
             self.pdata.getPNetVirial(self.tag,5));
        return v

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
    # \brief Return an iterator
    def __iter__(self):
        return force_data.force_data_iterator(self);

class force_data_proxy(object):
    R""" Access the force on a single particle via a proxy.

    force_data_proxy provides access to the current force, virial, and energy of a single particle due to a single
    force computation. See :py:mod:`hoomd.data` for examples.

    Attributes:
        force (tuple): (float, x, y, z) - the current force on the particle (force units)
        virial (tuple): This particle's contribution to the total virial tensor.
        energy (float): This particle's contribution to the total potential energy (energy units)
        torque (float): (float x, y, z) - current torque on the particle (torque units)

    """
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
    # \brief Return an iterator
    def __iter__(self):
        return bond_data.bond_data_iterator(self);

    ## \internal
    # \brief Return metadata for this bond_data instance
    def get_metadata(self):
        data = hoomd.meta._metadata.get_metadata(self)
        data['N'] = len(self)
        data['types'] = [self.bdata.getNameByType(i) for i in range(self.bdata.getNTypes())];
        return data

class bond_data_proxy(object):
    R""" Access a single bond via a proxy.

    bond_data_proxy provides access to all of the properties of a single bond in the system.
    See :py:mod:`hoomd.data` for examples.

    Attributes:
        tag (int): A unique integer attached to each bond (not in any particular range). A bond's tag remains fixed
                   during its lifetime. (Tags previously used by removed bonds may be recycled).
        typeid (int): Type id of the bond.
        a (int): The tag of the first particle in the bond.
        b (int): The tag of the second particle in the bond.
        type (str): Bond type name.

    In the current version of the API, only already defined type names can be used. A future improvement will allow
    dynamic creation of new type names from within the python API.
    """

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
    # \brief Get a constraint_data_proxy reference to the bond with tag \a tag
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
    # \brief Return an iterator
    def __iter__(self):
        return constraint_data.constraint_data_iterator(self);

    ## \internal
    # \brief Return metadata for this bond_data instance
    def get_metadata(self):
        data = hoomd.meta._metadata.get_metadata(self)
        data['N'] = len(self)
        return data

class constraint_data_proxy(object):
    R""" Access a single constraint via a proxy.

    constraint_data_proxy provides access to all of the properties of a single constraint in the system.
    See :py:mod:`hoomd.data` for examples.

    Attributes:
        tag (int): A unique integer attached to each constraint (not in any particular range). A constraint's tag remains fixed
                   during its lifetime. (Tags previously used by removed constraints may be recycled).
        d (float): The constraint distance.
        a (int): The tag of the first particle in the constraint.
        b (int): The tag of the second particle in the constraint.

    """
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
    # \param c Tag of the third particle in the angle
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
    # \brief Return an iterator
    def __iter__(self):
        return angle_data.angle_data_iterator(self);

    ## \internal
    # \brief Return metadata for this angle_data instance
    def get_metadata(self):
        data = hoomd.meta._metadata.get_metadata(self)
        data['N'] = len(self)
        data['types'] = [self.adata.getNameByType(i) for i in range(self.adata.getNTypes())];
        return data

class angle_data_proxy(object):
    R""" Access a single angle via a proxy.

    angle_data_proxy provides access to all of the properties of a single angle in the system.
    See :py:mod:`hoomd.data` for examples.

    Attributes:
        tag (int): A unique integer attached to each angle (not in any particular range). A angle's tag remains fixed
                   during its lifetime. (Tags previously used by removed angles may be recycled).
        typeid (int): Type id of the angle.
        a (int): The tag of the first particle in the angle.
        b (int): The tag of the second particle in the angle.
        c (int): The tag of the third particle in the angle.
        type (str): angle type name.

    In the current version of the API, only already defined type names can be used. A future improvement will allow
    dynamic creation of new type names from within the python API.
    """

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
    # \param c Tag of the third particle in the dihedral
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
    # \brief Return an iterator
    def __iter__(self):
        return dihedral_data.dihedral_data_iterator(self);

    ## \internal
    # \brief Return metadata for this dihedral_data instance
    def get_metadata(self):
        data = hoomd.meta._metadata.get_metadata(self)
        data['N'] = len(self)
        data['types'] = [self.ddata.getNameByType(i) for i in range(self.ddata.getNTypes())];
        return data

class dihedral_data_proxy(object):
    R""" Access a single dihedral via a proxy.

    dihedral_data_proxy provides access to all of the properties of a single dihedral in the system.
    See :py:mod:`hoomd.data` for examples.

    Attributes:
        tag (int): A unique integer attached to each dihedral (not in any particular range). A dihedral's tag remains fixed
                   during its lifetime. (Tags previously used by removed dihedrals may be recycled).
        typeid (int): Type id of the dihedral.
        a (int): The tag of the first particle in the dihedral.
        b (int): The tag of the second particle in the dihedral.
        c (int): The tag of the third particle in the dihedral.
        d (int): The tag of the fourth particle in the dihedral.
        type (str): dihedral type name.

    In the current version of the API, only already defined type names can be used. A future improvement will allow
    dynamic creation of new type names from within the python API.
    """

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

## \internal
# \brief Broadcast snapshot to all ranks
def broadcast_snapshot(cpp_snapshot):
    hoomd.context._verify_init();
    hoomd.util.print_status_line();
    # broadcast from rank 0
    cpp_snapshot._broadcast(0, hoomd.context.exec_conf);

## \internal
# \brief Broadcast snapshot to all ranks
def broadcast_snapshot_all(cpp_snapshot):
    hoomd.context._verify_init();
    hoomd.util.print_status_line();
    # broadcast from rank 0
    cpp_snapshot._broadcast_all(0, hoomd.context.exec_conf);

# Inject a box property into SnapshotSystemData that provides and accepts boxdim objects
_hoomd.SnapshotSystemData_float.box = property(get_snapshot_box, set_snapshot_box);
_hoomd.SnapshotSystemData_double.box = property(get_snapshot_box, set_snapshot_box);

# Inject broadcast methods into SnapshotSystemData
_hoomd.SnapshotSystemData_float.broadcast = broadcast_snapshot
_hoomd.SnapshotSystemData_double.broadcast = broadcast_snapshot
_hoomd.SnapshotSystemData_float.broadcast_all = broadcast_snapshot_all
_hoomd.SnapshotSystemData_double.broadcast_all = broadcast_snapshot_all

def make_snapshot(N, box, particle_types=['A'], bond_types=[], angle_types=[], dihedral_types=[], improper_types=[], pair_types=[], dtype='float'):
    R""" Make an empty snapshot.

    Args:
        N (int): Number of particles to create.
        box (:py:class:`hoomd.data.boxdim`): Simulation box parameters.
        particle_types (list): Particle type names (must not be zero length).
        bond_types (list): Bond type names (may be zero length).
        angle_types (list): Angle type names (may be zero length).
        dihedral_types (list): Dihedral type names (may be zero length).
        improper_types (list): Improper type names (may be zero length).
        pair_types(list): Special pair type names (may be zero length).
            .. versionadded:: 2.1
        dtype (str): Data type for the real valued numpy arrays in the snapshot. Must be either 'float' or 'double'.

    Examples::

        snapshot = data.make_snapshot(N=1000, box=data.boxdim(L=10))
        snapshot = data.make_snapshot(N=64000, box=data.boxdim(L=1, dimensions=2, volume=1000), particle_types=['A', 'B'])
        snapshot = data.make_snapshot(N=64000, box=data.boxdim(L=20), bond_types=['polymer'], dihedral_types=['dihedralA', 'dihedralB'], improper_types=['improperA', 'improperB', 'improperC'])
        ... set properties in snapshot ...
        init.read_snapshot(snapshot);

    :py:func:`hoomd.data.make_snapshot()` creates all particles with **default properties**. You must set reasonable
    values for particle properties before initializing the system with :py:func:`hoomd.init.read_snapshot()`.

    The default properties are:

    * position 0,0,0
    * velocity 0,0,0
    * image 0,0,0
    * orientation 1,0,0,0
    * typeid 0
    * charge 0
    * mass 1.0
    * diameter 1.0

    See Also:
        :py:func:`hoomd.init.read_snapshot()`
    """
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
    snapshot.pairs.types = pair_types;

    return snapshot;

def gsd_snapshot(filename, frame=0):
    R""" Read a snapshot from a GSD file.

    Args:
        filename (str): GSD file to read the snapshot from.
        frame (int): Frame to read from the GSD file. Negative values index from the end of the file.

    :py:func:`hoomd.data.gsd_snapshot()` opens the given GSD file and reads a snapshot from it.
    """
    hoomd.context._verify_init();

    reader = _hoomd.GSDReader(hoomd.context.exec_conf, filename, abs(frame), frame < 0);
    return reader.getSnapshot();


# Note: SnapshotParticleData should never be instantiated, it is a placeholder to generate sphinx documentation,
# as the real SnapshotParticleData lives in c++.
class SnapshotParticleData:
    R""" Snapshot of particle data properties.

    Users should not create SnapshotParticleData directly. Use :py:func:`hoomd.data.make_snapshot()`
    or :py:meth:`hoomd.data.system_data.take_snapshot()` to make snapshots.

    Attributes:
        N (int): Number of particles in the snapshot
        types (list): List of string type names (assignable)
        position (numpy.ndarray): (Nx3) numpy array containing the position of each particle (float or double)
        orientation (numpy.ndarray): (Nx4) numpy array containing the orientation quaternion of each particle (float or double)
        velocity (numpy.ndarray): (Nx3) numpy array containing the velocity of each particle (float or double)
        acceleration (numpy.ndarray): (Nx3) numpy array containing the acceleration of each particle (float or double)
        typeid (numpy.ndarray): Length N numpy array containing the type id of each particle (32-bit unsigned int)
        mass (numpy.ndarray): Length N numpy array containing the mass of each particle (float or double)
        charge (numpy.ndarray): Length N numpy array containing the charge of each particle (float or double)
        diameter (numpy.ndarray): Length N numpy array containing the diameter of each particle (float or double)
        image (numpy.ndarray): (Nx3) numpy array containing the image of each particle (32-bit int)
        body (numpy.ndarray): Length N numpy array containing the body of each particle (32-bit unsigned int). -1 indicates a free particle, and larger negative numbers indicate floppy bodies.
        moment_inertia (numpy.ndarray): (Nx3) numpy array containing the principal moments of inertia of each particle (float or double)
        angmom (numpy.ndarray): (Nx4) numpy array containing the angular momentum quaternion of each particle (float or double)

    See Also:
        :py:mod:`hoomd.data`
    """

    def resize(self, N):
        R""" Resize the snapshot to hold N particles.

        Args:
            N (int): new size of the snapshot.

        :py:meth:`resize()` changes the size of the arrays in the snapshot to hold *N* particles. Existing particle
        properties are preserved after the resize. Any newly created particles will have default values. After resizing,
        existing references to the numpy arrays will be invalid, access them again
        from `snapshot.particles.*`
        """
        pass
