# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

R""" Deprecated initialization routines.
"""

from hoomd.deprecated import _deprecated;
import hoomd;
import math
import os
from hoomd import _hoomd

def read_xml(filename, restart = None, time_step = None, wrap_coordinates = False):
    R""" ## Reads initial system state from an XML file

    Args:
        filename (str): File to read
        restart (str): If it exists, read *restart* instead of *filename*.
        time_step (int): (if specified) Time step number to use instead of the one stored in the XML file
        wrap_coordinates (bool): Wrap input coordinates back into the box

    .. deprecated:: 2.0
       GSD is the new default file format for HOOMD-blue. It can store everything that an XML file can in
       an efficient binary format that is easy to access. See :py:class:`hoomd.init.read_gsd`.

    Examples::

        deprecated.init.read_xml(filename="data.xml")
        deprecated.init.read_xml(filename="init.xml", restart="restart.xml")
        deprecated.init.read_xml(filename="directory/data.xml")
        deprecated.init.read_xml(filename="restart.xml", time_step=0)
        system = deprecated.init.read_xml(filename="data.xml")


    All particles, bonds, etc...  are read from the given XML file,
    setting the initial condition of the simulation.
    After this command completes, the system is initialized allowing
    other commands in hoomd to be run.

    For restartable jobs, specify the initial condition in *filename* and the restart file in *restart*.
    init.read_xml will read the restart file if it exists, otherwise it will read *filename*.

    All values are read in native units, see :ref:`page-units` for more information.

    If *time_step* is specified, its value will be used as the initial time
    step of the simulation instead of the one read from the XML file.

    If *wrap_coordinates* is set to True, input coordinates will be wrapped
    into the box specified inside the XML file. If it is set to False, out-of-box
    coordinates will result in an error.

    """
    hoomd.context._verify_init();
    hoomd.util.print_status_line();

    # check if initialization has already occurred
    if hoomd.init.is_initialized():
        hoomd.context.msg.error("Cannot initialize more than once\n");
        raise RuntimeError("Error reading XML file");

    filename_to_read = filename;
    if restart is not None:
        if os.path.isfile(restart):
            filename_to_read = restart;

    # read in the data
    initializer = _deprecated.HOOMDInitializer(hoomd.context.exec_conf,filename_to_read,wrap_coordinates);
    snapshot = initializer.getSnapshot()

    my_domain_decomposition = hoomd.init._create_domain_decomposition(snapshot._global_box);
    if my_domain_decomposition is not None:
        hoomd.context.current.system_definition = _hoomd.SystemDefinition(snapshot, hoomd.context.exec_conf, my_domain_decomposition);
    else:
        hoomd.context.current.system_definition = _hoomd.SystemDefinition(snapshot, hoomd.context.exec_conf);

    # initialize the system
    if time_step is None:
        hoomd.context.current.system = _hoomd.System(hoomd.context.current.system_definition, initializer.getTimeStep());
    else:
        hoomd.context.current.system = _hoomd.System(hoomd.context.current.system_definition, time_step);

    hoomd.init._perform_common_init_tasks();
    return hoomd.data.system_data(hoomd.context.current.system_definition);

def create_random(N, phi_p=None, name="A", min_dist=0.7, box=None, seed=1, dimensions=3):
    R""" Generates N randomly positioned particles of the same type.

    Args:
        N (int): Number of particles to create.
        phi_p (float): Packing fraction of particles in the simulation box (unitless).
        name (str): Name of the particle type to create.
        min_dist (float): Minimum distance particles will be separated by (in distance units).
        box (:py:class:`hoomd.data.boxdim`): Simulation box dimensions.
        seed (int): Random seed.
        dimensions (int): The number of dimensions in the simulation.

    .. deprecated:: 2.0 Random initialization is best left to specific methods tailored by the user for their work.

    Either *phi_p* or *box* must be specified. If *phi_p* is provided, it overrides the value of *box*.

    Examples::

        init.create_random(N=2400, phi_p=0.20)
        init.create_random(N=2400, phi_p=0.40, min_dist=0.5)
        system = init.create_random(N=2400, box=data.boxdim(L=20))

    When *phi_p* is set, the
    dimensions of the created box are such that the packing fraction
    of particles in the box is *phi_p*. The number density \e n
    is related to the packing fraction by :math:`n = 2d/\pi \cdot \phi_P`,
    where *d* is the dimension, and assumes the particles have a radius of 0.5.
    All particles are created with the same type, given by *name*.

    The result of :py:func:`hoomd.deprecated.init.create_random` can be saved in a variable and later used to read
    and/or change particle properties later in the script. See :py:mod:`hoomd.data` for more information.

    """
    hoomd.context._verify_init();
    hoomd.util.print_status_line();

    # check if initialization has already occurred
    if hoomd.init.is_initialized():
        hoomd.context.msg.error("Cannot initialize more than once\n");
        raise RuntimeError("Error initializing");

    # check that dimensions are appropriate
    if dimensions not in (2,3):
        raise ValueError('dimensions must be 2 or 3')

    # abuse the polymer generator to generate single particles

    if phi_p is not None:
        # calculate the box size
        L = math.pow(math.pi/(2.0*dimensions)*N / phi_p, 1.0/dimensions);
        box = hoomd.data.boxdim(L=L, dimensions=dimensions);

    if box is None:
        raise RuntimeError('box or phi_p must be specified');

    if not isinstance(box, hoomd.data.boxdim):
        hoomd.context.msg.error('box must be a data.boxdim object');
        raise TypeError('box must be a data.boxdim object');

    # create the generator
    generator = _deprecated.RandomGenerator(hoomd.context.exec_conf, box._getBoxDim(), seed, box.dimensions);

    # build type list
    type_vector = _hoomd.std_vector_string();
    type_vector.append(name);

    # empty bond lists for single particles
    bond_ab = _hoomd.std_vector_uint();
    bond_type = _hoomd.std_vector_string();

    # create the generator
    generator.addGenerator(int(N), _deprecated.PolymerParticleGenerator(hoomd.context.exec_conf, 1.0, type_vector, bond_ab, bond_ab, bond_type, 100, box.dimensions));

    # set the separation radius
    generator.setSeparationRadius(name, min_dist/2.0);

    # generate the particles
    generator.generate();

    # initialize snapshot
    snapshot = generator.getSnapshot()

    my_domain_decomposition = hoomd.init._create_domain_decomposition(snapshot._global_box);
    if my_domain_decomposition is not None:
        hoomd.context.current.system_definition = _hoomd.SystemDefinition(snapshot, hoomd.context.exec_conf, my_domain_decomposition);
    else:
        hoomd.context.current.system_definition = _hoomd.SystemDefinition(snapshot, hoomd.context.exec_conf);

    # initialize the system
    hoomd.context.current.system = _hoomd.System(hoomd.context.current.system_definition, 0);

    hoomd.init._perform_common_init_tasks();
    return hoomd.data.system_data(hoomd.context.current.system_definition);

def create_random_polymers(box, polymers, separation, seed=1):
    R""" Generates any number of randomly positioned polymers of configurable types.

    Args:
        box (:py:class:`hoomd.data.boxdim`): Simulation box dimensions
        polymers (list): Specification for the different polymers to create (see below)
        separation (dict): Separation radii for different particle types (see below)
        seed (int): Random seed to use

    .. deprecated:: 2.0 Random initialization is best left to specific methods tailored by the user for their work.

    Any number of polymers can be generated, of the same or different types, as
    specified in the argument *polymers*. Parameters for each polymer include
    bond length, particle type list, bond list, and count.

    The syntax is best shown by example. The below line specifies that 600 block copolymers
    A6B7A6 with a bond length of 1.2 be generated::

        polymer1 = dict(bond_len=1.2, type=['A']*6 + ['B']*7 + ['A']*6,
                        bond="linear", count=600)

    Here is an example for a second polymer, specifying just 100 polymers made of 5 B beads
    bonded in a branched pattern::

        polymer2 = dict(bond_len=1.2, type=['B']*5,
                        bond=[(0, 1), (1,2), (1,3), (3,4)] , count=100)

    The *polymers* argument can be given a list of any number of polymer types specified
    as above. *count* randomly generated polymers of each type in the list will be
    generated in the system.

    In detail:

    - bond_len defines the bond length of the generated polymers. This should
      not necessarily be set to the equilibrium bond length! The generator is dumb and doesn't know
      that bonded particles can be placed closer together than the separation (see below). Thus
      bond_len must be at a minimum set at twice the value of the largest separation radius. An
      error will be generated if this is not the case.
    - type is a python list of strings. Each string names a particle type in the order that
      they will be created in generating the polymer.
    - bond can be specified as "linear" in which case the generator connects all particles together
      with bonds to form a linear chain. bond can also be given a list if python tuples (see example
      above).
      - Each tuple in the form of \c (a,b) specifies that particle \c a of the polymer be bonded to
      particle \c b. These bonds are given the default type name of 'polymer' to be used when specifying parameters to
      bond forces such as bond.harmonic.
      - A tuple with three elements (a,b,type) can be used as above, but with a custom name for the bond. For example,
      a simple branched polymer with different bond types on each branch could be defined like so::

            bond=[(0,1), (1,2), (2,3,'branchA'), (3,4,'branchA), (2,5,'branchB'), (5,6,'branchB')]


    separation must contain one entry for each particle type specified in polymers
    ('A' and 'B' in the examples above). The value given is the separation radius of each
    particle of that type. The generated polymer system will have no two overlapping
    particles.

    Examples::

        init.create_random_polymers(box=data.boxdim(L=35),
                                    polymers=[polymer1, polymer2],
                                    separation=dict(A=0.35, B=0.35));

        init.create_random_polymers(box=data.boxdim(L=31),
                                    polymers=[polymer1],
                                    separation=dict(A=0.35, B=0.35), seed=52);

        # create polymers in an orthorhombic box
        init.create_random_polymers(box=data.boxdim(Lx=18,Ly=10,Lz=25),
                                    polymers=[polymer2],
                                    separation=dict(A=0.35, B=0.35), seed=12345);

        # create a triclinic box with tilt factors xy=0.1 xz=0.2 yz=0.3
        init.create_random_polymers(box=data.boxdim(L=18, xy=0.1, xz=0.2, yz=0.3),
                                    polymers=[polymer2],
                                    separation=dict(A=0.35, B=0.35));

    With all other parameters the same, create_random_polymers will always create the
    same system if seed is the same. Set a different seed (any integer) to create
    a different random system with the same parameters. Note that different versions
    of HOOMD \e may generate different systems even with the same seed due to programming
    changes.

    Note:
        For relatively dense systems (packing fraction 0.4 and higher) the simple random
        generation algorithm may fail to find room for all the particles and print an error message.
        There are two methods to solve this. First, you can lower the separation radii allowing particles
        to be placed closer together. Then setup integrate.nve with the limit option set to a
        relatively small value. A few thousand time steps should relax the system so that the simulation can be
        continued without the limit or with a different integrator. For extremely troublesome systems,
        generate it at a very low density and shrink the box with the command update.box_resize
        to the desired final size.

    Note:
        The polymer generator always generates polymers as if there were linear chains. If you
        provide a non-linear bond topology, the bonds in the initial configuration will be stretched
        significantly. This normally doesn't pose a problem for harmonic bonds (bond.harmonic) as
        the system will simply relax over a few time steps, but can cause the system to blow up with FENE
        bonds (bond.fene).

    """
    hoomd.context._verify_init();
    hoomd.util.print_status_line();

    # check if initialization has already occurred
    if hoomd.init.is_initialized():
        hoomd.context.msg.error("Cannot initialize more than once\n");
        raise RuntimeError("Error creating random polymers");

    if len(polymers) == 0:
        hoomd.context.msg.error("Polymers list cannot be empty.\n");
        raise RuntimeError("Error creating random polymers");

    if len(separation) == 0:
        hoomd.context.msg.error("Separation dict cannot be empty.\n");
        raise RuntimeError("Error creating random polymers");

    if not isinstance(box, hoomd.data.boxdim):
        hoomd.context.msg.error('Box must be a data.boxdim object\n');
        raise TypeError('box must be a data.boxdim object');

    # create the generator
    generator = _deprecated.RandomGenerator(hoomd.context.exec_conf,box._getBoxDim(), seed, box.dimensions);

    # make a list of types used for an eventual check vs the types in separation for completeness
    types_used = [];

    # track the minimum bond length
    min_bond_len = None;

    # build the polymer generators
    for poly in polymers:
        type_list = [];
        # check that all fields are specified
        if not 'bond_len' in poly:
            hoomd.context.msg.error('Polymer specification missing bond_len\n');
            raise RuntimeError("Error creating random polymers");

        if min_bond_len is None:
            min_bond_len = poly['bond_len'];
        else:
            min_bond_len = min(min_bond_len, poly['bond_len']);

        if not 'type' in poly:
            hoomd.context.msg.error('Polymer specification missing type\n');
            raise RuntimeError("Error creating random polymers");
        if not 'count' in poly:
            hoomd.context.msg.error('Polymer specification missing count\n');
            raise RuntimeError("Error creating random polymers");
        if not 'bond' in poly:
            hoomd.context.msg.error('Polymer specification missing bond\n');
            raise RuntimeError("Error creating random polymers");

        # build type list
        type_vector = _hoomd.std_vector_string();
        for t in poly['type']:
            type_vector.append(t);
            if not t in types_used:
                types_used.append(t);

        # build bond list
        bond_a = _hoomd.std_vector_uint();
        bond_b = _hoomd.std_vector_uint();
        bond_name = _hoomd.std_vector_string();

        # if the bond setting is 'linear' create a default set of bonds
        if poly['bond'] == 'linear':
            for i in range(0,len(poly['type'])-1):
                bond_a.append(i);
                bond_b.append(i+1);
                bond_name.append('polymer')
        #if it is a list, parse the user custom bonds
        elif type(poly['bond']) == type([]):
            for t in poly['bond']:
                # a 2-tuple gets the default 'polymer' name for the bond
                if len(t) == 2:
                    a,b = t;
                    name = 'polymer';
                # and a 3-tuple specifies the name directly
                elif len(t) == 3:
                    a,b,name = t;
                else:
                    hoomd.context.msg.error('Custom bond ' + str(t) + ' must have either two or three elements\n');
                    raise RuntimeError("Error creating random polymers");

                bond_a.append(a);
                bond_b.append(b);
                bond_name.append(name);
        else:
            hoomd.context.msg.error('Unexpected argument value for polymer bond\n');
            raise RuntimeError("Error creating random polymers");

        # create the generator
        generator.addGenerator(int(poly['count']), _deprecated.PolymerParticleGenerator(hoomd.context.exec_conf, poly['bond_len'], type_vector, bond_a, bond_b, bond_name, 100, box.dimensions));


    # check that all used types are in the separation list
    for t in types_used:
        if not t in separation:
            hoomd.context.msg.error("No separation radius specified for type " + str(t) + "\n");
            raise RuntimeError("Error creating random polymers");

    # set the separation radii, checking that it is within the minimum bond length
    for t,r in separation.items():
        generator.setSeparationRadius(t, r);
        if 2*r >= min_bond_len:
            hoomd.context.msg.error("Separation radius " + str(r) + " is too big for the minimum bond length of " + str(min_bond_len) + " specified\n");
            raise RuntimeError("Error creating random polymers");

    # generate the particles
    generator.generate();

    # copy over data to snapshot
    snapshot = generator.getSnapshot()

    my_domain_decomposition = hoomd.init._create_domain_decomposition(snapshot._global_box);
    if my_domain_decomposition is not None:
        hoomd.context.current.system_definition = _hoomd.SystemDefinition(snapshot, hoomd.context.exec_conf, my_domain_decomposition);
    else:
        hoomd.context.current.system_definition = _hoomd.SystemDefinition(snapshot, hoomd.context.exec_conf);

    # initialize the system
    hoomd.context.current.system = _hoomd.System(hoomd.context.current.system_definition, 0);

    hoomd.init._perform_common_init_tasks();
    return hoomd.data.system_data(hoomd.context.current.system_definition);
