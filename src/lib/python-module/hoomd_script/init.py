# -*- coding: iso-8859-1 -*-
# Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
# Source Software License
# Copyright (c) 2008 Ames Laboratory Iowa State University
# All rights reserved.

# Redistribution and use of HOOMD, in source and binary forms, with or
# without modification, are permitted, provided that the following
# conditions are met:

# * Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names HOOMD's
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.

# Disclaimer

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
# CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 

# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.

# $Id$
# $URL$
# Maintainer: joaander / All Developers are free to add commands for new features

from optparse import OptionParser;

import hoomd;
import globals;
import update;

import math;
import sys;
import util;
import gc;
import os;

## \internal
# \brief Parsed command line options
_options = {};

## \package hoomd_script.init
# \brief Data initialization commands
#
# Commands in the init package initialize the particle system. Initialization via
# any of the commands here must be done before any other command in hoomd_script can
# be run.
#
# \sa \ref page_quick_start

## Resets all hoomd_script variables
#
# After calling init.reset() all global variables used in hoomd_script are cleared and all allocated
# memory is freed so the simulation can begin anew without needing to launch hoomd again.
#
# \note There is a very important memory management issue that must be kept in mind when using
# reset(). If you have saved a variable such as an integrator or a force for changing parameters, 
# that saved object \b must be deleted before the reset() command is called. If all objects are 
# not deleted, then a memory leak will result causing repeated runs of even a small simulation 
# to eventually run the system out of memory. reset() will throw an error if it detects that this 
# is the case.
#
# \b Example:
# \code
# init.create_random(N=1000, phi_p = 0.2)
# lj = pair.lj(r_cut=3.0)
# .... setup and run simulation
# del lj
# init.reset()
# init.create_random(N=2000, phi_p = 0.2)
# .... setup and run simulation
# \endcode
def reset():
    if globals.system_definition == None:
        print "\n***Warning! Trying to reset an uninitialized system";
        return;

    # perform some reference counting magic to verify that the user has cleared all saved variables
    sysdef = globals.system_definition;
    globals._clear();
    
    gc.collect();
    count = sys.getrefcount(sysdef)

    # note: the check should be against 2, getrefcount counts the temporary reference 
    # passed to it in the argument
    expected_count = 2
    if count != expected_count:
        print "\n***Warning! Not all saved variables were cleared before calling reset()";
        print count-expected_count, "references to the particle data still exist somewhere\n"
        raise RuntimeError('Error resetting');

    del sysdef
    gc.collect();

## Reads initial system state from an XML file
#
# \param filename File to read
# \param time_step (if specified) Time step number to use instead of the one stored in the XML file
#
# \b Examples:
# \code
# init.read_xml(filename="data.xml")
# init.read_xml(filename="directory/data.xml")
# init.read_xml(filename="restart.xml", time_step=0)
# \endcode
#
# All particles, bonds, etc...  are read from the XML file given, 
# setting the initial condition of the simulation.
# After this command completes, the system is initialized allowing 
# other commands in hoomd_script to be run. For more details
# on the file format read by this command, see \ref page_xml_file_format.
#
# If \a time_step is specified, it's value will be used as the initial time 
# step of the simulation instead of the one read from the XML file.
#
def read_xml(filename, time_step = None):
    util.print_status_line();
    
    # parse command line
    _parse_command_line();

    # check if initialization has already occurred
    if (globals.system_definition != None):
        print >> sys.stderr, "\n***Error! Cannot initialize more than once\n";
        raise RuntimeError('Error initializing');

    # read in the data
    initializer = hoomd.HOOMDInitializer(filename);
    globals.system_definition = hoomd.SystemDefinition(initializer, _create_exec_conf());
    
    # initialize the system
    if time_step == None:
        globals.system = hoomd.System(globals.system_definition, initializer.getTimeStep());
    else:
        initializer.setTimeStep(time_step)
        globals.system = hoomd.System(globals.system_definition, initializer.getTimeStep());
    
    _perform_common_init_tasks();
    return globals.system_definition;


## Generates N randomly positioned particles of the same type
#
# \param N Number of particles to create
# \param phi_p Packing fraction of particles in the simulation box
# \param name Name of the particle type to create
# \param min_dist Minimum distance particles will be separated by
#
# \b Examples:
# \code
# init.create_random(N=2400, phi_p=0.20)
# init.create_random(N=2400, phi_p=0.40, min_dist=0.5)
# \endcode
#
# \a N particles are randomly placed in the simulation box. The 
# dimensions of the created box are such that the packing fraction
# of particles in the box is \a phi_p. The number density \e n
# is related to the packing fraction by \f$n = 6/\pi \cdot \phi_P\f$
# assuming the particles have a radius of 0.5.
# All particles are created with the same type, given by \a name.
#
def create_random(N, phi_p, name="A", min_dist=0.7):
    util.print_status_line();
    
    # parse command line
    _parse_command_line();
    my_exec_conf = _create_exec_conf();
    
    # check if initialization has already occurred
    if (globals.system_definition != None):
        print >> sys.stderr, "\n***Error! Cannot initialize more than once\n";
        raise RuntimeError('Error initializing');

    # abuse the polymer generator to generate single particles
    
    # calculat the box size
    L = math.pow(math.pi/6.0*N / phi_p, 1.0/3.0);
    box = hoomd.BoxDim(L);
    
    # create the generator
    generator = hoomd.RandomGenerator(box, 12345);
    
    # build type list
    type_vector = hoomd.std_vector_string();
    type_vector.append(name);
    
    # empty bond lists for single particles
    bond_ab = hoomd.std_vector_uint();
    bond_type = hoomd.std_vector_string();
        
    # create the generator
    generator.addGenerator(int(N), hoomd.PolymerParticleGenerator(1.0, type_vector, bond_ab, bond_ab, bond_type, 100));
    
    # set the separation radius
    generator.setSeparationRadius(name, min_dist/2.0);
        
    # generate the particles
    generator.generate();
    
    globals.system_definition = hoomd.SystemDefinition(generator, my_exec_conf);
    
    # initialize the system
    globals.system = hoomd.System(globals.system_definition, 0);
    
    _perform_common_init_tasks();
    return globals.system_definition;


## Generates any number of randomly positioned polymers of configurable types
#
# \param box BoxDim specifying the simulation box to generate the polymers in
# \param polymers Specification for the different polymers to create (see below)
# \param separation Separation radii for different particle types (see below)
# \param seed Random seed to use
#
# Any number of polymers can be generated, of the same or different types, as 
# specified in the argument \a polymers. Parameters for each polymer, include
# bond length, particle type list, bond list, and count.
#
# The syntax is best shown by example. The below line specifies that 600 block copolymers
# A6B7A6 with a %bond length of 1.2 be generated.
# \code
# polymer1 = dict(bond_len=1.2, type=['A']*6 + ['B']*7 + ['A']*6, 
#                 bond="linear", count=600)
# \endcode
# Here is an example for a second polymer, specifying just 100 polymers made of 4 B beads
# bonded in a branched pattern
# \code
# polymer2 = dict(bond_len=1.2, type=['B']*4, 
#                 bond=[(0, 1), (1,2), (1,3), (3,4)] , count=100)
# \endcode
# The \a polymers argument can be given a list of any number of polymer types specified
# as above. \a count randomly generated polymers of each type in the list will be
# generated in the system.
#
# In detail: 
# - \a bond_len defines the %bond length of the generated polymers. This should 
#   not necessarily be set to the equilibrium %bond length! The generator is dumb and doesn't know
#   that bonded particles can be placed closer together than the separation (see below). Thus
#   \a bond_len must be at a minimum set at twice the value of the largest separation radius. An 
#   error will be generated if this is not the case.
# - \a type is a python list of strings. Each string names a particle type in the order that
#   they will be created in generating the polymer.
# - \a %bond can be specified as "linear" in which case the generator connects all particles together
#   with bonds to form a linear chain. \a %bond can also be given a list if python tuples (see example
#   above). 
#   - Each tuple in the form of \c (a,b) specifies that particle \c a of the polymer be bonded to
#   particle \c b. These bonds are given the default type name of 'polymer' to be used when specifying parameters to 
#   bond forces such as bond.harmonic.
#   - A tuple with three elements (a,b,type) can be used as above, but with a custom name for the bond. For example,
#   a simple branched polymer with different bond types on each branch could be defined like so:
#\code
#bond=[(0,1), (1,2), (2,3,'branchA'), (3,4,'branchA), (2,5,'branchB'), (5,6,'branchB')]
#\endcode
# 
# \a separation \b must contain one entry for each particle type specified in \a polymers
# ('A' and 'B' in the examples above). The value given is the separation radius of each
# particle of that type. The generated polymer system will have no two overlapping 
# particles.
#
# \b Examples:
# \code
# init.create_random_polymers(box=hoomd.BoxDim(35), 
#                             polymers=[polymer1, polymer2], 
#                             separation=dict(A=0.35, B=0.35));
# 
# init.create_random_polymers(box=hoomd.BoxDim(31), 
#                             polymers=[polymer1], 
#                             separation=dict(A=0.35, B=0.35), seed=52);
# 
# init.create_random_polymers(box=hoomd.BoxDim(18,10,25), 
#                             polymers=[polymer2], 
#                             separation=dict(A=0.35, B=0.35), seed=12345);
# \endcode
#
# With all other parameters the same, create_random_polymers will always create the
# same system if \a seed is the same. Set a different \a seed (any integer) to create
# a different random system with the same parameters. Note that different versions
# of HOOMD \e may generate different systems even with the same seed due to programming
# changes.
#
# \note 1. For relatively dense systems (packing fraction 0.4 and higher) the simple random
# generation algorithm may fail to find room for all the particles and print an error message. 
# There are two methods to solve this. First, you can lower the separation radii allowing particles 
# to be placed closer together. Then setup integrate.nve with the \a limit option set to a 
# relatively small value. A few thousand time steps should relax the system so that the simulation can be
# continued without the limit or with a different integrator. For extremely troublesome systems,
# generate it at a very low density and shrink the box with the command update.box_resize
# to the desired final size.
#
# \note 2. The polymer generator always generates polymers as if there were linear chains. If you 
# provide a non-linear %bond topology, the bonds in the initial configuration will be stretched 
# significantly. This normally doesn't pose a problem for harmonic bonds (bond.harmonic) as
# the system will simply relax over a few time steps, but can cause the system to blow up with FENE 
# bonds (bond.fene). 
#
# \note 3. While the custom %bond list allows you to create ring shaped polymers, testing shows that
# such conformations have trouble relaxing and get stuck in tangled configurations. If you need 
# to generate a configuration of rings, you may need to write your own specialized initial configuration
# generator that writes HOOMD XML input files (see \ref page_xml_file_format). HOOMD's built-in polymer generator
# attempts to be as general as possible, but unfortunately cannot work in every possible case.
#
def create_random_polymers(box, polymers, separation, seed=1):
    util.print_status_line();
    
    # parse command line
    _parse_command_line();
    my_exec_conf = _create_exec_conf();
        
    # check if initialization has already occured
    if (globals.system_definition != None):
        print >> sys.stderr, "\n***Error! Cannot initialize more than once\n";
        raise RuntimeError("Error creating random polymers");
    
    if type(polymers) != type([]) or len(polymers) == 0:
        print >> sys.stderr, "\n***Error! polymers specified incorrectly. See the hoomd_script documentation\n";
        raise RuntimeError("Error creating random polymers");
    
    if type(separation) != type(dict()) or len(separation) == 0:
        print >> sys.stderr, "\n***Error! polymers specified incorrectly. See the hoomd_script documentation\n";
        raise RuntimeError("Error creating random polymers");
    
    # create the generator
    generator = hoomd.RandomGenerator(box, seed);
    
    # make a list of types used for an eventual check vs the types in separation for completeness
    types_used = [];
    
    # track the minimum bond length
    min_bond_len = None;
    
    # build the polymer generators
    for poly in polymers:
        type_list = [];
        # check that all fields are specified
        if not 'bond_len' in poly:
            print >> sys.stderr, '\n***Error! Polymer specification missing bond_len\n';
            raise RuntimeError("Error creating random polymers");
        
        if min_bond_len == None:
            min_bond_len = poly['bond_len'];
        else:
            min_bond_len = min(min_bond_len, poly['bond_len']);
        
        if not 'type' in poly:
            print >> sys.stderr, '\n***Error! Polymer specification missing type\n';
            raise RuntimeError("Error creating random polymers");
        if not 'count' in poly:
            print >> sys.stderr, '\n***Error! Polymer specification missing count\n';
            raise RuntimeError("Error creating random polymers");
        if not 'bond' in poly:
            print >> sys.stderr, '\n***Error! Polymer specification missing bond\n';
            raise RuntimeError("Error creating random polymers");
                
        # build type list
        type_vector = hoomd.std_vector_string();
        for t in poly['type']:
            type_vector.append(t);
            if not t in types_used:
                types_used.append(t);
        
        # build bond list
        bond_a = hoomd.std_vector_uint();
        bond_b = hoomd.std_vector_uint();
        bond_name = hoomd.std_vector_string();
        
        # if the bond setting is 'linear' create a default set of bonds
        if poly['bond'] == 'linear':
            for i in xrange(0,len(poly['type'])-1):
                bond_a.push_back(i);
                bond_b.push_back(i+1);
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
                    print >> sys.stderr, '\n***Error! Custom bond', t, 'must have either two or three elements\n';
                    raise RuntimeError("Error creating random polymers");
                                    
                bond_a.push_back(a);
                bond_b.push_back(b);
                bond_name.append(name);
        else:
            print >> sys.stderr, '\n***Error! Unexpected argument value for polymer bond\n';
            raise RuntimeError("Error creating random polymers");
        
        # create the generator
        generator.addGenerator(int(poly['count']), hoomd.PolymerParticleGenerator(poly['bond_len'], type_vector, bond_a, bond_b, bond_name, 100));
        
        
    # check that all used types are in the separation list
    for t in types_used:
        if not t in separation:
            print >> sys.stderr, "\n***Error! No separation radius specified for type ", t, "\n";
            raise RuntimeError("Error creating random polymers");
            
    # set the separation radii, checking that it is within the minimum bond length
    for t,r in separation.items():
        generator.setSeparationRadius(t, r);
        if 2*r >= min_bond_len:
            print >> sys.stderr, "\n***Error! Separation radius", r, "is too big for the minimum bond length of", min_bond_len, "specified\n";
            raise RuntimeError("Error creating random polymers");
        
    # generate the particles
    generator.generate();
    
    globals.system_definition = hoomd.SystemDefinition(generator, my_exec_conf);
    
    # initialize the system
    globals.system = hoomd.System(globals.system_definition, 0);
    
    _perform_common_init_tasks();
    return globals.system_definition;

## Performs common initialization tasks
#
# \internal
# Initialization tasks that are performed for every simulation are to
# be done here. For example, setting up the SFCPackUpdater, initializing
# the log writer, etc...
#
# Currently only creates the sorter
def _perform_common_init_tasks():
    # create the sorter, using the evil import __main__ trick to provide the user with a default variable
    import __main__;
    __main__.sorter = update.sort();

## Parses command line options
#
# \internal
# Parses all hoomd_script command line options into the module variable _options
def _parse_command_line():
    global _options;
    
    parser = OptionParser();
    parser.add_option("--mode", dest="mode", help="Execution mode (cpu or gpu)");
    parser.add_option("--gpu", dest="gpu", help="GPU to execute on");
    parser.add_option("--ngpu", dest="ngpu", help="Number of GPUs to execute on (requires that CUDA 2.2 compute-exclusive mode be enabled on all GPUs)");
    parser.add_option("--gpu_error_checking", dest="gpu_error_checking", action="store_true", default=False, help="Enable error checking on the GPU");
    parser.add_option("--minimize-cpu-usage", dest="min_cpu", action="store_true", default=False, help="Enable to keep the CPU usage of HOOMD to a bare minimum (will degrade overall performance somewhat)");
    parser.add_option("--ignore-display-gpu", dest="ignore_display", action="store_true", default=False, help="Attempt to avoid running on the display GPU");
    
    (_options, args) = parser.parse_args();
    
    # chedk for valid mode setting
    if _options.mode:
        if not (_options.mode == "cpu" or _options.mode == "gpu"):
            parser.error("--mode must be either cpu or gpu");
    
    # check for sane options
    if _options.mode == "cpu" and (_options.gpu or _options.ngpu):
        parser.error("It doesn't make sense to specify --mode=cpu and a value for --gpu")

    # set the mode to gpu if the gpu # was set
    if (_options.gpu and not _options.mode) or (_options.ngpu and not _options.mode):
        _options.mode = "gpu"
        
    if _options.gpu and _options.ngpu:
        parser.error("--gpu and --ngpu are mutually exclusive options")
        
    # if gpu_error_checking is set, enable it on the GPU
    if _options.gpu_error_checking:
        hoomd.set_gpu_error_checking(True);
    
## Initializes the execution configuration
#
# \internal
# Given an initializer, create a particle data with a properly configured ExecutionConfiguration
def _create_exec_conf():
    global _options;
    
    # if no command line options were specified, create a default ExecutionConfiguration
    if not _options.mode:
        exec_conf = hoomd.ExecutionConfiguration(_options.min_cpu, _options.ignore_display);
    else:
        # create a list of GPUs to execute on
        gpu_ids = hoomd.std_vector_int();
        if _options.gpu:
            # parse the list of gpus
            string_gpu_list = _options.gpu.split(",")
            for gpu in string_gpu_list:
                gpu_ids.append(int(gpu));
        elif _options.ngpu:
            for i in xrange(0, int(_options.ngpu)):
                gpu_ids.append(-1);
        else:
            # otherwise, assume the default GPU
            gpu_ids.append(-1);
        
        # create the specified configuration
        if _options.mode == "cpu":
            exec_conf = hoomd.ExecutionConfiguration(hoomd.ExecutionConfiguration.executionMode.CPU, _options.min_cpu, _options.ignore_display);
        elif _options.mode == "gpu":
            exec_conf = hoomd.ExecutionConfiguration(gpu_ids, _options.min_cpu, _options.ignore_display);
        else:
            raise RuntimeError("Error initializing");
        
    return exec_conf;

