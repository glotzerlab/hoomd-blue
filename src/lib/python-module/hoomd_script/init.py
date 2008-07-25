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

from optparse import OptionParser;

import hoomd;
import globals;
import update;

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

## Reads initial system state from an XML file
#
# \param filename File to read
#
# \b Examples:
# \code
# init.read_xml(file_name="data.xml")
# init.read_xml(file_name="directory/data.xml")
# \endcode
#
# All particles, bonds, etc...  are read from the XML file given, 
# setting the initial condition of the simulation.
# After this command completes, the system is initialized allowing 
# other commands in hoomd_script to be run. For more details
# on the file format read by this command, see \ref page_xml_file_format.
def read_xml(filename):
	print "init.read_xml(filename=", file_name, ")";
	
	# parse command line
	_parse_command_line();

	# check if initialization has already occured
	if (globals.particle_data != None):
		print "Error: Cannot initialize more than once";
		raise RuntimeError('Error initializing');

	# read in the data
	initializer = hoomd.HOOMDInitializer(filename);
	globals.particle_data = hoomd.ParticleData(initializer, _create_exec_conf());
	
	# TEMPORARY HACK for bond initialization
	globals.initializer = initializer;

	# initialize the system
	globals.system = hoomd.System(globals.particle_data, initializer.getTimeStep());
	
	_perform_common_init_tasks();
	return globals.particle_data;


## Generates N randomly positioned particles of the same type
#
# \param N Number of particles to create
# \param phi_p Packing fraction of particles in the simulation box
# \param name Name of the particle type to create
# \param min_dist Minimum distance particles will be separated by
# \param wall_offset (optional) If specified, walls are created a distance of 
#	\a wall_offset in from the edge of the simulation box
#
# \b Examples:
# \code
# init.create_random(N=2400, phi_p=0.20)
# init.create_random(N=2400, phi_p=0.40, min_dist=0.5)
# init.create_random(wall_offset=3.1, phi_p=0.10, N=6000)
# \endcode
#
# \a N particles are randomly placed in the simulation box. The 
# dimensions of the created box are such that the packing fraction
# of particles in the box is \a phi_p. The number density \e n
# is related to the packing fraction by \f$n = 6/\pi \cdot \phi_P\f$
# assuming the particles have a radius of 0.5.
# All particles are created with the same type, given by \a name.
#
def create_random(N, phi_p, name="A", min_dist=1.0, wall_offset=None):
	print "init.create_random(N =", N, ", phi_p =", phi_p, ", name = ", name, ", min_dist =", min_dist, ", wall_offset =", wall_offset, ")";
	
	# parse command line
	_parse_command_line();
	
	# check if initialization has already occured
	if (globals.particle_data != None):
		print "Error: Cannot initialize more than once";
		raise RuntimeError('Error initializing');

	# read in the data
	if wall_offset == None:
		initializer = hoomd.RandomInitializer(N, phi_p, min_dist, name);
	else:
		initializer = hoomd.RandomInitializerWithWalls(N, phi_p, min_dist, wall_offset, name);
		
	globals.particle_data = hoomd.ParticleData(initializer, _create_exec_conf());

	# initialize the system
	globals.system = hoomd.System(globals.particle_data, 0);
	
	_perform_common_init_tasks();
	return globals.particle_data;

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
# A6B7A6 with a bond length of 1.2 be generated.
# \code
# polymer1 = dict(bond_len=1.2, type=['A']*6 + ['B']*7 + ['A']*6, bond="TODO", count=600)
# \endcode
# Here is an example for a second polymer, specifying just 100 polymers made of 4 B beads
# \code
# polymer2 = dict(bond_len=1.2, type=['B']*4, bond="TODO", count=100)
# \endcode
# The \a polymers argument can be given a list of any number of polymer types specified
# as above. \a count randomly generated polymers of each type in the list will be
# generated in the system.
# 
# \a separation \b must contain one entry for each particle type specified in \a polymers
# ('A' and 'B' in the examples above). The value given is the separation radius of each
# particle of that type. The generated polymer system will have no two overlapping 
# particles.
#
# \b Examples:
# \code
# init.create_random_polymers(box=hoomd.BoxDim(35), polymers=[polymer1, polymer2], separation=dict(A=0.35, B=0.35));
# init.create_random_polymers(box=hoomd.BoxDim(31), polymers=[polymer1], separation=dict(A=0.35, B=0.35), seed=52);
# init.create_random_polymers(box=hoomd.BoxDim(18,10,25), polymers=[polymer2], separation=dict(A=0.35, B=0.35), seed=12345);
# \endcode
#
# With all other parameters the same, create_random_polymers will always create the
# same system if \a seed is the same. Set a different \a seed (any integer) to create
# a different random system with the same parameters. Note that different versions
# of HOOMD \e may generate different systems even with the same seed due to programming
# changes.
#
# \note For relatively dense systems (packing fraction 0.4 and higher) the simple random
# generation algorithm may fail to find room for all the particles and print an error message. 
# There are two methods to solve this. First, you can lower the separation radii allowing particles 
# to be placed closer together. Then setup integrate.nve with the \a limit option set to a 
# relatively small value. A few thousand timesteps should relax the system so that the simulation can be
# continued without the limit or with a different integrator. For extremely troublesome systems,
# generate it at a very low density and shrink the box with the command ___ (which isn't written yet)
# to the desired final size.
#
# \note Currently, create_random_polymers() always creates linear chains.
def create_random_polymers(box, polymers, separation, seed=1):
	print "init.create_random_polymers(box =", box, ", polymers =", polymers, ", separation = ", separation, ", seed =", seed, ")";
	
	# parse command line
	_parse_command_line();
	my_exec_conf = _create_exec_conf();
		
	# check if initialization has already occured
	if (globals.particle_data != None):
		print "Error: Cannot initialize more than once";
		raise RuntimeError("Error creating random polymers");
	
	if type(polymers) != type([]) or len(polymers) == 0:
		print "Argument error: polymers specified incorrectly. See the hoomd_script documentation"
		raise RuntimeError("Error creating random polymers");
	 
	if type(separation) != type(dict()) or len(separation) == 0:
		print "Argument error: polymers specified incorrectly. See the hoomd_script documentation"
		raise RuntimeError("Error creating random polymers");
	
	# create the generator
	generator = hoomd.RandomGenerator(box, seed);
	
	# make a list of types used for an eventual check vs the types in separation for completeness
	types_used = [];
	
	# build the polymer generators
	for poly in polymers:
		type_list = [];
		# check that all fields are specified
		if not 'bond_len' in poly:
			print 'Polymer specification missing bond_len';
			raise RuntimeError("Error creating random polymers");
		if not 'type' in poly:
			print 'Polymer specification missing type';
			raise RuntimeError("Error creating random polymers");
		if not 'count' in poly:	
			print 'Polymer specification missing count';
			raise RuntimeError("Error creating random polymers");
		
		# build type list
		type_vector = hoomd.std_vector_string();
		for t in poly['type']:
			type_vector.append(t);
			if not t in types_used:
				types_used.append(t);
		
		# create the generator
		generator.addGenerator(poly['count'], hoomd.PolymerParticleGenerator(poly['bond_len'], type_vector, 100));
		
		
	# check that all used types are in the separation list
	for t in types_used:
		if not t in separation:
			print "No separation radius specified for type ", t;
			raise RuntimeError("Error creating random polymers");
			
	# set the separation radii
	for t,r in separation.items():
		generator.setSeparationRadius(t, r);
		
	# generate the particles
	generator.generate();
	
	globals.particle_data = hoomd.ParticleData(generator, my_exec_conf);
	
	# TEMPORARY HACK for bond initialization
	globals.initializer = generator;
	
	# initialize the system
	globals.system = hoomd.System(globals.particle_data, 0);
	
	_perform_common_init_tasks();
	return globals.particle_data;


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
	parser.add_option("-e", "--mode", dest="mode", help="Execution mode (cpu or gpu)");
	parser.add_option("-g", "--gpu", dest="gpu", help="GPU to execute on");
	(_options, args) = parser.parse_args();
	
	# chedk for valid mode setting
	if _options.mode:
		if not (_options.mode == "cpu" or _options.mode == "gpu"):
			parser.error("--mode must be either cpu or gpu");
	
	# check for sane options
	if _options.mode == "cpu" and _options.gpu:
		parser.error("It doesn't make sense to specify --mode=cpu and a value for --gpu")

	# set the mode to gpu if the gpu # was set
	if _options.gpu and not _options.mode:
		_options.mode = "gpu"
	
## Initializes the execution configuration
#
# \internal
# Given an initializer, create a particle data with a properly configured ExecutionConfiguration
def _create_exec_conf():
	global _options;
	
	# if no command line options were specified, create a default ExecutionConfiguration
	if not _options.mode:
		exec_conf = hoomd.ExecutionConfiguration();
	else:
		if _options.gpu:
			gpu_id = int(_options.gpu);
		else:
			gpu_id = 0;
		
		# create the specified configuration
		if _options.mode == "cpu":
			exec_conf = hoomd.ExecutionConfiguration(hoomd.ExecutionConfiguration.executionMode.CPU, gpu_id);
		elif _options.mode == "gpu":
			exec_conf = hoomd.ExecutionConfiguration(hoomd.ExecutionConfiguration.executionMode.GPU, gpu_id);
		else:
			raise RuntimeError("Invalid value for _options.exec in initialization");
		
	return exec_conf;
		
