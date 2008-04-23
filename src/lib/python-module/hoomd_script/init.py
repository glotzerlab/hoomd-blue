# $Id$
# $URL$

import hoomd
import globals

## \package hoomd_script.init
# \brief Data initialization commands
#
# These commands initialize the particle data. The execution
# configuration must be defined first (todo: document this),
# and the initialization must be performed before any other 
# script commands can be run.  
#

## Reads particles from a hoomd_xml file
#
# \param file_name File to read in
# \return Initialized ParticleData
# \pre The execution configuration is set
# \pre The system is not yet initialized
# \post The system is initialized
#
# \b Examples:<br>
# init.read_xml(file_name="data.xml")<br>
# pdata = init.read_xml(file_name="directory/data.xml")<br>
#
# All particles, bonds, etc...  are read from the hoomd_xml file given.
# After this command completes, the system is initialized allowing 
# many other commands in hoomd_script to be run. For more details
# on the file format read by this command, see hoomd_xml.
#
# Initialization can only occur once. An error will be generated
# if any other initialization command is called after read_xml().
#
def read_xml(file_name):
	# check if initialization has already occured
	if (globals.particle_data != None):
		print "Error: Cannot initialize more than once";
		raise RuntimeError('Error initializing');

	# read in the data
	initializer = hoomd.HOOMDInitializer(file_name);
	globals.particle_data = hoomd.ParticleData(initializer);

	# initialize the system
	globals.system = hoomd.System(globals.particle_data, initializer.getTimeStep());
	
	_perform_common_init_tasks();
	return globals.particle_data;


## Generates randomly positioned particles
#
# \param N Number of particles to create
# \param phi_p Packing fraction of particles in the simulation box
# \param min_dist Minimum distance particles will be separated by
# \param wall_offset (optional) If specified, walls are created a distance of 
#	\a wall_offset in from the edge of the simulation box
# \return Initialized ParticleData
# \pre The execution configuration is set
# \pre The system is not yet initialized
# \post The system is initialized
#
# \b Examples:<br>
# init.create_random(N=2400, phi_p=0.20)<br>
# init.create_random(N=2400, phi_p=0.40, min_dist=0.5)<br>
# init.create_random(wall_offset=3.1, phi_p=0.10, N=6000)<br>
#
# \a N particles are randomly placed in the simulation box. The 
# dimensions of the created box are such that the packing fraction
# of particles in the box is \a phi_p. A number density \e n
# can be related to the packing fraction by \f$n = 6/\pi \cdot \phi_P\f$.
# All particles are created with the same type, 0.
# After this command completes, the system is initialized allowing 
# many other commands in hoomd_script to be run.
#
# Initialization can only occur once. An error will be generated
# if any other initialization command is called after create_random().
#
def create_random(N, phi_p, min_dist=1.0, wall_offset=None):
	# check if initialization has already occured
	if (globals.particle_data != None):
		print "Error: Cannot initialize more than once";
		raise RuntimeError('Error initializing');

	# read in the data
	if wall_offset == None:
		initializer = hoomd.RandomInitializer(N, phi_p, min_dist);
	else:
		initializer = hoomd.RandomInitializerWithWalls(N, phi_p, min_dist, wall_offset);
		
	globals.particle_data = hoomd.ParticleData(initializer);

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
# Currently does nothing
def _perform_common_init_tasks():
	pass
