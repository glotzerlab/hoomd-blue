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

import hoomd;
import sys;
import util;

## \package hoomd_script
# \brief Base module for the user-level scripting API
# 
# hoomd_script provides a very high level user interface for executing 
# simulations using HOOMD. This python module is designed to be imported
# into python with "from hoomd_script import *"
#
# More details to add later...

## \internal
# \brief Internal python variable 
__all__ = ["analyze", "bond", "dump", "force", "globals", "init", 
			"integrate", "pair", "update", "wall", "run", "group_tags", 
			"group_type", "group_all", "hoomd"];

## \brief Runs the simulation for a given number of time steps
#
# \param tsteps Number of timesteps to advance the simulation by
# \param profile Set to true to enable detailed profiling
# 
# \b Examples:
# \code
# run(1000)
# run(10e6)
# run(10000, profile=True)
# \endcode
#
# Execute the run() command to advance the simulation forward in time. 
# During the run, all previously specified \ref analyze "analyzers", 
# \ref dump "dumps", \ref update "updaters" and the \ref integrate "integrators"
# are executed every so many time steps as specified by their individual periods.
# 
# After run() completes, you may change parameters of the simulation (i.e. temperature)
# and continue the simulation by executing run() again. Time steps are added
# cumulatively, so calling run(1000) and then run(2000) would run the simulation
# up to time step 3000.
#
# run() cannot be executed before the system is \ref init "initialized". In most 
# cases, it also doesn't make sense to execute run() until after pair forces, bond forces,
# and an \ref integrate "integrator" have been created.
#
# When \a profile is \em True, a detailed breakdown of how much time was spent in each
# portion of the calculation is printed at the end of the run. Collecting this timing information
# can slow the simulation on the GPU by ~5 percent, so only enable profiling for testing
# and troubleshooting purposes.
def run(tsteps, profile=False):
	util.print_status_line();
	# check if initialization has occured
	if (globals.system == None):
		print >> sys.stderr, "\n***Error! Cannot run before initialization\n";
		raise RuntimeError('Error running');
		
	if (globals.integrator == None):
		print "***Warning! Starting a run without an integrator set";
	else:
		globals.integrator.update_forces();
	
	for logger in globals.loggers:
		logger.update_quantities();
	globals.system.enableProfiler(profile);
	
	print "** starting run **"
	globals.system.run(int(tsteps));
	print "** run complete **"

## Defines a group of particles
#
# group should not be created dirctly in hoomd_script code. The following methods can be used to create particle groups.
# - group_all()
# - group_type()
# - group_tags()
#
# The above methods assign a descriptive name based on the criteria chosen. That name can be easily changed if desired:
# \code
# groupA = group_type('A')
# groupA.name = "my new group name"
# \endcode
#
# Once a group has been created, it can be combined with others to form more complicated groups. To create a new group
# that contains the intersection of all the particles present in two different groups, use the & operator. Similarly, 
# the | operator creates a new group that is the a union of all particles in two different groups.
#
# \b Examles:
# \code
# # create a group containing all particles in group A and those with tags 100-199
# groupA = group_type('A')
# group100_199 = group_tags(100, 199);
# group_combined = groupA | group100_199;
#
# # create a group containing all particles in group A that also have tags 100-199
# groupA = group_type('A')
# group100_199 = group_tags(100, 199);
# group_combined = groupA & group100_199;
# \endcode
class group:
	## \internal
	# \brief Creates a group
	# 
	# \param name Name of the group
	# \param cpp_group an instance of hoomd.ParticleData that defines the group
	def __init__(self, name, cpp_group):
		# initialize the group
		self.name = name;
		self.cpp_group = cpp_group;
	
	## \internal
	# \brief Creates a new group as the intersection of two given groups
	# 
	# \param a group to perform the interesection with
	def __and__(self, a):
		new_name = '(' + self.name + ' & ' + a.name + ')';
		new_cpp_group = hoomd.ParticleGroup.groupIntersection(self.cpp_group, a.cpp_group);
		return group(new_name, new_cpp_group);
	
	## \internal
	# \brief Creates a new group as the union of two given groups
	# 
	# \param a group to perform the interesection with
	def __or__(self, a):
		new_name = '(' + self.name + ' | ' + a.name + ')';
		new_cpp_group = hoomd.ParticleGroup.groupUnion(self.cpp_group, a.cpp_group);
		return group(new_name, new_cpp_group);
		
## Groups particles by type
#
# \param type Name of the particle type to add to the group
# 
# Creates a particle group from particles that match the given type. The group can then be used by other hoomd_script commands
# (such as analyze.msd) to specify which particles should be operated on.
#
# Particle groups can be combined in various ways to build up more complicated matches. See group for information and examples.
# 
# \b Examples:
# \code
# groupA = group_type('A')
# groupB = group_type('B')
# \endcode
def group_type(type):
	util.print_status_line();
	
	# check if initialization has occured
	if globals.system == None:
		print >> sys.stderr, "\n***Error! Cannot create a group before initialization\n";
		raise RuntimeError('Error creating group');

	# create the group
	type_id = globals.particle_data.getTypeByName(type);
	name = 'type ' + type;
	cpp_group = hoomd.ParticleGroup(globals.particle_data, hoomd.ParticleGroup.criteriaOption.type, type_id, type_id);

	# notify the user of the created group
	print 'Group "' + name + '" created containing ' + str(cpp_group.getNumMembers()) + ' particles';

	# return it in the wrapper class
	return group(name, cpp_group);
	
## Groups particles by tag
#
# \param tag_min First tag in the range to include (inclusive)
# \param tag_max Last tag in the range to include (inclusive)
# 
# The second argument (tag_max) is optional. If it is not specified, then a single particle with tag=tag_min will be added to the group. 
#
# Creates a particle group from particles that match the given tag range. The group can then be used by other hoomd_script commands
# (such as analyze.msd) to specify which particles should be operated on.
#
# Particle groups can be combined in various ways to build up more complicated matches. See group for information and examples.
# 
# \b Examples:
# \code
# half1 = group_tags(0, 999)
# half2 = group_tags(1000, 1999)
# \endcode
def group_tags(tag_min, tag_max=None):
	util.print_status_line();
	
	# check if initialization has occured
	if globals.system == None:
		print >> sys.stderr, "\n***Error! Cannot create a group before initialization\n";
		raise RuntimeError('Error creating group');
	
	# handle the optional argument	
	if tag_max != None:
		name = 'tags ' + str(tag_min) + '-' + str(tag_max);
	else:
		# if the option is not specified, tag_max is set equal to tag_min to include only that particle in the range
		# and the name is chosen accordingly
		tag_max = tag_min;
		name = 'tag ' + str(tag_min);

	# create the group
	cpp_group = hoomd.ParticleGroup(globals.particle_data, hoomd.ParticleGroup.criteriaOption.tag, tag_min, tag_max);

	# notify the user of the created group
	print 'Group "' + name + '" created containing ' + str(cpp_group.getNumMembers()) + ' particles';

	# return it in the wrapper class
	return group(name, cpp_group);

## Groups all particles
#
# Creates a particle group from all particles in the simulation. The group can then be used by other hoomd_script commands
# (such as analyze.msd) to specify which particles should be operated on.
#
# Particle groups can be combined in various ways to build up more complicated matches. See group for information and examples.
# 
# \b Examples:
# \code
# all = group_all()
# \endcode
def group_all():
	util.print_status_line();

	# check if initialization has occured
	if globals.system == None:
		print >> sys.stderr, "\n***Error! Cannot create a group before initialization\n";
		raise RuntimeError('Error creating group');

	# choose the tag range
	tag_min = 0;
	tag_max = globals.particle_data.getN()-1;

	# create the group
	name = 'all';
	cpp_group = hoomd.ParticleGroup(globals.particle_data, hoomd.ParticleGroup.criteriaOption.tag, tag_min, tag_max);

	# notify the user of the created group
	print 'Group "' + name + '" created containing ' + str(cpp_group.getNumMembers()) + ' particles';

	# return it in the wrapper class
	return group(name, cpp_group);

