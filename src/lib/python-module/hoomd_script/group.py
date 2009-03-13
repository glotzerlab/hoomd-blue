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

# $Id: __init__.py 1521 2008-12-03 18:34:14Z joaander $
# $URL: https://svn2.assembla.com/svn/hoomd/branches/hoomd-0.8/src/lib/python-module/hoomd_script/__init__.py $

import hoomd;
import sys;
import util;
import globals;

## \package hoomd_script.group
# \brief Commands for grouping particles
#
# This package contains various commands for making groups of particles

## Defines a group of particles
#
# group should not be created dirctly in hoomd_script code. The following methods can be used to create particle groups.
# - group.all()
# - group.type()
# - group.tags()
#
# The above methods assign a descriptive name based on the criteria chosen. That name can be easily changed if desired:
# \code
# groupA = group.type('A')
# groupA.name = "my new group name"
# \endcode
#
# Once a group has been created, it can be combined with others to form more complicated groups. To create a new group
# that contains the intersection of all the particles present in two different groups, use the & operator. Similarly, 
# the | operator creates a new group that is the a union of all particles in two different groups.
#
# \b Examles:
# \code
# # create a group containing all particles in group A and those with 
# # tags 100-199
# groupA = group.type('A')
# group100_199 = group.tags(100, 199);
# group_combined = groupA | group100_199;
#
# # create a group containing all particles in group A that also have 
# # tags 100-199
# groupA = group.type('A')
# group100_199 = group.tags(100, 199);
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
# groupA = group.type('A')
# groupB = group.type('B')
# \endcode
def type(type):
	util.print_status_line();
	
	# check if initialization has occured
	if globals.system == None:
		print >> sys.stderr, "\n***Error! Cannot create a group before initialization\n";
		raise RuntimeError('Error creating group');

	# create the group
	type_id = globals.system_definition.getParticleData().getTypeByName(type);
	name = 'type ' + type;
	cpp_group = hoomd.ParticleGroup(globals.system_definition.getParticleData(), hoomd.ParticleGroup.criteriaOption.type, type_id, type_id);

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
# half1 = group.tags(0, 999)
# half2 = group.tags(1000, 1999)
# \endcode
def tags(tag_min, tag_max=None):
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
	cpp_group = hoomd.ParticleGroup(globals.system_definition.getParticleData(), hoomd.ParticleGroup.criteriaOption.tag, tag_min, tag_max);

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
# all = group.all()
# \endcode
def all():
	util.print_status_line();

	# check if initialization has occured
	if globals.system == None:
		print >> sys.stderr, "\n***Error! Cannot create a group before initialization\n";
		raise RuntimeError('Error creating group');

	# choose the tag range
	tag_min = 0;
	tag_max = globals.system_definition.getParticleData().getN()-1;

	# create the group
	name = 'all';
	cpp_group = hoomd.ParticleGroup(globals.system_definition.getParticleData(), hoomd.ParticleGroup.criteriaOption.tag, tag_min, tag_max);

	# notify the user of the created group
	print 'Group "' + name + '" created containing ' + str(cpp_group.getNumMembers()) + ' particles';

	# return it in the wrapper class
	return group(name, cpp_group);
