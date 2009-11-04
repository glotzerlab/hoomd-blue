# -*- coding: iso-8859-1 -*-
#Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
#(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
#Iowa State University and The Regents of the University of Michigan All rights
#reserved.

#HOOMD-blue may contain modifications ("Contributions") provided, and to which
#copyright is held, by various Contributors who have granted The Regents of the
#University of Michigan the right to modify and/or distribute such Contributions.

#Redistribution and use of HOOMD-blue, in source and binary forms, with or
#without modification, are permitted, provided that the following conditions are
#met:

#* Redistributions of source code must retain the above copyright notice, this
#list of conditions, and the following disclaimer.

#* Redistributions in binary form must reproduce the above copyright notice, this
#list of conditions, and the following disclaimer in the documentation and/or
#other materials provided with the distribution.

#* Neither the name of the copyright holder nor the names of HOOMD-blue's
#contributors may be used to endorse or promote products derived from this
#software without specific prior written permission.

#Disclaimer

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
#ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

#IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
#INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
#OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
#ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# $Id$
# $URL$
# Maintainer: joaander / All Developers are free to add commands for new features

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
# group should not be created directly in hoomd_script code. The following methods can be used to create particle 
# groups.
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
# \b Examples:
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
    # \param a group to perform the intersection with
    def __and__(self, a):
        new_name = '(' + self.name + ' & ' + a.name + ')';
        new_cpp_group = hoomd.ParticleGroup.groupIntersection(self.cpp_group, a.cpp_group);
        return group(new_name, new_cpp_group);
    
    ## \internal
    # \brief Creates a new group as the union of two given groups
    # 
    # \param a group to perform the union with
    def __or__(self, a):
        new_name = '(' + self.name + ' | ' + a.name + ')';
        new_cpp_group = hoomd.ParticleGroup.groupUnion(self.cpp_group, a.cpp_group);
        return group(new_name, new_cpp_group);
        
## Groups particles by type
#
# \param type Name of the particle type to add to the group
# 
# Creates a particle group from particles that match the given type. The group can then be used by other hoomd_script 
# commands (such as analyze.msd) to specify which particles should be operated on.
#
# Particle groups can be combined in various ways to build up more complicated matches. See group for information and 
# examples.
# 
# \b Examples:
# \code
# groupA = group.type('A')
# groupB = group.type('B')
# \endcode
def type(type):
    util.print_status_line();
    
    # check if initialization has occurred
    if globals.system == None:
        print >> sys.stderr, "\n***Error! Cannot create a group before initialization\n";
        raise RuntimeError('Error creating group');

    # create the group
    type_id = globals.system_definition.getParticleData().getTypeByName(type);
    name = 'type ' + type;
    selector = hoomd.ParticleSelectorType(globals.system_definition, type_id, type_id);
    cpp_group = hoomd.ParticleGroup(globals.system_definition, selector);

    # notify the user of the created group
    print 'Group "' + name + '" created containing ' + str(cpp_group.getNumMembers()) + ' particles';

    # return it in the wrapper class
    return group(name, cpp_group);
    
## Groups particles by tag
#
# \param tag_min First tag in the range to include (inclusive)
# \param tag_max Last tag in the range to include (inclusive)
# 
# The second argument (tag_max) is optional. If it is not specified, then a single particle with tag=tag_min will be
# added to the group. 
#
# Creates a particle group from particles that match the given tag range. The group can then be used by other
# hoomd_script commands
# (such as analyze.msd) to specify which particles should be operated on.
#
# Particle groups can be combined in various ways to build up more complicated matches. See group for information and
# examples.
# 
# \b Examples:
# \code
# half1 = group.tags(0, 999)
# half2 = group.tags(1000, 1999)
# \endcode
def tags(tag_min, tag_max=None):
    util.print_status_line();
    
    # check if initialization has occurred
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
    selector = hoomd.ParticleSelectorTag(globals.system_definition, tag_min, tag_max);
    cpp_group = hoomd.ParticleGroup(globals.system_definition, selector);

    # notify the user of the created group
    print 'Group "' + name + '" created containing ' + str(cpp_group.getNumMembers()) + ' particles';

    # return it in the wrapper class
    return group(name, cpp_group);

## Groups all particles
#
# Creates a particle group from all particles in the simulation. The group can then be used by other hoomd_script 
# commands (such as analyze.msd) to specify which particles should be operated on.
#
# Particle groups can be combined in various ways to build up more complicated matches. See group for information and 
# examples.
# 
# \b Examples:
# \code
# all = group.all()
# \endcode
def all():
    util.print_status_line();

    # check if initialization has occurred
    if globals.system == None:
        print >> sys.stderr, "\n***Error! Cannot create a group before initialization\n";
        raise RuntimeError('Error creating group');

    # choose the tag range
    tag_min = 0;
    tag_max = globals.system_definition.getParticleData().getN()-1;

    # create the group
    name = 'all';
    selector = hoomd.ParticleSelectorTag(globals.system_definition, tag_min, tag_max);
    cpp_group = hoomd.ParticleGroup(globals.system_definition, selector);

    # notify the user of the created group
    print 'Group "' + name + '" created containing ' + str(cpp_group.getNumMembers()) + ' particles';

    # return it in the wrapper class
    return group(name, cpp_group);

## Groups particles in a cuboid
#
# \param name User-assigned name for this group
# \param xmin (if set) Lower left x-coordinate of the cuboid
# \param xmax (if set) Upper right x-coordinate of the cuboid
# \param ymin (if set) Lower left y-coordinate of the cuboid
# \param ymax (if set) Upper right y-coordinate of the cuboid
# \param zmin (if set) Lower left z-coordinate of the cuboid
# \param zmax (if set) Upper right z-coordinate of the cuboid
#
# If any of the above parameters is not set, it will automatically be placed slightly outside of the simulation box
# dimension, allowing easy specification of slabs.
#
# Creates a particle group from particles that fall in the defined cuboid. Membership tests are performed via
# xmin <= x < xmax (and so forth for y and z) so that directly adjacent cuboids do not have overlapping group members.
#
# The group can then be used by other hoomd_script commands (such as analyze.msd) to specify which particles should be
# operated on.
#
# Particle groups can be combined in various ways to build up more complicated matches. See group for information and
# examples.
#
# \b Examples:
# \code
# slab = group.cuboid(name="slab", ymin=-3, ymax=3)
# cube = grouip.cuboid(name="cube", xmin=0, xmax=5, ymin=0, ymax=5, zmin=0, zmax=5)
# \endcode
def cuboid(name, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None):
    util.print_status_line();
    
    # check if initialization has occurred
    if globals.system == None:
        print >> sys.stderr, "\n***Error! Cannot create a group before initialization\n";
        raise RuntimeError('Error creating group');
    
    # handle the optional arguments
    box = globals.system_definition.getParticleData().getBox();
    if xmin == None:
        xmin = box.xlo - 0.5;
    if xmax == None:
        xmax = box.xhi + 0.5;
    if ymin == None:
        ymin = box.ylo - 0.5;
    if ymax == None:
        ymax = box.yhi + 0.5;
    if zmin == None:
        zmin = box.zlo - 0.5;
    if zmax == None:
        zmax = box.zhi + 0.5;
    
    ll = hoomd.Scalar3();
    ur = hoomd.Scalar3();
    ll.x = float(xmin);
    ll.y = float(ymin);
    ll.z = float(zmin);
    ur.x = float(xmax);
    ur.y = float(ymax);
    ur.z = float(zmax);
    
    # create the group
    selector = hoomd.ParticleSelectorCuboid(globals.system_definition, ll, ur);
    cpp_group = hoomd.ParticleGroup(globals.system_definition, selector);

    # notify the user of the created group
    print 'Group "' + name + '" created containing ' + str(cpp_group.getNumMembers()) + ' particles';

    # return it in the wrapper class
    return group(name, cpp_group);

