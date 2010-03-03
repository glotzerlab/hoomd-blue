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
import data;
import init;

## \package hoomd_script.group
# \brief Commands for grouping particles
#
# This package contains various commands for making groups of particles

## Defines a group of particles
#
# group should not be created directly in hoomd_script code. The following methods can be used to create particle 
# groups.
# - group.all()
# - group.cuboid()
# - group.nonrigid()
# - group.rigid()
# - group.tags()
# - group.type()
#
# The above methods assign a descriptive name based on the criteria chosen. That name can be easily changed if desired:
# \code
# groupA = group.type('A')
# groupA.name = "my new group name"
# \endcode
#
# Once a group has been created, it can be combined with others via set operations to form more complicated groups.
# Available operations are:
# - group.difference()
# - group.intersection()
# - group.union()
#
# \b Examples:
# \code
# # create a group containing all particles in group A and those with 
# # tags 100-199
# groupA = group.type('A')
# group100_199 = group.tags(100, 199);
# group_combined = group.union(name="combined", a=groupA, b=group100_199);
#
# # create a group containing all particles in group A that also have 
# # tags 100-199
# group_combined2 = group.intersection(name="combined2", a=groupA, b=group100_199);
#
# # create a group containing all particles that are not in group A
# all = group.all()
# group_notA = group.difference(name="notA", a=all, b=groupA)
# \endcode
#
# A group can also be queried with python sequence semantics.
#
# \b Examples:
# \code
# groupA = group.type('A')
# # print the number of particles in group A
# print len(groupA)
# # print the position of the first particle in the group
# print groupA[0].position
# # set the velocity of all particles in groupA to 0
# for p in groupA:
#     p.velocity = (0,0,0)
# \endcode
#
# For more information and examples on accessing the %data in this way, see hoomd_script.data.
# 
class group:
    ## \internal
    # \brief group iterator
    class group_iterator:
        def __init__(self, data):
            self.data = data;
            self.index = 0;
        def __iter__(self):
            return self;
        def next(self):
            if self.index == len(self.data):
                raise StopIteration;
            
            result = self.data[self.index];
            self.index += 1;
            return result;
    
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
    # \brief Get a particle_proxy reference to the i'th particle in the group
    # \param i Index of the particle in the group to get
    def __getitem__(self, i):
        if i >= len(self) or i < 0:
            raise IndexError;
        tag = self.cpp_group.getMemberTag(i);
        return data.particle_data_proxy(globals.system_definition.getParticleData(), tag);
    
    def __setitem__(self, i, p):
        raise RuntimeError('__setitem__ not implemented');

    ## \internal
    # \brief Get the number of particles in the group
    def __len__(self):
        return self.cpp_group.getNumMembers();

    ## \internal
    # \brief Get an informal string representing the object
    def __str__(self):
        result = "Particle Group " + self.name + " containing " + str(len(self)) + " particles";
        return result;

    ## \internal
    # \brief Return an interator
    def __iter__(self):
        return group.group_iterator(self);

## \name Group specifications

# {@

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
    if not init.is_initialized():
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
    if not init.is_initialized():
        print >> sys.stderr, "\n***Error! Cannot create a group before initialization\n";
        raise RuntimeError('Error creating group');
    
    # handle the optional arguments
    box = globals.system_definition.getParticleData().getBox();
    if xmin is None:
        xmin = box.xlo - 0.5;
    if xmax is None:
        xmax = box.xhi + 0.5;
    if ymin is None:
        ymin = box.ylo - 0.5;
    if ymax is None:
        ymax = box.yhi + 0.5;
    if zmin is None:
        zmin = box.zlo - 0.5;
    if zmax is None:
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

## Groups particles that do not belong to rigid bodieis
#
# Creates a particle group from particles. \b All particles that <b>do not</b> belong to a rigid body will be added to
# the group. The group can then be used by other hoomd_script commands (such as analyze.msd) to specify which particles
# should be operated on.
#
# The group is always named 'nonrigid'.
#
# Particle groups can be combined in various ways to build up more complicated matches. See group for information and 
# examples.
#
# \b Examples:
# \code
# nonrigid = group.nonrigid()
# \endcode
def nonrigid():
    util.print_status_line();
    
    # check if initialization has occurred
    if globals.system == None:
        print >> sys.stderr, "\n***Error! Cannot create a group before initialization\n";
        raise RuntimeError('Error creating group');

    # create the group
    name = 'nonrigid';
    selector = hoomd.ParticleSelectorRigid(globals.system_definition, False);
    cpp_group = hoomd.ParticleGroup(globals.system_definition, selector);

    # notify the user of the created group
    print 'Group "' + name + '" created containing ' + str(cpp_group.getNumMembers()) + ' particles';

    # return it in the wrapper class
    return group(name, cpp_group);

## Groups particles that belong to rigid bodieis
#
# Creates a particle group from particles. \b All particles that belong to a rigid body will be added to the group.
# The group can then be used by other hoomd_script commands (such as analyze.msd) to specify which particles should
# be operated on.
#
# The group is always named 'rigid'.
#
# Particle groups can be combined in various ways to build up more complicated matches. See group for information and 
# examples.
#
# \b Examples:
# \code
# rigid = group.rigid()
# \endcode
def rigid():
    util.print_status_line();
    
    # check if initialization has occurred
    if globals.system == None:
        print >> sys.stderr, "\n***Error! Cannot create a group before initialization\n";
        raise RuntimeError('Error creating group');

    # create the group
    name = 'rigid';
    selector = hoomd.ParticleSelectorRigid(globals.system_definition, True);
    cpp_group = hoomd.ParticleGroup(globals.system_definition, selector);

    # notify the user of the created group
    print 'Group "' + name + '" created containing ' + str(cpp_group.getNumMembers()) + ' particles';

    # return it in the wrapper class
    return group(name, cpp_group);

## Groups particles by tag
#
# \param tag_min First tag in the range to include (inclusive)
# \param tag_max Last tag in the range to include (inclusive)
# \param name User-assigned name for this group. If a name is not specified, a default one will be generated.
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
# half1 = group.tags(name="first-half", tag_min=0, tag_max=999)
# half2 = group.tags(name="second-half", tag_min=1000, tag_max=1999)
# \endcode
def tags(tag_min, tag_max=None, name=None):
    util.print_status_line();
    
    # check if initialization has occurred
    if not init.is_initialized():
        print >> sys.stderr, "\n***Error! Cannot create a group before initialization\n";
        raise RuntimeError('Error creating group');
    
    # handle the optional argument
    if tag_max is not None:
        if name is None:
            name = 'tags ' + str(tag_min) + '-' + str(tag_max);
    else:
        # if the option is not specified, tag_max is set equal to tag_min to include only that particle in the range
        # and the name is chosen accordingly
        tag_max = tag_min;
        if name is None:
            name = 'tag ' + str(tag_min);

    # create the group
    selector = hoomd.ParticleSelectorTag(globals.system_definition, tag_min, tag_max);
    cpp_group = hoomd.ParticleGroup(globals.system_definition, selector);

    # notify the user of the created group
    print 'Group "' + name + '" created containing ' + str(cpp_group.getNumMembers()) + ' particles';

    # return it in the wrapper class
    return group(name, cpp_group);

## Groups particles by type
#
# \param type Name of the particle type to add to the group
# \param name User-assigned name for this group. If a name is not specified, a default one will be generated.
#
# Creates a particle group from particles that match the given type. The group can then be used by other hoomd_script 
# commands (such as analyze.msd) to specify which particles should be operated on.
#
# Particle groups can be combined in various ways to build up more complicated matches. See group for information and 
# examples.
#
# \b Examples:
# \code
# groupA = group.type(name='a-particles', type='A')
# groupB = group.type(name='b-particles', type='B')
# \endcode
def type(type, name=None):
    util.print_status_line();
    
    # check if initialization has occurred
    if not init.is_initialized():
        print >> sys.stderr, "\n***Error! Cannot create a group before initialization\n";
        raise RuntimeError('Error creating group');

    # create the group
    type_id = globals.system_definition.getParticleData().getTypeByName(type);
    if name is None:
        name = 'type ' + type;
    selector = hoomd.ParticleSelectorType(globals.system_definition, type_id, type_id);
    cpp_group = hoomd.ParticleGroup(globals.system_definition, selector);

    # notify the user of the created group
    print 'Group "' + name + '" created containing ' + str(cpp_group.getNumMembers()) + ' particles';

    # return it in the wrapper class
    return group(name, cpp_group);

# @}

## \name Group combinations

# {@

## Create a new group from the set difference of two existing groups
#
# \param name User-assigned name for this group
# \param a First group
# \param b Second group
#
# A new group is created that contains particles that are present in group \a and \b not in \a group b. This can be
# useful for inverting the sense of a group (see below).
#
# \b Examples:
# \code
# groupA = group.type(name='groupA', type='A')
# all = group.all()
# nottypeA = group.union(name="particles-not-typeA", a=all, b=groupA)
# \endcode
def difference(name, a, b):
    new_cpp_group = hoomd.ParticleGroup.groupDifference(a.cpp_group, b.cpp_group);
    # notify the user of the created group
    print 'Group "' + name + '" created containing ' + str(new_cpp_group.getNumMembers()) + ' particles';
    return group(name, new_cpp_group);

## Create a new group from the set intersection of two existing groups
#
# \param name User-assigned name for this group
# \param a First group
# \param b Second group
#
# A new group is created that contains only those particles present in both groups \a a and \a b, and is given name
# \a name.
#
# \b Examples:
# \code
# groupA = group.type(name='groupA', type='A')
# group100_199 = group.tags(name='100_199', tag_min=100, tag_max=199);
# groupC = group.intersection(name="groupC", a=groupA, b=group100_199)
# \endcode
def intersection(name, a, b):
    new_cpp_group = hoomd.ParticleGroup.groupIntersection(a.cpp_group, b.cpp_group);
    # notify the user of the created group
    print 'Group "' + name + '" created containing ' + str(new_cpp_group.getNumMembers()) + ' particles';
    return group(name, new_cpp_group);

## Create a new group from the set union of two existing groups
#
# \param name User-assigned name for this group
# \param a First group
# \param b Second group
#
# A new group is created that contains all particles present in either of the two groups \a a and \a b, and is given 
# name \a name.
#
# \b Examples:
# \code
# groupA = group.type(name='groupA', type='A')
# groupB = group.type(name='groupB', type='B')
# groupAB = group.union(name="ab-particles", a=groupA, b=groupB)
# \endcode
def union(name, a, b):
    new_cpp_group = hoomd.ParticleGroup.groupUnion(a.cpp_group, b.cpp_group);
    # notify the user of the created group
    print 'Group "' + name + '" created containing ' + str(new_cpp_group.getNumMembers()) + ' particles';
    return group(name, new_cpp_group);

# @}

