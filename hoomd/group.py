# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander / All Developers are free to add commands for new features

R""" Commands for grouping particles

This package contains various commands for making groups of particles
"""

from hoomd import _hoomd;
import hoomd;
import sys;

class group(hoomd.meta._metadata):
    R""" Defines a group of particles

    group should not be created directly in user code. The following methods can be used to create particle
    groups.

    * :py:func:`hoomd.group.all()`
    * :py:func:`hoomd.group.cuboid()`
    * :py:func:`hoomd.group.nonrigid()`
    * :py:func:`hoomd.group.rigid()`
    * :py:func:`hoomd.group.tags()`
    * :py:func:`hoomd.group.tag_list()`
    * :py:func:`hoomd.group.type()`
    * :py:func:`hoomd.group.charged()`

    The above functions assign a descriptive name based on the criteria chosen. That name can be easily changed if desired::

        groupA = group.type('A')
        groupA.name = "my new group name"

    Once a group has been created, it can be combined with others via set operations to form more complicated groups.
    Available operations are:

    * :py:func:`hoomd.group.difference()`
    * :py:func:`hoomd.group.intersection()`
    * :py:func:`hoomd.group.union()`

    Note:
        Groups need to be consistent with the particle data. If a particle member is removed from the simulation,
        it will be temporarily removed from the group as well, that is, even though the group reports that tag as a member,
        it will act as if the particle was not existent. If a particle with the same tag is later added to the simulation,
        it will become member of the group again.

    Examples::

        # create a group containing all particles in group A and those with
        # tags 100-199
        groupA = group.type('A')
        group100_199 = group.tags(100, 199);
        group_combined = group.union(name="combined", a=groupA, b=group100_199);

        # create a group containing all particles in group A that also have
        # tags 100-199
        group_combined2 = group.intersection(name="combined2", a=groupA, b=group100_199);

        # create a group containing all particles that are not in group A
        all = group.all()
        group_notA = group.difference(name="notA", a=all, b=groupA)

    A group can also be queried with python sequence semantics.

    Examples::

        groupA = group.type('A')
        # print the number of particles in group A
        print len(groupA)
        # print the position of the first particle in the group
        print groupA[0].position
        # set the velocity of all particles in groupA to 0
        for p in groupA:
            p.velocity = (0,0,0)

    For more information and examples on accessing the data in this way, see :py:mod:`hoomd.data`.
    """

    ## \internal
    # \brief group iterator
    class group_iterator:
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
    # \brief Creates a group
    #
    # \param name Name of the group
    # \param cpp_group an instance of _hoomd.ParticleData that defines the group
    def __init__(self, name, cpp_group):
        # initialize the group
        self.name = name;
        self.cpp_group = cpp_group;

        # base class constructor
        hoomd.meta._metadata.__init__(self)
        self.metadata_fields = ['name']

    def force_update(self):
        R""" Force an update of the group.

        Re-evaluate all particles against the original group selection criterion and build a new
        member list based on the current state of the system. For example, call :py:meth:`hoomd.group.group.force_update()`
        set set a cuboid group's membership to particles that are currently in the defined region.

        Groups made by a combination (union, intersection, difference) of other groups will not
        update their membership, they are always static.
        """
        self.cpp_group.updateMemberTags(True);

    ## \internal
    # \brief Get a particle_proxy reference to the i'th particle in the group
    # \param i Index of the particle in the group to get
    def __getitem__(self, i):
        if i >= len(self) or i < 0:
            raise IndexError;
        tag = self.cpp_group.getMemberTag(i);
        return hoomd.data.particle_data_proxy(hoomd.context.current.system_definition.getParticleData(), tag);

    def __setitem__(self, i, p):
        raise RuntimeError('__setitem__ not implemented');

    ## \internal
    # \brief Get the number of particles in the group
    def __len__(self):
        return self.cpp_group.getNumMembersGlobal();

    ## \internal
    # \brief Get an informal string representing the object
    def __str__(self):
        result = "Particle Group " + self.name + " containing " + str(len(self)) + " particles";
        return result;

    ## \internal
    # \brief Return an iterator
    def __iter__(self):
        return group.group_iterator(self);

def all():
    R""" Groups all particles.

    Creates a particle group from all particles in the simulation.

    Examples::

        all = group.all()
    """

    hoomd.util.print_status_line();

    # check if initialization has occurred
    if not hoomd.init.is_initialized():
        hoomd.context.msg.error("Cannot create a group before initialization\n");
        raise RuntimeError('Error creating group');

    name = 'all';

    # the all group is special: when the first one is created, it is cached in the context and future calls to group.all()
    # return the cached version
    if hoomd.context.current.group_all is not None:
        expected_N = hoomd.context.current.system_definition.getParticleData().getNGlobal();

        if len(hoomd.context.current.group_all) != expected_N:
            hoomd.context.msg.error("hoomd.context.current.group_all does not appear to be the group of all particles!\n");
            raise RuntimeError('Error creating group');
        return hoomd.context.current.group_all;

    # create the group
    selector = _hoomd.ParticleSelectorAll(hoomd.context.current.system_definition)
    cpp_group = _hoomd.ParticleGroup(hoomd.context.current.system_definition, selector, True);

    # notify the user of the created group
    hoomd.context.msg.notice(2, 'Group "' + name + '" created containing ' + str(cpp_group.getNumMembersGlobal()) + ' particles\n');

    # cache it and then return it in the wrapper class
    hoomd.context.current.group_all = group(name, cpp_group);
    return hoomd.context.current.group_all;

def cuboid(name, xmin=None, xmax=None, ymin=None, ymax=None, zmin=None, zmax=None):
    R""" Groups particles in a cuboid.

    Args:
        name (str): User-assigned name for this group
        xmin (float): (if set) Lower left x-coordinate of the cuboid (in distance units)
        xmax (float): (if set) Upper right x-coordinate of the cuboid (in distance units)
        ymin (float): (if set) Lower left y-coordinate of the cuboid (in distance units)
        ymax (float): (if set) Upper right y-coordinate of the cuboid (in distance units)
        zmin (float): (if set) Lower left z-coordinate of the cuboid (in distance units)
        zmax (float): (if set) Upper right z-coordinate of the cuboid (in distance units)

    If any of the above parameters is not set, it will automatically be placed slightly outside of the simulation box
    dimension, allowing easy specification of slabs.

    Creates a particle group from particles that fall in the defined cuboid. Membership tests are performed via
    ``xmin <= x < xmax`` (and so forth for y and z) so that directly adjacent cuboids do not have overlapping group members.

    Note:
        Membership in :py:class:`cuboid` is defined at time of group creation. Once created,
        any particles added to the system will not be added to the group. Any particles that move
        into the cuboid region will not be added automatically, and any that move out will not be
        removed automatically.

    Between runs, you can force a group to update its membership with the particles currently
    in the originally defined region using :py:meth:`hoomd.group.group.force_update()`.

    Examples::

        slab = group.cuboid(name="slab", ymin=-3, ymax=3)
        cube = group.cuboid(name="cube", xmin=0, xmax=5, ymin=0, ymax=5, zmin=0, zmax=5)
        run(100)
        # Remove particles that left the region and add particles that entered the region.
        cube.force_update()

    """
    hoomd.util.print_status_line();

    # check if initialization has occurred
    if not hoomd.init.is_initialized():
        hoomd.context.msg.error("Cannot create a group before initialization\n");
        raise RuntimeError('Error creating group');

    # handle the optional arguments
    box = hoomd.context.current.system_definition.getParticleData().getGlobalBox();
    if xmin is None:
        xmin = box.getLo().x - 0.5;
    if xmax is None:
        xmax = box.getHi().x + 0.5;
    if ymin is None:
        ymin = box.getLo().y - 0.5;
    if ymax is None:
        ymax = box.getHi().y + 0.5;
    if zmin is None:
        zmin = box.getLo().z - 0.5;
    if zmax is None:
        zmax = box.getHi().z + 0.5;

    ll = _hoomd.Scalar3();
    ur = _hoomd.Scalar3();
    ll.x = float(xmin);
    ll.y = float(ymin);
    ll.z = float(zmin);
    ur.x = float(xmax);
    ur.y = float(ymax);
    ur.z = float(zmax);

    # create the group
    selector = _hoomd.ParticleSelectorCuboid(hoomd.context.current.system_definition, ll, ur);
    cpp_group = _hoomd.ParticleGroup(hoomd.context.current.system_definition, selector);

    # notify the user of the created group
    hoomd.context.msg.notice(2, 'Group "' + name + '" created containing ' + str(cpp_group.getNumMembersGlobal()) + ' particles\n');

    # return it in the wrapper class
    return group(name, cpp_group);

def rigid_center():
    R""" Groups particles that are center particles of rigid bodies.

    Creates a particle group from particles. All particles that are central particles of rigid bodies be added to the group.
    The group is always named 'rigid_center'.

    Examples::

        rigid = group.rigid_center()

    """
    hoomd.util.print_status_line();

    # check if initialization has occurred
    if not hoomd.init.is_initialized():
        hoomd.context.error("Cannot create a group before initialization\n");
        raise RuntimeError('Error creating group');

    # create the group
    name = 'rigid_center';
    selector = _hoomd.ParticleSelectorRigidCenter(hoomd.context.current.system_definition);
    cpp_group = _hoomd.ParticleGroup(hoomd.context.current.system_definition, selector, True);

    # notify the user of the created group
    hoomd.context.msg.notice(2, 'Group "' + name + '" created containing ' + str(cpp_group.getNumMembersGlobal()) + ' particles\n');
    if cpp_group.getNumMembersGlobal() == 0:
        hoomd.context.msg.notice(2, 'It is OK if there are zero particles in this group. The group will be updated after run().\n');

    # return it in the wrapper class
    return group(name, cpp_group);

def nonrigid():
    R""" Groups particles that do not belong to rigid bodies.

    Creates a particle group from particles. All particles that **do not** belong to a rigid body will be added to
    the group. The group is always named 'nonrigid'.

    Examples::

        nonrigid = group.nonrigid()

    """
    hoomd.util.print_status_line();

    # check if initialization has occurred
    if not hoomd.init.is_initialized():
        hoomd.context.msg.error("Cannot create a group before initialization\n");
        raise RuntimeError('Error creating group');

    # create the group
    name = 'nonrigid';
    selector = _hoomd.ParticleSelectorRigid(hoomd.context.current.system_definition, False);
    cpp_group = _hoomd.ParticleGroup(hoomd.context.current.system_definition, selector);

    # notify the user of the created group
    hoomd.context.msg.notice(2, 'Group "' + name + '" created containing ' + str(cpp_group.getNumMembersGlobal()) + ' particles\n');

    # return it in the wrapper class
    return group(name, cpp_group);

def rigid():
    R""" Groups particles that belong to rigid bodies.

    Creates a particle group from particles. All particles that belong to a rigid body will be added to the group.
    The group is always named 'rigid'.

    Examples::

        rigid = group.rigid()

    """
    hoomd.util.print_status_line();

    # check if initialization has occurred
    if not hoomd.init.is_initialized():
        hoomd.context.msg.error("Cannot create a group before initialization\n");
        raise RuntimeError('Error creating group');

    # create the group
    name = 'rigid';
    selector = _hoomd.ParticleSelectorRigid(hoomd.context.current.system_definition,True);
    cpp_group = _hoomd.ParticleGroup(hoomd.context.current.system_definition, selector);

    # notify the user of the created group
    hoomd.context.msg.notice(2, 'Group "' + name + '" created containing ' + str(cpp_group.getNumMembersGlobal()) + ' particles\n');

    # return it in the wrapper class
    return group(name, cpp_group);

def nonfloppy():
    R""" Groups particles that do not belong to any floppy body.

    Creates a particle group from particles. All particles that **do not** belong to a floppy body will be added to
    the group. The group is always named 'nonfloppy'.

    Examples::

        nonfloppy = group.nonfloppy()

    """
    hoomd.util.print_status_line();

    # check if initialization has occurred
    if not hoomd.init.is_initialized():
        hoomd.context.msg.error("Cannot create a group before initialization\n");
        raise RuntimeError('Error creating group');

    # create the group
    name = 'nonfloppy';
    selector = _hoomd.ParticleSelectorFloppy(hoomd.context.current.system_definition, False);
    cpp_group = _hoomd.ParticleGroup(hoomd.context.current.system_definition, selector);

    # notify the user of the created group
    hoomd.context.msg.notice(2, 'Group "' + name + '" created containing ' + str(cpp_group.getNumMembersGlobal()) + ' particles\n');

    # return it in the wrapper class
    return group(name, cpp_group);

def floppy():
    R""" Groups particles that belong to any floppy body.

    Creates a particle group from particles. All particles that belong to a floppy will be added to the group.
    The group is always named 'floppy'.

    Examples::

        floppy = group.floppy()

    """
    hoomd.util.print_status_line();

    # check if initialization has occurred
    if not hoomd.init.is_initialized():
        hoomd.context.msg.error("Cannot create a group before initialization\n");
        raise RuntimeError('Error creating group');

    # create the group
    name = 'floppy';
    selector = _hoomd.ParticleSelectorFloppy(hoomd.context.current.system_definition, True);
    cpp_group = _hoomd.ParticleGroup(hoomd.context.current.system_definition, selector);

    # notify the user of the created group
    hoomd.context.msg.notice(2, 'Group "' + name + '" created containing ' + str(cpp_group.getNumMembersGlobal()) + ' particles\n');

    # return it in the wrapper class
    return group(name, cpp_group);

def nonbody():
    R""" Groups particles that do not belong to any body.

    Creates a particle group from particles. All particles that **do not** belong to a body will be added to
    the group. The group is always named 'nonbody'.

    Examples::

        nonbody = group.nonbody()

    """
    hoomd.util.print_status_line();

    # check if initialization has occurred
    if not hoomd.init.is_initialized():
        hoomd.context.msg.error("Cannot create a group before initialization\n");
        raise RuntimeError('Error creating group');

    # create the group
    name = 'nonbody';
    selector = _hoomd.ParticleSelectorBody(hoomd.context.current.system_definition, False);
    cpp_group = _hoomd.ParticleGroup(hoomd.context.current.system_definition, selector);

    # notify the user of the created group
    hoomd.context.msg.notice(2, 'Group "' + name + '" created containing ' + str(cpp_group.getNumMembersGlobal()) + ' particles\n');

    # return it in the wrapper class
    return group(name, cpp_group);

def body():
    R""" Groups particles that belong to any bodies.

    Creates a particle group from particles. All particles that belong to a body will be added to the group.
    The group is always named 'body'.

    Examples::

        body = group.body()

    """
    hoomd.util.print_status_line();

    # check if initialization has occurred
    if not hoomd.init.is_initialized():
        hoomd.context.msg.error("Cannot create a group before initialization\n");
        raise RuntimeError('Error creating group');

    # create the group
    name = 'body';
    selector = _hoomd.ParticleSelectorBody(hoomd.context.current.system_definition,True);
    cpp_group = _hoomd.ParticleGroup(hoomd.context.current.system_definition, selector);

    # notify the user of the created group
    hoomd.context.msg.notice(2, 'Group "' + name + '" created containing ' + str(cpp_group.getNumMembersGlobal()) + ' particles\n');

    # return it in the wrapper class
    return group(name, cpp_group);

def tags(tag_min, tag_max=None, name=None, update=False):
    R""" Groups particles by tag.

    Args:
        tag_min (int): First tag in the range to include (inclusive)
        tag_max (int): Last tag in the range to include (inclusive)
        name (str): User-assigned name for this group. If a name is not specified, a default one will be generated.
        update (bool): When True, update list of group members when particles are added to or removed from the simulation.

    Creates a particle group from particles that match the given tag range.

    The *tag_max* is optional. If it is not specified, then a single particle with ``tag=tag_min`` will be
    added to the group.

    Examples::

        half1 = group.tags(name="first-half", tag_min=0, tag_max=999)
        half2 = group.tags(name="second-half", tag_min=1000, tag_max=1999)

    """
    hoomd.util.print_status_line();

    # check if initialization has occurred
    if not hoomd.init.is_initialized():
        hoomd.context.msg.error("Cannot create a group before initialization\n");
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
    selector = _hoomd.ParticleSelectorTag(hoomd.context.current.system_definition, tag_min, tag_max);
    cpp_group = _hoomd.ParticleGroup(hoomd.context.current.system_definition, selector, update);

    # notify the user of the created group
    hoomd.context.msg.notice(2, 'Group "' + name + '" created containing ' + str(cpp_group.getNumMembersGlobal()) + ' particles\n');

    # return it in the wrapper class
    return group(name, cpp_group);

def tag_list(name, tags):
    R""" Groups particles by tag list.

    Args:
        tags (list): List of particle tags to include in the group
        name (str): User-assigned name for this group.

    Creates a particle group from particles with the given tags. Can be used to implement advanced grouping not
    available with existing group commands.

    Examples::

        a = group.tag_list(name="a", tags = [0, 12, 18, 205])
        b = group.tag_list(name="b", tags = range(20,400))

    """
    hoomd.util.print_status_line();

    # check if initialization has occurred
    if not hoomd.init.is_initialized():
        hoomd.context.msg.error("Cannot create a group before initialization\n");
        raise RuntimeError('Error creating group');

    # build a vector of the tags
    cpp_list = _hoomd.std_vector_uint();
    for t in tags:
        cpp_list.append(t);

    # create the group
    cpp_group = _hoomd.ParticleGroup(hoomd.context.current.system_definition, cpp_list);

    # notify the user of the created group
    hoomd.context.msg.notice(2, 'Group "' + name + '" created containing ' + str(cpp_group.getNumMembersGlobal()) + ' particles\n');

    # return it in the wrapper class
    return group(name, cpp_group);

def type(type, name=None, update=False):
    R""" Groups particles by type.

    Args:
        type (str): Name of the particle type to add to the group.
        name (str): User-assigned name for this group. If a name is not specified, a default one will be generated.
        update (bool): When true, update list of group members when particles are added to or removed from the simulation.

    Creates a particle group from particles that match the given type. The group can then be used by other hoomd
    commands (such as analyze.msd) to specify which particles should be operated on.

    Note:
        Membership in :py:func:`hoomd.group.type()` is defined at time of group creation. Once created,
        any particles added to the system will be added to the group if *update* is set to *True*.
        However, if you change a particle type it will not be added to or removed from this group.

    Between runs, you can force a group to update its membership with the particles currently
    in the originally  specified type using :py:meth:`hoomd.group.group.force_update()`.

    Examples::

        groupA = group.type(name='a-particles', type='A')
        groupB = group.type(name='b-particles', type='B')
        groupB = group.type(name='b-particles', type='B',update=True)

    """
    hoomd.util.print_status_line();
    type = str(type);

    # check if initialization has occurred
    if not hoomd.init.is_initialized():
        hoomd.context.msg.error("Cannot create a group before initialization\n");
        raise RuntimeError('Error creating group');

    if name is None:
        name = 'type ' + type;

    # get a list of types from the particle data
    ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
    type_list = [];
    for i in range(0,ntypes):
        type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i));

    if type not in type_list:
        hoomd.context.msg.warning(str(type) + " does not exist in the system, creating an empty group\n");
        cpp_list = _hoomd.std_vector_uint();
        cpp_group = _hoomd.ParticleGroup(hoomd.context.current.system_definition, cpp_list);
    else:
        type_id = hoomd.context.current.system_definition.getParticleData().getTypeByName(type);
        selector = _hoomd.ParticleSelectorType(hoomd.context.current.system_definition, type_id, type_id);
        cpp_group = _hoomd.ParticleGroup(hoomd.context.current.system_definition, selector, update);

    # notify the user of the created group
    hoomd.context.msg.notice(2, 'Group "' + name + '" created containing ' + str(cpp_group.getNumMembersGlobal()) + ' particles\n');

    # return it in the wrapper class
    return group(name, cpp_group);

def charged(name='charged'):
    R""" Groups particles that are charged.

    Args:
        name (str): User-assigned name for this group.

    Creates a particle group containing all particles that have a non-zero charge.

    Warning:
        This group currently does not support being updated when the number of particles changes.

    Examples::

        a = group.charged()
        b = group.charged(name="cp")

    """
    hoomd.util.print_status_line();

    # check if initialization has occurred
    if not hoomd.init.is_initialized():
        hoomd.context.msg.error("Cannot create a group before initialization\n");
        raise RuntimeError('Error creating group');

    hoomd.util.quiet_status();

    # determine the group of particles that are charged
    charged_tags = [];
    sysdef = hoomd.context.current.system_definition;
    pdata = hoomd.data.particle_data(sysdef.getParticleData());
    for i in range(0,len(pdata)):
        if pdata[i].charge != 0.0:
            charged_tags.append(i);

    retval = tag_list(name, charged_tags);
    hoomd.util.unquiet_status();

    return retval;

def difference(name, a, b):
    R""" Create a new group from the set difference or complement of two existing groups.

    Args:
        name (str): User-assigned name for this group.
        a (:py:class:`group`): First group.
        b (:py:class:`group`): Second group.

    The set difference of *a* and *b* is defined to be the set of particles that are in *a* and not in *b*.
    This can be useful for inverting the sense of a group (see below).

    A new group called *name* is created.

    Warning:
        The group is static and will not update if particles are added to
        or removed from the system.

    Examples::

        groupA = group.type(name='groupA', type='A')
        all = group.all()
        nottypeA = group.difference(name="particles-not-typeA", a=all, b=groupA)

    """

    new_cpp_group = _hoomd.ParticleGroup.groupDifference(a.cpp_group, b.cpp_group);
    # notify the user of the created group
    hoomd.context.msg.notice(2, 'Group "' + name + '" created containing ' + str(new_cpp_group.getNumMembersGlobal()) + ' particles\n');
    return group(name, new_cpp_group);

def intersection(name, a, b):
    R""" Create a new group from the set intersection of two existing groups.

    Args:
        name (str): User-assigned name for this group.
        a (:py:class:`group`): First group.
        b (:py:class:`group`): Second group.

    A new group is created that contains all particles of *a* that are also in *b*, and is given the name
    *name*.

    Warning:
        The group is static and will not update if particles are added to
        or removed from the system.

    Examples::

        groupA = group.type(name='groupA', type='A')
        group100_199 = group.tags(name='100_199', tag_min=100, tag_max=199);
        groupC = group.intersection(name="groupC", a=groupA, b=group100_199)

    """

    new_cpp_group = _hoomd.ParticleGroup.groupIntersection(a.cpp_group, b.cpp_group);
    # notify the user of the created group
    hoomd.context.msg.notice(2, 'Group "' + name + '" created containing ' + str(new_cpp_group.getNumMembersGlobal()) + ' particles\n');
    return group(name, new_cpp_group);

def union(name, a, b):
    R""" Create a new group from the set union of two existing groups.

    Args:
        name (str): User-assigned name for this group.
        a (:py:class:`group`): First group.
        b (:py:class:`group`): Second group.

    A new group is created that contains all particles present in either group *a* or *b*, and is given the
    name *name*.

    Warning:
        The group is static and will not update if particles are added to
        or removed from the system.

    Examples::

        groupA = group.type(name='groupA', type='A')
        groupB = group.type(name='groupB', type='B')
        groupAB = group.union(name="ab-particles", a=groupA, b=groupB)

    """
    new_cpp_group = _hoomd.ParticleGroup.groupUnion(a.cpp_group, b.cpp_group);
    # notify the user of the created group
    hoomd.context.msg.notice(2, 'Group "' + name + '" created containing ' + str(new_cpp_group.getNumMembersGlobal()) + ' particles\n');
    return group(name, new_cpp_group);
