# -- start license --
# Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
# (HOOMD-blue) Open Source Software License Copyright 2009-2016 The Regents of
# the University of Michigan All rights reserved.

# HOOMD-blue may contain modifications ("Contributions") provided, and to which
# copyright is held, by various Contributors who have granted The Regents of the
# University of Michigan the right to modify and/or distribute such Contributions.

# You may redistribute, use, and create derivate works of HOOMD-blue, in source
# and binary forms, provided you abide by the following conditions:

# * Redistributions of source code must retain the above copyright notice, this
# list of conditions, and the following disclaimer both in the code and
# prominently in any materials provided with the distribution.

# * Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions, and the following disclaimer in the documentation and/or
# other materials provided with the distribution.

# * All publications and presentations based on HOOMD-blue, including any reports
# or published results obtained, in whole or in part, with HOOMD-blue, will
# acknowledge its use according to the terms posted at the time of submission on:
# http://codeblue.umich.edu/hoomd-blue/citations.html

# * Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
# http://codeblue.umich.edu/hoomd-blue/

# * Apart from the above required attributions, neither the name of the copyright
# holder nor the names of HOOMD-blue's contributors may be used to endorse or
# promote products derived from this software without specific prior written
# permission.

# Disclaimer

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
# WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -- end license --

# Maintainer: jproc

## \package hoomd_script.wall
# \brief Commands that specify %wall geometry and forces.
#
# The following geometries are currently supported by wall potentials:\n \link
# wall.sphere spheres\endlink, \link wall.cylinder cylinders\endlink, \link
# wall.plane planes\endlink \n\n The following potentials are currently supported
# by wall potentials:\n \link wall.lj lj\endlink, \link wall.gauss gauss\endlink,
# \link wall.slj slj\endlink, \link wall.yukawa yukawa\endlink, \link wall.morse
# morse\endlink, \link wall.force_shifted_lj force_shifted_lj\endlink, and \link
# wall.mie mie\endlink.
#
# Wall potentials can add forces to any particles within a certain distance, \f$
# r_{\mathrm{cut}} \f$, of each wall and in the \link wall.wallpotential extrapolated
# mode\endlink all particles deemed outside of the wall boundary as well.
#
# Wall geometries are used to specify half-spaces. There are two half spaces for
# each of the possible geometries included and each can be selected using the
# inside parameter. In order to fully specify space, it is necessary that one half
# space be closed and one open. It was chosen that the inside=True geometries
# would be closed half-spaces and inside=False would be open half-spaces. See
# wall.wallpotential for more infomation on how the concept of half-spaces is used
# in implementing wall forces.
#
# \note \par
# The current wall force implementation does not support NPT integrators.
#
# Wall groups are used to pass wall geometries to wall forces. See \link
# wall.group wall.group\endlink for more details. By themselves, wall groups do
# nothing. Only when you specify a wall force (i.e. wall.lj),  are forces actually
# applied between the wall and the particle. See wall.wallpotential
# for more details of implementing a force.

from hoomd_script import external;
from hoomd_script import globals;
from hoomd_script import force;
from hoomd_script import util;
from hoomd_script import meta;
import hoomd;
import math;

#           *** Helpers ***

## Defines the %wall group used by wall potentials.
#
# All wall forces use a wall group as an input so it is necessary to create a
# %wall.%group object before any wall.force can be created, however modifications
# of the created wall structure can occur at any time before run() command is
# used. Current supported geometries are spheres, cylinder, and planes. The
# maximum number of each type of wall is 20, 20, and 60 respectively per group.
#
# The <b>inside</b> parameter used in each wall geometry is used to specify the
# half-space that is to be used for the force implementation. See
# wall.wallpotential for more general details and  \link wall.sphere
# spheres\endlink, \link wall.cylinder cylinders\endlink, and \link wall.plane
# planes\endlink for the meaning of inside in each geometry.
#
# There are multiple ways to achieve the same input wall group. This apparent
# redundancy allows for several additional functionalies which will be
# demonstrated below and in wall.sphere.
#
# An effective use of wall forces <b>requires</b> considering the geometry of the
# system. Walls are only evaluated in one simulation box and are not periodic. It
# is therefore important to ensure that the walls that intersect with a periodic
# boundary meet.
#
# \note \par
# The entire structure can easily be viewed by printing the wall.group object.
# This can be seen in the example in \link hoomd_script.wall.sphere
# sphere\endlink.
# \note \par
# While all x,y,z coordinates can be given as a list or tuple, only origin
# parameters are points in x,y,z space. Normal and axis parameters are vectors and
# must have a nonzero magnitude. The examples in \link
# hoomd_script.wall.sphere sphere\endlink demonstrates this in the
# default parameters.
# \note \par
# Wall structure modifications between \link hoomd_script.run() run()\endlink
# calls will be implemented in the next run. However, modifications must be done
# carefully since moving the wall can result in particles moving to a relative
# position which causes exceptionally high forces resulting in particles moving
# many times the box length in one move.
#
#
# \b Example:
# \n In[0]:
# \code
# # Creating wall geometry defintions using convenience functions
# wallstructure=wall.group()
# wallstructure.add_sphere(r=1.0,origin=(0,1,3))
# wallstructure.add_sphere(1.0,[0,-1,3],inside=False)
# wallstructure.add_cylinder(r=1.0,origin=(1,1,1),axis=(0,0,3),inside=True)
# wallstructure.add_cylinder(4.0,[0,0,0],(1,0,1))
# wallstructure.add_cylinder(5.5,(1,1,1),(3,-1,1),False)
# wallstructure.add_plane(origin=(3,2,1),normal=(2,1,4))
# wallstructure.add_plane((0,0,0),(10,2,1))
# wallstructure.add_plane((0,0,0),(0,2,1))
# print(wallstructure)
# \endcode
# Out[0]:
# \code
# Wall_Data_Sturucture:
# spheres:2{
# [0:	Radius=1.0	Origin=(0.0, 1.0, 3.0)	Inside=True]
# [1:	Radius=1.0	Origin=(0.0, -1.0, 3.0)	Inside=False]}
# cylinders:3{
# [0:	Radius=1.0	Origin=(1.0, 1.0, 1.0)	Axis=(0.0, 0.0, 3.0)	Inside=True]
# [1:	Radius=4.0	Origin=(0.0, 0.0, 0.0)	Axis=(1.0, 0.0, 1.0)	Inside=True]
# [2:	Radius=5.5	Origin=(1.0, 1.0, 1.0)	Axis=(3.0, -1.0, 1.0)	Inside=False]}
# planes:3{
# [0:	Origin=(3.0, 2.0, 1.0)	Normal=(2.0, 1.0, 4.0)]
# [1:	Origin=(0.0, 0.0, 0.0)	Normal=(10.0, 2.0, 1.0)]
# [2:	Origin=(0.0, 0.0, 0.0)	Normal=(0.0, 2.0, 1.0)]}
# \endcode
# In[1]:
# \code
# # Deleting wall geometry defintions using convenience functions in all accepted types
# wallstructure.del_plane(range(3))
# wallstructure.del_cylinder([0,2])
# wallstructure.del_sphere(1)
# print(wallstructure)
# \endcode
# Out[1]:
# \code
# Wall_Data_Sturucture:
# spheres:1{
# [0:	Radius=1.0	Origin=(0.0, 1.0, 3.0)	Inside=True]}
# cylinders:1{
# [0:	Radius=4.0	Origin=(0.0, 0.0, 0.0)	Axis=(1.0, 0.0, 1.0)	Inside=True]}
# planes:0{}
# \endcode
# In[2]:
# \code
# # Modifying wall geometry defintions using convenience functions
# wallstructure.spheres[0].r=2.0
# wallstructure.cylinders[0].origin=[1,2,1]
# wallstructure.cylinders[0].axis=(0,0,1)
# print(wallstructure)
# \endcode
# Out[2]:
# \code
# Wall_Data_Sturucture:
# spheres:1{
# [0:	Radius=2.0	Origin=(0.0, 1.0, 3.0)	Inside=True]}
# cylinders:1{
# [0:	Radius=4.0	Origin=(1.0, 2.0, 1.0)	Axis=(0.0, 0.0, 1.0)	Inside=True]}
# planes:0{}
# \endcode
class group():

    ## Creates the wall group which can be named to easily find in the metadata.
    # Required to call and create an object before walls can be added to that
    # object.
    # \param name Name of the wall structure (string, defaults to empty string).
    # \param walls Wall objects to be included in the group
    #
    # \b Example:
    # \code
    # empty_wall_object=wall.group()
    # full_wall_object=wall.group([wall.sphere()]*20,[wall.cylinder()]*20,[wall.plane()]*60)
    # \endcode
    def __init__(self,*walls):
        self.spheres=[];
        self.cylinders=[];
        self.planes=[];
        for wall in walls:
            self.add(wall);

    ## Generic wall add for wall objects.
    # Generic convenience function to add any wall object to the group.
    # Accepts \link hoomd_script.wall.sphere sphere\endlink, \link
    # hoomd_script.wall.cylinder cylinder\endlink, \link
    # hoomd_script.wall.plane plane\endlink, and lists of any
    # combination of these.
    def add(self,wall,index=False):
        if (type(wall)==type(sphere())):
            self.spheres.append(wall);
        elif (type(wall)==type(cylinder())):
            self.cylinders.append(wall);
        elif (type(wall)==type(plane())):
            self.planes.append(wall);
        elif (type(wall)==list):
            for wall_el in wall:
                if (type(wall_el)==type(sphere())):
                    self.spheres.append(wall_el);
                elif (type(wall_el)==type(cylinder())):
                    self.cylinders.append(wall_el);
                elif (type(wall_el)==type(plane())):
                    self.planes.append(wall_el);
                else:
                    print("Input of type "+str(type(wall_el))+" is not allowed. Skipping invalid list element...");
        else:
            print("Input of type "+str(type(wall))+" is not allowed.");

    ## Adds a sphere to the %wall group.
    # Adds a sphere with the specified parameters to the wallgroup.spheres list.
    # \param r Sphere radius (in distance units)
    # \param origin Sphere origin (in x,y,z coordinates)
    # \param inside Selects the half-space to be used (bool)
    def add_sphere(self, r, origin, inside=True):
        self.spheres.append(sphere(r,origin,inside));

    ## Adds a cylinder to the %wall group.
    # Adds a cylinder with the specified parameters to the wallgroup.cylinders list.
    # \param r Cylinder radius (in distance units)
    # \param origin Cylinder origin (in x,y,z coordinates)
    # \param axis Cylinder axis vector (in x,y,z coordinates)
    # \param inside Selects the half-space to be used (bool)
    def add_cylinder(self, r, origin, axis, inside=True):
        self.cylinders.append(cylinder(r, origin, axis, inside));

    ## Adds a plane to the %wall group.
    # Adds a plane with the specified parameters to the wallgroup.planes list.
    # \param origin Plane origin (in x,y,z coordinates)
    # \param normal Plane normal vector (in x,y,z coordinates)
    # \param inside Selects the half-space to be used (bool)
    def add_plane(self, origin, normal, inside=True):
        self.planes.append(plane(origin, normal, inside));

    ## Deletes the sphere or spheres in index.
    # Removes the specified sphere or spheres from the wallgroup.spheres list.
    # \param index The index of sphere(s) desired to delete. Accepts int, range, and lists.
    def del_sphere(self, *indexs):
        for index in indexs:
            if type(index) is int: index = [index];
            elif type(index) is range: index = list(index);
            index=list(set(index));
            index.sort(reverse=True);
            for i in index:
                try:
                    del(self.spheres[i]);
                except IndexValueError:
                    globals.msg.error("Specified index for deletion is not valid.\n");
                    raise RuntimeError("del_sphere failed")

    ## Deletes the cylinder or cylinders in index.
    # Removes the specified cylinder or cylinders from the wallgroup.cylinders list.
    # \param index The index of cylinder(s) desired to delete. Accepts int, range, and lists.
    def del_cylinder(self, *indexs):
        for index in indexs:
            if type(index) is int: index = [index];
            elif type(index) is range: index = list(index);
            index=list(set(index));
            index.sort(reverse=True);
            for i in index:
                try:
                    del(self.cylinders[i]);
                except IndexValueError:
                    globals.msg.error("Specified index for deletion is not valid.\n");
                    raise RuntimeError("del_cylinder failed")

    ## Deletes the plane or planes in index.
    # Removes the specified plane or planes from the wallgroup.planes list.
    # \param index The index of plane(s) desired to delete. Accepts int, range, and lists.
    def del_plane(self, *indexs):
        for index in indexs:
            if type(index) is int: index = [index];
            elif type(index) is range: index = list(index);
            index=list(set(index));
            index.sort(reverse=True);
            for i in index:
                try:
                    del(self.planes[i]);
                except IndexValueError:
                    globals.msg.error("Specified index for deletion is not valid.\n");
                    raise RuntimeError("del_plane failed")

    ## \internal
    # \brief Return metadata for this wall structure
    def get_metadata(self):
        data = meta._metadata_from_dict(eval(str(self.__dict__)));
        return data;

    ## \internal
    # \brief Returns output for print
    def __str__(self):
        output="Wall_Data_Sturucture:\nspheres:%s{"%(len(self.spheres));
        for index in range(len(self.spheres)):
            output+="\n[%s:\t%s]"%(repr(index), str(self.spheres[index]));

        output+="}\ncylinders:%s{"%(len(self.cylinders));
        for index in range(len(self.cylinders)):
            output+="\n[%s:\t%s]"%(repr(index), str(self.cylinders[index]));

        output+="}\nplanes:%s{"%(len(self.planes));
        for index in range(len(self.planes)):
            output+="\n[%s:\t%s]"%(repr(index), str(self.planes[index]));

        output+="}";
        return output;

## Sphere wall object
# Class that populates the spheres[] array in the wall.group object.
# Object which contains all the geometric information needed to define the division of space by a sphere.\n
# \n inside = True selects the space inside the radius of the sphere and includes the sphere surface.
# \n inside = False selects the space outside the radius of the sphere.
#
# Can be used in function calls or by reference in the creation or modification of wall groups.
#
# The following example is intended to demonstrate cylinders and planes
# as well. Note that the distinction between points and vectors is reflected in
# the default parameters.
# \n \b Examples
# \n In[0]:\code
# # One line initialization
# one_line_walls=wall.group(wall.sphere(r=3,origin=(0,0,0)),wall.cylinder(r=2.5,axis=(0,0,1),inside=True), wall.plane(normal=(1,0,0)))
# print(one_line_walls)
# full_wall_object=wall.group([wall.sphere()]*20,[wall.cylinder()]*20,[wall.plane()]*60)
# # Sharing wall group elements and access by reference
# common_sphere=wall.sphere()
# linked_walls1=wall.group(common_sphere,wall.plane(origin=(3,0,0),normal=(-1,0,0)))
# linked_walls2=wall.group(common_sphere,wall.plane(origin=(-3,0,0),normal=(1,0,0)))
# common_sphere.r=5.0
# linked_walls1.spheres[0].origin=(0,0,1)
# print(linked_walls1)
# print(linked_walls2)
# \endcode
# Out[0]:\code
# Wall_Data_Sturucture:
# spheres:1{
# [0:	Radius=3	Origin=(0.0, 0.0, 0.0)	Inside=True]}
# cylinders:1{
# [0:	Radius=2.5	Origin=(0.0, 0.0, 0.0)	Axis=(0.0, 0.0, 1.0)	Inside=True]}
# planes:1{
# [0:	Origin=(0.0, 0.0, 0.0)	Normal=(1.0, 0.0, 0.0)]}
# Wall_Data_Sturucture:
# spheres:1{
# [0:	Radius=5.0	Origin=(0.0, 0.0, 1.0)	Inside=True]}
# cylinders:0{}
# planes:1{
# [0:	Origin=(3.0, 0.0, 0.0)	Normal=(-1.0, 0.0, 0.0)]}
# Wall_Data_Sturucture:
# spheres:1{
# [0:	Radius=5.0	Origin=(0.0, 0.0, 1.0)	Inside=True]}
# cylinders:0{}
# planes:1{
# [0:	Origin=(-3.0, 0.0, 0.0)	Normal=(1.0, 0.0, 0.0)]}
# \endcode
class sphere:
    ## Creates a sphere wall definition.
    # \param r Sphere radius (in distance units)\n <i>Default : 0.0</i>
    # \param origin Sphere origin (in x,y,z coordinates)\n <i>Default : (0.0, 0.0, 0.0)</i>
    # \param inside Selects the half-space to be used (bool)\n <i>Default : True</i>
    def __init__(self, r=0.0, origin=(0.0, 0.0, 0.0), inside=True):
        self.r = r;
        self._origin = hoomd.make_scalar3(*origin);
        self.inside = inside;

    @property
    def origin(self):
        return (self._origin.x, self._origin.y, self._origin.z);
    @origin.setter
    def origin(self, origin):
        self._origin = hoomd.make_scalar3(*origin);

    def __str__(self):
        return "Radius=%s\tOrigin=%s\tInside=%s" % (str(self.r), str(self.origin), str(self.inside));

    def __repr__(self):
        return "{'r': %s, 'origin': %s, 'inside': %s}" % (str(self.r), str(self.origin), str(self.inside));

## Cylinder wall object
# Class that populates the cylinders[] array in the wall.group object.
# Object which contains all the geometric information needed to define the division of space by a cylinder.\n
# \n inside = True selects the space inside the radius of the cylinder and includes the cylinder surface.
# \n inside = False selects the space outside the radius of the cylinder.
#
# Can be used in function calls or by reference in the creation or modification of wall groups.
#
# For an example see \link hoomd_script.wall.sphere sphere\endlink.
class cylinder:
    ## Creates a cylinder wall definition.
    # \param r Cylinder radius (in distance units)\n <i>Default : 0.0</i>
    # \param origin Cylinder origin (in x,y,z coordinates)\n <i>Default : (0.0, 0.0, 0.0)</i>
    # \param axis Cylinder axis vector (in x,y,z coordinates)\n <i>Default : (0.0, 0.0, 1.0)</i>
    # \param inside Selects the half-space to be used (bool)\n <i>Default : True</i>
    def __init__(self, r=0.0, origin=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), inside=True):
        self.r = r;
        self._origin = hoomd.make_scalar3(*origin);
        self._axis = hoomd.make_scalar3(*axis);
        self.inside = inside;

    @property
    def origin(self):
        return (self._origin.x, self._origin.y, self._origin.z);
    @origin.setter
    def origin(self, origin):
        self._origin = hoomd.make_scalar3(*origin);

    @property
    def axis(self):
        return (self._axis.x, self._axis.y, self._axis.z);
    @axis.setter
    def axis(self, axis):
        self._axis = hoomd.make_scalar3(*axis);

    def __str__(self):
        return "Radius=%s\tOrigin=%s\tAxis=%s\tInside=%s" % (str(self.r), str(self.origin), str(self.axis), str(self.inside));

    def __repr__(self):
        return "{'r': %s, 'origin': %s, 'axis': %s, 'inside': %s}" % (str(self.r), str(self.origin), str(self.axis), str(self.inside));

## Plane wall object
# Class that populates the planes[] array in the wall.group object.
# Object which contains all the geometric information needed to define the division of space by a plane.\n
# \n inside = True selects the space on the side of the plane to which the normal vector points and includes the plane surface.
# \n inside = False selects the space on the side of the plane opposite the normal vector.
#
# Can be used in function calls or by reference in the creation or modification of wall groups.
#
# For an example see \link hoomd_script.wall.sphere sphere\endlink.
class plane:
    ## Creates a plane wall definition.
    # \param origin Plane origin (in x,y,z coordinates)\n <i>Default : (0.0, 0.0, 0.0)</i>
    # \param normal Plane normal vector (in x,y,z coordinates)\n <i>Default : (0.0, 0.0, 1.0)</i>
    # \param inside Selects the half-space to be used (bool)\n <i>Default : True</i>
    def __init__(self, origin=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0), inside=True):
        self._origin = hoomd.make_scalar3(*origin);
        self._normal = hoomd.make_scalar3(*normal);
        self.inside = inside;

    @property
    def origin(self):
        return (self._origin.x, self._origin.y, self._origin.z);
    @origin.setter
    def origin(self, origin):
        self._origin = hoomd.make_scalar3(*origin);

    @property
    def normal(self):
        return (self._normal.x, self._normal.y, self._normal.z);
    @normal.setter
    def normal(self, normal):
        self._normal = hoomd.make_scalar3(*normal);

    def __str__(self):
        return "Origin=%s\tNormal=%s\tInside=%s" % (str(self.origin), str(self.normal), str(self.inside));

    def __repr__(self):
        return "{'origin':%s, 'normal': %s, 'inside': %s}" % (str(self.origin), str(self.normal), str(self.inside));

#           *** Potentials ***

## Generic %wall %force
#
# wall.wallpotential is not a command hoomd scripts should execute directly.
# Rather, it is a base command that provides common features to all standard %wall
# forces. Rather than repeating all of that documentation in many different
# places, it is collected here.
#
#  All %wall %force commands specify that a given potential energy and %force be
# computed on all particles in the system within a cutoff distance, \f$
# r_{\mathrm{cut}} \f$, from each wall in the given wall \link wall.group
# group\endlink. The %force \f$ \vec{F}\f$ is where \f$ \vec{r} \f$ is the vector
# pointing from the particle to the %wall or half-space boundary and \f$
# V_{\mathrm{pair}}(r) \f$ is the specific %pair potential chosen by the
# respective command. Wall forces are implemented with the concept of half-spaces
# in mind. There are two modes which are allowed currently in wall potentials:
# standard and extrapolated.\n\n
#
# <b>Standard Mode:</b>\n
# In the standard mode, when \f$ r_{\mathrm{extrap}} \le 0 \f$, the potential
# energy is only applied to the half-space specified in the wall group. \f$ V(r)
# \f$ is evaluated in the same manner as when mode is shift for the analogous pair
# potentials within the boundaries of the half-space.
# \f{eqnarray*}{ V(r)  = & V_{\mathrm{pair}}(r) - V_{\mathrm{pair}}(r_{\mathrm{cut}}) \f}
# For inside=True (closed) half-spaces:
# \f{eqnarray*}{ \vec{F}  = & -\nabla V(r) & 0 \le r < r_{\mathrm{cut}} \\ = & 0 &
# r \ge r_{\mathrm{cut}} \\ = & 0 & r < 0 \f}
# For inside=False (open) half-spaces:
# \f{eqnarray*}{ \vec{F}  = & -\nabla V(r) & 0 < r < r_{\mathrm{cut}} \\ = & 0 & r
# \ge r_{\mathrm{cut}} \\ = & 0 & r \le 0 \f} \n\n
#
# <b>Extrapolated Mode:</b>\n
# The wall potential can be linearly extrapolated beyond a minimum separation from the wall
# \f$r_{\mathrm{extrap}}\f$ in the active half-space. This can be useful for bringing particles outside the
# half-space into the active half-space. It also useful for pushing particles off the wall by
# effectively limiting the maximum force experienced by the particle. The potential is extrapolated into <b>both</b>
# half-spaces and the \f$ r_{\mathrm{cut}} \f$ only applies in the active half-space. The user should then be careful
# using this mode with multiple nested walls. It is intended to be used primarily for equilibration.
#
# The extrapolated potential has the following form:
# \f{eqnarray*}{
# V(r) =& V_{\mathrm{pair}}(r) &, r > r_{\rm extrap} \\
#      =& V_{\mathrm{pair}}(r_{\rm extrap}) + (r_{\rm extrap}-r)\vec{F}_{\rm pair}(r_{\rm extrap}) \cdot \vec{n}&, r \le r_{\rm extrap}
# \f}
# where \f$\vec{n}\f$ is the normal into the active half-space.
# This gives an effective force on the particle due to the wall:
# \f{eqnarray*}{
# \vec{F} =& \vec{F}_{\rm pair}(r) &, r > r_{\rm extrap} \\
#         =& \vec{F}_{\rm pair}(r_{\rm extrap}) &, r \le r_{\rm extrap}
# \f}
# where \f$\vec{F}_{\rm pair}\f$ is given by the gradient of the pair force
# \f{eqnarray*}{
# \vec{F}_{\rm pair} =& -\nabla V_{\rm pair}(r) &, r < r_{\rm cut} \\
#                    =& 0 &, r \ge r_{\mathrm{cut}}
# \f}
# In other words, if \f$r_{\rm extrap}\f$ is chosen so that the pair force would point into the active half-space,
# the extrapolated potential will push all particles into the active half-space. See wall.lj for a specific example.
#
# To use extrapolated mode, the following coefficients must be set per unique particle types.
# - All parameters required by the %pair potential base for the wall potential
# - \f$r_{\mathrm{cut}} \f$ - \c r_cut (in distance units) -<i>Optional: Defaults to global r_cut for the force if given or 0.0 if not</i>
# - \f$ r_{\mathrm{extrap}} \f$ -\c r_extrap (in distance units) -<i>Optional: Defaults to 0.0</i>
#
#
# <b>Generic Example:</b>\n
# Note that the walls object below must be created before it is given as an
# argument to the force object. However, walls can be modified at any time before
# run() is called and it will update itself appropriately. See \link wall.group
# wall.group()\endlink for more details about specifying the walls to be used.
# \code
# walls=wall.group()
# # Edit walls
# my_force=wall.pairpotential(walls)
# my_force.force_coeff.set('A', all required arguments)
# my_force.force_coeff.set(['B','C'],r_cut=0.3, all required arguments)
# my_force.force_coeff.set(['B','C'],r_extrap=0.3, all required arguments)
# \endcode
# A specific example can be found in wall.lj
#
# \note \par
# The current wall force implementation does not support NPT integrators.
#
# \note \par
# The virial due to walls is computed, but the pressure and reported by analyze.log is not well defined.
# The volume (area) of the box enters into the pressure computation, which is not correct in a
# confined system. It may not even be possible to define an appopriate volume with soft walls.
#
# \note \par
# An effective use of wall forces <b>requires</b> considering the geometry of the
# system. Each wall is only evaluated in one simulation box and thus is not
# periodic. Forces will be evaluated and added to all particles from all walls in
# the wall \link wall.group group\endlink. Additionally there are no safeguards
# requireing a wall to exist inside the box to have interactions. This means that
# an attractive force existing outside the simulation box would pull particles
# across the periodic boundary where they would immediately cease to have any
# interaction with that wall. It is therefore up to the user to use walls in a
# physically meaningful manner. This includes the geometry of the walls, their
# interactions, and as noted here their location.
# \note \par
# If \f$ r_{\mathrm{cut}} \le 0 \f$ or is set to False the particle type
# %wall interaction is excluded.
# \note \par
# While wall potentials are based on the same potential energy calculations
# as pair potentials, Features of pair potentials such as specified neighborlists,
# and alternative force shifting modes are not supported.
class wallpotential(external._external_force):
    ## Creates a generic wall force
    # \param walls Wall group containing half-space geometries for the force to act in.
    # \param r_cut The global r_cut value for the force. Defaults to False or 0 if not specified.
    # \param name The force name which will be used in the metadata and log files.
    def __init__(self, walls, r_cut, name=""):
        external._external_force.__init__(self, name);
        self.field_coeff = walls;
        self.required_coeffs = ["r_cut", "r_extrap"];
        self.force_coeff.set_default_coeff('r_extrap', 0.0);

        # convert r_cut False to a floating point type
        if (r_cut==False):
            r_cut = 0.0
        self.global_r_cut = r_cut;
        self.force_coeff.set_default_coeff('r_cut', self.global_r_cut);

    ## \internal
    # \brief passes the wall field
    def process_field_coeff(self, coeff):
        return hoomd.make_wall_field_params(coeff, globals.exec_conf);

    ## \internal
    # \brief Return metadata for this wall potential
    def get_metadata(self):
        data=external._external_force.get_metadata(self);
        data['walls_struct'] = self.field_coeff.get_metadata();
        return data

    ## \internal
    # \brief Fixes negative values to zero before squaring
    def update_coeffs(self):
        ntypes = globals.system_definition.getParticleData().getNTypes();
        for i in range(0,ntypes):
            type=globals.system_definition.getParticleData().getNameByType(i);
            if self.force_coeff.values[str(type)]['r_cut']<=0:
                self.force_coeff.values[str(type)]['r_cut']=0;
        external._external_force.update_coeffs(self);

## Lennard-Jones %wall %force
# Wall force evaluated using the Lennard-Jones potential.
# See pair.lj for force details and base parameters and wall.wallpotential for
# generalized %wall %force implementation
#
# <b>Example: Standard Mode</b>
# \code
# walls=wall.group()
# #add walls
# lj=wall.lj(walls, r_cut=3.0)
# lj.force_coeff.set('A', sigma=1.0,epsilon=1.0)  #plotted below in red
# lj.force_coeff.set('B', sigma=1.0,epsilon=1.0, r_cut=2.0**(1.0/2.0))
# lj.force_coeff.set(['A','B'], epsilon=2.0, sigma=1.0, alpha=1.0, r_cut=3.0)
# \endcode
#
#
# <b>Example: Extrapolated Mode</b>
# \code
# walls=wall.group()
# #add walls
# lj_extrap=wall.lj(walls, r_cut=3.0)
# lj_extrap.force_coeff.set('A', sigma=1.0,epsilon=1.0, r_extrap=1.1) #plotted in blue below
# \endcode
#
#
#
# <b>V(r) Plotted</b>
# \image html wall_extrap.png
class lj(wallpotential):
    ## Creates a lj wall force using the inputs
    # See wall.wallpotential and pair.lj for more details.
    def __init__(self, walls, r_cut=False, name=""):
        util.print_status_line();

        # tell the base class how we operate

        # initialize the base class
        wallpotential.__init__(self, walls, r_cut, name);

        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = hoomd.WallsPotentialLJ(globals.system_definition, self.name);
            self.cpp_class = hoomd.WallsPotentialLJ;
        else:

            self.cpp_force = hoomd.WallsPotentialLJGPU(globals.system_definition, self.name);
            self.cpp_class = hoomd.WallsPotentialLJGPU;

        globals.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficent options
        self.required_coeffs += ['epsilon', 'sigma', 'alpha'];
        self.force_coeff.set_default_coeff('alpha', 1.0);

    def process_coeff(self, coeff):
        epsilon = coeff['epsilon'];
        sigma = coeff['sigma'];
        alpha = coeff['alpha'];

        lj1 = 4.0 * epsilon * math.pow(sigma, 12.0);
        lj2 = alpha * 4.0 * epsilon * math.pow(sigma, 6.0);
        return hoomd.make_wall_lj_params(hoomd.make_scalar2(lj1, lj2), coeff['r_cut']*coeff['r_cut'], coeff['r_extrap']);

## Gaussian %wall %force
# Wall force evaluated using the Gaussian potential.
# See pair.gauss for force details and base parameters and wall.wallpotential for
# generalized %wall %force implementation
#
# \b Example:
# \code
# walls=wall.group()
# # add walls to interact with
# wall_force_gauss=wall.gauss(walls, r_cut=3.0)
# wall_force_gauss.force_coeff.set('A', epsilon=1.0, sigma=1.0)
# wall_force_gauss.force_coeff.set('A', epsilon=2.0, sigma=1.0, r_cut=3.0)
# wall_force_gauss.force_coeff.set(['C', 'D'], epsilon=3.0, sigma=0.5)
# \endcode
#
class gauss(wallpotential):
    ## Creates a gauss wall force using the inputs
    # See wall.wallpotential and pair.gauss for more details.
    def __init__(self, walls, r_cut=False, name=""):
        util.print_status_line();

        # tell the base class how we operate

        # initialize the base class
        wallpotential.__init__(self, walls, r_cut, name);
        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = hoomd.WallsPotentialGauss(globals.system_definition, self.name);
            self.cpp_class = hoomd.WallsPotentialGauss;
        else:

            self.cpp_force = hoomd.WallsPotentialGaussGPU(globals.system_definition, self.name);
            self.cpp_class = hoomd.WallsPotentialGaussGPU;

        globals.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficent options
        self.required_coeffs += ['epsilon', 'sigma'];

    def process_coeff(self, coeff):
        epsilon = coeff['epsilon'];
        sigma = coeff['sigma'];
        return hoomd.make_wall_gauss_params(hoomd.make_scalar2(epsilon, sigma), coeff['r_cut']*coeff['r_cut'], coeff['r_extrap']);

## Shifted Lennard-Jones %wall %force
# Wall force evaluated using the Shifted Lennard-Jones potential.
# Note that because slj is dependent upon particle diameters the following
# correction is necessary to the force details in the pair.slj description. \n
# \f$ \Delta = d_i/2 - 1 \f$ where \f$ d_i \f$ is the diameter of particle \f$ i
# \f$. \n
# See pair.slj for force details and base parameters and wall.wallpotential for
# generalized %wall %force implementation
#
# \b Example:
# \code
# walls=wall.group()
# # add walls to interact with
# wall_force_slj=wall.slj(walls, r_cut=3.0)
# wall_force_slj.force_coeff.set('A', epsilon=1.0, sigma=1.0)
# wall_force_slj.force_coeff.set('A', epsilon=2.0, sigma=1.0, r_cut=3.0)
# wall_force_slj.force_coeff.set('B', epsilon=1.0, sigma=1.0, r_cut=2**(1.0/6.0))
# \endcode
#
class slj(wallpotential):
    ## Creates a slj wall force using the inputs
    # See wall.wallpotential and pair.slj for more details.
    def __init__(self, walls, r_cut=False, d_max=None, name=""):
        util.print_status_line();

        # tell the base class how we operate

        # initialize the base class
        wallpotential.__init__(self, walls, r_cut, name);

        # update the neighbor list
        if d_max is None :
            sysdef = globals.system_definition;
            d_max = sysdef.getParticleData().getMaxDiameter()
            globals.msg.notice(2, "Notice: slj set d_max=" + str(d_max) + "\n");

        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = hoomd.WallsPotentialSLJ(globals.system_definition, self.name);
            self.cpp_class = hoomd.WallsPotentialSLJ;
        else:

            self.cpp_force = hoomd.WallsPotentialSLJGPU(globals.system_definition, self.name);
            self.cpp_class = hoomd.WallsPotentialSLJGPU;

        globals.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficient options
        self.required_coeffs += ['epsilon', 'sigma', 'alpha'];
        self.force_coeff.set_default_coeff('alpha', 1.0);

    def process_coeff(self, coeff):
        epsilon = coeff['epsilon'];
        sigma = coeff['sigma'];
        alpha = coeff['alpha'];

        lj1 = 4.0 * epsilon * math.pow(sigma, 12.0);
        lj2 = alpha * 4.0 * epsilon * math.pow(sigma, 6.0);
        return hoomd.make_wall_slj_params(hoomd.make_scalar2(lj1, lj2), coeff['r_cut']*coeff['r_cut'], coeff['r_extrap']);

## Yukawa %wall %force
# Wall force evaluated using the Yukawa potential.
# See pair.yukawa for force details and base parameters and wall.wallpotential for
# generalized %wall %force implementation
#
# \b Example:
# \code
# walls=wall.group()
# # add walls to interact with
# wall_force_yukawa=wall.yukawa(walls, r_cut=3.0)
# wall_force_yukawa.force_coeff.set('A', epsilon=1.0, kappa=1.0)
# wall_force_yukawa.force_coeff.set('A', epsilon=2.0, kappa=0.5, r_cut=3.0)
# wall_force_yukawa.force_coeff.set(['C', 'D'], epsilon=0.5, kappa=3.0)
# \endcode
#
class yukawa(wallpotential):
    ## Creates a yukawa wall force using the inputs
    # See wall.wallpotential and pair.yukawa for more details.
    def __init__(self, walls, r_cut=False, name=""):
        util.print_status_line();

        # tell the base class how we operate

        # initialize the base class
        wallpotential.__init__(self, walls, r_cut, name);

        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = hoomd.WallsPotentialYukawa(globals.system_definition, self.name);
            self.cpp_class = hoomd.WallsPotentialYukawa;
        else:
            self.cpp_force = hoomd.WallsPotentialYukawaGPU(globals.system_definition, self.name);
            self.cpp_class = hoomd.WallsPotentialYukawaGPU;

        globals.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficent options
        self.required_coeffs += ['epsilon', 'kappa'];

    def process_coeff(self, coeff):
        epsilon = coeff['epsilon'];
        kappa = coeff['kappa'];
        return hoomd.make_wall_yukawa_params(hoomd.make_scalar2(epsilon, kappa), coeff['r_cut']*coeff['r_cut'], coeff['r_extrap']);

## Morse %wall %force
# Wall force evaluated using the Morse potential.
# See pair.morse for force details and base parameters and wall.wallpotential for
# generalized %wall %force implementation
#
# \b Example:
# \code
# walls=wall.group()
# # add walls to interact with
# wall_force_morse=wall.morse(walls, r_cut=3.0)
# wall_force_morse.force_coeff.set('A', D0=1.0, alpha=3.0, r0=1.0)
# wall_force_morse.force_coeff.set('A', D0=1.0, alpha=3.0, r0=1.0, r_cut=3.0)
# wall_force_morse.force_coeff.set(['C', 'D'], D0=1.0, alpha=3.0)
# \endcode
#
class morse(wallpotential):
    ## Creates a morse wall force using the inputs
    # See wall.wallpotential and pair.morse for more details.
    def __init__(self, walls, r_cut=False, name=""):
        util.print_status_line();

        # tell the base class how we operate

        # initialize the base class
        wallpotential.__init__(self, walls, r_cut, name);

        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = hoomd.WallsPotentialMorse(globals.system_definition, self.name);
            self.cpp_class = hoomd.WallsPotentialMorse;
        else:

            self.cpp_force = hoomd.WallsPotentialMorseGPU(globals.system_definition, self.name);
            self.cpp_class = hoomd.WallsPotentialMorseGPU;

        globals.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficent options
        self.required_coeffs += ['D0', 'alpha', 'r0'];

    def process_coeff(self, coeff):
        D0 = coeff['D0'];
        alpha = coeff['alpha'];
        r0 = coeff['r0']

        return hoomd.make_wall_morse_params(hoomd.make_scalar4(D0, alpha, r0, 0.0), coeff['r_cut']*coeff['r_cut'], coeff['r_extrap']);

## Force-shifted Lennard-Jones %wall %force
# Wall force evaluated using the Force-shifted Lennard-Jones potential.
# See pair.force_shifted_lj for force details and base parameters and wall.wallpotential for generalized %wall %force implementation
#
# \b Example:
# \code
# walls=wall.group()
# # add walls to interact with
# wall_force_fslj=wall.force_shifted_lj(walls, r_cut=3.0)
# wall_force_fslj.force_coeff.set('A', epsilon=1.0, sigma=1.0)
# wall_force_fslj.force_coeff.set('B', epsilon=1.5, sigma=3.0, r_cut = 8.0)
# wall_force_fslj.force_coeff.set(['C','D'], epsilon=1.0, sigma=1.0, alpha = 1.5)
# \endcode
#
class force_shifted_lj(wallpotential):
    ## Creates a force_shifted_lj wall force using the inputs
    # See wall.wallpotential and pair.force_shifted_lj for more details.
    def __init__(self, walls, r_cut=False, name=""):
        util.print_status_line();

        # tell the base class how we operate

        # initialize the base class
        wallpotential.__init__(self, walls, r_cut, name);

        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = hoomd.WallsPotentialForceShiftedLJ(globals.system_definition, self.name);
            self.cpp_class = hoomd.WallsPotentialForceShiftedLJ;
        else:

            self.cpp_force = hoomd.WallsPotentialForceShiftedLJGPU(globals.system_definition, self.name);
            self.cpp_class = hoomd.WallsPotentialForceShiftedLJGPU;

        globals.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficent options
        self.required_coeffs += ['epsilon', 'sigma', 'alpha'];
        self.force_coeff.set_default_coeff('alpha', 1.0);

    def process_coeff(self, coeff):
        epsilon = coeff['epsilon'];
        sigma = coeff['sigma'];
        alpha = coeff['alpha'];

        lj1 = 4.0 * epsilon * math.pow(sigma, 12.0);
        lj2 = alpha * 4.0 * epsilon * math.pow(sigma, 6.0);
        return hoomd.make_wall_force_shifted_lj_params(hoomd.make_scalar2(lj1, lj2), coeff['r_cut']*coeff['r_cut'], coeff['r_extrap']);

## Mie potential %wall %force
# Wall force evaluated using the Mie potential.
# See pair.mie for force details and base parameters and wall.wallpotential for
# generalized %wall %force implementation
#
# \b Example:
# \code
# walls=wall.group()
# # add walls to interact with
# wall_force_mie=wall.mie(walls, r_cut=3.0)
# wall_force_mie.force_coeff.set('A', epsilon=1.0, sigma=1.0, n=12, m=6)
# wall_force_mie.force_coeff.set('A', epsilon=2.0, sigma=1.0, n=14, m=7, r_cut=3.0)
# wall_force_mie.force_coeff.set('B', epsilon=1.0, sigma=1.0, n=15.1, m=6.5, r_cut=2**(1.0/6.0))
# \endcode
#
class mie(wallpotential):
    ## Creates a mie wall force using the inputs
    # See wall.wallpotential and pair.mie for more details.
    def __init__(self, walls, r_cut=False, name=""):
        util.print_status_line();

        # tell the base class how we operate

        # initialize the base class
        wallpotential.__init__(self, walls, r_cut, name);

        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = hoomd.WallsPotentialMie(globals.system_definition, self.name);
            self.cpp_class = hoomd.WallsPotentialMie;
        else:

            self.cpp_force = hoomd.WallsPotentialMieGPU(globals.system_definition, self.name);
            self.cpp_class = hoomd.WallsPotentialMieGPU;

        globals.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficent options
        self.required_coeffs += ['epsilon', 'sigma', 'n', 'm'];

    def process_coeff(self, coeff):
        epsilon = float(coeff['epsilon']);
        sigma = float(coeff['sigma']);
        n = float(coeff['n']);
        m = float(coeff['m']);

        mie1 = epsilon * math.pow(sigma, n) * (n/(n-m)) * math.pow(n/m,m/(n-m));
        mie2 = epsilon * math.pow(sigma, m) * (n/(n-m)) * math.pow(n/m,m/(n-m));
        mie3 = n
        mie4 = m
        return hoomd.make_wall_mie_params(hoomd.make_scalar4(mie1, mie2, mie3, mie4), coeff['r_cut']*coeff['r_cut'], coeff['r_extrap']);
