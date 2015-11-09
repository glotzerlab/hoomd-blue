# -- start license --
# Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
# (HOOMD-blue) Open Source Software License Copyright 2009-2015 The Regents of
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
# http:/\codeblue.umich.edu/hoomd-blue/citations.html

# * Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
# http:/\codeblue.umich.edu/hoomd-blue/

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
#  Walls currently supports these geometries:\n \link wall.sphere_wall
# spheres\endlink, \link wall.cylinder_wall cylinders\endlink, \link
# wall.plane_wall planes\endlink \n Walls currently supports these
# potentials:\n \link wall.lj lj\endlink, \link wall.gauss gauss\endlink, \link
# wall.slj slj\endlink, \link wall.yukawa yukawa\endlink, \link wall.morse
# morse\endlink, \link wall.force_shifted_lj force_shifted_lj\endlink, and \link
# wall.mie mie\endlink.
#
# Walls can add forces to any particles within a certain distance of each wall.
#
# Walls are created using the \link wall.group group\endlink commands. See \link
# wall.group group\endlink for more details. By themselves, wall groups do
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
# An effective use of wall forces <b>requires</b> considering the geometry of the
# system. Walls are only evaluated in one simulation box and are not periodic. It
# is therefore important to ensure that the walls that intersect with a periodic
# boundary meet.
#
# \note \par
# The entire structure can easily be viewed by printing the wall.group object.
# Print intentionally only displays exactly what will be passed to the wall force.
# This can be seen in the example in \link hoomd_script.wall.sphere_wall
# sphere_wall\endlink.
# \note \par
# While all x,y,z coordinates can be given as a list or tuple, only origin
# parameters are points in x,y,z space. Normal and axis parameters are vectors and
# must have a nonzero magnitude. The examples in \link
# hoomd_script.wall.sphere_wall sphere_wall\endlink demonstrates this in the
# default parameters.
# \note \par
# Wall structure modifications between \link hoomd_script.run() run()\endlink
# calls will be implemented in the next run. However, modifications must be done
# carefully since moving the wall can result in particles moving to a relative
# positions which causes exceptionally high forces resulting in particles moving
# many times the box length in one move.
#
# \b Example:
# \n In[0]:
# \code
# wallstructure=wall.group(name="arbitrary name")
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
# Wall_Data_Sturucture:arbitrary name
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
# wallstructure.del_plane(range(3))
# wallstructure.del_cylinder([0,2])
# wallstructure.del_sphere(1)
# print(wallstructure)
# \endcode
# Out[1]:
# \code
# Wall_Data_Sturucture:arbitrary name
# spheres:1{
# [0:	Radius=1.0	Origin=(0.0, 1.0, 3.0)	Inside=True]}
# cylinders:1{
# [0:	Radius=4.0	Origin=(0.0, 0.0, 0.0)	Axis=(1.0, 0.0, 1.0)	Inside=True]}
# planes:0{}
# \endcode
# In[2]:
# \code
# wallstructure.spheres[0].r=2.0
# wallstructure.cylinders[0].origin=[1,2,1]
# wallstructure.cylinders[0].axis=(0,0,1)
# print(wallstructure)
# \endcode
# Out[2]:
# \code
# Wall_Data_Sturucture:arbitrary name
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
    # named_wall_object=wall.group(name='Arbitrary Wall Name')
    # full_wall_object=wall.group([wall.sphere_wall()]*20,[wall.cylinder_wall()]*20,[wall.plane_wall()]*60)
    # \endcode
    def __init__(self,*walls,name=''):
        self.name=name;
        self.spheres=[];
        self.cylinders=[];
        self.planes=[];
        for wall in walls:
            self.add(wall);

    ## Generic wall add for wall objects.
    # Generic convenience function to add any wall object to the group.
    # Accepts \link hoomd_script.wall.sphere_wall sphere_wall\endlink, \link
    # hoomd_script.wall.cylinder_wall cylinder_wall\endlink, \link
    # hoomd_script.wall.plane_wall plane_wall\endlink, and lists of any
    # combination of these.
    def add(self,wall):
        if (type(wall)==type(sphere_wall())):
            self.spheres.append(wall);
        elif (type(wall)==type(cylinder_wall())):
            self.cylinders.append(wall);
        elif (type(wall)==type(plane_wall())):
            self.planes.append(wall);
        elif (type(wall)==list):
            for wall_el in wall:
                if (type(wall_el)==type(sphere_wall())):
                    self.spheres.append(wall_el);
                elif (type(wall_el)==type(cylinder_wall())):
                    self.cylinders.append(wall_el);
                elif (type(wall_el)==type(plane_wall())):
                    self.planes.append(wall_el);
        else:
            print("Input of type "+str(type(wall))+" is not allowed.");

    ## Adds a sphere to the %wall group.
    # Adds a sphere with the specified parameters to the wallgroup.spheres list.
    # \param r Sphere radius (in distance units)
    # \param origin Sphere origin (in x,y,z coordinates)
    # \param inside Sphere distance evaluation from inside/outside surface (defaults to True)
    def add_sphere(self, r, origin, inside=True):
        self.spheres.append(sphere_wall(r,origin,inside));

    ## Adds a cylinder to the %wall group.
    # Adds a cylinder with the specified parameters to the wallgroup.cylinders list.
    # \param r Cylinder radius (in distance units)
    # \param origin Cylinder origin (in x,y,z coordinates)
    # \param axis Cylinder axis vector (in x,y,z coordinates)
    # \param inside Cylinder distance evaluation from inside/outside surface (defaults to True)
    def add_cylinder(self, r, origin, axis, inside=True):
        self.cylinders.append(cylinder_wall(r, origin, axis, inside));

    ## Adds a plane to the %wall group.
    # Adds a plane with the specified parameters to the wallgroup.planes list.
    # \param origin Plane origin (in x,y,z coordinates)
    # \param normal Plane normal vector (in x,y,z coordinates)
    def add_plane(self, origin, normal):
        self.planes.append(plane_wall(origin, normal));

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
        output="Wall_Data_Sturucture:%s\nspheres:%s{"%(self.name, len(self.spheres));
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
# Object which contains all the geometric information needed to define a sphere.
# This function is not recommended for common use and should mainly be used for
# reference. Helper functions \link wall.group.add_sphere add_sphere\endlink and
# \link wall.group.del_sphere del_sphere\endlink exist to properly update the
# entire wall structure. If these functions are used as in the example below, it
# is important to note that update must be called.
#
# This function is mainly available for utility of users in use cases like the one below
# and for reference in modifying the wall group structure's spheres members.
#
# The following example is intended to demonstrate cylinder_walls and plane_walls
# as well. Note that the distinction between points and vectors is reflected in
# the default parameters.
# \n \b Example
# \n In[0]:\code # walls=wall.group()
# walls.spheres+=[wall.sphere_wall()]*5
# walls.spheres+=[wall.sphere_wall(r=1)]*5
# walls.cylinders+=[wall.cylinder_wall(r=3)]*5
# walls.planes+=[wall.plane_wall()]*5
# print(walls)
# \endcode
# Out[0]:\code
# Wall_Data_Sturucture:
# spheres:10{
# [0:	Radius=0.0	Origin=(0.0, 0.0, 0.0)	Inside=True]
# [1:	Radius=0.0	Origin=(0.0, 0.0, 0.0)	Inside=True]
# [2:	Radius=0.0	Origin=(0.0, 0.0, 0.0)	Inside=True]
# [3:	Radius=0.0	Origin=(0.0, 0.0, 0.0)	Inside=True]
# [4:	Radius=0.0	Origin=(0.0, 0.0, 0.0)	Inside=True]
# [5:	Radius=1	Origin=(0.0, 0.0, 0.0)	Inside=True]
# [6:	Radius=1	Origin=(0.0, 0.0, 0.0)	Inside=True]
# [7:	Radius=1	Origin=(0.0, 0.0, 0.0)	Inside=True]
# [8:	Radius=1	Origin=(0.0, 0.0, 0.0)	Inside=True]
# [9:	Radius=1	Origin=(0.0, 0.0, 0.0)	Inside=True]}
# cylinders:5{
# [0:	Radius=3	Origin=(0.0, 0.0, 0.0)	Axis=(0.0, 0.0, 1.0)	Inside=True]
# [1:	Radius=3	Origin=(0.0, 0.0, 0.0)	Axis=(0.0, 0.0, 1.0)	Inside=True]
# [2:	Radius=3	Origin=(0.0, 0.0, 0.0)	Axis=(0.0, 0.0, 1.0)	Inside=True]
# [3:	Radius=3	Origin=(0.0, 0.0, 0.0)	Axis=(0.0, 0.0, 1.0)	Inside=True]
# [4:	Radius=3	Origin=(0.0, 0.0, 0.0)	Axis=(0.0, 0.0, 1.0)	Inside=True]}
# planes:5{
# [0:	Origin=(0.0, 0.0, 0.0)	Normal=(0.0, 0.0, 1.0)]
# [1:	Origin=(0.0, 0.0, 0.0)	Normal=(0.0, 0.0, 1.0)]
# [2:	Origin=(0.0, 0.0, 0.0)	Normal=(0.0, 0.0, 1.0)]
# [3:	Origin=(0.0, 0.0, 0.0)	Normal=(0.0, 0.0, 1.0)]
# [4:	Origin=(0.0, 0.0, 0.0)	Normal=(0.0, 0.0, 1.0)]}
# \endcode
class sphere_wall:
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
# Object which contains all the geometric information needed to define a cylinder.
# This function is not recommended for common use and should mainly be used for
# reference. Helper functions \link wall.group.add_cylinder add_cylinder\endlink and
# \link wall.group.del_cylinder del_cylinder\endlink exist to properly update the
# entire wall structure. If these functions are used as in the example below, it
# is important to note that update must be called.
#
# This function is mainly available for utility of users in use cases like the example
# and for reference in modifying the wall group structure's cylinders members.\n
#
# For an example see \link hoomd_script.wall.sphere_wall sphere_wall\endlink.
class cylinder_wall:
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
# Object which contains all the geometric information needed to define a plane.
# This function is not recommended for common use and should mainly be used for
# reference. Helper functions \link wall.group.add_plane add_plane\endlink and
# \link wall.group.del_plane del_plane\endlink exist to properly update the
# entire wall structure. If these functions are used as in the example below, it
# is important to note that update must be called.
#
# This function is mainly available for utility of users in use cases like the example
# and for reference in modifying the wall group structure's planes members.
#
# For an example see \link hoomd_script.wall.sphere_wall sphere_wall\endlink.
class plane_wall:
    def __init__(self, origin=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)):
        self._origin = hoomd.make_scalar3(*origin);
        self._normal = hoomd.make_scalar3(*normal);

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
        return "Origin=%s\tNormal=%s" % (str(self.origin), str(self.normal));

    def __repr__(self):
        return "{'origin':%s, 'normal': %s}" % (str(self.origin), str(self.normal));

#           *** Potentials ***

## Generic %wall %force
#
# wall.wallpotential is not a command hoomd scripts should execute directly.
# Rather, it is a base command that provides common features to all standard %wall
# forces. Rather than repeating all of that documentation in many different
# places, it is collected here.
#
# All %wall %force commands specify that a given potential energy and %force be
# computed on all particles in the system within a cutoff distance \f$
# r_{\mathrm{cut}} \f$ from each wall in the given wall \link wall.group
# group\endlink. The %force \f$ \vec{F}\f$ is where \f$ \vec{r} \f$ is the vector
# pointing from the particle to the %wall and \f$ V_{\mathrm{pair}}(r) \f$ is the
# specific %pair potential chosen by the respective command. \f{eqnarray*} \vec{F}  = &
# -\nabla V(r) & r_{\mathrm{min}} \le r < r_{\mathrm{cut}} \\ = & 0           & r
# \ge r_{\mathrm{cut}} \\ = & 0             & r < r_{\mathrm{min}} \f}   \f$ V(r)
# \f$ is evaluated in the same manner as when mode is shift for the analogous pair
# potentials. \f{eqnarray*} V(r)  = & V_{\mathrm{pair}}(r) -
# V_{\mathrm{pair}}(r_{\mathrm{cut}}) \\ \f}
#
# The following coefficients must be set per unique particle types.
# - \f$ r_{\mathrm{cut}} \f$ - \c r_cut (in distance units)
# - \f$ r_{\mathrm{min}} \f$ - \c r_min (in distance units)
#     -<i>Optional: Defaults to 0.0</i>
# - All parameters required by the %pair potential base for the wall potential
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
# my_force.force_coeff.set(['B','C'],r_min=0.3, all required arguments)
# \endcode
# A specific example can be found in wall.lj
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
# interactions, as noted here their location.
#
# There is no global r_cut intentionally since the r_cut value
# determines the shift of the potential and the parameter should be thought about.
# \note \par
# If \f$ r_{\mathrm{cut}} \le 0 \f$ or is set to False the particle type
# %wall interaction is excluded.
# \note \par
# While wall potentials are based on the same potential energy calculations
# as pair potentials, Features of pair potentials such as global r_cut,
# specified neighborlists, and alternative force shifting modes are not
# supported.
class wallpotential(external._external_force):
    def __init__(self, walls, r_cut, name=""):
        external._external_force.__init__(self, name);
        self.field_coeff = walls;
        self.required_coeffs = ["r_cut", "r_min"];
        self.force_coeff.set_default_coeff('r_min', 0.0);

        # convert r_cut False to a floating point type
        if r_cut is False:
            r_cut = -1.0
        self.global_r_cut = r_cut;
        self.force_coeff.set_default_coeff('r_cut', self.global_r_cut);

    def process_field_coeff(self, coeff):
        return hoomd.make_wall_field_params(coeff);

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
            if self.force_coeff.values[type]['r_cut']<=0:
                self.force_coeff.values[type]['r_cut']=0;
            if self.force_coeff.values[type]['r_min']<=0:
                self.force_coeff.values[type]['r_min']=0;
        external._external_force.update_coeffs(self);

## Lennard-Jones %wall %force
# Wall force evaluated using the Lennard-Jones potential.
# See pair.lj for force details and base parameters and wall.wallpotential for
# generalized %wall %force implementation
#
# \b Example\n
# Note that the base pair.lj requires the parameters <c>sigma</c> and <c>epsilon</c> and has a
# default <c>alpha</c>. The additional <c>r_cut</c> parameter is required per type by all wall
# potentials and all wall potentials have a default <c>r_min</c> of 0.0.
# \code
# walls=wall.group()
# lj=wall.lj(walls)
# lj.force_coeff.set('A',r_cut=3.0,r_min=0.0,sigma=1.0,epsilon=1.0)
# lj.force_coeff.set(['B','C'],r_cut=1.5,sigma=0.5,epsilon=1.0)
# \endcode
# \note \par
class lj(wallpotential):
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
        return hoomd.make_walls_lj_params(hoomd.make_scalar2(lj1, lj2), coeff['r_cut']*coeff['r_cut'], coeff['r_min']*coeff['r_min']);

## Gaussian %wall %force
# Wall force evaluated using the Gaussian potential.
# See pair.gauss for force details and base parameters and wall.wallpotential for
# generalized %wall %force implementation
class gauss(wallpotential):
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
        return hoomd.make_walls_gauss_params(hoomd.make_scalar2(epsilon, sigma), coeff['r_cut']*coeff['r_cut'], coeff['r_min']*coeff['r_min']);

## Shifted Lennard-Jones %wall %force
# Wall force evaluated using the Shifted Lennard-Jones potential.
# Note that because slj is dependent upon particle diameters the following
# correction is necessary to the force details in the pair.slj description. \n In
# wall.slj \f$ \Delta = d_i/2 - 1 \f$ where \f$ d_i \f$ is the diameter of
# particle \f$ i \f$. \n
# See pair.slj for force details and base parameters and wall.wallpotential for
# generalized %wall %force implementation
class slj(wallpotential):
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
        return hoomd.make_walls_slj_params(hoomd.make_scalar2(lj1, lj2), coeff['r_cut']*coeff['r_cut'], coeff['r_min']*coeff['r_min']);

## Yukawa %wall %force
# Wall force evaluated using the Yukawa potential.
# See pair.yukawa for force details and base parameters and wall.wallpotential for
# generalized %wall %force implementation
class yukawa(wallpotential):
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
        return hoomd.make_walls_yukawa_params(hoomd.make_scalar2(epsilon, kappa), coeff['r_cut']*coeff['r_cut'], coeff['r_min']*coeff['r_min']);

## Morse %wall %force
# Wall force evaluated using the Morse potential.
# See pair.morse for force details and base parameters and wall.wallpotential for
# generalized %wall %force implementation
class morse(wallpotential):
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

        return hoomd.make_walls_morse_params(hoomd.make_scalar4(D0, alpha, r0, 0.0), coeff['r_cut']*coeff['r_cut'], coeff['r_min']*coeff['r_min']);

## Force-shifted Lennard-Jones %wall %force
# Wall force evaluated using the Force-shifted Lennard-Jones potential.
# See pair.force_shifted_lj for force details and base parameters and wall.wallpotential for generalized %wall %force implementation
class force_shifted_lj(wallpotential):
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
        return hoomd.make_walls_force_shifted_lj_params(hoomd.make_scalar2(lj1, lj2), coeff['r_cut']*coeff['r_cut'], coeff['r_min']*coeff['r_min']);

## Mie potential %wall %force
# Wall force evaluated using the Mie potential.
# See pair.mie for force details and base parameters and wall.wallpotential for
# generalized %wall %force implementation
class mie(wallpotential):
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
        return hoomd.make_walls_mie_params(hoomd.make_scalar4(mie1, mie2, mie3, mie4), coeff['r_cut']*coeff['r_cut'], coeff['r_min']*coeff['r_min']);
                                                                                                                                                                                                                                                                                                             
