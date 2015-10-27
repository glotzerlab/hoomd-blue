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
# Walls currently supports these geometries: spheres, planes, and cylinders.
# Walls currently supports these potentials: lj, gauss, slj, yukawa, morse,
# force_shifted_lj, and mie.
#
# Walls can add forces to any particles within a certain distance of each wall.
#
# Walls are created using the group commands. See group for more details.
# By themselves, wall groups do nothing. Only when you specify a wall force (i.e.
# wall.lj),  are forces actually applied between the wall and the particle.
# See hoomd_script.wall.wallpotential for more details of implementing a force.

from hoomd_script import external;
from hoomd_script import globals;
from hoomd_script import force;
from hoomd_script import util;
from hoomd_script import meta;
import hoomd;
import math;

## Defines the %wall group.
#
# All wall forces use a wall group as an input so it is necessary to create a
# wall.group object before any wall.force can be created. Current supported
# geometries are spheres, cylinder, and planes. The maximum number of each type of
# wall is 20, 20, and 60 respectively.
#
# \note \par
# The entire structure can easily be viewed by printing the wall.group object.
#
# \note \par
# While all x,y,z coordinates can be given as a list or tuple, only origin
# parameters are points  in x,y,z space. Normal and axis parameters are vectors
# and must have a magnitude.
#
# \note \par
# Although members of the structure can be modified directly,  using the
# convenience functions (i.e. add_sphere, del_sphere) will keep track of the total
# number of each type implemented and give warnings if the maximum number of any
# type is reached. If the structure is modified outside the demonstrated scope,
# the group object update function should be called.
#
# \note \par
# Wall structure modifications between run() calls will be implemented in
# the next run. However, modifications must be done carefully since moving the
# wall can result in particles moving to a relative positions which causes
# exceptionally high forces resulting in particles moving many times the box length
# in one move.
#
# \b Example:\n
# In[0]:
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
    # Max number of each wall geometry must match c++ definitions
    _max_n_sphere_walls=20;
    _max_n_cylinder_walls=20;
    _max_n_plane_walls=60;

    ## Creates the wall group which can be named to easily find in the metadata.
    # Required to call and create an object before walls can be added to that
    # object.
    # \param name Name of the wall structure (string, defaults to empty string).
    #
    # \b Example:
    # \code
    # wall_object=wall.group()
    # named_wall_object=wall.group(name='Arbitrary Wall Name')
    # \endcode
    def __init__(self,name=''):
        self.name=name;
        self.num_spheres=0;
        self.num_cylinders=0;
        self.num_planes=0;
        self.spheres=[];
        self.cylinders=[];
        self.planes=[];

    ## Adds a sphere to the %wall group.
    #
    # \param r Sphere radius (in distance units)
    # \param origin Sphere origin (in x,y,z coordinates)
    # \param inside Sphere distance evaluation from inside/outside surface (defaults to True)
    def add_sphere(self, r, origin, inside=True):
        if self.num_spheres<group._max_n_sphere_walls:
            self.spheres.append(sphere_wall(r,origin,inside));
            self.num_spheres+=1;
        else:
            globals.msg.error("Trying to specify more than the maximum allowable number of sphere walls.\n");
            raise RuntimeError('Maximum number of sphere walls already used.');

    ## Adds a cylinder to the %wall group.
    #
    # \param r Cylinder radius (in distance units)
    # \param origin Cylinder origin (in x,y,z coordinates)
    # \param axis Cylinder axis vector (in x,y,z coordinates)
    # \param inside Cylinder distance evaluation from inside/outside surface (defaults to True)
    def add_cylinder(self, r, origin, axis, inside=True):
        if self.num_cylinders<group._max_n_cylinder_walls:
            self.cylinders.append(cylinder_wall(r, origin, axis, inside));
            self.num_cylinders+=1;
        else:
            globals.msg.error("Trying to specify more than the maximum allowable number of cylinder walls.\n");
            raise RuntimeError('Maximum number of cylinder walls already used.');

    ## Adds a plane to the %wall group.
    #
    # \param origin Plane origin (in x,y,z coordinates)
    # \param normal Plane normal vector (in x,y,z coordinates)
    def add_plane(self, origin, normal):
        if self.num_planes<group._max_n_plane_walls:
            self.planes.append(plane_wall(origin, normal));
            self.num_planes+=1;
        else:
            globals.msg.error("Trying to specify more than the maximum allowable number of plane walls.\n");
            raise RuntimeError('Maximum number of plane walls already used.');

    ## Deletes the sphere or spheres in index.
    # \param index The index of sphere(s) desired to delete. Accepts int, range, and lists.
    def del_sphere(self, index):
        if type(index) is int: index = [index];
        elif type(index) is range: index = list(index);
        index=list(set(index));
        index.sort(reverse=True);
        for i in index:
            try:
                del(self.spheres[i]);
                self.num_spheres-=1;
            except IndexValueError:
                globals.msg.error("Specified index for deletion is not valid.\n");
                raise RuntimeError("del_sphere failed")

    ## Deletes the cylinder or cylinders in index.
    # \param index The index of cylinder(s) desired to delete. Accepts int, range, and lists.
    def del_cylinder(self, index):
        if type(index) is int: index = [index];
        elif type(index) is range: index = list(index);
        index=list(set(index));
        index.sort(reverse=True);
        for i in index:
            try:
                del(self.cylinders[i]);
                self.num_cylinders-=1;
            except IndexValueError:
                globals.msg.error("Specified index for deletion is not valid.\n");
                raise RuntimeError("del_cylinder failed")

    ## Deletes the plane or planes in index.
    # \param index The index of plane(s) desired to delete. Accepts int, range, and lists.
    def del_plane(self, index):
        if type(index) is int: index = [index];
        elif type(index) is range: index = list(index);
        index=list(set(index));
        index.sort(reverse=True);
        for i in index:
            try:
                del(self.planes[i]);
                self.num_planes-=1;
            except IndexValueError:
                globals.msg.error("Specified index for deletion is not valid.\n");
                raise RuntimeError("del_plane failed")

    ## \internal
    # \brief Return metadata for this wall structure
    def get_metadata(self):
        data = meta._metadata_from_dict(eval(str(self.__dict__)));
        return data;

    ## Updates counting variables of the wall.group object and checks for validity of input
    def update(self):
        self.num_spheres=len(self.spheres);
        self.num_cylinders=len(self.cylinders);
        self.num_planes=len(self.planes);
        if self.num_spheres>group._max_n_sphere_walls:
            globals.msg.error("Trying to specify more than the maximum allowable number of sphere walls.\n");
            raise RuntimeError('Maximum number of sphere walls already used.');
        if self.num_cylinders>group._max_n_cylinder_walls:
            globals.msg.error("Trying to specify more than the maximum allowable number of cylinder walls.\n");
            raise RuntimeError('Maximum number of cylinder walls already used.');
        if self.num_planes>group._max_n_plane_walls:
            globals.msg.error("Trying to specify more than the maximum allowable number of plane walls.\n");
            raise RuntimeError('Maximum number of plane walls already used.');

    ## \internal
    # \brief Returns output for print
    def __str__(self):
        output="Wall_Data_Sturucture:%s\nspheres:%s{"%(self.name, self.num_spheres);
        for index in range(self.num_spheres):
            output+="\n[%s:\t%s]"%(repr(index), str(self.spheres[index]));

        output+="}\ncylinders:%s{"%(self.num_cylinders);
        for index in range(self.num_cylinders):
            output+="\n[%s:\t%s]"%(repr(index), str(self.cylinders[index]));

        output+="}\nplanes:%s{"%(self.num_planes);
        for index in range(self.num_planes):
            output+="\n[%s:\t%s]"%(repr(index), str(self.planes[index]));

        output+="}";
        return output;

## Class that populates the spheres[] array in the wall.group object.
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

## Class that populates the cylinders[] array in the wall.group object.
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

## Class that populates the planes[] array in the wall.group object.
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


## Generic %wall %force
#
# wall.wallpotential is not a command hoomd scripts should execute directly.
# Rather, it is a base command that provides common features to all standard
# %wall forces. Rather than repeating all of that documentation in many
# different places, it is collected here.
#
# All %wall %force commands specify that a given potential energy and %force be
# computed on all particles in the system within a cutoff distance \f$
# r_{\mathrm{cut}} \f$ from each wall in the given wall group
#
# The %force \f$ \vec{F}\f$ is \f{eqnarray*} \vec{F}  = & -\nabla V(r) & r <
# r_{\mathrm{cut}} \\ = & 0           & r \ge r_{\mathrm{cut}} \\ \f} where \f$
# \vec{r} \f$ is the vector pointing from the particle to the %wall, and \f$ V(r)
# \f$ is evaluated in the same manner as when mode is shift for the analogous pair
# potentials. \f{eqnarray*} V(r)  = & V_{\mathrm{pair}}(r) -
# V_{\mathrm{pair}}(r_{\mathrm{cut}}) \\ \f} and \f$ V_{\mathrm{pair}}(r) \f$ is
# the specific %pair potential chosen by the respective command.
#
# The following coefficients must be set per unique particle types.
# - \f$ r_{\mathrm{cut}} \f$ - \c r_cut (in distance units)
# - \f$ r_{\mathrm{on}} \f$ - \c r_on (in distance units)
# - All parameters required by the %pair potential base for the wall potential
#
#
# \b Example:
# \code
#
# \endcode
#
# \note \par
# If \f$ r_{\mathrm{cut}} \le 0 \f$ or is set to False the particle type
# %wall interaction is excluded.
# \note \par
# Wall potentials are only based on the same potential energy calculations
# as pair potentials. Features of pair potentials such as global r_cut,
# specified neighborlists, and alternative force shifting modes are not
# supported.
class wallpotential(external._external_force):
    def __init__(self, walls, name=""):
        external._external_force.__init__(self, name);
        self.field_coeff = walls;
        self.required_coeffs = ["r_cut", "r_on"];

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
            if self.force_coeff.values[type]['r_on']<=0:
                self.force_coeff.values[type]['r_on']=0;
        external._external_force.update_coeffs(self);

## Lennard-Jones %wall %force
# See pair.lj for force details and base parameters and wall.wallpotential for generalized %wall %force implementation
class lj(wallpotential):
    def __init__(self, walls, name=""):
        util.print_status_line();

        # tell the base class how we operate

        # initialize the base class
        wallpotential.__init__(self, walls, name);

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
        return hoomd.make_walls_lj_params(hoomd.make_scalar2(lj1, lj2), coeff['r_cut']*coeff['r_cut'], coeff['r_on']*coeff['r_on']);

## Gaussian %wall %force
# See pair.gauss for force details and base parameters and wall.wallpotential for generalized %wall %force implementation
class gauss(wallpotential):
    def __init__(self, walls, name=""):
        util.print_status_line();

        # tell the base class how we operate

        # initialize the base class
        wallpotential.__init__(self, walls, name);
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
        return hoomd.make_walls_gauss_params(hoomd.make_scalar2(epsilon, sigma), coeff['r_cut']*coeff['r_cut'], coeff['r_on']*coeff['r_on']);

## Shifted Lennard-Jones %wall %force
# See pair.slj for force details and base parameters and wall.wallpotential for generalized %wall %force implementation
class slj(wallpotential):
    def __init__(self, walls, d_max=None, name=""):
        util.print_status_line();

        # tell the base class how we operate

        # initialize the base class
        wallpotential.__init__(self, walls, name);

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
        self.pair_coeff.set_default_coeff('alpha', 1.0);

    def process_coeff(self, coeff):
        epsilon = coeff['epsilon'];
        sigma = coeff['sigma'];
        alpha = coeff['alpha'];

        lj1 = 4.0 * epsilon * math.pow(sigma, 12.0);
        lj2 = alpha * 4.0 * epsilon * math.pow(sigma, 6.0);
        return hoomd.make_walls_slj_params(hoomd.make_scalar2(lj1, lj2), coeff['r_cut']*coeff['r_cut'], coeff['r_on']*coeff['r_on']);

## Yukawa %wall %force
# See pair.yukawa for force details and base parameters and wall.wallpotential for generalized %wall %force implementation
class yukawa(wallpotential):
    def __init__(self, walls, name=""):
        util.print_status_line();

        # tell the base class how we operate

        # initialize the base class
        wallpotential.__init__(self, walls, name);

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
        return hoomd.make_walls_yukawa_params(hoomd.make_scalar2(epsilon, kappa), coeff['r_cut']*coeff['r_cut'], coeff['r_on']*coeff['r_on']);

## Morse %wall %force
# See pair.morse for force details and base parameters and wall.wallpotential for generalized %wall %force implementation
class morse(wallpotential):
    def __init__(self, walls, name=""):
        util.print_status_line();

        # tell the base class how we operate

        # initialize the base class
        wallpotential.__init__(self, walls, name);

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

        return hoomd.make_walls_morse_params(hoomd.make_scalar4(D0, alpha, r0, 0.0), coeff['r_cut']*coeff['r_cut'], coeff['r_on']*coeff['r_on']);

## Force-shifted Lennard-Jones %wall %force
# See pair.force_shifted_lj for force details and base parameters and wall.wallpotential for generalized %wall %force implementation
class force_shifted_lj(wallpotential):
    def __init__(self, walls, name=""):
        util.print_status_line();

        # tell the base class how we operate

        # initialize the base class
        wallpotential.__init__(self, walls, name);

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
        self.pair_coeff.set_default_coeff('alpha', 1.0);

    def process_coeff(self, coeff):
        epsilon = coeff['epsilon'];
        sigma = coeff['sigma'];
        alpha = coeff['alpha'];

        lj1 = 4.0 * epsilon * math.pow(sigma, 12.0);
        lj2 = alpha * 4.0 * epsilon * math.pow(sigma, 6.0);
        return hoomd.make_walls_force_shifted_lj_params(hoomd.make_scalar2(lj1, lj2), coeff['r_cut']*coeff['r_cut'], coeff['r_on']*coeff['r_on']);

## Mie potential %wall %force
# See pair.mie for force details and base parameters and wall.wallpotential for generalized %wall %force implementation
class mie(wallpotential):
    def __init__(self, walls, name=""):
        util.print_status_line();

        # tell the base class how we operate

        # initialize the base class
        wallpotential.__init__(self, walls, name);

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
        return hoomd.make_walls_mie_params(hoomd.make_scalar4(mie1, mie2, mie3, mie4), coeff['r_cut']*coeff['r_cut'], coeff['r_on']*coeff['r_on']);
