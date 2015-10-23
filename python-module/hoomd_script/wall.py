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

## \package hoomd_script.external
# \brief Commands that create external forces on particles

from hoomd_script import external;
from hoomd_script import globals;
from hoomd_script import force;
from hoomd_script import util;
from hoomd_script import meta;
import hoomd;

class group():
    # Max number of each wall geometry must match c++ defintions
    _max_n_sphere_walls=20;
    _max_n_cylinder_walls=20;
    _max_n_plane_walls=60;
    def __init__(self,name=''):
        self.name=name;
        self.num_spheres=0;
        self.num_cylinders=0;
        self.num_planes=0;
        self.spheres=[];
        self.cylinders=[];
        self.planes=[];

    def add_sphere(self, r, origin, inside=True):
        if self.num_spheres<group._max_n_sphere_walls:
            self.spheres.append(sphere_wall(r,origin,inside));
            self.num_spheres+=1;
        else:
            globals.msg.error("Trying to specify more than the maximum allowable number of sphere walls.\n");
            raise RuntimeError('Maximum number of sphere walls already used.');
    def add_cylinder(self, r, origin, axis, inside=True):
        if self.num_cylinders<group._max_n_cylinder_walls:
            self.cylinders.append(cylinder_wall(r, origin, axis, inside));
            self.num_cylinders+=1;
        else:
            globals.msg.error("Trying to specify more than the maximum allowable number of cylinder walls.\n");
            raise RuntimeError('Maximum number of cylinder walls already used.');
    def add_plane(self, origin, normal):
        if self.num_planes<group._max_n_plane_walls:
            self.planes.append(plane_wall(origin, normal));
            self.num_planes+=1;
        else:
            globals.msg.error("Trying to specify more than the maximum allowable number of plane walls.\n");
            raise RuntimeError('Maximum number of plane walls already used.');

    def del_sphere(self, index):
        if type(index) is list: index = set(index);
        elif type(index) is range: index = set(list(index));
        elif type(index) is not set: index = set([index]);
        if (self.num_spheres-index)>0:
            del(self.spheres[index]);
            self.num_spheres-=1;
        else:
            globals.msg.error("Specified index for deletion is not available.\n");
            raise RuntimeError("del_sphere failed")
    def del_cylinder(self, index):
        if type(index) is list: index = set(index);
        elif type(index) is range: index = set(list(index));
        elif type(index) is not set: index = set([index]);
        if (self.num_cylinders-index)>0:
            del(self.cylinders[index]);
            self.num_cylinders-=1;
        else:
            globals.msg.error("Specified index for deletion is not valid.\n");
            raise RuntimeError("del_cylinder failed")
    def del_plane(self, index):
        if type(index) is list: index = set(index);
        elif type(index) is range: index = set(list(index));
        elif type(index) is not set: index = set([index]);
        if (self.num_planes-index)>0:
            del(self.planes[index]);
            self.num_planes-=1;
        else:
            globals.msg.error("Specified index for deletion is not valid.\n");
            raise RuntimeError("del_plane failed")

    def get_metadata(self):
        data = meta._metadata_from_dict(eval(str(self._dict__)));
        return data;

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

class wallpotential(external._external_force):
    def __init__(self, walls, name=""):
        external._external_force.__init__(self, name);
        self.field_coeff = walls;
        self.required_coeffs = ["r_cut", "r_on"];

    def process_field_coeff(self, coeff):
        return hoomd.make_wall_field_params(coeff);

    def get_metadata(self):
        data=external._external_force.get_metadata(self);
        data['walls'] = self.field_coeff.get_metadata();
        return data



# class ewald(wallpotential):


# class moliere(wallpotential):


# class zbl(wallpotential):


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


class slj(wallpotential):
    def __init__(self, walls, r_cut, d_max=None, name=""):
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

    def set_params(self, mode=None):
        util.print_status_line();

        if mode == "xplor":
            globals.msg.error("XPLOR is smoothing is not supported with slj\n");
            raise RuntimeError("Error changing parameters in wallpotential force");

        wallpotential.set_params(self, mode=mode);

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
