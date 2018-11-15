# Copyright (c) 2009-2017 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

""" Shape data structures.
"""

import hoomd
import hoomd.hpmc
from hoomd.hpmc import _hpmc
import numpy

class param_dict(dict):
    R""" Manage shape parameters.

    The parameters for all hpmc integrator shapes (:py:mod:`hoomd.hpmc.integrate`) are specified using this class.
    Parameters are specified per particle type. Every HPMC integrator has a member shape_param that can read and
    set parameters of the shapes.

    :py:class:`param_dict` can be used as a dictionary to access parameters by type. You can read individual parameters
    or set parameters with :py:meth:`set`.

    Example::

        mc = hpmc.integrate.sphere();
        mc.shape_param['A'].set(diameter=2.0)
        mc.shape_param['B'].set(diameter=0.1)
        dA = mc.shape_param['A'].diameter
        dB = mc.shape_param['B'].diameter

    """

    def __init__(self, mc):
        dict.__init__(self);
        self.mc = mc;

    def __getitem__(self, key):
        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_names = [ hoomd.context.current.system_definition.getParticleData().getNameByType(i) for i in range(0,ntypes) ];
        if not key in type_names:
            raise RuntimeError("{} is not a known particle type".format(key));
        elif not key in self.keys():
            self.mc.initialize_shape_params(); # add any extra parameters in that exist at this time.
            if not key in self.keys():
                raise RuntimeError("could not create proxy for type {}".format(key));
        return super(param_dict, self).__getitem__(key);

    def set(self, types, **params):
        """ Sets parameters for particle type(s).

        Args:
            type (str): Particle type (string) or list of types
            params: Named parameters (see specific integrator for required parameters - :py:mod:`hoomd.hpmc.integrate`)

        Calling set() results in one or more parameters being set for a shape. Types are identified
        by name, and parameters are also added by name. Which parameters you need to specify depends on the hpmc
        integrator you are setting these coefficients for, see the corresponding documentation.

        All possible particle types types defined in the simulation box must be specified before executing :py:func:`hoomd.run()`.
        You will receive an error if you fail to do so. It is an error to specify coefficients for
        particle types that do not exist in the simulation.

        To set the same parameters for many particle types, provide a list of type names instead of a single
        one. All types in the list will be set to the same parameters. A convenient wildcard that lists all types
        of particles in the simulation can be gotten from a saved `sysdef` from the init command.

        Examples::

            mc.shape_param.set('A', diameter=1.0)
            mc.shape_param.set('B', diameter=2.0)
            mc.shape_param.set(['A', 'B'], diameter=2.0)


        Note:
            Single parameters can not be updated. If both *diameter* and *length* are required for a particle type,
            then executing coeff.set('A', diameter=1.5) will fail one must call coeff.set('A', diameter=1.5, length=2.0)

        """
        # listify the input
        types = hoomd.util.listify(types)

        for typei in types:
            self.__getitem__(typei).set(**params);


class _param(object):
    def __init__(self, mc, typid):
        self.__dict__.update(dict(_keys=['ignore_statistics'], mc=mc, typid=typid, is_set=False, _py_params=[]));

    @classmethod
    def ensure_list(cls, li):
        # this will be slow if called many times but it is more robust.
        ai = numpy.array(li);
        return ai.tolist();

    def get_metadata(self):
        data = {}
        for key in self._keys:
            data[key] = getattr(self, key);
        return data;

    def __setattr__(self, name, value):
        if not hasattr(self, name):
            raise AttributeError('{} instance has no attribute {!r}'.format(type(self).__name__, name));
        super(_param, self).__setattr__(name, value);

    def set(self, **params):
        self.is_set = True;

        # backwards compatibility
        if 'ignore_overlaps' in params:
            # ugly workaround
            super(_param,self).__setattr__('ignore_overlaps',params['ignore_overlaps'])
            # do not pass to C++
            params.pop('ignore_overlaps',None)

        for pname in self._py_params:
            if pname in params:
                self.__setattr__(pname, params.pop(pname, None))

        self.mc.cpp_integrator.setParam(self.typid, self.make_param(**params), True);

class sphere_params(_hpmc.sphere_param_proxy, _param):
    def __init__(self, mc, index):
        _hpmc.sphere_param_proxy.__init__(self, mc.cpp_integrator, index);
        _param.__init__(self, mc, index);
        self._keys += ['diameter','orientable'];

    def __str__(self):
        return "sphere(diameter = {})".format(self.diameter)

    @classmethod
    def make_param(cls, diameter, ignore_statistics=False, orientable=False):
        return _hpmc.make_sph_params(   float(diameter)/2.0,
                                        ignore_statistics,orientable);

class convex_polygon_params(_hpmc.convex_polygon_param_proxy, _param):
    def __init__(self, mc, index):
        _hpmc.convex_polygon_param_proxy.__init__(self, mc.cpp_integrator, index);
        _param.__init__(self, mc, index);
        self._keys += ['vertices'];

    def __str__(self):
        string = "convex polygon(vertices = {})".format(self.vertices);
        return string;

    @classmethod
    def make_param(cls, vertices, ignore_statistics=False):
        return _hpmc.make_poly2d_verts( cls.ensure_list(vertices),
                                        float(0.0),
                                        ignore_statistics);

class convex_spheropolygon_params(_hpmc.convex_spheropolygon_param_proxy, _param):
    def __init__(self, mc, index):
        _hpmc.convex_spheropolygon_param_proxy.__init__(self, mc.cpp_integrator, index);
        _param.__init__(self, mc, index);
        self._keys += ['vertices', 'sweep_radius'];

    def __str__(self):
        string = "convex spheropolygon(sweep radius = {}, , vertices = {})".format(self.sweep_radius, self.vertices);
        return string;

    @classmethod
    def make_param(cls, vertices, sweep_radius = 0.0, ignore_statistics=False):
        return _hpmc.make_poly2d_verts( cls.ensure_list(vertices),
                                        float(sweep_radius),
                                        ignore_statistics);

class simple_polygon_params(_hpmc.simple_polygon_param_proxy, _param):
    def __init__(self, mc, index):
        _hpmc.simple_polygon_param_proxy.__init__(self, mc.cpp_integrator, index);
        _param.__init__(self, mc, index);
        self._keys += ['vertices'];

    def __str__(self):
        string = "simple polygon(vertices = {})".format(self.vertices);
        return string;

    @classmethod
    def make_param(cls, vertices, ignore_statistics=False):
        return _hpmc.make_poly2d_verts( cls.ensure_list(vertices),
                                        float(0),
                                        ignore_statistics);

class convex_polyhedron_params(_hpmc.convex_polyhedron_param_proxy,_param):
    def __init__(self, mc, index):
        _hpmc.convex_polyhedron_param_proxy.__init__(self, mc.cpp_integrator, index);
        _param.__init__(self, mc, index);
        self._keys += ['vertices'];

    def __str__(self):
        string = "convex polyhedron(vertices = {})".format(self.vertices);
        return string;

    @classmethod
    def make_param(cls, vertices, ignore_statistics=False):
        return _hpmc.make_poly3d_verts( cls.ensure_list(vertices),
                                        float(0),
                                        ignore_statistics,
                                        hoomd.context.current.system_definition.getParticleData().getExecConf());

class convex_spheropolyhedron_params(_hpmc.convex_spheropolyhedron_param_proxy,_param):
    def __init__(self, mc, index):
        _hpmc.convex_spheropolyhedron_param_proxy.__init__(self, mc.cpp_integrator, index);
        _param.__init__(self, mc, index);
        self._keys += ['vertices', 'sweep_radius'];

    def __str__(self):
        string = "convex spheropolyhedron(sweep radius = {}, vertices = {})".format(self.sweep_radius, self.vertices);
        return string;

    @classmethod
    def make_param(cls, vertices, sweep_radius = 0.0, ignore_statistics=False):
        return _hpmc.make_poly3d_verts( cls.ensure_list(vertices),
                                        float(sweep_radius),
                                        ignore_statistics,
                                        hoomd.context.current.system_definition.getParticleData().getExecConf());

class polyhedron_params(_hpmc.polyhedron_param_proxy, _param):

    def __init__(self, mc, index):
        _hpmc.polyhedron_param_proxy.__init__(self, mc.cpp_integrator, index);
        _param.__init__(self, mc, index);
        self._keys += ['vertices', 'faces','overlap', 'colors', 'sweep_radius', 'capacity','origin','hull_only'];
        self.__dict__.update(dict(colors=None));
        self._py_params = ['colors'];

    def __str__(self):
        string = "polyhedron(vertices = {}, faces = {}, overlap = {}, colors= {}, sweep_radius = {}, capacity = {}, origin = {})".format(self.vertices, self.faces, self.overlap, self.colors, self.sweep_radius, self.capacity, self.hull_only);
        return string;

    @classmethod
    def make_param(cls, vertices, faces, sweep_radius=0.0, ignore_statistics=False, origin=(0,0,0), capacity=4, hull_only=True, overlap=None): #  colors=None
        face_offs = []
        face_verts = []
        offs = 0

        for face in faces:
            if len(face) != 3 and len(face) != 1:
                hoomd.context.msg.error("Only triangulated shapes and spheres are supported.\n")
                raise RuntimeError('Error setting shape parameters')
            face_offs.append(offs)
            for face_idx in face:
                face_verts.append(int(face_idx))
            offs += len(face)

        # end offset
        face_offs.append(offs)

        if overlap is None:
            overlap = [1 for f in faces]

        if sweep_radius < 0.0:
            hoomd.context.msg.warning("A rounding radius < 0 does not make sense.\n")

        # TODO: make this a class method
        if len(origin) != 3:
            hoomd.context.error("Origin must be a coordinate triple.\n")

        return _hpmc.make_poly3d_data( [cls.ensure_list(v) for v in vertices],
                                        cls.ensure_list(face_verts),
                                        cls.ensure_list(face_offs),
                                        cls.ensure_list(overlap),
                                        float(sweep_radius),
                                        ignore_statistics,
                                        capacity,
                                        cls.ensure_list(origin),
                                        int(hull_only),
                                        hoomd.context.current.system_definition.getParticleData().getExecConf());

class faceted_sphere_params(_hpmc.faceted_sphere_param_proxy, _param):
    def __init__(self, mc, index):
        _hpmc.faceted_sphere_param_proxy.__init__(self, mc.cpp_integrator, index);
        _param.__init__(self, mc, index);
        self._keys += ['vertices', 'normals', 'offsets', 'diameter', 'origin'];

    def __str__(self):
        string = "faceted sphere(vertices = {}, normals = {}, offsets = {})".format(self.vertices, self.normals, self.offsets);
        return string;

    @classmethod
    def make_param(cls, normals, offsets, vertices, diameter, origin=(0.0, 0.0, 0.0), ignore_statistics=False):
        return _hpmc.make_faceted_sphere(   cls.ensure_list(normals),
                                            cls.ensure_list(offsets),
                                            cls.ensure_list(vertices),
                                            float(diameter),
                                            tuple(origin),
                                            bool(ignore_statistics),
                                            hoomd.context.current.system_definition.getParticleData().getExecConf());

class sphinx_params(_hpmc.sphinx3d_param_proxy, _param):
    def __init__(self, mc, index):
        _hpmc.sphinx3d_param_proxy.__init__(self, mc.cpp_integrator, index);
        _param.__init__(self, mc, index);
        self.__dict__.update(dict(colors=None));
        self._keys += ['diameters', 'centers', 'diameter', 'colors'];
        self._py_params = ['colors'];
        self.colors = None;

    def __str__(self):
        string = "sphinx(centers = {}, diameters = {}, diameter = {})".format(self.centers, self.diameters, self.diameter);
        return string;

    @classmethod
    def make_param(cls, diameters, centers, ignore_statistics=False): # colors=None
        return _hpmc.make_sphinx3d_params(  cls.ensure_list(diameters),
                                            cls.ensure_list(centers),
                                            ignore_statistics);

class ellipsoid_params(_hpmc.ell_param_proxy, _param):
    def __init__(self, mc, index):
        _hpmc.ell_param_proxy.__init__(self, mc.cpp_integrator, index);
        _param.__init__(self, mc, index);
        self._keys += ['a', 'b', 'c'];

    def __str__(self):
        return "ellipsoid(a = {}, b = {}, c = {})".format(self.a, self.b, self.c)

    @classmethod
    def make_param(cls, a, b, c, ignore_statistics=False):
        return _hpmc.make_ell_params(   float(a),
                                        float(b),
                                        float(c),
                                        ignore_statistics);

class sphere_union_params(_hpmc.sphere_union_param_proxy,_param):
    def __init__(self, mc, index):
        _hpmc.sphere_union_param_proxy.__init__(self, mc.cpp_integrator, index); # we will add this base class later because of the size template
        _param.__init__(self, mc, index);
        self.__dict__.update(dict(colors=None));
        self._py_params = ['colors'];
        self.colors = None;
        self._keys += ['centers', 'orientations', 'diameter', 'colors','overlap'];

    def __str__(self):
        string = "sphere union(centers = {}, orientations = {}, diameter = {}, overlap = {}, capacity = {})\n".format(self.centers, self.orientations, self.diameter, self.overlap, self.capacity);
        ct = 0;
        members = self.members;
        for m in members:
            end = "\n" if ct < (len(members)-1) else "";
            string+="sphere-{}(d = {}){}".format(ct, m.diameter, end)
            ct+=1
        return string;

    def get_metadata(self):
        data = {}
        for key in self._keys:
            if key == 'diameters':
                val = [ m.diameter for m in self.members ];
            else:
                val = getattr(self, key);
            data[key] = val;
        return data;

    @classmethod
    def make_param(cls, diameters, centers, overlap=None, ignore_statistics=False, capacity=4):
        if overlap is None:
            overlap = [1 for c in centers]
        members = [_hpmc.make_sph_params(float(d)/2.0, False, False) for d in diameters];
        N = len(diameters)
        if len(centers) != N:
            raise RuntimeError("Lists of constituent particle parameters and centers must be equal length.")
        return _hpmc.make_sphere_union_params(cls.ensure_list(members),
                            cls.ensure_list(centers),
                            cls.ensure_list([[1,0,0,0] for i in range(N)]),
                            cls.ensure_list(overlap),
                            ignore_statistics,
                            capacity,
                            hoomd.context.current.system_definition.getParticleData().getExecConf());

class convex_spheropolyhedron_union_params(_hpmc.convex_polyhedron_union_param_proxy,_param):
    def __init__(self, mc, index):
        _hpmc.convex_polyhedron_union_param_proxy.__init__(self, mc.cpp_integrator, index); # we will add this base class later because of the size templated
        _param.__init__(self, mc, index);
        self.__dict__.update(dict(colors=None));
        self._keys += ['centers', 'orientations', 'vertices', 'colors','overlap','sweep_radii'];
        self.make_fn = _hpmc.make_convex_polyhedron_union_params;

    def __str__(self):
        # should we put this in the c++ side?
        string = "convex polyhedron union(centers = {}, orientations = {}, overlap = {})\n".format(self.centers, self.orientations, self.overlap);
        ct = 0;
        members = self.members;
        for m in members:
            end = "\n" if ct < (len(members)-1) else "";
            string+="convex polyhedron-{}(v = {}, R = {}){}".format(ct, m.vertices, m.sweep_radius, end)
            ct+=1
        return string;

    def get_metadata(self):
        data = {}
        for key in self._keys:
            if key == 'vertices':
                val = [ m.vertices for m in self.members ];
            elif key == 'sweep_radii':
                val = [ m.sweep_radius for m in self.members ]
            else:
                val = getattr(self, key);
            data[key] = val;
        return data;

    @classmethod
    def make_param(cls, centers, orientations, vertices, overlap=None, ignore_statistics=False, capacity=4): #colors=None,
        if overlap is None:
            overlap = [1 for c in centers]

        if sweep_radii is None:
            sweep_radii = [0 for c in centers]

        members = []
        for i,(verts, sweep_radius) in enumerate(zip(vertices, sweep_radii)):
            member_fn = _hpmc.make_poly3d_verts
            members.append(member_fn(cls.ensure_list(verts), float(0), ignore_statistics, hoomd.context.current.system_definition.getParticleData().getExecConf()))

        N = len(vertices)
        if len(centers) != N or len(orientations)!= N:
            raise RuntimeError("Lists of constituent particle parameters and centers must be equal length.")
        return _hpmc.make_convex_polyhedron_union_params(
                            cls.ensure_list(members),
                            cls.ensure_list(centers),
                            cls.ensure_list(orientations),
                            cls.ensure_list(overlap),
                            ignore_statistics,
                            capacity,
                            hoomd.context.current.system_definition.getParticleData().getExecConf());

class convex_polyhedron_union_params(convex_spheropolyhedron_union_params):
    # provided for backward compatibility
    def __init__(self, mc, index):
        # call base class constructor
        convex_spheropolyhedron_union_params.__init__(self,mc, index)
