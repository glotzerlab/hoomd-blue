# Copyright (c) 2009-2019 The Regents of the University of Michigan
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
        # hoomd.util.print_status_line();

        # listify the input
        types = hoomd.util.listify(types)

        for typei in types:
            self.__getitem__(typei).set(**params);


class _param(object):
    def __init__(self, mc, typid):
        self.__dict__.update(dict(_keys=['ignore_statistics'], mc=mc, typid=typid, make_fn=None, is_set=False));

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

        self.mc.cpp_integrator.setParam(self.typid, self.make_param(**params));

class sphere_params(_hpmc.sphere_param_proxy, _param):
    def __init__(self, mc, index):
        _hpmc.sphere_param_proxy.__init__(self, mc.cpp_integrator, index);
        _param.__init__(self, mc, index);
        self._keys += ['diameter','orientable'];
        self.make_fn = _hpmc.make_sph_params;

    def __str__(self):
        # should we put this in the c++ side?
        return "sphere(diameter = {})".format(self.diameter)

    def make_param(self, diameter, ignore_statistics=False,orientable=False):
        return self.make_fn(float(diameter)/2.0,ignore_statistics,orientable);

class convex_polygon_params(_hpmc.convex_polygon_param_proxy, _param):
    def __init__(self, mc, index):
        _hpmc.convex_polygon_param_proxy.__init__(self, mc.cpp_integrator, index);
        _param.__init__(self, mc, index);
        self._keys += ['vertices'];
        self.make_fn = _hpmc.make_poly2d_verts;

    def __str__(self):
        # should we put this in the c++ side?
        string = "convex polygon(vertices = {})".format(self.vertices);
        return string;

    def make_param(self, vertices, ignore_statistics=False):
        return self.make_fn(self.ensure_list(vertices),
                            float(0.0),
                            ignore_statistics);

class convex_spheropolygon_params(_hpmc.convex_spheropolygon_param_proxy, _param):
    def __init__(self, mc, index):
        _hpmc.convex_spheropolygon_param_proxy.__init__(self, mc.cpp_integrator, index);
        _param.__init__(self, mc, index);
        self._keys += ['vertices', 'sweep_radius'];
        self.make_fn = _hpmc.make_poly2d_verts;

    def __str__(self):
        # should we put this in the c++ side?
        string = "convex spheropolygon(sweep radius = {}, , vertices = {})".format(self.sweep_radius, self.vertices);
        return string;

    def make_param(self, vertices, sweep_radius = 0.0, ignore_statistics=False):
        return self.make_fn(self.ensure_list(vertices),
                            float(sweep_radius),
                            ignore_statistics);

class simple_polygon_params(_hpmc.simple_polygon_param_proxy, _param):
    def __init__(self, mc, index):
        _hpmc.simple_polygon_param_proxy.__init__(self, mc.cpp_integrator, index);
        _param.__init__(self, mc, index);
        self._keys += ['vertices'];
        self.make_fn = _hpmc.make_poly2d_verts;

    def __str__(self):
        # should we put this in the c++ side?
        string = "simple polygon(vertices = {})".format(self.vertices);
        return string;

    def make_param(self, vertices, ignore_statistics=False):
        return self.make_fn(self.ensure_list(vertices),
                            float(0),
                            ignore_statistics);

class convex_polyhedron_params(_hpmc.convex_polyhedron_param_proxy,_param):
    def __init__(self, mc, index):
        _hpmc.convex_polyhedron_param_proxy.__init__(self, mc.cpp_integrator, index);
        _param.__init__(self, mc, index);
        self._keys += ['vertices'];
        self.make_fn = _hpmc.make_poly3d_verts

    def __str__(self):
        # should we put this in the c++ side?
        string = "convex polyhedron(vertices = {})".format(self.vertices);
        return string;

    def make_param(self, vertices, ignore_statistics=False):
        return self.make_fn(self.ensure_list(vertices),
                            float(0),
                            ignore_statistics,
                            hoomd.context.current.system_definition.getParticleData().getExecConf());

class convex_spheropolyhedron_params(_hpmc.convex_spheropolyhedron_param_proxy,_param):
    def __init__(self, mc, index):
        _hpmc.convex_spheropolyhedron_param_proxy.__init__(self, mc.cpp_integrator, index);
        _param.__init__(self, mc, index);
        self._keys += ['vertices', 'sweep_radius'];
        self.make_fn = _hpmc.make_poly3d_verts;

    def __str__(self):
        # should we put this in the c++ side?
        string = "convex spheropolyhedron(sweep radius = {}, vertices = {})".format(self.sweep_radius, self.vertices);
        return string;

    def make_param(self, vertices, sweep_radius = 0.0, ignore_statistics=False):
        return self.make_fn(self.ensure_list(vertices),
                            float(sweep_radius),
                            ignore_statistics,
                            hoomd.context.current.system_definition.getParticleData().getExecConf());

class polyhedron_params(_hpmc.polyhedron_param_proxy, _param):
    def __init__(self, mc, index):
        _hpmc.polyhedron_param_proxy.__init__(self, mc.cpp_integrator, index);
        _param.__init__(self, mc, index);
        self._keys += ['vertices', 'faces','overlap', 'colors', 'sweep_radius', 'capacity','origin','hull_only'];
        self.make_fn = _hpmc.make_poly3d_data;
        self.__dict__.update(dict(colors=None));

    def __str__(self):
        # should we put this in the c++ side?
        string = "polyhedron(vertices = {}, faces = {}, overlap = {}, colors= {}, sweep_radius = {}, capacity = {}, origin = {})".format(self.vertices, self.faces, self.overlap, self.colors, self.sweep_radius, self.capacity, self.hull_only);
        return string;

    def make_param(self, vertices, faces, sweep_radius=0.0, ignore_statistics=False, origin=(0,0,0), capacity=4, hull_only=True, overlap=None, colors=None):
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

        self.colors = None if colors is None else self.ensure_list(colors);

        if overlap is None:
            overlap = [1 for f in faces]

        if sweep_radius < 0.0:
            hoomd.context.msg.warning("A rounding radius < 0 does not make sense.\n")

        if len(origin) != 3:
            hoomd.context.error("Origin must be a coordinate triple.\n")

        return self.make_fn([self.ensure_list(v) for v in vertices],
                            self.ensure_list(face_verts),
                            self.ensure_list(face_offs),
                            self.ensure_list(overlap),
                            float(sweep_radius),
                            ignore_statistics,
                            capacity,
                            self.ensure_list(origin),
                            int(hull_only),
                            hoomd.context.current.system_definition.getParticleData().getExecConf());

class faceted_ellipsoid_params(_hpmc.faceted_ellipsoid_param_proxy, _param):
    def __init__(self, mc, index):
        _hpmc.faceted_ellipsoid_param_proxy.__init__(self, mc.cpp_integrator, index);
        _param.__init__(self, mc, index);
        self._keys += ['vertices', 'normals', 'offsets', 'a', 'b', 'c', 'origin'];
        self.make_fn = _hpmc.make_faceted_ellipsoid;

    def __str__(self):
        string = "faceted ellipsoid(vertices = {}, normals = {}, offsets = {}, a = {}, b = {}, c = {})".format(
            self.vertices, self.normals, self.offsets, self.a, self.b, self.c);
        return string;

    def make_param(self, normals, offsets, vertices, a, b, c, origin=(0.0, 0.0, 0.0), ignore_statistics=False):
        return self.make_fn(self.ensure_list(normals),
                            self.ensure_list(offsets),
                            self.ensure_list(vertices),
                            float(a),
                            float(b),
                            float(c),
                            tuple(origin),
                            bool(ignore_statistics),
                            hoomd.context.current.system_definition.getParticleData().getExecConf());

# faceted sphere is a specialization of faceted ellipsoid
class faceted_sphere_params(_hpmc.faceted_ellipsoid_param_proxy, _param):
    def __init__(self, mc, index):
        _hpmc.faceted_ellipsoid_param_proxy.__init__(self, mc.cpp_integrator, index);
        _param.__init__(self, mc, index);
        self._keys += ['vertices', 'normals', 'offsets', 'diameter', 'origin'];
        self.make_fn = _hpmc.make_faceted_ellipsoid;

    def __str__(self):
        string = "faceted sphere(vertices = {}, normals = {}, offsets = {}, diameter = {})".format(self.vertices, self.normals, self.offsets, self.diameter);
        return string;

    def make_param(self, normals, offsets, vertices, diameter, origin=(0.0, 0.0, 0.0), ignore_statistics=False):
        return self.make_fn(self.ensure_list(normals),
                            self.ensure_list(offsets),
                            self.ensure_list(vertices),
                            float(diameter)/2.0,
                            float(diameter)/2.0,
                            float(diameter)/2.0,
                            tuple(origin),
                            bool(ignore_statistics),
                            hoomd.context.current.system_definition.getParticleData().getExecConf());


class sphinx_params(_hpmc.sphinx3d_param_proxy, _param):
    def __init__(self, mc, index):
        _hpmc.sphinx3d_param_proxy.__init__(self, mc.cpp_integrator, index);
        _param.__init__(self, mc, index);
        self.__dict__.update(dict(colors=None));
        self._keys += ['diameters', 'centers', 'diameter', 'colors'];
        self.make_fn = _hpmc.make_sphinx3d_params;

    def __str__(self):
        # should we put this in the c++ side?
        string = "sphinx(centers = {}, diameters = {}, diameter = {})".format(self.centers, self.diameters, self.diameter);
        return string;

    def make_param(self, diameters, centers, ignore_statistics=False, colors=None):
        self.colors = None if colors is None else self.ensure_list(colors);
        return self.make_fn(self.ensure_list(diameters),
                            self.ensure_list(centers),
                            ignore_statistics);

class ellipsoid_params(_hpmc.ell_param_proxy, _param):
    def __init__(self, mc, index):
        _hpmc.ell_param_proxy.__init__(self, mc.cpp_integrator, index);
        _param.__init__(self, mc, index);
        self._keys += ['a', 'b', 'c'];
        self.make_fn = _hpmc.make_ell_params;

    def __str__(self):
        # should we put this in the c++ side?
        return "ellipsoid(a = {}, b = {}, c = {})".format(self.a, self.b, self.c)

    def make_param(self, a, b, c, ignore_statistics=False):
        return self.make_fn(float(a),
                            float(b),
                            float(c),
                            ignore_statistics);

class sphere_union_params(_hpmc.sphere_union_param_proxy,_param):
    def __init__(self, mc, index):
        _hpmc.sphere_union_param_proxy.__init__(self, mc.cpp_integrator, index); # we will add this base class later because of the size template
        _param.__init__(self, mc, index);
        self.__dict__.update(dict(colors=None));
        self._keys += ['centers', 'orientations', 'diameter', 'colors','overlap'];
        self.make_fn = _hpmc.make_sphere_union_params;

    def __str__(self):
        # should we put this in the c++ side?
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

    def make_param(self, diameters, centers, overlap=None, ignore_statistics=False, colors=None, capacity=4):
        if overlap is None:
            overlap = [1 for c in centers]

        members = [_hpmc.make_sph_params(float(d)/2.0, False, False) for d in diameters];
        N = len(diameters)
        if len(centers) != N:
            raise RuntimeError("Lists of constituent particle parameters and centers must be equal length.")
        self.colors = None if colors is None else self.ensure_list(colors);
        return self.make_fn(self.ensure_list(members),
                            self.ensure_list(centers),
                            self.ensure_list([[1,0,0,0] for i in range(N)]),
                            self.ensure_list(overlap),
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

    def make_param(self, centers, orientations, vertices, overlap=None, sweep_radii=None, ignore_statistics=False, colors=None, capacity=4):
        if overlap is None:
            overlap = [1 for c in centers]

        if sweep_radii is None:
            sweep_radii = [0 for c in centers]

        members = []
        for i,(verts, sweep_radius) in enumerate(zip(vertices, sweep_radii)):
            member_fn = _hpmc.make_poly3d_verts
            members.append(member_fn(self.ensure_list(verts), float(sweep_radius), ignore_statistics, hoomd.context.current.system_definition.getParticleData().getExecConf()))

        N = len(vertices)
        if len(centers) != N or len(orientations)!= N:
            raise RuntimeError("Lists of constituent particle parameters and centers must be equal length.")
        self.colors = None if colors is None else self.ensure_list(colors);
        return self.make_fn(self.ensure_list(members),
                            self.ensure_list(centers),
                            self.ensure_list(orientations),
                            self.ensure_list(overlap),
                            ignore_statistics,
                            capacity,
                            hoomd.context.current.system_definition.getParticleData().getExecConf());

class convex_polyhedron_union_params(convex_spheropolyhedron_union_params):
    # provided for backward compatibility
    def __init__(self, mc, index):
        # call base class constructor
        convex_spheropolyhedron_union_params.__init__(self,mc, index)

class faceted_ellipsoid_union_params(_hpmc.faceted_ellipsoid_union_param_proxy,_param):
    def __init__(self, mc, index):
        _hpmc.faceted_ellipsoid_union_param_proxy.__init__(self, mc.cpp_integrator, index);
        _param.__init__(self, mc, index);
        self.__dict__.update(dict(colors=None));
        self._keys += ['centers', 'orientations', 'vertices', 'normals', 'offsets', 'colors','overlap','a', 'b', 'c', 'origins'];
        self.make_fn = _hpmc.make_faceted_ellipsoid_union_params

    def __str__(self):
        # should we put this in the c++ side?
        string = "faceted ellipsoid union(centers = {}, orientations = {}, overlap = {})\n".format(self.centers, self.orientations, self.overlap);
        ct = 0;
        members = self.members;
        for m in members:
            end = "\n" if ct < (len(members)-1) else "";
            string+="faceted ellipsoid-{}(v = {}, n = {}, offsets = {}, a = {}, b = {}, c = {}){}".format(
                ct, m.vertices, m.normals, m.offsets, m.a, m.b, m.c, mend)
            ct+=1
        return string;

    def get_metadata(self):
        data = {}
        for key in self._keys:
            if key == 'vertices':
                val = [ m.vertices for m in self.members ];
            if key == 'normals':
                val = [ m.normals for m in self.members ];
            if key == 'offsets':
                val = [ m.offsets for m in self.members ];
            if key == 'a':
                val = [ m.a for m in self.members ];
            if key == 'b':
                val = [ m.b for m in self.members ];
            if key == 'c':
                val = [ m.c for m in self.members ];
            if key == 'origins':
                val = [ m.origin for m in self.members ];
            else:
                val = getattr(self, key);
            data[key] = val;
        return data;

    def make_param(self, centers, orientations, vertices, normals, offsets, axes, origins=None, overlap=None,
        ignore_statistics=False, colors=None, capacity=4):
        if overlap is None:
            overlap = [1 for c in centers]

        if origins is None:
            origins = [(0,0,0) for c in centers]

        members = []
        for i,(cur_vertices, cur_normals, cur_offsets, cur_axes, cur_origin) in enumerate(zip(vertices, normals, offsets, axes, origins)):
            member_fn = _hpmc.make_faceted_ellipsoid
            members.append(member_fn(self.ensure_list(cur_normals),
                                     self.ensure_list(cur_offsets),
                                     self.ensure_list(cur_vertices),
                                     float(cur_axes[0]),
                                     float(cur_axes[1]),
                                     float(cur_axes[2]),
                                     tuple(cur_origin),
                                     False, # ignore_statistics
                                     hoomd.context.current.system_definition.getParticleData().getExecConf()))

        N = len(vertices)
        if len(centers) != N or len(orientations)!= N:
            raise RuntimeError("Lists of constituent particle parameters and centers must be equal length.")
        self.colors = None if colors is None else self.ensure_list(colors);

        return self.make_fn(self.ensure_list(members),
                            self.ensure_list(centers),
                            self.ensure_list(orientations),
                            self.ensure_list(overlap),
                            ignore_statistics,
                            capacity,
                            hoomd.context.current.system_definition.getParticleData().getExecConf());
