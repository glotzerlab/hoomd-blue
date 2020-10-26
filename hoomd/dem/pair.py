# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

R"""DEM pair potentials.
"""

import hoomd;
import hoomd.md;
import hoomd.md.nlist as nl;

from math import sqrt;
import json;

from hoomd.dem import _dem;
from hoomd.dem import params;
from hoomd.dem import utils;

class _DEMBase:
    def __init__(self, nlist):
        self.nlist = nlist;
        self.nlist.subscribe(self.get_rcut);
        self.nlist.update_rcut();
        self.cpp_force = None;

    def _initialize_types(self):
        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i));

        if self.dimensions == 2:
            for typ in type_list:
                self.setParams2D(typ, [[0, 0]], False);
        else:
            for typ in type_list:
                self.setParams3D(typ, [[0, 0, 0]], [], False);

    def _connect_gsd_shape_spec(self, gsd):
        # This is an internal method, and should not be called directly. See gsd.dump_shape() instead
        if isinstance(gsd, hoomd.dump.gsd) and hasattr(self.cpp_force, "connectDEMGSDShapeSpec"):
            self.cpp_force.connectDEMGSDShapeSpec(gsd.cpp_analyzer);
        else:
            raise NotImplementedError("GSD Schema is not implemented for {}".format(self.__class__.__name__));

    def setParams2D(self, type, vertices, center=False):
        """Set the vertices for a given particle type.

        Args:
            type (str): Name of the type to set the shape of
            vertices (list): List of (2D) points specifying the coordinates of the shape
            center (bool): If True, subtract the center of mass of the shape from the vertices before setting them for the shape

        Shapes are specified as a list of 2D coordinates. Edges will
        be made between all adjacent pairs of vertices, including one
        between the last and first vertex.
        """
        itype = hoomd.context.current.system_definition.getParticleData().getTypeByName(type);

        if not len(vertices):
            vertices = [(0, 0)];
            center = False;

        # explicitly turn into a list of tuples
        if center:
            vertices = [(float(p[0]), float(p[1])) for p in utils.center(vertices)];
        else:
            vertices = [(float(p[0]), float(p[1])) for p in vertices];

        # update the neighbor list
        rcutmax = 2*(sqrt(max(x*x + y*y for (x, y) in vertices)) + self.radius*2**(1./6));
        self.r_cut = max(self.r_cut, rcutmax);

        self.vertices[type] = vertices;
        self.cpp_force.setRcut(self.r_cut);
        self.cpp_force.setParams(itype, vertices);

    def setParams3D(self, type, vertices, faces, center=False):
        """Set the vertices for a given particle type.

        Args:
            type (str): Name of the type to set the shape of
            vertices (list): List of (3D) points specifying the coordinates of the shape
            faces (list): List of lists of indices specifying which coordinates comprise each face of a shape.
            center (bool): If True, subtract the center of mass of the shape from the vertices before setting them for the shape

        Shapes are specified as a list of coordinates (`vertices`) and
        another list containing one list for each polygonal face
        (`faces`). The elements of each list inside `faces` are
        integer indices specifying which vertex in `vertices` comprise
        the face.
        """
        itype = hoomd.context.current.system_definition.getParticleData().getTypeByName(type);

        if not len(vertices):
            vertices = [(0, 0, 0)];
            faces = [];
            center = False;

        # explicitly turn into python lists
        if center:
            vertices = [(float(p[0]), float(p[1]), float(p[2])) for p in utils.center(vertices, faces)];
        else:
            vertices = [(float(p[0]), float(p[1]), float(p[2])) for p in vertices];
        faces = [[int(i) for i in face] for face in faces];

        # update the neighbor list
        rcutmax = 2*(sqrt(max(x*x + y*y + z*z for (x, y, z) in vertices)) + self.radius*2**(1./6));
        self.r_cut = max(self.r_cut, rcutmax);

        self.vertices[type] = vertices;
        self.cpp_force.setRcut(self.r_cut);
        self.cpp_force.setParams(itype, vertices, faces);

    def get_type_shapes(self):
        """Get all the types of shapes in the current simulation.

        This assumes all 3D shapes are convex.

        Examples:
            Types depend on the number of shape vertices and system dimensionality.
            One vertex will yield a Sphere (2D and 3D), while multiple vertices will
            give a Polygon (2D) or ConvexPolyhedron (3D).

            >>> mc.get_type_shapes()  # one vertex in 3D
            [{'type': 'Sphere', 'diameter': 1.0}]
            >>> mc.get_type_shapes()  # one vertex in 2D
            [{'type': 'Sphere', 'diameter': 1.5}]
            >>> mc.get_type_shapes()  # multiple vertices in 3D
            [{'type': 'ConvexPolyhedron', 'rounding_radius': 0.1,
              'vertices': [[0.5, 0.5, 0.5], [0.5, -0.5, -0.5],
                           [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5]]}]
            >>> mc.get_type_shapes()  # multiple vertices in 2D
            [{'type': 'Polygon', 'rounding_radius': 0.1,
              'vertices': [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]}]

        Returns:
            A list of dictionaries, one for each particle type in the system.
        """
        type_shapes = self.cpp_force.getTypeShapesPy();
        ret = [ json.loads(json_string) for json_string in type_shapes ];
        return ret;

class WCA(hoomd.md.force._force, _DEMBase):
    R"""Specify a purely repulsive Weeks-Chandler-Andersen DEM force with a constant rounding radius.

    Args:
        nlist (:py:mod:`hoomd.md.nlist`): Neighbor list to use
        radius (float): Rounding radius :math:`r` to apply to the shape vertices

    The effect is as if a :py:class:`hoomd.md.pair.lj` interaction
    with :math:`r_{cut}=2^{1/6}\sigma` and :math:`\sigma=2\cdot r`
    were applied between the contact points of each pair of particles.

    Examples::

        # 2D system of squares
        squares = hoomd.dem.pair.WCA(radius=.5)
        squares.setParams('A', [[1, 1], [-1, 1], [-1, -1], [1, -1]])
        # 3D system of rounded square plates
        squarePlates = hoomd.dem.pair.WCA(radius=.5)
        squarePlates.setParams('A',
            vertices=[[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]],
            faces=[[0, 1, 2, 3]], center=False)
        # 3D system of some convex shape specified by vertices
        (vertices, faces) = hoomd.dem.utils.convexHull(vertices)
        shapes = hoomd.dem.pair.WCA(radius=.5)
        shapes.setParams('A', vertices=vertices, faces=faces)

    """

    def __init__(self, nlist, radius=1.):
        hoomd.util.print_status_line();
        friction = None;

        self.radius = radius;
        self.autotunerEnabled = True;
        self.autotunerPeriod = 100000;
        self.vertices = {};

        self.onGPU = hoomd.context.exec_conf.isCUDAEnabled();
        cppForces = {(2, None, 'cpu'): _dem.WCADEM2D,
             (2, None, 'gpu'): (_dem.WCADEM2DGPU if self.onGPU else None),
             (3, None, 'cpu'): _dem.WCADEM3D,
             (3, None, 'gpu'): (_dem.WCADEM3DGPU if self.onGPU else None)};

        self.dimensions = hoomd.context.current.system_definition.getNDimensions();

        # initialize the base class
        hoomd.md.force._force.__init__(self);

        # interparticle cutoff radius, will be updated as shapes are added
        self.r_cut = 2*radius*2**(1./6);

        if friction is None:
            potentialParams = params.WCA(radius=radius);
        else:
            raise RuntimeError('Unknown friction type: {}'.format(friction));

        _DEMBase.__init__(self, nlist);

        key = (self.dimensions, friction, 'gpu' if self.onGPU else 'cpu');
        cpp_force = cppForces[key];

        self.cpp_force = cpp_force(hoomd.context.current.system_definition,
                                   self.nlist.cpp_nlist, self.r_cut,
                                   potentialParams);

        if self.dimensions == 2:
            self.setParams = self.setParams2D;
        else:
            self.setParams = self.setParams3D;

        self._initialize_types();

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

    def update_coeffs(self):
        """Noop for this potential"""
        pass

    def setAutotunerParams(self, enable=None, period=None):
        if not self.onGPU:
            return;
        if enable is not None:
            self.autotunerEnabled = enable;
        if period is not None:
            self.autotunerPeriod = period;
        self.cpp_force.setAutotunerParams(self.autotunerEnabled, self.autotunerPeriod);

    def get_rcut(self):
        # self.log is True if the force is enabled
        if not self.log:
            return None;

        # go through the list of only the active particle types in the sim
        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i));

        # update the rcut by pair type
        r_cut_dict = nl.rcut();
        r_max_dict = {typ: sqrt(max(sum(p*p for p in point)
                                    for point in self.vertices[typ]))
                      for typ in self.vertices};
        for i in range(ntypes):
            for j in range(i, ntypes):
                (typei, typej) = type_list[i], type_list[j];
                r_cut_dict.set_pair(typei, typej,
                                    r_max_dict.get(typei, 0) + r_max_dict.get(typej, 0) + self.radius*2*2.0**(1./6));

        r_cut_dict.fill();

        return r_cut_dict;

class SWCA(hoomd.md.force._force, _DEMBase):
    R"""Specify a purely repulsive Weeks-Chandler-Andersen DEM force with a particle-varying rounding radius.

    Args:
        nlist (:py:mod:`hoomd.md.nlist`): Neighbor list to use
        radius (float): Unshifted rounding radius :math:`r` to apply to the shape vertices
        d_max (float): maximum rounding diameter among all particles in the system

    The SWCA potential enables simulation of particles with
    heterogeneous rounding radii. The effect is as if a
    :py:class:`hoomd.md.pair.slj` interaction with
    :math:`r_{cut}=2^{1/6}\sigma` and :math:`\sigma=2\cdot r` were
    applied between the contact points of each pair of particles.

    Examples::

        # 2D system of squares
        squares = hoomd.dem.pair.SWCA(radius=.5)
        squares.setParams('A', [[1, 1], [-1, 1], [-1, -1], [1, -1]])
        # 3D system of rounded square plates
        squarePlates = hoomd.dem.pair.SWCA(radius=.5)
        squarePlates.setParams('A',
            vertices=[[1, 1, 0], [-1, 1, 0], [-1, -1, 0], [1, -1, 0]],
            faces=[[0, 1, 2, 3]], center=False)
        # 3D system of some convex shape specified by vertices
        (vertices, faces) = hoomd.dem.utils.convexHull(vertices)
        shapes = hoomd.dem.pair.SWCA(radius=.5)
        shapes.setParams('A', vertices=vertices, faces=faces)

    """
    def __init__(self, nlist, radius=1., d_max=None):
        hoomd.util.print_status_line();
        friction = None;

        self.radius = radius;
        self.autotunerEnabled = True;
        self.autotunerPeriod = 100000;
        self.vertices = {};

        self.onGPU = hoomd.context.exec_conf.isCUDAEnabled();
        cppForces = {(2, None, 'cpu'): _dem.SWCADEM2D,
             (2, None, 'gpu'): (_dem.SWCADEM2DGPU if self.onGPU else None),
             (3, None, 'cpu'): _dem.SWCADEM3D,
             (3, None, 'gpu'): (_dem.SWCADEM3DGPU if self.onGPU else None)};

        self.dimensions = hoomd.context.current.system_definition.getNDimensions();

        # Error out in MPI simulations
        if (hoomd._hoomd.is_MPI_available()):
            if hoomd.context.current.system_definition.getParticleData().getDomainDecomposition():
                hoomd.context.msg.error("pair.SWCA is not supported in multi-processor simulations.\n\n");
                raise RuntimeError("Error setting up pair potential.");

        # initialize the base class
        hoomd.md.force._force.__init__(self);

        # update the neighbor list
        if d_max is None :
            sysdef = hoomd.context.current.system_definition;
            self.d_max = max(x.diameter for x in hoomd.data.particle_data(sysdef.getParticleData()));
            hoomd.context.msg.notice(2, "Notice: swca set d_max=" + str(self.d_max) + "\n");

        # interparticle cutoff radius, will be updated as shapes are added
        self.r_cut = 2*2*self.radius*2**(1./6);

        if friction is None:
            potentialParams = params.SWCA(radius=radius);
        else:
            raise RuntimeError('Unknown friction type: {}'.format(friction));

        _DEMBase.__init__(self, nlist);

        key = (self.dimensions, friction, 'gpu' if self.onGPU else 'cpu');
        cpp_force = cppForces[key];

        self.cpp_force = cpp_force(hoomd.context.current.system_definition,
                                   self.nlist.cpp_nlist, self.r_cut,
                                   potentialParams);

        if self.dimensions == 2:
            self.setParams = self.setParams2D;
        else:
            self.setParams = self.setParams3D;

        self._initialize_types();

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

    def update_coeffs(self):
        """Noop for this potential"""
        pass

    def setAutotunerParams(self, enable=None, period=None):
        if not self.onGPU:
            return;
        if enable is not None:
            self.autotunerEnabled = enable;
        if period is not None:
            self.autotunerPeriod = period;
        self.cpp_force.setAutotunerParams(self.autotunerEnabled, self.autotunerPeriod);

    def get_rcut(self):
        if not self.log:
            return None;

        # go through the list of only the active particle types in the sim
        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i));

        # update the rcut by pair type
        r_cut_dict = nl.rcut();
        r_max_dict = {typ: sqrt(max(sum(p*p for p in point)
                                    for point in self.vertices[typ]))
                      for typ in self.vertices};
        for i in range(ntypes):
            for j in range(i, ntypes):
                (typei, typej) = type_list[i], type_list[j];
                r_cut_dict.set_pair(typei, typej,
                                    r_max_dict.get(typei, 0) + r_max_dict.get(typej, 0) +
                                    self.radius*2*2.0**(1./6) + self.d_max - 1);

        r_cut_dict.fill();

        return r_cut_dict;
