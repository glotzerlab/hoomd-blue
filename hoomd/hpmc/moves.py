# Copyright (c) 2009-2016 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

""" Shape move functions
"""

import hoomd
import hoomd.hpmc
from hoomd.hpmc import _hpmc
from . import integrate
import numpy

def python_shape_move(self, callback, params, stepsize, param_ratio):
    move_cls = None;
    if isinstance(self.mc, integrate.sphere):
        move_cls = _hpmc.PythonShapeMoveSphere;
    elif isinstance(self.mc, integrate.convex_polygon):
        move_cls = _hpmc.PythonShapeMoveConvexPolygon;
    elif isinstance(self.mc, integrate.simple_polygon):
        move_cls = _hpmc.PythonShapeMoveSimplePolygon;
    elif isinstance(self.mc, integrate.convex_polyhedron):
        move_cls = integrate._get_sized_entry('PyhtonShapeMoveConvexPolyhedron', self.mc.max_verts);
    elif isinstance(self.mc, integrate.convex_spheropolyhedron):
        move_cls = integrate._get_sized_entry('PythonShapeMoveSpheropolyhedron', self.mc.max_verts);
    elif isinstance(self.mc, integrate.ellipsoid):
        move_cls = _hpmc.PythonShapeMoveEllipsoid;
    elif isinstance(self.mc, integrate.convex_spheropolygon):
        move_cls = _hpmc.PythonShapeMoveConvexSphereopolygon;
    elif isinstance(self.mc, integrate.polyhedron):
        move_cls = _hpmc.PythonShapeMovePolyhedron;
    elif isinstance(self.mc, integrate.sphinx):
        move_cls = _hpmc.PythonShapeMoveSphinx;
    elif isinstance(self.mc, integrate.sphere_union):
        move_cls = _hpmc.PythonShapeMoveSphereUnion;
    else:
        globals.msg.error("update.shape_update.vertex_shifter: Unsupported integrator.\n");
        raise RuntimeError("Error initializing update.shape_update");

    if not move_cls:
        globals.msg.error("update.shape_update: Unsupported integrator.\n");
        raise RuntimeError("Error initializing update.shape_update");

    ntypes = globals.system_definition.getParticleData().getNTypes();
    self.cpp_updater.registerShapeMove(move_cls(ntypes, callback, params, stepsize, param_ratio));


def vertex_shifter(self, stepsize=0.01, mixratio=0.25, volume=1.0):
    move_cls = None;
    if isinstance(self.mc, integrate.sphere):
        pass;
    elif isinstance(self.mc, integrate.convex_polygon):
        pass;
    elif isinstance(self.mc, integrate.simple_polygon):
        pass;
    elif isinstance(self.mc, integrate.convex_polyhedron):
        move_cls = integrate._get_sized_entry('GeneralizedShapeMoveConvexPolyhedron', self.mc.max_verts);
    elif isinstance(self.mc, integrate.convex_spheropolyhedron):
        pass;
    elif isinstance(self.mc, integrate.ellipsoid):
        pass;
    elif isinstance(self.mc, integrate.convex_spheropolygon):
        pass;
    elif isinstance(self.mc, integrate.patchy_sphere):
        pass;
    elif isinstance(self.mc, integrate.polyhedron):
        pass;
    elif isinstance(self.mc, integrate.sphinx):
        pass;
    elif isinstance(self.mc, integrate.sphere_union):
        pass;
    else:
        globals.msg.error("update.shape_update.vertex_shifter: Unsupported integrator.\n");
        raise RuntimeError("Error initializing update.shape_update");

    if not move_cls:
        globals.msg.error("update.shape_update: Unsupported integrator.\n");
        raise RuntimeError("Error initializing update.shape_update");

    ntypes = globals.system_definition.getParticleData().getNTypes();
    self.cpp_updater.registerShapeMove(move_cls(ntypes, stepsize, mixratio, volume));
