from __future__ import division
from __future__ import print_function

import hoomd
from hoomd import context, data, init
import hoomd.deprecated
from hoomd import hpmc

import unittest
import os
import numpy as np
import itertools
import sys
import time

context.initialize();

class shape_move_funciton(object):
    def __init__(self, mc, shape_params):
        self.mc = mc;
        self.shape_params = shape_params;

    def __call__(self, params):
        return self.mc.shape_class.make_param(**self.shape_params);

def create_empty(**kwargs):
    snap = data.make_snapshot(**kwargs);
    return init.read_snapshot(snap);

class shape_updater_test(unittest.TestCase):
    def setUp(self):
        self.types = [str(i) for i in range(8)];
        self.system = create_empty(N=8, box=hoomd.data.boxdim(L=5), particle_types=self.types);
        positions = [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]]
        for p in self.system.particles:
            p.position = positions[p.tag];
            p.type = self.types[p.tag];
        self.updater = None
        self.mc = None

    def tearDown(self):
        del self.mc
        del self.system
        del self.updater
        context.initialize()

    def test_sphere_python(self):
        # sphere
        diam = 1.0;
        self.mc = hpmc.integrate.sphere(seed=2398, d=0.01)
        self.mc.shape_param.set(self.types, diameter=diam)
        with self.assertRaises(RuntimeError):
            self.updater = hpmc.update.alchemy(mc=self.mc, move_ratio=0.1, seed=3832765, period=1, nselect=1);
            hoomd.run(10, quiet=True);

        fun = shape_move_funciton(mc=self.mc, shape_params=dict(diameter=diam));
        p = {};
        for t in self.types:
            p.update({t: [0.1]});

        with self.assertRaises(RuntimeError):
            self.updater.python_shape_move(callback=fun, params=p, stepsize=0.0, param_ratio=1.0);
            hoomd.run(10, quiet=True);

    def test_sphere_elastic(self):
        # sphere
        diam = 1.0;
        self.mc = hpmc.integrate.sphere(seed=2398, d=0.01)
        self.mc.shape_param.set(self.types, diameter=diam)
        with self.assertRaises(RuntimeError):
            self.updater = hpmc.update.elastic_shape(mc=self.mc, move_ratio=0.1, seed=3832765, stiffness=10.0, reference=dict(diameter=diam))

    def test_ellipsoid(self):
        # ellipsoid
        # a = 0.00038920117896296716, p = 0.2035860456051452
        # d = 0.0014225507698958867, p = 0.19295361127422195
        a = 0.5;
        b = 0.54;
        c = 0.35;
        # self.system = init.create_lattice(hoomd.lattice.sc(a=2.0, type_name='A'), n=2);
        # self.mc = hpmc.integrate.ellipsoid(seed=2398, d=0.0, a=0.0)
        # self.mc.shape_param.set('A', a=a, b=b, c=c)
        # self.run_test(latticep=lattice3d, latticeq=latticeq, k=k, kalt=kalt, q=k*10.0, qalt=kalt*10.0, uein=3.0, snapshot_s=self.snapshot3d_s, eng_check=(eng_check3d+eng_checkq), a = 0.000389, d = 0.001423);
        pass;

    def test_convex_polygon(self):
        # convex_polygon
        # a = 0.001957745443687172, p = 0.19863574351978172
        # d = 0.0017185407622231329, p = 0.2004306126443531
        # self.system = init.create_lattice(hoomd.lattice.hex(a=2.0, type_name='A'), n=2);
        # v = 0.33*np.array([(-1,-1), (1,-1), (1,1), (-1,1)]);
        # self.mc = hpmc.integrate.convex_polygon(seed=2398, d=0.0, a=0.0)
        # self.mc.shape_param.set('A', vertices=v)
        # self.run_test(latticep=lattice2d, latticeq=latticeq, k=k, kalt=kalt, q=k*10.0, qalt=kalt*10.0, uein=1.5, snapshot_s=self.snapshot2d_s, eng_check=(eng_check2d+eng_checkq), a = 0.001958, d = 0.001719);
        pass;

    def test_convex_spheropolygon(self):
        # convex_spheropolygon
        # self.system = init.create_lattice(hoomd.lattice.hex(a=2.0, type_name='A'), n=2);
        # v = 0.33*np.array([(-1,-1), (1,-1), (1,1), (-1,1)]);
        # r = 0.1234;
        # self.mc = hpmc.integrate.convex_spheropolygon(seed=2398, d=0.0, a=0.0)
        # self.mc.shape_param.set('A', vertices=v, sweep_radius=r)
        # diff = (np.array(v) - np.array(self.mc.shape_param['A'].vertices)).flatten();
        # self.run_test(latticep=lattice2d, latticeq=latticeq, k=k, kalt=kalt, q=k*10.0, qalt=kalt*10.0, uein=None, snapshot_s=self.snapshot2d_s, eng_check=(eng_check2d+eng_checkq));
        pass;

    def test_simple_polygon(self):
        # simple_polygon
        # self.system = init.create_lattice(hoomd.lattice.hex(a=2.0, type_name='A'), n=2);
        # v = 0.33*np.array([(-1,-1), (1,-1), (1,1), (-1,1)]);
        # self.mc = hpmc.integrate.simple_polygon(seed=2398, d=0.0, a=0.0)
        # self.mc.shape_param.set('A', vertices=v)
        # self.run_test(latticep=lattice2d, latticeq=latticeq, k=k, kalt=kalt, q=k*10.0, qalt=kalt*10.0, uein=None, snapshot_s=self.snapshot2d_s, eng_check=(eng_check2d+eng_checkq));
        pass;

    def test_polyhedron(self):
        # polyhedron
        # self.system = init.create_lattice(hoomd.lattice.sc(a=2.0, type_name='A'), n=2);
        # v = 0.33*np.array([(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5)]);
        # f = [(7, 3, 1, 5), (7, 5, 4, 6), (7, 6, 2, 3), (3, 2, 0, 1), (0, 2, 6, 4), (1, 0, 4, 5)];
        # r = 0.0;
        # self.mc = hpmc.integrate.polyhedron(seed=10);
        # self.mc.shape_param.set('A', vertices=v, faces =f, sweep_radius=r);
        # diff = (np.array(v) - np.array(self.mc.shape_param['A'].vertices)).flatten();
        # self.assertAlmostEqual(diff.dot(diff), 0);
        # diff = (np.array(f) - np.array(self.mc.shape_param['A'].faces)).flatten();
        # self.assertAlmostEqual(diff.dot(diff), 0);
        # self.assertAlmostEqual(self.mc.shape_param['A'].sweep_radius, r);
        pass;

    def test_convex_polyhedron_python(self):
        # convex_polyhedron
        v = 0.33*np.array([(1,1,1), (1,-1,1), (-1,-1,1), (-1,1,1),(1,1,-1), (1,-1,-1), (-1,-1,-1), (-1,1,-1)]);
        self.mc = hpmc.integrate.convex_polyhedron(seed=2398, d=0.0, a=0.0)
        self.mc.shape_param.set(self.types, vertices=v)

        self.updater = hpmc.update.alchemy(mc=self.mc, move_ratio=0.1, seed=3832765, period=1, nselect=1);
        hoomd.run(10, quiet=True);
        fun = shape_move_funciton(mc=self.mc, shape_params=dict(vertices=v));

        p = {};
        for t in self.types:
            p.update({t: [0.1]});

        self.updater.python_shape_move(callback=fun, params=p, stepsize=0.1, param_ratio=1.0);
        hoomd.run(10, quiet=True);

    def test_convex_polyhedron_elastic(self):
        # convex_polyhedron
        v = 0.33*np.array([(1,1,1), (1,-1,1), (-1,-1,1), (-1,1,1),(1,1,-1), (1,-1,-1), (-1,-1,-1), (-1,1,-1)]);
        self.mc = hpmc.integrate.convex_polyhedron(seed=2398, d=0.0, a=0.0)
        self.mc.shape_param.set(self.types, vertices=v)

        self.pos = None; #hoomd.deprecated.dump.pos("shape_updater.pos", period=10)
        self.updater = hpmc.update.elastic_shape(mc=self.mc, move_ratio=0.1, seed=3832765, stiffness=1.0, reference=dict(vertices=v), pos=self.pos, nselect=3)
        self.updater.scale_shear_shape_move(scale_max=0.1, shear_max=0.1);

        hoomd.run(10000, quiet=True);

    def test_convex_spheropolyhedron(self):
        # convex_spheropolyhedron
        # self.system = init.create_lattice(hoomd.lattice.sc(a=2.0, type_name='A'), n=2);
        # v = 0.33*np.array([(1,1,1), (1,-1,1), (-1,-1,1), (-1,1,1),(1,1,-1), (1,-1,-1), (-1,-1,-1), (-1,1,-1)]);
        # r = 0.1234;
        # self.mc = hpmc.integrate.convex_spheropolyhedron(seed=2398, d=0.0, a=0.0)
        # self.mc.shape_param.set('A', vertices=v, sweep_radius=r)
        # self.run_test(latticep=lattice3d, latticeq=latticeq, k=k, kalt=kalt, q=k*10.0, qalt=kalt*10.0, uein=None, snapshot_s=self.snapshot3d_s, eng_check=(eng_check3d+eng_checkq));
        pass;

    def test_faceted_sphere(self):
        # faceted_sphere
        # self.system = init.create_lattice(hoomd.lattice.sc(a=2.0, type_name='A'), n=2);
        # v =  0.33*np.array([(-1,-1,-1),(-1,-1,1),(-1,1,-1),(-1,1,1),(1,-1,-1),(1,-1,1),(1,1,-1),(1,1,1)]);
        # offs = [-1]*6;
        # norms =[(-1,0,0), (1,0,0), (0,1,0,), (0,-1,0), (0,0,1), (0,0,-1)];
        # diam = 1.0;
        # orig = (0,0,0);
        # self.mc = hpmc.integrate.faceted_sphere(seed=10, d=0.0, a=0.0);
        # self.mc.shape_param.set('A', normals=norms,
        #                             offsets=offs,
        #                             vertices=v,
        #                             diameter=diam,
        #                             origin=orig);
        # self.run_test(latticep=lattice3d, latticeq=latticeq, k=k, kalt=kalt, q=k*10.0, qalt=kalt*10.0, uein=None, snapshot_s=self.snapshot3d_s, eng_check=(eng_check3d+eng_checkq));
        pass;

    def test_sphinx(self):
        # sphinx
        # self.system = init.create_lattice(hoomd.lattice.sc(a=2.0, type_name='A'), n=2);
        # cent = [(0,0,0), (0,0,1.15), (0,0,-1.15)]
        # diams = [1,-1.2,-1.2];
        # self.mc = hpmc.integrate.sphinx(seed=10, d=0.0, a=0.0);
        # self.mc.shape_param.set('A', diameters=diams, centers=cent);
        # self.run_test(latticep=lattice3d, latticeq=latticeq, k=k, kalt=kalt, q=k*10.0, qalt=kalt*10.0, uein=None, snapshot_s=self.snapshot3d_s, eng_check=(eng_check3d+eng_checkq));
        pass;

    def test_sphere_union(self):
        # sphere_union
        # self.system = init.create_lattice(hoomd.lattice.sc(a=2.0, type_name='A'), n=2);
        # cent = [(0,0,0), (0,0,0.15), (0,0,-0.15)]
        # diams = [1,1,1];
        # self.mc = hpmc.integrate.sphere_union(seed=10, d=0.0, a=0.0);
        # self.mc.shape_param.set('A', diameters=diams, centers=cent);
        # self.run_test(latticep=lattice3d, latticeq=latticeq, k=k, kalt=kalt, q=k*10.0, qalt=kalt*10.0, uein=None, snapshot_s=self.snapshot3d_s, eng_check=(eng_check3d+eng_checkq));
        pass;

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
