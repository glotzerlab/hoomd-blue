from __future__ import division
from __future__ import print_function

import hoomd
from hoomd import context, data, init
# import hoomd.deprecated
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
        sp = self.mc.shape_class(self.mc, 0)
        return sp.make_param(**self.shape_params);

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
            self.updater = hpmc.update.alchemy(mc=self.mc, move_ratio=1.0, seed=3832765, period=1, nselect=1);
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
            self.updater = hpmc.update.elastic_shape(mc=self.mc, stepsize=0.001, move_ratio=1.0, seed=3832765, stiffness=10.0, reference=dict(diameter=diam), nsweeps=2, param_ratio=0.5)


    def test_convex_polyhedron_python(self):
        # convex_polyhedron
        v = 0.33*np.array([(1,1,1), (1,-1,1), (-1,-1,1), (-1,1,1),(1,1,-1), (1,-1,-1), (-1,-1,-1), (-1,1,-1)]);
        self.mc = hpmc.integrate.convex_polyhedron(seed=2398, d=0.0, a=0.0)
        self.mc.shape_param.set(self.types, vertices=v)

        self.updater = hpmc.update.alchemy(mc=self.mc, move_ratio=1.0, seed=3832765, period=1, nselect=1);
        hoomd.run(10, quiet=True);
        fun = shape_move_funciton(mc=self.mc, shape_params=dict(vertices=v));

        p = {};
        for t in self.types:
            p.update({t: [0.1]});

        self.updater.python_shape_move(callback=fun, params=p, stepsize=0.1, param_ratio=1.0);
        hoomd.run(10, quiet=False);

        self.tearDown();
        self.setUp();

        v = np.random.random((15,3));
        self.mc = hpmc.integrate.convex_polyhedron(seed=2398, d=0.0, a=0.0)
        self.mc.shape_param.set(self.types, vertices=v)

        self.updater = hpmc.update.alchemy(mc=self.mc, move_ratio=1.0, seed=3832765, period=1, nselect=1);
        hoomd.run(10, quiet=True);
        fun = shape_move_funciton(mc=self.mc, shape_params=dict(vertices=v));

        p = {};
        for t in self.types:
            p.update({t: [0.1]});

        self.updater.python_shape_move(callback=fun, params=p, stepsize=0.1, param_ratio=1.0);
        hoomd.run(10, quiet=True);

        self.tearDown();
        self.setUp();

        v = np.random.random((31,3));
        self.mc = hpmc.integrate.convex_polyhedron(seed=2398, d=0.0, a=0.0)
        self.mc.shape_param.set(self.types, vertices=v)

        self.updater = hpmc.update.alchemy(mc=self.mc, move_ratio=1.0, seed=3832765, period=1, nselect=1);
        hoomd.run(10, quiet=True);
        fun = shape_move_funciton(mc=self.mc, shape_params=dict(vertices=v));

        p = {};
        for t in self.types:
            p.update({t: [0.1]});

        self.updater.python_shape_move(callback=fun, params=p, stepsize=0.1, param_ratio=1.0);
        hoomd.run(10, quiet=True);

        self.tearDown();
        self.setUp();

        v = np.random.random((63,3));
        self.mc = hpmc.integrate.convex_polyhedron(seed=2398, d=0.0, a=0.0)
        self.mc.shape_param.set(self.types, vertices=v)

        self.updater = hpmc.update.alchemy(mc=self.mc, move_ratio=1.0, seed=3832765, period=1, nselect=1);
        hoomd.run(10, quiet=True);
        fun = shape_move_funciton(mc=self.mc, shape_params=dict(vertices=v));

        p = {};
        for t in self.types:
            p.update({t: [0.1]});

        self.updater.python_shape_move(callback=fun, params=p, stepsize=0.1, param_ratio=1.0);
        hoomd.run(10, quiet=True);

        self.tearDown();
        self.setUp();

        v = np.random.random((127,3));
        self.mc = hpmc.integrate.convex_polyhedron(seed=2398, d=0.0, a=0.0)
        self.mc.shape_param.set(self.types, vertices=v)

        self.updater = hpmc.update.alchemy(mc=self.mc, move_ratio=1.0, seed=3832765, period=1, nselect=1);
        hoomd.run(10, quiet=True);
        fun = shape_move_funciton(mc=self.mc, shape_params=dict(vertices=v));

        p = {};
        for t in self.types:
            p.update({t: [0.1]});

        self.updater.python_shape_move(callback=fun, params=p, stepsize=0.1, param_ratio=1.0);
        hoomd.run(10, quiet=True);

    def test_convex_polyhedron_vertex(self):
        # convex_polyhedron
        v = 0.33*np.array([(1,1,1), (1,-1,1), (-1,-1,1), (-1,1,1),(1,1,-1), (1,-1,-1), (-1,-1,-1), (-1,1,-1)]);
        self.mc = hpmc.integrate.convex_polyhedron(seed=2398, d=0.0, a=0.0)
        self.mc.shape_param.set(self.types, vertices=v)

        self.updater = hpmc.update.alchemy(mc=self.mc, move_ratio=1.0, seed=3832765, period=1, nselect=1);
        hoomd.run(10, quiet=True);
        self.updater.vertex_shape_move(stepsize=0.01, param_ratio=0.2, volume=1.0);
        hoomd.run(10, quiet=True);

        self.tearDown();
        self.setUp();

        v = np.random.random((15,3));
        self.mc = hpmc.integrate.convex_polyhedron(seed=2398, d=0.0, a=0.0)
        self.mc.shape_param.set(self.types, vertices=v)

        self.updater = hpmc.update.alchemy(mc=self.mc, move_ratio=1.0, seed=3832765, period=1, nselect=1);
        hoomd.run(10, quiet=True);
        self.updater.vertex_shape_move(stepsize=0.01, param_ratio=0.2, volume=1.0);
        hoomd.run(10, quiet=True);

        self.tearDown();
        self.setUp();

        v = np.random.random((31,3));
        self.mc = hpmc.integrate.convex_polyhedron(seed=2398, d=0.0, a=0.0)
        self.mc.shape_param.set(self.types, vertices=v)

        self.updater = hpmc.update.alchemy(mc=self.mc, move_ratio=1.0, seed=3832765, period=1, nselect=1);
        hoomd.run(10, quiet=True);
        self.updater.vertex_shape_move(stepsize=0.01, param_ratio=0.2, volume=1.0);
        hoomd.run(10, quiet=True);

        self.tearDown();
        self.setUp();

        v = np.random.random((63,3));
        self.mc = hpmc.integrate.convex_polyhedron(seed=2398, d=0.0, a=0.0)
        self.mc.shape_param.set(self.types, vertices=v)

        self.updater = hpmc.update.alchemy(mc=self.mc, move_ratio=1.0, seed=3832765, period=1, nselect=1);
        hoomd.run(10, quiet=True);
        self.updater.vertex_shape_move(stepsize=0.01, param_ratio=0.2, volume=1.0);
        hoomd.run(10, quiet=True);

        self.tearDown();
        self.setUp();

        v = np.random.random((127,3));
        self.mc = hpmc.integrate.convex_polyhedron(seed=2398, d=0.0, a=0.0)
        self.mc.shape_param.set(self.types, vertices=v)

        self.updater = hpmc.update.alchemy(mc=self.mc, move_ratio=1.0, seed=3832765, period=1, nselect=1);
        hoomd.run(10, quiet=True);
        self.updater.vertex_shape_move(stepsize=0.01, param_ratio=0.2, volume=1.0);
        hoomd.run(10, quiet=True);

    def test_convex_polyhedron_elastic(self):
        # convex_polyhedron
        v = 0.33*np.array([(1,1,1), (1,-1,1), (-1,-1,1), (-1,1,1),(1,1,-1), (1,-1,-1), (-1,-1,-1), (-1,1,-1)]);
        self.mc = hpmc.integrate.convex_polyhedron(seed=2398, d=0.0, a=0.0)
        self.mc.shape_param.set(self.types, vertices=v)
        self.pos = None; #hoomd.deprecated.dump.pos("shape_updater.pos", period=10)
        self.updater = hpmc.update.elastic_shape(mc=self.mc, stepsize=0.001, move_ratio=1.0, seed=3832765, stiffness=1.0, reference=dict(vertices=v), pos=self.pos, nselect=3, nsweeps=2, param_ratio=0.5)
        # self.updater.scale_shear_shape_move(stepsize=0.1);
        hoomd.run(10, quiet=True);

        self.tearDown();
        self.setUp();
        v = np.random.random((15,3));
        self.mc = hpmc.integrate.convex_polyhedron(seed=2398, d=0.0, a=0.0)
        self.mc.shape_param.set(self.types, vertices=v)
        self.pos = None; #hoomd.deprecated.dump.pos("shape_updater.pos", period=10)
        self.updater = hpmc.update.elastic_shape(mc=self.mc, stepsize=0.001, move_ratio=1.0, seed=3832765, stiffness=1.0, reference=dict(vertices=v), pos=self.pos, nselect=3, nsweeps=2, param_ratio=0.5)
        # self.updater.scale_shear_shape_move(stepsize=0.1);
        hoomd.run(10, quiet=True);

        self.tearDown();
        self.setUp();
        v = np.random.random((31,3));
        self.mc = hpmc.integrate.convex_polyhedron(seed=2398, d=0.0, a=0.0)
        self.mc.shape_param.set(self.types, vertices=v)
        self.pos = None; #hoomd.deprecated.dump.pos("shape_updater.pos", period=10)
        self.updater = hpmc.update.elastic_shape(mc=self.mc, stepsize=0.001, move_ratio=1.0, seed=3832765, stiffness=1.0, reference=dict(vertices=v), pos=self.pos, nselect=3, nsweeps=2, param_ratio=0.5)
        # self.updater.scale_shear_shape_move(stepsize=0.1);
        hoomd.run(10, quiet=True);

        self.tearDown();
        self.setUp();
        v = np.random.random((63,3));
        self.mc = hpmc.integrate.convex_polyhedron(seed=2398, d=0.0, a=0.0)
        self.mc.shape_param.set(self.types, vertices=v)
        self.pos = None; #hoomd.deprecated.dump.pos("shape_updater.pos", period=10)
        self.updater = hpmc.update.elastic_shape(mc=self.mc, stepsize=0.001, move_ratio=1.0, seed=3832765, stiffness=1.0, reference=dict(vertices=v), pos=self.pos, nselect=3, nsweeps=2, param_ratio=0.5)
        # self.updater.scale_shear_shape_move(stepsize=0.1);
        hoomd.run(10, quiet=True);

        self.tearDown();
        self.setUp();
        v = np.random.random((127,3));
        self.mc = hpmc.integrate.convex_polyhedron(seed=2398, d=0.0, a=0.0)
        self.mc.shape_param.set(self.types, vertices=v)
        self.pos = None; #hoomd.deprecated.dump.pos("shape_updater.pos", period=10)
        self.updater = hpmc.update.elastic_shape(mc=self.mc, stepsize=0.001, move_ratio=1.0, seed=3832765, stiffness=1.0, reference=dict(vertices=v), pos=self.pos, nselect=3, nsweeps=2, param_ratio=0.5)
        # self.updater.scale_shear_shape_move(stepsize=0.1);
        hoomd.run(10, quiet=True);


    def test_tuner(self):
        # convex_polyhedron
        v = 0.33*np.array([(1,1,1), (1,-1,1), (-1,-1,1), (-1,1,1),(1,1,-1), (1,-1,-1), (-1,-1,-1), (-1,1,-1)]);
        self.mc = hpmc.integrate.convex_polyhedron(seed=2398, d=0.1, a=0.1)
        self.mc.shape_param.set(self.types, vertices=v)
        self.pos = None; #hoomd.deprecated.dump.pos("shape_updater.pos", period=10)
        self.updater = hpmc.update.elastic_shape(mc=self.mc, stepsize=0.001, move_ratio=1.0, seed=3832765, stiffness=100.0, reference=dict(vertices=v), pos=self.pos, nselect=3, nsweeps=2, param_ratio=0.5)
        # self.updater.scale_shear_shape_move(stepsize=0.1);
        tuner = self.updater.get_tuner();
        for _ in range(10):
            # hoomd.util.quiet_status();
            # print( "stepsizes: ", [ self.updater.get_step_size(i) for i in range(len(self.types))] );
            # print( "acceptance: ", [ self.updater.get_move_acceptance(i) for i in range(len(self.types))] );
            # hoomd.util.unquiet_status();
            hoomd.run(100, quiet=True);
            tuner.update();

    def test_tuner_average(self):
        # convex_polyhedron
        v = 0.33*np.array([(1,1,1), (1,-1,1), (-1,-1,1), (-1,1,1),(1,1,-1), (1,-1,-1), (-1,-1,-1), (-1,1,-1)]);
        self.mc = hpmc.integrate.convex_polyhedron(seed=2398, d=0.1, a=0.1)
        self.mc.shape_param.set(self.types, vertices=v)
        self.pos = None; #hoomd.deprecated.dump.pos("shape_updater.pos", period=10)
        self.updater = hpmc.update.elastic_shape(mc=self.mc, stepsize=0.1, move_ratio=1.0, seed=3832765, stiffness=100.0, reference=dict(vertices=v), pos=self.pos, nselect=3, nsweeps=2, param_ratio=0.5)
        # self.updater.scale_shear_shape_move(stepsize=0.1);
        tuner = self.updater.get_tuner(average=True);
        for _ in range(100):
            print( "stepsizes: ", [ self.updater.get_step_size(i) for i in range(len(self.types))] );
            print( "acceptance: ", [ self.updater.get_move_acceptance(i) for i in range(len(self.types))] );
            hoomd.run(100, quiet=True);
            tuner.update();

    def test_logger(self):
        v = 0.33*np.array([(1,1,1), (1,-1,1), (-1,-1,1), (-1,1,1),(1,1,-1), (1,-1,-1), (-1,-1,-1), (-1,1,-1)]);
        self.mc = hpmc.integrate.convex_polyhedron(seed=2398, d=0.1, a=0.1)
        self.mc.shape_param.set(self.types, vertices=v)
        self.pos = None; #hoomd.deprecated.dump.pos("shape_updater.pos", period=10)
        stiffness = hoomd.variant.linear_interp([(0,100), (100, 80)]);
        self.updater = hpmc.update.elastic_shape(mc=self.mc, stepsize=0.1, move_ratio=1.0, seed=3832765, stiffness=stiffness, reference=dict(vertices=v), pos=self.pos, nselect=3, nsweeps=2, param_ratio=0.5)
        k = hoomd.variant.linear_interp([(0,128), (100, 64)]);
        self.updater.set_stiffness(k)
        hoomd.analyze.log(filename="shape_updater.log", period=10, overwrite=True, quantities=['shape_move_energy', 'shape_move_stiffness', 'shape_move_acceptance_ratio', 'shape_move_particle_volume'])
        hoomd.run(101)

    def test_iq_logging(self):
        v = np.array([[x, y, z] for x, y, z in itertools.product((-0.5, 0.5), repeat=3)])
        self.mc = hpmc.integrate.convex_polyhedron(seed=0, d=0.1, a=0.1)
        self.mc.shape_param.set(self.types, vertices=v)
        self.updater = hpmc.update.alchemy(mc=self.mc, move_ratio=1.0, seed=0,
                period=1, nselect=1);
        self.updater.vertex_shape_move(stepsize=0.0, param_ratio=0.2, volume=1.0);
        log1, log2 = 'shape_isoperimetric_quotient_1', 'shape_isoperimetric_quotient'
        iq_logger = hoomd.analyze.log(filename="iq.log", period=1, overwrite=True,
                quantities=[log1, log2])
        hoomd.run(10, quiet=True);
        self.assertEqual(iq_logger.query(log1), iq_logger.query(log2))
        self.assertAlmostEqual(0.5235987756, iq_logger.query(log2))  # cube


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
