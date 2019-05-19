from __future__ import division
from __future__ import print_function

import hoomd
from hoomd import context, data, init
from hoomd import hpmc

import unittest
import os
import numpy as np
import itertools
import sys

context.initialize()

def create_empty(**kwargs):
    snap = data.make_snapshot(**kwargs);
    return init.read_snapshot(snap);

class hpmc_gsd_state(unittest.TestCase):
    def setUp(self):
        self._name = None;
        hexuc = hoomd.lattice.hex(a=2.0);
        system = init.read_snapshot(hexuc.get_snapshot());
        system.replicate(nx=8, ny=8, nz=1);
        self.snapshot2d = system.take_snapshot(particles=True);
        context.initialize();
        bccuc = hoomd.lattice.bcc(a=4.0);
        system = init.read_snapshot(bccuc.get_snapshot());
        system.replicate(nx=4, ny=4, nz=4);
        self.snapshot3d = system.take_snapshot(particles=True);
        # self.snapshot3d = data.make_snapshot(N=20, box=system.box, particle_types=['A']);
        context.initialize();
        v2d = 0.33*np.array([(-1,-1), (1,-1), (1,1), (-1,1)]);
        v2dup = v2d+0.1*np.array([(1,-1), (-1,-1), (1,-1), (1,1)]);
        r = 0.1234;
        rup = 0.3;
        v3d = 0.33*np.array([(1,1,1), (1,-1,1), (-1,-1,1), (-1,1,1),(1,1,-1), (1,-1,-1), (-1,-1,-1), (-1,1,-1)]);
        v3dup = v3d + 0.1*np.array([(-1,1,-1), (1,1,1), (1,-1,-1), (-1,1,-1),(-1,-1,-1), (1,1,1), (-1,1,-1), (-1,1,1)]);
        self.a = 0.5;
        self.d = 0.5;
        self.params = dict(
            sphere=dict(first=dict(diameter=1.5,orientable=False), second=dict(diameter=3.0,orientable=True)),
            ellipsoid=dict(first=dict(a=0.5, b=0.54, c=0.35), second=dict(a=0.98*0.5, b=1.05*0.54, c=1.1*0.35)),
            convex_polygon=dict(first=dict(vertices=v2d), second=dict(vertices=v2dup)),
            convex_spheropolygon=dict(first=dict(vertices=v2d, sweep_radius=r), second=dict(vertices=v2dup, sweep_radius=rup)),
            simple_polygon=dict(first=dict(vertices=v2d), second=dict(vertices=v2dup)),
            convex_polyhedron=dict(first=dict(vertices=v3d), second=dict(vertices=v3dup)),
            convex_spheropolyhedron=dict(first=dict(vertices=v3d, sweep_radius=r), second=dict(vertices=v3dup, sweep_radius=rup))
        )

    def tear_down(self):
        del self.gsd
        del self.mc
        del self.system
        context.initialize()

    def tearDown(self):
        for name in self.params:
            filename = "{}.gsd".format(name)
            if hoomd.comm.get_rank() == 0 and os.path.exists(filename):
                os.remove(filename);

    def run_test_1(self, name, dim):
        filename = "{}.gsd".format(name)
        mc_cls = hoomd.hpmc.integrate.__dict__[name]
        if dim == 3:
            self.system = init.read_snapshot(self.snapshot3d)
        else:
            self.system = init.read_snapshot(self.snapshot2d)
        self.mc = mc_cls(seed=2398)
        self.mc.shape_param.set('A', **self.params[name]['first'])
        self.gsd = hoomd.dump.gsd(filename, group=hoomd.group.all(), period=1, overwrite=True);
        self.gsd.dump_state(self.mc)
        hoomd.run(5);
        self.mc.shape_param.set('A', **self.params[name]['second']);
        self.mc.set_params(a=self.a, d=self.d);
        hoomd.run(5);
        self.gsd.disable();
        self.gsd = None;
        with self.assertRaises(RuntimeError):
            self.mc.restore_state();

    def run_test_2(self, name, dim):
        filename = "{}.gsd".format(name)
        mc_cls = hoomd.hpmc.integrate.__dict__[name]
        self.system = init.read_gsd(filename=filename, frame = 9);
        self.mc = mc_cls(seed=2398, d=0.0)
        self.mc.shape_param.set('A', **self.params[name]['first'])
        self.mc.restore_state();
        self.gsd = None;

    def run_test_3(self, name, dim):
        filename = "{}.gsd".format(name)
        mc_cls = hoomd.hpmc.integrate.__dict__[name]
        self.system = init.read_gsd(filename=filename, frame = 2);
        self.mc = mc_cls(seed=2398, d=0.0, restore_state=True)
        # self.mc.restore_state();
        self.gsd = None;
        # os.remove(filename);

    def test_gsd(self):

        # sphere
        # print("****************************************")
        # print("*               sphere                 *")
        # print("****************************************")
        self.run_test_1('sphere', 3);
        self.tear_down();

        self.run_test_2('sphere', 3);
        self.assertAlmostEqual(self.mc.shape_param['A'].diameter, self.params['sphere']['second']['diameter']);
        self.assertAlmostEqual(self.mc.shape_param['A'].orientable, self.params['sphere']['second']['orientable']);
        self.assertAlmostEqual(self.mc.get_a(), self.a);
        self.assertAlmostEqual(self.mc.get_d(), self.d);
        self.tear_down()

        self.run_test_3('sphere', 3);
        self.assertAlmostEqual(self.mc.shape_param['A'].diameter, self.params['sphere']['first']['diameter']);
        self.assertAlmostEqual(self.mc.shape_param['A'].orientable, self.params['sphere']['first']['orientable']);
        # a is set to garbage value in memory (usually zero) if type is not orientable. Might be better to explicitly
        # set it it to a default value to avoid this test to randomly fail in the future.
        self.assertAlmostEqual(self.mc.get_a(), 0.0);
        self.assertAlmostEqual(self.mc.get_d(), 0.1);
        self.tear_down()

        # ellipsoid
        # print("****************************************")
        # print("*              ellipsoid               *")
        # print("****************************************")
        self.run_test_1('ellipsoid', 3);
        self.tear_down();

        self.run_test_2('ellipsoid', 3);
        self.assertAlmostEqual(self.mc.shape_param['A'].a, self.params['ellipsoid']['second']['a']);
        self.assertAlmostEqual(self.mc.shape_param['A'].b, self.params['ellipsoid']['second']['b']);
        self.assertAlmostEqual(self.mc.shape_param['A'].c, self.params['ellipsoid']['second']['c']);
        self.assertAlmostEqual(self.mc.get_a(), self.a);
        self.assertAlmostEqual(self.mc.get_d(), self.d);

        self.tear_down();

        self.run_test_3('ellipsoid', 3);
        self.assertAlmostEqual(self.mc.shape_param['A'].a, self.params['ellipsoid']['first']['a']);
        self.assertAlmostEqual(self.mc.shape_param['A'].b, self.params['ellipsoid']['first']['b']);
        self.assertAlmostEqual(self.mc.shape_param['A'].c, self.params['ellipsoid']['first']['c']);
        self.assertAlmostEqual(self.mc.get_a(), 0.1);
        self.assertAlmostEqual(self.mc.get_d(), 0.1);

        self.tear_down();

        # convex_polygon
        # print("****************************************")
        # print("*           convex_polygon             *")
        # print("****************************************")
        self.run_test_1('convex_polygon', 2);
        self.tear_down();

        self.run_test_2('convex_polygon', 2);
        vup = self.params['convex_polygon']['second']['vertices'];
        diff = (np.array(vup) - np.array(self.mc.shape_param['A'].vertices)).flatten();
        self.assertAlmostEqual(diff.dot(diff), 0);
        self.tear_down();

        self.run_test_3('convex_polygon', 2);
        v = self.params['convex_polygon']['first']['vertices'];
        diff = (np.array(v) - np.array(self.mc.shape_param['A'].vertices)).flatten();
        self.assertAlmostEqual(diff.dot(diff), 0);
        self.tear_down();

        # convex_spheropolygon
        # print("****************************************")
        # print("*        convex_spheropolygon          *")
        # print("****************************************")
        self.run_test_1('convex_spheropolygon', 2);
        self.tear_down();

        self.run_test_2('convex_spheropolygon', 2);
        vup = self.params['convex_spheropolygon']['second']['vertices'];
        rup = self.params['convex_spheropolygon']['second']['sweep_radius'];
        diff = (np.array(vup) - np.array(self.mc.shape_param['A'].vertices)).flatten();
        self.assertAlmostEqual(diff.dot(diff), 0);
        self.assertAlmostEqual(self.mc.shape_param['A'].sweep_radius, rup);
        self.tear_down();

        self.run_test_3('convex_spheropolygon', 2);
        v = self.params['convex_spheropolygon']['first']['vertices'];
        r = self.params['convex_spheropolygon']['first']['sweep_radius'];
        diff = (np.array(v) - np.array(self.mc.shape_param['A'].vertices)).flatten();
        self.assertAlmostEqual(diff.dot(diff), 0);
        self.assertAlmostEqual(self.mc.shape_param['A'].sweep_radius, r);
        self.tear_down();

        #simple_polygon
        # print("****************************************")
        # print("*           simple_polygon             *")
        # print("****************************************")
        self.run_test_1('simple_polygon', 2);
        self.tear_down();

        self.run_test_2('simple_polygon', 2);
        vup = self.params['simple_polygon']['second']['vertices'];
        diff = (np.array(vup) - np.array(self.mc.shape_param['A'].vertices)).flatten();
        self.assertAlmostEqual(diff.dot(diff), 0);
        self.tear_down();

        self.run_test_3('simple_polygon', 2);
        v = self.params['simple_polygon']['first']['vertices'];
        diff = (np.array(v) - np.array(self.mc.shape_param['A'].vertices)).flatten();
        self.assertAlmostEqual(diff.dot(diff), 0);
        self.tear_down();

        # polyhedron
        # print("****************************************")
        # print("*             polyhedron               *")
        # print("****************************************")
        #
        # self.system = init.read_snapshot(self.snapshot3d)
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
        # del self.mc
        # del self.system
        # context.initialize()

        # convex_polyhedron
        # print("****************************************")
        # print("*          convex_polyhedron           *")
        # print("****************************************")
        self.run_test_1('convex_polyhedron', 3);
        self.tear_down();

        self.run_test_2('convex_polyhedron', 3);
        vup = self.params['convex_polyhedron']['second']['vertices'];
        diff = (np.array(vup) - np.array(self.mc.shape_param['A'].vertices)).flatten();
        self.assertAlmostEqual(diff.dot(diff), 0);
        self.tear_down();

        self.run_test_3('convex_polyhedron', 3);
        v = self.params['convex_polyhedron']['first']['vertices'];
        diff = (np.array(v) - np.array(self.mc.shape_param['A'].vertices)).flatten();
        self.assertAlmostEqual(diff.dot(diff), 0);
        self.tear_down();

        # convex_spheropolyhedron
        # print("****************************************")
        # print("*       convex_spheropolyhedron        *")
        # print("****************************************")
        self.run_test_1('convex_spheropolyhedron', 3);
        self.tear_down();

        self.run_test_2('convex_spheropolyhedron', 3);
        vup = self.params['convex_spheropolyhedron']['second']['vertices'];
        rup = self.params['convex_spheropolyhedron']['second']['sweep_radius'];
        diff = (np.array(vup) - np.array(self.mc.shape_param['A'].vertices)).flatten();
        self.assertAlmostEqual(diff.dot(diff), 0);
        self.assertAlmostEqual(self.mc.shape_param['A'].sweep_radius, rup);
        self.tear_down();

        self.run_test_3('convex_spheropolyhedron', 3);
        v = self.params['convex_spheropolyhedron']['first']['vertices'];
        r = self.params['convex_spheropolyhedron']['first']['sweep_radius'];
        diff = (np.array(v) - np.array(self.mc.shape_param['A'].vertices)).flatten();
        self.assertAlmostEqual(diff.dot(diff), 0);
        self.assertAlmostEqual(self.mc.shape_param['A'].sweep_radius, r);
        self.tear_down();

        # faceted_sphere
        # print("****************************************")
        # print("*            faceted_sphere            *")
        # print("****************************************")
        #
        # self.system = init.read_snapshot(self.snapshot3d)
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
        # self.tear_down()
        #
        # # sphinx
        # print("****************************************")
        # print("*               sphinx                 *")
        # print("****************************************")
        #
        # self.system = init.read_snapshot(self.snapshot3d)
        # cent = [(0,0,0), (0,0,1.15), (0,0,-1.15)]
        # diams = [1,-1.2,-1.2];
        # self.mc = hpmc.integrate.sphinx(seed=10, d=0.0, a=0.0);
        # self.mc.shape_param.set('A', diameters=diams, centers=cent);
        # self.run_test(latticep=lattice3d, latticeq=latticeq, k=k, kalt=kalt, q=k*10.0, qalt=kalt*10.0, uein=None, snapshot_s=self.snapshot3d_s, eng_check=(eng_check3d+eng_checkq));
        # self.tear_down()
        #
        # # sphere_union
        # print("****************************************")
        # print("*            sphere_union              *")
        # print("****************************************")
        #
        # self.system = init.read_snapshot(self.snapshot3d)
        # cent = [(0,0,0), (0,0,0.15), (0,0,-0.15)]
        # diams = [1,1,1];
        # self.mc = hpmc.integrate.sphere_union(seed=10, d=0.0, a=0.0);
        # self.mc.shape_param.set('A', diameters=diams, centers=cent);
        # self.run_test(latticep=lattice3d, latticeq=latticeq, k=k, kalt=kalt, q=k*10.0, qalt=kalt*10.0, uein=None, snapshot_s=self.snapshot3d_s, eng_check=(eng_check3d+eng_checkq));
        # self.tear_down()

class hpmc_gsd_check_restore_state(unittest.TestCase):


    def setUp(self):

        context.initialize()
        self.lattice = hoomd.lattice.sc(a=1.5, type_name='A');
        self.system = hoomd.init.create_lattice(self.lattice, n=3);

        self.gsd = hoomd.dump.gsd('init.gsd', period=100, group=hoomd.group.all(), overwrite=True);
        self.gsd.write_restart();

    def tearDown(self):
        del self.lattice
        del self.gsd
        del self.system
        filename = "init.gsd"
        if hoomd.comm.get_rank() == 0 and os.path.exists(filename):
            os.remove(filename);
        context.initialize()

    def test_sphere(self):

        context.initialize()
        self.system = hoomd.init.read_gsd(filename='init.gsd')

        with self.assertRaises(RuntimeError):
            self.mc = hpmc.integrate.sphere(seed=2234, d=0.3, restore_state=True);

    def test_ellipsoid(self):

        context.initialize()
        self.system = hoomd.init.read_gsd(filename='init.gsd')

        with self.assertRaises(RuntimeError):
            self.mc = hpmc.integrate.ellipsoid(seed=2234, d=0.3, a=0.4, restore_state=True);

    def test_convex_polygon(self):

        context.initialize()
        self.system = hoomd.init.read_gsd(filename='init.gsd')

        with self.assertRaises(RuntimeError):
            self.mc = hpmc.integrate.convex_polygon(seed=2234, d=0.3, a=0.4, restore_state=True);

    def test_convex_polyhedron(self):

        context.initialize()
        self.system = hoomd.init.read_gsd(filename='init.gsd')

        with self.assertRaises(RuntimeError):
            self.mc = hpmc.integrate.convex_polyhedron(seed=2234, d=0.3, a=0.4, restore_state=True);

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
