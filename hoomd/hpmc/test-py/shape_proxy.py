from __future__ import division
from __future__ import print_function

import hoomd
from hoomd import context, data, init
from hoomd import hpmc

import unittest
import os
import numpy as np
import itertools



def create_empty(**kwargs):
    snap = data.make_snapshot(**kwargs);
    return init.read_snapshot(snap);

class shape_proxy_sanity_checks (unittest.TestCase):
    def test_access(self):
        N=2
        L=10
        context.initialize()
        self.snapshot = data.make_snapshot(N=N, box=data.boxdim(L=L, dimensions=2), particle_types=['A'])

        # sphere
        diam = 1.125;
        self.system = init.read_snapshot(self.snapshot)
        self.mc = hpmc.integrate.sphere(seed=2398, d=0.0)
        self.mc.shape_param.set('A', diameter=diam)
        self.assertAlmostEqual(self.mc.shape_param['A'].diameter, diam);
        del self.mc
        del self.system
        context.initialize()
        # ellipsoid
        a = 1.125;
        b = 0.238;
        c = 2.25;
        self.system = init.read_snapshot(self.snapshot)
        self.mc = hpmc.integrate.ellipsoid(seed=2398, d=0.0)
        self.mc.shape_param.set('A', a=a, b=b, c=c)
        self.assertAlmostEqual(self.mc.shape_param['A'].a, a);
        self.assertAlmostEqual(self.mc.shape_param['A'].b, b);
        self.assertAlmostEqual(self.mc.shape_param['A'].c, c);
        del self.mc
        del self.system
        context.initialize()


        # convex_polygon
        v = [(-1,-1), (1,-1), (1,1), (-1,1)];
        self.system = init.read_snapshot(self.snapshot)
        self.mc = hpmc.integrate.convex_polygon(seed=2398, d=0.1, a=0.1)
        self.mc.shape_param.set('A', vertices=v)
        diff = (np.array(v) - np.array(self.mc.shape_param['A'].vertices)).flatten();
        self.assertAlmostEqual(diff.dot(diff), 0);
        del self.mc
        del self.system
        context.initialize()

        # convex_spheropolygon
        v = [(-1,-1), (1,-1), (1,1), (-1,1)];
        r = 0.1234;
        self.system = init.read_snapshot(self.snapshot)
        self.mc = hpmc.integrate.convex_spheropolygon(seed=2398, d=0.1, a=0.1)
        self.mc.shape_param.set('A', vertices=v, sweep_radius=r)
        diff = (np.array(v) - np.array(self.mc.shape_param['A'].vertices)).flatten();
        self.assertAlmostEqual(diff.dot(diff), 0);
        self.assertAlmostEqual(self.mc.shape_param['A'].sweep_radius, r);
        del self.mc
        del self.system
        context.initialize()

        #simple_polygon
        v = [(-1,-1), (1,-1), (1,1), (-1,1)];
        self.system = init.read_snapshot(self.snapshot)
        self.mc = hpmc.integrate.simple_polygon(seed=2398, d=0.1, a=0.1)
        self.mc.shape_param.set('A', vertices=v)
        diff = (np.array(v) - np.array(self.mc.shape_param['A'].vertices)).flatten();
        self.assertAlmostEqual(diff.dot(diff), 0);
        del self.mc
        del self.system
        context.initialize()

        # polyhedron
        import math
        v = [(-0.5, -0.5, 0), (-0.5, 0.5, 0), (0.5, -0.5, 0), (0.5, 0.5, 0), (0,0, 1.0/math.sqrt(2)),(0,0,-1.0/math.sqrt(2))];
        f = [(0,4,1),(1,4,2),(2,4,3),(3,4,0),(0,5,1),(1,5,2),(2,5,3),(3,5,0)]
        r = 0.0;
        self.system = init.read_snapshot(self.snapshot)
        self.mc = hpmc.integrate.polyhedron(seed=10);
        self.mc.shape_param.set('A', vertices=v, faces =f, sweep_radius=r);
        diff = (np.array(v) - np.array(self.mc.shape_param['A'].vertices)).flatten();
        self.assertAlmostEqual(diff.dot(diff), 0);
        diff = (np.array(f) - np.array(self.mc.shape_param['A'].faces)).flatten();
        self.assertAlmostEqual(diff.dot(diff), 0);
        self.assertAlmostEqual(self.mc.shape_param['A'].sweep_radius, r);
        del self.mc
        del self.system
        context.initialize()

        # convex_polyhedron
        v = [(1,1,1), (1,-1,1), (-1,-1,1), (-1,1,1),(1,1,-1), (1,-1,-1), (-1,-1,-1), (-1,1,-1)];
        self.system = init.read_snapshot(self.snapshot)
        self.mc = hpmc.integrate.convex_polyhedron(seed=2398, d=0.1, a=0.1)
        self.mc.shape_param.set('A', vertices=v)
        diff = (np.array(v) - np.array(self.mc.shape_param['A'].vertices)).flatten();
        self.assertAlmostEqual(diff.dot(diff), 0);
        del self.mc
        del self.system
        context.initialize()

        # convex_spheropolyhedron
        v = [(1,1,1), (1,-1,1), (-1,-1,1), (-1,1,1),(1,1,-1), (1,-1,-1), (-1,-1,-1), (-1,1,-1)];
        r = 0.1234;
        self.system = init.read_snapshot(self.snapshot)
        self.mc = hpmc.integrate.convex_spheropolyhedron(seed=2398, d=0.1, a=0.1)
        self.mc.shape_param.set('A', vertices=v, sweep_radius=r)
        diff = (np.array(v) - np.array(self.mc.shape_param['A'].vertices)).flatten();
        self.assertAlmostEqual(diff.dot(diff), 0);
        self.assertAlmostEqual(self.mc.shape_param['A'].sweep_radius, r);
        del self.mc
        del self.system
        context.initialize()

        # faceted_sphere
        v =  [(-1,-1,-1),(-1,-1,1),(-1,1,-1),(-1,1,1),(1,-1,-1),(1,-1,1),(1,1,-1),(1,1,1)];
        offs = [-1]*6;
        norms =[(-1,0,0), (1,0,0), (0,1,0,), (0,-1,0), (0,0,1), (0,0,-1)];
        diam = 2;
        orig = (0,0,0);
        self.system = init.read_snapshot(self.snapshot)
        self.mc = hpmc.integrate.faceted_sphere(seed=10);
        self.mc.shape_param.set('A', normals=norms,
                                    offsets=offs,
                                    vertices=v,
                                    diameter=diam,
                                    origin=orig);

        diff = (np.array(v) - np.array(self.mc.shape_param['A'].vertices)).flatten();
        self.assertAlmostEqual(diff.dot(diff), 0);
        diff = (np.array(offs) - np.array(self.mc.shape_param['A'].offsets)).flatten();
        self.assertAlmostEqual(diff.dot(diff), 0);
        diff = (np.array(norms) - np.array(self.mc.shape_param['A'].normals)).flatten();
        self.assertAlmostEqual(diff.dot(diff), 0);
        diff = (np.array(orig) - np.array(self.mc.shape_param['A'].origin)).flatten();
        self.assertAlmostEqual(diff.dot(diff), 0);
        self.assertAlmostEqual(self.mc.shape_param['A'].diameter, diam);
        del self.mc
        del self.system
        context.initialize()

        # sphinx
        # GPU Sphinx is not built on most the time
        if not hoomd.context.exec_conf.isCUDAEnabled():
            cent = [(0,0,0), (0,0,1.15), (0,0,-1.15)]
            diams = [2,-2.2,-2.2];
            self.system = init.read_snapshot(self.snapshot)
            self.mc = hpmc.integrate.sphinx(seed=10);
            self.mc.shape_param.set('A', diameters=diams, centers=cent);
            diff = (np.array(cent) - np.array(self.mc.shape_param['A'].centers)).flatten();
            self.assertAlmostEqual(diff.dot(diff), 0);
            diff = (np.array(diams) - np.array(self.mc.shape_param['A'].diameters)).flatten();
            self.assertAlmostEqual(diff.dot(diff), 0);
            del self.mc
            del self.system
            context.initialize()

        # sphere_union
        cent = [(0,0,0), (0,0,1.15), (0,0,-1.15)]
        diams = [2,2.2,1.75];
        self.system = init.read_snapshot(self.snapshot)
        self.mc = hpmc.integrate.sphere_union(seed=10);
        self.mc.shape_param.set('A', diameters=diams, centers=cent);
        diff = (np.array(cent) - np.array(self.mc.shape_param['A'].centers)).flatten();
        self.assertAlmostEqual(diff.dot(diff), 0);
        for i,m in enumerate(self.mc.shape_param['A'].members):
            self.assertAlmostEqual(m.diameter, diams[i]);
        del self.mc
        del self.system
        del self.snapshot
        context.initialize()

    def test_ensurelist(self):
        enlist = hpmc.data._param.ensure_list;
        li = [1,2,3];
        self.assertEqual(enlist(li) == li , True);
        li = ["1","2","3"];
        self.assertEqual(enlist(li) == li , True);
        li = [["1"],["2"],["3"]];
        self.assertEqual(enlist(li) == li , True);
        li = [[(1,2,3), (1,2,4)],[(1,6,3), (7,2,4)],[(1,9,3), (11,121,141)]];
        lit = [[[1,2,3], [1,2,4]],[[1,6,3], [7,2,4]],[[1,9,3], [11,121,141]]];
        self.assertEqual(enlist(li) == lit , True);
        li =[
                [np.array([1,2,3]), np.array([1,2,4])],
                [np.array([1,6,3]), np.array([7,2,4])],
                [np.array([1,9,3]), np.array([11,121,141])]
            ];
        self.assertEqual(enlist(li) == lit , True);
        li = ['', '', ''];
        self.assertEqual(enlist(li) == li , True);
        li = 'test'
        self.assertEqual(enlist(li) == li , True);
        li = ''
        self.assertEqual(enlist(li) == li , True);
        li = 1
        self.assertEqual(enlist(li) == li , True);
        li = 10.0
        self.assertEqual(enlist(li) == li , True);
        li = []
        self.assertEqual(enlist(li) == li , True);


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
