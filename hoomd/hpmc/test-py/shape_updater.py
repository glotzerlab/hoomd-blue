from __future__ import division
from __future__ import print_function

import hoomd
import hoomd.deprecated
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

class shape_updater_test(unittest.TestCase):
    def tear_down(self):
        del self.mc
        del self.system
        del self.lattice
        del self.remove_drift
        context.initialize()

    def run_test(self):
        pass;


    def test_lattice(self):
        N=2;
        self.system = init.read_snapshot(hexuc.get_snapshot());
        self.system.replicate(nx=8, ny=8, nz=1);

        self.snapshot2d = self.system.take_snapshot(particles=True); #data.make_snapshot(N=N, box=data.boxdim(L=L, dimensions=3), particle_types=['A'])
        lattice2d = [];
        if hoomd.comm.get_rank() == 0:
            lattice2d = self.snapshot2d.particles.position[:];

        self.snapshot2d_s = data.make_snapshot(N=N, box=self.system.box, particle_types=['A']);
        if hoomd.comm.get_rank() == 0:
            self.snapshot2d_s.particles.position[:] = self.snapshot2d.particles.position[:]+dx2d;
            self.snapshot2d_s.particles.orientation[:] = np.array([dq for _ in range(N)]);
        del self.system
        context.initialize();

        bccuc = hoomd.lattice.bcc(a=2.0);
        self.system = init.read_snapshot(bccuc.get_snapshot());
        self.system.replicate(nx=4, ny=4, nz=4);
        self.snapshot3d = self.system.take_snapshot(particles=True); #data.make_snapshot(N=N, box=data.boxdim(L=L, dimensions=3), particle_types=['A'])
        lattice3d = [];
        if hoomd.comm.get_rank() == 0:
            lattice3d = self.snapshot3d.particles.position[:];
        self.snapshot3d_s = data.make_snapshot(N=N, box=self.system.box, particle_types=['A']);
        if hoomd.comm.get_rank() == 0:
            self.snapshot3d_s.particles.position[:] = self.snapshot3d.particles.position[:]+dx3d;
            self.snapshot3d_s.particles.orientation[:] = np.array([dq for _ in range(N)]);
        del self.system
        context.initialize();

        # sphere
        print("****************************************")
        print("*               sphere                 *")
        print("****************************************")
        # d = 0.0014284726343172743, p = 0.20123046875
        uein = 1.5 # kT
        diam = 1.0;
        self.system = init.read_snapshot(self.snapshot3d)
        self.mc = hpmc.integrate.sphere(seed=2398, d=0.0)
        self.mc.shape_param.set('A', diameter=diam)
        self.updater = hpmc.update.shape_updater();
        self.tear_down()

        # ellipsoid
        print("****************************************")
        print("*              ellipsoid               *")
        print("****************************************")
        # a = 0.00038920117896296716, p = 0.2035860456051452
        # d = 0.0014225507698958867, p = 0.19295361127422195
        a = 0.5;
        b = 0.54;
        c = 0.35;
        uein = 3.0 # kT
        self.system = init.read_snapshot(self.snapshot3d)
        self.mc = hpmc.integrate.ellipsoid(seed=2398, d=0.0, a=0.0)
        self.mc.shape_param.set('A', a=a, b=b, c=c)
        self.run_test(latticep=lattice3d, latticeq=latticeq, k=k, kalt=kalt, q=k*10.0, qalt=kalt*10.0, uein=3.0, snapshot_s=self.snapshot3d_s, eng_check=(eng_check3d+eng_checkq), a = 0.000389, d = 0.001423);
        self.tear_down()

        # convex_polygon
        print("****************************************")
        print("*           convex_polygon             *")
        print("****************************************")
        # a = 0.001957745443687172, p = 0.19863574351978172
        # d = 0.0017185407622231329, p = 0.2004306126443531
        self.system = init.read_snapshot(self.snapshot2d)
        v = 0.33*np.array([(-1,-1), (1,-1), (1,1), (-1,1)]);
        self.mc = hpmc.integrate.convex_polygon(seed=2398, d=0.0, a=0.0)
        self.mc.shape_param.set('A', vertices=v)
        self.run_test(latticep=lattice2d, latticeq=latticeq, k=k, kalt=kalt, q=k*10.0, qalt=kalt*10.0, uein=1.5, snapshot_s=self.snapshot2d_s, eng_check=(eng_check2d+eng_checkq), a = 0.001958, d = 0.001719);
        self.tear_down()


        # convex_spheropolygon
        print("****************************************")
        print("*        convex_spheropolygon          *")
        print("****************************************")

        self.system = init.read_snapshot(self.snapshot2d)
        v = 0.33*np.array([(-1,-1), (1,-1), (1,1), (-1,1)]);
        r = 0.1234;
        self.mc = hpmc.integrate.convex_spheropolygon(seed=2398, d=0.0, a=0.0)
        self.mc.shape_param.set('A', vertices=v, sweep_radius=r)
        diff = (np.array(v) - np.array(self.mc.shape_param['A'].vertices)).flatten();
        self.run_test(latticep=lattice2d, latticeq=latticeq, k=k, kalt=kalt, q=k*10.0, qalt=kalt*10.0, uein=None, snapshot_s=self.snapshot2d_s, eng_check=(eng_check2d+eng_checkq));
        self.tear_down()

        #simple_polygon
        print("****************************************")
        print("*           simple_polygon             *")
        print("****************************************")

        self.system = init.read_snapshot(self.snapshot2d)
        v = 0.33*np.array([(-1,-1), (1,-1), (1,1), (-1,1)]);
        self.mc = hpmc.integrate.simple_polygon(seed=2398, d=0.0, a=0.0)
        self.mc.shape_param.set('A', vertices=v)
        self.run_test(latticep=lattice2d, latticeq=latticeq, k=k, kalt=kalt, q=k*10.0, qalt=kalt*10.0, uein=None, snapshot_s=self.snapshot2d_s, eng_check=(eng_check2d+eng_checkq));
        self.tear_down()

        # polyhedron
        print("****************************************")
        print("*             polyhedron               *")
        print("****************************************")

        self.system = init.read_snapshot(self.snapshot3d)
        v = 0.33*np.array([(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5)]);
        f = [(7, 3, 1, 5), (7, 5, 4, 6), (7, 6, 2, 3), (3, 2, 0, 1), (0, 2, 6, 4), (1, 0, 4, 5)];
        r = 0.0;
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
        print("****************************************")
        print("*          convex_polyhedron           *")
        print("****************************************")
        #a = 0.00038920117896296716, p = 0.2035860456051452
        #d = 0.0014225507698958867, p = 0.19295361127422195
        self.system = init.read_snapshot(self.snapshot3d)
        v = 0.33*np.array([(1,1,1), (1,-1,1), (-1,-1,1), (-1,1,1),(1,1,-1), (1,-1,-1), (-1,-1,-1), (-1,1,-1)]);
        self.mc = hpmc.integrate.convex_polyhedron(seed=2398, d=0.0, a=0.0)
        self.mc.shape_param.set('A', vertices=v)
        self.run_test(latticep=lattice3d, latticeq=latticeq, k=k, kalt=kalt, q=k*10.0, qalt=kalt*10.0, uein=3.0, snapshot_s=self.snapshot3d_s, eng_check=(eng_check3d+eng_checkq), a = 0.0003892, d = 0.00142255);
        self.tear_down()

        # convex_spheropolyhedron
        print("****************************************")
        print("*       convex_spheropolyhedron        *")
        print("****************************************")

        self.system = init.read_snapshot(self.snapshot3d)
        v = 0.33*np.array([(1,1,1), (1,-1,1), (-1,-1,1), (-1,1,1),(1,1,-1), (1,-1,-1), (-1,-1,-1), (-1,1,-1)]);
        r = 0.1234;
        self.mc = hpmc.integrate.convex_spheropolyhedron(seed=2398, d=0.0, a=0.0)
        self.mc.shape_param.set('A', vertices=v, sweep_radius=r)
        self.run_test(latticep=lattice3d, latticeq=latticeq, k=k, kalt=kalt, q=k*10.0, qalt=kalt*10.0, uein=None, snapshot_s=self.snapshot3d_s, eng_check=(eng_check3d+eng_checkq));
        self.tear_down()

        # faceted_sphere
        print("****************************************")
        print("*            faceted_sphere            *")
        print("****************************************")

        self.system = init.read_snapshot(self.snapshot3d)
        v =  0.33*np.array([(-1,-1,-1),(-1,-1,1),(-1,1,-1),(-1,1,1),(1,-1,-1),(1,-1,1),(1,1,-1),(1,1,1)]);
        offs = [-1]*6;
        norms =[(-1,0,0), (1,0,0), (0,1,0,), (0,-1,0), (0,0,1), (0,0,-1)];
        diam = 1.0;
        orig = (0,0,0);
        self.mc = hpmc.integrate.faceted_sphere(seed=10, d=0.0, a=0.0);
        self.mc.shape_param.set('A', normals=norms,
                                    offsets=offs,
                                    vertices=v,
                                    diameter=diam,
                                    origin=orig);
        self.run_test(latticep=lattice3d, latticeq=latticeq, k=k, kalt=kalt, q=k*10.0, qalt=kalt*10.0, uein=None, snapshot_s=self.snapshot3d_s, eng_check=(eng_check3d+eng_checkq));
        self.tear_down()

        # sphinx
        print("****************************************")
        print("*               sphinx                 *")
        print("****************************************")

        self.system = init.read_snapshot(self.snapshot3d)
        cent = [(0,0,0), (0,0,1.15), (0,0,-1.15)]
        diams = [1,-1.2,-1.2];
        self.mc = hpmc.integrate.sphinx(seed=10, d=0.0, a=0.0);
        self.mc.shape_param.set('A', diameters=diams, centers=cent);
        self.run_test(latticep=lattice3d, latticeq=latticeq, k=k, kalt=kalt, q=k*10.0, qalt=kalt*10.0, uein=None, snapshot_s=self.snapshot3d_s, eng_check=(eng_check3d+eng_checkq));
        self.tear_down()

        # sphere_union
        print("****************************************")
        print("*            sphere_union              *")
        print("****************************************")

        self.system = init.read_snapshot(self.snapshot3d)
        cent = [(0,0,0), (0,0,0.15), (0,0,-0.15)]
        diams = [1,1,1];
        self.mc = hpmc.integrate.sphere_union(seed=10, d=0.0, a=0.0);
        self.mc.shape_param.set('A', diameters=diams, centers=cent);
        self.run_test(latticep=lattice3d, latticeq=latticeq, k=k, kalt=kalt, q=k*10.0, qalt=kalt*10.0, uein=None, snapshot_s=self.snapshot3d_s, eng_check=(eng_check3d+eng_checkq));
        self.tear_down()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
