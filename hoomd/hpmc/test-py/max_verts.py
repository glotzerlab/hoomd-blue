from __future__ import print_function
from __future__ import division
from hoomd import *
from hoomd import hpmc
import hoomd
import numpy
import math
import sys
import os
import unittest
import tempfile

context.initialize()

class convex_polyhedron(unittest.TestCase):
    def setUp(self):
        # setup the MC integration
        snap = data.make_snapshot(N=32, box=data.boxdim(Lx=20, Ly=20, Lz=20, dimensions=3), particle_types=['A', 'B']);
        # no need to initialize particles, we are just testing construction of integrators
        init.read_snapshot(snap);

        if comm.get_rank() == 0:
            tmp = tempfile.mkstemp(suffix='.hpmc-test-sdf');
            self.tmp_file = tmp[1];
        else:
            self.tmp_file = "invalid";

    def test_8(self):
        xmax=0.02
        dx=1e-4
        mc = hpmc.integrate.convex_polyhedron(seed=10, d=0.1, max_verts=8);
        mc.shape_param.set(['A', 'B'], vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        hpmc.analyze.sdf(mc=mc, filename=self.tmp_file, xmax=xmax, dx=dx, navg=800, period=10, phase=0)
        hpmc.compute.free_volume(mc=mc, seed=123, test_type='A', nsample=1000)

        # run
        run(1, quiet=True);

    def test_8_implicit(self):
        xmax=0.02
        dx=1e-4
        mc = hpmc.integrate.convex_polyhedron(seed=10, d=0.1, max_verts=8, implicit=True);
        mc.set_params(nselect=8,nR=3,depletant_type='B')
        mc.shape_param.set(['A', 'B'], vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        hpmc.analyze.sdf(mc=mc, filename=self.tmp_file, xmax=xmax, dx=dx, navg=800, period=10, phase=0)
        hpmc.compute.free_volume(mc=mc, seed=123, test_type='A', nsample=1000)

        # run
        run(1, quiet=True);


    def test_16(self):
        xmax=0.02
        dx=1e-4
        mc = hpmc.integrate.convex_polyhedron(seed=10, d=0.1, max_verts=16);
        mc.shape_param.set(['A', 'B'], vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        hpmc.analyze.sdf(mc=mc, filename=self.tmp_file, xmax=xmax, dx=dx, navg=800, period=10, phase=0)
        hpmc.compute.free_volume(mc=mc, seed=123, test_type='A', nsample=1000)

        # run
        run(1, quiet=True);

    def test_16_implicit(self):
        xmax=0.02
        dx=1e-4
        mc = hpmc.integrate.convex_polyhedron(seed=10, d=0.1, max_verts=16, implicit=True);
        mc.set_params(nselect=8,nR=3,depletant_type='B')
        mc.shape_param.set(['A', 'B'], vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        hpmc.analyze.sdf(mc=mc, filename=self.tmp_file, xmax=xmax, dx=dx, navg=800, period=10, phase=0)
        hpmc.compute.free_volume(mc=mc, seed=123, test_type='A', nsample=1000)

        # run
        run(1, quiet=True);

    def test_32(self):
        xmax=0.02
        dx=1e-4
        mc = hpmc.integrate.convex_polyhedron(seed=10, d=0.1, max_verts=32);
        mc.shape_param.set(['A', 'B'], vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        hpmc.analyze.sdf(mc=mc, filename=self.tmp_file, xmax=xmax, dx=dx, navg=800, period=10, phase=0)
        hpmc.compute.free_volume(mc=mc, seed=123, test_type='A', nsample=1000)

        # run
        run(1, quiet=True);

    def test_32_implicit(self):
        xmax=0.02
        dx=1e-4
        mc = hpmc.integrate.convex_polyhedron(seed=10, d=0.1, max_verts=32, implicit=True);
        mc.set_params(nselect=8,nR=3,depletant_type='B')
        mc.shape_param.set(['A', 'B'], vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        hpmc.analyze.sdf(mc=mc, filename=self.tmp_file, xmax=xmax, dx=dx, navg=800, period=10, phase=0)
        hpmc.compute.free_volume(mc=mc, seed=123, test_type='A', nsample=1000)

        # run
        run(1, quiet=True);

    def test_64(self):
        xmax=0.02
        dx=1e-4
        mc = hpmc.integrate.convex_polyhedron(seed=10, d=0.1, max_verts=64);
        mc.shape_param.set(['A', 'B'], vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        hpmc.analyze.sdf(mc=mc, filename=self.tmp_file, xmax=xmax, dx=dx, navg=800, period=10, phase=0)
        hpmc.compute.free_volume(mc=mc, seed=123, test_type='A', nsample=1000)

        # run
        run(1, quiet=True);

    def test_64_implicit(self):
        xmax=0.02
        dx=1e-4
        mc = hpmc.integrate.convex_polyhedron(seed=10, d=0.1, max_verts=64, implicit=True);
        mc.set_params(nselect=8,nR=3,depletant_type='B')
        mc.shape_param.set(['A', 'B'], vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        hpmc.analyze.sdf(mc=mc, filename=self.tmp_file, xmax=xmax, dx=dx, navg=800, period=10, phase=0)
        hpmc.compute.free_volume(mc=mc, seed=123, test_type='A', nsample=1000)

        # run
        run(1, quiet=True);

    def test_128(self):
        xmax=0.02
        dx=1e-4
        mc = hpmc.integrate.convex_polyhedron(seed=10, d=0.1, max_verts=128);
        mc.shape_param.set(['A', 'B'], vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        hpmc.analyze.sdf(mc=mc, filename=self.tmp_file, xmax=xmax, dx=dx, navg=800, period=10, phase=0)
        hpmc.compute.free_volume(mc=mc, seed=123, test_type='A', nsample=1000)

        # run
        run(1, quiet=True);

    def test_128_implicit(self):
        xmax=0.02
        dx=1e-4
        mc = hpmc.integrate.convex_polyhedron(seed=10, d=0.1, max_verts=128, implicit=True);
        mc.set_params(nselect=8,nR=3,depletant_type='B')
        mc.shape_param.set(['A', 'B'], vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        hpmc.analyze.sdf(mc=mc, filename=self.tmp_file, xmax=xmax, dx=dx, navg=800, period=10, phase=0)
        hpmc.compute.free_volume(mc=mc, seed=123, test_type='A', nsample=1000)

        # run
        run(1, quiet=True);

    def test_129(self):
        self.assertRaises(Exception, hpmc.integrate.convex_polyhedron, seed=10, d=0.1, max_verts=129);

    def tearDown(self):
        context.initialize();

        if comm.get_rank() == 0:
            os.remove(self.tmp_file);


class convex_polyhedron_fl(unittest.TestCase):
    def setUp(self):
        # setup the MC integration
        snap = data.make_snapshot(N=1, box=data.boxdim(Lx=20, Ly=20, Lz=20, dimensions=3), particle_types=['A', 'B']);
        # no need to initialize particles, we are just testing construction of integrators
        init.read_snapshot(snap);

    def test_8(self):
        if hoomd.context.exec_conf.isCUDAEnabled():
            return;
        mc = hpmc.integrate.convex_polyhedron(seed=10, d=0.1, max_verts=8);
        mc.shape_param.set(['A', 'B'], vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        # run
        run(1, quiet=True);

    def test_16(self):
        if hoomd.context.exec_conf.isCUDAEnabled():
            return;
        mc = hpmc.integrate.convex_polyhedron(seed=10, d=0.1, max_verts=16);
        mc.shape_param.set(['A', 'B'], vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        # run
        run(1, quiet=True);

    def test_32(self):
        if hoomd.context.exec_conf.isCUDAEnabled():
            return;
        mc = hpmc.integrate.convex_polyhedron(seed=10, d=0.1, max_verts=32);
        mc.shape_param.set(['A', 'B'], vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        # run
        run(1, quiet=True);

    def test_64(self):
        if hoomd.context.exec_conf.isCUDAEnabled():
            return;
        mc = hpmc.integrate.convex_polyhedron(seed=10, d=0.1, max_verts=64);
        mc.shape_param.set(['A', 'B'], vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        # run
        run(1, quiet=True);

    def test_128(self):
        if hoomd.context.exec_conf.isCUDAEnabled():
            return;
        mc = hpmc.integrate.convex_polyhedron(seed=10, d=0.1, max_verts=128);
        mc.shape_param.set(['A', 'B'], vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        # run
        run(1, quiet=True);

    def tearDown(self):
        context.initialize();

class convex_spheropolyhedron(unittest.TestCase):
    def setUp(self):
        # setup the MC integration
        snap = data.make_snapshot(N=32, box=data.boxdim(Lx=20, Ly=20, Lz=20, dimensions=3), particle_types=['A', 'B']);
        # no need to initialize particles, we are just testing construction of integrators
        init.read_snapshot(snap);

        if comm.get_rank() == 0:
            tmp = tempfile.mkstemp(suffix='.hpmc-test-sdf');
            self.tmp_file = tmp[1];
        else:
            self.tmp_file = "invalid";

    def test_8(self):
        xmax=0.02
        dx=1e-4
        mc = hpmc.integrate.convex_spheropolyhedron(seed=10, d=0.1, max_verts=8);
        mc.shape_param.set(['A', 'B'], vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        hpmc.analyze.sdf(mc=mc, filename=self.tmp_file, xmax=xmax, dx=dx, navg=800, period=10, phase=0)
        hpmc.compute.free_volume(mc=mc, seed=123, test_type='A', nsample=1000)

        # run
        run(1, quiet=True);

    def test_8_implicit(self):
        xmax=0.02
        dx=1e-4
        mc = hpmc.integrate.convex_spheropolyhedron(seed=10, d=0.1, max_verts=8, implicit=True);
        mc.set_params(nselect=8,nR=3,depletant_type='B')
        mc.shape_param.set(['A', 'B'], vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        hpmc.analyze.sdf(mc=mc, filename=self.tmp_file, xmax=xmax, dx=dx, navg=800, period=10, phase=0)
        hpmc.compute.free_volume(mc=mc, seed=123, test_type='A', nsample=1000)

        # run
        run(1, quiet=True);

    def test_16(self):
        xmax=0.02
        dx=1e-4
        mc = hpmc.integrate.convex_spheropolyhedron(seed=10, d=0.1, max_verts=16);
        mc.shape_param.set(['A', 'B'], vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        hpmc.analyze.sdf(mc=mc, filename=self.tmp_file, xmax=xmax, dx=dx, navg=800, period=10, phase=0)
        hpmc.compute.free_volume(mc=mc, seed=123, test_type='A', nsample=1000)

        # run
        run(1, quiet=True);

    def test_16_implicit(self):
        xmax=0.02
        dx=1e-4
        mc = hpmc.integrate.convex_spheropolyhedron(seed=10, d=0.1, max_verts=16, implicit=True);
        mc.set_params(nselect=8,nR=3,depletant_type='B')
        mc.shape_param.set(['A', 'B'], vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        hpmc.analyze.sdf(mc=mc, filename=self.tmp_file, xmax=xmax, dx=dx, navg=800, period=10, phase=0)
        hpmc.compute.free_volume(mc=mc, seed=123, test_type='A', nsample=1000)

        # run
        run(1, quiet=True);

    def test_32(self):
        xmax=0.02
        dx=1e-4
        mc = hpmc.integrate.convex_spheropolyhedron(seed=10, d=0.1, max_verts=32);
        mc.shape_param.set(['A', 'B'], vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        hpmc.analyze.sdf(mc=mc, filename=self.tmp_file, xmax=xmax, dx=dx, navg=800, period=10, phase=0)
        hpmc.compute.free_volume(mc=mc, seed=123, test_type='A', nsample=1000)

        # run
        run(1, quiet=True);

    def test_32_implicit(self):
        xmax=0.02
        dx=1e-4
        mc = hpmc.integrate.convex_spheropolyhedron(seed=10, d=0.1, max_verts=32, implicit=True);
        mc.set_params(nselect=8,nR=3,depletant_type='B')
        mc.shape_param.set(['A', 'B'], vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        hpmc.analyze.sdf(mc=mc, filename=self.tmp_file, xmax=xmax, dx=dx, navg=800, period=10, phase=0)
        hpmc.compute.free_volume(mc=mc, seed=123, test_type='A', nsample=1000)

        # run
        run(1, quiet=True);

    def test_64(self):
        xmax=0.02
        dx=1e-4
        mc = hpmc.integrate.convex_spheropolyhedron(seed=10, d=0.1, max_verts=64);
        mc.shape_param.set(['A', 'B'], vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        hpmc.analyze.sdf(mc=mc, filename=self.tmp_file, xmax=xmax, dx=dx, navg=800, period=10, phase=0)
        hpmc.compute.free_volume(mc=mc, seed=123, test_type='A', nsample=1000)

        # run
        run(1, quiet=True);

    def test_64_implicit(self):
        xmax=0.02
        dx=1e-4
        mc = hpmc.integrate.convex_spheropolyhedron(seed=10, d=0.1, max_verts=64, implicit=True);
        mc.set_params(nselect=8,nR=3,depletant_type='B')
        mc.shape_param.set(['A', 'B'], vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        hpmc.analyze.sdf(mc=mc, filename=self.tmp_file, xmax=xmax, dx=dx, navg=800, period=10, phase=0)
        hpmc.compute.free_volume(mc=mc, seed=123, test_type='A', nsample=1000)

        # run
        run(1, quiet=True);

    def test_128(self):
        xmax=0.02
        dx=1e-4
        mc = hpmc.integrate.convex_spheropolyhedron(seed=10, d=0.1, max_verts=128);
        mc.shape_param.set(['A', 'B'], vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        hpmc.analyze.sdf(mc=mc, filename=self.tmp_file, xmax=xmax, dx=dx, navg=800, period=10, phase=0)
        hpmc.compute.free_volume(mc=mc, seed=123, test_type='A', nsample=1000)

        # run
        run(1, quiet=True);

    def test_128_implicit(self):
        xmax=0.02
        dx=1e-4
        mc = hpmc.integrate.convex_spheropolyhedron(seed=10, d=0.1, max_verts=128, implicit=True);
        mc.set_params(nselect=8,nR=3,depletant_type='B')
        mc.shape_param.set(['A', 'B'], vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        hpmc.analyze.sdf(mc=mc, filename=self.tmp_file, xmax=xmax, dx=dx, navg=800, period=10, phase=0)
        hpmc.compute.free_volume(mc=mc, seed=123, test_type='A', nsample=1000)

        # run
        run(1, quiet=True);

    def test_129(self):
        self.assertRaises(Exception, hpmc.integrate.convex_spheropolyhedron, seed=10, d=0.1, max_verts=129);

    def tearDown(self):
        context.initialize();

        if comm.get_rank() == 0:
            os.remove(self.tmp_file);


class convex_spheropolyhedron_fl(unittest.TestCase):
    def setUp(self):
        # setup the MC integration
        snap = data.make_snapshot(N=1, box=data.boxdim(Lx=20, Ly=20, Lz=20, dimensions=3), particle_types=['A', 'B']);
        # no need to initialize particles, we are just testing construction of integrators
        init.read_snapshot(snap);

    def test_8(self):
        if hoomd.context.exec_conf.isCUDAEnabled():
            return;
        mc = hpmc.integrate.convex_spheropolyhedron(seed=10, d=0.1, max_verts=8);
        mc.shape_param.set(['A', 'B'], vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        # run
        run(1, quiet=True);

    def test_16(self):
        if hoomd.context.exec_conf.isCUDAEnabled():
            return;
        mc = hpmc.integrate.convex_spheropolyhedron(seed=10, d=0.1, max_verts=16);
        mc.shape_param.set(['A', 'B'], vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        # run
        run(1, quiet=True);

    def test_32(self):
        if hoomd.context.exec_conf.isCUDAEnabled():
            return;
        mc = hpmc.integrate.convex_spheropolyhedron(seed=10, d=0.1, max_verts=32);
        mc.shape_param.set(['A', 'B'], vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        # run
        run(1, quiet=True);

    def test_64(self):
        if hoomd.context.exec_conf.isCUDAEnabled():
            return;
        mc = hpmc.integrate.convex_spheropolyhedron(seed=10, d=0.1, max_verts=64);
        mc.shape_param.set(['A', 'B'], vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        # run
        run(1, quiet=True);

    def test_128(self):
        if hoomd.context.exec_conf.isCUDAEnabled():
            return;
        mc = hpmc.integrate.convex_spheropolyhedron(seed=10, d=0.1, max_verts=128);
        mc.shape_param.set(['A', 'B'], vertices=[(-2,-1,-1),
                                               (-2,1,-1),
                                               (-2,-1,1),
                                               (-2,1,1),
                                               (2,-1,-1),
                                               (2,1,-1),
                                               (2,-1,1),
                                               (2,1,1)]);

        # run
        run(1, quiet=True);

    def tearDown(self):
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
