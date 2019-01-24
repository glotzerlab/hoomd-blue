# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

import hoomd;
import hoomd.dem;
hoomd.context.initialize();

import itertools
import unittest

def not_on_mpi(f):
    def noop(*args, **kwargs):
        return;
    if hoomd.comm.get_num_ranks() > 1:
        return noop;
    else:
        return f;

class sphere_sphere(unittest.TestCase):

    def test_potential_wca_2d(self):
        self._test_potential(hoomd.dem.pair.WCA, twoD=True, radius=.5);

    @not_on_mpi
    def test_potential_swca_2d(self):
        self._test_potential(hoomd.dem.pair.SWCA, twoD=True, radius=.5);

    def test_potential_wca_3d(self):
        self._test_potential(hoomd.dem.pair.WCA, twoD=False, radius=.5);

    @not_on_mpi
    def test_potential_swca_3d(self):
        self._test_potential(hoomd.dem.pair.SWCA, twoD=False, radius=.5);

    def _test_potential(self, typ, twoD, **params):
        box = hoomd.data.boxdim(L=80, dimensions=(2 if twoD else 3));
        snap = hoomd.data.make_snapshot(N=4, box=box);
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[0] = (0, 0, 0);
            snap.particles.position[1] = (1, 0, 0);
            snap.particles.position[2] = (0, 2**(1./6), 0);
            snap.particles.position[3] = (1, 2**(1./6), 0);

        system = hoomd.init.read_snapshot(snap);
        nl = hoomd.md.nlist.cell();

        potential = typ(nlist=nl, **params);
        nve = hoomd.md.integrate.nve(group=hoomd.group.all());
        mode = hoomd.md.integrate.mode_standard(dt=0);

        hoomd.run(1);

        for p in system.particles:
            self.assertAlmostEqual(p.net_energy, 0.5);

        potential.disable();

    def setUp(self):
        hoomd.context.initialize();

    def tearDown(self):
        hoomd.comm.barrier();

class spherocylinder_sphere(unittest.TestCase):

    def test_potential_wca_2d(self):
        self._test_potential(hoomd.dem.pair.WCA, twoD=True, radius=.5);

    @not_on_mpi
    def test_potential_swca_2d(self):
        self._test_potential(hoomd.dem.pair.SWCA, twoD=True, radius=.5);

    def test_potential_wca_3d(self):
        self._test_potential(hoomd.dem.pair.WCA, twoD=False, radius=.5);

    @not_on_mpi
    def test_potential_swca_3d(self):
        self._test_potential(hoomd.dem.pair.SWCA, twoD=False, radius=.5);

    def _test_potential(self, typ, twoD, **params):
        box = hoomd.data.boxdim(L=80, dimensions=(2 if twoD else 3));
        snap = hoomd.data.make_snapshot(N=4, box=box);
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[0] = (0, 0, 0);
            snap.particles.position[1] = (1, 0, 0);
            snap.particles.position[2] = (0, 2 + 2**(1./6), 0);
            snap.particles.position[3] = (1, 2 + 2**(1./6), 0);

            snap.particles.typeid[:] = [1, 0, 1, 0];

            snap.particles.types = ['A', 'B'];

        system = hoomd.init.read_snapshot(snap);
        nl = hoomd.md.nlist.cell();

        potential = typ(nlist=nl, **params);
        nve = hoomd.md.integrate.nve(group=hoomd.group.all());
        mode = hoomd.md.integrate.mode_standard(dt=0);

        if twoD:
            vertices = [[0, 1], [0, -1]];
            potential.setParams('A', vertices, center=False);
        else:
            vertices = [[0, 1, 0], [0, -1, 0]];
            faces = [[0, 1]];
            potential.setParams('A', vertices, faces, center=False);

        hoomd.run(1);

        for p in system.particles:
            self.assertAlmostEqual(p.net_energy, 0.5);

        potential.disable();

    def setUp(self):
        hoomd.context.initialize();

    def tearDown(self):
        hoomd.comm.barrier();

class shape_sphere(unittest.TestCase):

    def test_potential_wca_2d(self):
        self._test_potential(hoomd.dem.pair.WCA, twoD=True, radius=.5);

    @not_on_mpi
    def test_potential_swca_2d(self):
        self._test_potential(hoomd.dem.pair.SWCA, twoD=True, radius=.5);

    def test_potential_wca_3d(self):
        self._test_potential(hoomd.dem.pair.WCA, twoD=False, radius=.5);

    @not_on_mpi
    def test_potential_swca_3d(self):
        self._test_potential(hoomd.dem.pair.SWCA, twoD=False, radius=.5);

    def _test_potential(self, typ, twoD, **params):
        box = hoomd.data.boxdim(L=80, dimensions=(2 if twoD else 3));
        snap = hoomd.data.make_snapshot(N=4, box=box);
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[0] = (0, 0, 0);
            snap.particles.position[1] = (3.5, 0, 0);
            snap.particles.position[2] = (0, 5 + 2**(1./6), 0);
            snap.particles.position[3] = (3.5, 5 + 2**(1./6), 0);

            snap.particles.typeid[:] = [1, 0, 1, 0];

            snap.particles.types = ['A', 'B'];

        system = hoomd.init.read_snapshot(snap);
        nl = hoomd.md.nlist.cell();

        potential = typ(nlist=nl, **params);
        nve = hoomd.md.integrate.nve(group=hoomd.group.all());
        mode = hoomd.md.integrate.mode_standard(dt=0);

        if twoD:
            vertices = [[2.5, 2.5], [-2.5, 2.5], [-2.5, -2.5], [2.5, -2.5]];
            potential.setParams('A', vertices, center=False);
        else:
            vertices = [[-2.5, 2.5, 2.5], [-2.5, -2.5, 2.5], [-2.5, -2.5, -2.5], [-2.5, 2.5, -2.5]];
            faces = [[0, 1, 2, 3]];
            potential.setParams('A', vertices, faces, center=False);

        hoomd.run(1);

        for p in system.particles:
            self.assertAlmostEqual(p.net_energy, 0.5);

        potential.disable();

    def setUp(self):
        hoomd.context.initialize();

    def tearDown(self):
        hoomd.comm.barrier();

class shape_shape(unittest.TestCase):

    def test_potential_wca_2d(self):
        self._test_potential(hoomd.dem.pair.WCA, twoD=True, radius=.5);

    @not_on_mpi
    def test_potential_swca_2d(self):
        self._test_potential(hoomd.dem.pair.SWCA, twoD=True, radius=.5);

    def test_potential_wca_3d(self):
        self._test_potential(hoomd.dem.pair.WCA, twoD=False, radius=.5);

    @not_on_mpi
    def test_potential_swca_3d(self):
        self._test_potential(hoomd.dem.pair.SWCA, twoD=False, radius=.5);

    def _test_potential(self, typ, twoD, **params):
        box = hoomd.data.boxdim(L=80, dimensions=(2 if twoD else 3));
        snap = hoomd.data.make_snapshot(N=4, box=box);

        dz = (0 if twoD else 2.5);

        if hoomd.comm.get_rank() == 0:
            snap.particles.position[0] = (0, 0, 0);
            snap.particles.position[1] = (6, 2.5, dz);
            snap.particles.position[2] = (0, 5 + 2**(1./6), 0);
            snap.particles.position[3] = (6, 2.5 + 5 + 2**(1./6), dz);

        system = hoomd.init.read_snapshot(snap);
        nl = hoomd.md.nlist.cell();

        potential = typ(nlist=nl, **params);
        nve = hoomd.md.integrate.nve(group=hoomd.group.all());
        mode = hoomd.md.integrate.mode_standard(dt=0);

        # there is cross-interaction between two of the particles the
        # way they are arranged, so the energies are not all the same
        if twoD:
            vertices = [[2.5, 2.5], [-2.5, 2.5], [-2.5, -2.5], [2.5, -2.5]];
            potential.setParams('A', vertices, center=False);
            expected_energies = [1, 2, 2, 1];
        else:
            # cube of edge length 5
            vertices = list(itertools.product(*(3*[[-2.5,2.5]])));
            faces = [[4, 0, 2, 6],
                     [1, 0, 4, 5],
                     [5, 4, 6, 7],
                     [2, 0, 1, 3],
                     [6, 2, 3, 7],
                     [3, 1, 5, 7]];
            potential.setParams('A', vertices, faces, center=False);
            expected_energies = [2, 4, 4, 2];

        hoomd.run(1);

        for (p, U) in zip(system.particles, expected_energies):
            self.assertAlmostEqual(p.net_energy, U);

        potential.disable();
        del potential;
        del system;

    def setUp(self):
        hoomd.context.initialize();

    def tearDown(self):
        hoomd.comm.barrier();

if __name__ == '__main__':
    unittest.main(argv = ['test_potentials.py', '-v']);
