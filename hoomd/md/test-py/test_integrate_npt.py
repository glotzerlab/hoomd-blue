# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os

# unit tests for md.integrate.npt
class integrate_npt_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_lattice(unitcell=lattice.sc(a=1.0), n=13);
        md.force.constant(fx=0.1, fy=0.1, fz=0.1)
        nl = md.nlist.cell()
        lj = md.pair.lj(r_cut=2.5, nlist = nl)
        lj.pair_coeff.set('A','A', epsilon=1.0, sigma=1.0)
        context.current.sorter.set_params(grid=8)

    # tests basic creation of the integrator
    def test(self):
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.npt(all, kT=1.2, tau=0.5, P=1.0, tauP=0.5);
        run(1);

    def test_mtk_cubic(self):
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.npt(all, kT=1.2, tau=0.5, P=1.0, tauP=0.5);
        run(1);

    def test_mtk_orthorhombic(self):
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.npt(all, kT=1.2, tau=0.5, P=1.0, tauP=0.5, couple="none");
        run(1);

    def test_mtk_tetragonal(self):
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.npt(all, kT=1.2, tau=0.5, P=1.0, tauP=0.5, couple="xy");
        run(1);

    def test_mtk_triclinic(self):
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.npt(all, kT=1.2, tau=0.5, P=1.0, tauP=0.5, couple="none", all=True);
        run(1);

    # test set_params
    def test_set_params(self):
        md.integrate.mode_standard(dt=0.005);
        all = group.all();
        npt = md.integrate.npt(all, kT=1.2, tau=0.5, P=1.0, tauP=0.5);
        npt.set_params(kT=1.3);
        npt.set_params(tau=0.6);
        npt.set_params(P=0.5);
        npt.set_params(tauP=0.6);
        npt.set_params(rescale_all=True)
        run(1);

    # tests randomize_velocities()
    def test(self):
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        npt = md.integrate.npt(all, kT=1.2, tau=0.5, P=1.0, tauP=0.5,couple='none');
        npt.randomize_velocities(seed=42)
        run(1);


    # test w/ empty group
    def test_empty(self):
        # currently cannot catch run-time errors in MPI simulations
        pass
        #empty = group.cuboid(name="empty", xmin=-100, xmax=-100, ymin=-100, ymax=-100, zmin=-100, zmax=-100)
        #mode = md.integrate.mode_standard(dt=0.005);
        #with self.assertRaises(RuntimeError):
        #    nve = md.integrate.npt(group=empty, kT=1.0, P=1.0, tau=0.5, tauP=0.5)
        #    run(1);

    def tearDown(self):
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
