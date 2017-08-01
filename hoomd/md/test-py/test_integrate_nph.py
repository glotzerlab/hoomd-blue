# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os

# unit tests for md.integrate.nph
class integrate_nph_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4]); #target a packing fraction of 0.05
        md.force.constant(fx=0.1, fy=0.1, fz=0.1)

        context.current.sorter.set_params(grid=8)

    # tests basic creation of the integrator
    def test(self):
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nph(group=all, tau=0.5, P=1.0, tauP=0.5);
        run(100);

    def test_mtk_cubic(self):
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nph(group=all, tau=0.5, P=1.0, tauP=0.5);
        run(100);

    def test_mtk_orthorhombic(self):
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nph(group=all, tau=0.5, P=1.0, tauP=0.5, couple="none");
        run(100);

    def test_mtk_tetragonal(self):
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nph(group=all, tau=0.5, P=1.0, tauP=0.5, couple="xy");
        run(100);

    def test_mtk_triclinic(self):
        all = group.all();
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nph(group=all, tau=0.5, P=1.0, tauP=0.5, couple="none", all=True);
        run(100);

    # test set_params
    def test_set_params(self):
        md.integrate.mode_standard(dt=0.005);
        all = group.all();
        nph = md.integrate.nph(group=all, tau=0.5, P=1.0, tauP=0.5);
        nph.set_params(kT=1.3);
        nph.set_params(tau=0.6);
        nph.set_params(P=0.5);
        nph.set_params(tauP=0.6);
        run(100);

    # test w/ empty group
    def test_empty_mtk(self):
        # currently cannot catch run-time errors in MPI simulations
        pass

        #empty = group.cuboid(name="empty", xmin=-100, xmax=-100, ymin=-100, ymax=-100, zmin=-100, zmax=-100)
        #mode = md.integrate.mode_standard(dt=0.005);
        #with self.assertRaises(RuntimeError):
        #    nph = md.integrate.nph(group=empty, P=1.0, tau=0.5, tauP=0.5)
        #    run(1);

    def tearDown(self):
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
