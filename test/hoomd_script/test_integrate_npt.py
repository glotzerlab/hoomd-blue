# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# unit tests for integrate.npt
class integrate_npt_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);
        force.constant(fx=0.1, fy=0.1, fz=0.1)
        import __main__;
        __main__.sorter.set_params(grid=8)
        
    # tests basic creation of the integrator
    def test(self):
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.npt(all, T=1.2, tau=0.5, P=1.0, tauP=0.5,mtk=False);
        run(100);

    def test_mtk_cubic(self):
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.npt(all, T=1.2, tau=0.5, P=1.0, tauP=0.5);
        run(100);

    def test_mtk_orthorhombic(self):
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.npt(all, T=1.2, tau=0.5, P=1.0, tauP=0.5, mode="orthorhombic");
        run(100);

    def test_mtk_tetragonal(self):
        all = group.all();
        integrate.mode_standard(dt=0.005);
        integrate.npt(all, T=1.2, tau=0.5, P=1.0, tauP=0.5, mode="tetragonal");
        run(100);


    # test set_params
    def test_set_params(self):
        integrate.mode_standard(dt=0.005);
        all = group.all();
        npt = integrate.npt(all, T=1.2, tau=0.5, P=1.0, tauP=0.5,mtk=False);
        npt.set_params(T=1.3);
        npt.set_params(tau=0.6);
        npt.set_params(P=0.5);
        npt.set_params(tauP=0.6);
        npt.set_params(partial_scale=True);
        run(100);

    # test w/ empty group
    def test_empty(self):
        empty = group.cuboid(name="empty", xmin=-100, xmax=-100, ymin=-100, ymax=-100, zmin=-100, zmax=-100)
        mode = integrate.mode_standard(dt=0.005);
        nve = integrate.npt(group=empty, T=1.0, P=1.0, tau=0.5, tauP=0.5,mtk=False)
        run(1);

    # test w/ empty group
    def test_empty_mtk(self):
        empty = group.cuboid(name="empty", xmin=-100, xmax=-100, ymin=-100, ymax=-100, zmin=-100, zmax=-100)
        mode = integrate.mode_standard(dt=0.005);
        nve = integrate.npt(group=empty, T=1.0, P=1.0, tau=0.5, tauP=0.5)
        run(1);
    
    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

