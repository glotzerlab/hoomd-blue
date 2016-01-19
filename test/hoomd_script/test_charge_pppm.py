# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# charge.pppm
class charge_pppm_tests (unittest.TestCase):
    def setUp(self):
        print
        self.s = init.create_random(N=100, phi_p=0.05);

        sorter.set_params(grid=8)
        for i in range(0,50):
            self.s.particles[i].charge = -1;

        for i in range(50,100):
            self.s.particles[i].charge = 1;

    # basic test of creation and param setting
    def test(self):
        all = group.all()
        c = charge.pppm(all);
        c.set_params(Nx=16, Ny=16, Nz=16, order=4, rcut=2.0);
        integrate.mode_standard(dt=0.005);
        integrate.nve(all);
        run(100);

        del all
        del c

    # Cannot test pppm multiple times currently because of implementation limitations
    ## test missing coefficients
    #def test_set_missing_coeff(self):
        #all = group.all()
        #c = charge.pppm(all);
        #integrate.mode_standard(dt=0.005);
        #integrate.nve(all);
        #self.assertRaises(RuntimeError, run, 100);

    ## test enable/disable
    #def test_enable_disable(self):
        #all = group.all()
        #c = charge.pppm(all);
        #c.set_params(Nx=16, Ny=16, Nz=16, order=4, rcut=2.0);

        #c.disable(log=True);
        #c.enable();

    def tearDown(self):
        del self.s
        init.reset();

# charge.pppm
class charge_pppm_twoparticle_tests (unittest.TestCase):
    def setUp(self):
        print
        # initialize a two particle system in a triclinic box
        snap = data.make_snapshot(N=2, particle_types=['A'], box = data.boxdim(xy=0.5,xz=0.5,yz=0.5,L=10))

        if comm.get_rank() == 0:
            snap.particles.position[0] = (0,0,0)
            snap.particles.position[1] = (3,3,3)
            snap.particles.charge[0] = 1
            snap.particles.charge[1] = -1

        self.s = init.read_snapshot(snap);

    # basic test of creation and param setting
    def test(self):
        all = group.all()
        c = charge.pppm(all);
        c.set_params(Nx=128, Ny=128, Nz=128, order=3, rcut=2.0);
        log = analyze.log(quantities = ['pppm_energy','pressure_xx','pressure_xy','pressure_xz', 'pressure_yy','pressure_yz', 'pressure_zz'], period = 1, filename=None);
        integrate.mode_standard(dt=0.0);
        integrate.nve(all);
        run(1);

        self.assertAlmostEqual(c.forces[0].force[0], 0.00904953, 5)
        self.assertAlmostEqual(c.forces[0].force[1], 0.0101797, 5)
        self.assertAlmostEqual(c.forces[0].force[2], 0.0124804, 5)
        self.assertAlmostEqual(c.forces[0].energy, 0, 5)
        self.assertAlmostEqual(c.forces[0].virial[0], 0, 5)
        self.assertAlmostEqual(c.forces[0].virial[1], 0, 5)
        self.assertAlmostEqual(c.forces[0].virial[2], 0, 5)
        self.assertAlmostEqual(c.forces[0].virial[3], 0, 5)
        self.assertAlmostEqual(c.forces[0].virial[4], 0, 5)
        self.assertAlmostEqual(c.forces[0].virial[5], 0, 5)

        self.assertAlmostEqual(c.forces[1].force[0], -0.00904953, 5)
        self.assertAlmostEqual(c.forces[1].force[1], -0.0101797, 5)
        self.assertAlmostEqual(c.forces[1].force[2], -0.0124804, 5)
        self.assertAlmostEqual(c.forces[1].energy, 0, 5)
        self.assertAlmostEqual(c.forces[1].virial[0], 0, 5)
        self.assertAlmostEqual(c.forces[1].virial[1], 0, 5)
        self.assertAlmostEqual(c.forces[1].virial[2], 0, 5)
        self.assertAlmostEqual(c.forces[1].virial[3], 0, 5)
        self.assertAlmostEqual(c.forces[1].virial[4], 0, 5)
        self.assertAlmostEqual(c.forces[1].virial[5], 0, 5)

        vol = self.s.box.get_volume()
        self.assertAlmostEqual(log.query('pppm_energy'), -0.2441,4)
        self.assertAlmostEqual(log.query('pressure_xx'), -5.7313404e-05, 2)
        self.assertAlmostEqual(log.query('pressure_xy'), -4.5494677e-05, 2)
        self.assertAlmostEqual(log.query('pressure_xz'), -3.9889249e-05, 2)
        self.assertAlmostEqual(log.query('pressure_yy'), -7.8745142e-05, 2)
        self.assertAlmostEqual(log.query('pressure_yz'), -4.8501155e-05, 2)
        self.assertAlmostEqual(log.query('pressure_zz'), -0.00010732774, 2)

        del all
        del c
        del log

    # Cannot test pppm multiple times currently because of implementation limitations
    ## test missing coefficients
    #def test_set_missing_coeff(self):
        #all = group.all()
        #c = charge.pppm(all);
        #integrate.mode_standard(dt=0.005);
        #integrate.nve(all);
        #self.assertRaises(RuntimeError, run, 100);

    ## test enable/disable
    #def test_enable_disable(self):
        #all = group.all()
        #c = charge.pppm(all);
        #c.set_params(Nx=16, Ny=16, Nz=16, order=4, rcut=2.0);

        #c.disable(log=True);
        #c.enable();

    def tearDown(self):
        del self.s
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
