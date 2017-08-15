# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os

# tests for md.update.rescale_temp
class update_rescale_temp_tests (unittest.TestCase):
    def setUp(self):
        print
        self.system = init.create_lattice(lattice.sc(a=3.7411019268182444),n=[20,20,25]); #target a packing fraction of 0.01
        self.nl = md.nlist.cell()
        context.current.sorter.set_params(grid=8)

    # tests basic creation of the updater
    def test(self):
        md.update.rescale_temp(kT=1.0)
        run(100);

    # tests with phase
    def test_phase(self):
        md.update.rescale_temp(kT=1.0, period=10, phase=0)
        run(100);

    # test variable periods
    def test_variable(self):
        md.update.rescale_temp(kT=1.0, period=lambda n: n*10)
        run(100);

    # test enable/disable
    def test_enable_disable(self):
        upd = md.update.rescale_temp(kT=1.0)
        upd.disable();
        self.assert_(not upd.enabled);
        upd.disable();
        self.assert_(not upd.enabled);
        upd.enable();
        self.assert_(upd.enabled);
        upd.enable();
        self.assert_(upd.enabled);

    # test set_period
    def test_set_period(self):
        upd = md.update.rescale_temp(kT=1.0)
        upd.set_period(10);
        upd.disable();
        self.assertEqual(10, upd.prev_period);
        upd.set_period(50);
        self.assertEqual(50, upd.prev_period);
        upd.enable();

    # test set_params
    def test_set_params(self):
        upd = md.update.rescale_temp(kT=1.0);
        upd.set_params(kT=1.2);

    # test functionality with siotropic particles
    def test_aniso(self):
        lj = md.pair.lj(r_cut=3.0, nlist = self.nl);
        lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=0.7);

        integrator = md.integrate.mode_standard(dt=0.005)
        langevin = md.integrate.langevin(kT=1.0,group=group.all(),seed=123)
        log = analyze.log(filename=None,quantities=['temperature'],period=1)
        run(100)

        upd = md.update.rescale_temp(kT=5.0)
        upd.set_period(1);
        # disable integration
        langevin.disable()
        # dummy integrator with zero timestep
        nve =  md.integrate.nve(group=group.all())
        integrator.set_params(dt=0.0)
        run(10)
        del integrator
        del langevin
        del nve
        T = log.query('temperature')
        self.assertAlmostEqual(T, 5.0, 3)


    # test functionality with ansiotropic particles
    def test_aniso(self):
        gb = md.pair.gb(r_cut=3.0, nlist = self.nl);
        gb.pair_coeff.set('A', 'A', epsilon=1.0, lperp=.45, lpar=0.25);

        for p in self.system.particles:
            p.moment_inertia = (1,1,1)

        integrator = md.integrate.mode_standard(dt=0.005)
        langevin = md.integrate.langevin(kT=1.0,group=group.all(),seed=123)
        log = analyze.log(filename=None,quantities=['temperature','rotational_temperature'],period=1)
        run(100)
        upd = md.update.rescale_temp(kT=5.0)
        upd.set_period(1);
        # disable integration
        langevin.disable()
        # dummy integrator with zero timestep
        nve =  md.integrate.nve(group=group.all())
        integrator.set_params(dt=0.0)
        run(10)
        del integrator
        del langevin
        del nve
        T = log.query('temperature')
        self.assertAlmostEqual(T, 5.0, 3)

    def tearDown(self):
        del self.system, self.nl
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
