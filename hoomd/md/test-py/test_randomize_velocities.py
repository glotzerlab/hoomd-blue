# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md
context.initialize()
import unittest
import os
import numpy

# unit tests for velocity randomization
class velocity_randomization_tests (unittest.TestCase):

    def setUp(self):
        print
        option.set_notice_level(0)
        util.quiet_status()

        # Target a packing fraction of 0.05
        # Set an orientation and a moment of inertia for the anisotropic test
        a = 2.1878096788957757
        uc = lattice.unitcell(N=1,
                              a1=[a, 0, 0],
                              a2=[0, a, 0],
                              a3=[0, 0, a],
                              position=[[0, 0, 0]],
                              type_name=['A'],
                              mass=[1],
                              moment_inertia=[[1, 1, 0]],
                              orientation=[[1, 0, 0, 0]])

        # The system has to be reasonably large to ensure that the measured
        # temperature is approximately equal to the desired temperature after
        # removing the center of mass momentum (eliminating drift)
        self.system = init.create_lattice(unitcell=uc, n=[50,50,40])

        # Set one particle to be really massive for validation
        self.system.particles[0].mass = 10000

        md.integrate.mode_standard(dt=0.)
        self.all = group.all()
        self.quantities = ['N',
                           'kinetic_energy',
                           'translational_kinetic_energy',
                           'rotational_kinetic_energy',
                           'temperature',
                           'momentum']
        self.log = analyze.log(filename=None, quantities=self.quantities, period=None)

    def check_quantities(self):
        for q in self.quantities:
            if 'energy' in q:
                print('average', q, self.log.query(q) / self.log.query('N'))
            else:
                print(q, self.log.query(q))
        self.assertAlmostEqual(self.log.query('temperature'), self.kT, 2)
        self.assertAlmostEqual(self.log.query('momentum'), 0, 6)
        #self.assertAlmostEqual(self.log.query('kinetic_energy')

    def test_nvt(self):
        self.kT = 1.0
        integrator = md.integrate.nvt(group=self.all, kT=self.kT, tau=0.5)
        integrator.randomize_velocities(kT=1.0, seed=42)
        run(1)
        self.check_quantities()

    def test_npt(self):
        self.kT = 1.0
        integrator = md.integrate.npt(group=self.all, kT=self.kT, tau=0.5, tauP=1.0, P=2.0)
        integrator.randomize_velocities(kT=self.kT, seed=42)
        run(1)
        self.check_quantities()

    def test_nph(self):
        self.kT = 1.0
        integrator = md.integrate.nph(group=self.all, P=2.0, tauP=1.0)
        integrator.randomize_velocities(kT=self.kT, seed=42)
        run(1)
        self.check_quantities()

    def test_nve(self):
        self.kT = 1.0
        integrator = md.integrate.nve(group=self.all)
        integrator.randomize_velocities(kT=self.kT, seed=42)
        run(1)
        self.check_quantities()

    def test_nvt_anisotropic(self):
        # Sets up an anisotropic pair potential, so that rotational degrees of
        # freedom are given some energy
        nlist = md.nlist.cell()
        dipole = md.pair.dipole(r_cut=2, nlist=nlist)
        dipole.pair_coeff.set('A', 'A', mu=0.0, A=1.0, kappa=1.0)

        self.kT = 1.0
        integrator = md.integrate.nvt(group=self.all, kT=self.kT, tau=0.5)
        integrator.randomize_velocities(kT=self.kT, seed=42)
        run(1)
        self.check_quantities()

    def tearDown(self):
        context.initialize()


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
