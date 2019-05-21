from hoomd import *
from hoomd import md;
import unittest
import os
context.initialize()

# tests for md.update.dynamic_bond
class update_dynamic_bond_tests (unittest.TestCase):
    def setUp(self):
        self.system = init.create_lattice(unitcell=lattice.sc(a=1.5), n=10);
        snapshot = self.system.take_snapshot(bonds=True)
        snapshot.particles.types = ['A']
        snapshot.bonds.types = ['test_bond']
        snapshot.bonds.resize(1)
        snapshot.bonds.group[0] = [0, 1]
        self.system.restore_snapshot(snapshot)
        self.nl = md.nlist.cell()

        slj = md.pair.slj(r_cut=2**(1/6), nlist=self.nl);
        slj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0);

        harmonic = md.bond.harmonic()
        harmonic.bond_coeff.set(k=1, r0=1.5, type='test_bond')
        md.integrate.mode_standard(dt=0.005);
        md.integrate.langevin(group=group.all(), kT=0.2, seed=43);


    # tests basic creation of the updater
    def test(self):
        updater = md.update.dynamic_bond(group=group.all(), nlist=self.nl, seed=1994, period=1)
        run(10);

    def test_set_params(self):
        updater = md.update.dynamic_bond(group=group.all(), nlist=self.nl, seed=1994, period=1)
        updater.set_params(r_cut=1.2, bond_type='test_bond', delta_G=8.0, n_polymer=400, nK=40)
        run(10);

    # def test_integrate(self):
    #     md.integrate.mode_standard(dt=0.01)
    #     brownian = md.integrate.brownian(group=group.all(), kT=1, seed=342)
    #     updater = md.update.dynamic_bond(group=group.all(), nlist=self.nl, seed=1994, period=1)
    #     run(100);
    #
    # def tearDown(self):
    #     del self.s
    #     context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
