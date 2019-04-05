from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os


# tests for md.update.dynamic_bond
class update_dynamic_bond_tests (unittest.TestCase):
    def setUp(self):
        print
        self.s = init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4]); #target a packing fraction of 0.05
        self.nl = md.nlist.cell()
    # tests basic creation of the updater
    def test(self):

        md.update.dynamic_bond(r_cut=2, nlist=self.nl, bond_type='A', seed=1994)
        run(100);

    def tearDown(self):
        del self.s
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
