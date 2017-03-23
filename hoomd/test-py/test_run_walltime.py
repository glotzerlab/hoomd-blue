# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

import hoomd
hoomd.context.initialize()
import unittest
import os

class analyze_callback_tests(unittest.TestCase):

    def setUp(self):
        sysdef = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=2.0),
                                           n=[1,2]);
        self.a = -1;

    def test_walltime_exception(self):
        os.environ['HOOMD_WALLTIME_STOP'] = "0"
        self.assertRaises(hoomd.WalltimeLimitReached, hoomd.run, 10);

    def tearDown(self):
        hoomd.context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
