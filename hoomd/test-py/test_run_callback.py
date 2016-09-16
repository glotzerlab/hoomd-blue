# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

import hoomd
hoomd.context.initialize()
import unittest

class analyze_callback_tests(unittest.TestCase):

    def setUp(self):
        sysdef = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=2.0),
                                           n=[1,2]);
        self.a = -1;

    def test_simple(self):
        def cb(step):
            self.a = step;

        self.a = -1;
        hoomd.run(10, callback=cb);
        self.assertEqual(self.a, 10);

    def test_period(self):
        def cb(step):
            self.a = step;

        self.a = -1;
        hoomd.run(10, callback=cb, callback_period=7);
        self.assertEqual(self.a, 7);

    def test_cancel(self):
        def cb(step):
            self.a = step;
            if step == 3:
                return -1;
            else:
                return 0;

        self.a = -1;
        hoomd.run(10, callback=cb, callback_period=1);
        self.assertEqual(self.a, 3);

    def tearDown(self):
        hoomd.context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
