# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# tests for update.rescale_temp
class update_rescale_temp_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);
        import __main__;
        __main__.sorter.set_params(grid=8)

    # tests basic creation of the updater
    def test(self):
        update.rescale_temp(T=1.0)
        run(100);

    # test variable periods
    def test_variable(self):
        update.rescale_temp(T=1.0, period=lambda n: n*10)
        run(100);

    # test enable/disable
    def test_enable_disable(self):
        upd = update.rescale_temp(T=1.0)
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
        upd = update.rescale_temp(T=1.0)
        upd.set_period(10);
        upd.disable();
        self.assertEqual(10, upd.prev_period);
        upd.set_period(50);
        self.assertEqual(50, upd.prev_period);
        upd.enable();

    # test set_params
    def test_set_params(self):
        upd = update.rescale_temp(T=1.0);
        upd.set_params(T=1.2);

    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
