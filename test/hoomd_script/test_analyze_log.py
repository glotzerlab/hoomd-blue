# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import hoomd_script;
context.initialize()
import unittest
import os
import tempfile

# unit tests for analyze.log
class analyze_log_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);

        sorter.set_params(grid=8)

        if comm.get_rank() == 0:
            tmp = tempfile.mkstemp(suffix='.test.log');
            self.tmp_file = tmp[1];
        else:
            self.tmp_file = "invalid";

    # tests basic creation of the analyzer
    def test(self):
        analyze.log(quantities = ['test1', 'test2', 'test3'], period = 10, filename=self.tmp_file);
        run(100);

    # tests with phase
    def test_phase(self):
        analyze.log(quantities = ['test1', 'test2', 'test3'], period = 10, filename=self.tmp_file, phase=0);
        run(100);

    # test set_params
    def test_set_params(self):
        ana = analyze.log(quantities = ['test1', 'test2', 'test3'], period = 10, filename=self.tmp_file);
        ana.set_params(quantities = ['test1']);
        run(100);
        ana.set_params(delimiter = ' ');
        run(100);
        ana.set_params(quantities = ['test2', 'test3'], delimiter=',')
        run(100);
        ana.set_params(quantities = [u'test4', u'test5'], delimiter=',')
        run(100);

    # test variable period
    def test_variable(self):
        ana = analyze.log(quantities = ['test1', 'test2', 'test3'], period = lambda n: n*10, filename=self.tmp_file);
        run(100);

    # test the initialization checks
    def test_init_checks(self):
        ana = analyze.log(quantities = ['test1', 'test2', 'test3'], period = 10, filename=self.tmp_file);
        ana.cpp_analyzer = None;

        self.assertRaises(RuntimeError, ana.enable);
        self.assertRaises(RuntimeError, ana.disable);

    def tearDown(self):
        context.initialize();
        if (comm.get_rank()==0):
            os.remove(self.tmp_file);


# test analyze.log with query
class analyze_log_query_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.005);
        self.pair = pair.lj(r_cut=2.5)
        self.pair.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
        integrate.mode_standard(dt=0.005);
        integrate.langevin(group.all(), seed=1, T=1.0);

        sorter.set_params(grid=8)

    # tests query with no output file
    def test(self):
        log = analyze.log(quantities = ['potential_energy', 'kinetic_energy'], period = 10, filename=None);
        run(102);
        t0 = log.query('timestep');
        U0 = log.query('potential_energy');
        K0 = log.query('kinetic_energy');

        run(2);
        t1 = log.query('timestep');
        U1 = log.query('potential_energy');
        K1 = log.query('kinetic_energy');

        self.assertEqual(int(t0), 102)
        self.assertNotEqual(K0, 0);
        self.assertNotEqual(U0, 0);
        self.assertEqual(int(t1), 104)
        self.assertNotEqual(U0, U1);
        self.assertNotEqual(K0, K1);

    # tests basic creation of the analyzer
    def test_with_file(self):
        if comm.get_rank() == 0:
            tmp = tempfile.mkstemp(suffix='.test.log');
            self.tmp_file = tmp[1];
        else:
            self.tmp_file = "invalid";

        log = analyze.log(quantities = ['potential_energy', 'kinetic_energy'], period = 10, filename=self.tmp_file);
        run(11);
        t0 = log.query('timestep');
        U0 = log.query('potential_energy');
        K0 = log.query('kinetic_energy');

        run(2);
        t1 = log.query('timestep');
        U1 = log.query('potential_energy');
        K1 = log.query('kinetic_energy');

        self.assertEqual(int(t0), 10)
        self.assertNotEqual(K0, 0);
        self.assertNotEqual(U0, 0);
        self.assertEqual(int(t1), 10)
        self.assertEqual(U0, U1);
        self.assertEqual(K0, K1);

    def tearDown(self):
        self.pair = None;
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

