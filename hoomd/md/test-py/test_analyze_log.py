# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

import hoomd;
import hoomd.md;
from hoomd import *
hoomd.context.initialize()
import unittest
import os
import tempfile
import numpy

# unit tests for analyze.log
class analyze_log_tests (unittest.TestCase):
    def setUp(self):
        self.system = init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4]);

        hoomd.context.current.sorter.set_params(grid=8)

        if hoomd.context.current.device.comm.rank == 0:
            tmp = tempfile.mkstemp(suffix='.test.log');
            self.tmp_file = tmp[1];
        else:
            self.tmp_file = "invalid";

    # tests basic creation of the analyzer
    def test(self):
        hoomd.analyze.log(quantities = ['test1', 'test2', 'test3'], period = 10, filename=self.tmp_file);
        hoomd.run(100);

    # tests with phase
    def test_phase(self):
        hoomd.analyze.log(quantities = ['test1', 'test2', 'test3'], period = 10, filename=self.tmp_file, phase=0);
        hoomd.run(100);

    # test set_params
    def test_set_params(self):
        ana = hoomd.analyze.log(quantities = ['test1', 'test2', 'test3'], period = 10, filename=self.tmp_file);
        ana.set_params(quantities = ['test1']);
        hoomd.run(100);
        ana.set_params(delimiter = ' ');
        hoomd.run(100);
        ana.set_params(quantities = ['test2', 'test3'], delimiter=',')
        hoomd.run(100);
        ana.set_params(quantities = [u'test4', u'test5'], delimiter=',')
        hoomd.run(100);

    # test variable period
    def test_variable(self):
        ana = hoomd.analyze.log(quantities = ['test1', 'test2', 'test3'], period = lambda n: n*10, filename=self.tmp_file);
        hoomd.run(100);

    # test the initialization checks
    def test_init_checks(self):
        ana = hoomd.analyze.log(quantities = ['test1', 'test2', 'test3'], period = 10, filename=self.tmp_file);
        ana.cpp_analyzer = None;

        self.assertRaises(RuntimeError, ana.enable);
        self.assertRaises(RuntimeError, ana.disable);

    def test_callback(self):
        ana = hoomd.analyze.log(quantities = ['test1', 'test2', 'test3'], period = 10, filename=self.tmp_file);
        ana.register_callback('phi_p', lambda timestep: len(self.system.particles)/self.system.box.get_volume() * math.pi / 4.0)

    def tearDown(self):
        hoomd.context.initialize();
        if (hoomd.context.current.device.comm.rank==0):
            os.remove(self.tmp_file);


# test analyze.log with query
class analyze_log_query_tests (unittest.TestCase):
    def setUp(self):
        init.create_lattice(lattice.sc(a=1.5),n=[8,8,8]); # must be close enough to interact
        nl = hoomd.md.nlist.cell()
        self.pair = hoomd.md.pair.lj(r_cut=2.5, nlist = nl)
        self.pair.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
        hoomd.md.integrate.mode_standard(dt=0.005);
        hoomd.md.integrate.langevin(hoomd.group.all(), seed=1, kT=1.0);

        hoomd.context.current.sorter.set_params(grid=8)

    # tests query with no output file
    def test(self):
        log = hoomd.analyze.log(quantities = ['potential_energy', 'kinetic_energy'], period = 10, filename=None);
        hoomd.run(102);
        t0 = log.query('timestep');
        U0 = log.query('potential_energy');
        K0 = log.query('kinetic_energy');

        hoomd.run(2);
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
        if hoomd.context.current.device.comm.rank == 0:
            tmp = tempfile.mkstemp(suffix='.test.log');
            self.tmp_file = tmp[1];
        else:
            self.tmp_file = "invalid";

        log = hoomd.analyze.log(quantities = ['potential_energy', 'kinetic_energy'], period = 10, filename=self.tmp_file);
        hoomd.run(11);
        t0 = log.query('timestep');
        U0 = log.query('potential_energy');
        K0 = log.query('kinetic_energy');

        hoomd.run(2);
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
        hoomd.context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
