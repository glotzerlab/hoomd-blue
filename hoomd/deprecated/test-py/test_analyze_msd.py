# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

import hoomd;
hoomd.context.initialize()
from hoomd import deprecated
import unittest
import os
import tempfile

# unit tests for deprecated.analyze.msd
class analyze_msd_tests (unittest.TestCase):
    def setUp(self):
        print
        self.s = deprecated.init.create_random(N=100, phi_p=0.05);

        hoomd.context.current.sorter.set_params(grid=8)

        if hoomd.comm.get_rank() == 0:
            tmp = tempfile.mkstemp(suffix='.test.log');
            self.tmp_file = tmp[1];
        else:
            self.tmp_file = "invalid";


    # tests basic creation of the analyzer
    def test(self):
        deprecated.analyze.msd(period = 10, filename=self.tmp_file, groups=[hoomd.group.all()]);
        hoomd.run(100);

    # tests with phase
    def test_phase(self):
        deprecated.analyze.msd(period = 10, filename=self.tmp_file, groups=[hoomd.group.all()], phase=0);
        hoomd.run(100);

    # test variable period
    def test_variable(self):
        deprecated.analyze.msd(period = lambda n: n*10, filename=self.tmp_file, groups=[hoomd.group.all()]);
        hoomd.run(100);

    # test error if no groups defined
    def test_no_gropus(self):
        self.assertRaises(RuntimeError, deprecated.analyze.msd, period=10, filename=self.tmp_file, groups=[]);

    # test set_params
    def test_set_params(self):
        ana = deprecated.analyze.msd(period = 10, filename=self.tmp_file, groups=[hoomd.group.all()]);
        ana.set_params(delimiter = ' ');
        hoomd.run(100);

    # test behavior upon changing number of particles
    def test_change_num_ptls(self):
        self.s.particles.types.add('B')
        self.s.particles.add('B')
        groupA = hoomd.group.type('A',update=True)
        groupB = hoomd.group.type('B',update=True)
        self.assertEqual(len(groupA),100)
        self.assertEqual(len(groupB),1)
        ana_A_ = deprecated.analyze.msd(period = 10, filename=self.tmp_file, groups=[groupA]);
        self.s.particles.add('B')
        ana_B = deprecated.analyze.msd(period = 10, filename=self.tmp_file+'_B', groups=[groupB]);
        self.assertRaises(RuntimeError,self.s.particles.add, type='B')
        if hoomd.comm.get_rank() == 0:
            os.remove(self.tmp_file+'_B');

    def tearDown(self):
        hoomd.context.initialize();
        if hoomd.comm.get_rank() == 0:
            os.remove(self.tmp_file);

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
