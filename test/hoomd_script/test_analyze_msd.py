# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
context.initialize()
import unittest
import os
import tempfile

# unit tests for analyze.msd
class analyze_msd_tests (unittest.TestCase):
    def setUp(self):
        print
        self.s = init.create_random(N=100, phi_p=0.05);

        sorter.set_params(grid=8)

        if comm.get_rank() == 0:
            tmp = tempfile.mkstemp(suffix='.test.log');
            self.tmp_file = tmp[1];
        else:
            self.tmp_file = "invalid";


    # tests basic creation of the analyzer
    def test(self):
        analyze.msd(period = 10, filename=self.tmp_file, groups=[group.all()]);
        run(100);

    # tests with phase
    def test_phase(self):
        analyze.msd(period = 10, filename=self.tmp_file, groups=[group.all()], phase=0);
        run(100);

    # test variable period
    def test_variable(self):
        analyze.msd(period = lambda n: n*10, filename=self.tmp_file, groups=[group.all()]);
        run(100);

    # test error if no groups defined
    def test_no_gropus(self):
        self.assertRaises(RuntimeError, analyze.msd, period=10, filename=self.tmp_file, groups=[]);

    # test set_params
    def test_set_params(self):
        ana = analyze.msd(period = 10, filename=self.tmp_file, groups=[group.all()]);
        ana.set_params(delimiter = ' ');
        run(100);

    # test behavior upon changing number of particles
    def test_change_num_ptls(self):
        self.s.particles.types.add('B')
        self.s.particles.add('B')
        groupA = group.type('A',update=True)
        groupB = group.type('B',update=True)
        self.assertEqual(len(groupA),100)
        self.assertEqual(len(groupB),1)
        ana_A_ = analyze.msd(period = 10, filename=self.tmp_file, groups=[groupA]);
        self.s.particles.add('B')
        ana_B = analyze.msd(period = 10, filename=self.tmp_file+'_B', groups=[groupB]);
        self.assertRaises(RuntimeError,self.s.particles.add, type='B')
        if comm.get_rank() == 0:
            os.remove(self.tmp_file+'_B');

    def tearDown(self):
        init.reset();
        if comm.get_rank() == 0:
            os.remove(self.tmp_file);

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
