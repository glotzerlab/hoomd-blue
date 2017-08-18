# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import deprecated
import hoomd;
context.initialize()
import unittest
import os
import tempfile

# unit tests for deprecated.dump.xml
class dmp_xml_tests (unittest.TestCase):
    def setUp(self):
        print
        if hoomd.comm.get_rank() == 0:
            tmp = tempfile.mkstemp(suffix='.test.xml')
            self.tmp_file = tmp[1]
        else:
            self.tmp_file = 'invalid'

        self.s = deprecated.init.create_random(N=100, phi_p=0.05);

        context.current.sorter.set_params(grid=8)

    # tests basic creation of the dump
    def test(self):
        deprecated.dump.xml(group=group.all(), filename=self.tmp_file, period=100);
        run(102);

    # tests with phase
    def test(self):
        deprecated.dump.xml(group=group.all(), filename=self.tmp_file, period=100, phase=0);
        run(102);

    # test variable period
    def test_variable(self):
        deprecated.dump.xml(group=group.all(), filename=self.tmp_file, period=lambda n: n*100);
        run(102);

    # test with restart
    def test_restart(self):
        deprecated.dump.xml(group=group.all(), filename=self.tmp_file, period=100, restart=True).write_restart();
        run(102);

    # test set_params
    def test_set_params(self):
        xml = deprecated.dump.xml(group=group.all(), filename=self.tmp_file, period=100);
        xml.set_params(position=True);
        xml.set_params(velocity=True);
        xml.set_params(mass=False);
        xml.set_params(diameter=False);
        xml.set_params(type=True);
        xml.set_params(bond=True);
        xml.set_params(image=True);
        xml.set_params(all=True);
        xml.set_params(angmom=True);

    def test_group(self):
        tag = group.tags(tag_min=0,tag_max=1)

        # make sure topology doesn't go out
        with self.assertRaises(ValueError):
            deprecated.dump.xml(group=tag, filename=self.tmp_file, bond=True)
        with self.assertRaises(ValueError):
            deprecated.dump.xml(group=tag, filename=self.tmp_file, angle=True)
        with self.assertRaises(ValueError):
            deprecated.dump.xml(group=tag, filename=self.tmp_file, dihedral=True)
        with self.assertRaises(ValueError):
            deprecated.dump.xml(group=tag, filename=self.tmp_file, improper=True)
        with self.assertRaises(ValueError):
            deprecated.dump.xml(group=tag, filename=self.tmp_file, constraint=True)

        # now this should proceed okay
        deprecated.dump.xml(group=tag, filename=self.tmp_file, position=True)

    def test_delete_particles(self):
        del self.s.particles[0]
        self.assertEqual(len(self.s.particles), 99)
        deprecated.dump.xml(group=group.all(), filename=self.tmp_file, position=True)

    def tearDown(self):
        if comm.get_rank() == 0:
            os.remove(self.tmp_file)
        del self.s
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
