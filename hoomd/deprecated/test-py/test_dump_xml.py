# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import deprecated
import hoomd;
context.initialize()
import unittest
import os

# unit tests for deprecated.dump.xml
class dmp_xml_tests (unittest.TestCase):
    def setUp(self):
        print
        deprecated.init.create_random(N=100, phi_p=0.05);

        context.current.sorter.set_params(grid=8)

    # tests basic creation of the dump
    def test(self):
        deprecated.dump.xml(group=group.all(), filename="dump_xml", period=100);
        run(102);

    # tests with phase
    def test(self):
        deprecated.dump.xml(group=group.all(), filename="dump_xml", period=100, phase=0);
        run(102);

    # test variable period
    def test_variable(self):
        deprecated.dump.xml(group=group.all(), filename="dump_xml", period=lambda n: n*100);
        run(102);

    # test with restart
    def test_restart(self):
        deprecated.dump.xml(group=group.all(), filename="restart.xml", period=100, restart=True).write_restart();
        run(102);

    # test set_params
    def test_set_params(self):
        xml = deprecated.dump.xml(group=group.all(), filename="dump_xml", period=100);
        xml.set_params(position=True);
        xml.set_params(velocity=True);
        xml.set_params(mass=False);
        xml.set_params(diameter=False);
        xml.set_params(type=True);
        xml.set_params(bond=True);
        xml.set_params(image=True);
        xml.set_params(all=True);
        xml.set_params(angmom=True);

    def tearDown(self):
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
