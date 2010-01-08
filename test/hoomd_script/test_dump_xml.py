# -*- coding: iso-8859-1 -*-
# $Id$
# $URL$
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# unit tests for dump.xml
class dmp_xml_tests (unittest.TestCase):
    def setUp(self):
        print
        init.create_random(N=100, phi_p=0.05);

    # tests basic creation of the dump
    def test(self):
        dump.xml(filename="dump_xml", period=100);

    # test variable period
    def test_variable(self):
        dump.xml(filename="dump_xml", period=lambda n: n*100);
    
    # test set_params
    def test_set_params(self):
        xml = dump.xml(filename="dump_xml", period=100);
        xml.set_params(position=True);
        xml.set_params(velocity=True);
        xml.set_params(mass=False);
        xml.set_params(diameter=False);
        xml.set_params(type=True);
        xml.set_params(wall=True);
        xml.set_params(bond=True);
        xml.set_params(image=True);
        xml.set_params(all=True);
    
    def tearDown(self):
        init.reset();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

