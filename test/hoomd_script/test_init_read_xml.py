# -*- coding: iso-8859-1 -*-
# $Id$
# $URL$
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# unit tests for init.read_xml
class init_read_xml_tests (unittest.TestCase):
    def setUp(self):
        print
        f = open("test.xml", "w");
        f.write('''<?xml version="1.0" encoding="UTF-8"?>
<hoomd_xml version="1.0">
<configuration time_step="0">
<box units="sigma"  lx="8" ly="8" lz="8"/>
<position units="sigma">
-1 2 3
2 1 -3
3 -2 1
</position>
<type>
A B C
</type>
</configuration>
</hoomd_xml>
''');
        

    # tests basic creation of the random initializer
    def test(self):
        init.read_xml('test.xml');
        self.assert_(globals.system_definition);
        self.assert_(globals.system);
        self.assertEqual(globals.system_definition.getParticleData().getN(), 3);
    
    # tests creation with a few more arugments specified
    def test_moreargs(self):
        init.read_xml('test.xml', time_step=100);
        self.assert_(globals.system_definition);
        self.assert_(globals.system);
        self.assertEqual(globals.system_definition.getParticleData().getN(), 3);
        
    # checks for an error if initialized twice
    def test_inittwice(self):
        init.read_xml('test.xml');
        self.assertRaises(RuntimeError, init.read_xml, 'test.xml');
    
    def tearDown(self):
        os.remove("test.xml");
        init.reset();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
