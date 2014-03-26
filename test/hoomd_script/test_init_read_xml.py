# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# unit tests for init.read_xml
class init_read_xml_tests (unittest.TestCase):
    def setUp(self):
        print
        if (comm.get_rank()==0):
            f = open("test.xml", "w");
            f.write('''<?xml version="1.0" encoding="UTF-8"?>
<hoomd_xml version="1.0">
<configuration time_step="0">
<box lx="8" ly="8" lz="8"/>
<position>
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
            f = open("test_out_of_box.xml", "w");
            f.write('''<?xml version="1.0" encoding="UTF-8"?>
<hoomd_xml version="1.0">
<configuration time_step="0">
<box lx="4" ly="4" lz="4"/>
<position>
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
        self.assertEqual(globals.system_definition.getParticleData().getNGlobal(), 3);

    # tests creation with a few more arugments specified
    def test_moreargs(self):
        init.read_xml('test.xml', time_step=100);
        self.assert_(globals.system_definition);
        self.assert_(globals.system);
        self.assertEqual(globals.system_definition.getParticleData().getNGlobal(), 3);

    # tests creation with out of box particles
    def test_out_of_box_1(self):
        sys=init.read_xml('test_out_of_box.xml')
        self.assert_(globals.system_definition);
        self.assert_(globals.system);
        self.assertEqual(globals.system_definition.getParticleData().getNGlobal(), 3);
        self.assertAlmostEqual(sys.particles[0].position[2],3,5)

    # tests creation with out of box particles
    def test_out_of_box_2(self):
        sys=init.read_xml('test_out_of_box.xml',wrap_coordinates=True)
        self.assert_(globals.system_definition);
        self.assert_(globals.system);
        self.assertEqual(globals.system_definition.getParticleData().getNGlobal(), 3);
        self.assertAlmostEqual(sys.particles[0].position[2],-1,5)



    # checks for an error if initialized twice
    def test_inittwice(self):
        init.read_xml('test.xml');
        self.assertRaises(RuntimeError, init.read_xml, 'test.xml');

    def tearDown(self):
        if (comm.get_rank()==0):
            os.remove("test.xml");
        init.reset();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
