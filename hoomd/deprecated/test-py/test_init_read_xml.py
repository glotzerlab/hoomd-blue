# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import deprecated
import hoomd;
context.initialize()
import unittest
import os
import tempfile

# unit tests for init.read_xml
class init_read_xml_tests (unittest.TestCase):
    def setUp(self):
        print

        if (comm.get_rank()==0):
            tmp = tempfile.mkstemp(suffix='.test.xml');
            self.tmp_file = tmp[1];

            f = open(self.tmp_file, "w");
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

            tmp = tempfile.mkstemp(suffix='.test_out_of_box.xml');
            self.tmp_file2 = tmp[1];

            f = open(self.tmp_file2, "w");
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
        else:
            self.tmp_file = "invalid";
            self.tmp_file2 = "invalid";

    # tests basic creation of the random initializer
    def test(self):
        deprecated.init.read_xml(self.tmp_file);
        self.assert_(hoomd.context.current.system_definition);
        self.assert_(hoomd.context.current.system);
        self.assertEqual(hoomd.context.current.system_definition.getParticleData().getNGlobal(), 3);

    # tests creation with a few more arguments specified
    def test_moreargs(self):
        deprecated.init.read_xml(self.tmp_file, time_step=100);
        self.assert_(hoomd.context.current.system_definition);
        self.assert_(hoomd.context.current.system);
        self.assertEqual(hoomd.context.current.system_definition.getParticleData().getNGlobal(), 3);

    # tests creation with out of box particles
    def test_out_of_box_1(self):
        self.assertRaises(RuntimeError, deprecated.init.read_xml, self.tmp_file2)

    # tests creation with out of box particles
    def test_out_of_box_2(self):
        sys=deprecated.init.read_xml(self.tmp_file2,wrap_coordinates=True)
        self.assert_(hoomd.context.current.system_definition);
        self.assert_(hoomd.context.current.system);
        self.assertEqual(hoomd.context.current.system_definition.getParticleData().getNGlobal(), 3);
        self.assertAlmostEqual(sys.particles[0].position[2],-1,5)

    # test read restart file
    def test_read_restart(self):
        sys=deprecated.init.read_xml(self.tmp_file, self.tmp_file2,wrap_coordinates=True)
        self.assert_(hoomd.context.current.system_definition);
        self.assert_(hoomd.context.current.system);
        self.assertEqual(hoomd.context.current.system_definition.getParticleData().getNGlobal(), 3);
        self.assertAlmostEqual(sys.particles[0].position[2],-1,5)

    # checks for an error if initialized twice
    def test_inittwice(self):
        deprecated.init.read_xml(self.tmp_file);
        self.assertRaises(RuntimeError, deprecated.init.read_xml, self.tmp_file);

    def tearDown(self):
        if (comm.get_rank()==0):
            os.remove(self.tmp_file);
            os.remove(self.tmp_file2);
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
