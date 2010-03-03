# -*- coding: iso-8859-1 -*-
# $Id$
# $URL$
# Maintainer: joaander

from hoomd_script import *
import unittest
import os

# group - test grouping commands
class pair_group_tests (unittest.TestCase):
    def setUp(self):
        print
        sysdef = init.create_empty(N=11, box=(4,4,4), n_particle_types=2);
        sysdef.particles[0].position = (0,0,0);
        sysdef.particles[0].type = 'A';
        sysdef.particles[1].position = (1,1,1);
        sysdef.particles[1].type = 'B';
        sysdef.particles[2].position = (1,1,1);
        sysdef.particles[2].type = 'B';
        sysdef.particles[3].position = (-1,-1,-1);
        sysdef.particles[3].type = 'A';
        sysdef.particles[4].position = (-1,-1,-1);
        sysdef.particles[4].type = 'A';
        sysdef.particles[5].position = (2,0,0);
        sysdef.particles[5].type = 'B';
        sysdef.particles[6].position = (0,2,0);
        sysdef.particles[6].type = 'A';
        sysdef.particles[7].position = (0,0,2);
        sysdef.particles[7].type = 'A';
        sysdef.particles[8].position = (-2,0,0);
        sysdef.particles[8].type = 'B';
        sysdef.particles[9].position = (0,-2,0);
        sysdef.particles[9].type = 'B';
        sysdef.particles[10].position = (0,0,-2);
        sysdef.particles[10].type = 'B';

    def test_all(self):
        all = group.all()
        tags = [(x.tag) for x in all]
        self.assertEqual(tags, range(11))
    
    def test_tags(self):
        one = group.tags(5)
        tags = [(x.tag) for x in one]
        self.assertEqual(tags, [5])
        
        part = group.tags(tag_min=5, tag_max=8)
        tags = [(x.tag) for x in part]
        self.assertEqual(tags, range(5,9))
    
    def test_type(self):
        A = group.type(type='A')
        tags = [(x.tag) for x in A]
        self.assertEqual(tags, [0, 3, 4, 6, 7])
        
        B = group.type(type='B')
        tags = [(x.tag) for x in B]
        self.assertEqual(tags, [1, 2, 5, 8, 9, 10])
    
    def test_cuboid_xmin(self):
        g = group.cuboid(name='test', xmin=None)
        tags = [(x.tag) for x in g]
        self.assertEqual(tags, range(11))
        
        g = group.cuboid(name='test', xmin=0.99)
        tags = [(x.tag) for x in g]
        self.assertEqual(tags, [1,2,5])
    
    def test_cuboid_xmax(self):
        g = group.cuboid(name='test', xmax=None)
        tags = [(x.tag) for x in g]
        self.assertEqual(tags, range(11))
        
        g = group.cuboid(name='test', xmax=-0.99)
        tags = [(x.tag) for x in g]
        self.assertEqual(tags, [3,4,8])
    
    def test_cuboid_ymin(self):
        g = group.cuboid(name='test', ymin=None)
        tags = [(x.tag) for x in g]
        self.assertEqual(tags, range(11))
        
        g = group.cuboid(name='test', ymin=0.99)
        tags = [(x.tag) for x in g]
        self.assertEqual(tags, [1,2,6])
    
    def test_cuboid_ymax(self):
        g = group.cuboid(name='test', ymax=None)
        tags = [(x.tag) for x in g]
        self.assertEqual(tags, range(11))
        
        g = group.cuboid(name='test', ymax=-0.99)
        tags = [(x.tag) for x in g]
        self.assertEqual(tags, [3,4,9])
    
    def test_cuboid_zmin(self):
        g = group.cuboid(name='test', zmin=None)
        tags = [(x.tag) for x in g]
        self.assertEqual(tags, range(11))
        
        g = group.cuboid(name='test', zmin=0.99)
        tags = [(x.tag) for x in g]
        self.assertEqual(tags, [1,2,7])
    
    def test_cuboid_zmax(self):
        g = group.cuboid(name='test', zmax=None)
        tags = [(x.tag) for x in g]
        self.assertEqual(tags, range(11))
        
        g = group.cuboid(name='test', zmax=-0.99)
        tags = [(x.tag) for x in g]
        self.assertEqual(tags, [3,4,10])

    def test_union(self):
        A = group.type(type='A')
        B = group.type(type='B')
        
        union = group.union(name='test', a=A,b=B)
        tags = [(x.tag) for x in union]
        self.assertEqual(tags, range(11))
    
    def test_intersection(self):
        A = group.type(type='A')
        B = group.type(type='B')
        all = group.all();

        isectAB = group.intersection(name='test', a=A,b=B)
        tags = [(x.tag) for x in isectAB]
        self.assertEqual(tags, [])
        
        isectAall = group.intersection(name='test', a=A,b=all)
        tags = [(x.tag) for x in isectAall]
        self.assertEqual(tags, [0, 3, 4, 6, 7])
    
    def test_difference(self):
        B = group.type(type='B')
        all = group.all();

        diffBall = group.difference(name='test', a=all,b=B)
        tags = [(x.tag) for x in diffBall]
        self.assertEqual(tags, [0, 3, 4, 6, 7])

    def tearDown(self):
        init.reset();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

