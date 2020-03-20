# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
import hoomd;
context.initialize()
import unittest
import os
import gc

def create_empty(**kwargs):
    snap = data.make_snapshot(**kwargs);
    return init.read_snapshot(snap);

# group - test grouping commands
class pair_group_tests (unittest.TestCase):
    def setUp(self):
        print
        self.s = create_empty(N=11, box=data.boxdim(L=5), particle_types=['A', 'B']);
        self.s.particles[0].position = (0,0,0);
        self.s.particles[0].type = 'A';
        self.s.particles[1].position = (1,1,1);
        self.s.particles[1].type = 'B';
        self.s.particles[2].position = (1,1,1);
        self.s.particles[2].type = 'B';
        self.s.particles[3].position = (-1,-1,-1);
        self.s.particles[3].type = 'A';
        self.s.particles[4].position = (-1,-1,-1);
        self.s.particles[4].type = 'A';
        self.s.particles[5].position = (2,0,0);
        self.s.particles[5].type = 'B';
        self.s.particles[6].position = (0,2,0);
        self.s.particles[6].type = 'A';
        self.s.particles[7].position = (0,0,2);
        self.s.particles[7].type = 'A';
        self.s.particles[8].position = (-2,0,0);
        self.s.particles[8].type = 'B';
        self.s.particles[9].position = (0,-2,0);
        self.s.particles[9].type = 'B';
        self.s.particles[10].position = (0,0,-2);
        self.s.particles[10].type = 'B';

        context.current.sorter.set_params(grid=8)

    def test_all(self):
        all = group.all()
        tags = [(x.tag) for x in all]
        self.assertEqual(tags, list(range(11)))

        # add a particle
        self.s.particles.add('A')
        tags = [(x.tag) for x in all]
        self.assertEqual(tags, list(range(12)))

    def test_tags(self):
        one = group.tags(5)
        tags = [(x.tag) for x in one]
        self.assertEqual(tags, [5])

        part = group.tags(tag_min=5, tag_max=8)
        tags = [(x.tag) for x in part]
        self.assertEqual(tags, list(range(5,9)))

        part_update = group.tags(tag_min=5, tag_max=8,update=True)
        tags = [(x.tag) for x in part_update]
        self.assertEqual(tags, list(range(5,9)))

        # remove a particle
        del self.s.particles[7]
        tags = [(x.tag) for x in part]
        self.assertEqual(tags, list(range(5,9)))
        tags = [(x.tag) for x in part_update]
        self.assertEqual(tags, [5,6,8])

    def test_type(self):
        A = group.type(type='A')
        A_update = group.type(type='A',update=True)
        tags = [(x.tag) for x in A]
        self.assertEqual(tags, [0, 3, 4, 6, 7])
        tags = [(x.tag) for x in A_update]
        self.assertEqual(tags, [0, 3, 4, 6, 7])

        B = group.type(type='B')
        B_update = group.type(type='B',update=True)
        tags = [(x.tag) for x in B]
        self.assertEqual(tags, [1, 2, 5, 8, 9, 10])

        # remove a particle
        del self.s.particles[7]

        # add a particle
        t = self.s.particles.add('B')
        self.assertEqual(t,7)

        tags = [(x.tag) for x in A]
        self.assertEqual(tags, [0, 3, 4, 6, 7])
        tags = [(x.tag) for x in A_update]
        self.assertEqual(tags, [0, 3, 4, 6])

        tags = [(x.tag) for x in B]
        self.assertEqual(tags, [1, 2, 5, 8, 9, 10])
        tags = [(x.tag) for x in B_update]
        self.assertEqual(tags, [1, 2, 5, 7, 8, 9, 10])


    def test_cuboid_xmin(self):
        g = group.cuboid(name='test', xmin=None)
        tags = [(x.tag) for x in g]
        self.assertEqual(tags, list(range(11)))

        g = group.cuboid(name='test', xmin=0.99)
        tags = [(x.tag) for x in g]
        self.assertEqual(tags, [1,2,5])

    def test_cuboid_xmax(self):
        g = group.cuboid(name='test', xmax=None)
        tags = [(x.tag) for x in g]
        self.assertEqual(tags, list(range(11)))

        g = group.cuboid(name='test', xmax=-0.99)
        tags = [(x.tag) for x in g]
        self.assertEqual(tags, [3,4,8])

    def test_cuboid_ymin(self):
        g = group.cuboid(name='test', ymin=None)
        tags = [(x.tag) for x in g]
        self.assertEqual(tags, list(range(11)))

        g = group.cuboid(name='test', ymin=0.99)
        tags = [(x.tag) for x in g]
        self.assertEqual(tags, [1,2,6])

    def test_cuboid_ymax(self):
        g = group.cuboid(name='test', ymax=None)
        tags = [(x.tag) for x in g]
        self.assertEqual(tags, list(range(11)))

        g = group.cuboid(name='test', ymax=-0.99)
        tags = [(x.tag) for x in g]
        self.assertEqual(tags, [3,4,9])

    def test_cuboid_zmin(self):
        g = group.cuboid(name='test', zmin=None)
        tags = [(x.tag) for x in g]
        self.assertEqual(tags, list(range(11)))

        g = group.cuboid(name='test', zmin=0.99)
        tags = [(x.tag) for x in g]
        self.assertEqual(tags, [1,2,7])

    def test_cuboid_zmax(self):
        g = group.cuboid(name='test', zmax=None)
        tags = [(x.tag) for x in g]
        self.assertEqual(tags, list(range(11)))

        g = group.cuboid(name='test', zmax=-0.99)
        tags = [(x.tag) for x in g]
        self.assertEqual(tags, [3,4,10])

    def test_tag_list(self):
        g = group.tag_list(name='a', tags=[0, 5, 9]);
        tags = [(x.tag) for x in g]
        self.assertEqual(tags, [0,5,9])

    def test_union(self):
        A = group.type(type='A')
        B = group.type(type='B')

        union = group.union(name='test', a=A,b=B)
        tags = [(x.tag) for x in union]
        self.assertEqual(tags, list(range(11)))

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

    def test_cuboid_update(self):
        g = group.cuboid(name='test', xmin=0.99)
        tags = [(x.tag) for x in g]
        self.assertEqual(tags, [1,2,5])

        # move one particle out and another in
        self.s.particles[5].position = (-2,0,0);
        self.s.particles[9].position = (1,-2,0);
        g.force_update();
        tags = [(x.tag) for x in g]
        self.assertEqual(tags, [1,2,9])

    def test_type_update(self):
        B = group.type(type='B')
        tags = [(x.tag) for x in B]
        self.assertEqual(tags, [1, 2, 5, 8, 9, 10])

        self.s.particles[5].type = 'A';
        self.s.particles[6].type = 'B';
        B.force_update();
        tags = [(x.tag) for x in B]
        self.assertEqual(tags, [1, 2, 6, 8, 9, 10])

    def tearDown(self):
        del self.s
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
