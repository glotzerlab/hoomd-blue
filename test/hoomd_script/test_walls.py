# -*- coding: iso-8859-1 -*-
# Maintainer: jproc

from hoomd_script import *
import unittest
import os

#test wall.group()
class wall_group_tests(unittest.TestCase):
    def setUp(self):
        init.create_random(N=100, box=data.boxdim(L=5), Period=None);
        updater=update.box_resize(L = 10);
        updater.set_params(scale_particles = False);

    # basic test of creation for walls structure
    def test(self):
        walls=wall.group();

    # test each type of geometry can be added and deleted
    def test_add_sphere(self):
        walls=wall.group();
        walls.add_sphere(r=4, origin=(0.0, 0.0, 0.0), inside=True);
        walls.del_sphere(0);

    def test_add_cylinder(self):
        walls=wall.group();
        walls.add_cylinder(r=4, origin=(0.0, 0.0, 0.0), axis=(0.0, 0.0, 1.0), inside=True);
        walls.del_cylinder(0);

    def test_add_plane(self):
        walls=wall.group();
        walls.add_plane(normal=(1.0, 0.0, 0.0), origin=(-4.0, 0.0, 0.0));
        walls.add_plane(normal=(-1.0, 0.0, 0.0), origin=(4.0, 0.0, 0.0));
        walls.del_plane([0,1]);

    # test forces
    #TODO:add after api is finalized
