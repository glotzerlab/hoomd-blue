# -*- coding: iso-8859-1 -*-

from hoomd import *
from hoomd import md;

context.initialize()
import unittest
import os

Nslabs = 20
min_slab = 0
max_slab = 10
dt = 0.005

# tests for md.update.mueller_plathe_flow
class update_mueller_plathe_flow (unittest.TestCase):
    def setUp(self):
        print
        self.system = init.create_lattice(lattice.sc(a=1.2039980656902276),n=[5,5,8]); #target a packing fraction of 0.3

    # tests basic creation of the updater
    def test(self):
        const_flow = variant.linear_interp(  [(0,0),(1e8,0.03*dt*1e8)] )
        #simple creation
        flow = md.update.mueller_plathe_flow(group.all(),const_flow,md.update.mueller_plathe_flow.Z,md.update.mueller_plathe_flow.X,Nslabs)
        run(10);

    # tests with X slab direction and Y shear direction
    def test_XY(self):
        const_flow = variant.linear_interp(  [(0,0),(1e8,0.03*dt*1e8)] )
        flow = md.update.mueller_plathe_flow(group.all(),const_flow,md.update.mueller_plathe_flow.X,md.update.mueller_plathe_flow.Y,Nslabs,max_slab,min_slab)
        run(10);

    # tests with X slab direction and Z shear direction
    def test_XZ(self):
        const_flow = variant.linear_interp(  [(0,0),(1e8,0.03*dt*1e8)] )
        flow = md.update.mueller_plathe_flow(group.all(),const_flow,md.update.mueller_plathe_flow.X,md.update.mueller_plathe_flow.Y,Nslabs,max_slab,min_slab)
        run(10);

    # tests with Y slab direction and X shear direction
    def test_YX(self):
        const_flow = variant.linear_interp(  [(0,0),(1e8,0.03*dt*1e8)] )
        flow = md.update.mueller_plathe_flow(group.all(),const_flow,md.update.mueller_plathe_flow.Y,md.update.mueller_plathe_flow.X,Nslabs,max_slab,min_slab)
        run(10);

    # tests with Y slab direction and Z shear direction
    def test_YX(self):
        const_flow = variant.linear_interp(  [(0,0),(1e8,0.03*dt*1e8)] )
        flow = md.update.mueller_plathe_flow(group.all(),const_flow,md.update.mueller_plathe_flow.Y,md.update.mueller_plathe_flow.Z,Nslabs,max_slab,min_slab)
        run(10);

    # tests with Y slab direction and Z shear direction
    def test_YZ(self):
        const_flow = variant.linear_interp(  [(0,0),(1e8,0.03*dt*1e8)] )
        flow = md.update.mueller_plathe_flow(group.all(),const_flow,md.update.mueller_plathe_flow.Y,md.update.mueller_plathe_flow.Z,Nslabs,max_slab,min_slab)
        run(10);

    # tests with Z slab direction and X shear direction
    def test_ZX(self):
        const_flow = variant.linear_interp(  [(0,0),(1e8,0.03*dt*1e8)] )
        flow = md.update.mueller_plathe_flow(group.all(),const_flow,md.update.mueller_plathe_flow.Z,md.update.mueller_plathe_flow.X,Nslabs,max_slab,min_slab)
        run(10);

    # tests with Z slab direction and Y shear direction
    def test_ZY(self):
        const_flow = variant.linear_interp(  [(0,0),(1e8,0.03*dt*1e8)] )
        flow = md.update.mueller_plathe_flow(group.all(),const_flow,md.update.mueller_plathe_flow.Z,md.update.mueller_plathe_flow.Y,Nslabs,max_slab,min_slab)
        run(10);


    # test member functions
    def test_mem_func(self):
        max_slab_loc = max_slab
        min_slab_loc = min_slab

        const_flow = variant.linear_interp(  [(0,0),(1e8,0.03*dt*1e8)] )
        #simple creation
        flow = md.update.mueller_plathe_flow(group.all(),const_flow,md.update.mueller_plathe_flow.Z,md.update.mueller_plathe_flow.X,Nslabs,max_slab_loc,min_slab_loc)

        assert flow.get_n_slabs() == Nslabs
        assert flow.get_min_slab() == min_slab_loc
        assert flow.get_max_slab() == max_slab_loc
        epsilon = flow.get_flow_epsilon()
        epsilon *= 2
        flow.set_flow_epsilon(epsilon)
        assert flow.get_flow_epsilon() == epsilon
        assert flow.get_summed_exchanged_momentum() == 0

        run(10);


    def tearDown(self):
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
