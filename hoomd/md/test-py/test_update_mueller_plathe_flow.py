# -*- coding: iso-8859-1 -*-

from hoomd import *
from hoomd import deprecated
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
        self.system = deprecated.init.create_random(N=20000, phi_p=0.3,name='A');
        nlist = md.nlist.cell(3.0)
        lj = md.pair.lj(3.0,nlist)
        lj.pair_coeff.set('A','A',epsilon=1.0,sigma=1.0)
        md.integrate.mode_standard(dt=dt)
        md.integrate.nvt(group=group.all(),kT=1.2,tau=0.5)

    # tests basic creation of the updater
    def test(self):
        const_flow = variant.linear_interp(  [(0,0),(1e8,0.03*dt*1e8)] )
        #simple creation
        flow = md.update.mueller_plathe_flow(group.all(),const_flow,2,0,Nslabs,max_slab,min_slab)
        run(10);

    # tests with X slab direction and Y shear direction
    def test_XY(self):
        const_flow = variant.linear_interp(  [(0,0),(1e8,0.03*dt*1e8)] )
        flow = md.update.mueller_plathe_flow(group.all(),const_flow,0,1,Nslabs,max_slab,min_slab)
        run(10);

    # tests with X slab direction and Z shear direction
    def test_XZ(self):
        const_flow = variant.linear_interp(  [(0,0),(1e8,0.03*dt*1e8)] )
        flow = md.update.mueller_plathe_flow(group.all(),const_flow,0,2,Nslabs,max_slab,min_slab)
        run(10);

    # tests with Y slab direction and X shear direction
    def test_YX(self):
        const_flow = variant.linear_interp(  [(0,0),(1e8,0.03*dt*1e8)] )
        flow = md.update.mueller_plathe_flow(group.all(),const_flow,1,0,Nslabs,max_slab,min_slab)
        run(10);

    # tests with Y slab direction and Z shear direction
    def test_YX(self):
        const_flow = variant.linear_interp(  [(0,0),(1e8,0.03*dt*1e8)] )
        flow = md.update.mueller_plathe_flow(group.all(),const_flow,1,2,Nslabs,max_slab,min_slab)
        run(10);

    # tests with Y slab direction and Z shear direction
    def test_YX(self):
        const_flow = variant.linear_interp(  [(0,0),(1e8,0.03*dt*1e8)] )
        flow = md.update.mueller_plathe_flow(group.all(),const_flow,1,2,Nslabs,max_slab,min_slab)
        run(10);

    # tests with Z slab direction and X shear direction
    def test_YX(self):
        const_flow = variant.linear_interp(  [(0,0),(1e8,0.03*dt*1e8)] )
        flow = md.update.mueller_plathe_flow(group.all(),const_flow,2,0,Nslabs,max_slab,min_slab)
        run(10);

    # tests with Z slab direction and Y shear direction
    def test_YX(self):
        const_flow = variant.linear_interp(  [(0,0),(1e8,0.03*dt*1e8)] )
        flow = md.update.mueller_plathe_flow(group.all(),const_flow,2,1,Nslabs,max_slab,min_slab)
        run(10);


    # test member functions
    def test_mem_func(self):
        max_slab_loc = max_slab
        min_slab_loc = min_slab

        const_flow = variant.linear_interp(  [(0,0),(1e8,0.03*dt*1e8)] )
        #simple creation
        flow = md.update.mueller_plathe_flow(group.all(),const_flow,2,0,Nslabs,max_slab_loc,min_slab_loc)

        assert flow.get_n_slabs() == Nslabs
        assert flow.get_min_slab() == min_slab_loc
        assert flow.get_max_slab() == max_slab_loc
        epsilon = flow.get_flow_epsilon()
        epsilon *= 2
        flow.set_flow_epsilon(epsilon)
        assert flow.get_flow_epsilon() == epsilon
        assert flow.get_summed_exchanged_momentum() == 0

        tmp = max_slab_loc
        max_slab_loc = min_slab_loc
        min_slab_loc = tmp
        flow.swap_min_max()
        assert flow.get_min_slab() == min_slab_loc
        assert flow.get_max_slab() == max_slab_loc

        run(100);
        flow.update_domain_decomposition()
        run(100)
        snapshot = data.system_data.take_snapshot(self.system)
        area = snapshot.box.Ly * snapshot.box.Lz
        expected_flow = 0.03*dt*200
        assert abs(flow.get_summed_exchanged_momentum()/area -expected_flow) < epsilon+1e-2, str( (flow.get_summed_exchanged_momentum()/area,expected_flow) )


    def tearDown(self):
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
