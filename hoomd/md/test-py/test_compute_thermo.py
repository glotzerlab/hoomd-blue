# -*- coding: iso-8859-1 -*-
# Maintainer: joaander
from __future__ import print_function
from __future__ import division

from hoomd import *
from hoomd import md
context.initialize()
import unittest
import os
import numpy

numpy.random.seed(1)

# unit tests for analyze.msd
class compute_thermo_tests (unittest.TestCase):
    def setUp(self):
        self.N = 1000
        self.snap = data.make_snapshot(N=self.N, particle_types=['A'], box=data.boxdim(L=100))

        self.v = (numpy.random.rand(self.N,3)-0.5)*10
        self.m = (numpy.random.rand(self.N)+1)

        if comm.get_rank() == 0:
            self.snap.particles.velocity[:] = self.v
            self.snap.particles.mass[:] = self.m

        init.read_snapshot(self.snap)
        context.current.sorter.set_params(grid=8)

    # API test: tests basic creation of the compute
    def test_api(self):
        typeA = group.type(name='typeA', type='A')
        compute.thermo(group=typeA);
        run(1);

    # Unit test: Validate temperature computation
    def test_temperature(self):
        # Initialize the compute thermo
        typeA = group.type(name='A', type='A')
        compute.thermo(group=typeA);

        # Logger to access computed quantities
        quantities=['num_particles_A',
                    'ndof_A',
                    'translational_ndof_A',
                    'rotational_ndof_A',
                    'potential_energy_A',
                    'kinetic_energy_A',
                    'translational_kinetic_energy_A',
                    'rotational_kinetic_energy_A',
                    'temperature_A'];

        log = analyze.log(filename=None, quantities=quantities, period=None);

        # dummy integrator to apply appropriate degrees of freedom
        md.integrate.mode_standard(dt=0.0);
        md.integrate.nve(group=group.all());

        run(1);

        numpy.testing.assert_allclose(log.query('num_particles_A'), self.N)
        numpy.testing.assert_allclose(log.query('ndof_A'), 3*self.N-3)
        numpy.testing.assert_allclose(log.query('translational_ndof_A'), 3*self.N-3)
        numpy.testing.assert_allclose(log.query('rotational_ndof_A'), 0, atol=1e-7)
        numpy.testing.assert_allclose(log.query('potential_energy_A'), 0, atol=1e-7)

        m = self.m;
        v = self.v;
        K_ref = 1/2 * numpy.sum(m * (v[:,0]**2 + v[:,1]**2 + v[:,2]**2))

        numpy.testing.assert_allclose(log.query('kinetic_energy_A'), K_ref)
        numpy.testing.assert_allclose(log.query('translational_kinetic_energy_A'), K_ref)
        numpy.testing.assert_allclose(log.query('rotational_kinetic_energy_A'), 0, atol=1e-7)
        numpy.testing.assert_allclose(log.query('temperature_A'), 2.0 / (3*self.N-3) * K_ref)


    def tearDown(self):
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
