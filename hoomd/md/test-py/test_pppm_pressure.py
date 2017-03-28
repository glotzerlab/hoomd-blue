# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md;
context.initialize()
import unittest
import os
import numpy
import math

class pppm_pressure_tests (unittest.TestCase):

    # test to ensure that the PPPM pressure remains stable
    # based on https://bitbucket.org/kimNle/pppm_test/src/2048eb67f063520f00fed711eab3cbfb03cec15e/pppm_test.py?at=master&fileviewer=file-view-default
    def test(self):

        snapshot = data.make_snapshot(N=2,particle_types=['A', 'B'],box=data.boxdim(L=10),bond_types=['test'])
        system = init.read_snapshot(snapshot)

        system.bonds.add('test',system.particles[0].tag,system.particles[1].tag)
        harmonic = md.bond.harmonic();
        harmonic.bond_coeff.set('test', k=1.0, r0=1.0)


        all = group.all();

        system.particles[0].position=(0,0,1)
        system.particles[0].charge = -1;
        system.particles[1].charge = 1;


        nl = md.nlist.cell()
        nl.reset_exclusions(exclusions=['bond'])
        c = md.charge.pppm(all, nlist = nl);
        c.set_params(Nx=16, Ny=16, Nz=16, order=4, rcut=2.0);
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);

        logdata = ['temperature',
                   'pressure',
                   'potential_energy',
                   'kinetic_energy',
                   'volume',
                   'pppm_energy'
                   ]

        log = analyze.log(filename=None, quantities=logdata, period=10, overwrite=True)

        pressure_measure = []
        def log_callback(timestep):
            v = log.query('pressure');
            pressure_measure.append(v)
            if comm.get_rank() == 0:
                print('pressure =', v);

        run(5000,callback=log_callback, callback_period=50)

        # this test checks for a bug where PPPM pressure monotonically increases.
        # it simply checks if the average of the last half of the run is significantly bigger than the first half

        h = int(len(pressure_measure)/2)
        a1 = numpy.mean(pressure_measure[:h])
        a2 = numpy.mean(pressure_measure[h:])

        self.assertLess(math.fabs(a2), math.fabs(a1*1.5))

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
