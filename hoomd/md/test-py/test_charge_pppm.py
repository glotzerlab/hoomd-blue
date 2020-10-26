# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
from hoomd import md
import unittest
import os

context.initialize()

# charge.pppm
class charge_pppm_tests (unittest.TestCase):
    def setUp(self):
        self.s = init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4]); #target a packing fraction of 0.05

        for i in range(0,50):
            self.s.particles[i].charge = -1;

        for i in range(50,100):
            self.s.particles[i].charge = 1;

    # basic test of creation and param setting
    def test(self):
        all = group.all()
        nl = md.nlist.cell()
        c = md.charge.pppm(all, nlist = nl);
        c.set_params(Nx=16, Ny=16, Nz=16, order=4, rcut=2.0);
        md.integrate.mode_standard(dt=0.005);
        md.integrate.nve(all);
        run(100);

        del all
        del c

    # Cannot test pppm multiple times currently because of implementation limitations
    ## test missing coefficients
    #def test_set_missing_coeff(self):
        #all = group.all()
        #c = charge.pppm(all);
        #integrate.mode_standard(dt=0.005);
        #integrate.nve(all);
        #self.assertRaises(RuntimeError, run, 100);

    ## test enable/disable
    #def test_enable_disable(self):
        #all = group.all()
        #c = charge.pppm(all);
        #c.set_params(Nx=16, Ny=16, Nz=16, order=4, rcut=2.0);

        #c.disable(log=True);
        #c.enable();

    def tearDown(self):
        del self.s
        context.initialize()

# charge.pppm
class charge_pppm_bond_exclusions_test(unittest.TestCase):
    def test_exclusion_energy(self):
        # initialize a two particle system in a large box, to minimize effect of PBC
        snap = data.make_snapshot(N=2, particle_types=[u'A1'], bond_types=['bondA'], box = data.boxdim(L=50))

        if comm.get_rank() == 0:
            snap.particles.position[0] = (0,0,0)
            snap.particles.position[1] = (1,1,1)
            snap.particles.charge[0] = 1
            snap.particles.charge[1] = -1

        self.s = init.read_snapshot(snap);

        # measure the Coulomb energy with sufficient number of grid cells so that the exclusions are accurate
        nl = md.nlist.cell()
        c = md.charge.pppm(group.all(), nlist = nl);

        # rcut should be larger than distance of pair, so we have a split into short range and long range energy
        c.set_params(Nx=128, Ny=128, Nz=128, order=7, rcut=3);
        log = analyze.log(quantities = ['potential_energy','pair_ewald_energy','pppm_energy'], period = 1, filename=None);
        md.integrate.mode_standard(dt=0.0);
        md.integrate.nve(group.all());
        # trick to allow larger decompositions
        nl.set_params(r_buff=0.1)
        context.current.sorter.disable()

        run(1)
        pppm_energy_nobond = log.query('pppm_energy')
        ewald_energy_nobond = log.query('pair_ewald_energy')
        potential_energy_nobond = log.query('potential_energy')

        # check that it is **roughly** equal to the Coulomb energy (not accounting for PBC)
        import numpy as np
        self.assertAlmostEqual(potential_energy_nobond,
            -1/np.linalg.norm(np.array(self.s.particles[1].position)-np.array(self.s.particles[0].position)),3)

        # now introduce a bond
        self.s.bonds.add('bondA', 0,1)

        # we need to manually add it to the exclusions currently
        nl.reset_exclusions(exclusions=['bond'])
        run(1)
        pppm_energy_bond = log.query('pppm_energy')
        ewald_energy_bond = log.query('pair_ewald_energy')
        potential_energy_bond = log.query('potential_energy')

        # energy should be practically zero
        # but not exactly -- exclusions are computed analytically, the long range part using fft
        self.assertAlmostEqual(pppm_energy_bond, 0.0, 3)
        self.assertAlmostEqual(potential_energy_bond, 0.0, 3)

        # but strictly zero for the short range part
        self.assertEqual(ewald_energy_bond, 0.0)

    def tearDown(self):
        del self.s
        context.initialize();


# charge.pppm
class charge_pppm_twoparticle_tests (unittest.TestCase):
    def setUp(self):
        print
        # initialize a two particle system in a triclinic box
        snap = data.make_snapshot(N=2, particle_types=[u'A1'], box = data.boxdim(xy=0.5,xz=0.5,yz=0.5,L=10))

        if comm.get_rank() == 0:
            snap.particles.position[0] = (0,0,0)
            snap.particles.position[1] = (3,3,3)
            snap.particles.charge[0] = 1
            snap.particles.charge[1] = -1

        self.s = init.read_snapshot(snap);

    # basic test of creation and param setting
    def test(self):
        all = group.all()
        nl = md.nlist.cell()
        c = md.charge.pppm(all, nlist = nl);
        c.set_params(Nx=128, Ny=128, Nz=128, order=3, rcut=2.0);
        log = analyze.log(quantities = ['pppm_energy','pressure_xx','pressure_xy','pressure_xz', 'pressure_yy','pressure_yz', 'pressure_zz'], period = 1, filename=None);
        md.integrate.mode_standard(dt=0.0);
        md.integrate.nve(all);
        # trick to allow larger decompositions
        nl.set_params(r_buff=0.1)
        run(1);

        self.assertAlmostEqual(c.forces[0].force[0], 0.00904953, 5)
        self.assertAlmostEqual(c.forces[0].force[1], 0.0101797, 5)
        self.assertAlmostEqual(c.forces[0].force[2], 0.0124804, 5)
        self.assertAlmostEqual(c.forces[0].energy, 0, 5)
        self.assertAlmostEqual(c.forces[0].virial[0], 0, 5)
        self.assertAlmostEqual(c.forces[0].virial[1], 0, 5)
        self.assertAlmostEqual(c.forces[0].virial[2], 0, 5)
        self.assertAlmostEqual(c.forces[0].virial[3], 0, 5)
        self.assertAlmostEqual(c.forces[0].virial[4], 0, 5)
        self.assertAlmostEqual(c.forces[0].virial[5], 0, 5)

        self.assertAlmostEqual(c.forces[1].force[0], -0.00904953, 5)
        self.assertAlmostEqual(c.forces[1].force[1], -0.0101797, 5)
        self.assertAlmostEqual(c.forces[1].force[2], -0.0124804, 5)
        self.assertAlmostEqual(c.forces[1].energy, 0, 5)
        self.assertAlmostEqual(c.forces[1].virial[0], 0, 5)
        self.assertAlmostEqual(c.forces[1].virial[1], 0, 5)
        self.assertAlmostEqual(c.forces[1].virial[2], 0, 5)
        self.assertAlmostEqual(c.forces[1].virial[3], 0, 5)
        self.assertAlmostEqual(c.forces[1].virial[4], 0, 5)
        self.assertAlmostEqual(c.forces[1].virial[5], 0, 5)

        vol = self.s.box.get_volume()
        self.assertAlmostEqual(log.query('pppm_energy'), -0.2441,4)
        self.assertAlmostEqual(log.query('pressure_xx'), -5.7313404e-05, 2)
        self.assertAlmostEqual(log.query('pressure_xy'), -4.5494677e-05, 2)
        self.assertAlmostEqual(log.query('pressure_xz'), -3.9889249e-05, 2)
        self.assertAlmostEqual(log.query('pressure_yy'), -7.8745142e-05, 2)
        self.assertAlmostEqual(log.query('pressure_yz'), -4.8501155e-05, 2)
        self.assertAlmostEqual(log.query('pressure_zz'), -0.00010732774, 2)

        del all
        del c
        del log

    # Cannot test pppm multiple times currently because of implementation limitations
    ## test missing coefficients
    #def test_set_missing_coeff(self):
        #all = group.all()
        #c = md.charge.pppm(all);
        #integrate.mode_standard(dt=0.005);
        #integrate.nve(all);
        #self.assertRaises(RuntimeError, run, 100);

    ## test enable/disable
    #def test_enable_disable(self):
        #all = group.all()
        #c = md.charge.pppm(all);
        #c.set_params(Nx=16, Ny=16, Nz=16, order=4, rcut=2.0);

        #c.disable(log=True);
        #c.enable();

    def tearDown(self):
        del self.s
        context.initialize();

# charge.pppm
class charge_pppm_screening_test(unittest.TestCase):
    def setUp(self):
        print
        # initialize a two particle system in a triclinic box
        snap = data.make_snapshot(N=2, particle_types=[u'A1'], box = data.boxdim(L=10))

        if comm.get_rank() == 0:
            snap.particles.position[0] = (0,0,0)
            snap.particles.position[1] = (0,0,1.2)
            snap.particles.charge[0] = -1
            snap.particles.charge[1] = 1

        self.s = init.read_snapshot(snap);

    # test PPPM for Yukawa interaction
    def test_screening(self):
        all = group.all()
        nl = md.nlist.cell()
        c = md.charge.pppm(all, nlist = nl);
        # use a reasonably long screening length so the charges are actually interacting via
        # the periodic boundaries
        c.set_params(Nx=128, Ny=128, Nz=128, order=7, rcut=1.5, alpha=0.1);
        log = analyze.log(quantities = ['potential_energy'], period = 1, filename=None);
        md.integrate.mode_standard(dt=0.0);
        md.integrate.nve(all);
        # trick to allow larger decompositions
        nl.set_params(r_buff=0.1)
        context.current.sorter.disable()
        run(1);

        # the reference forces and energies have been obtained by
        # summing the (derivatives of) the ionic crystal with Yukawa potential over nmax=6
        # periodic images in Mathematica
        self.assertAlmostEqual(self.s.particles[0].net_force[0], 0, 5)
        self.assertAlmostEqual(self.s.particles[0].net_force[1], 0, 5)
        self.assertAlmostEqual(self.s.particles[0].net_force[2], 0.685379, 5)

        self.assertAlmostEqual(self.s.particles[1].net_force[0], 0, 5)
        self.assertAlmostEqual(self.s.particles[1].net_force[1], 0, 5)
        self.assertAlmostEqual(self.s.particles[1].net_force[2], -0.685379, 5)

        pe = log.query('potential_energy')
        self.assertAlmostEqual(pe,-0.741706,5)

        del all
        del c
        del log

    def tearDown(self):
        del self.s
        context.initialize();

# charge.pppm
class charge_pppm_rigid_body_test(unittest.TestCase):
    def setUp(self):
        # initialize a two particle system in a cubic box
        snap = data.make_snapshot(N=1, particle_types=[u'A'], box = data.boxdim(L=15))
        if comm.get_rank() == 0:
            snap.particles.position[0] = (0,0,0)
            snap.particles.orientation[0] = (1,0,0,0)
            snap.particles.charge[0] = 0
            snap.particles.moment_inertia[0] = (1,1,1)

        self.s = init.read_snapshot(snap);

        self.rigid = md.constrain.rigid()
        self.s.particles.types.add('A1')
        self.rigid.set_param('A',types=['A1']*2,positions=[(0,0,0),(0,0,1.2)],charges=[-1,1])

    # test PPPM for Yukawa interaction
    def test_screening_noexclusion(self):
        # create rigid particles
        self.rigid.create_bodies()

        all = group.all()
        nl = md.nlist.cell()
        c = md.charge.pppm(all, nlist = nl);
        c.set_params(Nx=128, Ny=128, Nz=128, order=7, rcut=1.5, alpha=0.1);
        log = analyze.log(quantities = ['potential_energy'], period = 1, filename=None);
        md.integrate.mode_standard(dt=0.0);
        md.integrate.nve(group.rigid_center());
        # trick to allow larger decompositions
        nl.set_params(r_buff=0.1)
        context.current.sorter.disable()

        nl.reset_exclusions()
        run(1);

        # pairwise forces on a rigid body cancel out
        self.assertAlmostEqual(self.s.particles[0].net_force[0], 0, 5)
        self.assertAlmostEqual(self.s.particles[0].net_force[1], 0, 5)
        self.assertAlmostEqual(self.s.particles[0].net_force[2], 0, 5)

        pe = log.query('potential_energy')
        self.assertAlmostEqual(pe,-0.739741,5) # Mathematica ionic crystal calculation

        del all
        del c
        del log

    # test PPPM for Yukawa interaction with rigid body exclusions
    def test_screening_exclusions_set(self):
        import math
        import numpy as np

        # create rigid particles
        self.rigid.create_bodies()

        all = group.all()
        nl = md.nlist.cell()
        c = md.charge.pppm(all, nlist = nl);
        kappa = 0.1
        c.set_params(Nx=128, Ny=128, Nz=128, order=7, rcut=1.5, alpha=kappa);
        log = analyze.log(quantities = ['potential_energy','pppm_energy','pair_ewald_energy'], period = 1, filename=None);
        md.integrate.mode_standard(dt=0.0);
        md.integrate.nve(group.rigid_center());
        # trick to allow larger decompositions
        nl.set_params(r_buff=0.1)
        context.current.sorter.disable()

        run(1);

        # pairwise forces on a rigid body cancel out
        self.assertAlmostEqual(self.s.particles[0].net_force[0], 0, 5)
        self.assertAlmostEqual(self.s.particles[0].net_force[1], 0, 5)
        self.assertAlmostEqual(self.s.particles[0].net_force[2], 0, 5)

        pe = log.query('potential_energy')
        dx = np.array(self.s.particles[2].position)-np.array(self.s.particles[1].position)
        r = math.sqrt(np.dot(dx,dx))
        pe_primary_image = -math.exp(-kappa*r)/r
        self.assertAlmostEqual(pe_primary_image,-0.7391,4)

        self.assertAlmostEqual(pe,-0.739741-pe_primary_image,5)

        del all
        del c
        del log


    def tearDown(self):
        del self.rigid
        del self.s
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
