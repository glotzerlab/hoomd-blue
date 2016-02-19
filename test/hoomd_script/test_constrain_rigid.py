from hoomd_script import *
import math
import unittest

context.initialize()

# test the constrain.rigid() functionality
class test_constrain_rigid(unittest.TestCase):
    def setUp(self):
        # particle radius
        N_A = 100
        N_B = 100
        phi_ini = 0.05

        r_rounding = .5
        species_A = dict(bond_len=2.1, type=['A'], bond="linear", count=N_A)
        species_B = dict(bond_len=2.1, type=['B'], bond="linear", count=N_B)

        # generate a system of N=8 AB diblocks
        self.system=init.create_random_polymers(box=data.boxdim(L=20), polymers=[species_A,species_B], separation=dict(A=1.0, B=1.0));

        for p in self.system.particles:
            p.inertia = (1,2,3)

    def test_energy_conservation(self):
        # create rigid spherocylinders out of two particles (not including the central particle)
        len_cyl = .5

        # create constituent particle types
        self.system.particles.types.add('A_const')
        self.system.particles.types.add('B_const')

        rigid = constrain.rigid()
        rigid.set_param('A', types=['A_const','A_const'], positions=[(0,0,-len_cyl/2),(0,0,len_cyl/2)])
        rigid.set_param('B', types=['B_const','B_const'], positions=[(0,0,-len_cyl/2),(0,0,len_cyl/2)])

        lj = pair.lj(r_cut=False)
        # central particles do not interact
        lj.pair_coeff.set(self.system.particles.types, self.system.particles.types, epsilon=0, sigma=0)

        # constituent particle coefficients
        lj.pair_coeff.set('A_const','A_const', epsilon=1.0, sigma=1.0, r_cut=2.5)
        lj.pair_coeff.set('B_const','B_const', epsilon=1.0, sigma=1.0, r_cut=2.5)
        lj.pair_coeff.set('A_const','B_const', epsilon=-1.0, sigma=1.0, r_cut=2.5)
        lj.set_params(mode="xplor")

        integrate.mode_standard(dt=0.005)
        nve = integrate.nve(group=group.rigid_center())

        log = analyze.log(filename=None, quantities=['potential_energy','kinetic_energy'], period=100)
        run(1)
        E0 = log.query('potential_energy') + log.query('kinetic_energy')
        run(1000)
        E1 = log.query('potential_energy') + log.query('kinetic_energy')

        self.assertAlmostEqual(E0,E1,2)
        del rigid
        del lj
        del log
        del nve

    def tearDown(self):
        del self.system
        init.reset();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
