import hoomd
from hoomd import md
import numpy as np
import unittest

hoomd.context.initialize()

class test_floppy_bodies(unittest.TestCase):
    """Test that floppy bodies enforced via the body flag work as expected"""
    def test_floppy(self):
        """Test case is an octahedron"""
        # Add some random noise to positions so that the bonds have some energy
        points = np.array(
            [[ 1,  0,  0],
             [-1,  0,  0],
             [ 0,  1,  0],
             [ 0, -1,  0],
             [ 0,  0,  1],
             [ 0,  0, -1]], dtype=np.float64)
        np.random.seed(3)
        points += 0.1*np.random.random_sample(points.shape)-0.05
        all_points = np.append([[0, 0, 0]], points, axis=0)

        # Add one to avoid the center particle
        bonds = np.array(
            [[0, 2], [0, 3], [0, 4], [0, 5],
             [1, 2], [1, 3], [1, 4], [1, 5],
             [2, 4], [2, 5],
             [3, 4], [3, 5]]) + 1

        box = hoomd.data.boxdim(L=10)  # Avoid self-interaction
        snapshot = hoomd.data.make_snapshot(N=points.shape[0]+1, box=box, particle_types=['center', 'constituent'], bond_types=['test_bond'])
        if hoomd.comm.get_rank() == 0:
            snapshot.particles.position[0] = [0, 0, 0]
            snapshot.particles.position[1:] = points
            snapshot.particles.typeid[0] = 0
            snapshot.particles.typeid[1:] = 1
            snapshot.particles.body[:] = 0  # Initially rigid, tagged to first particle

            snapshot.bonds.resize(bonds.shape[0])
            snapshot.bonds.typeid[:] = 0
            snapshot.bonds.group[:] = bonds

        system = hoomd.init.read_snapshot(snapshot)

        standard = md.integrate.mode_standard(dt=0.001)
        nvt = md.integrate.nvt(group=hoomd.group.rigid_center(), kT=1, tau=1)

        # Set up pair potentials and bonds. We need both to confirm that
        # everything is behaving as expected with bodies and exclusions.
        nl = md.nlist.stencil()
        yukawa = md.pair.yukawa(r_cut=2.1, nlist=nl)
        yukawa.pair_coeff.set('center',['center', 'constituent'], epsilon=0, kappa=0)
        yukawa.pair_coeff.set('constituent','constituent', epsilon=1.0, kappa=1.0)

        harmonic = md.bond.harmonic()
        harmonic.bond_coeff.set('test_bond', k=0.001, r0=np.sqrt(2))

        # Need to log quantities for testing.
        log = hoomd.analyze.log(filename=None,
                                quantities=['potential_energy',
                                            'kinetic_energy',
                                            'momentum',
                                            'pair_yukawa_energy',
                                            'bond_harmonic_energy'],
                                period=10)

        rigid = hoomd.md.constrain.rigid()
        rigid.set_param('center',
                        types=['constituent']*points.shape[0],
                        positions=points)
        rigid.validate_bodies()

        hoomd.run(1)
        original_bond_energy = log.query('bond_harmonic_energy')
        self.assertNotEqual(original_bond_energy, 0)

        # While rigid, there should be no potential energy, and the particles
        # should not have moved. However, there will be a nonzero bond energy
        hoomd.run(100)
        current_snapshot = system.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            self.assertTrue(np.allclose(current_snapshot.particles.position, all_points))
            self.assertEqual(log.query('bond_harmonic_energy'), original_bond_energy)
            self.assertEqual(log.query('pair_yukawa_energy'), 0)

        # Now convert them all to floppy bonds, and move the particles a little bit
        for p in system.particles:
            p.body = -2

        nvt.disable()
        rigid.disable()
        nvt = md.integrate.nvt(group=hoomd.group.all(), kT=1, tau=1)
        hoomd.run(100)

        # Remove the central particle to avoid any confusion.
        for p in system.particles:
            if p.typeid == 0:
                tag_delete = p.tag
        system.particles.remove(tag_delete)

        # Now there should be a different bond energy because particles should
        # have moved, but still no pair potential.
        current_snapshot = system.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            self.assertTrue(not np.allclose(current_snapshot.particles.position, points))
            self.assertNotEqual(log.query('bond_harmonic_energy'), 0)
            self.assertNotEqual(log.query('bond_harmonic_energy'), original_bond_energy)
            self.assertEqual(log.query('pair_yukawa_energy'), 0)

        # Now convert them all to separate particles
        for p in system.particles:
            p.body = -1

        hoomd.run(100)

        # Now there should be a both bond and pair potential energy
        current_snapshot = system.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            self.assertTrue(np.all(current_snapshot.particles.position != points))
            self.assertNotEqual(log.query('bond_harmonic_energy'), 0)
            self.assertNotEqual(log.query('bond_harmonic_energy'), original_bond_energy)
            self.assertNotEqual(log.query('pair_yukawa_energy'), 0)


    def tearDown(self):
        hoomd.context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
