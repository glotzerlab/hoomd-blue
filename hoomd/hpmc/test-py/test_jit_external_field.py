"""This module contains the operation functions for this project."""
from __future__ import print_function, division, absolute_import

import unittest

import hoomd
hoomd.context.initialize()

class jit_external_field(unittest.TestCase):

    def test_gravity(self):
        """This test simulates a sedimentation experiment by using an elongated
        box in the z-dimension and adding an effective gravitational
        potential with a wall. Note that it is technically probabilistic in
        nature, but we use enough particles and a strong enough gravitational
        potential that the probability of particles rising in the simulation is
        vanishingly small."""
        from hoomd import hpmc, jit
        import numpy as np


        # Just creating a simple cubic lattice # is fine here.
        system = hoomd.init.create_lattice(hoomd.lattice.sc(
            1), n=5)

        mc = hpmc.integrate.sphere(80391, d=0.1, a=0.1)
        mc.overlap_checks.set('A', 'A', False)
        mc.shape_param.set('A', diameter=1)

        # Expand system, add walls, and add gravity
        hoomd.update.box_resize(Lx=system.box.Lx*1.5, Ly = system.box.Ly*1.5,
                                Lz=system.box.Lz*20, scale_particles=False,
                                period=None)
        wall = hpmc.field.wall(mc)
        wall.add_plane_wall([0, 0, 1], [0, 0, -system.box.Lz/2])
        gravity_field = hoomd.jit.external.user(mc=mc, code="return 1000*(r_i.z + box.getL().z/2);")
        comp = hpmc.field.external_field_composite(mc, [wall, gravity_field])

        snapshot = system.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            old_avg_z = np.mean(snapshot.particles.position[:, 2])

        log = hoomd.analyze.log(filename=None, quantities=['external_field_jit'], period=None);

        hoomd.run(1)
        original_energy = log.query('external_field_jit')
        hoomd.run(1e3)

        snapshot = system.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            self.assertLess(np.mean(snapshot.particles.position[:, 2]), old_avg_z)

        if hoomd.comm.get_rank() == 0:
            self.assertLess(log.query('external_field_jit'), original_energy)


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
