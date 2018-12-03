"""This module contains the operation functions for this project."""
from __future__ import print_function, division, absolute_import

import unittest

class enthalpic_dipole_interaction(unittest.TestCase):

    def test_gravity(self):
        """This operation simulates a sedimentation experiment by using an elongated box in the z-dimension and adding an effective gravitational potential (in the absence of depletion)."""
        import hoomd
        from hoomd import hpmc, jit
        import numpy as np

        hoomd.context.initialize();

        # Just creating a simple cubic lattice # is fine here.
        system = hoomd.init.create_lattice(hoomd.lattice.sc(
            1), n=10)

        mc = hpmc.integrate.sphere(80391, d=0.1, a=0.1)
        mc.overlap_checks.set('A', 'A', False)
        mc.shape_param.set('A', diameter=1)

        # Expand system, add walls, and add gravity
        hoomd.update.box_resize(Lx=system.box.Lx*1.5, Ly = system.box.Ly*1.5,
                                Lz=system.box.Lz*20, scale_particles=False,
                                period=None)
        wall = hpmc.field.wall(mc)
        wall.add_plane_wall([0, 0, 1], [0, 0, -system.box.Lz/2])
        gravity_field = hoomd.jit.force.user(mc=mc, code="return pos.z + box.getL().z/2;")

        snapshot = system.take_snapshot()
        old_avg_z = np.mean(snapshot.particles.position[:, 2])

        hoomd.run(1e3)

        snapshot = system.take_snapshot()
        self.assertTrue(np.mean(snapshot.particles.position[:, 2]) < old_avg_z)

if __name__ == '__main__':
    unittest.main()
