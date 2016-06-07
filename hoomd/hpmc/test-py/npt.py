# Test npt updater for basic sanity. Proper validation requires thermodynamic data from much longer runs.
from __future__ import division, print_function
from hoomd import *
from hoomd import hpmc
import unittest
import numpy as np

context.initialize()

def create_empty(**kwargs):
    snap = data.make_snapshot(**kwargs);
    return init.read_snapshot(snap);

class npt_sanity_checks (unittest.TestCase):
    # This test runs a system at high enough pressure and for enough steps to ensure a dense system.
    # After adequate compression, it confirms at 1000 different steps in the simulation, the NPT
    # updater does not introduce overlaps.
    def test_prevents_overlaps(self):
        N=64
        L=20
        system = create_empty(N=N, box=data.boxdim(L=L, dimensions=2), particle_types=['A'])
        mc = hpmc.integrate.convex_polygon(seed=1, d=0.1, a=0.1)
        npt = hpmc.update.npt(mc, seed=1, P=1000, dLx=0.01, dLy=0.01, dLz=0.01, dxy=0.01, dxz=0.01, dyz=0.01)
        mc.shape_param.set('A', vertices=[(-1,-1), (1,-1), (1,1), (-1,1)])

        # place particles
        a = L / 8.
        for k in range(N):
            i = k % 8
            j = k // 8 % 8
            system.particles[k].position = (i*a - 9.9, j*a - 9.9, 0)

        run(0)
        self.assertEqual(mc.count_overlaps(), 0)
        run(1000)
        overlaps = 0
        for i in range(100):
            run(10)
            overlaps += mc.count_overlaps()
        self.assertEqual(overlaps, 0)
        print(system.box)

        del npt
        del mc
        del system
        context.initialize()

    # This test places two particles that overlap significantly.
    # The maximum move displacement is set so that the overlap cannot be removed.
    # It then performs an NPT run and ensures that no volume or shear moves were accepted.
    def test_rejects_overlaps(self):
        system = create_empty(N=2, box=data.boxdim(L=100), particle_types=['A'])
        mc = hpmc.integrate.convex_polyhedron(seed=1, d=0.1, a=0.1,max_verts=8)
        npt = hpmc.update.npt(mc, seed=1, P=1000, dLx=0.1, dLy=0.1, dLz=0.1, dxy=0.01, dxz=0.01, dyz=0.01)
        mc.shape_param.set('A', vertices=[  (1,1,1), (1,-1,1), (-1,-1,1), (-1,1,1),
                                            (1,1,-1), (1,-1,-1), (-1,-1,-1), (-1,1,-1) ])

        system.particles[1].position = (0.7,0,0)

        run(0)
        overlaps = mc.count_overlaps()
        self.assertGreater(overlaps, 0)

        run(100)
        self.assertEqual(overlaps, mc.count_overlaps())
        self.assertEqual(npt.get_volume_acceptance(), 0)
        self.assertEqual(npt.get_shear_acceptance(), 0)

        del npt
        del mc
        del system
        context.initialize()

    # This test runs a single-particle NPT system to test whether NPT allows the box to invert.
    def test_box_inversion(self):
        for i in range(5):
            system = create_empty(N=1, box=data.boxdim(L=4), particle_types=['A'])
            mc = hpmc.integrate.sphere(seed=i, d=0.0)
            npt = hpmc.update.npt(mc, seed=1, P=100, dLx=10.0, dLy=10.0, dLz=10.0, dxy=0, dxz=0, dyz=0, move_ratio=1)
            mc.shape_param.set('A', diameter=1.0)

            system.particles[0].position = (0,0,0)

            for j in range(10):
                run(10)
                self.assertGreater(system.box.get_volume(), 0)
                print(system.box)

            del npt
            del mc
            del system
            context.initialize()

# this test takes far too long to run for a unit test and should be migrated to a validation test suite
# class npt_thermodynamic_tests (unittest.TestCase):
#     # This test checks the NPT updater against the ideal gas equation of state
#     def test_ideal_gas(self):
#         N=100
#         L=2.0
#         nsteps = 1e4
#         nsamples = 1e3
#         sample_period = int(nsteps/nsamples)
#         class accumulator:
#             def __init__(self,nsamples,system):
#                 self.volumes = np.empty((nsamples),)
#                 self.i = 0
#                 self.system = system
#             def callback(self,timestep):
#                 if self.i < nsamples:
#                     self.volumes[self.i] = self.system.box.get_volume()
#                 self.i += 1
#             def get_volumes(self):
#                 return self.volumes[:self.i]
#         system = create_empty(N=N, box=data.boxdim(L=L, dimensions=3), particle_types=['A'])
#         mc = hpmc.integrate.sphere(seed=1, d=0.0)
#         npt = hpmc.update.npt(mc, seed=1, P=N, dLx=0.2, move_ratio=1.0, isotropic=True)
#         mc.shape_param.set('A', diameter=0.0)

#         # place particles
#         positions = np.random.random((N,3))*L - 0.5*L
#         for k in range(N):
#             system.particles[k].position = positions[k]

#         my_acc = accumulator(nsamples, system)
#         run(1e5, callback_period=sample_period, callback=my_acc.callback)
#         # for beta P == N the ideal gas law says V must be 1.0. We'll grant 10% error
#         self.assertLess(np.abs(my_acc.get_volumes().mean() - 1.0), 0.1)

#         del my_acc
#         del npt
#         del mc
#         del system
#         context.initialize()


if __name__ == '__main__':
    # this test works on the CPU only and only on a single rank
    if comm.get_num_ranks() > 1:
        raise RuntimeError("This test only works on 1 rank");

    unittest.main(argv = ['test.py', '-v'])
