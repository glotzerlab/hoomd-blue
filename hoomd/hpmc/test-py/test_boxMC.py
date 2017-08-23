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

class boxMC_sanity_checks (unittest.TestCase):
    # This test runs a system at high enough pressure and for enough steps to ensure a dense system.
    # After adequate compression, it confirms at 1000 different steps in the simulation, the NPT
    # updater does not introduce overlaps.
    def test_prevents_overlaps(self):
        N=64
        L=20
        self.snapshot = data.make_snapshot(N=N, box=data.boxdim(L=L, dimensions=2), particle_types=['A'])
        self.system = init.read_snapshot(self.snapshot)
        self.mc = hpmc.integrate.convex_polygon(seed=1, d=0.1, a=0.1)
        self.mc.set_params(deterministic=True)
        self.boxMC = hpmc.update.boxmc(self.mc, betaP=1000, seed=1)
        self.boxMC.volume(delta=0.1, weight=1)
        self.mc.shape_param.set('A', vertices=[(-1,-1), (1,-1), (1,1), (-1,1)])

        # place particles
        a = L / 8.
        for k in range(N):
            i = k % 8
            j = k // 8 % 8
            self.system.particles[k].position = (i*a - 9.9, j*a - 9.9, 0)

        run(0)
        self.assertEqual(self.mc.count_overlaps(), 0)
        run(1000)
        overlaps = 0
        for i in range(100):
            run(10, quiet=True)
            overlaps += self.mc.count_overlaps()
        self.assertEqual(overlaps, 0)
        #print(system.box)

        del self.boxMC
        del self.mc
        del self.system
        del self.snapshot
        context.initialize()

    # This test places two particles that overlap significantly.
    # The maximum move displacement is set so that the overlap cannot be removed.
    # It then performs an NPT run and ensures that no volume or shear moves were accepted.
    def test_rejects_overlaps(self):
        self.snapshot = data.make_snapshot(N=2, box=data.boxdim(L=4), particle_types=['A'])
        self.system = init.read_snapshot(self.snapshot)
        self.mc = hpmc.integrate.convex_polyhedron(seed=1, d=0.1, a=0.1)
        self.mc.set_params(deterministic=True)
        self.boxMC = hpmc.update.boxmc(self.mc, betaP=1000, seed=1)
        self.boxMC.volume(delta=0.1, weight=1)
        self.mc.shape_param.set('A', vertices=[  (1,1,1), (1,-1,1), (-1,-1,1), (-1,1,1),
                                            (1,1,-1), (1,-1,-1), (-1,-1,-1), (-1,1,-1) ])

        self.system.particles[1].position = (0.7,0,0)

        run(0)
        overlaps = self.mc.count_overlaps()
        self.assertGreater(overlaps, 0)

        run(100)
        self.assertEqual(overlaps, self.mc.count_overlaps())
        self.assertEqual(self.boxMC.get_volume_acceptance(), 0)
        self.assertEqual(self.boxMC.get_shear_acceptance(), 0)

        del self.boxMC
        del self.mc
        del self.system
        del self.snapshot
        context.initialize()

    # This test places two particles that overlap significantly.
    # The maximum move displacement is set so that the overlap cannot be removed.
    # It then performs an NPT run and ensures that no volume or shear moves were accepted.
    def test_rejects_overlaps_lnV(self):
        self.snapshot = data.make_snapshot(N=2, box=data.boxdim(L=4), particle_types=['A'])
        self.system = init.read_snapshot(self.snapshot)
        self.mc = hpmc.integrate.convex_polyhedron(seed=1, d=0.1, a=0.1)
        self.mc.set_params(deterministic=True)
        self.boxMC = hpmc.update.boxmc(self.mc, betaP=1000, seed=1)
        self.boxMC.ln_volume(delta=0.01, weight=1)
        self.mc.shape_param.set('A', vertices=[  (1,1,1), (1,-1,1), (-1,-1,1), (-1,1,1),
                                            (1,1,-1), (1,-1,-1), (-1,-1,-1), (-1,1,-1) ])

        self.system.particles[1].position = (0.7,0,0)

        run(0)
        overlaps = self.mc.count_overlaps()
        self.assertGreater(overlaps, 0)

        run(100)
        self.assertEqual(overlaps, self.mc.count_overlaps())
        self.assertEqual(self.boxMC.get_ln_volume_acceptance(), 0)
        self.assertEqual(self.boxMC.get_shear_acceptance(), 0)

        del self.boxMC
        del self.mc
        del self.system
        del self.snapshot
        context.initialize()

    # This test runs an orthorhombic simple cubic lattice in the NPT ensemble to ensure
    # that the aspect ratios are preserved by volume moves.
    def test_VolumeMove_box_aspect_ratio(self):
        for i in range(1):
            self.system = init.create_lattice(lattice.sc(a = 1), n = [1, 2, 3])
            self.mc = hpmc.integrate.sphere(seed=i, d=0.0)
            self.boxMC = hpmc.update.boxmc(self.mc, betaP=100, seed=1)
            self.boxMC.volume(delta=1.0, weight=1)
            self.mc.shape_param.set('A', diameter=0.0)

            a1 = self.system.box.Lx/self.system.box.Ly
            a2 = self.system.box.Lx/self.system.box.Lz
            assert np.allclose([a1, a2], [1/2, 1/3]) # Sanity check

            for j in range(10):
                run(10, quiet = True)
                a1_after = self.system.box.Lx/self.system.box.Ly
                a2_after = self.system.box.Lx/self.system.box.Lz
                self.assertAlmostEqual(a1_after, a1)
                self.assertAlmostEqual(a2_after, a2)

            del self.boxMC
            del self.mc
            del self.system
            context.initialize()

    # This test runs a single-particle NPT system to test whether NPT allows the box to invert.
    def test_VolumeMove_box_inversion(self):
        for i in range(5):
            self.snapshot = data.make_snapshot(N=1, box=data.boxdim(L=0.1), particle_types=['A'])
            self.snapshot.particles.position[:] = (0,0,0)
            self.system = init.read_snapshot(self.snapshot)
            self.mc = hpmc.integrate.sphere(seed=i, d=0.0)
            self.mc.set_params(deterministic=True)
            self.boxMC = hpmc.update.boxmc(self.mc, betaP=100, seed=1)
            self.boxMC.volume(delta=1.0, weight=1)
            self.mc.shape_param.set('A', diameter=0.0)

            for j in range(10):
                run(10)
                self.assertGreater(self.system.box.get_volume(), 0)
                #print(system.box)

            del self.boxMC
            del self.mc
            del self.system
            del self.snapshot
            context.initialize()

    # This test runs a single-particle NPT system to test whether NPT allows the box to invert.
    def test_LengthMove_box_inversion(self):
        for i in range(5):
            self.snapshot = data.make_snapshot(N=1, box=data.boxdim(L=0.1), particle_types=['A'])
            self.snapshot.particles.position[:] = (0,0,0)
            self.system = init.read_snapshot(self.snapshot)
            self.mc = hpmc.integrate.sphere(seed=i, d=0.0)
            self.mc.set_params(deterministic=True)
            self.boxMC = hpmc.update.boxmc(self.mc, betaP=100, seed=1)
            self.boxMC.length(delta=[1.0, 1.0, 1.0], weight=1)
            self.mc.shape_param.set('A', diameter=0.0)

            for j in range(10):
                run(10)
                self.assertGreater(self.system.box.get_volume(), 0)
                #print(system.box)

            del self.boxMC
            del self.mc
            del self.system
            del self.snapshot
            context.initialize()

    # This test runs a single-particle NPT system to test whether NPT allows the box to invert.
    def test_AspectMove_box_inversion(self):
        for i in range(5):
            self.snapshot = data.make_snapshot(N=1, box=data.boxdim(L=0.1), particle_types=['A'])
            self.snapshot.particles.position[:] = (0,0,0)
            self.system = init.read_snapshot(self.snapshot)
            self.mc = hpmc.integrate.sphere(seed=i, d=0.0)
            self.mc.set_params(deterministic=True)
            self.boxMC = hpmc.update.boxmc(self.mc, betaP=100, seed=1)
            self.boxMC.aspect(delta=0.5, weight=1)
            self.mc.shape_param.set('A', diameter=0.05)

            for j in range(10):
                run(10)
                self.assertGreater(self.system.box.get_volume(), 0)
                #print(system.box)

            del self.boxMC
            del self.mc
            del self.system
            del self.snapshot
            context.initialize()


# These tests check the methods for functionality
class boxMC_test_methods (unittest.TestCase):
    def setUp(self):
        snapshot = data.make_snapshot(N=1, box=data.boxdim(L=4), particle_types=['A'])
        snapshot.particles.position[0] = (0, 0, 0)
        self.system = init.read_snapshot(snapshot)
        self.mc = hpmc.integrate.sphere(seed=1)
        self.mc.set_params(deterministic=True)
        self.mc.shape_param.set('A', diameter = 1.0)
        self.boxMC = hpmc.update.boxmc(self.mc, betaP=100, seed=1)

    def tearDown(self):
        del self.boxMC
        del self.mc
        del self.system
        context.initialize();

    def test_methods_setVolumeMove(self):
        boxMC = self.boxMC
        boxMC.volume(delta=1.0)
        boxMC.volume(delta=1.0, weight=1)

    def test_methods_setlnVMove(self):
        boxMC = self.boxMC
        boxMC.ln_volume(delta=1.0)
        boxMC.ln_volume(delta=1.0, weight=1)

    def test_methods_setLengthMove(self):
        boxMC = self.boxMC
        # test scalar delta
        boxMC.length(delta=10.0)
        # test list delta
        boxMC.length(delta=(1,1,1))
        boxMC.length(delta=(1,1,1), weight=2)

    def test_methods_setShearMove(self):
        boxMC = self.boxMC
        # test scalar delta
        boxMC.shear(delta=1.0)
        # test list delta
        boxMC.shear(delta=(1,1,1))
        boxMC.shear(delta=(1,1,1), weight=2)

    def test_get_deltas(self):
        boxMC = self.boxMC
        delta = tuple([float(n) for n in (0.1, 0.2, 0.3)])
        boxMC.length(delta=delta)
        self.assertEqual(tuple([float(n) for n in boxMC.length()['delta']]), delta)
        boxMC.shear(delta=delta)
        self.assertEqual(tuple([float(n) for n in boxMC.shear()['delta']]), delta)
        boxMC.volume(delta=0.1)
        self.assertEqual(boxMC.volume()['delta'], 0.1)
        boxMC.ln_volume(delta=0.01)
        self.assertEqual(boxMC.ln_volume()['delta'], 0.01)
        boxMC.aspect(delta=0.1)
        self.assertEqual(boxMC.aspect()['delta'], 0.1)


# This test takes too long to run. Validation tests do not need to be run on every commit.
# class boxMC_thermodynamic_tests (unittest.TestCase):
#     # This test checks the BoxMC updater against the ideal gas equation of state
#     def test_volume_ideal_gas(self):
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
#         self.system = create_empty(N=N, box=data.boxdim(L=L, dimensions=3), particle_types=['A'])
#         self.mc = hpmc.integrate.sphere(seed=1, d=0.0)
#         npt = hpmc.update.boxMC(self.mc, betaP=N, seed=1)
#         npt.setVolumeMove(delta=0.2, weight=1)
#         self.mc.shape_param.set('A', diameter=0.0)

#         # place particles
#         positions = np.random.random((N,3))*L - 0.5*L
#         for k in range(N):
#             self.system.particles[k].position = positions[k]

#         my_acc = accumulator(nsamples, self.system)
#         run(1e5, callback_period=sample_period, callback=my_acc.callback)
#         # for beta P == N the ideal gas law says V must be 1.0. We'll grant 10% error
#         self.assertLess(np.abs(my_acc.get_volumes().mean() - 1.0), 0.1)

#         del my_acc
#         del npt
#         del self.mc
#         del self.system
#         context.initialize()

if __name__ == '__main__':
    # this test works on the CPU only and only on a single rank
    if comm.get_num_ranks() > 1:
        raise RuntimeError("This test only works on 1 rank");

    unittest.main(argv = ['test.py', '-v'])
