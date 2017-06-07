from __future__ import division, print_function
from hoomd import *
from hoomd import hpmc
import unittest
import os
import numpy

context.initialize()

def create_empty(**kwargs):
    snap = data.make_snapshot(**kwargs);
    return init.read_snapshot(snap);

# This test ensures that the small box code path is enabled at the correct box sizes and works correctly
# It performs two tests
# 1) Initialize a system with known overlaps (or not) and verify that count_overlaps produces the correct result
# 2) Run many steps tracking overlap counts and trial moves accepted
#
# Success condition: Correctly functioning code should enable the small box code path and report overlaps when they
# are created and none during the run(). Some moves should be accepted and some should be rejected.
#
# Failure mode 1: If the box size is between 1 and 2 diameters and the cell list code path activates, it is an error
# This is now unused. HPMC always runs a small box capable path on the CPU.
#
# Failure mode 2: If the small box trial move code path does not correctly check the updated orientation when checking
# particles vs its own image - some number of overlaps will show up during the run().
#
# To detect these failure modes, a carefully designed system is needed. Place a cube (side length 1) in a cubic box
# 1 < L < sqrt(2). This allows the square to rotate to many possible orientations - but some are disallowed.
# For example, the corners of the square will overlap at 45 degrees.

class pair_smallbox3d_test1 (unittest.TestCase):
    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=1.9, dimensions=3), particle_types=['A'])

        self.mc = hpmc.integrate.convex_polyhedron(seed=10);
        self.mc.set_params(deterministic=True)
        self.mc.shape_param.set("A", vertices=[(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5)]);

        context.current.sorter.set_params(grid=8)

    def test_cell_list(self):
        # check 1, see if there are any overlaps. There should be none as the square is oriented along the box and L>1
        self.system.particles[0].position = (0,0,0);
        self.system.particles[0].orientation = (1,0,0,0);

        # run one step
        run(1);

        # verify that there are no overlaps
        self.assertEqual(self.mc.count_overlaps(), 0);

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize();


class pair_smallbox3d_test2 (unittest.TestCase):
    def setUp(self):
        self.system = create_empty(N=1, box=data.boxdim(L=1.2, dimensions=3), particle_types=['A'])

        self.mc = hpmc.integrate.convex_polyhedron(seed=10);
        self.mc.set_params(deterministic=True)
        self.mc.shape_param.set("A", vertices=[(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5)]);

        context.current.sorter.set_params(grid=8)

    def test_no_overlap(self):
        # check 1, see if there are any overlaps. There should be none as the square is oriented along the box and L>1
        self.system.particles[0].position = (0,0,0);
        self.system.particles[0].orientation = (1,0,0,0);

        run(0)
        self.assertEqual(self.mc.count_overlaps(), 0);

    def test_overlap(self):
        # check 2, rotate the box by 45 degrees and verify that there are overlaps
        self.system.particles[0].position = (0,0,0);
        self.system.particles[0].orientation = (0.9238795325112867,0,0,0.3826834323650898);

        run(0)
        self.assertGreater(self.mc.count_overlaps(), 0);

    def test_run(self):
        self.system.particles[0].position = (0,0,0);
        self.system.particles[0].orientation = (1,0,0,0);

        analyze.log(filename='small-box-3d.log', quantities=['hpmc_overlap_count'], period=1, overwrite=True);
        run(50000);

        # check 3 - verify that trial moves were both accepted and rejected
        count = self.mc.get_counters();
        # do not check translation moves because in a single particle system all translation moves are accepted
        self.assertGreater(count['rotate_accept_count'], 0);
        self.assertGreater(count['rotate_reject_count'], 0);

        # check 4 - verify that no overlaps resulted
        overlaps = numpy.genfromtxt('small-box-3d.log', skip_header=1);
        # os.remove('small-box-3d.log');

        self.assertEqual(numpy.max(overlaps[:,1]), 0);

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize();

if __name__ == '__main__':
    # this test works on the CPU only and only on a single rank
    if comm.get_num_ranks() > 1:
        raise RuntimeError("This test only works on 1 rank");

    unittest.main(argv = ['test.py', '-v'])
