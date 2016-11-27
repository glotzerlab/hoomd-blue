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

# Tests that randomly generated systems of multi-object particles have the expected logical overlap status.

## Helper function to check if two dumbbells of unit spheres overlap
# \param A 2x3 list of sphere coordinates for first dumbbell
# \param B 2x3 list of sphere coordinates for first dumbbell
# \param box box dimensions (Lx, Ly, Lz)
# \returns True if dumbbells overlap, else False
def dumbbell_overlap(A, B, box):
    d = 1.0 # diameter of spheres
    for r_i in numpy.asarray(A, dtype=float):
        for r_j in numpy.asarray(B, dtype=float):
            r_ij = r_j - r_i
            for k in (0,1,2):
                if r_ij[k] < -box[k]/2.:
                    r_ij[k] += box[k]
                if r_ij[k] > box[k]/2.:
                    r_ij[k] -= box[k]
            r2 = numpy.dot(r_ij, r_ij)
            if r2 < d:
                return True
    return False

# Rotate a vector with a unit quaternion
# \param q rotation quaternion
# \param v 3d vector to be rotated
# \returns rotated vector
def quatRot(q, v):
    v = numpy.asarray(v)
    q = numpy.asarray(q)
    # assume q is a unit quaternion
    w = q[0]
    r = q[1:]
    vnew = numpy.empty((3,), dtype=v.dtype)
    return v + 2*numpy.cross(r, numpy.cross(r,v) + w*v)

class helper_functions(unittest.TestCase):
    def test_dumbbell_overlap(self):
        box = [10,10,10]
        A = numpy.array([[-0.5, 0, 0], [0.5, 0, 0]])
        B = numpy.array(A)
        self.assertTrue(dumbbell_overlap(A, B, box))

        B = A + [0, 0.99, 0]
        self.assertTrue(dumbbell_overlap(A, B, box))

        B = A + [0, 1.01, 0]
        self.assertTrue(not dumbbell_overlap(A, B, box))

        B = A + [1.99, 0, 0]
        self.assertTrue(dumbbell_overlap(A, B, box))

        B = A + [2.01, 0, 0]
        self.assertTrue(not dumbbell_overlap(A, B, box))

        B = A + [-1.0, 0, 0.99]
        self.assertTrue(dumbbell_overlap(A, B, box))

        B = A + [-1.0, 0, 1.01]
        self.assertTrue(not dumbbell_overlap(A, B, box))

class sphere_union(unittest.TestCase):
    # Test randomly generated dumbbells
    # 1) generate N dumbbell positions and orientations
    # 2) place 2N spheres and determine if system has overlapping dumbbells
    # 3) place N dumbbells and determine if system has overlapping dumbbells
    # 4) confirm agreement between tests
    # 5) wash, rinse, repeat
    def test_dumbbells (self):
        # number of dumbbells, size of box, and number of iterations are chosen empirically for
        #  * a good mix of overlapping and non-overlapping configurations
        #  * a reasonable opportunity to detect errors (early testing had errors that did not appear until iteration 15)
        #  * reasonable run time
        # Run time could be improved with better code in the brute force overlap check (such as vectorizing dumbbell_overlap()
        # and and moving the N^2 loop to C via numpy, or turning the N^2 loop into an O(N) loop by pre-filtering neighbors),
        # but this might cost readability and developer time better spent elsewhere.
        #
        N = 90        # number of dumbbells
        L = 40        # box edge length
        num_iter = 50 # number of times to generate new configurations
        seed = 1      # PRNG seed
        ndim = 3

        spheres = numpy.array([[-0.5, 0, 0],[0.5, 0, 0]])      # positions of spheres in dumbbell coordinates
        system = create_empty(N=N, box=data.boxdim(L=L, dimensions=ndim), particle_types=['A'])

        # decrease initialization time with smaller grid for Hilbert curve
        context.current.sorter.set_params(grid=8)

        mc = hpmc.integrate.sphere_union(seed=seed);
        mc.shape_param.set("A", diameters=[1.0, 1.0], centers=spheres);

        # use fixed seed
        numpy.random.seed(seed)

        for i in range(num_iter):
            # randomly create "dumbbells" as pairs of spheres located anywhere in the box
            positions = (numpy.random.random((N,ndim)) - 0.5) * L  # positions of dumbbells in box
            # not uniformly sampling orientations, but that's okay
            orientations = numpy.random.random((N,4)) - 0.5
            # normalize to unit quaternions
            o2 = numpy.einsum('ij,ij->i', orientations, orientations)
            orientations = numpy.einsum('ij,i->ij', orientations, 1./numpy.sqrt(o2)) # orientations of dumbbells

            dumbbell = numpy.array([[quatRot(q, spheres[0]) + r, quatRot(q, spheres[1]) + r] for r,q in zip(positions, orientations)])

            # perform brute force overlap check
            overlaps = False
            for i in range(N-1):
                for j in range(i+1, N):
                    if dumbbell_overlap(dumbbell[i], dumbbell[j], [L,L,L]):
                        overlaps = True
                        break
                if overlaps == True:
                    break
            sphere_overlaps = overlaps

            # use HPMC overlap check
            for p, r, q in zip(system.particles, positions, orientations):
                p.position = r
                p.orientation = q

            dumbbell_overlaps = False
            run(0)
            if mc.count_overlaps() > 0:
                dumbbell_overlaps = True

            # verify agreement on configurations with overlaps
            self.assertEqual(sphere_overlaps, dumbbell_overlaps);

        del p
        del mc
        del system
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
