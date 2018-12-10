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

## Helper function to check if two convex polyhedron dimers overlap by HAND (treating each dimer as separate polyhedra)
# \param system hoomd system
# \param verts list of vertices for all cubes
# \param Ap 2x3 list of cube coordinates for first dimer
# \param Aq 2x4 list of cube quaternions for first dimer
# \param Bp 2x3 list of cube coordinates for second dimer
# \param Bq 2x4 list of cube quaternions for second dimer
# \returns True if dimers overlap, else False
def dimer_overlap_manual(system, verts, Ap, Aq, Bp, Bq):
    # for restoration later
    backup = system.take_snapshot()

    mc = hpmc.integrate.convex_polyhedron(seed=27)
    mc.shape_param.set('A', vertices=verts)

    result = False

    for r_i, q_i in zip(numpy.asarray(Ap, dtype=float), numpy.asarray(Aq, dtype=float)):
        for r_j, q_j in zip(numpy.asarray(Bp, dtype=float), numpy.asarray(Bq, dtype=float)):
            system.particles[0].position = r_i
            system.particles[0].orientation = q_i
            system.particles[1].position = r_j
            system.particles[1].orientation = q_j

            # we are only concerned with the overlap between the 0th and 1st particles.
            # I couldn't find a more elegant way to do this
            for m in range(2, len(system.particles)):
                system.particles[m].position = [0,0,0]
                system.particles[m].orientation = [1,0,0,0]

            # test for overlaps
            run(0, quiet=True)
            overlap_map = mc.map_overlaps()

            if overlap_map[0][1] > 0:
                result = True;

    # clean up
    del mc
    system.restore_snapshot(backup)

    return result

## Helper function to check if two convex polyhedron dimers overlap using the convex_polyhedron_union methods
# \param system hoomd system
# \param verts list of vertices for all cubes
# \param Ap 2x3 list of cube coordinates for first dimer
# \param Aq 2x4 list of cube quaternions for first dimer
# \param Bp 2x3 list of cube coordinates for second dimer
# \param Bq 2x4 list of cube quaternions for second dimer
# \returns True if dimers overlap, else False
def dimer_overlap_union(system, verts, Ap, Aq, Bp, Bq):
    # for restoration later
    backup = system.take_snapshot()

    mc = hpmc.integrate.convex_spheropolyhedron_union(seed=27);
    mc.shape_param.set("A", vertices=[verts, verts], centers=Ap, orientations=Aq);
    mc.shape_param.set("B", vertices=[verts, verts], centers=Bp, orientations=Bq);

    result = False

    # set the particle unions at the origin with no rotation, since locations of dimer constituents
    # has already been taken care of
    system.particles[0].position = [0,0,0]
    system.particles[0].orientation = [1,0,0,0]
    system.particles[0].type = "A"
    system.particles[1].position = [0,0,0]
    system.particles[1].orientation = [1,0,0,0]
    system.particles[1].type = "B"

    # we are only concerned with the overlap between the 0th and 1st particles.
    # I couldn't find a more elegant way to do this
    for m in range(2, len(system.particles)):
        system.particles[m].position = [0,0,0]
        system.particles[m].orientation = [1,0,0,0]
        system.particles[m].type = "A"

    # test for overlaps
    run(0, quiet=True)
    overlap_map = mc.map_overlaps()

    if overlap_map[0][1] > 0:
        result = True

    # clean up
    del mc
    system.restore_snapshot(backup)
    return result

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

# Multiply two quaternions
# From util.py
# \param q1 quaternion
# \param q2 quaternion
# \returns q1*q2
def quatMult(q1, q2):
    s = q1[0]
    v = q1[1:]
    t = q2[0]
    w = q2[1:]
    q = numpy.empty((4,), dtype=float)
    q[0] = s*t - numpy.dot(v, w)
    q[1:] = s*w + t*v + numpy.cross(v,w)
    return q


class helper_functions(unittest.TestCase):
    def setUp(self):
        box = data.boxdim(L=10, dimensions=3)
        self.system = create_empty(N=2, box=box, particle_types=['A'])
        context.current.sorter.set_params(grid=8)

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

    def test_dimer_overlap(self):
        # half side length of cubes
        hs = 0.5
        cube_verts = [[-hs,-hs,-hs],
                      [-hs,-hs,hs],
                      [-hs,hs,-hs],
                      [-hs,hs,hs],
                      [hs,-hs,-hs],
                      [hs,-hs,hs],
                      [hs,hs,-hs],
                      [hs,hs,hs]]

        # use this for testing an alternate simulation context
        tmp_sim = context.SimulationContext()
        with tmp_sim:
            system = create_empty(N=2, box=data.boxdim(L=10, dimensions=3), particle_types=['A','B'])

        Aq = numpy.array([[1,0,0,0],[1,0,0,0]])
        Bq = numpy.array(Aq)

        Ap = numpy.array([[-hs, 0, 0], [hs, 0, 0]])
        Bp = numpy.array(Ap)

        self.assertTrue(dimer_overlap_manual(self.system, cube_verts, Ap, Aq, Bp, Bq))
        with tmp_sim:
            self.assertTrue(dimer_overlap_union(system, cube_verts, Ap, Aq, Bp, Bq))

        Bp = Ap + [0, 2*hs-0.01, 0]
        self.assertTrue(dimer_overlap_manual(self.system, cube_verts, Ap, Aq, Bp, Bq))
        with tmp_sim:
            self.assertTrue(dimer_overlap_union(system, cube_verts, Ap, Aq, Bp, Bq))

        Bp = Ap + [0, 2*hs+0.01, 0]
        self.assertTrue(not dimer_overlap_manual(self.system, cube_verts, Ap, Aq, Bp, Bq))
        with tmp_sim:
            self.assertTrue(not dimer_overlap_union(system, cube_verts, Ap, Aq, Bp, Bq))

        Bp = Ap + [4*hs-0.01, 0, 0]
        self.assertTrue(dimer_overlap_manual(self.system, cube_verts, Ap, Aq, Bp, Bq))
        with tmp_sim:
            self.assertTrue(dimer_overlap_union(system, cube_verts, Ap, Aq, Bp, Bq))

        Bp = Ap + [4*hs+0.01, 0, 0]
        self.assertTrue(not dimer_overlap_manual(self.system, cube_verts, Ap, Aq, Bp, Bq))
        with tmp_sim:
            self.assertTrue(not dimer_overlap_union(system, cube_verts, Ap, Aq, Bp, Bq))

        Bp = Ap + [-2*hs, 0, 2*hs-0.01]
        self.assertTrue(dimer_overlap_manual(self.system, cube_verts, Ap, Aq, Bp, Bq))
        with tmp_sim:
            self.assertTrue(dimer_overlap_union(system, cube_verts, Ap, Aq, Bp, Bq))

        Bp = Ap + [-2*hs, 0, 2*hs+0.01]
        self.assertTrue(not dimer_overlap_manual(self.system, cube_verts, Ap, Aq, Bp, Bq))
        with tmp_sim:
            self.assertTrue(not dimer_overlap_union(system, cube_verts, Ap, Aq, Bp, Bq))

        # test differing orientations
        Aq = numpy.array([[1,0,0,0],[1,0,0,0]])
        # rotate cubes in second dimer by 90 degrees about the x axis. shouldn't make a difference.
        Bq = numpy.array([[numpy.cos(numpy.pi/2./2.),numpy.sin(numpy.pi/2./2.),0.,0.],
                          [numpy.cos(numpy.pi/2./2.),numpy.sin(numpy.pi/2./2.),0.,0.]])

        Ap = numpy.array([[-hs, 0, 0], [hs, 0, 0]])
        Bp = numpy.array(Ap)
        self.assertTrue(dimer_overlap_manual(self.system, cube_verts, Ap, Aq, Bp, Bq))
        with tmp_sim:
            self.assertTrue(dimer_overlap_union(system, cube_verts, Ap, Aq, Bp, Bq))


    def tearDown(self):
        del self.system
        context.initialize()

class shape_union(unittest.TestCase):
    def setUp(self):
        # number of dumbbells, size of box, and number of iterations are chosen empirically for
        #  * a good mix of overlapping and non-overlapping configurations
        #  * a reasonable opportunity to detect errors (early testing had errors that did not appear until iteration 15)
        #  * reasonable run time
        # Run time could be improved with better code in the brute force overlap check (such as vectorizing dumbbell_overlap()
        # and and moving the N^2 loop to C via numpy, or turning the N^2 loop into an O(N) loop by pre-filtering neighbors),
        # but this might cost readability and developer time better spent elsewhere.
        #
        self.N = 90        # number of dumbbells
        self.L = 40        # box edge length
        self.seed = 27      # PRNG seed
        self.ndim = 3

        self.system = create_empty(N=self.N, box=data.boxdim(L=self.L, dimensions=self.ndim), particle_types=['A'])
        # decrease initialization time with smaller grid for Hilbert curve
        context.current.sorter.set_params(grid=8)

    # Test randomly generated dumbbells
    # 1) generate N dumbbell positions and orientations
    # 2) place 2N spheres and determine if system has overlapping dumbbells
    # 3) place N dumbbells and determine if system has overlapping dumbbells
    # 4) confirm agreement between tests
    # 5) wash, rinse, repeat
    def test_dumbbells (self):
        spheres = numpy.array([[-0.5, 0, 0],[0.5, 0, 0]])      # positions of spheres in dumbbell coordinates

        # use fixed seed
        numpy.random.seed(self.seed)

        self.mc = hpmc.integrate.sphere_union(seed=self.seed);
        self.mc.shape_param.set("A", diameters=[1.0, 1.0], centers=spheres);

        num_iter = 50 # number of times to generate new configurations
        for i in range(num_iter):
            # randomly create "dumbbells" as pairs of spheres located anywhere in the box
            positions = (numpy.random.random((self.N,self.ndim)) - 0.5) * self.L  # positions of dumbbells in box
            # not uniformly sampling orientations, but that's okay
            orientations = numpy.random.random((self.N,4)) - 0.5
            # normalize to unit quaternions
            o2 = numpy.einsum('ij,ij->i', orientations, orientations)
            orientations = numpy.einsum('ij,i->ij', orientations, 1./numpy.sqrt(o2)) # orientations of dumbbells

            dumbbell = numpy.array([[quatRot(q, spheres[0]) + r, quatRot(q, spheres[1]) + r] for r,q in zip(positions, orientations)])

            # perform brute force overlap check
            overlaps = False
            for i in range(self.N-1):
                for j in range(i+1, self.N):
                    if dumbbell_overlap(dumbbell[i], dumbbell[j], [self.L,self.L,self.L]):
                        overlaps = True
                        break
                if overlaps == True:
                    break
            sphere_overlaps = overlaps

            # use HPMC overlap check
            for p, r, q in zip(self.system.particles, positions, orientations):
                p.position = r
                p.orientation = q

            dumbbell_overlaps = False
            run(0, quiet=True)
            if self.mc.count_overlaps() > 0:
                dumbbell_overlaps = True

            # verify agreement on configurations with overlaps
            self.assertEqual(sphere_overlaps, dumbbell_overlaps);

        del p
        del self.mc

    # Test randomly generated dumbbells (special case of faceted ellipsoid)
    def test_dumbbells_fellipsoid (self):
        spheres = numpy.array([[-0.5, 0, 0],[0.5, 0, 0]])      # positions of spheres in dumbbell coordinates

        # use fixed seed
        numpy.random.seed(self.seed)

        self.mc = hpmc.integrate.faceted_ellipsoid_union(seed=self.seed);
        self.mc.shape_param.set("A", axes=[(0.5,0.5,0.5)]*2, offsets=[[]]*2, normals=[[]]*2,vertices=[[]]*2,
            centers=spheres, orientations=[(1,0,0,0)]*2);

        num_iter = 50 # number of times to generate new configurations
        for i in range(num_iter):
            # randomly create "dumbbells" as pairs of spheres located anywhere in the box
            positions = (numpy.random.random((self.N,self.ndim)) - 0.5) * self.L  # positions of dumbbells in box
            # not uniformly sampling orientations, but that's okay
            orientations = numpy.random.random((self.N,4)) - 0.5
            # normalize to unit quaternions
            o2 = numpy.einsum('ij,ij->i', orientations, orientations)
            orientations = numpy.einsum('ij,i->ij', orientations, 1./numpy.sqrt(o2)) # orientations of dumbbells

            dumbbell = numpy.array([[quatRot(q, spheres[0]) + r, quatRot(q, spheres[1]) + r] for r,q in zip(positions, orientations)])

            # perform brute force overlap check
            overlaps = False
            for i in range(self.N-1):
                for j in range(i+1, self.N):
                    if dumbbell_overlap(dumbbell[i], dumbbell[j], [self.L,self.L,self.L]):
                        overlaps = True
                        break
                if overlaps == True:
                    break
            sphere_overlaps = overlaps

            # use HPMC overlap check
            for p, r, q in zip(self.system.particles, positions, orientations):
                p.position = r
                p.orientation = q

            dumbbell_overlaps = False
            run(0, quiet=True)
            if self.mc.count_overlaps() > 0:
                dumbbell_overlaps = True

            # verify agreement on configurations with overlaps
            self.assertEqual(sphere_overlaps, dumbbell_overlaps);

        del p
        del self.mc


    # Test randomly generated cube dimers
    # 1) generate N cube dimer positions and orientations
    # 2) place 2N cubes and determine if system has overlapping cube dimers
    # 3) place N cube dimers and determine if system has overlapping cube dimers
    # 4) confirm agreement between tests
    # 5) wash, rinse, repeat
    def test_dimers (self):
        # half side length of cubes
        hs = 0.5
        cube_verts = [[-hs,-hs,-hs],
                      [-hs,-hs,hs],
                      [-hs,hs,-hs],
                      [-hs,hs,hs],
                      [hs,-hs,-hs],
                      [hs,-hs,hs],
                      [hs,hs,-hs],
                      [hs,hs,hs]]

        box = data.boxdim(L=self.L, dimensions=self.ndim)

        # positions of cubes in dimer coordinates
        cubes = numpy.array([[-hs, 0, 0],[hs, 0, 0]])
        # rotate cubes in dimer by 90 degrees about the x axis. shouldn't make a difference.
        cube_ors = numpy.array([[numpy.cos(numpy.pi/2./2.),numpy.sin(numpy.pi/2./2.),0.,0.],
                                [numpy.cos(numpy.pi/2./2.),numpy.sin(numpy.pi/2./2.),0.,0.]])

        # use fixed seed
        numpy.random.seed(self.seed)

        # only perform the test 5 times, rather than 50 in the case of test_dumbbells.
        num_iter = 5
        for i in range(num_iter):
            # randomly create "dumbbells" as pairs of cubes located anywhere in the box
            positions = (numpy.random.random((self.N, self.ndim)) - 0.5) * self.L  # positions of dimers in box
            # not uniformly sampling orientations, but that's okay
            orientations = numpy.random.random((self.N,4)) - 0.5
            # normalize to unit quaternions
            o2 = numpy.einsum('ij,ij->i', orientations, orientations)
            orientations = numpy.einsum('ij,i->ij', orientations, 1./numpy.sqrt(o2))

            dimer_pos = numpy.array([[quatRot(q, cubes[0]) + r, quatRot(q, cubes[1]) + r] for r,q in zip(positions, orientations)])
            dimer_ors = numpy.array([[quatMult(q, cube_ors[0]), quatMult(q, cube_ors[1])] for q in orientations])

            # perform brute force overlap check
            #backup = self.system.take_snapshot()

            tmp_sim = context.SimulationContext()
            with tmp_sim:
                system = create_empty(N=2, box=data.boxdim(L=self.L, dimensions=self.ndim), particle_types=['A'])
                overlaps = False
                for m in range(self.N-1):
                    for n in range(m+1, self.N):
                        #print(dimer_overlap(system, cube_verts, dimer_pos[i], dimer_ors[i], dimer_pos[j], dimer_ors[j]))
                        if dimer_overlap_manual(system, cube_verts, dimer_pos[m], dimer_ors[m], dimer_pos[n], dimer_ors[n]):
                            overlaps = True
                            break
                    if overlaps == True:
                        break

            #self.system.restore_snapshot(backup)
            cube_overlaps = overlaps

            self.mc = hpmc.integrate.convex_spheropolyhedron_union(seed=self.seed);
            self.mc.shape_param.set("A", vertices=[cube_verts, cube_verts], centers=cubes, orientations=cube_ors);

            # use HPMC overlap check
            for p, r, q in zip(self.system.particles, positions, orientations):
                p.position = r
                p.orientation = q

            dimer_overlaps = False
            run(0, quiet=True)
            if self.mc.count_overlaps() > 0:
                dimer_overlaps = True

            # verify agreement on configurations with overlaps
            self.assertEqual(cube_overlaps, dimer_overlaps);

        del p
        del tmp_sim
        del self.mc

    # Test randomly generated dumbbells (with spheropolyhedron shape)
    # 1) generate N dumbbell positions and orientations
    # 2) place 2N spheres and determine if system has overlapping dumbbells
    # 3) place N dumbbells and determine if system has overlapping dumbbells
    # 4) confirm agreement between tests
    # 5) wash, rinse, repeat
    def test_dumbbells_spheropolyhedron (self):
        spheres = numpy.array([[-0.5, 0, 0],[0.5, 0, 0]])      # positions of spheres in dumbbell coordinates

        # use fixed seed
        numpy.random.seed(self.seed)

        self.mc = hpmc.integrate.convex_spheropolyhedron_union(seed=self.seed);
        self.mc.shape_param.set("A", sweep_radii=[0.5, 0.5], centers=spheres, orientations=[(1,0,0,0),(1,0,0,0)], vertices=[[(0,0,0)],[(0,0,0)]]);

        num_iter = 50 # number of times to generate new configurations
        for i in range(num_iter):
            # randomly create "dumbbells" as pairs of spheres located anywhere in the box
            positions = (numpy.random.random((self.N,self.ndim)) - 0.5) * self.L  # positions of dumbbells in box
            # not uniformly sampling orientations, but that's okay
            orientations = numpy.random.random((self.N,4)) - 0.5
            # normalize to unit quaternions
            o2 = numpy.einsum('ij,ij->i', orientations, orientations)
            orientations = numpy.einsum('ij,i->ij', orientations, 1./numpy.sqrt(o2)) # orientations of dumbbells

            dumbbell = numpy.array([[quatRot(q, spheres[0]) + r, quatRot(q, spheres[1]) + r] for r,q in zip(positions, orientations)])

            # perform brute force overlap check
            overlaps = False
            for i in range(self.N-1):
                for j in range(i+1, self.N):
                    if dumbbell_overlap(dumbbell[i], dumbbell[j], [self.L,self.L,self.L]):
                        overlaps = True
                        break
                if overlaps == True:
                    break
            sphere_overlaps = overlaps

            # use HPMC overlap check
            for p, r, q in zip(self.system.particles, positions, orientations):
                p.position = r
                p.orientation = q

            dumbbell_overlaps = False
            run(0, quiet=True)
            if self.mc.count_overlaps() > 0:
                dumbbell_overlaps = True

            # verify agreement on configurations with overlaps
            self.assertEqual(sphere_overlaps, dumbbell_overlaps);

        del p
        del self.mc


    def tearDown(self):
        del self.system
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
