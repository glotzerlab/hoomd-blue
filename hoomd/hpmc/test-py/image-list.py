from __future__ import division, print_function
from hoomd import *
from hoomd import hpmc
import unittest
import os
import numpy as np
import random

context.initialize()

def create_empty(**kwargs):
    snap = data.make_snapshot(**kwargs);
    return init.read_snapshot(snap);

random.seed(10)
np.random.seed(10)
rectangle = np.array([(-1.0,-0.5, 0), (1.0, -0.5, 0), (1.0, 0.5, 0), (-1.0, 0.5, 0)])
poly = np.array([(-1.0,-0.5, -0.5),
                  (1.0, -0.5, -0.5),
                  (1.0, 0.5, -0.5),
                  (-1.0, 0.5, -0.5),
                  (-1.0,-0.5, 0.5),
                  (1.0, -0.5, 0.5),
                  (1.0, 0.5, 0.5),
                  (-1.0, 0.5, 0.5)])

def quatMult(q1, q2):
    # per http://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation
    s = q1[0]
    v = q1[1:]
    t = q2[0]
    w = q2[1:]
    q = np.empty((4,), dtype=float)
    q[0] = s*t - np.dot(v, w)
    q[1:] = s*w + t*v + np.cross(v,w)
    return q

def quatRot(q, v):
    v = np.asarray(v)
    q = np.asarray(q)
    # assume q is a unit quaternion
    w = q[0]
    r = q[1:]
    vnew = np.empty((3,), dtype=v.dtype)
    return v + 2*np.cross(r, np.cross(r,v) + w*v)

# This test ensures that the image list construction algorithm is sufficiently robust.
#
# The two likeliest failure modes are presumed to be
# 1) an insufficiently far-reaching list of images
# 2) some sort of logic error associated with optimizations that prevent certain particle pairs from being checked.
#
# The first failure mode is likeliest to occur with small, dense systems of high aspect ratio particles, but also
# possible with low aspect ratio particles in a high aspect ratio box. For a one-particle system, it is easy to see that
# the two situations can be mapped to each other by a linear transformation. The easiest way to construct such a system
# is to construct a simple system with a known overlap occurring across the periodic boundary and then to find
# equivalent lattices which cause the overlap to occur at a further image. Equivalent lattices can be created randomly
# to search a large parameter space.
#
# As the box becomes more distorted, the circumsphere of the original particle becomes small relative to the box edge
# lengths, which may obscure some bugs. This can be addressed by stretching the box along a random vector and then
# applying the same transformation to the original particle shape.
#
# The second failure mode can be checked by constructing a system as above but with multiple particles and permute the
# particle numbers such that cases are tested for both cases of overlaps only occurring for particles i < j and for
# particles i > j, where i is a particle in the primary box and j is a particle in an image box.
#
# Success condition: count_overlaps returns the correct integer value.

# Given a set of lattice vectors, return a randomly generated distorted but equivalent set of lattice vectors.
def rand_equiv_lattice(a1=[1.,0.,0.], a2=[0.,1.,0.], a3=[0.,0.,1.], ndim=3, iterations=5):
    a = np.asarray([a1, a2, a3])
    for trial in range(iterations):
        # pick a lattice vector and direction by which to shear box and apply to remaining lattice vectors
        i = random.randint(0, ndim-1)
        e = a[i] * random.choice([-1,1])
        a[np.arange(3) != i] += e
    b1, b2, b3 = a[0:3]
    return b1, b2, b3

# Given a set of lattice vectors, rotate to produce an upper triangular right-handed box
# as a hoomd boxdim object and a rotation quaternion that brings particles in the original coordinate system to the new one.
# Mirroring shouldn't be necessary since rand_equiv_lattice() and randDistortion() preserve the handedness of the input.
# (E.g. returns (data.boxdim(Lx=10, Ly=20, Lz=30, xy=1.0, xz=0.5, yz=0.1), q) )
def latticeToHoomd(a1, a2, a3=[0.,0.,1.], ndim=3):
    xhat = np.array([1.,0.,0.])
    yhat = np.array([0.,1.,0.])
    zhat = np.array([0.,0.,1.])

    # Find quaternion to rotate first lattice vector to x axis
    a1mag = np.sqrt(np.dot(a1,a1))
    v1 = a1/a1mag + xhat
    v1mag = np.sqrt(np.dot(v1,v1))
    if v1mag > 1e-6:
        u1 = v1/v1mag
    else:
        # a1 is antialigned with xhat, so rotate around any unit vector perpendicular to xhat
        u1 = yhat
    q1 = np.concatenate(([np.cos(np.pi/2)], np.sin(np.pi/2)*u1))

    # Find quaternion to rotate second lattice vector to xy plane after applying above rotation
    a2prime = quatRot(q1, a2)
    angle = -1*np.arctan2(a2prime[2], a2prime[1])
    q2 = np.concatenate(([np.cos(angle/2)], np.sin(angle/2)*xhat))

    q = quatMult(q2,q1)

    Lx = np.sqrt(np.dot(a1, a1))
    a2x = np.dot(a1, a2) / Lx
    Ly = np.sqrt(np.dot(a2,a2) - a2x*a2x)
    xy = a2x / Ly
    v0xv1 = np.cross(a1, a2)
    v0xv1mag = np.sqrt(np.dot(v0xv1, v0xv1))
    Lz = np.dot(a3, v0xv1) / v0xv1mag
    a3x = np.dot(a1, a3) / Lx
    xz = a3x / Lz
    yz = (np.dot(a2, a3) - a2x*a3x) / (Ly*Lz)

    box = data.boxdim(Lx=Lx, Ly=Ly, Lz=Lz, xy=xy, xz=xz, yz=yz, dimensions=ndim)

    return box, q

# Generate a random distortion transformation matrix that can be applied to a box matrix or particle vertices.
# Returns a numpy matrix object.
def randDistortion(ndim=3):
    transform = np.identity(3)
    # Want to keep a valid HOOMD box, so leave first axis untouched. Distort second axis
    # without affecting handedness.
    # Deform second axis, keeping it in the xy plane
    transform[0,1] = (np.random.rand() + 0.5)*np.random.choice([-1,1])
    transform[1,1] = np.random.rand() + 0.5
    if ndim == 3:
        # Deform third axis, preserving handedness
        transform[0,2] = (np.random.rand() + 0.5)*np.random.choice([-1,1])
        transform[1,2] = (np.random.rand() + 0.5)*np.random.choice([-1,1])
        transform[2,2] = np.random.rand() + 0.5
    return np.matrix(transform)

def matFromBox(box):
    Lx, Ly, Lz = box.Lx, box.Ly, box.Lz
    xy = box.xy
    xz = box.xz
    yz = box.yz
    return np.matrix([[Lx, xy*Ly, xz*Lz], [0, Ly, yz*Lz], [0, 0, Lz]])

# Test latticeToHoomd() function
class helper_function_sanity_check (unittest.TestCase):
    def test_x_with_z_component(self):
        a1 = np.asarray([1.,0,0])
        a2 = np.asarray([0.,1.,0.])
        a3 = np.asarray([0.,0.,1.])
        box, q = latticeToHoomd(a1+a3,a2,a3)
        vecs = np.asarray(matFromBox(box)).transpose()
        v1 = quatRot(q,a1+a3)
        v2 = vecs[0]
        v = v2 - v1
        self.assertLess(np.abs(np.dot(v,v)), 1e-6)

    def test_x_with_y_component(self):
        a1 = np.asarray([1.,0,0])
        a2 = np.asarray([0.,1.,0.])
        a3 = np.asarray([0.,0.,1.])
        box, q = latticeToHoomd(a1+a2,a2,a3)
        vecs = np.asarray(matFromBox(box)).transpose()
        v1 = quatRot(q,a1+a2)
        v2 = vecs[0]
        v = v2 - v1
        self.assertLess(np.abs(np.dot(v,v)), 1e-6)

    def test_y_with_z_component(self):
        a1 = np.asarray([1.,0,0])
        a2 = np.asarray([0.,1.,0.])
        a3 = np.asarray([0.,0.,1.])
        box, q = latticeToHoomd(a1,a2+2*a3,a3)
        vecs = np.asarray(matFromBox(box)).transpose()
        v1 = quatRot(q,a2+2*a3)
        v2 = vecs[1]
        v = v2 - v1
        self.assertLess(np.abs(np.dot(v,v)), 1e-6)

    def test_z_with_y_component(self):
        a1 = np.asarray([1.,0,0])
        a2 = np.asarray([0.,1.,0.])
        a3 = np.asarray([0.,0.,1.])
        box, q = latticeToHoomd(a1,a2,a2+a3)
        vecs = np.asarray(matFromBox(box)).transpose()
        v1 = quatRot(q,a2+a3)
        v2 = vecs[2]
        v = v2 - v1
        self.assertLess(np.abs(np.dot(v,v)), 1e-6)

    def test_handedness(self):
        rherror = 0
        for i in range(100):
            b1, b2, b3 = rand_equiv_lattice(ndim=2, iterations=15)
            if np.dot(np.cross(b1,b2),b3) <= 0:
                rherror += 1
        self.assertEquals(rherror, 0)
        for i in range(100):
            b1, b2, b3 = rand_equiv_lattice(iterations=15)
            if np.dot(np.cross(b1,b2),b3) <= 0:
                rherror += 1

## 2d Tests

# Test single spheres in a small box.
class imagelist2d_test1 (unittest.TestCase):
    def test_overlaps(self):
        for i in range(100):
            a1 = [0.99,0.,0.]
            a2 = [0.,1.01,0.]
            a3 = [0.,0.,1.01]

            b1, b2, b3 = rand_equiv_lattice(a1, a2, a3, ndim=2)
            box, q = latticeToHoomd(b1, b2, b3, ndim=2)
            self.system = create_empty(N=1, box=box, particle_types=['A'])

            self.mc = hpmc.integrate.sphere(seed=10, d=0.0);
            self.mc.shape_param.set("A", diameter=1.0);

            # verify that overlaps are detected
            run(0)
            self.assertGreater(self.mc.count_overlaps(), 0);

            del self.mc
            del self.system
            context.initialize();

    def test_no_overlaps(self):
        for i in range(100):
            a1 = [0.99,0.,0.]
            a2 = [0.,1.01,0.]
            a3 = [0.,0.,1.01]

            b1, b2, b3 = rand_equiv_lattice(a1, a2, a3, ndim=2)
            box, q = latticeToHoomd(b1, b2, b3, ndim=2)
            self.system = create_empty(N=1, box=box, particle_types=['A'])

            self.mc = hpmc.integrate.sphere(seed=10, d=0.0);
            self.mc.shape_param.set("A", diameter=1.0);

            # verify that overlaps aren't detected
            self.mc.shape_param.set("A", diameter=0.98)
            run(0)
            self.assertEquals(self.mc.count_overlaps(), 0);

            del self.mc
            del self.system
            context.initialize();

# Test single rectangles in a suboptimal box.
class imagelist2d_test2 (unittest.TestCase):
    def test_test3_sanity(self):
        for i in range(100):
            a1 = [2.01,0.,0.]
            a2 = [0.,1.01,0.]
            a3 = [0.,0.,1.0]
            b1, b2, b3 = rand_equiv_lattice(a1, a2, a3, ndim=2)
            box, q = latticeToHoomd(b1, b2, b3, ndim=2)
            self.system = create_empty(N=1, box=box, particle_types=['A'])
            self.system.particles[0].orientation = q

            self.mc = hpmc.integrate.convex_polygon(seed=10, d=0.0);
            self.mc.shape_param.set("A", vertices=rectangle);

            # verify that overlaps are detected
            run(0)
            self.assertEquals(self.mc.count_overlaps(), 0);

            del self.mc
            del self.system
            context.initialize();

# Test single rectangles in a suboptimal box.
class imagelist2d_test3 (unittest.TestCase):
    def test_image_list(self):
        for i in range(100):
            a1 = [1.99,0.,0.]
            a2 = [0.,1.01,0.]
            a3 = [0.,0.,1.0]
            b1, b2, b3 = rand_equiv_lattice(a1, a2, a3, ndim=2)
            box, q = latticeToHoomd(b1, b2, b3, ndim=2)
            self.system = create_empty(N=1, box=box, particle_types=['A'])
            self.system.particles[0].orientation = q

            self.mc = hpmc.integrate.convex_polygon(seed=10, d=0.0);
            self.mc.shape_param.set("A", vertices=rectangle);

            # verify that overlaps are detected
            run(0)
            self.assertGreater(self.mc.count_overlaps(), 0);

            del self.mc
            del self.system
            context.initialize();

# Test single parallelepipeds in a distorted box to make sure extra overlaps aren't detected.
class imagelist2d_test4 (unittest.TestCase):
    def test_test5_sanity(self):
        for i in range(100):
            a1 = [2.01,0.,0.]
            a2 = [0.,1.01,0.]
            a3 = [0.,0.,1.0]

            verts = np.array(rectangle)
            T = randDistortion(ndim=2)
            a1 = np.asarray(T*np.matrix(a1).transpose()).reshape((3,))
            a2 = np.asarray(T*np.matrix(a2).transpose()).reshape((3,))
            a3 = np.asarray(T*np.matrix(a3).transpose()).reshape((3,))
            for i in range(len(verts)):
                verts[i] = np.asarray(T * verts[i].reshape((3,1))).reshape((3,))

            b1, b2, b3 = rand_equiv_lattice(a1, a2, a3, ndim=2)
            box, q = latticeToHoomd(b1, b2, b3, ndim=2)
            self.system = create_empty(N=1, box=box, particle_types=['A'])
            self.system.particles[0].orientation = q

            self.mc = hpmc.integrate.convex_polygon(seed=10, d=0.0);
            self.mc.shape_param.set("A", vertices=verts);

            # verify that no overlaps are detected
            run(0)
            self.assertEquals(self.mc.count_overlaps(), 0);

            del self.mc
            del self.system
            context.initialize();

# Test single parallelepipeds in a distorted box to make sure overlaps are detected.
class imagelist2d_test5 (unittest.TestCase):
    def test_image_list(self):
        for i in range(100):
            a1 = [1.99,0.,0.]
            a2 = [0.,1.01,0.]
            a3 = [0.,0.,1.0]

            verts = np.array(rectangle)
            T = randDistortion(ndim=2)
            a1 = np.asarray(T*np.matrix(a1).transpose()).reshape((3,))
            a2 = np.asarray(T*np.matrix(a2).transpose()).reshape((3,))
            a3 = np.asarray(T*np.matrix(a3).transpose()).reshape((3,))
            for i in range(len(verts)):
                verts[i] = np.asarray(T * verts[i].reshape((3,1))).reshape((3,))

            b1, b2, b3 = rand_equiv_lattice(a1, a2, a3, ndim=2)
            box, q = latticeToHoomd(b1, b2, b3, ndim=2)
            self.system = create_empty(N=1, box=box, particle_types=['A'])
            self.system.particles[0].orientation = q

            self.mc = hpmc.integrate.convex_polygon(seed=10, d=0.0);
            self.mc.shape_param.set("A", vertices=verts);

            # verify that overlaps are detected
            run(0)
            self.assertGreater(self.mc.count_overlaps(), 0);

            del self.mc
            del self.system
            context.initialize();

###########
## 3d Tests

# Test single spheres in a small box.
class imagelist3d_test1 (unittest.TestCase):
    def test_image_list(self):
        for i in range(100):
            a1 = [0.99,0.,0.]
            a2 = [0.,1.01,0.]
            a3 = [0.,0.,1.01]
            b1, b2, b3 = rand_equiv_lattice(a1, a2, a3, ndim=3)
            box, q = latticeToHoomd(b1, b2, b3, ndim=3)
            self.system = create_empty(N=1, box=box, particle_types=['A'])

            self.mc = hpmc.integrate.sphere(seed=10, d=0.0);
            self.mc.shape_param.set("A", diameter=1.0);

            # verify that overlaps are detected
            run(0)
            self.assertGreater(self.mc.count_overlaps(), 0);

            del self.mc
            del self.system
            context.initialize();

# Test single rectangular prism in a suboptimal box.
class imagelist3d_test3 (unittest.TestCase):
    def test_test3_sanity(self):
        for i in range(100):
            a1 = [2.01,0.,0.]
            a2 = [0.,1.01,0.]
            a3 = [0.,0.,1.01]
            b1, b2, b3 = rand_equiv_lattice(a1, a2, a3, ndim=3)
            box, q = latticeToHoomd(b1, b2, b3, ndim=3)
            self.system = create_empty(N=1, box=box, particle_types=['A'])
            self.system.particles[0].orientation = q

            verts = np.array(poly)
            self.mc = hpmc.integrate.convex_polyhedron(seed=10, d=0.0);
            self.mc.shape_param.set("A", vertices=verts);

            # verify that overlaps are detected
            run(0)
            self.assertEquals(self.mc.count_overlaps(), 0);

            del self.mc
            del self.system
            context.initialize();

# Test single rectangular prism in a suboptimal box.
class imagelist3d_test3 (unittest.TestCase):
    def test_image_list(self):
        for i in range(100):
            a1 = [1.99,0.,0.]
            a2 = [0.,1.01,0.]
            a3 = [0.,0.,1.01]
            b1, b2, b3 = rand_equiv_lattice(a1, a2, a3, ndim=3)
            box, q = latticeToHoomd(b1, b2, b3, ndim=3)
            self.system = create_empty(N=1, box=box, particle_types=['A'])
            self.system.particles[0].orientation = q

            verts = np.array(poly)
            self.mc = hpmc.integrate.convex_polyhedron(seed=10, d=0.0);
            self.mc.shape_param.set("A", vertices=verts);

            # verify that overlaps are detected
            run(0)
            self.assertGreater(self.mc.count_overlaps(), 0);

            del self.mc
            del self.system
            context.initialize();

# Test single parallelepipeds in a distorted box to make sure there aren't extra overlaps
class imagelist3d_test4 (unittest.TestCase):
    def test_test5_sanity(self):
        for i in range(100):
            a1 = [2.01,0.,0.]
            a2 = [0.,1.01,0.]
            a3 = [0.,0.,1.01]

            verts = np.array(poly)
            T = randDistortion()
            a1 = np.asarray(T*np.matrix(a1).transpose()).reshape((3,))
            a2 = np.asarray(T*np.matrix(a2).transpose()).reshape((3,))
            a3 = np.asarray(T*np.matrix(a3).transpose()).reshape((3,))
            for i in range(len(verts)):
                verts[i] = np.asarray(T * verts[i].reshape((3,1))).reshape((3,))

            b1, b2, b3 = rand_equiv_lattice(a1, a2, a3, ndim=3)
            box, q = latticeToHoomd(b1, b2, b3, ndim=3)
            self.system = create_empty(N=1, box=box, particle_types=['A'])
            self.system.particles[0].orientation = q

            self.mc = hpmc.integrate.convex_polyhedron(seed=10, d=0.0);
            self.mc.shape_param.set("A", vertices=verts);

            self.verts = verts
            self.T = T

            # verify that no overlaps are detected
            run(0)
            self.assertEquals(self.mc.count_overlaps(), 0);

            del self.mc
            del self.system
            context.initialize();

# Test single parallelepipeds in a distorted box to make sure overlaps are detected.
class imagelist3d_test5 (unittest.TestCase):
    def test_image_list(self):
        for i in range(100):
            a1 = [1.99,0.,0.]
            a2 = [0.,1.01,0.]
            a3 = [0.,0.,1.01]

            verts = np.array(poly)
            T = randDistortion()
            a1 = np.asarray(T*np.matrix(a1).transpose()).reshape((3,))
            a2 = np.asarray(T*np.matrix(a2).transpose()).reshape((3,))
            a3 = np.asarray(T*np.matrix(a3).transpose()).reshape((3,))
            for i in range(len(verts)):
                verts[i] = np.asarray(T * verts[i].reshape((3,1))).reshape((3,))

            b1, b2, b3 = rand_equiv_lattice(a1, a2, a3, ndim=3)
            box, q = latticeToHoomd(b1, b2, b3, ndim=3)
            self.system = create_empty(N=1, box=box, particle_types=['A'])
            self.system.particles[0].orientation = q

            self.mc = hpmc.integrate.convex_polyhedron(seed=10, d=0.0);
            self.mc.shape_param.set("A", vertices=verts);

            # verify that overlaps are detected
            run(0)
            self.assertGreater(self.mc.count_overlaps(), 0);

            del self.mc
            del self.system
            context.initialize();

if __name__ == '__main__':
    # this test works on the CPU only and only on a single rank
    if comm.get_num_ranks() > 1:
        raise RuntimeError("This test only works on 1 rank");

    unittest.main(argv = ['test.py', '-v'])
