# Test the helper functions in hpmc.util
from __future__ import division, print_function
from hoomd import *
from hoomd import hpmc
import unittest
import tempfile
import os
import numpy as np

context.initialize()

def create_empty(**kwargs):
    snap = data.make_snapshot(**kwargs);
    return init.read_snapshot(snap);

Zr4Al3_pos = """boxMatrix    1.18455 0   4.3983  0.018416    4.5582  0.018416    -4.3983 0   -1.18455
def A1  "sphere 0.8 ff0000"
def A2  "sphere 0.8 0000ff"
A2  -2.791425   0   2.791425
A2  -2.791425   -2.297516   2.791425
A1  0.603183    1.148758    -2.10529
A1  -2.10529    1.111926    0.603183
A1  4.4408921E-16   1.130342    -0
A1  -0.891623   -1.148758   0.891623
A1  0.891623    -1.148758   -0.891623
eof
"""

sc2d_pos = """box 1.1 1.1 0
def A "sphere 1 ff0000"
A 0 0 0
"""

# Test latticeToHoomd() function
class latticeToHoomd (unittest.TestCase):
    def test_x_with_z_component(self):
        a1 = np.asarray([1.,0,0])
        a2 = np.asarray([0.,2.,0.])
        a3 = np.asarray([0.,0.,3.])
        a1 = a1+a3
        box, q = hpmc.util.latticeToHoomd(a1,a2,a3)
        vecs = np.asarray(hpmc.util.matFromBox(box)).transpose()
        v1 = hpmc.util.quatRot(q,a1)
        v2 = vecs[0]
        v = v2 - v1
        self.assertAlmostEqual(np.dot(v,v), 0, places=6)
        volume = np.dot(a1,np.cross(a2,a3))
        volume -= np.dot(vecs[0], np.cross(vecs[1],vecs[2]))
        self.assertAlmostEqual(volume, 0, places=6)

    def test_x_with_y_component(self):
        a1 = np.asarray([1.,0,0])
        a2 = np.asarray([0.,2.,0.])
        a3 = np.asarray([0.,0.,3.])
        a1 = a1+a2
        box, q = hpmc.util.latticeToHoomd(a1,a2,a3)
        vecs = np.asarray(hpmc.util.matFromBox(box)).transpose()
        v1 = hpmc.util.quatRot(q,a1)
        v2 = vecs[0]
        v = v2 - v1
        self.assertAlmostEqual(np.dot(v,v), 0, places=6)
        volume = np.dot(a1,np.cross(a2,a3))
        volume -= np.dot(vecs[0], np.cross(vecs[1],vecs[2]))
        self.assertAlmostEqual(volume, 0, places=6)

    def test_y_with_z_component(self):
        a1 = np.asarray([1.,0,0])
        a2 = np.asarray([0.,2.,0.])
        a3 = np.asarray([0.,0.,3.])
        a2 = a2+2*a3
        box, q = hpmc.util.latticeToHoomd(a1,a2,a3)
        vecs = np.asarray(hpmc.util.matFromBox(box)).transpose()
        v1 = hpmc.util.quatRot(q,a2)
        v2 = vecs[1]
        v = v2 - v1
        self.assertAlmostEqual(np.dot(v,v), 0, places=6)
        volume = np.dot(a1,np.cross(a2,a3))
        volume -= np.dot(vecs[0], np.cross(vecs[1],vecs[2]))
        self.assertAlmostEqual(volume, 0, places=6)

    def test_z_with_y_component(self):
        a1 = np.asarray([1.,1,0])
        a2 = np.asarray([-1.,2.,0.])
        a3 = np.asarray([0.,0.,3.])
        a3 = a2-a3
        box, q = hpmc.util.latticeToHoomd(a1,a2,a3)
        vecs = np.asarray(hpmc.util.matFromBox(box)).transpose()
        v1 = hpmc.util.quatRot(q,a3)
        v2 = vecs[2]
        v = v2 - v1
        self.assertAlmostEqual(np.dot(v,v), 0, places=6)
        volume = np.dot(a1,np.cross(a2,a3))
        volume -= np.dot(vecs[0], np.cross(vecs[1],vecs[2]))
        self.assertAlmostEqual(volume, 0, places=6)

    def test_handedness(self):
        for i in range(1000):
            a1, a2, a3 = np.random.random((3,3))
            box, q = hpmc.util.latticeToHoomd(a1,a2,a3)
            b1 = box.get_lattice_vector(0)
            b2 = box.get_lattice_vector(1)
            b3 = box.get_lattice_vector(2)
            self.assertAlmostEqual(np.dot(np.cross(a1,a2),a3), np.dot(np.cross(b1,b2),b3), places=5)

class read_pos (unittest.TestCase):
    def setUp(self):
        # create temporary pos file
        fd, self.fname = tempfile.mkstemp(suffix='read_pos_test.pos')
    # read a simple 2d simple cubic unit cell
    def test_trivial_2d(self):
        fh = open(self.fname, 'w')
        fh.write(sc2d_pos)
        fh.close()
        input = hpmc.util.read_pos(self.fname, ndim=2)
        self.assertEqual((input['positions'][0] == np.array([0,0,0])).all(), True)
        self.assertEqual(input['param_dict']['A']['shape'], 'sphere')
        self.assertEqual(input['param_dict']['A']['diameter'], 1.0)
        self.assertEqual(set(input['types']), set(['A']))
        self.assertEqual(input['box'].Ly, 1.1)
    def test_read_triclinic(self):
        fh = open(self.fname, 'w')
        fh.write(Zr4Al3_pos)
        fh.close()
        input = hpmc.util.read_pos(self.fname)
        self.assertEqual(input['param_dict']['A2']['shape'], 'sphere')
        self.assertEqual(input['param_dict']['A1']['diameter'], 0.8)
        self.assertEqual(set(input['types']), set(['A1','A2']))

        # compare volumes to see that the box hasn't been grossly distorted...
        bmatrix = Zr4Al3_pos.split('\n')[0].split()[1:]
        bmatrix = np.array([float(n) for n in bmatrix])
        bmatrix.resize((3,3))
        a1, a2, a3 = bmatrix.transpose()
        b1 = input['box'].get_lattice_vector(0)
        b2 = input['box'].get_lattice_vector(1)
        b3 = input['box'].get_lattice_vector(2)
        self.assertAlmostEqual(np.dot(a1,np.cross(a2,a3)), np.dot(b1, np.cross(b2,b3)), places=5)

        # check that q rotates a1 to b1
        q = input['q']
        v1 = hpmc.util.quatRot(q,a1)
        v2 = b1
        v12 = v2-v1
        self.assertAlmostEqual(np.dot(v12,v12), 0.0, places=5)

        # Check the first A1 particle position in original frame against A1 particle in new frame
        with open(self.fname, 'r') as fh:
            for line in fh:
                if line.startswith('A1'):
                    r1 = np.array([float(n) for n in line.rstrip().split()[-3:]])
                    break
        #print("len squared r1 = {}".format(np.dot(r1,r1)))
        i = input['types'].index('A1')
        r = input['positions'][i]
        #print("len squared new r = {}".format(np.dot(r,r)))
        q = input['q']
        qconj = q * [1,-1,-1,-1]
        r2 = hpmc.util.quatRot(qconj, r)
        r12 = r2-r1
        self.assertAlmostEqual(np.dot(r12,r12), 0.0, places=5)
    def tearDown(self):
        # remove pos file
        os.remove(self.fname)

# modeled on dense_pack example
class compressor (unittest.TestCase):
    def setUp(self):
        # create temporary log file
        fd, self.fname = tempfile.mkstemp()
        self.args = {'ptypes':['A'],
                     'pnums':[1],
                     'pvolumes':[4./3. * np.pi * 0.5**3],
                     'pverts':[],
                     'num_comp_steps':5e4,
                     'log_file':self.fname,
                     'pf_tol':0.01,
                     'relax':1e4}
    # one sphere should compress easily
    def test_1sphere(self):
        system = create_empty(N=1, box=data.boxdim(L=3), particle_types=['A'])
        mc = hpmc.integrate.sphere(seed=1)
        mc.set_params(d=0.1)
        mc.shape_param.set('A', diameter=1.0)
        npt = hpmc.update.boxmc(mc, betaP=5.0, seed=1)
        npt.length(delta=0.1, weight=1)
        npt.shear(delta=0.1, weight=1, reduce=0.6)
        compressor = hpmc.util.compress(mc=mc,
                                        npt_updater=npt,
                                        **self.args)
        etas, snaps = compressor.run(1)
        etas = np.array(etas)
        self.assertGreater(etas.max(), 0.7)
        del snaps
        del compressor
        del npt
        del mc
        del system
        context.initialize()
    # Two spheres require all box and particle move types to compress.
    # Larger unit cell requires more sensitivity for convergence.
    # Initialize with overlaps to test overlap resolution.
    def test_2sphere(self):
        self.args['ptypes'] = ['A']
        self.args['pnums'] = [2]
        self.args['pvolumes'] = [4./3. * np.pi * 0.5**3]
        self.args['num_comp_steps'] = 4e5
        self.args['pf_tol'] = 1e-4
        self.args['pmin'] = 5
        self.args['relax'] = 1e3
        system = create_empty(N=2, box=data.boxdim(L=3), particle_types=['A'])
        system.particles[1].position = (0.9,0,0)
        mc = hpmc.integrate.sphere(seed=1, nselect=1)
        mc.set_params(d=0.1)
        mc.shape_param.set('A', diameter=1.0)
        npt = hpmc.update.boxmc(mc, betaP=5.0, seed=1)
        npt.length(delta=0.1, weight=1)
        npt.shear(delta=0.1, weight=1, reduce=0.6)
        compressor = hpmc.util.compress(mc=mc,
                                        npt_updater=npt,
                                        **self.args)
        etas, snaps = compressor.run(2)
        etas = np.array(etas)
        self.assertGreater(etas.max(), 0.7)
        del snaps
        del compressor
        del npt
        del mc
        del system
        context.initialize()
    # Test two orientable shapes
    def test_2cube(self):
        self.args['ptypes'] = ['A']
        self.args['pnums'] = [2]
        self.args['pvolumes'] = [8.]
        self.args['num_comp_steps'] = 2e5
        self.args['pf_tol'] = 1e-5
        self.args['pmin'] = 5
        self.args['relax'] = 1e3
        system = create_empty(N=2, box=data.boxdim(L=3), particle_types=['A'])
        system.particles[1].position = (2.0,0,0)
        mc = hpmc.integrate.convex_polyhedron(seed=1)
        mc.set_params(d=0.1, a=0.1)
        mc.shape_param.set('A', vertices=[ (1,1,1), (1,-1,1), (-1,-1,1), (-1,1,1),
           (1,1,-1), (1,-1,-1), (-1,-1,-1), (-1,1,-1) ])
        npt = hpmc.update.boxmc(mc, betaP=5.0, seed=1)
        npt.length(delta=0.1, weight=1)
        npt.shear(delta=0.1, weight=1, reduce=0.6)
        compressor = hpmc.util.compress(mc=mc,
                                        npt_updater=npt,
                                        **self.args)
        etas, snaps = compressor.run(2)
        etas = np.array(etas)
        self.assertGreater(etas.max(), 0.9)
        del snaps
        del compressor
        del npt
        del mc
        del system
        context.initialize()
    def tearDown(self):
        os.remove(self.fname)

# TO DO
class snapshot (unittest.TestCase):
    # show that the snapshot class can properly extract data from a simulation
    def test_2_poly_types(self):
        pass

class tune (unittest.TestCase):
    def setUp(self):
        self.system = create_empty(N=2, box=data.boxdim(L=4.5), particle_types=['A'])
        self.system.particles[1].position = (2.0,0,0)
        self.mc = hpmc.integrate.convex_polyhedron(seed=1)
        self.mc.set_params(d=0.1, a=0.1)
        self.mc.shape_param.set('A', vertices=[ (1,1,1), (1,-1,1), (-1,-1,1), (-1,1,1),
           (1,1,-1), (1,-1,-1), (-1,-1,-1), (-1,1,-1) ])

    # show that the tuner will adjust d to achieve a reasonable acceptance ratio
    def test_d(self):
        # Set up
        self.mc.set_params(d=1, a=1, move_ratio=0.5)
        target = 0.8
        old_acceptance = self.mc.get_translate_acceptance()
        old_d = self.mc.get_d()

        # Create and run the tuner
        tuner = hpmc.util.tune(self.mc, tunables=['d'], max_val=[2], target=target, gamma=0.0)
        for i in range(5):
                    run(2e2)
                    tuner.update()

        # Check that the new acceptance has improved
        new_acceptance = self.mc.get_translate_acceptance()
        self.assertLess(abs(new_acceptance - target), abs(old_acceptance - target)) 
        self.assertNotEqual(old_d, self.mc.get_d())
        del tuner

    # show that the tuner can reduce a to achieve a reasonable acceptance ratio
    def test_a(self):
        # Set up
        self.mc.set_params(d=0.4, a=1, move_ratio=0.5)
        target = 0.8
        old_acceptance = self.mc.get_rotate_acceptance()
        old_a = self.mc.get_a()

        # Create and run the tuner
        tuner = hpmc.util.tune(self.mc, tunables=['a'], max_val=[2], target=target, gamma=0.0)
        for i in range(5):
                    run(2e2)
                    tuner.update()

        # Check that the new acceptance has improved
        new_acceptance = self.mc.get_rotate_acceptance()
        self.assertLess(abs(new_acceptance - target), abs(old_acceptance - target)) 
        self.assertNotEqual(old_a, self.mc.get_a())
        del tuner

    # show that the tuner can tune both d and a simultaneously
    def test_multiple_tunables(self):
        # Set up
        self.mc.set_params(d=1, a=1, move_ratio=0.5)
        target = 0.8
        old_translate_acceptance = self.mc.get_translate_acceptance()
        old_rotate_acceptance = self.mc.get_rotate_acceptance()
        old_a = self.mc.get_a()
        old_d = self.mc.get_d()

        # Create and run the tuner
        tuner = hpmc.util.tune(self.mc, tunables=['d', 'a'], max_val=[1, 1], target=target, gamma=0.0)
        for i in range(5):
            run(2e2)
            tuner.update()

        # Check that the new acceptance has improved
        new_translate_acceptance = self.mc.get_translate_acceptance()
        new_rotate_acceptance = self.mc.get_rotate_acceptance()
        self.assertLess(abs(new_translate_acceptance - target), abs(old_translate_acceptance - target)) 
        self.assertLess(abs(new_rotate_acceptance - target), abs(old_rotate_acceptance - target)) 
        self.assertNotEqual(old_a, self.mc.get_a())
        self.assertNotEqual(old_d, self.mc.get_d())
        del tuner

    # show that the npt tuner can reasonably handle volume changes
    def test_npt_noshear(self):
        target = 0.5
        self.mc.set_params(d=0.1, a=0.01, move_ratio=0.5)
        updater = hpmc.update.boxmc(self.mc, betaP=10.0, seed=1)
        updater.length(delta=(0.01,0.01,0.01), weight=1)
        tuner = hpmc.util.tune_npt(updater, tunables=['dLx', 'dLy', 'dLz'], target=target, gamma=0.0)
        for i in range(5):
            run(1e2)
            tuner.update()
            print("npt_noshear: ", *updater.length()['delta'])
        acceptance = updater.get_volume_acceptance()
        self.assertGreater(acceptance, 0.)
        self.assertLess(acceptance, 1.0)
        del tuner
        del updater

    # show that the npt tuner can properly handle shear
    def test_npt_shear(self):
        target = 0.5
        self.mc.set_params(d=0.02, a=0.01, move_ratio=0.5)
        updater = hpmc.update.boxmc(self.mc, seed=1, betaP=10)
        updater.length(delta=(0.1, 0.1, 0.1), weight=1)
        updater.shear(delta=(0.1, 0.1, 0.1), weight=1)
        tuner = hpmc.util.tune_npt(updater, tunables=['dxy', 'dyz', 'dxz'], target=target, gamma=0.5)
        for i in range(5):
            run(1e2)
            tuner.update()
        acceptance = updater.get_shear_acceptance()
        self.assertGreater(acceptance, 0.)
        self.assertLess(acceptance, 1.0)
        del tuner
        del updater

    # check the tuner for isotropic mode
    def test_npt_isotropic(self):
        target = 0.5
        self.mc.set_params(d=0.1, a=0.01, move_ratio=0.5)
        updater = hpmc.update.boxmc(self.mc, seed=1, betaP=10)
        updater.volume(delta=0.1, weight=1)
        tuner = hpmc.util.tune_npt(updater, tunables=['dV'], target=target, gamma=0.0)
        for i in range(5):
            run(1e2)
            tuner.update()
            print("npt_isotropic: ", updater.volume()['delta'])
        acceptance = updater.get_volume_acceptance()
        self.assertGreater(acceptance, 0.)
        self.assertLess(acceptance, 1.0)
        del tuner
        del updater

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize()

# Test tuning of systems where we specify the type
class tune_by_type(unittest.TestCase):
    def setUp(self):
        self.system = create_empty(N=2, box=data.boxdim(L=4.5), particle_types=['A', 'B'])
        self.system.particles[0].position = (1.0,0,0)
        self.system.particles[1].position = (-1.0,0,0)
        self.system.particles.types = ['A', 'B']
        self.mc = hpmc.integrate.convex_polyhedron(seed=1)
        self.mc.set_params(d=0.5, a=0.5)
        self.mc.shape_param.set('A', vertices=[ (1,1,1), (1,-1,1), (-1,-1,1), (-1,1,1),
           (1,1,-1), (1,-1,-1), (-1,-1,-1), (-1,1,-1) ])
        self.mc.shape_param.set('B', vertices=[ (1,1,1), (1,-1,1), (-1,-1,1), (-1,1,1),
           (1,1,-1), (1,-1,-1), (-1,-1,-1), (-1,1,-1) ])

    # show that the tuner will adjust d to achieve a reasonable acceptance ratio
    def test_d(self):
        # Set up
        self.mc.set_params(d=0.5, a=0.5, move_ratio=0.5)
        target = 0.8
        old_translate_acceptance = self.mc.get_translate_acceptance()
        old_d = self.mc.get_d("A")
        old_d_fixed = self.mc.get_d("B")

        # Create and run the tuner. Make sure to ignore statistics for the unused type
        self.mc.shape_param["B"].ignore_statistics = True
        tuner = hpmc.util.tune(self.mc, type='A', tunables=['d'], max_val=[1], target=target, gamma=0.0)
        for i in range(5):
            run(2e2)
            tuner.update()

        # Check that the new acceptance has improved
        new_translate_acceptance = self.mc.get_translate_acceptance()
        self.assertLess(abs(new_translate_acceptance - target), abs(old_translate_acceptance - target)) 
        self.assertNotEqual(old_d, self.mc.get_d("A"))
        self.assertEqual(old_d_fixed, self.mc.get_d("B"))
        del tuner

    # Test per-type tuning
    def test_a(self):
        # Set up
        self.mc.set_params(d=0.5, a=0.5, move_ratio=0.5)
        target = 0.8
        old_rotate_acceptance = self.mc.get_rotate_acceptance()
        old_a = self.mc.get_a("A")
        old_a_fixed = self.mc.get_a("B")

        # Create and run the tuner. Make sure to ignore statistics for the unused type
        self.mc.shape_param["B"].ignore_statistics = True
        tuner = hpmc.util.tune(self.mc, type='A', tunables=['a'], max_val=[1], target=target, gamma=0.0)
        for i in range(5):
            run(2e2)
            tuner.update()

        # Check that the new acceptance has improved
        new_rotate_acceptance = self.mc.get_rotate_acceptance()
        self.assertLess(abs(new_rotate_acceptance - target), abs(old_rotate_acceptance - target)) 
        self.assertNotEqual(old_a, self.mc.get_a("A"))
        self.assertEqual(old_a_fixed, self.mc.get_a("B"))
        del tuner

    # Test per-type tuning
    def test_multiple_tunables(self):
        # Set up
        self.mc.set_params(d=0.5, a=0.5, move_ratio=0.5)
        target = 0.8
        old_translate_acceptance = self.mc.get_translate_acceptance()
        old_rotate_acceptance = self.mc.get_rotate_acceptance()
        old_a = self.mc.get_a("A")
        old_d = self.mc.get_d("A")
        old_a_fixed = self.mc.get_a("B")
        old_d_fixed = self.mc.get_d("B")

        # Create and run the tuner. Make sure to ignore statistics for the unused type
        self.mc.shape_param["B"].ignore_statistics = True
        tuner = hpmc.util.tune(self.mc, type='A', tunables=['d', 'a'], max_val=[1, 1], target=target, gamma=0.0)
        for i in range(5):
            run(2e2)
            tuner.update()

        # Check that the new acceptance has improved
        new_translate_acceptance = self.mc.get_translate_acceptance()
        new_rotate_acceptance = self.mc.get_rotate_acceptance()
        self.assertLess(abs(new_translate_acceptance - target), abs(old_translate_acceptance - target)) 
        self.assertLess(abs(new_rotate_acceptance - target), abs(old_rotate_acceptance - target)) 
        self.assertNotEqual(old_a, self.mc.get_a("A"))
        self.assertNotEqual(old_d, self.mc.get_d("A"))
        self.assertEqual(old_a_fixed, self.mc.get_a("B"))
        self.assertEqual(old_d_fixed, self.mc.get_d("B"))
        del tuner

    def tearDown(self):
        del self.mc
        del self.system
        context.initialize()

# Test handling of extreme values
class tune_extreme (unittest.TestCase):
    # show that the tuner can reduce d to a small value without making it too small
    def test_d_small(self):
        return # This test will fail until hoomd.util.tune has better minimum value checking.
        minimum = 1e-6 # minimum should come from the tuner ultimately
        target = 0.99

        N = 27
        snapshot = data.make_snapshot(N=N, box=data.boxdim(6.0001,6.0001,6.0001), particle_types=['A'])
        positions = np.array([ (i,j,k) for i in range(3) for j in range(3) for k in range(3) ], dtype=np.float32)
        positions -= (1,1,1)
        positions *= 2.00001
        snapshot.particles.position[:] = positions
        self.system = init.read_snapshot(snapshot)
        self.mc = hpmc.integrate.convex_polyhedron(seed=1)
        self.mc.set_params(d=0.1, a=0.1)
        self.mc.shape_param.set('A', vertices=[ (1,1,1), (1,-1,1), (-1,-1,1), (-1,1,1),
           (1,1,-1), (1,-1,-1), (-1,-1,-1), (-1,1,-1) ])

        #tuner = hpmc.util.tune(mc, tunables=['d', 'a'], target=0.2, gamma=0.0)
        tuner = hpmc.util.tune(self.mc, tunables=['d'], target=target, gamma=0.0)
        for i in range(5):
                    run(2e2)
                    tuner.update()
        d = self.mc.get_d()
        self.assertGreater(d, minimum)
        del tuner
        del self.mc
        del self.system
        context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
