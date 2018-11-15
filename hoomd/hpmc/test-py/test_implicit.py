from __future__ import print_function
from __future__ import division
from hoomd import *
from hoomd import hpmc
import math
import unittest

context.initialize()

#
# These tests can be run with the '--user=long' option, in which case
# the results will be tested for thermodynamical correctness.
#
# Per default, it will only be checked that the simulation runs, i.e. does not crash
#
class implicit_test_cube(unittest.TestCase):
    def setUp(self):
        # setup the MC integration
        phi_c_ini = 0.01
        phi_c = 0.1
        self.V_cube = 1.0
        n = [10] * 3
        N = math.pow(10,3)

        self.long = False
        self.num_samples = 0
        self.steps = 10
        if len(option.get_user()) > 0 and option.get_user()[0]=="long":
            self.long = True
            self.num_samples = 10
            self.steps = 1000

        L_ini= math.pow(N*self.V_cube/phi_c_ini,1.0/3.0)
        L_target= math.pow(N*self.V_cube/phi_c,1.0/3.0)

        self.system = init.create_lattice(lattice.sc(a=L_ini/float(n[0])),n=n)

        self.mc = hpmc.integrate.convex_polyhedron(seed=123,implicit=True,depletant_mode='overlap_regions')
        self.mc.set_params(d=0.1,a=0.15)

        etap=1.0
        self.nR = etap/self.V_cube

        self.system.particles.types.add('B')
        self.mc.set_params(nR=0,depletant_type='B')

        cube_verts=[(-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5)]
        self.mc.shape_param.set('A', vertices=cube_verts)
        self.mc.shape_param.set('B', vertices=cube_verts)

        import tempfile
        tmp = tempfile.mkstemp(suffix='.hpmc-test-implicit2');
        self.tmp_file = tmp[1];

        nsample = 10000
        self.free_volume = hpmc.compute.free_volume(mc=self.mc, seed=987, nsample=nsample, test_type='B')
        self.log=analyze.log(filename=self.tmp_file, quantities=['volume','hpmc_free_volume'], overwrite=True,period=100)

        if self.long:
            mc_tune = hpmc.util.tune(self.mc, tunables=['d','a'],max_val=[4,0.5],gamma=1,target=0.4)

            # shrink the box to the target size (if needed)
            scale = 0.99;
            L = L_ini;
            while L_target < L:
                # shrink the box
                L = max(L*scale, L_target);

                update.box_resize(Lx=L, Ly=L, Lz=L, period=None);
                overlaps = self.mc.count_overlaps();
                context.msg.notice(1,"phi =%f: overlaps = %d " % (((N*self.V_cube) / (L*L*L)), overlaps));

                # run until all overlaps are removed
                while overlaps > 0:
                    mc_tune.update()
                    run(100, quiet=True);
                    overlaps = self.mc.count_overlaps();
                    context.msg.notice(1,"%d\n" % overlaps)

                context.msg.notice(1,"\n");

            # set the target L (this expands to the final L if it is larger than the start)
            update.box_resize(Lx=L_target, Ly=L_target, Lz=L_target, period=None);

    def test_implicit(self):
        # use depletants
        self.mc.set_params(nR=self.nR)

        # warm up
        run(self.steps)

        avg_eta_p = 0
        for i in range(self.num_samples):
            run(self.steps)
            n_overlap = self.mc.count_overlaps()
            self.assertEqual(n_overlap,0)
            vol = self.log.query('volume')
            free_vol = self.log.query('hpmc_free_volume')
            eta_p = self.V_cube*free_vol/vol*self.nR
            avg_eta_p += eta_p/self.num_samples

        # check equation of state with very rough tolerance
        context.msg.notice(1,'eta_p = {0}\n'.format(avg_eta_p))
        if self.long:
            self.assertAlmostEqual(avg_eta_p,0.4,delta=0.1)

    def tearDown(self):
        if comm.get_rank() == 0:
            os.remove(self.tmp_file);

        del self.free_volume
        del self.log
        del self.mc
        del self.system
        context.initialize();

class implicit_test_sphere_new (unittest.TestCase):
    def setUp(self):
        # setup the MC integration
        self.system = init.create_lattice(lattice.sc(a=1.3782337338022654),n=[10,10,10]) #target a packing fraction of 0.2

        self.num_samples = 0
        self.steps = 10

        self.mc = hpmc.integrate.sphere(seed=123,implicit=True, depletant_mode='overlap_regions')
        self.mc.set_params(d=0.1)

        q=1.0
        etap=1.0
        self.nR = etap/(math.pi/6.0*math.pow(q,3.0))

        self.system.particles.types.add('B')
        self.mc.set_params(nR=self.nR,depletant_type='B')

        self.mc.shape_param.set('A', diameter=1.0)
        self.mc.shape_param.set('B', diameter=q)

        import tempfile
        tmp = tempfile.mkstemp(suffix='.hpmc-test-implicit');
        self.tmp_file = tmp[1];

        nsample = 10000
        self.free_volume = hpmc.compute.free_volume(mc=self.mc, seed=987, nsample=nsample, test_type='B')
        self.log=analyze.log(filename=self.tmp_file, quantities=['volume','hpmc_free_volume'], overwrite=True,period=100)

    def test_implicit(self):
        # warm up
        run(self.steps)

        avg_eta_p = 0

        for i in range(self.num_samples):
            run(self.steps)
            n_overlap = self.mc.count_overlaps()
            self.assertEqual(n_overlap,0)
            vol = self.log.query('volume')
            free_vol = self.log.query('hpmc_free_volume')
            eta_p = math.pi/6.0*free_vol/vol*self.nR
            avg_eta_p += eta_p/self.num_samples

        context.msg.notice(1,'eta_p = {0}\n'.format(avg_eta_p))

    def tearDown(self):
        if comm.get_rank() == 0:
            os.remove(self.tmp_file);

        del self.free_volume
        del self.log
        del self.mc
        del self.system
        context.initialize();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
