from hoomd import *
from hoomd import md
import math
import unittest

context.initialize()

# test the md.constrain.rigid() functionality
class test_constrain_rigid(unittest.TestCase):
    def setUp(self):
        self.system = init.read_gsd(os.path.join(os.path.dirname(__file__),'test_data_diblock_copolymer_system.gsd'));

        # generate a system of N=8 AB diblocks
        self.nl = md.nlist.cell()

        for p in self.system.particles:
            p.moment_inertia = (.5,.5,1)
            #p.moment_inertia = (0,0,0)

    def test_unequal_argument_lengths(self):
        # create constituent particle types
        self.system.particles.types.add('A_const')

        rigid = md.constrain.rigid()
        # try passing inconsistent numbers of arguments
        self.assertRaises(RuntimeError,rigid.set_param,'A', types=['A_const']*3, positions=[(1,2,3),(4,5,6)])
        self.assertRaises(RuntimeError,rigid.set_param,'A', types=['A_const']*2, positions=[(1,2,3),(4,5,6)], charges=[0])
        self.assertRaises(RuntimeError,rigid.set_param,'A', types=['A_const']*2, positions=[(1,2,3),(4,5,6)], diameters=[0])

    def test_energy_conservation(self):
        # create rigid spherocylinders out of two particles (not including the central particle)
        len_cyl = .5

        # create constituent particle types
        self.system.particles.types.add('A_const')
        self.system.particles.types.add('B_const')

        md.integrate.mode_standard(dt=0.001)

        lj = md.pair.lj(r_cut=False, nlist = self.nl)

        # central particles
        lj.pair_coeff.set(['A','B'], self.system.particles.types, epsilon=1.0, sigma=1.0, r_cut=2.5)

        # constituent particle coefficients
        lj.pair_coeff.set('A_const','A_const', epsilon=1.0, sigma=1.0, r_cut=2.5)
        lj.pair_coeff.set('B_const','B_const', epsilon=1.0, sigma=1.0, r_cut=2.5)
        lj.pair_coeff.set('A_const','B_const', epsilon=1.0, sigma=1.0, r_cut=2.5)
        lj.set_params(mode="xplor")

        rigid = md.constrain.rigid()
        rigid.set_param('A', types=['A_const','A_const'], positions=[(0,0,-len_cyl/2),(0,0,len_cyl/2)])
        rigid.set_param('B', types=['B_const','B_const'], positions=[(0,0,-len_cyl/2),(0,0,len_cyl/2)])
        rigid.create_bodies()

        center = group.rigid_center()

        # thermalize
        langevin = md.integrate.langevin(group=center,kT=1.0,seed=123)
        langevin.set_gamma('A',2.0)
        langevin.set_gamma('B',2.0)
        run(50)
        langevin.disable()

        nve = md.integrate.nve(group=center)

        log = analyze.log(filename=None,quantities=['potential_energy','kinetic_energy'],period=10)

        # warm up
        run(50)

        # measure
        E0 = log.query('potential_energy') + log.query('kinetic_energy')
        run(50)
        E1 = log.query('potential_energy') + log.query('kinetic_energy')

        # two sig figs
        self.assertAlmostEqual(E0/round(E0),E1/round(E0),2)
        del rigid
        del lj
        del log
        del nve

    def test_npt(self):
        # create rigid spherocylinders out of two particles (not including the central particle)
        len_cyl = .5

        # create constituent particle types
        self.system.particles.types.add('A_const')
        self.system.particles.types.add('B_const')

        md.integrate.mode_standard(dt=0.001)

        lj = md.pair.lj(r_cut=False, nlist = self.nl)

        # central particles
        lj.pair_coeff.set(['A','B'], self.system.particles.types, epsilon=1.0, sigma=1.0, r_cut=2.5)

        # constituent particle coefficients
        lj.pair_coeff.set('A_const','A_const', epsilon=1.0, sigma=1.0, r_cut=2.5)
        lj.pair_coeff.set('B_const','B_const', epsilon=1.0, sigma=1.0, r_cut=2.5)
        lj.pair_coeff.set('A_const','B_const', epsilon=1.0, sigma=1.0, r_cut=2.5)
        lj.set_params(mode="xplor")

        rigid = md.constrain.rigid()
        rigid.set_param('A', types=['A_const','A_const'], positions=[(0,0,-len_cyl/2),(0,0,len_cyl/2)])
        rigid.set_param('B', types=['B_const','B_const'], positions=[(0,0,-len_cyl/2),(0,0,len_cyl/2)])
        rigid.create_bodies()

        center = group.rigid_center()

        # thermalize
        langevin = md.integrate.langevin(group=center,kT=1.0,seed=123)
        langevin.set_gamma('A',2.0)
        langevin.set_gamma('B',2.0)
        run(50)
        langevin.disable()

        P = 2.5
        npt = md.integrate.npt(group=center,P=P,tauP=0.5,kT=1.0,tau=1.0)

        log = analyze.log(filename=None,quantities=['potential_energy','kinetic_energy','npt_thermostat_energy','npt_barostat_energy','volume'],period=10)

        # warm up
        run(50)

        # measure
        E0 = log.query('potential_energy') + log.query('kinetic_energy') + log.query('npt_thermostat_energy') + log.query('npt_barostat_energy') + P*log.query('volume')
        run(50)
        E1 = log.query('potential_energy') + log.query('kinetic_energy') + log.query('npt_thermostat_energy') + log.query('npt_barostat_energy') + P*log.query('volume')

        # two sig figs
        self.assertAlmostEqual(E0/round(E0),E1/round(E0),2)
        del rigid
        del lj
        del log
        del npt

    def test_reinit(self):
        # create rigid spherocylinders out of two particles (not including the central particle)
        len_cyl = .5

        # create constituent particle types
        self.system.particles.types.add('A_const')
        self.system.particles.types.add('B_const')

        md.integrate.mode_standard(dt=0.001)

        lj = md.pair.lj(r_cut=False, nlist = self.nl)

        # central particles
        lj.pair_coeff.set(['A','B'], self.system.particles.types, epsilon=1.0, sigma=1.0, r_cut=2.5)

        # constituent particle coefficients
        lj.pair_coeff.set('A_const','A_const', epsilon=1.0, sigma=1.0, r_cut=2.5)
        lj.pair_coeff.set('B_const','B_const', epsilon=1.0, sigma=1.0, r_cut=2.5)
        lj.pair_coeff.set('A_const','B_const', epsilon=1.0, sigma=1.0, r_cut=2.5)
        lj.set_params(mode="xplor")

        rigid = md.constrain.rigid()
        rigid.set_param('A', types=['A_const','A_const'], positions=[(0,0,-len_cyl/2),(0,0,len_cyl/2)],diameters=[1,2],charges=[-1,1])
        rigid.set_param('B', types=['B_const','B_const'], positions=[(0,0,-len_cyl/2),(0,0,len_cyl/2)],diameters=[3,4],charges=[-2,2])
        rigid.create_bodies()

        center = group.rigid_center()

        nve = md.integrate.nve(group=center)

        # create rigid bodies
        run(1)

        self.assertEqual(self.system.particles[0].type,'A')
        self.assertEqual(self.system.particles[4000].type,'A_const')
        self.assertEqual(self.system.particles[4001].type,'A_const')
        self.assertEqual(self.system.particles[4000].diameter,1.0)
        self.assertEqual(self.system.particles[4000].charge,-1.0)
        self.assertEqual(self.system.particles[4001].diameter,2)
        self.assertEqual(self.system.particles[4001].charge,1.0)

        self.assertEqual(self.system.particles[2000].type,'B')
        self.assertEqual(self.system.particles[8000].type,'B_const')
        self.assertEqual(self.system.particles[8001].type,'B_const')
        self.assertEqual(self.system.particles[8000].diameter,3.0)
        self.assertEqual(self.system.particles[8000].charge,-2.0)
        self.assertEqual(self.system.particles[8001].diameter,4)
        self.assertEqual(self.system.particles[8001].charge,2.0)


        snap = self.system.take_snapshot()
        self.system.restore_snapshot(snap)

        # validate rigid bodies
        run(1)

        del rigid
        del lj
        del nve

    def test_box_resize(self):
        # create rigid spherocylinders out of two particles (not including the central particle)
        len_cyl = .5

        # create constituent particle types
        self.system.particles.types.add('A_const')
        self.system.particles.types.add('B_const')

        rigid = md.constrain.rigid()
        rigid.set_param('A', types=['A_const','A_const'], positions=[(0,0,-len_cyl/2),(0,0,len_cyl/2)])
        rigid.set_param('B', types=['B_const','B_const'], positions=[(0,0,-len_cyl/2),(0,0,len_cyl/2)])

        update.box_resize(L = variant.linear_interp([(0, 50), (100, 100)]))
        run(100)

    def test_metadata(self):
        # create rigid spherocylinders out of two particles (not including the central particle)
        len_cyl = .5

        # create constituent particle types
        self.system.particles.types.add('A_const')
        self.system.particles.types.add('B_const')

        rigid = md.constrain.rigid()
        rigid.set_param('A', types=['A_const','A_const'], positions=[(0,0,-len_cyl/2),(0,0,len_cyl/2)])
        rigid.set_param('B', types=['B_const','B_const'], positions=[(0,0,-len_cyl/2),(0,0,len_cyl/2)])

        meta.dump_metadata()

    def tearDown(self):
        del self.system, self.nl
        context.initialize();

def test_self_interaction(system,nlist):
    # create long cylinders slightly shorter than the box dimension
    # adding the LJ cut-off should make them longer than the box
    len_cyl = system.box.Lx-1

    # create constituent particle types
    system.particles.types.add('const')

    md.integrate.mode_standard(dt=0)
    nve = md.integrate.nve(group=group.rigid_center())

    lj = md.pair.lj(r_cut=False, nlist = nlist)

    # central particles
    lj.pair_coeff.set(['A'], system.particles.types, epsilon=0, sigma=0, r_cut=False)

    # constituent particle coefficients
    lj.pair_coeff.set('const','const', epsilon=1.0, sigma=1.0, r_cut=2.5)

    rigid = md.constrain.rigid()
    rigid.set_param('A', types=['const','const'], positions=[(0,0,-len_cyl/2),(0,0,len_cyl/2)])
    rigid.create_bodies()

    # we should get an error from NeighborList
    run(1)

# test that self-interactions are not possible
class test_constrain_rigid_self_interactions(unittest.TestCase):
    def setUp(self):
        snap = data.make_snapshot(N=1, particle_types=['A'], box=data.boxdim(L=10))
        if comm.get_rank() == 0:
            snap.particles.position[0] = (0,0,0)
            snap.particles.orientation[0] = (1,0,0,0)
        self.system = init.read_snapshot(snap)

    def test_self_interaction_nlist_cell(self):
        nlist = md.nlist.cell()
        self.assertRaises(RuntimeError,test_self_interaction,self.system,nlist)

    def test_self_interaction_nlist_stencil(self):
        nlist = md.nlist.stencil()
        self.assertRaises(RuntimeError,test_self_interaction,self.system,nlist)

    def test_self_interaction_nlist_tree(self):
        nlist = md.nlist.tree()
        self.assertRaises(RuntimeError,test_self_interaction,self.system,nlist)

    def tearDown(self):
        del self.system
        context.initialize();

# test that mixtures of rigid and nonrigid particles are possible
class test_constrain_rigid_nonrigid(unittest.TestCase):
    def setUp(self):
        self.s = init.create_lattice(lattice.sc(a=2,type_name='A'),n=[10,10,10]);

        # 50% of the particles are B
        self.s.particles.types.add('B')
        for i,p in enumerate(self.s.particles):
            if i % 2:
                p.type = 'B'

    def test_rigid_nonrigid(self):
        self.s.particles.types.add('A_const')
        rigid = md.constrain.rigid()
        rigid.set_param('A',types=['A_const']*2,positions=[(-.5,0,0),(.5,0,0)])

        rigid.create_bodies()

        nlist = md.nlist.cell()
        lj = md.pair.lj(r_cut=False, nlist = nlist)

        # central particles
        lj.pair_coeff.set(['A'], self.s.particles.types, epsilon=0, sigma=0, r_cut=False)
        lj.pair_coeff.set('B', ['B','A_const'], epsilon=1, sigma=1, r_cut=2.5)
        lj.pair_coeff.set('A_const','A_const', epsilon=1.0, sigma=1.0, r_cut=2.5)

        center = group.rigid_center()
        nonrigid = group.nonrigid()
        g = group.union(a=center,b=nonrigid, name='intgroup')

        md.integrate.mode_standard(dt=0.005)
        langevin = md.integrate.langevin(group=g,kT=1.0,seed=1234)

        run(1000)

    def tearDown(self):
        del self.s
        context.initialize();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
