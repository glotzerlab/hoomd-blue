from __future__ import division
import hoomd
import hoomd.md
import unittest

hoomd.context.initialize();

# test the md.constrain.rigid() functionality
class test_log_energy_upon_run_command(unittest.TestCase):
    def test_log(self):
        uc = hoomd.lattice.unitcell(N = 1,
                                    a1 = [10.8, 0,   0],
                                    a2 = [0,    1.2, 0],
                                    a3 = [0,    0,   1.2],
                                    dimensions = 3,
                                    position = [[0,0,0]],
                                    type_name = ['R'],
                                    mass = [1.0],
                                    moment_inertia = [[0,
                                                       1/12*1.0*8**2,
                                                       1/12*1.0*8**2]],
                                    orientation = [[1, 0, 0, 0]]);
        system = hoomd.init.create_lattice(unitcell=uc, n=[8,18,18]);
        system.particles.types.add('A')
        rigid = hoomd.md.constrain.rigid()
        rigid.set_param('R',
                        types=['A']*8,
                        positions=[(-4,0,0),(-3,0,0),(-2,0,0),(-1,0,0),
                                   (1,0,0),(2,0,0),(3,0,0),(4,0,0)]);

        rigid.create_bodies()
        nl = hoomd.md.nlist.cell()
        lj = hoomd.md.pair.lj(r_cut=3.0, nlist=nl)
        lj.set_params(mode='shift')
        lj.pair_coeff.set(['R', 'A'], ['R', 'A'], epsilon=1.0, sigma=1.0)
        hoomd.md.integrate.mode_standard(dt=0.001);
        rigid_gr = hoomd.group.rigid_center();
        integrator=hoomd.md.integrate.langevin(group=rigid_gr, kT=1.0, seed=42);
        log = hoomd.analyze.log(filename=None,
                          quantities=['potential_energy',
                                      'translational_kinetic_energy',
                                      'rotational_kinetic_energy', 'pressure'],
                          period=1,
                          overwrite=True);
        hoomd.run(100);

        self.last_l = None
        def cb(timestep):
            l = log.query('potential_energy')
            if self.last_l is not None:
                rel_dl = abs(l)/abs(self.last_l)

            else:
                rel_dl = 1.0
            # the log value shouldn't change abruptly
            self.assertTrue(rel_dl > 0.5)
            self.assertTrue(rel_dl < 1.5)
            self.last_l = l
        for i in range(10):
            hoomd.run(10,callback=cb, callback_period=1)

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])

