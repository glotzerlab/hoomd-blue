# -*- coding: iso-8859-1 -*-

from hoomd import *
from hoomd import md

context.initialize()
import unittest
import os

# tests for md.update.popbd
class update_popbd(unittest.TestCase):
    def setUp(self):
        snap = data.make_snapshot(
            N=2,
            box=data.boxdim(Lx=80, Ly=80, Lz=80),
            bond_types=["polymer"],
            particle_types=["colloid"],
        )

        if comm.get_rank() == 0:
            snap.particles.diameter[:] = [1.0 * 2] * 2
            snap.particles.position[0] = [-1, 0, 0]
            snap.particles.position[1] = [1, 0, 0]

        self.s = init.read_snapshot(snap)
        self.nl = md.nlist.tree()
        self.integrator = md.integrate.mode_standard(dt=0.001)
        md.integrate.brownian(group=group.all(), kT=1, seed=0)

    # tests basic creation of the updater
    def test_create(self):
        # nl = md.nlist.tree();
        dybond = md.update.popbd(
            group=group.all(),
            nlist=self.nl,
            seed=1,
            period=1,
            integrator=self.integrator,
            table_width=10,
        )
        run(10)

    # tests formation of a bond within a cutoff radius
    def test_set_params(self):
        dybond = md.update.popbd(
            group=group.all(),
            nlist=self.nl,
            seed=1,
            period=1,
            integrator=self.integrator,
            table_width=10,
        )
        dybond.set_params(
            r_cut=2.0,
            bond_type="harmonic",
            n_polymer=10,
        )

    def test_set_from_file(self):
        dybond = md.update.popbd(
            group=group.all(),
            nlist=self.nl,
            seed=1,
            period=1,
            integrator=self.integrator,
            table_width=3,
        )
        dybond.set_from_file(os.path.join(os.path.dirname(__file__), 'popbd.dat'))

    def tearDown(self):
        del self.s
        context.initialize()


if __name__ == "__main__":
    unittest.main(argv=["test.py", "-v"])

