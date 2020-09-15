# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Test write_debug_data for MD simulations."""

import hoomd


def test_write_debug_data(simulation_factory, lattice_snapshot_factory,
                          tmp_path):
    """Test write_debug_data for MD simulations."""
    sim = simulation_factory(lattice_snapshot_factory())
    md = hoomd.md.Integrator(dt=0.005)

    cell = hoomd.md.nlist.Cell()
    lj = hoomd.md.pair.LJ(nlist=cell)
    lj.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.2)
    lj.r_cut[('A', 'A')] = 2.5

    gauss = hoomd.md.pair.Gauss(nlist=cell)
    gauss.params[('A', 'A')] = dict(epsilon=1.5, sigma=0.9)
    gauss.r_cut[('A', 'A')] = 3.5

    md.forces = [lj, gauss]
    md.methods = [
        hoomd.md.methods.Langevin(kT=1.5, seed=6, filter=hoomd.filter.All())
    ]

    sim.operations.integrator = md

    snap = sim.state.snapshot
    snap.particles.types = ['A']
    snap.replicate(2, 2, 2)
    sim.state.snapshot = snap

    sim.write_debug_data(tmp_path / 'test_unscheduled.json')

    sim.run(10)

    sim.write_debug_data(tmp_path / 'test_scheduled.json')
