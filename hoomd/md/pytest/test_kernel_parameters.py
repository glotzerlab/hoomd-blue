# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import pytest


@pytest.mark.gpu
@pytest.mark.validate
def test_combined_kernel_parameters(simulation_factory,
                                    lattice_snapshot_factory):

    snap = lattice_snapshot_factory(particle_types=['A'], n=7, a=1.7, r=0.01)
    sim = simulation_factory(snap)

    nlist = hoomd.md.nlist.Cell(buffer=0.4)
    lj = hoomd.md.pair.LJ(nlist=nlist, default_r_cut=2.5)
    lj.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0)
    langevin = hoomd.md.methods.Langevin(kT=1.5, filter=hoomd.filter.All())
    sim.operations.integrator = hoomd.md.Integrator(dt=0.005,
                                                    methods=[langevin],
                                                    forces=[lj])

    sim.run(0)
    while not sim.operations.is_tuning_complete:
        sim.run(1000)

        # Prevent infinite loops:
        if sim.timestep > 100_000:
            raise RuntimeError("Tuning is not completing as expected.")

    assert nlist.is_tuning_complete
    assert lj.is_tuning_complete
    assert langevin.is_tuning_complete
    assert sim.operations.integrator.is_tuning_complete

    # The nlist tuning complete signal is chained through lj and the integrator.
    # Check that this chain is followed.
    nlist.tune_kernel_parameters()
    assert not nlist.is_tuning_complete
    assert not lj.is_tuning_complete
    assert not sim.operations.integrator.is_tuning_complete

    # langevin should remained tuned:
    assert langevin.is_tuning_complete
