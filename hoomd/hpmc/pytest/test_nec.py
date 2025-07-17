# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import pytest


@pytest.mark.serial
def test_nec(simulation_factory, lattice_snapshot_factory):
    snap = lattice_snapshot_factory(particle_types=["A"])
    simulation = simulation_factory(snap)
    sphere = hoomd.hpmc.nec.integrate.Sphere()
    sphere.shape["A"] = {"diameter": 1.0}
    simulation.operations.integrator = sphere
    simulation.run(10)
