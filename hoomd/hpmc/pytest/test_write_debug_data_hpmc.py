# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Test write_debug_data for HPMC simulations."""

import hoomd


def test_write_debug_data(simulation_factory, lattice_snapshot_factory,
                          tmp_path):
    """Test write_debug_data for MD simulations."""
    sim = simulation_factory(lattice_snapshot_factory())

    mc = hoomd.hpmc.integrate.ConvexPolyhedron(seed=2)
    mc.shape['A'] = dict(vertices=[
        (-0.5, 0, 0),
        (0.5, 0, 0),
        (0, -0.5, 0),
        (0, 0.5, 0),
        (0, 0, -0.5),
        (0, 0, 0.5),
    ])

    sim.operations.integrator = mc

    sim.write_debug_data(tmp_path / 'test_unscheduled.json')

    sim.run(10)

    sim.write_debug_data(tmp_path / 'test_scheduled.json')
