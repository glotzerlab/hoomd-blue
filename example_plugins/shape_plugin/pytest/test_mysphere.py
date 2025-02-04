# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

from hoomd import shape_plugin


def test_attach(simulation_factory, two_particle_snapshot_factory):
    mc = shape_plugin.integrate.MySphere()
    mc.shape["A"] = dict(radius=0.5)
    mc.d["A"] = 0.1
    mc.a["A"] = 0.1
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = mc

    sim.run(0)
