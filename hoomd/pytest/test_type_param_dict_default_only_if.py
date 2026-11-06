# Copyright (c) 2009-2026 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

from hoomd.hpmc.pair import Step
from hoomd.util import make_example_simulation
from hoomd.hpmc.integrate import Sphere

def test_default_can_be_set_despite_only_if():
    step = Step()
    step.params.default = {"epsilon": [1, 2], "r": [1, 2]}

    sim = make_example_simulation()
    integrator = Sphere()
    integrator.shape["A"] = {"diameter": 1}
    integrator.pair_potentials.append(h)
    sim.operations.integrator = integrator
    sim.run(0)

    assert sim.operations.integrator.pair_potentials[0].params.default == {
        "epsilon": [1.0, 2.0], "r": [1.0, 2.0]
    }
