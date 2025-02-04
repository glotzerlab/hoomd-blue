# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import pytest
from itertools import permutations
from hoomd.logging import LoggerCategories
from hoomd.conftest import logging_check, autotuned_kernel_parameter_check

_directions = list(permutations(['x', 'y', 'z'], 2))


@pytest.mark.parametrize("slab_direction, flow_direction", _directions)
def test_before_attaching(slab_direction, flow_direction):
    filt = hoomd.filter.All()
    ramp = hoomd.variant.Ramp(0.0, 0.1e8, 0, int(1e8))
    n_slabs = 20
    mpf = hoomd.md.update.ReversePerturbationFlow(filt, ramp, slab_direction,
                                                  flow_direction, n_slabs)

    assert mpf.filter == filt
    assert mpf.flow_target == ramp
    assert mpf.slab_direction == slab_direction
    assert mpf.flow_direction == flow_direction
    assert mpf.n_slabs == n_slabs
    assert mpf.max_slab == n_slabs / 2
    assert mpf.min_slab == 0
    assert mpf.trigger == hoomd.trigger.Periodic(1)
    assert mpf.flow_epsilon == 1e-2
    with pytest.raises(hoomd.error.DataAccessError):
        mpf.summed_exchanged_momentum


@pytest.mark.parametrize("slab_direction, flow_direction", _directions)
def test_after_attaching(simulation_factory, two_particle_snapshot_factory,
                         slab_direction, flow_direction):
    filt = hoomd.filter.All()
    ramp = hoomd.variant.Ramp(0.0, 0.1e8, 0, int(1e8))
    n_slabs = 20
    mpf = hoomd.md.update.ReversePerturbationFlow(filt, ramp, slab_direction,
                                                  flow_direction, n_slabs)

    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[nve])
    sim.operations.add(mpf)
    sim.run(0)

    assert mpf.flow_target == ramp
    assert mpf.slab_direction == slab_direction
    assert mpf.flow_direction == flow_direction
    assert mpf.n_slabs == n_slabs
    assert mpf.max_slab == n_slabs / 2
    assert mpf.min_slab == 0
    assert mpf.trigger == hoomd.trigger.Periodic(1)
    assert mpf.summed_exchanged_momentum == 0

    with pytest.raises(AttributeError):
        # filter cannot be set after scheduling
        mpf.filter = hoomd.filter.Type(['A'])
    with pytest.raises(AttributeError):
        # flow_target cannot be set after scheduling
        mpf.flow_target = hoomd.variant.Ramp(0.0, 0.1e7, 0, int(1e7))
    with pytest.raises(AttributeError):
        # slab_direction cannot be set after scheduling
        mpf.slab_direction = flow_direction
    with pytest.raises(AttributeError):
        # flow_direction cannot be set after scheduling
        mpf.flow_direction = slab_direction
    with pytest.raises(AttributeError):
        # n_slabs cannot be set after scheduling
        mpf.n_slabs = 15
    with pytest.raises(AttributeError):
        # min_slab cannot be set after scheduling
        mpf.min_slab = 2
    with pytest.raises(AttributeError):
        # max_slab cannot be set after scheduling
        mpf.max_slab = 10
    with pytest.raises(AttributeError):
        # summed_exchanged_momentum cannot be set
        mpf.summed_exchanged_momentum = 1.5

    sim.run(10)

    if sim.device.communicator.num_ranks == 1:
        # ReversePerturbationFlow doesn't execute its kernel on all ranks,
        # test only on serial simulations.
        autotuned_kernel_parameter_check(instance=mpf,
                                         activate=lambda: sim.run(1),
                                         all_optional=True)


def test_logging():
    logging_check(
        hoomd.md.update.ReversePerturbationFlow, ('md', 'update'), {
            'summed_exchanged_momentum': {
                'category': LoggerCategories.scalar,
                'default': True
            }
        })
