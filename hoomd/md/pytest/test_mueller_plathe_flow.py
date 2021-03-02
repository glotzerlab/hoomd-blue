import hoomd
import pytest
import numpy
from itertools import permutations


_directions = permutations([hoomd.md.update.MuellerPlatheFlow.X,
                            hoomd.md.update.MuellerPlatheFlow.Y,
                            hoomd.md.update.MuellerPlatheFlow.Z],
                           2)

@pytest.mark.parametrize("slab_direction, flow_direction", _directions)
def test_before_attaching(slab_direction, flow_direction):
    filt = hoomd.filter.All()
    ramp = hoomd.variant.Ramp(0.0, 0.1e8, 0, int(1e8))
    n_slabs = 20
    mpf = hoomd.md.update.MuellerPlatheFlow(filt, ramp, slab_direction, flow_direction, n_slabs)

    assert mpf.filter is filt
    assert mpf.flow_target is ramp
    assert mpf.slab_direction is slab_direction
    assert mpf.flow_direction is flow_direction
    assert mpf.n_slabs == n_slabs
    assert mpf.max_slab == n_slabs / 2
    assert mpf.min_slab == 0
    assert mpf.trigger == hoomd.trigger.Periodic(1)
    assert mpf.has_max_slab is None
    assert mpf.has_min_slab is None
    assert mpf.summed_exchanged_momentum is None
    assert mpf.flow_epsilon == 1e-2


@pytest.mark.parametrize("slab_direction, flow_direction", _directions)
def test_after_attaching(simulation_factory, two_particle_snapshot_factory,
                         slab_direction, flow_direction):
    filt = hoomd.filter.All()
    ramp = hoomd.variant.Ramp(0.0, 0.1e8, 0, int(1e8))
    n_slabs = 20
    mpf = hoomd.md.update.MuellerPlatheFlow(filt, ramp, slab_direction, flow_direction, n_slabs)

    nve = hoomd.md.methods.NVE(filter=hoomd.filter.All())
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[nve])
    sim.operations.add(mpf)
    sim.run(0)

    assert mpf.flow_target is ramp
    assert mpf.slab_direction == slab_direction
    assert mpf.flow_direction == flow_direction
    assert mpf.n_slabs == n_slabs
    assert mpf.max_slab == n_slabs / 2
    assert mpf.min_slab == 0
    assert mpf.trigger == hoomd.trigger.Periodic(1)
    assert isinstance(mpf.has_max_slab, bool)
    assert isinstance(mpf.has_min_slab, bool)
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
        # has_max_slab cannot be set
        mpf.has_max_slab = False
    with pytest.raises(AttributeError):
        # has_min_slab cannot be set
        mpf.has_max_slab = False
    with pytest.raises(AttributeError):
        # summed_exchanged_momentum cannot be set
        mpf.summed_exchanged_momentum = 1.5

    sim.run(1)
    flow_epsilon = mpf.flow_epsilon
    flow_epsilon *= 2
    mpf.flow_epsilon = flow_epsilon
    assert mpf.flow_epsilon == flow_epsilon

    sim.run(10)
