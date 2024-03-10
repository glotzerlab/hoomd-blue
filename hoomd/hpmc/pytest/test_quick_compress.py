# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test hoomd.hpmc.update.QuickCompress."""

import hoomd
from hoomd.conftest import operation_pickling_check
import pytest
import math
import numpy as np

# note: The parameterized tests validate parameters so we can't pass in values
# here that require preprocessing
valid_constructor_args = [
    # 3d box from constant box variant
    dict(trigger=hoomd.trigger.Periodic(10),
         target_box=hoomd.variant.box.Constant(hoomd.Box.from_box([10, 10,
                                                                   10]))),
    # 3d box with box object
    dict(trigger=hoomd.trigger.After(100),
         target_box=hoomd.Box.from_box([10, 20, 40]),
         max_overlaps_per_particle=0.2),
    # 2d box with box object
    dict(trigger=hoomd.trigger.Before(100),
         target_box=hoomd.Box.from_box([50, 50]),
         min_scale=0.75),
    # 2d box with box variant
    dict(trigger=hoomd.trigger.Before(100),
         target_box=hoomd.variant.box.Constant(hoomd.Box.from_box([50, 50])),
         min_scale=0.75),
    dict(trigger=hoomd.trigger.Periodic(1000),
         target_box=hoomd.variant.box.Constant(
             hoomd.Box.from_box([80, 50, 40, 0.2, 0.4, 0.5])),
         max_overlaps_per_particle=0.2,
         min_scale=0.999,
         allow_unsafe_resize=True),
    dict(trigger=hoomd.trigger.Periodic(1000),
         target_box=hoomd.variant.box.Constant(
             hoomd.Box.from_box([80, 50, 40, -0.2, 0.4, 0.5])),
         max_overlaps_per_particle=0.2,
         min_scale=0.999),
    dict(trigger=hoomd.trigger.Periodic(1000),
         target_box=hoomd.variant.box.Interpolate(
             hoomd.Box.from_box([10, 20, 30]), hoomd.Box.from_box([24, 7, 365]),
             hoomd.variant.Ramp(0, 1, 0, 100)),
         max_overlaps_per_particle=0.2,
         min_scale=0.999),
    dict(trigger=hoomd.trigger.Periodic(1000),
         target_box=hoomd.variant.box.InverseVolumeRamp(
             hoomd.Box.from_box([10, 20, 30]), 3000, 10, 100),
         max_overlaps_per_particle=0.2,
         min_scale=0.999),
]

valid_attrs = [
    ('trigger', hoomd.trigger.Periodic(10000)),
    ('trigger', hoomd.trigger.After(100)),
    ('trigger', hoomd.trigger.Before(12345)),
    ('target_box', hoomd.variant.box.Constant(hoomd.Box.from_box([10, 20,
                                                                  30]))),
    ('target_box', hoomd.variant.box.Constant(hoomd.Box.from_box([50, 50]))),
    ('target_box', hoomd.variant.box.Constant(hoomd.Box.from_box([50, 50]))),
    ('target_box',
     hoomd.variant.box.Interpolate(hoomd.Box.from_box([10, 20, 30]),
                                   hoomd.Box.from_box([24, 7, 365]),
                                   hoomd.variant.Ramp(0, 1, 0, 100))),
    ('target_box',
     hoomd.variant.box.InverseVolumeRamp(hoomd.Box.from_box([10, 20, 30]), 3000,
                                         10, 100)),
    ('max_overlaps_per_particle', 0.2),
    ('max_overlaps_per_particle', 0.5),
    ('max_overlaps_per_particle', 2.5),
    ('min_scale', 0.1),
    ('min_scale', 0.5),
    ('min_scale', 0.9999),
    ('allow_unsafe_resize', True),
]


@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction(constructor_args):
    """Test that QuickCompress can be constructed with valid arguments."""
    qc = hoomd.hpmc.update.QuickCompress(**constructor_args)

    # validate the params were set properly
    for attr, value in constructor_args.items():
        if type(value) is hoomd.Box:
            assert getattr(qc, attr) == hoomd.variant.box.Constant(value)
        else:
            assert getattr(qc, attr) == value


@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction_and_attach(simulation_factory,
                                       two_particle_snapshot_factory,
                                       constructor_args):
    """Test that QuickCompress can be attached with valid arguments."""
    qc = hoomd.hpmc.update.QuickCompress(**constructor_args)

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.updaters.append(qc)

    # QuickCompress requires an HPMC integrator
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=1)
    sim.operations.integrator = mc

    sim.operations._schedule()

    # validate the params were set properly
    for attr, value in constructor_args.items():
        if type(value) is hoomd.Box:
            assert getattr(qc, attr) == hoomd.variant.box.Constant(value)
        else:
            assert getattr(qc, attr) == value


@pytest.mark.parametrize("attr,value", valid_attrs)
def test_valid_setattr(attr, value):
    """Test that QuickCompress can get and set attributes."""
    qc = hoomd.hpmc.update.QuickCompress(trigger=hoomd.trigger.Periodic(10),
                                         target_box=hoomd.Box.from_box(
                                             [10, 10, 10]))

    setattr(qc, attr, value)
    assert getattr(qc, attr) == value


@pytest.mark.parametrize("attr,value", valid_attrs)
def test_valid_setattr_attached(attr, value, simulation_factory,
                                two_particle_snapshot_factory):
    """Test that QuickCompress can get and set attributes while attached."""
    qc = hoomd.hpmc.update.QuickCompress(trigger=hoomd.trigger.Periodic(10),
                                         target_box=hoomd.variant.box.Constant(
                                             hoomd.Box.from_box([10, 10, 10])))

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.updaters.append(qc)

    # QuickCompress requires an HPMC integrator
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=1)
    sim.operations.integrator = mc

    sim.operations._schedule()

    setattr(qc, attr, value)
    assert getattr(qc, attr) == value


@pytest.mark.parametrize("xy", [-0.4, 0, 0.2])
@pytest.mark.parametrize("xz", [-0.3, 0, 0.3])
@pytest.mark.parametrize("yz", [-0.2, 0, 0.4])
@pytest.mark.parametrize("phi", [0.01, 0.4, 0.6])
@pytest.mark.cpu
@pytest.mark.validate
def test_sphere_compression_triclinic(xy, xz, yz, phi, simulation_factory,
                                      lattice_snapshot_factory, device):
    """Test that QuickCompress can resize and reshape triclinic boxes."""
    n = 7 if device.communicator.num_ranks > 1 else 3
    if isinstance(device, hoomd.device.GPU):
        n = 8  # Increase simulation size even further to accomodate GPU

    snap = lattice_snapshot_factory(n=n, a=1.1)
    snap.configuration.box = hoomd.Box.from_box([20, 18, 16, xy, xz, yz])

    # Generate random tilts in [-1,1] and apply to the target box
    tilts = np.random.rand(3) * 2 - 1
    target_box = hoomd.Box.from_box([0.95, 1.05, 1, *tilts])

    v_particle = 4 / 3 * math.pi * (0.5)**3
    target_box.volume = n**3 * v_particle / phi

    qc = hoomd.hpmc.update.QuickCompress(trigger=hoomd.trigger.Periodic(25),
                                         target_box=target_box)

    sim = simulation_factory(snap)

    mc = hoomd.hpmc.integrate.Sphere(default_d=0.05)
    mc.shape['A'] = dict(diameter=1)
    sim.operations.integrator = mc
    sim.operations.updaters.append(qc)
    sim.run(1)

    # compression should not be complete yet
    assert not qc.complete
    # run long enough to compress the box
    while not qc.complete and sim.timestep < 1e5:
        sim.run(100)

    # Check that compression is complete and debug which statement is incorrect
    assert sim.state.box == target_box, f"{sim.state.box}!={target_box}"
    assert mc.overlaps == 0
    assert qc.complete


@pytest.mark.parametrize("phi", [0.2, 0.3, 0.4, 0.5, 0.55, 0.58, 0.6])
@pytest.mark.parametrize("allow_unsafe_resize", [False, True])
@pytest.mark.validate
def test_sphere_compression(phi, allow_unsafe_resize, simulation_factory,
                            lattice_snapshot_factory):
    """Test that QuickCompress can compress (and expand) simulation boxes."""
    if allow_unsafe_resize and phi > math.pi / 6:
        pytest.skip("Skipped impossible compression.")

    n = 7
    snap = lattice_snapshot_factory(n=n, a=1.1)
    v_particle = 4 / 3 * math.pi * (0.5)**3
    target_box = hoomd.Box.cube((n * n * n * v_particle / phi)**(1 / 3))

    qc = hoomd.hpmc.update.QuickCompress(
        trigger=hoomd.trigger.Periodic(10),
        target_box=target_box,
        allow_unsafe_resize=allow_unsafe_resize)

    sim = simulation_factory(snap)
    sim.operations.updaters.append(qc)

    mc = hoomd.hpmc.integrate.Sphere(default_d=0.05)
    mc.shape['A'] = dict(diameter=1)
    sim.operations.integrator = mc

    if allow_unsafe_resize:
        mc.d['A'] = 0

    sim.run(1)

    # compression should not be complete yet
    assert not qc.complete

    # run long enough to compress the box
    while not qc.complete and sim.timestep < 1e5:
        sim.run(100)

    # compression should end the run early
    assert qc.complete
    assert mc.overlaps == 0
    assert sim.state.box == target_box


@pytest.mark.parametrize("phi", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
@pytest.mark.validate
def test_disk_compression(phi, simulation_factory, lattice_snapshot_factory):
    """Test that QuickCompress can compress (and expand) simulation boxes."""
    n = 7
    snap = lattice_snapshot_factory(dimensions=2, n=n, a=1.1)
    v_particle = math.pi * (0.5)**2
    target_box = hoomd.Box.square((n * n * v_particle / phi)**(1 / 2))

    qc = hoomd.hpmc.update.QuickCompress(trigger=hoomd.trigger.Periodic(10),
                                         target_box=target_box)

    sim = simulation_factory(snap)
    sim.operations.updaters.append(qc)

    mc = hoomd.hpmc.integrate.Sphere(default_d=0.05)
    mc.shape['A'] = dict(diameter=1)
    sim.operations.integrator = mc

    sim.run(1)

    # compression should not be complete yet
    assert not qc.complete

    while not qc.complete and sim.timestep < 1e5:
        sim.run(100)

    # compression should end the run early
    assert qc.complete
    assert mc.overlaps == 0
    assert sim.state.box == target_box


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("same_initial_and_final_box", [True, False])
def test_inverse_volume_slow_compress(ndim, simulation_factory,
                                      same_initial_and_final_box,
                                      lattice_snapshot_factory):
    """Test that InverseVolumeRamp compresses at an appropriate rate.

    Ensure that the volume is no less than the set point determined by the
    specified t_start and t_ramp.
    """
    n = 6
    snap = lattice_snapshot_factory(dimensions=ndim, n=n, a=1.2)
    sim = simulation_factory(snap)
    if ndim == 2:
        close_packing = np.pi / 2 / np.sqrt(3)
        v_p = np.pi * 0.5**2
        fraction_close_packing = 0.91
    else:
        fraction_close_packing = 0.83
        close_packing = np.pi * np.sqrt(2) / 6
        v_p = 4 / 3 * np.pi * 0.5**3
    if same_initial_and_final_box:
        final_volume = sim.state.box.volume
    else:
        final_packing_fraction = fraction_close_packing * close_packing
        final_volume = n**ndim * v_p / final_packing_fraction
    t_ramp = 1000
    target_box = hoomd.variant.box.InverseVolumeRamp(sim.state.box,
                                                     final_volume, 0, t_ramp)
    qc = hoomd.hpmc.update.QuickCompress(trigger=hoomd.trigger.Periodic(10),
                                         target_box=target_box)

    sim.operations.updaters.append(qc)

    mc = hoomd.hpmc.integrate.Sphere(default_d=0.05)
    mc.shape['A'] = dict(diameter=1)
    sim.operations.integrator = mc

    while sim.timestep < t_ramp or (not qc.complete):
        sim.run(10)
        assert not sim.state.box.volume < hoomd.Box(
            *qc.target_box(sim.timestep)).volume


def test_pickling(simulation_factory, two_particle_snapshot_factory):
    """Test that QuickCompress objects are picklable."""
    qc = hoomd.hpmc.update.QuickCompress(trigger=hoomd.trigger.Periodic(10),
                                         target_box=hoomd.Box.square(10.))

    sim = simulation_factory(two_particle_snapshot_factory())
    mc = hoomd.hpmc.integrate.Sphere(default_d=0.05)
    mc.shape['A'] = dict(diameter=1)
    sim.operations.integrator = mc
    operation_pickling_check(qc, sim)
