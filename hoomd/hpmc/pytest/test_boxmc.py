# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test hoomd.hpmc.update.BoxMC."""

import hoomd
from hoomd.conftest import operation_pickling_check, logging_check
from hoomd.data.collections import _HOOMDSyncedCollection
from hoomd.logging import LoggerCategories
import pytest
import numpy as np

valid_constructor_args = [
    dict(trigger=hoomd.trigger.Periodic(10), betaP=10),
    dict(trigger=hoomd.trigger.After(100),
         betaP=hoomd.variant.Ramp(1, 5, 0, 100)),
    dict(trigger=hoomd.trigger.Before(100),
         betaP=hoomd.variant.Cycle(1, 5, 0, 10, 20, 10, 15)),
    dict(trigger=hoomd.trigger.Periodic(1000),
         betaP=hoomd.variant.Power(1, 5, 3, 0, 100)),
]

valid_attrs = [('betaP', hoomd.variant.Constant(10)),
               ('betaP', hoomd.variant.Ramp(1, 5, 0, 100)),
               ('betaP', hoomd.variant.Cycle(1, 5, 0, 10, 20, 10, 15)),
               ('betaP', hoomd.variant.Power(1, 5, 3, 0, 100)),
               ('volume', {
                   'mode': 'standard',
                   'weight': 0.7,
                   'delta': 0.3
               }), ('volume', {
                   'mode': 'ln',
                   'weight': 0.1,
                   'delta': 1.2
               }), ('aspect', {
                   'weight': 0.3,
                   'delta': 0.1
               }), ('length', {
                   'weight': 0.5,
                   'delta': [0.8] * 3
               }), ('shear', {
                   'weight': 0.7,
                   'delta': [0.3] * 3,
                   'reduce': 0.1
               })]

box_moves_attrs = [{
    'move': 'volume',
    "params": {
        'mode': 'standard',
        'weight': 1,
        'delta': 0.001
    }
}, {
    'move': 'volume',
    "params": {
        'mode': 'ln',
        'weight': 1,
        'delta': 0.001
    }
}, {
    'move': 'aspect',
    "params": {
        'weight': 1,
        'delta': 0.001
    }
}, {
    'move': 'shear',
    "params": {
        'weight': 1,
        'delta': (0.001,) * 3,
        'reduce': 0.2
    }
}, {
    'move': 'length',
    "params": {
        'weight': 1,
        'delta': (0.001,) * 3
    }
}]


@pytest.fixture
def counter_attrs():
    return {
        'volume': "volume_moves",
        'length': "volume_moves",
        'aspect': "aspect_moves",
        'shear': "shear_moves"
    }


def _is_close(v1, v2):
    if isinstance(v1, _HOOMDSyncedCollection):
        v1 = v1.to_base()
    if isinstance(v2, _HOOMDSyncedCollection):
        v2 = v2.to_base()

    return v1 == v2 if isinstance(v1, str) else np.allclose(v1, v2)


def obj_attr_check(boxmc, mapping):
    for attr, value in mapping.items():
        obj_value = getattr(boxmc, attr)
        if (isinstance(obj_value, hoomd.variant.Constant)
                and not isinstance(value, hoomd.variant.Constant)):
            assert obj_value(0) == value
            continue
        assert getattr(boxmc, attr) == value


@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction(constructor_args):
    """Test that BoxMC can be constructed with valid arguments."""
    boxmc = hoomd.hpmc.update.BoxMC(**constructor_args)

    # validate the params were set properly
    obj_attr_check(boxmc, constructor_args)


@pytest.mark.parametrize("constructor_args", valid_constructor_args)
def test_valid_construction_and_attach(simulation_factory,
                                       two_particle_snapshot_factory,
                                       constructor_args):
    """Test that BoxMC can be attached with valid arguments."""
    boxmc = hoomd.hpmc.update.BoxMC(**constructor_args)

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.updaters.append(boxmc)

    # BoxMC requires an HPMC integrator
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=1)
    sim.operations.integrator = mc

    # create C++ mirror classes and set parameters
    sim.run(0)

    # validate the params were set properly
    obj_attr_check(boxmc, constructor_args)


@pytest.mark.parametrize("attr,value", valid_attrs)
def test_valid_setattr(attr, value):
    """Test that BoxMC can get and set attributes."""
    boxmc = hoomd.hpmc.update.BoxMC(trigger=hoomd.trigger.Periodic(10),
                                    betaP=10)

    setattr(boxmc, attr, value)
    if isinstance(value, dict):
        # check if we have the same keys
        assert value.keys() == getattr(boxmc, attr).keys()
        for k in value.keys():
            assert _is_close(value[k], getattr(boxmc, attr)[k])
    else:
        assert getattr(boxmc, attr) == value


@pytest.mark.parametrize("attr,value", valid_attrs)
def test_valid_setattr_attached(attr, value, simulation_factory,
                                two_particle_snapshot_factory):
    """Test that BoxMC can get and set attributes while attached."""
    boxmc = hoomd.hpmc.update.BoxMC(trigger=hoomd.trigger.Periodic(10),
                                    betaP=10)

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.updaters.append(boxmc)

    # BoxMC requires an HPMC integrator
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=1)
    sim.operations.integrator = mc

    # create C++ mirror classes and set parameters
    sim.run(0)

    setattr(boxmc, attr, value)
    if isinstance(value, dict):
        # check if we have the same keys
        assert value.keys() == getattr(boxmc, attr).keys()
        for k in value.keys():
            assert _is_close(value[k], getattr(boxmc, attr)[k])
    else:
        assert getattr(boxmc, attr) == value


@pytest.mark.parametrize("betaP", [1, 3, 5, 7, 10])
@pytest.mark.parametrize("box_move", box_moves_attrs)
def test_sphere_compression(betaP, box_move, simulation_factory,
                            lattice_snapshot_factory):
    """Test that BoxMC can compress (and expand) simulation boxes."""
    n = 7
    snap = lattice_snapshot_factory(dimensions=3, n=n, a=1.3)

    boxmc = hoomd.hpmc.update.BoxMC(betaP=betaP, trigger=1)

    sim = simulation_factory(snap)
    initial_box = sim.state.box

    sim.operations.updaters.append(boxmc)
    mc = hoomd.hpmc.integrate.Sphere(default_d=0.05)
    mc.shape['A'] = dict(diameter=1)
    sim.operations.integrator = mc

    # run w/o setting any of the box moves
    sim.run(1)

    # check that the box remains unchanged
    assert mc.overlaps == 0
    assert sim.state.box == initial_box

    # add a box move
    setattr(boxmc, box_move['move'], box_move['params'])
    sim.run(5)

    # check that box is changed
    assert mc.overlaps == 0
    assert sim.state.box != initial_box


@pytest.mark.parametrize("betaP", [1, 3, 5, 7, 10])
@pytest.mark.parametrize("box_move", box_moves_attrs)
def test_disk_compression(betaP, box_move, simulation_factory,
                          lattice_snapshot_factory):
    """Test that BoxMC can compress (and expand) simulation boxes."""
    n = 7
    snap = lattice_snapshot_factory(dimensions=2, n=n, a=1.3)

    boxmc = hoomd.hpmc.update.BoxMC(betaP=betaP, trigger=1)

    sim = simulation_factory(snap)
    initial_box = sim.state.box

    sim.operations.updaters.append(boxmc)
    mc = hoomd.hpmc.integrate.Sphere(default_d=0.05)
    mc.shape['A'] = dict(diameter=1)
    sim.operations.integrator = mc

    # run w/o setting any of the box moves
    sim.run(10)

    # check that the box remains unchanged
    assert mc.overlaps == 0
    assert sim.state.box == initial_box

    # add a box move
    setattr(boxmc, box_move['move'], box_move['params'])
    sim.run(50)

    # check that box is changed
    assert mc.overlaps == 0
    assert sim.state.box != initial_box


@pytest.mark.parametrize("box_move", box_moves_attrs)
def test_counters(box_move, simulation_factory, lattice_snapshot_factory,
                  counter_attrs):
    """Test that BoxMC counters count corectly."""
    boxmc = hoomd.hpmc.update.BoxMC(betaP=3, trigger=1)
    # check result when box object is unattached
    for v in counter_attrs.values():
        assert getattr(boxmc, v) == (0, 0)

    n = 7
    snap = lattice_snapshot_factory(dimensions=2, n=n, a=1.3)
    sim = simulation_factory(snap)

    sim.operations.updaters.append(boxmc)
    mc = hoomd.hpmc.integrate.Sphere(default_d=0.05)
    mc.shape['A'] = dict(diameter=1)
    sim.operations.integrator = mc

    # run w/o setting any of the box moves
    sim.run(1)

    # check results after attaching but with zero weights and deltas
    for v in counter_attrs.values():
        assert getattr(boxmc, v) == (0, 0)

    # add a box move
    setattr(boxmc, box_move['move'], box_move['params'])
    # run with box move
    sim.run(10)

    # check some moves are accepted after properly setting a box move
    for (k, v) in counter_attrs.items():
        if k == box_move['move']:
            ctr = getattr(boxmc, v)
            assert ctr[0] > 0
            assert ctr[0] + ctr[1] == 10


@pytest.mark.parametrize("box_move", box_moves_attrs)
def test_pickling(box_move, simulation_factory, two_particle_snapshot_factory):
    boxmc = hoomd.hpmc.update.BoxMC(betaP=3, trigger=1)
    setattr(boxmc, box_move['move'], box_move['params'])
    sim = simulation_factory(two_particle_snapshot_factory())
    mc = hoomd.hpmc.integrate.Sphere()
    mc.shape['A'] = dict(diameter=1)
    sim.operations.integrator = mc
    operation_pickling_check(boxmc, sim)


def test_logging():
    logging_check(
        hoomd.hpmc.update.BoxMC, ('hpmc', 'update'), {
            'aspect_moves': {
                'category': LoggerCategories.sequence,
                'default': True
            },
            'shear_moves': {
                'category': LoggerCategories.sequence,
                'default': True
            },
            'volume_moves': {
                'category': LoggerCategories.sequence,
                'default': True
            }
        })
