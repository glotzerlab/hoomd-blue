# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
from hoomd import md
from hoomd.conftest import expected_loggable_params
from hoomd.conftest import (logging_check, pickling_check,
                            autotuned_kernel_parameter_check)
import pytest
import numpy

import itertools
# Test parameters: class, class keyword arguments (blank), bond params, positions, orientations, force, energy, torques

sqrt2inv = 1/numpy.sqrt(2)

patch_test_parameters = [
    (
        hoomd.md.pair.aniso.JanusLJ,
        {},
        {"pair_params": {"epsilon": 1,"sigma": 1},
         "envelope_params": {"alpha": numpy.pi/2,
                             "omega": 10,
                             "ni": (1,0,0),
                             "nj": (1,0,0)
                             }
         },
        [[0,0,0], [2,0,0]], # positions
        [[1,0,0,0], [sqrt2inv, 0, 0, sqrt2inv]], # orientations
        [0,0,0],
        -2.7559e-6,
        [[0,0,0], [0,0,0]]
    )
]


@pytest.fixture(scope='session')
def patchy_snapshot_factory(device):

    def make_snapshot(position_i = numpy.array([0,0,0]),
                      position_j = numpy.array([2,0,0]),
                      orientation_i = (1,0,0),
                      orientation_j = (1,0,0),
                      dimensions = 3,
                      L=20                      
                      ):
        snapshot = hoomd.Snapshot(device.communicator)
        if snapshot.communicator.rank == 0:
            N = 2
            box = [L, L, L, 0, 0, 0]
            if dimensions == 2:
                box[2] = 0
            snapshot.configuration.box = box
            snapshot.particles.N = N
            snapshot.particles.position[:] = [position_i, position_j]
            snapshot.particles.types = ['A']
            snapshot.particles.typeid[:] = 0
            snapshot.particles.moment_inertia[:] = [(1,1,1)]*N
            snapshot.particles.angmom[:] = [(0,0,0,0)]*N
        return snapshot

    return make_snapshot


@pytest.mark.parametrize('patch_cls, patch_args, params, positions, orientations, force, energy, torques',
                         patch_test_parameters)
def test_before_attaching(patch_cls, patch_args, params, positions, orientations, force, energy, torques):
    potential = patch_cls(nlist = hoomd.md.nlist.Cell(buffer=0.4), default_r_cut=4, **patch_args)
    potential.params[('A','A')] = params
    for key in params:
        assert potential.params[('A','A')][key] == pytest.approx(params[key])

        
@pytest.mark.parametrize('patch_cls, patch_args, params, positions, orientations, force, energy, torques',
                         patch_test_parameters)
def test_after_attaching(patchy_snapshot_factory, simulation_factory,
                         patch_cls, patch_args, params, positions, orientations, force, energy, torques):
    sim = simulation_factory(patchy_snapshot_factory())
    potential = patch_cls(nlist = hoomd.md.nlist.Cell(buffer=0.4), default_r_cut=4, **patch_args)
    potential.params[('A','A')] = params

    sim.operations.integrator = hoomd.md.Integrator(dt = 0.05,
                                                    forces = [potential])
    sim.run(0)
    for key in params:
        assert potential.params[('A','A')][key] == pytest.approx(params[key])


@pytest.mark.parametrize('patch_cls, patch_args, params, positions, orientations, force, energy, torques',
                         patch_test_parameters)
def test_forces_energies_torques(patchy_snapshot_factory, simulation_factory,
                                 patch_cls, patch_args, params, positions, orientations, force, energy, torques):

    snapshot = patchy_snapshot_factory(position_i = positions[0],
                                       position_j = positions[1],
                                       orientation_i = orientations[0],
                                       orientation_j = orientations[1])
    sim = simulation_factory(snapshot)

    potential = patch_cls(nlist = hoomd.md.nlist.Cell(buffer=0.4), default_r_cut=4, **patch_args)
    potential.params[('A','A')] = params

    sim.operations.integrator = hoomd.md.Integrator(dt = 0.005,
                                                    forces = [potential])

    sim.run(0)

    sim_forces = potential.forces
    sim_energy = potential.energy
    sim_torques = potential.torques
    if sim.device.communicator.rank == 0:
        assert sim_energy == pytest.approx(energy, rel=1e-2)
        
        # numpy.testing.assert_allclose(sim_forces[0],
        #                               force_array)

        # test force


        # test torque
    
    
