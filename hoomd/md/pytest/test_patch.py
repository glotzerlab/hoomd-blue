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


TOLERANCES = {"rtol": 1e-2, "atol": 1e-5}

# patch_test_parameters = [
#     (
#         hoomd.md.pair.aniso.JanusLJ,
#         {},
#         {"pair_params": {"epsilon": 1, "sigma": 1},
#          "envelope_params": {"alpha": numpy.pi/2,
#                              "omega": 10,
#                              "ni": (1,0,0),
#                              "nj": (1,0,0)
#                              }
#          },
#         [[0,0,0], [2,0,0]], # positions
#         [[1,0,0,0], [1, 0, 0, 0]], # orientations
#         [-8.245722889538097e-6, 0, 0],
#         -2.79291e-6, # energy
#         [[0,0,0], [0,0,0]] # todo put in right torque values
#     ),
#     (
#         hoomd.md.pair.aniso.JanusLJ,
#         {},
#         {"pair_params": {"epsilon": 1, "sigma": 1},
#          "envelope_params": {"alpha": 1.5707963267948966,
#                              "omega": 10,
#                              "ni": (1, 0, 0),
#                              "nj": (1, 0, 0)
#                              }
#          },
#         [[0, 0, 0], [0, 2, 1]],
#         [[1., 0., 0., 0.], [1., 0., 0., 0.]],
#         [0.03549087093887666, 0.0188928, 0.0094464],
#         -0.007936,
#         [[0., -0.01774543546943833, 0.03549087093887666],
#          [0., 0.01774543546943833, -0.03549087093887666]])
# ]

patch_test_parameters = [(hoomd.md.pair.aniso.PatchyLJ, {}, {"pair_params": {"epsilon": 1, "sigma": 1}, "envelope_params": {"alpha": 45., "omega": 20}}, [[0, 0, 0], [0.8426488874308758, 0.7070663706551933, 0]], [[1., 0., 0., 0.], [0.3420201433256688, 0., 0., 0.9396926207859083]], [-0.00025376481426376174, 0.00025849718238020263, 0.], -0.9833724493736826, [[0, 0, 0], [0, 0, 0]])]



@pytest.fixture(scope='session')
def patchy_snapshot_factory(device):

    def make_snapshot(position_i = numpy.array([0,0,0]),
                      position_j = numpy.array([2,0,0]),
                      orientation_i = (1,0,0,0),
                      orientation_j = (1,0,0,0),
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
            snapshot.particles.orientation[:] = [orientation_i, orientation_j]
            snapshot.particles.types = ['A']
            snapshot.particles.typeid[:] = 0
            snapshot.particles.moment_inertia[:] = [(1,1,1)]*N
            snapshot.particles.angmom[:] = [(0,0,0,0)]*N
        return snapshot

    return make_snapshot

# TODO: exclude the one that is not normalized
@pytest.mark.parametrize('patch_cls, patch_args, params, positions, orientations, force, energy, torques',
                         patch_test_parameters)
def test_before_attaching(patch_cls, patch_args, params, positions, orientations, force, energy, torques):
    potential = patch_cls(nlist = hoomd.md.nlist.Cell(buffer=0.4), default_r_cut=4, **patch_args)
    potential.params[('A','A')] = params
    for key in params:
        # ni and nj are returned normalized, so replace them in the params we check
        if key == "envelope_params":
            for nkey in ("ni", "nj"):
                nn = numpy.array(params[key][nkey])
                params[key][nkey] = tuple(nn / numpy.linalg.norm(nn))
            assert potential.params[('A','A')][key] == pytest.approx(params[key])
        else:
            assert potential.params[('A','A')][key] == pytest.approx(params[key])


# TODO: exclude the one that is not normalized
@pytest.mark.parametrize('patch_cls, patch_args, params, positions, orientations, force, energy, torques',
                         patch_test_parameters)
def test_after_attaching(patchy_snapshot_factory, simulation_factory,
                         patch_cls, patch_args, params, positions, orientations, force, energy, torques):
    sim = simulation_factory(patchy_snapshot_factory())
    potential = patch_cls(nlist = hoomd.md.nlist.Cell(buffer=0.4), default_r_cut=4, **patch_args)
    potential.params[('A','A')] = params

    sim.operations.integrator = hoomd.md.Integrator(dt = 0.05,
                                                    forces = [potential],
                                                    integrate_rotational_dof = True)
    sim.run(0)
    for key in params:
        # ni and nj are returned normalized, so replace them in the params we check
        if key == "envelope_params":
            for nkey in ("ni", "nj"):
                nn = numpy.array(params[key][nkey])
                params[key][nkey] = tuple(nn / numpy.linalg.norm(nn))
            assert potential.params[('A','A')][key] == pytest.approx(params[key])
        else:
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
                                                    forces = [potential],
                                                    integrate_rotational_dof = True)
    sim.run(0)

    sim_forces = potential.forces
    sim_energy = potential.energy
    sim_torques = potential.torques
    if sim.device.communicator.rank == 0:

        sim_orientations = snapshot.particles.orientation

        numpy.testing.assert_allclose(sim_orientations, orientations, **TOLERANCES)
        
        numpy.testing.assert_allclose(sim_energy, energy, **TOLERANCES)

        numpy.testing.assert_allclose(sim_forces[0], force, **TOLERANCES)

        numpy.testing.assert_allclose(sim_forces[1],  [-force[0], -force[1], -force[2]], **TOLERANCES)

        numpy.testing.assert_allclose(sim_torques[0], torques[0], **TOLERANCES)

        numpy.testing.assert_allclose(sim_torques[1], torques[1], **TOLERANCES)
    
    
