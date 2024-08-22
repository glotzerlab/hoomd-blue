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

patch_test_parameters = [(hoomd.md.pair.aniso.PatchyLJ, {}, {"pair_params": {"epsilon": 1, "sigma": 1}, "envelope_params": {"alpha": 0.7853981633974483, "omega": 20}}, [[1, 0, 0]], [[0, 0, 0], [0.8426488874308758, 0.7070663706551933, 0]], [[1., 0., 0., 0.], [0.3420201433256688, 0., 0., 0.9396926207859083]], [-0.0002552170704854609, 0.000259976521204464, 0.], -0.000017584424790767245, [[0., 0., 0.00005318682937777501], [0., 0., 0.00034633750473072174]]),
                         (hoomd.md.pair.aniso.PatchyLJ, {}, {"pair_params": {"epsilon": 1, "sigma": 1}, "envelope_params": {"alpha": 0.7853981633974483, "omega": 2}}, [[1, 0, 0]], [[0, 0, 0], [0.8426488874308758, 0.7070663706551933, 0]], [[1., 0., 0., 0.], [0.3420201433256688, 0., 0., 0.9396926207859083]], [-0.7623810742102363, 0.1693907687094742, 0.], -0.29421101626557683, [[0., 0., 0.18937616300473295], [0., 0., 0.49241479898740603]]),
                         (hoomd.md.pair.aniso.PatchyGaussian, {}, {"pair_params": {"epsilon": 1, "sigma": 1}, "envelope_params": {"alpha": 0.7853981633974483, "omega": 2}}, [[1, 0, 0]], [[0, 0, 0], [0.8426488874308758, 0.7070663706551933, 0]], [[1., 0., 0., 0.], [0.3420201433256688, 0., 0., 0.9396926207859083]], [0.08356804332549303, -0.3791801250032115, 0.], 0.16337768270876143, [[0., 0., -0.10516206722885364], [0., 0., -0.2734417963379153]]),
                         (hoomd.md.pair.aniso.PatchyGaussian, {}, {"pair_params": {"epsilon": 1, "sigma": 1}, "envelope_params": {"alpha": 0.7853981633974483, "omega": 2}}, [[1, 0, 0]], [[0, 0, 0], [0.8426488874308758, 0.7070663706551933, 0]], [[1., 0., 0., 0.], [0.3420201433256688, 0., 0., 0.9396926207859083]], [0.08356804332549309, -0.37918012500321163, 0.], 0.16337768270876143, [[0., 0., -0.10516206722885363], [0., 0., -0.2734417963379153]]),
                         (hoomd.md.pair.aniso.PatchyYukawa, {}, {"pair_params": {"epsilon": 1, "kappa": 1}, "envelope_params": {"alpha": 0.7853981633974483, "omega": 2}}, [[1, 0, 0]], [[0, 0, 0], [0.8426488874308758, 0.7070663706551933, 0]], [[1., 0., 0., 0.], [0.3420201433256688, 0., 0., 0.9396926207859083]], [-0.009804880564917191, -0.2572104096366956, 0.], 0.09053662025522569, [[0., 0., -0.058276124303498], [0., 0., -0.15152923989675965]]),
                         (hoomd.md.pair.aniso.PatchyYukawa, {}, {"pair_params": {"epsilon": 1, "kappa": 1}, "envelope_params": {"alpha": 0.7853981633974483, "omega": 2}}, [[0.5773502691896258, 0.5773502691896258, 0.5773502691896258]], [[0, 0, 0], [0.8426488874308758, 0.7070663706551933, 0]], [[1., 0., 0., 0.], [0.3420201433256688, 0., 0., 0.9396926207859083]], [-0.22747920796066617, -0.29303852855614165, -0.01626378797975883], 0.1899441858809721, [[0.06687983781663602, -0.07970428699009408, 0.012824449173458055], [-0.07837941535658961, 0.09340894983664945, -0.09891014126340553]]),
                         (hoomd.md.pair.aniso.PatchyYukawa, {}, {"pair_params": {"epsilon": 1, "kappa": 1}, "envelope_params": {"alpha": 0.7853981633974483, "omega": 2}}, [[-1., 0., 0.]], [[0, 0, 0], [-0.19101299543362338, 1.0832885283134288, 0]], [[1., 0., 0., 0.], [0.3420201433256688, 0., 0., 0.9396926207859083]], [-0.15782069012466707, -0.20333706075613178, 0.], 0.09053662025522574, [[0, 0., 0.1515292398967598], [0., 0., 0.05827612430349796]])]


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
            snapshot.particles.types = ['A', 'B']
            snapshot.particles.typeid[:] = [0, 1]
            snapshot.particles.moment_inertia[:] = [(1,1,1)]*N
            snapshot.particles.angmom[:] = [(0,0,0,0)]*N
        return snapshot

    return make_snapshot


@pytest.mark.parametrize('patch_cls, patch_args, params, patches_A, patches_B, positions, orientations, force, energy, torques',
                         patch_test_parameters)
def test_before_attaching(patch_cls, patch_args, params, patches_A, patches_B, positions, orientations, force, energy, torques):
    potential = patch_cls(nlist = hoomd.md.nlist.Cell(buffer=0.4), default_r_cut=4, **patch_args)
    potential.params.default = params
    potential.patches['A'] = patches_A
    potential.patches['B'] = patches_B
    for key in params:
        assert potential.params[('A','A')][key] == pytest.approx(params[key])
    for i,patch in enumerate(patches_A):
        # patch is returned normalized, so normalize it before checking
        nn = numpy.array(patch)
        patch = tuple(nn / numpy.linalg.norm(nn))
        assert potential.patches['A'][i] == pytest.approx(nn)
    for i,patch in enumerate(patches_B):
        # patch is returned normalized, so normalize it before checking
        nn = numpy.array(patch)
        patch = tuple(nn / numpy.linalg.norm(nn))
        assert potential.patches['B'][i] == pytest.approx(nn)


@pytest.mark.parametrize('patch_cls, patch_args, params, patches_A, patches_B, positions, orientations, force, energy, torques',
                         patch_test_parameters)
def test_after_attaching(patchy_snapshot_factory, simulation_factory,
                         patch_cls, patch_args, params, patches_A, patches_B, positions, orientations, force, energy, torques):
    sim = simulation_factory(patchy_snapshot_factory())
    potential = patch_cls(nlist = hoomd.md.nlist.Cell(buffer=0.4), default_r_cut=4, **patch_args)
    potential.params.default = params
    potential.patches['A'] = patches_A
    potential.patches['B'] = patches_B

    sim.operations.integrator = hoomd.md.Integrator(dt = 0.05,
                                                    forces = [potential],
                                                    integrate_rotational_dof = True)
    sim.run(0)
    for key in params:
        assert potential.params[('A','A')][key] == pytest.approx(params[key])
    for i,patch in enumerate(patches_A):
        # patch is returned normalized, so normalize it before checking
        nn = numpy.array(patch)
        patch = tuple(nn / numpy.linalg.norm(nn))
        assert potential.patches['A'][i] == pytest.approx(nn)
    for i,patch in enumerate(patches_B):
        # patch is returned normalized, so normalize it before checking
        nn = numpy.array(patch)
        patch = tuple(nn / numpy.linalg.norm(nn))
        assert potential.patches['B'][i] == pytest.approx(nn)


@pytest.mark.parametrize('patch_cls, patch_args, params, patches_A, patches_B, positions, orientations, force, energy, torques',
                         patch_test_parameters)
def test_forces_energies_torques(patchy_snapshot_factory, simulation_factory,
                                 patch_cls, patch_args, params, patches_A, patches_B, positions, orientations, force, energy, torques):

    snapshot = patchy_snapshot_factory(position_i = positions[0],
                                       position_j = positions[1],
                                       orientation_i = orientations[0],
                                       orientation_j = orientations[1])
    sim = simulation_factory(snapshot)

    potential = patch_cls(nlist = hoomd.md.nlist.Cell(buffer=0.4), default_r_cut=4, **patch_args)
    potential.params.default = params
    potential.patches['A'] = patches_A
    potential.patches['B'] = patches_B

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
    

# Move this to validation
# @pytest.mark.parametrize('patch_cls, patch_args, params, positions, orientations, force, energy, torques',
#                          patch_test_parameters)
# def test_energy_drift(patchy_snapshot_factory, simulation_factory,
#                       patch_cls, patch_args, params, positions, orientations, force, energy, torques):
