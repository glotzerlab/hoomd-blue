# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

from collections import namedtuple
from collections.abc import Mapping, Sequence
import itertools
import json
import math
from numbers import Number
from pathlib import Path

import numpy as np
import pytest

import hoomd
from hoomd.conftest import (pickling_check, logging_check,
                            autotuned_kernel_parameter_check)
from hoomd.logging import LoggerCategories
from hoomd import md
from hoomd.error import TypeConversionError


def _equivalent_data_structures(struct_1, struct_2):
    """Compare arbitrary data structures for equality.

    ``struct_1`` is expected to be the reference data structure. Cannot handle
    set like data structures.
    """
    if isinstance(struct_1, np.ndarray):
        return np.allclose(struct_1, struct_2)
    if isinstance(struct_1, Mapping):
        if set(struct_1.keys()) != set(struct_2.keys()):
            return False
        return all(
            _equivalent_data_structures(struct_1[key], struct_2[key])
            for key in struct_1)
    if isinstance(struct_1, Sequence):
        if len(struct_1) != len(struct_2):
            return False
        return all(
            _equivalent_data_structures(value_1, value_2)
            for value_1, value_2 in zip(struct_1, struct_2))
    if isinstance(struct_1, Number):
        return math.isclose(struct_1, struct_2)
    return False


def assert_equivalent_data_structures(struct_1, struct_2):
    assert _equivalent_data_structures(struct_1, struct_2)


def make_langevin_integrator(force):
    integrator = md.Integrator(dt=0.005, integrate_rotational_dof=True)
    integrator.forces.append(force)
    integrator.methods.append(md.methods.Langevin(hoomd.filter.All(), kT=1))
    return integrator


@pytest.fixture
def make_two_particle_simulation(two_particle_snapshot_factory,
                                 simulation_factory):

    def make_simulation(force, d=1, types=None, dimensions=3):
        if types is None:
            types = ['A']
        snap = two_particle_snapshot_factory(dimensions=dimensions,
                                             d=d,
                                             particle_types=types)
        if snap.communicator.rank == 0:
            snap.particles.charge[:] = 1.
            snap.particles.moment_inertia[0] = [1., 1., 1.]
            snap.particles.moment_inertia[1] = [1., 2., 1.]
        sim = simulation_factory(snap)
        sim.operations.integrator = make_langevin_integrator(force)
        return sim

    return make_simulation


@pytest.mark.parametrize("mode", [('none', 'shift'), ('shift', 'none')])
def test_mode(make_two_particle_simulation, mode):
    """Test that all modes are correctly set on construction."""
    cell = md.nlist.Cell(buffer=0.4)
    # Test setting on construction
    gay_berne = md.pair.aniso.GayBerne(nlist=cell,
                                       default_r_cut=2.5,
                                       mode=mode[0])
    assert gay_berne.mode == mode[0]

    # Test setting
    gay_berne.mode = mode[1]
    assert gay_berne.mode == mode[1]

    # Ensure that mode remains the same after attaching
    gay_berne.params[('A', 'A')] = {'epsilon': 1, 'lpar': 0.5, 'lperp': 1.0}
    sim = make_two_particle_simulation(dimensions=3, d=0.5, force=gay_berne)
    sim.run(0)
    assert gay_berne.mode == mode[1]


@pytest.mark.parametrize("mode", ['foo', 1, True, 'xplor'])
def test_mode_invalid(mode):
    """Test mode validation on construction and setting."""
    # Test errors on construction
    with pytest.raises(TypeConversionError):
        gay_berne = md.pair.aniso.GayBerne(nlist=md.nlist.Cell(buffer=0.4),
                                           default_r_cut=2.5,
                                           mode=mode)
    gay_berne = md.pair.aniso.GayBerne(nlist=md.nlist.Cell(buffer=0.4),
                                       default_r_cut=2.5)
    gay_berne.params[('A', 'A')] = {'epsilon': 1, 'lpar': 0.5, 'lperp': 1.0}
    # Test errors on setting
    with pytest.raises(TypeConversionError):
        gay_berne.mode = mode


@pytest.mark.parametrize("r_cut", [2.5, 4.0, 1.0, 0.01])
def test_rcut(make_two_particle_simulation, r_cut):
    """Test that r_cut is correctly set and settable."""
    cell = md.nlist.Cell(buffer=0.4)
    # Test construction
    gay_berne = md.pair.aniso.GayBerne(nlist=cell, default_r_cut=r_cut)
    assert gay_berne.r_cut.default == r_cut

    # Test setting
    new_r_cut = r_cut * 1.1
    gay_berne.r_cut[('A', 'A')] = new_r_cut
    assert gay_berne.r_cut[('A', 'A')] == new_r_cut

    expected_r_cut = {('A', 'A'): new_r_cut}
    assert_equivalent_data_structures(gay_berne.r_cut.to_base(), expected_r_cut)

    gay_berne.params[('A', 'A')] = {'epsilon': 1, 'lpar': 0.5, 'lperp': 1.0}
    sim = make_two_particle_simulation(dimensions=3, d=.5, force=gay_berne)

    # Check after attaching
    sim.run(0)
    assert_equivalent_data_structures(gay_berne.r_cut.to_base(), expected_r_cut)


@pytest.mark.parametrize("r_cut", [-1., 'foo', None])
def test_rcut_invalid(r_cut):
    """Test r_cut validation logic."""
    cell = md.nlist.Cell(buffer=0.4)
    # Test construction error
    if r_cut is not None:
        with pytest.raises(TypeConversionError):
            gay_berne = md.pair.aniso.GayBerne(nlist=cell, default_r_cut=r_cut)
    # Test setting error
    gay_berne = md.pair.aniso.GayBerne(nlist=cell, default_r_cut=2.5)
    with pytest.raises(ValueError):
        gay_berne.r_cut[('A', 'B')] = r_cut


def isclose(value, reference, rtol=5e-6):
    """Return if two values are close while automatically managing atol."""
    if isinstance(reference, (Sequence, np.ndarray)):
        ref = np.asarray(reference, np.float64)
        val = np.asarray(reference, np.float64)
        min_value = np.min(np.abs(reference))
        atol = 1e-6 if min_value == 0 else min_value / 1e4
        return np.allclose(val, ref, rtol=rtol, atol=atol)
    else:
        atol = 1e-6 if reference == 0 else 0
        return math.isclose(value, reference, rel_tol=rtol, abs_tol=atol)


def expand_dict(iterable_dict):
    for values in zip(*iterable_dict.values()):
        yield dict(zip(iterable_dict.keys(), values))


AnisoPotentialSpecification = namedtuple("AnisoParametersSpecification",
                                         ("cls", "type_parameters"))


def make_aniso_spec(cls, type_parameters=None):
    if type_parameters is None:
        type_parameters = {}
    return AnisoPotentialSpecification(cls, type_parameters)


def _valid_params(particle_types=['A', 'B']):
    """Create valid full specifications for anisotropic potentials."""

    def to_type_parameter_dicts(types, argument_dict):
        """Converts a list of types and type parameter values into dicts.

        Args:
            types (list):
                A list of valid types
            argument_dict (dict):
                A dictionary structured with the keys as type parameters for the
                given class, and the values tuples with the form (iterable,
                num_types) where num_types is the number of types per key. The
                interable in the tuple is an iterable of valid values for this
                type parameter. If passing a dict, the dict should have all its
                values be iterables of the same length and the dict is
                transformed into a an iterable of dicts using the iterable
                values of the dict.

        Returns:
            A dictionary of the form {type_parameter_name: value} where value
            specifies the values for all keys given types.
        """
        type_parameters_dicts = {}
        for name, (values, num_types) in argument_dict.items():
            if num_types > 1:
                type_keys = itertools.combinations_with_replacement(
                    particle_types, num_types)
            else:
                type_keys = particle_types

            if isinstance(values, Mapping):
                tp_spec = {
                    type_key: spec
                    for type_key, spec in zip(type_keys, expand_dict(values))
                }
            else:
                tp_spec = {
                    type_key: spec for type_key, spec in zip(type_keys, values)
                }
            type_parameters_dicts[name] = tp_spec
        return type_parameters_dicts

    valid_params_list = []

    dipole_arg_dict = {
        'params': ({
            'A': [0.5, 1.5, 3.47],
            'kappa': [4., 1.2, 0.3]
        }, 2),
        'mu': ([(1.0, 0, 0), (0.5, 0, 0)], 1)
    }

    valid_params_list.append(
        make_aniso_spec(
            md.pair.aniso.Dipole,
            to_type_parameter_dicts(particle_types, dipole_arg_dict)))

    gay_berne_arg_dict = {
        'params': ({
            'epsilon': [0.5, 0.25, 0.1],
            'lperp': [0.5, 0.45, 0.3],
            'lpar': [.7, 0.2, 0.375]
        }, 2)
    }

    valid_params_list.append(
        make_aniso_spec(
            md.pair.aniso.GayBerne,
            to_type_parameter_dicts(particle_types, gay_berne_arg_dict)))

    alj_arg_dict0 = {
        'params': ({
            'epsilon': [0.5, 1.1, 0.147],
            'sigma_i': [0.4, 0.12, 0.3],
            'sigma_j': [4., 1.2, 0.3],
            'alpha': [0, 1, 3],
            'contact_ratio_i': [0.15, 0.3, 0.145],
            'contact_ratio_j': [0.15, 0.3, 0.145],
            'average_simplices': [True, False, True]
        }, 2),
        'shape': ({
            "vertices": [[], []],
            "rounding_radii": [(0.1, 0.2, 0.15), (0.3, 0.3, 0.3)],
            "faces": [[], []]
        }, 1)
    }

    valid_params_list.append(
        make_aniso_spec(md.pair.aniso.ALJ,
                        to_type_parameter_dicts(particle_types, alj_arg_dict0)))

    shape_vertices = [
        # octahedron
        [(0.5, 0, 0), (-0.5, 0, 0), (0, 0.5, 0), (0, -0.5, 0), (0, 0, 0.5),
         (0, 0, -0.5)],
        # cube
        [(0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5),
         (-0.5, 0.5, -0.5), (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5),
         (-0.5, -0.5, -0.5)],
    ]
    shape_faces = [
        # octahedron
        [[5, 3, 1], [0, 3, 5], [1, 3, 4], [4, 3, 0], [5, 2, 0], [1, 2, 5],
         [0, 2, 4], [4, 2, 1]],
        # cube
        [[4, 3, 2, 1], [0, 1, 2, 6], [2, 3, 5, 6], [7, 4, 1, 0], [6, 5, 7, 0],
         [3, 4, 7, 5]]
    ]

    alj_arg_dict1 = {
        'params': ({
            'epsilon': [0.5, 1.1, 0.147],
            'sigma_i': [0.4, 0.12, 0.3],
            'sigma_j': [4., 1.2, 0.3],
            'alpha': [0, 1, 3],
            'contact_ratio_i': [0.15, 0.3, 0.145],
            'contact_ratio_j': [0.15, 0.3, 0.145],
            'average_simplices': [True, False, True]
        }, 2),
        'shape': ({
            "vertices": shape_vertices,
            "rounding_radii": [(0.1, 0.01, 0.15), (0.0, 0.0, 0.0)],
            "faces": shape_faces
        }, 1)
    }

    valid_params_list.append(
        make_aniso_spec(md.pair.aniso.ALJ,
                        to_type_parameter_dicts(particle_types, alj_arg_dict1)))

    return valid_params_list


class PotentialId:

    def __init__(self):
        self.cls_dict = {}

    def __call__(self, obj):
        self.cls_dict.setdefault(obj.cls, 0)
        self.cls_dict[obj.cls] += 1
        return f"{obj.cls.__name__}-{self.cls_dict[obj.cls]}"


@pytest.mark.parametrize('pair_potential_spec',
                         _valid_params(),
                         ids=PotentialId())
def test_setting_params_and_shape(make_two_particle_simulation,
                                  pair_potential_spec):
    pair_potential = pair_potential_spec.cls(nlist=md.nlist.Cell(buffer=0.4),
                                             default_r_cut=2.5)
    for key, value in pair_potential_spec.type_parameters.items():
        setattr(pair_potential, key, value)
        assert_equivalent_data_structures(value, getattr(pair_potential, key))

    sim = make_two_particle_simulation(types=['A', 'B'],
                                       dimensions=3,
                                       d=0.5,
                                       force=pair_potential)
    sim.run(0)
    for key, value in pair_potential_spec.type_parameters.items():
        assert_equivalent_data_structures(value, getattr(pair_potential, key))


def _aniso_forces_and_energies():
    """Return reference force and energy values.

    Reference force and energy values were calculated using HOOMD-blue v3 beta
    1.  Values were calculated at distances of 0.75 and 1.5 for each argument
    dictionary as well as second particle orientations of
    [0.86615809, 0.4997701, 0, 0] and [0.70738827, 0, 0, 0.70682518]. The first
    particle is always oriented [1, 0, 0, 0].
    """
    # holds the forces, energies, and torques associated with an anisotropic
    # pair potential.
    FETtuple = namedtuple('FETtuple', [
        'pair_potential', 'pair_potential_params', 'forces', 'energies',
        'torques'
    ])

    path = Path(__file__).parent / "aniso_forces_and_energies.json"
    with path.open() as f:
        computations = json.load(f)
        fet_list = []
        for pot in computations:
            for i, params in enumerate(expand_dict(
                    computations[pot]["params"])):
                fet_list.append(
                    FETtuple(
                        getattr(md.pair.aniso, pot),
                        params,
                        computations[pot]["forces"][i],
                        computations[pot]["energies"][i],
                        computations[pot]["torques"][i],
                    ))
    return fet_list


@pytest.fixture(scope="function", params=_valid_params(), ids=PotentialId())
def pair_potential(request):
    spec = request.param
    pair_potential = spec.cls(nlist=md.nlist.Cell(buffer=0.4),
                              default_r_cut=2.5)
    for key, value in spec.type_parameters.items():
        setattr(pair_potential, key, value)
    return pair_potential


def test_run(simulation_factory, lattice_snapshot_factory, pair_potential):
    snap = lattice_snapshot_factory(particle_types=['A', 'B'],
                                    n=7,
                                    a=2.0,
                                    r=0.01)
    if snap.communicator.rank == 0:
        snap.particles.typeid[:] = np.random.randint(0,
                                                     len(snap.particles.types),
                                                     snap.particles.N)
    sim = simulation_factory(snap)
    integrator = md.Integrator(dt=0.005, integrate_rotational_dof=True)
    integrator.forces.append(pair_potential)
    integrator.methods.append(md.methods.Langevin(hoomd.filter.All(), kT=1))
    sim.operations.integrator = integrator
    old_snap = sim.state.get_snapshot()
    sim.run(5)
    new_snap = sim.state.get_snapshot()
    forces = pair_potential.forces
    energies = pair_potential.energies
    if new_snap.communicator.rank == 0:
        assert not np.allclose(new_snap.particles.position,
                               old_snap.particles.position)
        assert np.any(energies != 0)
        assert np.any(forces != 0)

    autotuned_kernel_parameter_check(instance=pair_potential,
                                     activate=lambda: sim.run(1))


@pytest.mark.parametrize("aniso_forces_and_energies",
                         _aniso_forces_and_energies(),
                         ids=lambda x: x.pair_potential.__name__)
def test_aniso_force_computes(make_two_particle_simulation,
                              aniso_forces_and_energies):
    r"""These are pure regression tests from HOOMD-blue version 3.0 beta 1.

    This tests 2 conditions with three parameter values for each pair potential.
    The particle distances and orientations are:

    .. math::

        r_1 = (0, 0, 0.1) \ r_2 = (0, 0, 0.85) \\
        \theta_1 = (1, 0, 0, 0) \ \theta_2 = (0.86615809, 0.4997701, 0, 0) \\
        \\
        r_1 = (0, 0, 0.1) \ r_2 = (0, 0, 1.6) \\
        \theta_1 = (1, 0, 0, 0) \ \theta_2 = (0.70738827, 0, 0, 0.70682518) \\

    """
    pot = aniso_forces_and_energies.pair_potential(
        nlist=md.nlist.Cell(buffer=0.4), default_r_cut=2.5)
    for param, value in aniso_forces_and_energies.pair_potential_params.items():
        getattr(pot, param)[('A', 'A')] = value
    sim = make_two_particle_simulation(types=['A'], d=0.75, force=pot)
    sim.run(0)
    particle_distances = [0.75, 1.5]
    orientations = [[0.86615809, 0.4997701, 0.0, 0.0],
                    [0.70738827, 0.0, 0.0, 0.70682518]]
    for i, (distance,
            orientation) in enumerate(zip(particle_distances, orientations)):
        snap = sim.state.get_snapshot()
        # Set up proper distances and orientations
        if snap.communicator.rank == 0:
            snap.particles.position[0] = [0, 0, .1]
            snap.particles.position[1] = [0, 0, distance + .1]
            snap.particles.orientation[1] = orientation
        sim.state.set_snapshot(snap)

        # Grab all quantities to test for accuracy
        sim_energies = sim.operations.integrator.forces[0].energies
        sim_forces = sim.operations.integrator.forces[0].forces
        sim_torques = sim.operations.integrator.forces[0].torques
        # Compare the gathered quantities for the potential
        if sim_energies is not None:
            assert isclose(sim_energies[0],
                           aniso_forces_and_energies.energies[i])
            assert isclose(sim_forces[0], aniso_forces_and_energies.forces[i])
            assert isclose(-sim_forces[1], aniso_forces_and_energies.forces[i])
            assert isclose(sim_torques, aniso_forces_and_energies.torques[i])


@pytest.mark.parametrize('pair_potential_spec',
                         _valid_params(),
                         ids=PotentialId())
def test_pickling(make_two_particle_simulation, pair_potential_spec):
    pair_potential = pair_potential_spec.cls(nlist=md.nlist.Cell(buffer=0.4),
                                             default_r_cut=2.5)
    for key, value in pair_potential_spec.type_parameters.items():
        setattr(pair_potential, key, value)

    sim = make_two_particle_simulation(types=['A', 'B'],
                                       dimensions=3,
                                       d=0.5,
                                       force=pair_potential)
    pickling_check(pair_potential)
    sim.run(0)
    pickling_check(pair_potential)


def _base_expected_loggable(include_type_shapes=False):
    base = {
        "forces": {
            "category": hoomd.logging.LoggerCategories["particle"],
            "default": True
        },
        "torques": {
            "category": hoomd.logging.LoggerCategories["particle"],
            "default": True
        },
        "virials": {
            "category": hoomd.logging.LoggerCategories["particle"],
            "default": True
        },
        "energies": {
            "category": hoomd.logging.LoggerCategories["particle"],
            "default": True
        },
        "energy": {
            "category": hoomd.logging.LoggerCategories["scalar"],
            "default": True
        }
    }
    if include_type_shapes:
        base["type_shapes"] = {
            'category': LoggerCategories.object,
            'default': True
        }
    return base


@pytest.mark.parametrize(
    "cls,log_check_params",
    ((cls, log_check_params) for cls, log_check_params in zip((
        md.pair.aniso.GayBerne, md.pair.aniso.Dipole,
        md.pair.aniso.ALJ), (_base_expected_loggable(True),
                             _base_expected_loggable(),
                             _base_expected_loggable(True)))))
def test_logging(cls, log_check_params):
    logging_check(cls, ('md', 'pair', 'aniso'), log_check_params)
