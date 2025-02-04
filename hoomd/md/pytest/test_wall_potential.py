# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import numpy as np
import pytest

import hoomd
import hoomd.md as md
from hoomd.conftest import expected_loggable_params
from hoomd.conftest import (logging_check, pickling_check,
                            autotuned_kernel_parameter_check)

import itertools


@pytest.fixture
def simulation(simulation_factory, two_particle_snapshot_factory):
    sim = simulation_factory(two_particle_snapshot_factory())
    integrator = md.Integrator(0.005)
    integrator.methods.append(md.methods.ConstantVolume(hoomd.filter.All()))
    sim.operations += integrator
    sim.state.thermalize_particle_momenta(hoomd.filter.All(), kT=1.0)
    return sim


class WallGenerator:
    rng = np.random.default_rng(1264556)
    scale = 1e3

    @classmethod
    def generate(cls, types=("Sphere", "Cylinder", "Plane")):
        type = cls.rng.choice(types)
        origin = (cls.float(), cls.float(), cls.float())
        inside = cls.rng.choice((True, False))
        if type == "Sphere":
            return hoomd.wall.Sphere(radius=cls.float(),
                                     origin=origin,
                                     inside=inside)
        if type == "Cylinder":
            return hoomd.wall.Cylinder(
                radius=cls.float(),
                origin=origin,
                axis=(cls.float(), cls.float(), cls.float()),
                inside=inside,
            )
        if type == "Plane":
            normal = np.array((cls.float(), cls.float(), cls.float()))
            vector_norm = np.linalg.norm(normal)
            if vector_norm == 0:
                assert "Generated invalid normal."
            normal /= vector_norm
            return hoomd.wall.Plane(origin=origin, normal=normal)

    @classmethod
    def float(cls):
        return cls.rng.random() * cls.scale

    @classmethod
    def generate_n(cls, N):
        for _ in range(N):
            yield cls.generate()


_potential_cls = (
    md.external.wall.LJ,
    md.external.wall.Gaussian,
    md.external.wall.Yukawa,
    md.external.wall.Morse,
    md.external.wall.ForceShiftedLJ,
    md.external.wall.Mie,
)


@pytest.mark.parametrize("cls", _potential_cls)
def test_construction(cls):
    walls = [w for w in WallGenerator.generate_n(10)]
    wall_pot = cls(walls)
    assert all(a is b for a, b in zip(walls, wall_pot.walls))
    assert len(wall_pot.walls) == 10


@pytest.mark.parametrize("wall_pot", (cls([]) for cls in _potential_cls))
def test_wall_setting(wall_pot):
    walls = [w for w in WallGenerator.generate_n(10)]
    wall_pot.walls = walls
    assert all(a is b for a, b in zip(walls, wall_pot.walls))
    assert len(wall_pot.walls) == 10
    walls = [w for w in WallGenerator.generate_n(5)]
    wall_pot.walls = walls
    assert all(a is b for a, b in zip(walls, wall_pot.walls))
    assert len(wall_pot.walls) == 5


def _params(r_cut=None, r_extrap=None):
    """Generate compatible potential parameters for _potential_cls.

    Sets a random ``r_cut`` and ``r_extrap`` if not specified.

    Args:
        r_cut (float, optional): Provide an explicit r_cut for all parameters.
        r_extrap (float, optional): Provide an explicit r_extrap for all
            parameters.
    """
    base = (
        {
            "sigma": 1.0,
            "epsilon": 1.0
        },
        {
            "sigma": 1.0,
            "epsilon": 5.5
        },
        {
            "kappa": 1.0,
            "epsilon": 1.5
        },
        {
            "r0": 1.0,
            "D0": 1.0,
            "alpha": 1.0
        },
        {
            "sigma": 1.0,
            "epsilon": 1.0
        },
        {
            "sigma": 1.0,
            "epsilon": 1.0,
            "m": 10,
            "n": 20
        },
    )
    for p in base:
        if r_cut is None:
            p["r_cut"] = _params.rng.random() * 5
        else:
            p["r_cut"] = r_cut
        if r_extrap is None:
            p["r_extrap"] = _params.rng.random() * 5
        else:
            p["r_extrap"] = r_extrap
    return base


_params.rng = np.random.default_rng(26456)


@pytest.mark.parametrize("cls, params", zip(_potential_cls, _params()))
def test_potential_params(cls, params):
    wall_pot = cls(WallGenerator.generate_n(2))
    wall_pot.params["A"] = params
    for attr in params:
        assert wall_pot.params["A"][attr] == params[attr]

    wall_pot.params.default = params
    for attr in params:
        assert wall_pot.params["foo"][attr] == params[attr]


@pytest.mark.parametrize("cls, params", zip(_potential_cls, _params()))
def test_attaching(simulation, cls, params):
    wall_pot = cls(WallGenerator.generate_n(2))
    simulation.operations.integrator.forces.append(wall_pot)
    with pytest.raises(hoomd.error.IncompleteSpecificationError):
        simulation.run(0)
    wall_pot.params["A"] = params
    simulation.run(0)
    for attr in params:
        assert np.isclose(wall_pot.params["A"][attr], params[attr])


@pytest.mark.parametrize("cls, params", zip(_potential_cls, _params(2.5, 0.0)))
def test_plane(simulation, cls, params):
    """Test that particles stay in box slice defined by two plane walls."""
    wall_pot = cls([
        hoomd.wall.Plane(normal=(0, 0, -1), origin=(0, 0, 1)),
        hoomd.wall.Plane(normal=(0, 0, 1), origin=(0, 0, -1)),
    ])
    simulation.operations.integrator.forces.append(wall_pot)
    wall_pot.params["A"] = params
    for _ in range(10):
        simulation.run(10)
        with simulation.state.cpu_local_snapshot as snap:
            assert np.all(snap.particles.position[:, 2] < 1)
            assert np.all(snap.particles.position[:, 2] > -1)


@pytest.mark.parametrize("cls, params", zip(_potential_cls, _params(2.5, 0.0)))
def test_sphere(simulation, cls, params):
    """Test that particles stay within a sphere wall."""
    radius = 5
    wall_pot = cls(
        [hoomd.wall.Sphere(radius=radius, origin=(0, 0, 0), inside=True)])
    simulation.operations.integrator.forces.append(wall_pot)
    wall_pot.params["A"] = params
    for _ in range(10):
        simulation.run(10)
        with simulation.state.cpu_local_snapshot as snap:
            assert np.all(
                np.linalg.norm(snap.particles.position, axis=1) < radius)


@pytest.mark.parametrize("cls, params", zip(_potential_cls, _params(2.5, 0.0)))
def test_cylinder(simulation, cls, params):
    """Test that particles stay within the pipe defined by a cylinder wall."""
    n = np.array([1, 1, 1])
    radius = 5
    wall_pot = cls([
        hoomd.wall.Cylinder(radius=radius,
                            origin=(0, 0, 0),
                            axis=n,
                            inside=True)
    ])
    simulation.operations.integrator.forces.append(wall_pot)
    wall_pot.params["A"] = params
    for _ in range(10):
        simulation.run(10)
        with simulation.state.cpu_local_snapshot as snap:
            for i in range(len(snap.particles.position)):
                r = snap.particles.position[i]
                assert np.linalg.norm(r - (np.dot(r, n) * n)) < radius


@pytest.mark.parametrize("cls, params", zip(_potential_cls, _params(2.5, 0.0)))
def test_outside(simulation, cls, params):
    """Test that particles stay outside a sphere wall when inside=False."""
    radius = 5.0
    wall_pot = cls(
        [hoomd.wall.Sphere(radius=radius, origin=(0, 0, 0), inside=False)])
    simulation.operations.integrator.forces.append(wall_pot)
    wall_pot.params["A"] = params
    snap = simulation.state.get_snapshot()
    if simulation.device.communicator.rank == 0:
        snap.particles.position[:] = [[0, 0, 6.5], [0, 0, -6.5]]
    simulation.state.set_snapshot(snap)

    for _ in range(10):
        simulation.run(50)
        with simulation.state.cpu_local_snapshot as snap:
            assert np.all(
                np.linalg.norm(snap.particles.position, axis=1) > radius)


@pytest.mark.parametrize("cls, params", zip(_potential_cls, _params(2.5, 1.1)))
def test_r_extrap(simulation, cls, params):
    """Test a force is generated in the other half space with r_extrap set."""
    radius = 5.0
    wall_pot = cls(
        [hoomd.wall.Sphere(radius=radius, origin=(0, 0, 0), inside=False)])
    simulation.operations.integrator.forces.append(wall_pot)
    wall_pot.params["A"] = params
    snap = simulation.state.get_snapshot()
    if simulation.device.communicator.rank == 0:
        snap.particles.position[:] = [[0, 0, 4.8], [0, 0, -4.8]]
    simulation.state.set_snapshot(snap)

    simulation.run(0)
    energies = wall_pot.energy
    forces = wall_pot.forces
    if simulation.device.communicator.rank == 0:
        assert np.all(energies != 0)
        assert np.all(np.any(forces != 0, axis=1))


# Test Logging
@pytest.mark.parametrize(
    'cls, expected_namespace, expected_loggables',
    zip(_potential_cls, itertools.repeat(('md', 'external', 'wall')),
        itertools.repeat(expected_loggable_params)))
def test_logging(cls, expected_namespace, expected_loggables):
    logging_check(cls, expected_namespace, expected_loggables)


@pytest.mark.parametrize("cls, params", zip(_potential_cls, _params()))
def test_kernel_parameters(simulation, cls, params):
    wall_pot = cls(WallGenerator.generate_n(2))
    simulation.operations.integrator.forces.append(wall_pot)
    wall_pot.params["A"] = params

    simulation.run(0)

    autotuned_kernel_parameter_check(instance=wall_pot,
                                     activate=lambda: simulation.run(1))


# Pickle Testing
@pytest.mark.parametrize("cls, params", zip(_potential_cls, _params()))
def test_pickling(simulation, cls, params):
    """Test pickling on a small simulation."""
    wall_pot = cls(WallGenerator.generate_n(2))
    simulation.operations.integrator.forces.append(wall_pot)
    wall_pot.params["A"] = params

    pickling_check(wall_pot)
    simulation.run(0)
    pickling_check(wall_pot)
