import numpy as np
import pytest

import hoomd
import hoomd.md as md


@pytest.fixture
def simulation(simulation_factory, two_particle_snapshot_factory):
    sim = simulation_factory(two_particle_snapshot_factory())
    integrator = md.Integrator(0.005)
    integrator.methods.append(md.methods.NVE(hoomd.filter.All()))
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
            return hoomd.wall.Plane(origin=origin, normal=normal, inside=inside)

    @classmethod
    def float(cls):
        return cls.rng.random() * cls.scale

    @classmethod
    def generate_n(cls, N):
        for _ in range(N):
            yield cls.generate()


_potential_cls = (
    md.external.wall.LJ,
    md.external.wall.Gauss,
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


@pytest.mark.parametrize("cls", _potential_cls)
def test_wall_setting(cls):
    wall_pot = cls()
    walls = [w for w in WallGenerator.generate_n(10)]
    wall_pot.walls = walls
    assert all(a is b for a, b in zip(walls, wall_pot.walls))
    assert len(wall_pot.walls) == 10
    walls = [w for w in WallGenerator.generate_n(5)]
    wall_pot.walls = walls
    assert all(a is b for a, b in zip(walls, wall_pot.walls))
    assert len(wall_pot.walls) == 5


def _params(r_cut=None, r_extrap=None):
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
def test_params(cls, params):
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
    wall_pot = cls([
        hoomd.wall.Plane(normal=(0, 0, -1), origin=(0, 0, 1), inside=True),
        hoomd.wall.Plane(normal=(0, 0, 1), origin=(0, 0, -1), inside=True),
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
    radius = 5
    wall_pot = cls([
        hoomd.wall.Cylinder(radius=radius,
                            origin=(0, 0, 0),
                            axis=(0, 0, 1),
                            inside=True)
    ])
    simulation.operations.integrator.forces.append(wall_pot)
    wall_pot.params["A"] = params
    for _ in range(10):
        simulation.run(10)
        with simulation.state.cpu_local_snapshot as snap:
            assert np.all(
                np.linalg.norm(snap.particles.position[:, :2], axis=1) < radius)


@pytest.mark.parametrize("cls, params", zip(_potential_cls, _params(2.5, 0.0)))
def test_outside(simulation, cls, params):
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
    assert np.all(wall_pot.energy != 0)
    assert np.all(np.any(wall_pot.forces != 0, axis=1))
