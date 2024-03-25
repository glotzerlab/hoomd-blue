# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

from hoomd.snapshot import Snapshot
import hoomd
import numpy
import pytest


@pytest.fixture(scope='function')
def snap(device):
    s = Snapshot(device.communicator)
    N = 1000

    if s.communicator.rank == 0:
        s.configuration.box = [20, 20, 20, 0, 0, 0]

        s.particles.N = N
        s.particles.position[:] = numpy.random.uniform(-10, 10, size=(N, 3))
        s.particles.velocity[:] = numpy.random.uniform(-1, 1, size=(N, 3))
        s.particles.typeid[:] = numpy.random.randint(0, 3, size=N)
        s.particles.mass[:] = numpy.random.uniform(1, 2, size=N)
        s.particles.charge[:] = numpy.random.uniform(-2, 2, size=N)
        s.particles.image[:] = numpy.random.randint(-8, 8, size=(N, 3))
        s.particles.orientation[:] = numpy.random.uniform(-1, 1, size=(N, 4))
        s.particles.moment_inertia[:] = numpy.random.uniform(1, 5, size=(N, 3))
        s.particles.angmom[:] = numpy.random.uniform(-1, 1, size=(N, 4))
        s.particles.types = ['A', 'B', 'C', 'D']

        s.bonds.N = N - 1
        for i in range(s.bonds.N):
            s.bonds.group[i, :] = [i, i + 1]

        s.bonds.typeid[:] = numpy.random.randint(0, 3, size=s.bonds.N)
        s.bonds.types = ['bondA', 'bondB', 'bondC', 'bondD']

        s.angles.N = N - 2
        for i in range(s.angles.N):
            s.angles.group[i, :] = [i, i + 1, i + 2]

        s.angles.typeid[:] = numpy.random.randint(0, 3, size=s.angles.N)
        s.angles.types = ['angleA', 'angleB', 'angleC', 'angleD']

        s.dihedrals.N = N - 3
        for i in range(s.dihedrals.N):
            s.dihedrals.group[i, :] = [i, i + 1, i + 2, i + 3]

        s.dihedrals.typeid[:] = numpy.random.randint(0, 3, size=s.dihedrals.N)
        s.dihedrals.types = ['dihedralA', 'dihedralB', 'dihedralC', 'dihedralD']

        s.impropers.N = N - 3
        for i in range(s.impropers.N):
            s.impropers.group[i, :] = [i, i + 1, i + 2, i + 3]

        s.impropers.typeid[:] = numpy.random.randint(0, 3, size=s.impropers.N)
        s.impropers.types = ['improperA', 'improperB', 'improperC', 'improperD']

        s.pairs.N = N - 1
        for i in range(s.pairs.N):
            s.pairs.group[i, :] = [i, i + 1]

        s.pairs.typeid[:] = numpy.random.randint(0, 3, size=s.pairs.N)
        s.pairs.types = ['pairA', 'pairB', 'pairC', 'pairD']

        s.constraints.N = N - 1
        for i in range(s.constraints.N):
            s.constraints.group[i, :] = [i, i + 1]

        s.constraints.value[:] = numpy.random.uniform(1,
                                                      10,
                                                      size=s.constraints.N)

    return s


def assert_snapshots_equal(s1, s2):
    if s1.communicator.rank == 0:
        numpy.testing.assert_allclose(s1.configuration.box,
                                      s2.configuration.box)
        numpy.testing.assert_allclose(s1.configuration.dimensions,
                                      s2.configuration.dimensions)

        assert s1.particles.N == s2.particles.N
        assert s1.particles.types == s2.particles.types
        numpy.testing.assert_allclose(s1.particles.position,
                                      s2.particles.position)
        numpy.testing.assert_allclose(s1.particles.velocity,
                                      s2.particles.velocity)
        numpy.testing.assert_allclose(s1.particles.acceleration,
                                      s2.particles.acceleration)
        numpy.testing.assert_equal(s1.particles.typeid, s2.particles.typeid)
        numpy.testing.assert_allclose(s1.particles.mass, s2.particles.mass)
        numpy.testing.assert_allclose(s1.particles.charge, s2.particles.charge)
        numpy.testing.assert_allclose(s1.particles.diameter,
                                      s2.particles.diameter)
        numpy.testing.assert_equal(s1.particles.image, s2.particles.image)
        numpy.testing.assert_equal(s1.particles.body, s2.particles.body)
        numpy.testing.assert_allclose(s1.particles.orientation,
                                      s2.particles.orientation)
        numpy.testing.assert_allclose(s1.particles.moment_inertia,
                                      s2.particles.moment_inertia)
        numpy.testing.assert_allclose(s1.particles.angmom, s2.particles.angmom)
        numpy.testing.assert_allclose(s1.particles.diameter,
                                      s2.particles.diameter)

        assert s1.bonds.N == s2.bonds.N
        assert s1.bonds.types == s2.bonds.types
        numpy.testing.assert_equal(s1.bonds.typeid, s2.bonds.typeid)
        numpy.testing.assert_equal(s1.bonds.group, s2.bonds.group)

        assert s1.angles.N == s2.angles.N
        assert s1.angles.types == s2.angles.types
        numpy.testing.assert_equal(s1.angles.typeid, s2.angles.typeid)
        numpy.testing.assert_equal(s1.angles.group, s2.angles.group)

        assert s1.dihedrals.N == s2.dihedrals.N
        assert s1.dihedrals.types == s2.dihedrals.types
        numpy.testing.assert_equal(s1.dihedrals.typeid, s2.dihedrals.typeid)
        numpy.testing.assert_equal(s1.dihedrals.group, s2.dihedrals.group)

        assert s1.impropers.N == s2.impropers.N
        assert s1.impropers.types == s2.impropers.types
        numpy.testing.assert_equal(s1.impropers.typeid, s2.impropers.typeid)
        numpy.testing.assert_equal(s1.impropers.group, s2.impropers.group)

        assert s1.pairs.N == s2.pairs.N
        assert s1.pairs.types == s2.pairs.types
        numpy.testing.assert_equal(s1.pairs.typeid, s2.pairs.typeid)
        numpy.testing.assert_equal(s1.pairs.group, s2.pairs.group)

        assert s1.constraints.N == s2.constraints.N
        numpy.testing.assert_allclose(s1.constraints.value,
                                      s2.constraints.value)
        numpy.testing.assert_equal(s1.constraints.group, s2.constraints.group)


def test_create_from_snapshot(simulation_factory, snap):
    sim = simulation_factory(snap)

    if snap.communicator.rank == 0:
        assert sim.state.particle_types == snap.particles.types
        assert sim.state.bond_types == snap.bonds.types
        assert sim.state.angle_types == snap.angles.types
        assert sim.state.dihedral_types == snap.dihedrals.types
        assert sim.state.improper_types == snap.impropers.types
        assert sim.state.special_pair_types == snap.pairs.types
        # TODO: test box, dimensions
        # assert
        # TODO: what other state properties should be accessible and valid here?


def test_get_snapshot(simulation_factory, snap):
    sim = simulation_factory()
    sim.create_state_from_snapshot(snap)

    snap2 = sim.state.get_snapshot()
    assert_snapshots_equal(snap, snap2)


def test_modify_snapshot(simulation_factory, snap):
    sim = simulation_factory()
    sim.create_state_from_snapshot(snap)

    if snap.communicator.rank == 0:
        snap.particles.N = snap.particles.N // 2
        snap.bonds.N = snap.bonds.N // 4
        snap.angles.N = snap.angles.N // 4
        snap.dihedrals.N = snap.dihedrals.N // 4
        snap.impropers.N = snap.impropers.N // 4
        snap.pairs.N = snap.pairs.N // 4
        snap.constraints.N = snap.constraints.N // 4

    sim.state.set_snapshot(snap)

    snap2 = sim.state.get_snapshot()
    assert_snapshots_equal(snap, snap2)


def test_thermalize_particle_velocity(simulation_factory,
                                      lattice_snapshot_factory):
    snap = lattice_snapshot_factory()
    sim = simulation_factory(snap)
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.5)

    snapshot = sim.state.get_snapshot()
    if snapshot.communicator.rank == 0:
        v = snapshot.particles.velocity[:]
        m = snapshot.particles.mass[:]
        p = m * v.T
        p_com = numpy.mean(p, axis=1)

        numpy.testing.assert_allclose(p_com, [0, 0, 0], atol=1e-14)

        K = numpy.sum(1 / 2 * m * (v[:, 0]**2 + v[:, 1]**2 + v[:, 2]**2))
        # check that K is somewhat close to the target - the fluctuations are
        # too large for an allclose check.
        expected_K = (3 * snap.particles.N - 3) / 2 * 1.5
        assert K > expected_K * 3 / 4 and K < expected_K * 4 / 3


def test_thermalize_angular_momentum(simulation_factory,
                                     lattice_snapshot_factory):
    snap = lattice_snapshot_factory()
    I = [1, 2, 3]  # noqa: E741 - allow ambiguous variable name

    if snap.communicator.rank == 0:
        snap.particles.moment_inertia[:] = I

    sim = simulation_factory(snap)
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.5)

    snapshot = sim.state.get_snapshot()
    if snapshot.communicator.rank == 0:
        # Note: this conversion assumes that all particles have (1, 0, 0, 0)
        # orientations.
        L = snapshot.particles.angmom[:, 1:4] / 2

        K = numpy.sum(
            1 / 2 * (L[:, 0]**2 / I[0] + L[:, 1]**2 / I[1] + L[:, 2]**2 / I[2]))
        # check that K is somewhat close to the target - the fluctuations are
        # too large for an allclose check.
        expected_K = (3 * snap.particles.N) / 2 * 1.5
        assert K > expected_K * 3 / 4 and K < expected_K * 4 / 3


def test_zero_particle_velocity_angmom():
    snapshot = hoomd.Snapshot()
    snapshot.configuration.box = (10, 10, 10, 0, 0, 0)
    if snapshot.communicator.rank == 0:
        snapshot.particles.N = 4
        snapshot.particles.types = ['A']
        snapshot.particles.body[:] = [0, 0, 2, 2]
        snapshot.particles.moment_inertia[:] = [[1, 1, 1]] * 4

    sim = hoomd.Simulation(device=hoomd.device.CPU())
    sim.create_state_from_snapshot(snapshot)
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.0)
    thermalized_snapshot = sim.state.get_snapshot()

    if snapshot.communicator.rank == 0:
        numpy.testing.assert_allclose(
            thermalized_snapshot.particles.velocity[1], [0, 0, 0])
        numpy.testing.assert_allclose(
            thermalized_snapshot.particles.velocity[3], [0, 0, 0])
        numpy.testing.assert_allclose(thermalized_snapshot.particles.angmom[1],
                                      [0, 0, 0, 0])
        numpy.testing.assert_allclose(thermalized_snapshot.particles.angmom[3],
                                      [0, 0, 0, 0])


def test_replicate(simulation_factory, lattice_snapshot_factory):
    initial_snapshot = lattice_snapshot_factory(a=10, n=1)

    sim = simulation_factory(initial_snapshot)

    initial_snapshot.replicate(2, 2, 2)
    if initial_snapshot.communicator.rank == 0:
        numpy.testing.assert_allclose(initial_snapshot.particles.position, [
            [-5, -5, -5],
            [-5, -5, 5],
            [-5, 5, -5],
            [-5, 5, 5],
            [5, -5, -5],
            [5, -5, 5],
            [5, 5, -5],
            [5, 5, 5],
        ])

    sim.state.replicate(2, 2, 2)
    new_snapshot = sim.state.get_snapshot()
    assert_snapshots_equal(initial_snapshot, new_snapshot)


def test_domain_decomposition(device, simulation_factory,
                              lattice_snapshot_factory):
    snapshot = lattice_snapshot_factory()

    if device.communicator.num_ranks == 1:
        sim = simulation_factory(snapshot)
        assert sim.state.domain_decomposition == (1, 1, 1)
        assert sim.state.domain_decomposition_split_fractions == ([], [], [])
    elif device.communicator.num_ranks == 2:
        sim = simulation_factory(snapshot)
        assert sim.state.domain_decomposition == (1, 1, 2)
        assert sim.state.domain_decomposition_split_fractions == ([], [], [0.5])

        sim = simulation_factory(snapshot, domain_decomposition=(None, 1, 1))
        assert sim.state.domain_decomposition == (2, 1, 1)
        assert sim.state.domain_decomposition_split_fractions == ([0.5], [], [])

        sim = simulation_factory(snapshot, domain_decomposition=(2, 1, 1))
        assert sim.state.domain_decomposition == (2, 1, 1)
        assert sim.state.domain_decomposition_split_fractions == ([0.5], [], [])

        sim = simulation_factory(snapshot, domain_decomposition=(1, None, 1))
        assert sim.state.domain_decomposition == (1, 2, 1)
        assert sim.state.domain_decomposition_split_fractions == ([], [0.5], [])

        sim = simulation_factory(snapshot, domain_decomposition=(1, 2, 1))
        assert sim.state.domain_decomposition == (1, 2, 1)
        assert sim.state.domain_decomposition_split_fractions == ([], [0.5], [])

        sim = simulation_factory(snapshot,
                                 domain_decomposition=(None, None, [0.25,
                                                                    0.75]))
        assert sim.state.domain_decomposition == (1, 1, 2)
        assert sim.state.domain_decomposition_split_fractions == ([], [],
                                                                  [0.25])
    else:
        raise RuntimeError("Test only supports 1 and 2 ranks")
