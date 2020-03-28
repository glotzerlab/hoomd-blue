from hoomd.snapshot import Snapshot
from hoomd.simulation import Simulation
import numpy
import pytest


@pytest.fixture(scope='function')
def snap(device):
    s = Snapshot(device.comm)
    N = 1000

    if s.exists:
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

        s.bonds.N = N-1
        for i in range(s.bonds.N):
            s.bonds.group[i, :] = [i, i+1]

        s.bonds.typeid[:] = numpy.random.randint(0, 3, size=s.bonds.N)
        s.bonds.types = ['bondA', 'bondB', 'bondC', 'bondD']

        s.angles.N = N-2
        for i in range(s.angles.N):
            s.angles.group[i, :] = [i, i+1, i+2]

        s.angles.typeid[:] = numpy.random.randint(0, 3, size=s.angles.N)
        s.angles.types = ['angleA', 'angleB', 'angleC', 'angleD']

        s.dihedrals.N = N-3
        for i in range(s.dihedrals.N):
            s.dihedrals.group[i, :] = [i, i+1, i+2, i+3]

        s.dihedrals.typeid[:] = numpy.random.randint(0, 3, size=s.dihedrals.N)
        s.dihedrals.types = ['dihedralA', 'dihedralB', 'dihedralC', 'dihedralD']

        s.impropers.N = N-3
        for i in range(s.impropers.N):
            s.impropers.group[i, :] = [i, i+1, i+2, i+3]

        s.impropers.typeid[:] = numpy.random.randint(0, 3, size=s.impropers.N)
        s.impropers.types = ['improperA', 'improperB', 'improperC', 'improperD']

        s.pairs.N = N-1
        for i in range(s.pairs.N):
            s.pairs.group[i, :] = [i, i+1]

        s.pairs.typeid[:] = numpy.random.randint(0, 3, size=s.pairs.N)
        s.pairs.types = ['pairA', 'pairB', 'pairC', 'pairD']

        s.constraints.N = N-1
        for i in range(s.constraints.N):
            s.constraints.group[i, :] = [i, i+1]

        s.constraints.value[:] = numpy.random.uniform(1, 10, size=s.constraints.N)

    return s


def assert_snapshots_equal(s1, s2):
    if s1.exists:
        numpy.testing.assert_allclose(s1.configuration.box, s2.configuration.box)
        numpy.testing.assert_allclose(s1.configuration.dimensions, s2.configuration.dimensions)

        assert s1.particles.N == s2.particles.N
        assert s1.particles.types == s2.particles.types
        numpy.testing.assert_allclose(s1.particles.position, s2.particles.position)
        numpy.testing.assert_allclose(s1.particles.velocity, s2.particles.velocity)
        numpy.testing.assert_allclose(s1.particles.acceleration, s2.particles.acceleration)
        numpy.testing.assert_equal(s1.particles.typeid, s2.particles.typeid)
        numpy.testing.assert_allclose(s1.particles.mass, s2.particles.mass)
        numpy.testing.assert_allclose(s1.particles.charge, s2.particles.charge)
        numpy.testing.assert_allclose(s1.particles.diameter, s2.particles.diameter)
        numpy.testing.assert_equal(s1.particles.image, s2.particles.image)
        numpy.testing.assert_equal(s1.particles.body, s2.particles.body)
        numpy.testing.assert_allclose(s1.particles.orientation, s2.particles.orientation)
        numpy.testing.assert_allclose(s1.particles.moment_inertia, s2.particles.moment_inertia)
        numpy.testing.assert_allclose(s1.particles.angmom, s2.particles.angmom)
        numpy.testing.assert_allclose(s1.particles.diameter, s2.particles.diameter)

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
        numpy.testing.assert_allclose(s1.constraints.value, s2.constraints.value)
        numpy.testing.assert_equal(s1.constraints.group, s2.constraints.group)

def test_create_from_snapshot(device, snap):
    sim = Simulation(device)
    sim.create_state_from_snapshot(snap)

    if snap.exists:
        assert sim.state.particle_types == snap.particles.types
        assert sim.state.bond_types == snap.bonds.types
        assert sim.state.angle_types == snap.angles.types
        assert sim.state.dihedral_types == snap.dihedrals.types
        assert sim.state.improper_types == snap.impropers.types
        assert sim.state.special_pair_types == snap.pairs.types
        # TODO: test box, dimensions
        # assert
        # TODO: what other state properties should be accessible and valid here?

def test_get_snapshot(device, snap):
    sim = Simulation(device)
    sim.create_state_from_snapshot(snap)

    snap2 = sim.state.snapshot
    assert_snapshots_equal(snap, snap2)


def test_modify_snapshot(device, snap):
    sim = Simulation(device)
    sim.create_state_from_snapshot(snap)

    if snap.exists:
        snap.particles.N = snap.particles.N // 2
        snap.bonds.N = snap.bonds.N // 4
        snap.angles.N = snap.angles.N // 4
        snap.dihedrals.N = snap.dihedrals.N // 4
        snap.impropers.N = snap.impropers.N // 4
        snap.pairs.N = snap.pairs.N // 4
        snap.constraints.N = snap.constraints.N // 4

    sim.state.snapshot = snap

    snap2 = sim.state.snapshot
    assert_snapshots_equal(snap, snap2)
