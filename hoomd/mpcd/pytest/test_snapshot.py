# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import pytest

import hoomd
import numpy as np

# default testing configuration, used in setting values below
test_positions = np.array([[1, 2, 3], [-9, -6, -3], [7, 8, 9]])
test_velocities = np.array([[-1, -2, -3], [4, 5, 6], [-7, -8, -9]])
test_typeids = np.array([1, 0, 2])
test_types = ["M", "P", "H"]
test_mass = 1.5


@pytest.fixture
def snap():
    snap_ = hoomd.Snapshot()
    snap_.configuration.box = [20, 20, 20, 0, 0, 0]
    return snap_


def test_get_set_mpcd_data(snap):
    """Test basic manipulation of MPCD data in snapshhot."""
    if snap.communicator.rank == 0:
        assert snap.mpcd.N == 0
        assert len(snap.mpcd.position) == 0
        assert len(snap.mpcd.velocity) == 0
        assert len(snap.mpcd.typeid) == 0
        assert len(snap.mpcd.types) == 0
        assert snap.mpcd.mass == 1.0

        # resize and test default fillings
        snap.mpcd.N = 3
        assert snap.mpcd.N == 3
        assert len(snap.mpcd.position) == 3
        assert len(snap.mpcd.velocity) == 3
        assert len(snap.mpcd.typeid) == 3
        np.testing.assert_array_equal(snap.mpcd.position, np.zeros((3, 3)))
        np.testing.assert_array_equal(snap.mpcd.velocity, np.zeros((3, 3)))
        np.testing.assert_array_equal(snap.mpcd.typeid, np.zeros(3))

        # test setting values in the snapshot
        snap.mpcd.N = test_positions.shape[0]
        snap.mpcd.position[:] = tuple(test_positions)
        snap.mpcd.velocity[:] = list(test_velocities)
        snap.mpcd.typeid[:] = test_typeids
        snap.mpcd.types = test_types
        snap.mpcd.mass = test_mass
        np.testing.assert_array_equal(snap.mpcd.position, test_positions)
        np.testing.assert_array_equal(snap.mpcd.velocity, test_velocities)
        np.testing.assert_array_equal(snap.mpcd.typeid, test_typeids)
        np.testing.assert_array_equal(snap.mpcd.types, test_types)


def test_resize(snap):
    """Test that MPCD data resizes with snapshot."""
    if snap.communicator.rank == 0:
        # start the snapshot from one particle
        snap.mpcd.N = 1
        snap.mpcd.position[:] = test_positions[0]
        snap.mpcd.velocity[:] = test_velocities[0]
        snap.mpcd.typeid[:] = test_typeids[0]
        snap.mpcd.types = test_types
        np.testing.assert_array_equal(snap.mpcd.position, [test_positions[0]])
        np.testing.assert_array_equal(snap.mpcd.velocity, [test_velocities[0]])
        np.testing.assert_array_equal(snap.mpcd.typeid, [test_typeids[0]])

        # grow the snapshot by one, and make sure first entry is retained, and
        # it is padded by zeros
        snap.mpcd.N = 2
        np.testing.assert_array_equal(snap.mpcd.position,
                                      [test_positions[0], [0, 0, 0]])
        np.testing.assert_array_equal(snap.mpcd.velocity,
                                      [test_velocities[0], [0, 0, 0]])
        np.testing.assert_array_equal(snap.mpcd.typeid, [test_typeids[0], 0])

        # grow the snapshot to the "standard" size and fill it back in
        snap.mpcd.N = test_positions.shape[0]
        snap.mpcd.position[:] = test_positions
        snap.mpcd.velocity[:] = test_velocities
        snap.mpcd.typeid[:] = test_typeids
        np.testing.assert_array_equal(snap.mpcd.position, test_positions)
        np.testing.assert_array_equal(snap.mpcd.velocity, test_velocities)
        np.testing.assert_array_equal(snap.mpcd.typeid, test_typeids)

        # resize down one, and make sure the first two entries are retained
        snap.mpcd.N = 2
        np.testing.assert_array_equal(snap.mpcd.position, test_positions[:2])
        np.testing.assert_array_equal(snap.mpcd.velocity, test_velocities[:2])
        np.testing.assert_array_equal(snap.mpcd.typeid, test_typeids[:2])


def test_replicate(snap):
    """Test that MPCD data is replicated with snapshot."""
    L = 5.0

    # initial configuration is a system with two particles
    pos0 = np.array([[-2, 0.5, -1], [2.0, -0.5, 1]])
    vel0 = [[1, 2, 3], [4, 5, 6]]
    typeid0 = [1, 0]
    types0 = ["A", "B"]
    nx, ny, nz = 2, 3, 4
    if snap.communicator.rank == 0:
        snap.configuration.box = [L, L, L, 0, 0, 0]
        snap.mpcd.N = 2
        snap.mpcd.position[:] = pos0
        snap.mpcd.velocity[:] = vel0
        snap.mpcd.typeid[:] = typeid0
        snap.mpcd.types = types0

    snap.replicate(nx=nx, ny=ny, nz=nz)

    if snap.communicator.rank == 0:
        # replicate the numpy position array along x, y, and z
        pos1 = []
        vel1 = []
        typeid1 = []
        new_origin = L * np.array([nx / 2, ny / 2, nz / 2])
        for i in range(0, nx):
            for j in range(0, ny):
                for k in range(0, nz):
                    for r, v, tid in zip(pos0, vel0, typeid0):
                        dr = np.array([i * L, j * L, k * L])
                        pos1 += [list(r + dr + 0.5 * L - new_origin)]
                        vel1 += [v]
                        typeid1 += [tid]

        scale = nx * ny * nz
        assert snap.mpcd.N == len(pos0) * scale
        np.testing.assert_allclose(snap.mpcd.position, pos1)
        np.testing.assert_allclose(snap.mpcd.velocity, vel1)
        np.testing.assert_array_equal(snap.mpcd.typeid, typeid1)
        np.testing.assert_equal(snap.mpcd.types, types0)


def test_create_and_restore_from_snap(snap, simulation_factory):
    """Test simulation can be created and restored with MPCD data."""
    if snap.communicator.num_ranks > 2:
        pytest.skip("Test must be run on 1 or 2 ranks")

    def reap_mpcd_pdata(simulation):
        """Reap MPCD particle data as a NumPy array, sorted by tag."""
        pdata = simulation._state._cpp_sys_def.getMPCDParticleData()
        dat = []
        for i in range(0, pdata.N):
            pos_i = pdata.getPosition(i)
            vel_i = pdata.getVelocity(i)
            type_i = pdata.getType(i)
            tag_i = pdata.getTag(i)
            dat += [[
                tag_i, pos_i.x, pos_i.y, pos_i.z, vel_i.x, vel_i.y, vel_i.z,
                type_i
            ]]
        return np.array(sorted(dat, key=lambda p: p[0]))

    # set snap values and initialize
    if snap.communicator.rank == 0:
        snap.mpcd.N = test_positions.shape[0]
        snap.mpcd.position[:] = test_positions
        snap.mpcd.velocity[:] = test_velocities
        snap.mpcd.typeid[:] = test_typeids
        snap.mpcd.types = test_types
        snap.mpcd.mass = test_mass
    sim = simulation_factory(snap, (1, 1, 2))

    # do a per-processor check that data is on the right rank
    dat = reap_mpcd_pdata(sim)
    if snap.communicator.num_ranks > 1:
        if snap.communicator.rank == 0:
            assert dat[0, 0] == 1
            np.testing.assert_array_equal(dat[0, 1:4], test_positions[1])
            np.testing.assert_array_equal(dat[0, 4:7], test_velocities[1])
            assert dat[0, 7] == test_typeids[1]
        else:
            np.testing.assert_array_equal(dat[:, 0], [0, 2])
            np.testing.assert_array_equal(dat[:, 1:4], test_positions[0::2])
            np.testing.assert_array_equal(dat[:, 4:7], test_velocities[0::2])
            np.testing.assert_array_equal(dat[:, 7], test_typeids[0::2])
    else:
        np.testing.assert_array_equal(dat[:, 0], [0, 1, 2])
        np.testing.assert_array_equal(dat[:, 1:4], test_positions)
        np.testing.assert_array_equal(dat[:, 4:7], test_velocities)
        np.testing.assert_array_equal(dat[:, 7], test_typeids)

    # next, shuffle particle data and restore simulation to those values
    targets = {}
    targets["position"] = 0.5 * np.roll(test_positions, 1, axis=0)
    targets["velocity"] = 2.0 * np.roll(test_velocities, 1, axis=0)
    targets["typeid"] = np.roll(test_typeids, 1, axis=0)
    targets["mass"] = 0.9
    snap2 = sim.state.get_snapshot()
    if snap2.communicator.rank == 0:
        snap2.mpcd.position[:] = targets["position"]
        snap2.mpcd.velocity[:] = targets["velocity"]
        snap2.mpcd.typeid[:] = targets["typeid"]
        snap2.mpcd.mass = targets["mass"]
    sim.state.set_snapshot(snap2)

    # do another per-processor check
    dat = reap_mpcd_pdata(sim)
    if snap.communicator.num_ranks > 1:
        if snap.communicator.rank == 0:
            assert dat[0, 0] == 2
            np.testing.assert_array_equal(dat[0, 1:4], targets["position"][2])
            np.testing.assert_array_equal(dat[0, 4:7], targets["velocity"][2])
            assert dat[0, 7] == targets["typeid"][2]
        else:
            np.testing.assert_array_equal(dat[:, 0], [0, 1])
            np.testing.assert_array_equal(dat[:, 1:4], targets["position"][0:2])
            np.testing.assert_array_equal(dat[:, 4:7], targets["velocity"][0:2])
            np.testing.assert_array_equal(dat[:, 7], targets["typeid"][0:2])
    else:
        np.testing.assert_array_equal(dat[:, 0], [0, 1, 2])
        np.testing.assert_array_equal(dat[:, 1:4], targets["position"])
        np.testing.assert_array_equal(dat[:, 4:7], targets["velocity"])
        np.testing.assert_array_equal(dat[:, 7], targets["typeid"])


def test_bad_typeid(snap, simulation_factory):
    """Test that out-of-range typeid in MPCD data is an error."""
    if snap.communicator.rank == 0:
        snap.mpcd.N = test_positions.shape[0]
        snap.mpcd.typeid[:] = test_typeids
        snap.mpcd.types = ["A", "B"]
    sim = simulation_factory()
    with pytest.raises(RuntimeError):
        sim.create_state_from_snapshot(snap)


def test_bad_positions(snap, simulation_factory):
    """Test that out-of-bounds position in MPCD data is an error."""
    if snap.communicator.rank == 0:
        snap.mpcd.N = 2
        snap.mpcd.position[1] = [11.0, 0.0, 0.0]
    sim = simulation_factory()
    with pytest.raises(RuntimeError):
        sim.create_state_from_snapshot(snap)
