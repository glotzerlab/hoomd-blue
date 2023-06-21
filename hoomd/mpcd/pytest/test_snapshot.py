# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import numpy as np
import pytest

# default testing configuration, used in setting values below
test_positions = np.array([[1, 2, 3], [-4, -5, -6], [7, 8, 9]])
test_velocities = np.array([[-1, -2, -3], [4, 5, 6], [-7, -8, -9]])
test_typeids = np.array([1, 0, 2])
test_types = ['M', 'P', 'H']
test_mass = 1.5


@pytest.fixture
def snap():
    return hoomd.Snapshot()


def test_make(snap):
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
        snap.mpcd.position[:] = tuple(test_positions)
        snap.mpcd.velocity[:] = list(test_velocities)
        snap.mpcd.typeid[:] = test_typeids
        snap.mpcd.types = test_types
        snap.mpcd.mass = test_mass
        np.testing.assert_array_equal(snap.mpcd.position, test_positions)
        np.testing.assert_array_equal(snap.mpcd.velocity, test_velocities)
        np.testing.assert_array_equal(snap.mpcd.typeid, test_typeids)
        np.testing.assert_array_equal(snap.mpcd.types, test_types)


# test that snapshot can be resized without losing data
def test_resize(snap):
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

        # # grow the snapshot by one, and make sure first entry is retained, and it is padded by zeros
        snap.mpcd.N = 2
        np.testing.assert_array_equal(snap.mpcd.position,
                                      [test_positions[0], [0, 0, 0]])
        np.testing.assert_array_equal(snap.mpcd.velocity,
                                      [test_velocities[0], [0, 0, 0]])
        np.testing.assert_array_equal(snap.mpcd.typeid, [test_typeids[0], 0])

        # # grow the snapshot to the "standard" size and fill it back in
        snap.mpcd.N = 3
        snap.mpcd.position[:] = test_positions
        snap.mpcd.velocity[:] = test_velocities
        snap.mpcd.typeid[:] = test_typeids
        np.testing.assert_array_equal(snap.mpcd.position, test_positions)
        np.testing.assert_array_equal(snap.mpcd.velocity, test_velocities)
        np.testing.assert_array_equal(snap.mpcd.typeid, test_typeids)

        # # resize down one, and make sure the first two entries are retained
        snap.mpcd.N = 2
        np.testing.assert_array_equal(snap.mpcd.position, test_positions[:2])
        np.testing.assert_array_equal(snap.mpcd.velocity, test_velocities[:2])
        np.testing.assert_array_equal(snap.mpcd.typeid, test_typeids[:2])


def test_replicate(snap):
    L = 5.

    # initial configuration is a system with two particles
    pos0 = np.array([[-2, 0.5, -1], [2., -0.5, 1]])
    vel0 = [[1, 2, 3], [4, 5, 6]]
    typeid0 = [1, 0]
    types0 = ['A', 'B']
    nx, ny, nz = 2, 3, 4
    if snap.communicator.rank == 0:
        snap.configuration.box = [L, L, L, 0, 0, 0]
        snap.mpcd.N = 2
        snap.mpcd.position[:] = pos0
        snap.mpcd.velocity[:] = vel0
        snap.mpcd.typeid[:] = typeid0
        snap.mpcd.types = types0

        # replicate the system before initialization
        snap.replicate(nx=nx, ny=ny, nz=nz)

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
    if snap.communicator.rank == 0:
        assert snap.mpcd.N == len(pos0) * scale
        np.testing.assert_allclose(snap.mpcd.position, pos1)
        np.testing.assert_allclose(snap.mpcd.velocity, vel1)
        np.testing.assert_array_equal(snap.mpcd.typeid, typeid1)
        np.testing.assert_equal(snap.mpcd.types, types0)
