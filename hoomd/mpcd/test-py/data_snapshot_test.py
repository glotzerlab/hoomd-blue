# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: mphoward

import unittest
import numpy as np
import hoomd
from hoomd import mpcd

# unit tests for snapshots with mpcd particle data
class mpcd_snapshot(unittest.TestCase):
    def setUp(self):
        # establish the simulation context
        hoomd.context.initialize()

        # set the decomposition in z for mpi builds
        if hoomd.comm.get_num_ranks() > 1:
            hoomd.comm.decomposition(nz=2)

        # default testing configuration
        self.s = None
        self.positions = np.array( [[1,2,3],[-4,-5,-6],[7,8,9]] )
        self.velocities = np.array( [[-1,-2,-3],[4,5,6],[-7,-8,-9]] )
        self.typeids = np.array([1,0,2])
        self.types = ['M','P','H']
        self.mass = 1.5

    # test making and reading a snapshot with pure mpcd
    def test_make_read(self):
        hoomd.init.read_snapshot(hoomd.data.make_snapshot(N=0, box=hoomd.data.boxdim(L=20.)))
        snap = mpcd.data.make_snapshot()

        if hoomd.comm.get_rank() == 0:
            # mpcd should be empty in a snapshot by default
            self.assertEqual(snap.particles.N, 0)
            self.assertEqual(len(snap.particles.position), 0)
            self.assertEqual(len(snap.particles.velocity), 0)
            self.assertEqual(len(snap.particles.typeid), 0)
            self.assertEqual(len(snap.particles.types), 0)
            self.assertAlmostEqual(snap.particles.mass, 1.0)

            # resize and test default fillings
            snap.particles.resize(3)
            self.assertEqual(snap.particles.N, 3)
            self.assertEqual(len(snap.particles.position), 3)
            self.assertEqual(len(snap.particles.velocity), 3)
            self.assertEqual(len(snap.particles.typeid), 3)
            np.testing.assert_allclose(snap.particles.position, np.zeros((3,3)))
            np.testing.assert_allclose(snap.particles.velocity, np.zeros((3,3)))
            np.testing.assert_array_equal(snap.particles.typeid, np.zeros(3))

            # test setting values in the snapshot
            snap.particles.position[:] = tuple(self.positions)
            snap.particles.velocity[:] = list(self.velocities)
            snap.particles.typeid[:] = self.typeids
            snap.particles.types = self.types
            snap.particles.mass = self.mass
            np.testing.assert_allclose(snap.particles.position, self.positions)
            np.testing.assert_allclose(snap.particles.velocity, self.velocities)
            np.testing.assert_array_equal(snap.particles.typeid, self.typeids)
            np.testing.assert_array_equal(snap.particles.types, self.types)

        # restore the system from the snapshot
        self.s = mpcd.init.read_snapshot(snap)

        # check on the initialization
        pdata = self.s.particles
        self.assertEqual(pdata.N_global, 3)
        if hoomd.comm.get_num_ranks() > 1:
            # mpi test by rank
            if hoomd.comm.get_rank() == 0:
                self.assertEqual(pdata.N, 1)
            else:
                self.assertEqual(pdata.N, 2)
        else:
            self.assertEqual(pdata.N, 3)

        # reap all of the particle data into checkable list sorted by tag
        dat = []
        for i in range(0,pdata.N):
            pos_i = pdata.getPosition(i)
            vel_i = pdata.getVelocity(i)
            type_i = pdata.getType(i)
            tag_i = pdata.getTag(i)
            dat += [[tag_i, pos_i.x, pos_i.y, pos_i.z, vel_i.x, vel_i.y, vel_i.z, type_i]]
        dat = np.array( sorted(dat, key=lambda p : p[0]) )

        # now we do a per-processor check
        if hoomd.comm.get_num_ranks() > 1:
            if hoomd.comm.get_rank() == 0:
                self.assertEqual(dat[0,0], 1)
                np.testing.assert_allclose(dat[0,1:4], self.positions[1])
                np.testing.assert_allclose(dat[0,4:7], self.velocities[1])
                self.assertEqual(dat[0,7], self.typeids[1])
            else:
                np.testing.assert_array_equal(dat[:,0],[0,2])
                np.testing.assert_allclose(dat[:,1:4], self.positions[0::2])
                np.testing.assert_allclose(dat[:,4:7], self.velocities[0::2])
                np.testing.assert_array_equal(dat[:,7], self.typeids[0::2])
        else:
            np.testing.assert_array_equal(dat[:,0],[0,1,2])
            np.testing.assert_allclose(dat[:,1:4], self.positions)
            np.testing.assert_allclose(dat[:,4:7], self.velocities)
            np.testing.assert_array_equal(dat[:,7], self.typeids)

        # check type mapping, which should be present on all ranks
        self.assertEqual(pdata.n_types, 3)
        np.testing.assert_array_equal(pdata.types, self.types)
        np.testing.assert_array_equal([pdata.getNameByType(i) for i in range(0,len(self.types))], self.types)
        np.testing.assert_array_equal([pdata.getTypeByName(t) for t in self.types], range(0,len(self.types)))

        # check the particle mass
        self.assertAlmostEqual(pdata.mass, self.mass)

    # test that system can be restored from a snapshot
    def test_take_restore(self):
        hoomd.init.read_snapshot(hoomd.data.make_snapshot(N=0, box=hoomd.data.boxdim(L=20.)))

        # initialize the system from the starting snapshot
        snap = mpcd.data.make_snapshot()
        snap.particles.resize(3)
        snap.particles.position[:] = tuple(self.positions)
        snap.particles.velocity[:] = list(self.velocities)
        snap.particles.typeid[:] = self.typeids
        snap.particles.types = self.types
        snap.particles.mass = self.mass
        self.s = mpcd.init.read_snapshot(snap)
        del snap

        # take a snapshot and validate that the data matches what we fed in
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            self.assertEqual(snap.particles.N, 3)
            np.testing.assert_allclose(snap.particles.position, self.positions)
            np.testing.assert_allclose(snap.particles.velocity, self.velocities)
            np.testing.assert_array_equal(snap.particles.typeid, self.typeids)
            np.testing.assert_array_equal(snap.particles.types, self.types)
            self.assertAlmostEqual(snap.particles.mass, self.mass)

        # roll and rescale elements in the arrays for a fresh initializing
        targets = {}
        targets['position'] = 0.5*np.roll(self.positions, 1, axis=0)
        targets['velocity'] = 2.0*np.roll(self.velocities, 1, axis=0)
        targets['typeid'] = np.roll(self.typeids, 1, axis=0)
        targets['types'] = ['R','L','GH']
        targets['mass'] = 0.9
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[:] = targets['position']
            snap.particles.velocity[:] = targets['velocity']
            snap.particles.typeid[:] = targets['typeid']
            snap.particles.types = targets['types']
            snap.particles.mass = targets['mass']
        self.s.restore_snapshot(snap)

        # check the particle data as it was reinitialized matches what we set
        pdata = self.s.particles
        self.assertEqual(pdata.N_global, 3)
        if hoomd.comm.get_num_ranks() > 1:
            # mpi test by rank
            if hoomd.comm.get_rank() == 0:
                self.assertEqual(pdata.N, 1)
            else:
                self.assertEqual(pdata.N, 2)
        else:
            self.assertEqual(pdata.N, 3)

        # reap all of the particle data into checkable list sorted by tag
        dat = []
        for i in range(0,pdata.N):
            pos_i = pdata.getPosition(i)
            vel_i = pdata.getVelocity(i)
            type_i = pdata.getType(i)
            tag_i = pdata.getTag(i)
            dat += [[tag_i, pos_i.x, pos_i.y, pos_i.z, vel_i.x, vel_i.y, vel_i.z, type_i]]
        dat = np.array( sorted(dat, key=lambda p : p[0]) )

        # now we do a per-processor check
        if hoomd.comm.get_num_ranks() > 1:
            if hoomd.comm.get_rank() == 0:
                self.assertEqual(dat[0,0], 2)
                np.testing.assert_allclose(dat[0,1:4], targets['position'][2])
                np.testing.assert_allclose(dat[0,4:7], targets['velocity'][2])
                self.assertEqual(dat[0,7], targets['typeid'][2])
            else:
                np.testing.assert_array_equal(dat[:,0],[0,1])
                np.testing.assert_allclose(dat[:,1:4], targets['position'][0:2])
                np.testing.assert_allclose(dat[:,4:7], targets['velocity'][0:2])
                np.testing.assert_array_equal(dat[:,7], targets['typeid'][0:2])
        else:
            np.testing.assert_array_equal(dat[:,0],[0,1,2])
            np.testing.assert_allclose(dat[:,1:4], targets['position'])
            np.testing.assert_allclose(dat[:,4:7], targets['velocity'])
            np.testing.assert_array_equal(dat[:,7], targets['typeid'])

        # check type mapping, which should be present on all ranks
        self.assertEqual(pdata.n_types, 3)
        np.testing.assert_array_equal(pdata.types, targets['types'])
        np.testing.assert_array_equal([pdata.getNameByType(i) for i in range(0,len(targets['types']))], targets['types'])
        np.testing.assert_array_equal([pdata.getTypeByName(t) for t in targets['types']], range(0,len(targets['types'])))

        # check the mass
        self.assertAlmostEqual(pdata.mass, targets['mass'])

    # test that snapshot can be resized without losing data
    def test_resize(self):
        hoomd.init.read_snapshot(hoomd.data.make_snapshot(N=0, box=hoomd.data.boxdim(L=20.)))
        snap = mpcd.data.make_snapshot()

        # start the snapshot from one particle
        snap.particles.resize(1)
        snap.particles.position[:] = self.positions[0]
        snap.particles.velocity[:] = self.velocities[0]
        snap.particles.typeid[:] = self.typeids[0]
        snap.particles.types = self.types
        np.testing.assert_array_equal(snap.particles.position, [self.positions[0]])
        np.testing.assert_array_equal(snap.particles.velocity, [self.velocities[0]])
        np.testing.assert_array_equal(snap.particles.typeid, [self.typeids[0]])

        # grow the snapshot by one, and make sure first entry is retained, and it is padded by zeros
        snap.particles.resize(2)
        np.testing.assert_array_equal(snap.particles.position, [self.positions[0], [0,0,0]])
        np.testing.assert_array_equal(snap.particles.velocity, [self.velocities[0], [0,0,0]])
        np.testing.assert_array_equal(snap.particles.typeid, [self.typeids[0], 0])

        # grow the snapshot to the "standard" size and fill it back in
        snap.particles.resize(3)
        snap.particles.position[:] = self.positions
        snap.particles.velocity[:] = self.velocities
        snap.particles.typeid[:] = self.typeids
        np.testing.assert_array_equal(snap.particles.position, self.positions)
        np.testing.assert_array_equal(snap.particles.velocity, self.velocities)
        np.testing.assert_array_equal(snap.particles.typeid, self.typeids)

        # resize down one, and make sure the first two entries are retained
        snap.particles.resize(2)
        np.testing.assert_array_equal(snap.particles.position, self.positions[:2])
        np.testing.assert_array_equal(snap.particles.velocity, self.velocities[:2])
        np.testing.assert_array_equal(snap.particles.typeid, self.typeids[:2])

    # test the systems ability to replicate itself
    def test_replicate(self):
        L = 5.
        hoomd_sys = hoomd.init.read_snapshot(hoomd.data.make_snapshot(N=0, box=hoomd.data.boxdim(L=L)))
        snap = mpcd.data.make_snapshot()

        # initial configuration is a system with two particles
        pos0 = np.array([[-2,0.5,-1],[2.,-0.5, 1]])
        vel0 = [[1,2,3],[4,5,6]]
        typeid0 = [1,0]
        types0 = ['A','B']
        snap.particles.resize(2)
        snap.particles.position[:] = pos0
        snap.particles.velocity[:] = vel0
        snap.particles.typeid[:] = typeid0
        snap.particles.types = types0

        # replicate the system before initialization
        nx=2; ny=2; nz=2
        hoomd_sys.replicate(nx=nx,ny=ny,nz=nz)
        # initialization should fail because snapshot has not been replicated
        with self.assertRaises(RuntimeError):
            self.s = mpcd.init.read_snapshot(snap)
        self.assertEqual(self.s, None)
        # now box is resized, and it should proceed
        snap.replicate(nx=nx,ny=ny,nz=nz)
        self.s = mpcd.init.read_snapshot(snap)

        # replicate the numpy position array along x, y, and z
        pos1 = []
        vel1 = []
        typeid1 = []
        for i in range(0,nx):
            for j in range(0,ny):
                for k in range(0,nz):
                    for r,v,tid in zip(pos0, vel0, typeid0):
                        dr = np.array([i*L, j*L, k*L])
                        pos1 += [list(r + dr - 0.5*L)]
                        vel1 += [v]
                        typeid1 += [tid]

        scale = nx*ny*nz
        snap = self.s.take_snapshot()
        if hoomd.comm.get_rank() == 0:
            self.assertEqual(snap.particles.N, len(pos0)*scale)
            # python is borking the rounding (c++ is good), and everything is at
            # least 0.5 away so we can just use a very loose tolerance here
            np.testing.assert_allclose(snap.particles.position, pos1, atol=0.1)
            np.testing.assert_allclose(snap.particles.velocity, vel1)
            np.testing.assert_array_equal(snap.particles.typeid, typeid1)
            np.testing.assert_equal(snap.particles.types, types0)

        # after initialization, replication should throw an error
        with self.assertRaises(RuntimeError):
            hoomd_sys.replicate(nx=2,ny=1,nz=1)

    # test for typeid out of range
    def test_bad_typeid(self):
        hoomd.init.read_snapshot(hoomd.data.make_snapshot(N=0, box=hoomd.data.boxdim(L=20.)))
        snap = mpcd.data.make_snapshot()
        snap.particles.resize(3)
        snap.particles.typeid[:] = self.typeids
        snap.particles.types = ['A','B']
        with self.assertRaises(RuntimeError):
            mpcd.init.read_snapshot(snap)

    # check for position outside the box
    def test_bad_positions(self):
        hoomd.init.read_snapshot(hoomd.data.make_snapshot(N=0, box=hoomd.data.boxdim(L=20.)))
        snap = mpcd.data.make_snapshot()
        snap.particles.resize(2)
        snap.particles.position[1] = [11., 0., 0.]
        with self.assertRaises(RuntimeError):
            mpcd.init.read_snapshot(snap)

    def tearDown(self):
        del self.s

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
