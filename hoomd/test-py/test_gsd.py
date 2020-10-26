# -*- coding: iso-8859-1 -*-
# Maintainer: joaander

from hoomd import *
import hoomd;
import unittest
import os
import numpy
import tempfile

# unit tests for dump.gsd
class gsd_write_tests (unittest.TestCase):
    def setUp(self):
        context.initialize()
        if hoomd.comm.get_rank() == 0:
            tmp = tempfile.mkstemp(suffix='.test.gsd');
            self.tmp_file = tmp[1];
        else:
            self.tmp_file = "invalid";

        self.snapshot = data.make_snapshot(N=4, box=data.boxdim(Lx=10, Ly=20, Lz=30), dtype='float');
        if comm.get_rank() == 0:
            # particles
            self.snapshot.particles.position[0] = [0,1,2];
            self.snapshot.particles.position[1] = [1,2,3];
            self.snapshot.particles.position[2] = [0,-1,-2];
            self.snapshot.particles.position[3] = [-1, -2, -3];
            self.snapshot.particles.velocity[0] = [10, 11, 12];
            self.snapshot.particles.velocity[1] = [11, 12, 13];
            self.snapshot.particles.velocity[2] = [12, 13, 14];
            self.snapshot.particles.velocity[3] = [13, 14, 15];
            self.snapshot.particles.acceleration[0] = [20, 21, 22];
            self.snapshot.particles.acceleration[1] = [21, 22, 23];
            self.snapshot.particles.acceleration[2] = [22, 23, 24];
            self.snapshot.particles.acceleration[3] = [23, 24, 25];
            self.snapshot.particles.typeid[:] = [0,0,1,1];
            self.snapshot.particles.mass[:] = [33, 34, 35,  36];
            self.snapshot.particles.charge[:] = [44, 45, 46, 47];
            self.snapshot.particles.diameter[:] = [55, 56, 57, 58];
            self.snapshot.particles.image[0] = [60, 61, 62];
            self.snapshot.particles.image[1] = [61, 62, 63];
            self.snapshot.particles.image[2] = [62, 63, 64];
            self.snapshot.particles.image[3] = [63, 64, 65];
            self.snapshot.particles.types = ['p1', 'p2'];

            # bonds
            self.snapshot.bonds.types = ['b1', 'b2'];
            self.snapshot.bonds.resize(2);
            self.snapshot.bonds.typeid[:] = [0, 1];
            self.snapshot.bonds.group[0] = [0, 1];
            self.snapshot.bonds.group[1] = [2, 3];

            # angles
            self.snapshot.angles.types = ['a1', 'a2'];
            self.snapshot.angles.resize(2);
            self.snapshot.angles.typeid[:] = [1, 0];
            self.snapshot.angles.group[0] = [0, 1, 2];
            self.snapshot.angles.group[1] = [2, 3, 0];

            # dihedrals
            self.snapshot.dihedrals.types = ['d1'];
            self.snapshot.dihedrals.resize(1);
            self.snapshot.dihedrals.typeid[:] = [0];
            self.snapshot.dihedrals.group[0] = [0, 1, 2, 3];

            # impropers
            self.snapshot.impropers.types = ['i1'];
            self.snapshot.impropers.resize(1);
            self.snapshot.impropers.typeid[:] = [0];
            self.snapshot.impropers.group[0] = [3, 2, 1, 0];

            # constraints
            self.snapshot.constraints.resize(1)
            self.snapshot.constraints.group[0] = [0, 1]
            self.snapshot.constraints.value[0] = 2.5

            # special pairs
            self.snapshot.pairs.types = ['p1', 'p2'];
            self.snapshot.pairs.resize(2);
            self.snapshot.pairs.typeid[:] = [0, 1];
            self.snapshot.pairs.group[0] = [0, 1];
            self.snapshot.pairs.group[1] = [2, 3];


        self.s = init.read_snapshot(self.snapshot);

        context.current.sorter.set_params(grid=8)

    # tests basic creation of the dump
    def test(self):
        dump.gsd(filename=self.tmp_file, group=group.all(), period=1, overwrite=True);
        run(5);
        # ensure 5 frames are written to the file
        data.gsd_snapshot(self.tmp_file, frame=4);
        if comm.get_rank() == 0:
            self.assertRaises(RuntimeError, data.gsd_snapshot, self.tmp_file, frame=5);

    # tests with phase
    def test_phase(self):
        dump.gsd(filename=self.tmp_file, group=group.all(), period=1, phase=0, overwrite=True);
        run(1);
        data.gsd_snapshot(self.tmp_file, frame=0);
        if comm.get_rank() == 0:
            self.assertRaises(RuntimeError, data.gsd_snapshot, self.tmp_file, frame=1);

    # tests overwrite
    def test_overwrite(self):
        if comm.get_rank() == 0:
            with open(self.tmp_file, 'wt') as f:
                f.write('Hello');

        dump.gsd(filename=self.tmp_file, group=group.all(), period=1, overwrite=True);
        run(1);
        data.gsd_snapshot(self.tmp_file, frame=0);
        if comm.get_rank() == 0:
            self.assertRaises(RuntimeError, data.gsd_snapshot, self.tmp_file, frame=1);

    # tests truncate
    def test_truncate(self):
        dump.gsd(filename=self.tmp_file, group=group.all(), period=1, truncate=True, overwrite=True);
        run(5);
        data.gsd_snapshot(self.tmp_file, frame=0);
        if comm.get_rank() == 0:
            self.assertRaises(RuntimeError, data.gsd_snapshot, self.tmp_file, frame=1);

    # tests write_restart
    def test_write_restart(self):
        g = dump.gsd(filename=self.tmp_file, group=group.all(), period=1000000, truncate=True, overwrite=True);
        run(5);
        g.write_restart();
        data.gsd_snapshot(self.tmp_file, frame=0);
        if comm.get_rank() == 0:
            self.assertRaises(RuntimeError, data.gsd_snapshot, self.tmp_file, frame=1);

    # test all static quantities
    def test_all_static(self):
        dump.gsd(filename=self.tmp_file, group=group.all(), period=1, static=['attribute', 'property', 'momentum', 'topology'], overwrite=True);
        run(1);
        data.gsd_snapshot(self.tmp_file, frame=0);
        if comm.get_rank() == 0:
            self.assertRaises(RuntimeError, data.gsd_snapshot, self.tmp_file, frame=1);

    def test_dynamic(self):
        dump.gsd(filename=self.tmp_file, group=group.all(), period=1, dynamic=['momentum'], overwrite=True);
        run(1);
        data.gsd_snapshot(self.tmp_file, frame=0);
        if comm.get_rank() == 0:
            self.assertRaises(RuntimeError, data.gsd_snapshot, self.tmp_file, frame=1);

    # test write file
    def test_write_immediate(self):
        dump.gsd(filename=self.tmp_file, group=group.all(), period=None, time_step=1000, overwrite=True);
        data.gsd_snapshot(self.tmp_file, frame=0);
        if comm.get_rank() == 0:
            self.assertRaises(RuntimeError, data.gsd_snapshot, self.tmp_file, frame=1);

    # tests init.read_gsd
    def test_read_gsd(self):
        dump.gsd(filename=self.tmp_file, group=group.all(), period=1, overwrite=True);
        run(5);

        context.initialize();
        init.read_gsd(filename=self.tmp_file, frame=4);
        self.assertEqual(get_step(), 4)
        if comm.get_rank() == 0:
            self.assertRaises(RuntimeError, init.read_gsd, self.tmp_file, frame=5);

    # tests init.read_gsd time_step
    def test_read_gsd_time_step(self):
        dump.gsd(filename=self.tmp_file, group=group.all(), period=1, overwrite=True);
        run(5);

        context.initialize();
        # test that time_step is set appropriately
        init.read_gsd(filename=self.tmp_file, frame=4, time_step=1000);
        self.assertEqual(get_step(), 1000)

        # when restart is present, the time_step field should be ignored
        context.initialize();
        init.read_gsd(filename=self.tmp_file, restart=self.tmp_file, frame=4, time_step=1000);
        self.assertEqual(get_step(), 4)

    # tests with zero particles
    def test_zero_particles(self):
        self.s.particles.remove(0)
        self.s.particles.remove(1)
        self.s.particles.remove(2)
        self.s.particles.remove(3)
        dump.gsd(filename=self.tmp_file, group=group.all(), period=1, overwrite=True);
        run(1)

    def test_log(self):
        gsd = dump.gsd(filename=self.tmp_file, group=group.all(), period=1, overwrite=True);
        gsd.log['uint8'] = lambda step: numpy.array([1, 2, 3, 4], dtype=numpy.uint8)
        gsd.log['uint16'] = lambda step: numpy.array([1, 2, 3, 4], dtype=numpy.uint16)
        gsd.log['uint32'] = lambda step: numpy.array([1, 2, 3, 4], dtype=numpy.uint32)
        gsd.log['uint64'] = lambda step: numpy.array([1, 2, 3, 4], dtype=numpy.uint64)

        gsd.log['int8'] = lambda step: numpy.array([1, 2, 3, 4, -10], dtype=numpy.int8)
        gsd.log['int16'] = lambda step: numpy.array([1, 2, 3, 4, -10], dtype=numpy.int16)
        gsd.log['int32'] = lambda step: numpy.array([1, 2, 3, 4, -10], dtype=numpy.int32)
        gsd.log['int64'] = lambda step: numpy.array([1, 2, 3, 4, -10], dtype=numpy.int64)

        gsd.log['float32'] = lambda step: numpy.array([1, 2, 3, 4], dtype=numpy.float32)
        gsd.log['float64'] = lambda step: numpy.array([1, 2, 3, 4], dtype=numpy.float64)

        gsd.log['2d'] = lambda step: numpy.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=numpy.float64)
        run(1)

        # skip these tests in MPI, only the root rank raises the runtime error resulting in deadlock
        if hoomd.comm.get_num_ranks() == 1:
            gsd.log['complex64'] = lambda step: numpy.array([1, 2, 3, 4], dtype=numpy.complex64)
            self.assertRaises(RuntimeError, run, 1);

            del gsd.log['complex64'];
            run(1)

            gsd.log['3d'] = lambda step: numpy.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=numpy.float64)
            self.assertRaises(RuntimeError, run, 1);

            del gsd.log['3d']
            run(1)

            gsd.log['scalar'] = lambda step: 5
            self.assertRaises(RuntimeError, run, 1);


    def tearDown(self):
        if (hoomd.comm.get_rank()==0):
            os.remove(self.tmp_file);

        comm.barrier_all();

# unit tests for dump.gsd
class gsd_read_tests (unittest.TestCase):
    def setUp(self):
        context.initialize()
        if hoomd.comm.get_rank() == 0:
            tmp = tempfile.mkstemp(suffix='.test.gsd');
            self.tmp_file = tmp[1];
        else:
            self.tmp_file = "invalid";

        self.snapshot = data.make_snapshot(N=4, box=data.boxdim(L=10), dtype='float');
        if comm.get_rank() == 0:
            # particles
            self.snapshot.particles.position[0] = [0,1,2];
            self.snapshot.particles.position[1] = [1,2,3];
            self.snapshot.particles.position[2] = [0,-1,-2];
            self.snapshot.particles.position[3] = [-1, -2, -3];
            self.snapshot.particles.velocity[0] = [10, 11, 12];
            self.snapshot.particles.velocity[1] = [11, 12, 13];
            self.snapshot.particles.velocity[2] = [12, 13, 14];
            self.snapshot.particles.velocity[3] = [13, 14, 15];
            self.snapshot.particles.orientation[0] = [19, 20, 21, 22];
            self.snapshot.particles.orientation[1] = [20, 21, 22, 23];
            self.snapshot.particles.orientation[2] = [21, 22, 23, 24];
            self.snapshot.particles.orientation[3] = [22, 23, 24, 25];
            self.snapshot.particles.angmom[0] = [119, 220, 321, 422];
            self.snapshot.particles.angmom[1] = [120, 221, 322, 423];
            self.snapshot.particles.angmom[2] = [121, 222, 323, 424];
            self.snapshot.particles.angmom[3] = [122, 223, 324, 425];
            self.snapshot.particles.typeid[:] = [0,0,1,1];
            self.snapshot.particles.mass[:] = [33, 34, 35,  36];
            self.snapshot.particles.charge[:] = [44, 45, 46, 47];
            self.snapshot.particles.diameter[:] = [55, 56, 57, 58];
            self.snapshot.particles.body[:] = [-1, -1, -1, -1];
            self.snapshot.particles.image[0] = [60, 61, 62];
            self.snapshot.particles.image[1] = [61, 62, 63];
            self.snapshot.particles.image[2] = [62, 63, 64];
            self.snapshot.particles.image[3] = [63, 64, 65];
            self.snapshot.particles.moment_inertia[0] = [50, 51, 52];
            self.snapshot.particles.moment_inertia[1] = [51, 52, 53];
            self.snapshot.particles.moment_inertia[2] = [52, 53, 54];
            self.snapshot.particles.moment_inertia[3] = [53, 54, 55];
            self.snapshot.particles.types = ['p1', 'p2'];

            # bonds
            self.snapshot.bonds.types = ['b1', 'b2'];
            self.snapshot.bonds.resize(2);
            self.snapshot.bonds.typeid[:] = [0, 1];
            self.snapshot.bonds.group[0] = [0, 1];
            self.snapshot.bonds.group[1] = [2, 3];

            # angles
            self.snapshot.angles.types = ['a1', 'a2'];
            self.snapshot.angles.resize(2);
            self.snapshot.angles.typeid[:] = [1, 0];
            self.snapshot.angles.group[0] = [0, 1, 2];
            self.snapshot.angles.group[1] = [2, 3, 0];

            # dihedrals
            self.snapshot.dihedrals.types = ['d1'];
            self.snapshot.dihedrals.resize(1);
            self.snapshot.dihedrals.typeid[:] = [0];
            self.snapshot.dihedrals.group[0] = [0, 1, 2, 3];

            # impropers
            self.snapshot.impropers.types = ['i1'];
            self.snapshot.impropers.resize(1);
            self.snapshot.impropers.typeid[:] = [0];
            self.snapshot.impropers.group[0] = [3, 2, 1, 0];

            # constraints
            self.snapshot.constraints.resize(1)
            self.snapshot.constraints.group[0] = [0, 1]
            self.snapshot.constraints.value[0] = 2.5

            # pairs
            self.snapshot.pairs.types = ['p1', 'p2'];
            self.snapshot.pairs.resize(2);
            self.snapshot.pairs.typeid[:] = [0, 1];
            self.snapshot.pairs.group[0] = [0, 1];
            self.snapshot.pairs.group[1] = [2, 3];


        self.s = init.read_snapshot(self.snapshot);
        context.current.sorter.set_params(grid=8)

    # tests data.gsd_snapshot
    def test_gsd_snapshot(self):
        dump.gsd(filename=self.tmp_file, group=group.all(), period=None, overwrite=True);

        snap = data.gsd_snapshot(self.tmp_file, frame=0);
        if comm.get_rank() == 0:
            self.assertEqual(snap.box.dimensions, self.snapshot.box.dimensions);
            self.assertEqual(snap.box.Lx, self.snapshot.box.Lx);
            self.assertEqual(snap.box.Ly, self.snapshot.box.Ly);
            self.assertEqual(snap.box.Lz, self.snapshot.box.Lz);
            self.assertEqual(snap.box.xy, self.snapshot.box.xy);
            self.assertEqual(snap.box.xz, self.snapshot.box.xz);
            self.assertEqual(snap.box.yz, self.snapshot.box.yz);

            self.assertEqual(snap.particles.N, self.snapshot.particles.N);
            self.assertEqual(snap.particles.types, self.snapshot.particles.types);

            numpy.testing.assert_array_equal(snap.particles.typeid, self.snapshot.particles.typeid);
            numpy.testing.assert_array_equal(snap.particles.mass, self.snapshot.particles.mass);
            numpy.testing.assert_array_equal(snap.particles.charge, self.snapshot.particles.charge);
            numpy.testing.assert_array_equal(snap.particles.diameter, self.snapshot.particles.diameter);
            numpy.testing.assert_array_equal(snap.particles.body, self.snapshot.particles.body);
            numpy.testing.assert_array_equal(snap.particles.moment_inertia, self.snapshot.particles.moment_inertia);
            numpy.testing.assert_array_equal(snap.particles.position, self.snapshot.particles.position);
            numpy.testing.assert_array_equal(snap.particles.orientation, self.snapshot.particles.orientation);
            numpy.testing.assert_array_equal(snap.particles.velocity, self.snapshot.particles.velocity);
            numpy.testing.assert_array_equal(snap.particles.angmom, self.snapshot.particles.angmom);
            numpy.testing.assert_array_equal(snap.particles.image, self.snapshot.particles.image);

            self.assertEqual(snap.bonds.N, self.snapshot.bonds.N);
            self.assertEqual(snap.bonds.types, self.snapshot.bonds.types);
            numpy.testing.assert_array_equal(snap.bonds.typeid, self.snapshot.bonds.typeid);
            numpy.testing.assert_array_equal(snap.bonds.group, self.snapshot.bonds.group);

            self.assertEqual(snap.angles.N, self.snapshot.angles.N);
            self.assertEqual(snap.angles.types, self.snapshot.angles.types);
            numpy.testing.assert_array_equal(snap.angles.typeid, self.snapshot.angles.typeid);
            numpy.testing.assert_array_equal(snap.angles.group, self.snapshot.angles.group);

            self.assertEqual(snap.dihedrals.N, self.snapshot.dihedrals.N);
            self.assertEqual(snap.dihedrals.types, self.snapshot.dihedrals.types);
            numpy.testing.assert_array_equal(snap.dihedrals.typeid, self.snapshot.dihedrals.typeid);
            numpy.testing.assert_array_equal(snap.dihedrals.group, self.snapshot.dihedrals.group);

            self.assertEqual(snap.impropers.N, self.snapshot.impropers.N);
            self.assertEqual(snap.impropers.types, self.snapshot.impropers.types);
            numpy.testing.assert_array_equal(snap.impropers.typeid, self.snapshot.impropers.typeid);
            numpy.testing.assert_array_equal(snap.impropers.group, self.snapshot.impropers.group);

            self.assertEqual(snap.constraints.N, self.snapshot.constraints.N);
            numpy.testing.assert_array_equal(snap.constraints.group, self.snapshot.constraints.group);
            numpy.testing.assert_array_equal(snap.constraints.value, self.snapshot.constraints.value);

            self.assertEqual(snap.pairs.N, self.snapshot.pairs.N);
            self.assertEqual(snap.pairs.types, self.snapshot.pairs.types);
            numpy.testing.assert_array_equal(snap.pairs.typeid, self.snapshot.pairs.typeid);
            numpy.testing.assert_array_equal(snap.pairs.group, self.snapshot.pairs.group);


    # test changing the order particles
    def test_remove(self):
        # remove particle so that tag 2 points to no particle, and particle tags are no longer contiguous
        self.s.particles.remove(2)
        self.snapshot = self.s.take_snapshot(all=True)
        dump.gsd(filename=self.tmp_file, group=group.all(), period=None, overwrite=True);

        snap = data.gsd_snapshot(self.tmp_file, frame=0);
        if comm.get_rank() == 0:
            self.assertEqual(snap.box.dimensions, self.snapshot.box.dimensions);
            self.assertEqual(snap.box.Lx, self.snapshot.box.Lx);
            self.assertEqual(snap.box.Ly, self.snapshot.box.Ly);
            self.assertEqual(snap.box.Lz, self.snapshot.box.Lz);
            self.assertEqual(snap.box.xy, self.snapshot.box.xy);
            self.assertEqual(snap.box.xz, self.snapshot.box.xz);
            self.assertEqual(snap.box.yz, self.snapshot.box.yz);

            self.assertEqual(snap.particles.N, self.snapshot.particles.N);
            self.assertEqual(snap.particles.types, self.snapshot.particles.types);

            numpy.testing.assert_array_equal(snap.particles.typeid, self.snapshot.particles.typeid);
            numpy.testing.assert_array_equal(snap.particles.mass, self.snapshot.particles.mass);
            numpy.testing.assert_array_equal(snap.particles.charge, self.snapshot.particles.charge);
            numpy.testing.assert_array_equal(snap.particles.diameter, self.snapshot.particles.diameter);
            numpy.testing.assert_array_equal(snap.particles.body, self.snapshot.particles.body);
            numpy.testing.assert_array_equal(snap.particles.moment_inertia, self.snapshot.particles.moment_inertia);
            numpy.testing.assert_array_equal(snap.particles.position, self.snapshot.particles.position);
            numpy.testing.assert_array_equal(snap.particles.orientation, self.snapshot.particles.orientation);
            numpy.testing.assert_array_equal(snap.particles.velocity, self.snapshot.particles.velocity);
            numpy.testing.assert_array_equal(snap.particles.angmom, self.snapshot.particles.angmom);
            numpy.testing.assert_array_equal(snap.particles.image, self.snapshot.particles.image);

            self.assertEqual(snap.bonds.N, self.snapshot.bonds.N);
            self.assertEqual(snap.bonds.types, self.snapshot.bonds.types);
            numpy.testing.assert_array_equal(snap.bonds.typeid, self.snapshot.bonds.typeid);
            numpy.testing.assert_array_equal(snap.bonds.group, self.snapshot.bonds.group);

            self.assertEqual(snap.angles.N, self.snapshot.angles.N);
            self.assertEqual(snap.angles.types, self.snapshot.angles.types);
            numpy.testing.assert_array_equal(snap.angles.typeid, self.snapshot.angles.typeid);
            numpy.testing.assert_array_equal(snap.angles.group, self.snapshot.angles.group);

            self.assertEqual(snap.dihedrals.N, self.snapshot.dihedrals.N);
            self.assertEqual(snap.dihedrals.types, self.snapshot.dihedrals.types);
            numpy.testing.assert_array_equal(snap.dihedrals.typeid, self.snapshot.dihedrals.typeid);
            numpy.testing.assert_array_equal(snap.dihedrals.group, self.snapshot.dihedrals.group);

            self.assertEqual(snap.impropers.N, self.snapshot.impropers.N);
            self.assertEqual(snap.impropers.types, self.snapshot.impropers.types);
            numpy.testing.assert_array_equal(snap.impropers.typeid, self.snapshot.impropers.typeid);
            numpy.testing.assert_array_equal(snap.impropers.group, self.snapshot.impropers.group);

            self.assertEqual(snap.constraints.N, self.snapshot.constraints.N);
            numpy.testing.assert_array_equal(snap.constraints.group, self.snapshot.constraints.group);
            numpy.testing.assert_array_equal(snap.constraints.value, self.snapshot.constraints.value);

            self.assertEqual(snap.pairs.N, self.snapshot.pairs.N);
            self.assertEqual(snap.pairs.types, self.snapshot.pairs.types);
            numpy.testing.assert_array_equal(snap.pairs.typeid, self.snapshot.pairs.typeid);
            numpy.testing.assert_array_equal(snap.pairs.group, self.snapshot.pairs.group);


    # tests init.read_gsd
    def test_read_gsd(self):
        dump.gsd(filename=self.tmp_file, group=group.all(), period=None, overwrite=True);
        context.initialize();

        init.read_gsd(filename=self.tmp_file, frame=-1);

    def tearDown(self):
        if comm.get_rank() == 0:
            os.remove(self.tmp_file);
        comm.barrier_all();

# unit tests for dump.gsd with default type
class gsd_default_type (unittest.TestCase):
    def setUp(self):
        context.initialize()
        if hoomd.comm.get_rank() == 0:
            tmp = tempfile.mkstemp(suffix='.test.gsd');
            self.tmp_file = tmp[1];
        else:
            self.tmp_file = "invalid";

        self.snapshot = data.make_snapshot(N=4, box=data.boxdim(L=10), dtype='float');
        if comm.get_rank() == 0:
            # particles
            self.snapshot.particles.position[0] = [0,1,2];
            self.snapshot.particles.position[1] = [1,2,3];
            self.snapshot.particles.position[2] = [0,-1,-2];
            self.snapshot.particles.position[3] = [-1, -2, -3];
            self.snapshot.particles.velocity[0] = [10, 11, 12];
            self.snapshot.particles.velocity[1] = [11, 12, 13];
            self.snapshot.particles.velocity[2] = [12, 13, 14];
            self.snapshot.particles.velocity[3] = [13, 14, 15];
            self.snapshot.particles.types = ['A'];

        self.s = init.read_snapshot(self.snapshot);
        context.current.sorter.set_params(grid=8)

    # tests data.gsd_snapshot
    def test_gsd(self):
        dump.gsd(filename=self.tmp_file, group=group.all(), period=None, overwrite=True);

        snap = data.gsd_snapshot(self.tmp_file, frame=-1);
        if comm.get_rank() == 0:
            self.assertEqual(snap.particles.N, self.snapshot.particles.N);
            self.assertEqual(snap.particles.types, self.snapshot.particles.types);

    # tests init.read_gsd
    def test_read_gsd(self):
        dump.gsd(filename=self.tmp_file, group=group.all(), period=None, overwrite=True);
        context.initialize();

        init.read_gsd(filename=self.tmp_file);

    def tearDown(self):
        if comm.get_rank() == 0:
            os.remove(self.tmp_file);
        comm.barrier_all();

class gsd_default_type (unittest.TestCase):
    def setUp(self):
        context.initialize();
        if hoomd.comm.get_rank() == 0:
            tmp = tempfile.mkstemp(suffix='.test.gsd');
            self.tmp_file = tmp[1];
        else:
            self.tmp_file = "invalid";

    def validate_append(self, name, default_val, nondefault_val):
        self.snapshot = data.make_snapshot(N=4, box=data.boxdim(L=10), dtype='float');
        if comm.get_rank() == 0:
            # particles
            self.snapshot.particles.types = ['A', 'B', 'C'];
            print(dir(self.snapshot.particles))
            getattr(self.snapshot.particles, name)[:] = nondefault_val

        self.s = init.read_snapshot(self.snapshot);
        context.current.sorter.set_params(grid=8)

        # write out frame 0
        dump.gsd(filename=self.tmp_file, group=group.all(), period=None, overwrite=True);

        # reset values to default and write out the second frame
        if comm.get_rank() == 0:
            getattr(self.snapshot.particles, name)[:] = default_val

        self.s.restore_snapshot(self.snapshot);
        run(1)
        dump.gsd(filename=self.tmp_file, group=group.all(), dynamic=['attribute', 'momentum'], period=None);

        # validate the resulting gsd file
        snap = data.gsd_snapshot(self.tmp_file, frame=0);
        if comm.get_rank() == 0:
            self.assertEqual(snap.particles.N, self.snapshot.particles.N);
            numpy.testing.assert_array_equal(getattr(snap.particles, name), nondefault_val);

        snap = data.gsd_snapshot(self.tmp_file, frame=1);
        if comm.get_rank() == 0:
            self.assertEqual(snap.particles.N, self.snapshot.particles.N);
            numpy.testing.assert_array_equal(getattr(snap.particles, name), default_val);


    def validate_fullwrite(self, name, default_val, nondefault_val):
        self.snapshot = data.make_snapshot(N=4, box=data.boxdim(L=10), dtype='float');
        if comm.get_rank() == 0:
            # particles
            self.snapshot.particles.types = ['A', 'B', 'C'];
            print(dir(self.snapshot.particles))
            getattr(self.snapshot.particles, name)[:] = nondefault_val

        self.s = init.read_snapshot(self.snapshot);
        context.current.sorter.set_params(grid=8)

        # write out frame 0
        dump.gsd(filename=self.tmp_file, group=group.all(), period=1, overwrite=True, dynamic=['attribute', 'momentum']);
        run(1)

        # reset values to default and write out the second frame
        if comm.get_rank() == 0:
            getattr(self.snapshot.particles, name)[:] = default_val

        self.s.restore_snapshot(self.snapshot);
        run(1)

        # validate the resulting gsd file
        snap = data.gsd_snapshot(self.tmp_file, frame=0);
        if comm.get_rank() == 0:
            self.assertEqual(snap.particles.N, self.snapshot.particles.N);
            numpy.testing.assert_array_equal(getattr(snap.particles, name), nondefault_val);

        snap = data.gsd_snapshot(self.tmp_file, frame=1);
        if comm.get_rank() == 0:
            self.assertEqual(snap.particles.N, self.snapshot.particles.N);
            numpy.testing.assert_array_equal(getattr(snap.particles, name), default_val);

    def test_nondefault_typeid(self):
        self.validate_append(name='typeid',
                             default_val = [0, 0, 0, 0],
                             nondefault_val = [2, 1, 0, 2])

    def test_nondefault_typeid2(self):
        self.validate_fullwrite(name='typeid',
                                default_val = [0, 0, 0, 0],
                                nondefault_val = [2, 1, 0, 2])

    def test_nondefault_mass(self):
        self.validate_append(name='mass',
                             default_val = [1, 1, 1, 1],
                             nondefault_val = [3, 2, 1, 3])

    def test_nondefault_mass2(self):
        self.validate_fullwrite(name='mass',
                                default_val = [1, 1, 1, 1],
                                nondefault_val = [3, 2, 1, 3])

    def test_nondefault_charge(self):
        self.validate_append(name='charge',
                             default_val = [0, 0, 0, 0],
                             nondefault_val = [1, -1, 3, -3])

    def test_nondefault_charge2(self):
        self.validate_fullwrite(name='charge',
                                default_val = [0, 0, 0, 0],
                                nondefault_val = [1, -1, 3, -3])

    def test_nondefault_diameter(self):
        self.validate_append(name='diameter',
                             default_val = [1, 1, 1, 1],
                             nondefault_val = [2, 3, 4, 1])

    def test_nondefault_diameter2(self):
        self.validate_fullwrite(name='diameter',
                                default_val = [1, 1, 1, 1],
                                nondefault_val = [2, 3, 4, 1])

    def test_nondefault_body(self):
        self.validate_append(name='body',
                             default_val = [4294967295, 4294967295, 4294967295, 4294967295],
                             nondefault_val = [0, 1, 2, 4294967295])

    def test_nondefault_body2(self):
        self.validate_fullwrite(name='body',
                                default_val = [4294967295, 4294967295, 4294967295, 4294967295],
                                nondefault_val = [0, 1, 2, 4294967295])

    def test_nondefault_moment_inertia(self):
        self.validate_append(name='moment_inertia',
                             default_val = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                             nondefault_val = [[1, 0, 0], [1, 2, 0], [1, 1, 1], [2, 3, 4]])

    def test_nondefault_moment_inertia2(self):
        self.validate_fullwrite(name='moment_inertia',
                                default_val = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                nondefault_val = [[1, 0, 0], [1, 2, 0], [1, 1, 1], [2, 3, 4]])


    def test_nondefault_velocity(self):
        self.validate_append(name='velocity',
                             default_val = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                             nondefault_val = [[1, 0, 0], [1, 2, 0], [1, 1, 1], [2, 3, 4]])

    def test_nondefault_velocity2(self):
        self.validate_fullwrite(name='velocity',
                                default_val = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                nondefault_val = [[1, 0, 0], [1, 2, 0], [1, 1, 1], [2, 3, 4]])

    def test_nondefault_image(self):
        self.validate_append(name='image',
                             default_val = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                             nondefault_val = [[1, 0, 0], [1, 2, 0], [1, 1, 1], [2, 3, 4]])

    def test_nondefault_image2(self):
        self.validate_fullwrite(name='image',
                                default_val = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                nondefault_val = [[1, 0, 0], [1, 2, 0], [1, 1, 1], [2, 3, 4]])

    def test_nondefault_orientation(self):
        self.validate_append(name='orientation',
                             default_val = [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
                             nondefault_val = [[1, 1, 0, 0], [1, 1, 2, 0], [1, 1, 1, 1], [1, 2, 3, 4]])

    def test_nondefault_orientation2(self):
        self.validate_fullwrite(name='orientation',
                                default_val = [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
                                nondefault_val = [[1, 1, 0, 0], [1, 1, 2, 0], [1, 1, 1, 1], [1, 2, 3, 4]])

    def test_nondefault_angmom(self):
        self.validate_append(name='angmom',
                             default_val = [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
                             nondefault_val = [[1, 1, 0, 0], [1, 1, 2, 0], [1, 1, 1, 1], [1, 2, 3, 4]])

    def test_nondefault_angmom2(self):
        self.validate_fullwrite(name='angmom',
                                default_val = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                                nondefault_val = [[1, 1, 0, 0], [1, 1, 2, 0], [1, 1, 1, 1], [1, 2, 3, 4]])

    def tearDown(self):
        if comm.get_rank() == 0:
            os.remove(self.tmp_file);
        comm.barrier_all();


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
