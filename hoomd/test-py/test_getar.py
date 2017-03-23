from __future__ import print_function;
import json;
import os;
import random;
import sys;
import unittest;
import tempfile;
import zipfile;

import numpy as np;
import hoomd;

try:
    import gtar;
except ImportError:
    print('Warning: gtar python module not found, some getar tests not run',
          file=sys.stderr);
    gtar = None;

def get_tmp(suffix):
    if hoomd.comm.get_rank() == 0:
        tmp = tempfile.mkstemp(suffix);
        return tmp[1];
    else:
        return "invalid";

class RandomSystem:
    def __init__(self, seed):
        self.seed = seed;
        self.restoreProps = ['box'];
        random.seed(seed);

        self.nParticles = random.randint(1, 64);
        self.randomizeTypes();
        self.randomizeScalar('position', self.nParticles, 3, .25);
        self.randomizeScalar('velocity', self.nParticles, 3);
        self.randomizeScalar('acceleration', self.nParticles, 3);
        self.randomizeScalar('mass', self.nParticles);
        self.randomizeScalar('charge', self.nParticles);
        self.randomizeScalar('diameter', self.nParticles);
        self.randomizeBonds();
        self.box = [14, 12., 11.2];

    def randomizeTypes(self):
        self.restoreProps.extend(['type', 'type_names.json']);
        self.nTypes = random.randint(1, 4);
        self.types = [random.randint(0, self.nTypes - 1) for _ in range(self.nParticles)];
        self.typeNames = [self.randName() for _ in range(self.nTypes)];

    def randomizeScalar(self, name, N, width=1, thresh=.5):
        if random.random() < thresh:
            setattr(self, name, []);
        else:
            self.restoreProps.append(name);
            setattr(self, name,
                    [(tuple(random.random() for _ in range(width)) if width > 1
                      else random.random())
                    for _ in range(N)]);

    def randName(self):
        size = random.randint(1, 8);
        return ''.join(random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for _ in range(size));

    def randomizeBonds(self):
        self.nBondTypes = random.randint(0, 4);

        if self.nBondTypes:
            self.nBonds = random.randint(0, 40);
            self.bonds = [(random.randint(0, self.nBondTypes - 1),
                           random.randint(0, self.nParticles - 1),
                           random.randint(0, self.nParticles - 1)) for _ in range(self.nBonds)];
            self.bondNames = [self.randName() for _ in range(self.nBondTypes)];

        if not self.nBondTypes or any(i == j for (_, i, j) in self.bonds):
            self.nBonds = 0;
            self.bonds = [];
            self.bondNames = [];
        else:
            self.restoreProps.extend(['bond_type_names', 'bond_type', 'bond_tag']);

    def writeXML(self, filename='test.xml'):
        with open(filename, 'w') as inpFile:
            inpFile.write('\n'.join(['<?xml version="1.1" encoding="UTF-8"?>',
                                     '<hoomd_xml version="{}">'.format('1.6'),
                                     '<configuration time_step = "0" dimensions = "3">']));
            inpFile.write('<box lx="{x}" ly="{y}" lz={z} />\n'.format(
                x=self.box[0], y=self.box[1], z=self.box[2]));
            inpFile.write('<position>\n' + '\n'.join(
                '{} {} {}'.format(*r) for r in self.position) +
                '</position>\n');
            inpFile.write('<velocity>\n' + '\n'.join(
                '{} {} {}'.format(*r) for r in self.velocity) +
                '</velocity>\n');
            inpFile.write('<acceleration>\n' + '\n'.join(
                '{} {} {}'.format(*r) for r in self.acceleration) +
                '</acceleration>\n');
            inpFile.write('<type>\n' + '\n'.join(self.typeNames[typ] for typ in self.types) + '</type>\n');
            inpFile.write('<mass>\n' + '\n'.join(map(str, self.mass)) + '</mass>\n');
            inpFile.write('<charge>\n' + '\n'.join(map(str, self.charge)) + '</charge>\n');
            inpFile.write('<diameter>\n' + '\n'.join(map(str, self.diameter)) + '</diameter>\n');
            inpFile.write('\n</configuration>\n</hoomd_xml>\n');

    def writeGetar(self, filename='test.zip'):
        with gtar.GTAR(filename, 'w') as traj:
            traj.writePath('type_names.json', json.dumps(self.typeNames));
            traj.writePath('type.u32.ind', self.types);

            for name in ['position', 'velocity', 'acceleration', 'mass', 'charge', 'diameter']:
                if getattr(self, name):
                    traj.writePath('{}.f32.ind'.format(name), getattr(self, name));
            traj.writePath('box.f32.uni', self.box);

            if self.bonds:
                traj.writePath('bond/type_names.json', json.dumps(self.bondNames));
                traj.writePath('bond/type.u32.ind', [t for (t, _, _) in self.bonds]);
                traj.writePath('bond/tag.u32.ind', [(l, r) for (_, l, r) in self.bonds]);

    def readSystem(self, system):
        self.types = [p.typeid for p in system.particles];
        typeNames = system.particles.types;
        self.typeNames = [typeNames[i] for i in range(len(typeNames))];
        self.nTypes = len(self.typeNames);

        self.position = [p.position for p in system.particles];
        self.velocity = [p.velocity for p in system.particles];
        self.acceleration = [p.acceleration for p in system.particles];
        self.mass = [p.mass for p in system.particles];
        self.charge = [p.charge for p in system.particles];
        self.diameter = [p.diameter for p in system.particles];

        box = system.box;
        self.box = [box.Lx, box.Ly, box.Lz];

        del box;

    def __eq__(self, other):
        result = True;
        for name in ['types', 'typeNames', 'nTypes']:
            result = result and getattr(self, name) == getattr(other, name);

            if not result:
                print('Seed {}: error with {}:'.format(self.seed, name));
                print(getattr(self, name));
                print(getattr(other, name));
                return False;

        for name in ['position', 'velocity', 'acceleration', 'mass',
                     'charge', 'diameter', 'box']:
            result = (not getattr(self, name) or not getattr(other, name) or
                      result and np.allclose(getattr(self, name), getattr(other, name)));

            if not result:
                print('Seed {}: error with {}:'.format(self.seed, name));
                print(getattr(self, name));
                print(getattr(other, name));
                return False;

        return result;

# skip this test if the gtar python module is not available
if gtar is not None:
    class test_random_read_write(unittest.TestCase):
        def test_zip(self):
            tmp_file = get_tmp(suffix='dump.zip');
            self._test_procedure(tmp_file);

        def test_tar(self):
            tmp_file = get_tmp(suffix='dump.tar');
            self._test_procedure(tmp_file);

        def test_sqlite(self):
            tmp_file = get_tmp(suffix='dump.sqlite');
            self._test_procedure(tmp_file);

        def _test_procedure(self, fname):
            random.seed();
            self.last_fname = fname;
            hoomd.util.quiet_status();

            for _ in range(10):
                with hoomd.context.initialize():
                    seed = random.randint(1, 2**32 - 1);
                    sys1 = RandomSystem(seed);
                    if hoomd.comm.get_rank() == 0:
                        sys1.writeGetar(fname);

                    sys2 = RandomSystem(0);
                    hoomd.comm.barrier_all();
                    system = hoomd.init.read_getar(fname,
                                                   {'any': 'any'});
                    sys2.readSystem(system);
                    del system;

                    if hoomd.comm.get_rank() == 0:
                        self.assertEqual(sys1, sys2);

        def setUp(self):
            hoomd.context.initialize();

        def tearDown(self):
            if hoomd.comm.get_rank() == 0:
                os.remove(self.last_fname);
            hoomd.comm.barrier_all();

class test_basic_io(unittest.TestCase):
    def test_basic(self):
        N = 10;
        box = hoomd.data.boxdim(20*N, 40*N, 60*N);
        snap = hoomd.data.make_snapshot(N, box);
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[:] = [(i, 2*i, 3*i) for i in range(N)];
        hoomd.init.read_snapshot(snap);

        vel_prop = hoomd.dump.getar.DumpProp(
            'velocity', True, hoomd.dump.getar.Compression.NoCompress);
        for suffix in ['zip', 'tar', 'sqlite']:
            fname_suffix = 'dump.{}'.format(suffix);
            tmp_file = get_tmp(suffix=fname_suffix);

            hoomd.dump.getar.immediate(tmp_file, static=['type'],
                                       dynamic=['position', vel_prop]);

            hoomd.comm.barrier_all();

            hoomd.init.restore_getar(tmp_file);

    def test_simple(self):
        N = 10;
        box = hoomd.data.boxdim(20*N, 40*N, 60*N);
        snap = hoomd.data.make_snapshot(N, box);
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[:] = [(i, 2*i, 3*i) for i in range(N)];
        hoomd.init.read_snapshot(snap);

        for suffix in ['zip', 'tar', 'sqlite']:
            fname_suffix = 'dump.{}'.format(suffix);
            tmp_file = get_tmp(suffix=fname_suffix);
            dump = hoomd.dump.getar.simple(tmp_file, mode='w', period=1e3,
                                           static=['viz_static'],
                                           dynamic=['viz_dynamic']);
            hoomd.run(1);
            dump.close();
            dump.disable();
            hoomd.comm.barrier_all();

            hoomd.init.restore_getar(tmp_file);

    def test_periodic(self):
        N = 10;
        box = hoomd.data.boxdim(20*N, 40*N, 60*N);
        snap = hoomd.data.make_snapshot(N, box);
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[:] = [(i, 2*i, 3*i) for i in range(N)];
        hoomd.init.read_snapshot(snap);

        for suffix in ['zip', 'tar', 'sqlite']:
            fname_suffix = 'dump.{}'.format(suffix);
            tmp_file = get_tmp(suffix=fname_suffix);

            dump = hoomd.dump.getar(tmp_file, mode='w', static=['viz_static'],
                                    dynamic={'viz_aniso_dynamic': 1e3});
            hoomd.run(1);
            dump.close();
            dump.disable();
            hoomd.comm.barrier_all();

            hoomd.init.restore_getar(tmp_file);

    def test_write_json(self):
        N = 10;
        box = hoomd.data.boxdim(20*N, 40*N, 60*N);
        snap = hoomd.data.make_snapshot(N, box);
        if hoomd.comm.get_rank() == 0:
            snap.particles.position[:] = [(i, 2*i, 3*i) for i in range(N)];
        hoomd.init.read_snapshot(snap);

        for suffix in ['zip', 'tar', 'sqlite']:
            fname_suffix = 'dump.{}'.format(suffix);
            tmp_file = get_tmp(suffix=fname_suffix);

            dump = hoomd.dump.getar(tmp_file, mode='w', static=['viz_static'],
                                    dynamic={'viz_aniso_dynamic': 1e3});

            dump.writeJSON('test.json', dict(testQuantity=True), False)

            hoomd.run(1);

            dump.writeJSON('test.json', dict(testQuantity=hoomd.comm.get_rank()), True)

            dump.close();
            dump.disable();
            hoomd.comm.barrier_all();

            hoomd.init.restore_getar(tmp_file);

            if hoomd.comm.get_rank() == 0:
                if suffix == 'zip':
                    traj = zipfile.ZipFile(tmp_file, 'r')
                    json_result = traj.read('frames/1/test.json').decode()
                    # only rank 0 should have written
                    self.assertEqual(json.loads(json_result)['testQuantity'], 0)

    def setUp(self):
        hoomd.context.initialize();

    def tearDown(self):
        hoomd.comm.barrier_all();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v']);
