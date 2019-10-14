from hoomd import *
import hoomd
import unittest
import os
import tempfile
import numpy as np
from hoomd import _hoomd, md
import json

def parse_shape_spec(type_shapes):
    ret = [ json.loads(json_string) for json_string in type_shapes ];
    return ret;

class md_gsd_shape_spec(unittest.TestCase):

    def setUp(self):
        hoomd.context.initialize()
        if hoomd.comm.get_rank() == 0:
            tmp = tempfile.mkstemp(suffix='.test.gsd');
            self.tmp_file = tmp[1];
        else:
            self.tmp_file = "invalid";

    def setup_system(self, cls, shape_params, expected_shapespec, filename, dim):
        if dim == 2:
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sq(a=5.50),n=5);
        elif dim == 3:
            system = hoomd.init.create_lattice(unitcell=hoomd.lattice.sc(a=5.50),n=5);
        snapshot = system.take_snapshot(all=True)
        bindex = np.random.choice(range(5**dim),int(0.5*5**dim),replace=False)
        if comm.get_rank() == 0:
            snapshot.particles.types = ['A', 'B']
            snapshot.particles.typeid[bindex] = 1
        hoomd.context.initialize()
        system = hoomd.init.read_snapshot(snapshot)
        md.integrate.mode_standard(dt=0.001);
        obj = cls(r_cut=3, nlist=md.nlist.cell())
        obj.pair_coeff.set('A', 'A', **shape_params['A'])
        obj.pair_coeff.set('B', 'B', **shape_params['B'])
        obj.pair_coeff.set('A', 'B', **shape_params['B'])
        md.integrate.nvt(group=group.all(), kT=1.0, tau=0.5)
        dumper = dump.gsd(filename=filename, group=group.all(), period=1, overwrite=True);
        dumper.dump_shape(obj);
        steps = 5
        hoomd.run(steps);
        reader = _hoomd.GSDReader(hoomd.context.exec_conf, filename, 0, False);
        if comm.get_rank() == 0:
            for i in range(steps):
                shape_spec = parse_shape_spec(reader.readTypeShapesPy(i));
                self.assertEqual(shape_spec[0], expected_shapespec[0]);
                self.assertEqual(shape_spec[1], expected_shapespec[1]);

    def test_gay_berne(self):
        shape_params = dict(A=dict(epsilon=1, lperp=0.5, lpar=1), \
                            B=dict(epsilon=2, lperp=0.5, lpar=1));
        expected_shapespec = [dict(type='Ellipsoid', a=shape_params['A']['lperp'], \
                                   b=shape_params['A']['lperp'], \
                                   c=shape_params['A']['lpar']), \
                              dict(type='Ellipsoid', a=shape_params['B']['lperp'], \
                                   b=shape_params['B']['lperp'], \
                                   c=shape_params['B']['lpar']) ];
        self.setup_system(cls=md.pair.gb, shape_params=shape_params, \
                          expected_shapespec=expected_shapespec, filename=self.tmp_file, dim=3);
    def tearDown(self):
        if comm.get_rank() == 0:
            os.remove(self.tmp_file);
        comm.barrier_all();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
