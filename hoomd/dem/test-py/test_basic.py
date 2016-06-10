# Copyright (c) 2009-2016 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

import hoomd;
import hoomd.dem;
hoomd.context.initialize();

import unittest;

def not_on_mpi(f):
    def noop(*args, **kwargs):
        return;
    if hoomd.comm.get_num_ranks() > 1:
        return noop;
    else:
        return f;

class basic(unittest.TestCase):

    def test_set_params_wca_2d(self):
        self._test_set_params(hoomd.dem.pair.WCA, twoD=True, radius=.5);

    @not_on_mpi
    def test_set_params_swca_2d(self):
        self._test_set_params(hoomd.dem.pair.SWCA, twoD=True, radius=.75);

    def test_set_params_wca_3d(self):
        self._test_set_params(hoomd.dem.pair.WCA, twoD=False, radius=.5);

    @not_on_mpi
    def test_set_params_swca_3d(self):
        self._test_set_params(hoomd.dem.pair.SWCA, twoD=False, radius=.75);

    def _test_set_params(self, typ, twoD, **params):
        box = hoomd.data.boxdim(10, dimensions=(2 if twoD else 3));
        snap = hoomd.data.make_snapshot(N=4, box=box);
        system = hoomd.init.read_snapshot(snap);
        nl = hoomd.md.nlist.cell();

        potential = typ(nlist=nl, **params);

        if twoD:
            potential.setParams('A', [[1, 0], [0, 1], [-1, -1]], center=False);
        else:
            potential.setParams('A', [[1, 0, 0], [0, 1, 0], [-1, -1, 0]], [[0, 1, 2]],
                                center=False);

        potential.disable();

    def setUp(self):
        hoomd.context.initialize();

    def tearDown(self):
        hoomd.comm.barrier();

if __name__ == '__main__':
    unittest.main(argv = ['test_basic.py', '-v']);
