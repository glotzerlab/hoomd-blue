
# -*- coding: iso-8859-1 -*-
# Maintainer: mphoward

from hoomd import *
import hoomd;
context.initialize()
import unittest

## Domain decomposition balancing tests
class decomposition_tests (unittest.TestCase):
    ## Test that no errors are raised if a uniform decomposition should be done
    def test_uniform(self):
        if comm.get_num_ranks() > 1:
            box = data.boxdim(L=10)
            boxdim = box._getBoxDim()

            # test the default constructor, which should make a uniform decomposition
            comm.decomposition()
            dd = hoomd.context.current.decomposition._make_cpp_decomposition(boxdim)

            self.assertEquals(len(dd.getCumulativeFractions(0)), 3)
            self.assertEquals(len(dd.getCumulativeFractions(1)), 3)
            self.assertEquals(len(dd.getCumulativeFractions(2)), 3)

            self.assertAlmostEquals(dd.getCumulativeFractions(0)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[1], 0.5)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[2], 1.0)

            self.assertAlmostEquals(dd.getCumulativeFractions(1)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[1], 0.5)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[2], 1.0)

            self.assertAlmostEquals(dd.getCumulativeFractions(2)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[1], 0.5)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[2], 1.0)

            # explicitly set the grid dimensions in the constructor
            comm.decomposition(nx=2, ny=2, nz=2)
            dd = hoomd.context.current.decomposition._make_cpp_decomposition(boxdim)

            self.assertEquals(len(dd.getCumulativeFractions(0)), 3)
            self.assertEquals(len(dd.getCumulativeFractions(1)), 3)
            self.assertEquals(len(dd.getCumulativeFractions(2)), 3)

            self.assertAlmostEquals(dd.getCumulativeFractions(0)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[1], 0.5)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[2], 1.0)

            self.assertAlmostEquals(dd.getCumulativeFractions(1)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[1], 0.5)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[2], 1.0)

            self.assertAlmostEquals(dd.getCumulativeFractions(2)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[1], 0.5)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[2], 1.0)

            # shuffle dimensions
            comm.decomposition(nx=2, ny=4, nz=1)
            dd = hoomd.context.current.decomposition._make_cpp_decomposition(boxdim)

            self.assertEquals(len(dd.getCumulativeFractions(0)), 3)
            self.assertEquals(len(dd.getCumulativeFractions(1)), 5)
            self.assertEquals(len(dd.getCumulativeFractions(2)), 2)

            self.assertAlmostEquals(dd.getCumulativeFractions(0)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[1], 0.5)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[2], 1.0)

            self.assertAlmostEquals(dd.getCumulativeFractions(1)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[1], 0.25)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[2], 0.5)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[3], 0.75)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[4], 1.0)

            self.assertAlmostEquals(dd.getCumulativeFractions(2)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[1], 1.0)

            # shuffle dimensions
            comm.decomposition(nx=4, ny=1, nz=2)
            dd = hoomd.context.current.decomposition._make_cpp_decomposition(boxdim)

            self.assertEquals(len(dd.getCumulativeFractions(0)), 5)
            self.assertEquals(len(dd.getCumulativeFractions(1)), 2)
            self.assertEquals(len(dd.getCumulativeFractions(2)), 3)

            self.assertAlmostEquals(dd.getCumulativeFractions(0)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[1], 0.25)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[2], 0.5)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[3], 0.75)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[4], 1.0)

            self.assertAlmostEquals(dd.getCumulativeFractions(1)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[1], 1.0)

            self.assertAlmostEquals(dd.getCumulativeFractions(2)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[1], 0.5)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[2], 1.0)

            # shuffle dimensions
            comm.decomposition(nx=1, ny=2, nz=4)
            dd = hoomd.context.current.decomposition._make_cpp_decomposition(boxdim)

            self.assertEquals(len(dd.getCumulativeFractions(0)), 2)
            self.assertEquals(len(dd.getCumulativeFractions(1)), 3)
            self.assertEquals(len(dd.getCumulativeFractions(2)), 5)

            self.assertAlmostEquals(dd.getCumulativeFractions(0)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[1], 1.0)

            self.assertAlmostEquals(dd.getCumulativeFractions(1)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[1], 0.5)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[2], 1.0)

            self.assertAlmostEquals(dd.getCumulativeFractions(2)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[1], 0.25)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[2], 0.5)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[3], 0.75)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[4], 1.0)

    ## Test the fractions are set correctly
    def test_basic_balance(self):
        if comm.get_num_ranks() == 1:
            comm.decomposition(x=0.5)
            self.assertEqual(hoomd.context.current.decomposition, None)
        elif comm.get_num_ranks() > 1:
            box = data.boxdim(L=10)
            boxdim = box._getBoxDim()
            comm.decomposition(z=0.2)
            dd = hoomd.context.current.decomposition._make_cpp_decomposition(boxdim)

            self.assertEquals(len(dd.getCumulativeFractions(0)), 3)
            self.assertEquals(len(dd.getCumulativeFractions(1)), 3)
            self.assertEquals(len(dd.getCumulativeFractions(2)), 3)

            self.assertAlmostEquals(dd.getCumulativeFractions(0)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[1], 0.5)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[2], 1.0)

            self.assertAlmostEquals(dd.getCumulativeFractions(1)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[1], 0.5)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[2], 1.0)

            self.assertAlmostEquals(dd.getCumulativeFractions(2)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[1], 0.2)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[2], 1.0)

            comm.decomposition(x=[0.2,0.3,0.1], y=0.3)
            dd = hoomd.context.current.decomposition._make_cpp_decomposition(boxdim)
            self.assertEquals(len(dd.getCumulativeFractions(0)), 5)
            self.assertEquals(len(dd.getCumulativeFractions(1)), 3)
            self.assertEquals(len(dd.getCumulativeFractions(2)), 2)

            self.assertAlmostEquals(dd.getCumulativeFractions(0)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[1], 0.2)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[2], 0.5)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[3], 0.6)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[4], 1.0)

            self.assertAlmostEquals(dd.getCumulativeFractions(1)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[1], 0.3)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[2], 1.0)

            self.assertAlmostEquals(dd.getCumulativeFractions(2)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[1], 1.0)

    ## Test that script gracefully chooses among available options
    def test_overspecify(self):
        if comm.get_num_ranks() > 1:
            box = data.boxdim(L=10)
            boxdim = box._getBoxDim()

            # this is wrong to set both, so it should fail
            with self.assertRaises(RuntimeError):
                comm.decomposition(x=[0.2,0.3,0.1], nx=2)

            # set a global value, and try with nx arg set
            hoomd.context.options.nx = 8
            comm.decomposition(nx=4)
            dd = hoomd.context.current.decomposition._make_cpp_decomposition(boxdim)

            self.assertEquals(len(dd.getCumulativeFractions(0)), 5)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[1], 0.25)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[2], 0.5)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[3], 0.75)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[4], 1.0)

            # now fallback to the global one
            comm.decomposition()
            dd = hoomd.context.current.decomposition._make_cpp_decomposition(boxdim)

            self.assertEquals(len(dd.getCumulativeFractions(0)), 9)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[1], 0.125)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[2], 0.25)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[3], 0.375)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[4], 0.5)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[5], 0.625)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[6], 0.75)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[7], 0.875)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[8], 1.0)
            hoomd.context.options.nx = None # undo this so that it doesn't contaminate other dimensions

            # this is wrong to set both, so it should fail
            with self.assertRaises(RuntimeError):
                comm.decomposition(y=[0.2,0.3,0.1], ny=2)

            # set a global value, and try with ny arg set
            hoomd.context.options.ny = 8
            comm.decomposition(ny=4)
            dd = hoomd.context.current.decomposition._make_cpp_decomposition(boxdim)

            self.assertEquals(len(dd.getCumulativeFractions(1)), 5)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[1], 0.25)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[2], 0.5)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[3], 0.75)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[4], 1.0)

            # now fallback to the global one
            comm.decomposition()
            dd = hoomd.context.current.decomposition._make_cpp_decomposition(boxdim)

            self.assertEquals(len(dd.getCumulativeFractions(1)), 9)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[1], 0.125)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[2], 0.25)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[3], 0.375)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[4], 0.5)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[5], 0.625)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[6], 0.75)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[7], 0.875)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[8], 1.0)
            hoomd.context.options.ny = None # undo this so that it doesn't contaminate other dimensions

            # this is wrong to set both, so it should fail
            with self.assertRaises(RuntimeError):
                comm.decomposition(z=[0.2,0.3,0.1], nz=2)

            # set a global value, and try with nz arg set
            hoomd.context.options.nz = 8
            comm.decomposition(nz=4)
            dd = hoomd.context.current.decomposition._make_cpp_decomposition(boxdim)

            self.assertEquals(len(dd.getCumulativeFractions(2)), 5)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[1], 0.25)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[2], 0.5)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[3], 0.75)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[4], 1.0)

            # now fallback to the global one
            comm.decomposition()
            dd = hoomd.context.current.decomposition._make_cpp_decomposition(boxdim)

            self.assertEquals(len(dd.getCumulativeFractions(2)), 9)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[1], 0.125)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[2], 0.25)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[3], 0.375)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[4], 0.5)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[5], 0.625)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[6], 0.75)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[7], 0.875)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[8], 1.0)

            # the linear command should take precedence over the z command
            hoomd.context.options.nz = 4
            hoomd.context.options.linear = True
            comm.decomposition()
            dd = hoomd.context.current.decomposition._make_cpp_decomposition(boxdim)

            self.assertEquals(len(dd.getCumulativeFractions(2)), 9)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[1], 0.125)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[2], 0.25)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[3], 0.375)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[4], 0.5)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[5], 0.625)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[6], 0.75)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[7], 0.875)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[8], 1.0)

            # clear out these options so they don't contaminate other tests
            hoomd.context.options.nz = None
            hoomd.context.options.linear = None

    ## Test that balancing fails after initialization
    def test_wrong_order(self):
        init.create_lattice(lattice.sc(a=2.1878096788957757),n=[5,5,4]); #target a packing fraction of 0.05
        with self.assertRaises(RuntimeError):
            comm.decomposition(y=0.3)
        context.initialize()

    ## Test that errors are raised if fractional divisions exceed 1.0
    def test_bad_fractions(self):
        if comm.get_num_ranks() > 1:
            box = data.boxdim(L=10)
            boxdim = box._getBoxDim()
            with self.assertRaises(RuntimeError):
                comm.decomposition(x=-0.2)
                hoomd.context.current.decomposition._make_cpp_decomposition(boxdim)

            with self.assertRaises(RuntimeError):
                comm.decomposition(x=1.2)
                hoomd.context.current.decomposition._make_cpp_decomposition(boxdim)

            with self.assertRaises(RuntimeError):
                comm.decomposition(x=[0.2,0.9])
                hoomd.context.current.decomposition._make_cpp_decomposition(boxdim)

            with self.assertRaises(RuntimeError):
                comm.decomposition(x=[0.3,-0.1])
                hoomd.context.current.decomposition._make_cpp_decomposition(boxdim)

            with self.assertRaises(RuntimeError):
                comm.decomposition(y=-0.2)
                hoomd.context.current.decomposition._make_cpp_decomposition(boxdim)

            with self.assertRaises(RuntimeError):
                comm.decomposition(y=1.2)
                hoomd.context.current.decomposition._make_cpp_decomposition(boxdim)

            with self.assertRaises(RuntimeError):
                comm.decomposition(y=[0.2,0.9])
                hoomd.context.current.decomposition._make_cpp_decomposition(boxdim)

            with self.assertRaises(RuntimeError):
                comm.decomposition(y=[0.3,-0.1])
                hoomd.context.current.decomposition._make_cpp_decomposition(boxdim)

            with self.assertRaises(RuntimeError):
                comm.decomposition(z=-0.2)
                hoomd.context.current.decomposition._make_cpp_decomposition(boxdim)

            with self.assertRaises(RuntimeError):
                comm.decomposition(z=1.2)
                hoomd.context.current.decomposition._make_cpp_decomposition(boxdim)

            with self.assertRaises(RuntimeError):
                comm.decomposition(z=[0.2,0.9])
                hoomd.context.current.decomposition._make_cpp_decomposition(boxdim)

            with self.assertRaises(RuntimeError):
                comm.decomposition(z=[0.3,-0.1])
                hoomd.context.current.decomposition._make_cpp_decomposition(boxdim)

    ## Test that parameters are set correctly
    def test_set_params(self):
        if comm.get_num_ranks() > 1:
            dd = comm.decomposition(x=[0.3],y=[0.4],z=[0.6])

            # check that the grid was set correctly in the constructor (via set_params)
            self.assertFalse(hoomd.context.current.decomposition.uniform_x)
            self.assertFalse(hoomd.context.current.decomposition.uniform_y)
            self.assertFalse(hoomd.context.current.decomposition.uniform_z)
            self.assertAlmostEqual(hoomd.context.current.decomposition.x[0], 0.3)
            self.assertAlmostEqual(hoomd.context.current.decomposition.y[0], 0.4)
            self.assertAlmostEqual(hoomd.context.current.decomposition.z[0], 0.6)

            # switch everything to a uniform grid (doesn't matter that it is infeasible, we aren't actually constructing it)
            dd.set_params(nx=4)
            dd.set_params(ny=5)
            dd.set_params(nz=6)
            self.assertTrue(hoomd.context.current.decomposition.uniform_x)
            self.assertTrue(hoomd.context.current.decomposition.uniform_y)
            self.assertTrue(hoomd.context.current.decomposition.uniform_z)
            self.assertEqual(hoomd.context.current.decomposition.nx, 4)
            self.assertEqual(hoomd.context.current.decomposition.ny, 5)
            self.assertEqual(hoomd.context.current.decomposition.nz, 6)

            # do it all in one function call to make sure this works
            dd.set_params(nx=2, ny=3, nz=4)
            self.assertTrue(hoomd.context.current.decomposition.uniform_x)
            self.assertTrue(hoomd.context.current.decomposition.uniform_y)
            self.assertTrue(hoomd.context.current.decomposition.uniform_z)
            self.assertEqual(hoomd.context.current.decomposition.nx, 2)
            self.assertEqual(hoomd.context.current.decomposition.ny, 3)
            self.assertEqual(hoomd.context.current.decomposition.nz, 4)

            # now back to a new non-uniform spacing
            dd.set_params(x=0.6)
            dd.set_params(y=0.4)
            dd.set_params(z=0.3)
            self.assertFalse(hoomd.context.current.decomposition.uniform_x)
            self.assertFalse(hoomd.context.current.decomposition.uniform_y)
            self.assertFalse(hoomd.context.current.decomposition.uniform_z)
            self.assertAlmostEqual(hoomd.context.current.decomposition.x[0], 0.6)
            self.assertAlmostEqual(hoomd.context.current.decomposition.y[0], 0.4)
            self.assertAlmostEqual(hoomd.context.current.decomposition.z[0], 0.3)

            # do it all at once
            dd.set_params(x=[0.2,0.3], y=[0.4,0.1], z=[0.25,0.25])
            self.assertFalse(hoomd.context.current.decomposition.uniform_x)
            self.assertFalse(hoomd.context.current.decomposition.uniform_y)
            self.assertFalse(hoomd.context.current.decomposition.uniform_z)
            self.assertAlmostEqual(hoomd.context.current.decomposition.x[0], 0.2)
            self.assertAlmostEqual(hoomd.context.current.decomposition.x[1], 0.3)
            self.assertAlmostEqual(hoomd.context.current.decomposition.y[0], 0.4)
            self.assertAlmostEqual(hoomd.context.current.decomposition.y[1], 0.1)
            self.assertAlmostEqual(hoomd.context.current.decomposition.z[0], 0.25)
            self.assertAlmostEqual(hoomd.context.current.decomposition.z[1], 0.25)

            # try a mixture of things
            dd.set_params(nx=3, y=0.8, nz=2)
            self.assertTrue(hoomd.context.current.decomposition.uniform_x)
            self.assertFalse(hoomd.context.current.decomposition.uniform_y)
            self.assertTrue(hoomd.context.current.decomposition.uniform_z)
            self.assertEqual(hoomd.context.current.decomposition.nx, 3)
            self.assertAlmostEqual(hoomd.context.current.decomposition.y[0], 0.8)
            self.assertEqual(hoomd.context.current.decomposition.nz, 2)

            with self.assertRaises(RuntimeError):
                dd.set_params(x=0.2, nx=4)
            with self.assertRaises(RuntimeError):
                dd.set_params(y=0.2, ny=4)
            with self.assertRaises(RuntimeError):
                dd.set_params(z=0.2, nz=4)

## Test for MPI barriers
class barrier_tests(unittest.TestCase):
    def test_barrier(self):
        comm.barrier();

    def test_barrier_all(self):
        comm.barrier_all();

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
