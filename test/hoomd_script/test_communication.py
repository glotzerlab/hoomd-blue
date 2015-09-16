
# -*- coding: iso-8859-1 -*-
# Maintainer: mphoward

from hoomd_script import *
context.initialize()
import hoomd
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
            dd = globals.decomposition._make_cpp_decomposition(boxdim)

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
            dd = globals.decomposition._make_cpp_decomposition(boxdim)

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
            dd = globals.decomposition._make_cpp_decomposition(boxdim)

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
            dd = globals.decomposition._make_cpp_decomposition(boxdim)

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
            dd = globals.decomposition._make_cpp_decomposition(boxdim)

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
            self.assertEqual(globals.decomposition, None)
        elif comm.get_num_ranks() > 1:
            box = data.boxdim(L=10)
            boxdim = box._getBoxDim()
            comm.decomposition(z=0.2)
            dd = globals.decomposition._make_cpp_decomposition(boxdim)

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
            dd = globals.decomposition._make_cpp_decomposition(boxdim)
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
            globals.options.nx = 8
            comm.decomposition(nx=4)
            dd = globals.decomposition._make_cpp_decomposition(boxdim)

            self.assertEquals(len(dd.getCumulativeFractions(0)), 5)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[1], 0.25)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[2], 0.5)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[3], 0.75)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[4], 1.0)

            # now fallback to the global one
            comm.decomposition()
            dd = globals.decomposition._make_cpp_decomposition(boxdim)

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
            globals.options.nx = None # undo this so that it doesn't contaminate other dimensions

            # this is wrong to set both, so it should fail
            with self.assertRaises(RuntimeError):
                comm.decomposition(y=[0.2,0.3,0.1], ny=2)

            # set a global value, and try with ny arg set
            globals.options.ny = 8
            comm.decomposition(ny=4)
            dd = globals.decomposition._make_cpp_decomposition(boxdim)

            self.assertEquals(len(dd.getCumulativeFractions(1)), 5)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[1], 0.25)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[2], 0.5)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[3], 0.75)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[4], 1.0)

            # now fallback to the global one
            comm.decomposition()
            dd = globals.decomposition._make_cpp_decomposition(boxdim)

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
            globals.options.ny = None # undo this so that it doesn't contaminate other dimensions

            # this is wrong to set both, so it should fail
            with self.assertRaises(RuntimeError):
                comm.decomposition(z=[0.2,0.3,0.1], nz=2)

            # set a global value, and try with nz arg set
            globals.options.nz = 8
            comm.decomposition(nz=4)
            dd = globals.decomposition._make_cpp_decomposition(boxdim)

            self.assertEquals(len(dd.getCumulativeFractions(2)), 5)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[1], 0.25)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[2], 0.5)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[3], 0.75)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[4], 1.0)

            # now fallback to the global one
            comm.decomposition()
            dd = globals.decomposition._make_cpp_decomposition(boxdim)

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
            globals.options.nz = 4
            globals.options.linear = True
            comm.decomposition()
            dd = globals.decomposition._make_cpp_decomposition(boxdim)

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
            globals.options.nz = None
            globals.options.linear = None

    ## Test that balancing fails after initialization
    def test_wrong_order(self):
        init.create_random(N=100, phi_p=0.05)        
        with self.assertRaises(RuntimeError):
            comm.decomposition(y=0.3)
        init.reset()

    ## Test that errors are raised if fractional divisions exceed 1.0
    def test_bad_fractions(self):
        if comm.get_num_ranks() > 1:
            box = data.boxdim(L=10)
            boxdim = box._getBoxDim()
            with self.assertRaises(RuntimeError):
                comm.decomposition(x=-0.2)
                globals.decomposition._make_cpp_decomposition(boxdim)

            with self.assertRaises(RuntimeError):
                comm.decomposition(x=1.2)
                globals.decomposition._make_cpp_decomposition(boxdim)

            with self.assertRaises(RuntimeError):
                comm.decomposition(x=[0.2,0.9])
                globals.decomposition._make_cpp_decomposition(boxdim)

            with self.assertRaises(RuntimeError):
                comm.decomposition(x=[0.3,-0.1])
                globals.decomposition._make_cpp_decomposition(boxdim)

            with self.assertRaises(RuntimeError):
                comm.decomposition(y=-0.2)
                globals.decomposition._make_cpp_decomposition(boxdim)

            with self.assertRaises(RuntimeError):
                comm.decomposition(y=1.2)
                globals.decomposition._make_cpp_decomposition(boxdim)

            with self.assertRaises(RuntimeError):
                comm.decomposition(y=[0.2,0.9])
                globals.decomposition._make_cpp_decomposition(boxdim)

            with self.assertRaises(RuntimeError):
                comm.decomposition(y=[0.3,-0.1])
                globals.decomposition._make_cpp_decomposition(boxdim)

            with self.assertRaises(RuntimeError):
                comm.decomposition(z=-0.2)
                globals.decomposition._make_cpp_decomposition(boxdim)

            with self.assertRaises(RuntimeError):
                comm.decomposition(z=1.2)
                globals.decomposition._make_cpp_decomposition(boxdim)

            with self.assertRaises(RuntimeError):
                comm.decomposition(z=[0.2,0.9])
                globals.decomposition._make_cpp_decomposition(boxdim)

            with self.assertRaises(RuntimeError):
                comm.decomposition(z=[0.3,-0.1])
                globals.decomposition._make_cpp_decomposition(boxdim)

    ## Test that parameters are set correctly
    def test_set_params(self):
        if comm.get_num_ranks() > 1:
            dd = comm.decomposition(x=[0.3],y=[0.4],z=[0.6])

            # check that the grid was set correctly in the constructor (via set_params)
            self.assertFalse(globals.decomposition.uniform_x)
            self.assertFalse(globals.decomposition.uniform_y)
            self.assertFalse(globals.decomposition.uniform_z)
            self.assertAlmostEqual(globals.decomposition.x[0], 0.3)
            self.assertAlmostEqual(globals.decomposition.y[0], 0.4)
            self.assertAlmostEqual(globals.decomposition.z[0], 0.6)

            # switch everything to a uniform grid (doesn't matter that it is infeasible, we aren't actually constructing it)
            dd.set_params(nx=4)
            dd.set_params(ny=5)
            dd.set_params(nz=6)
            self.assertTrue(globals.decomposition.uniform_x)
            self.assertTrue(globals.decomposition.uniform_y)
            self.assertTrue(globals.decomposition.uniform_z)
            self.assertEqual(globals.decomposition.nx, 4)
            self.assertEqual(globals.decomposition.ny, 5)
            self.assertEqual(globals.decomposition.nz, 6)

            # do it all in one function call to make sure this works
            dd.set_params(nx=2, ny=3, nz=4)
            self.assertTrue(globals.decomposition.uniform_x)
            self.assertTrue(globals.decomposition.uniform_y)
            self.assertTrue(globals.decomposition.uniform_z)
            self.assertEqual(globals.decomposition.nx, 2)
            self.assertEqual(globals.decomposition.ny, 3)
            self.assertEqual(globals.decomposition.nz, 4)

            # now back to a new non-uniform spacing
            dd.set_params(x=0.6)
            dd.set_params(y=0.4)
            dd.set_params(z=0.3)
            self.assertFalse(globals.decomposition.uniform_x)
            self.assertFalse(globals.decomposition.uniform_y)
            self.assertFalse(globals.decomposition.uniform_z)
            self.assertAlmostEqual(globals.decomposition.x[0], 0.6)
            self.assertAlmostEqual(globals.decomposition.y[0], 0.4)
            self.assertAlmostEqual(globals.decomposition.z[0], 0.3)

            # do it all at once
            dd.set_params(x=[0.2,0.3], y=[0.4,0.1], z=[0.25,0.25])
            self.assertFalse(globals.decomposition.uniform_x)
            self.assertFalse(globals.decomposition.uniform_y)
            self.assertFalse(globals.decomposition.uniform_z)
            self.assertAlmostEqual(globals.decomposition.x[0], 0.2)
            self.assertAlmostEqual(globals.decomposition.x[1], 0.3)
            self.assertAlmostEqual(globals.decomposition.y[0], 0.4)
            self.assertAlmostEqual(globals.decomposition.y[1], 0.1)
            self.assertAlmostEqual(globals.decomposition.z[0], 0.25)
            self.assertAlmostEqual(globals.decomposition.z[1], 0.25)

            # try a mixture of things
            dd.set_params(nx=3, y=0.8, nz=2)
            self.assertTrue(globals.decomposition.uniform_x)
            self.assertFalse(globals.decomposition.uniform_y)
            self.assertTrue(globals.decomposition.uniform_z)
            self.assertEqual(globals.decomposition.nx, 3)
            self.assertAlmostEqual(globals.decomposition.y[0], 0.8)
            self.assertEqual(globals.decomposition.nz, 2)

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
