
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

            # this is wrong to set both, so it should just use the x list
            comm.decomposition(x=[0.2,0.3,0.1], nx=2)
            dd = globals.decomposition._make_cpp_decomposition(boxdim)

            self.assertEquals(len(dd.getCumulativeFractions(0)), 5)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[1], 0.2)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[2], 0.5)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[3], 0.6)
            self.assertAlmostEquals(dd.getCumulativeFractions(0)[4], 1.0)

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

            # this is wrong to set both, so it should just use the y list
            comm.decomposition(y=[0.2,0.3,0.1], ny=2)
            dd = globals.decomposition._make_cpp_decomposition(boxdim)

            self.assertEquals(len(dd.getCumulativeFractions(1)), 5)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[1], 0.2)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[2], 0.5)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[3], 0.6)
            self.assertAlmostEquals(dd.getCumulativeFractions(1)[4], 1.0)

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

            # this is wrong to set both, so it should just use the z list
            comm.decomposition(z=[0.2,0.3,0.1], nz=2)
            dd = globals.decomposition._make_cpp_decomposition(boxdim)

            self.assertEquals(len(dd.getCumulativeFractions(2)), 5)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[0], 0.0)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[1], 0.2)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[2], 0.5)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[3], 0.6)
            self.assertAlmostEquals(dd.getCumulativeFractions(2)[4], 1.0)

            # set a global value, and try with ny arg set
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
            globals.options.nz = None # undo this so that it doesn't contaminate other dimensions

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

## Dynamic load balancing tests
class load_balance_tests (unittest.TestCase):
    def setUp(self):
        box = data.boxdim(L=10)
        boxdim = box._getBoxDim()
        comm.decomposition(nx=2,ny=2,nz=2)
        init.create_random(N=100, phi_p=0.05)

    def test_basic(self):
        comm.balance()

    def test_set_params(self):
        lb = comm.balance(x=False, y=False, z=False, tolerance=0.02, maxiter=2, period=4, phase=1)
        lb.set_params(x=True, y=True, z=True, tolerance=0.05, maxiter=1)
        

    def tearDown(self):
        init.reset()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
