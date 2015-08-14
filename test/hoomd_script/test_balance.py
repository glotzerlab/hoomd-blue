# -*- coding: iso-8859-1 -*-
# Maintainer: mphoward

from hoomd_script import *
context.initialize()
import hoomd
import unittest

## Domain decomposition balancing tests
class balance_tests (unittest.TestCase):
    ## Test that no errors are raised if a uniform decomposition should be done
    def test_uniform(self):
        comm.balance()
        self.assertEqual(globals.decomposition, None)

    ## Test that balance does nothing on a single rank
    def test_single_rank(self):
        if comm.get_num_ranks() == 1:
            comm.balance(x=0.5)
            self.assertEqual(globals.decomposition, None)

    ## Test there fractions are set correctly
    def test_basic_balance(self):
        if comm.get_num_ranks() > 1:
            box = data.boxdim(L=10)
            boxdim = box._getBoxDim()
            comm.balance(z=0.2)
            dd = globals.decomposition._make_cpp_decomposition(boxdim)

            self.assertEquals(len(dd.getFractions(0)), 2)
            self.assertEquals(len(dd.getFractions(1)), 2)
            self.assertEquals(len(dd.getFractions(2)), 2)

            self.assertAlmostEquals(dd.getFractions(0)[0], 0.5)
            self.assertAlmostEquals(dd.getFractions(0)[1], 0.5)

            self.assertAlmostEquals(dd.getFractions(0)[0], 0.5)
            self.assertAlmostEquals(dd.getFractions(0)[1], 0.5)
        
            self.assertAlmostEquals(dd.getFractions(2)[0], 0.2)
            self.assertAlmostEquals(dd.getFractions(2)[1], 0.8)
        
            comm.balance(x=[0.2,0.3,0.1], y=0.3)
            dd = globals.decomposition._make_cpp_decomposition(boxdim)
            self.assertEquals(len(dd.getFractions(0)), 4)
            self.assertEquals(len(dd.getFractions(1)), 2)
            self.assertEquals(len(dd.getFractions(2)), 1)

            self.assertAlmostEquals(dd.getFractions(0)[0], 0.2)
            self.assertAlmostEquals(dd.getFractions(0)[1], 0.3)
            self.assertAlmostEquals(dd.getFractions(0)[2], 0.1)
            self.assertAlmostEquals(dd.getFractions(0)[3], 0.4)        

            self.assertAlmostEquals(dd.getFractions(1)[0], 0.3)
            self.assertAlmostEquals(dd.getFractions(1)[1], 0.7)
        
            self.assertAlmostEquals(dd.getFractions(2)[0], 1.0)
        
            comm.balance(x=[0.2,0.3],y=[0.2,0.3],z=[0.2,0.3])
            dd = globals.decomposition._make_cpp_decomposition(boxdim)

    ## Test that balancing fails after initialization
    def test_wrong_order(self):
        init.create_random(N=100, phi_p=0.05)        
        with self.assertRaises(RuntimeError):
            comm.balance(y=0.3)

    ## Test that errors are raised if fractional divisions exceed 1.0
    def test_bad_fractions(self):
        if comm.get_num_ranks() > 1:
            box = data.boxdim(L=10)
            boxdim = box._getBoxDim()
            with self.assertRaises(RuntimeError):
                comm.balance(x=-0.2)
                globals.decomposition._make_cpp_decomposition(boxdim)

                comm.balance(x=1.2)
                globals.decomposition._make_cpp_decomposition(boxdim)

                comm.balance(x=[0.2,0.9])
                globals.decomposition._make_cpp_decomposition(boxdim)
                
                comm.balance(x=[0.3,-0.1])
                globals.decomposition._make_cpp_decomposition(boxdim)

                comm.balance(y=-0.2)
                globals.decomposition._make_cpp_decomposition(boxdim)

                comm.balance(y=1.2)
                globals.decomposition._make_cpp_decomposition(boxdim)

                comm.balance(y=[0.2,0.9])
                globals.decomposition._make_cpp_decomposition(boxdim)

                comm.balance(y=[0.3,-0.1])
                globals.decomposition._make_cpp_decomposition(boxdim)

                comm.balance(z=-0.2)
                globals.decomposition._make_cpp_decomposition(boxdim)

                comm.balance(z=1.2)
                globals.decomposition._make_cpp_decomposition(boxdim)

                comm.balance(z=[0.2,0.9])
                globals.decomposition._make_cpp_decomposition(boxdim)

                comm.balance(z=[0.3,-0.1])
                globals.decomposition._make_cpp_decomposition(boxdim)

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
