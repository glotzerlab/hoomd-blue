from hoomd import *
from hoomd import md
from hoomd import metal
import unittest
import numpy
import os
context.initialize()

class eam_tests(unittest.TestCase):
    # setUp is called before the start of every test method
    def setUp(self):
        ltconst = 20.0
        type1 = 'Al'
        type2 = 'Ni'
        mass1 = 26.982
        mass2 = 58.710
        pos1 = (-0.5 * ltconst, -0.5 * ltconst, -0.5 * ltconst)
        pos2 = (-0.5 * ltconst, -0.5 * ltconst, -0.4 * ltconst)
        pos3 = (-0.5 * ltconst, -0.4 * ltconst, -0.4 * ltconst)
        pos4 = (-0.4 * ltconst, -0.4 * ltconst, -0.4 * ltconst)
        typelst = [type1, type2, type2, type2]
        masslst = [mass1, mass2, mass2, mass2]
        poslst = [pos1, pos2, pos3, pos4]
        # - generate system -
        snapshot = data.make_snapshot(N=4, box=data.boxdim(L=ltconst), particle_types=[type1, type2])
        system = init.read_snapshot(snapshot)
        for p in system.particles:
            p.position = poslst[p.tag]
            p.type = typelst[p.tag]
            p.mass = masslst[p.tag]
        cwd = os.getcwd()
        # the unit test uses a potential, which is sparsed from G.Purja Pun & Y. Mishin, 2009
        tmpd = cwd + '/eamtemp/'
        os.system('rm -rf ' + tmpd)
        os.system('mkdir -p ' + tmpd)
        potf = tmpd + 'testpot'
        with open(potf, 'w') as outf:
            outf.write('test potential sparse from:\n Mishin-Ni-Al-2009.eam.alloy\n Alloy\n 2 Ni Al\n 20 0.250661 20 0.31436 6.28721\n 28 58.71 3.52 fcc\n -0.0225464 -1.76636 -2.37638 -2.58753 -2.56335 \n -2.44363 -2.1936 -1.69669 -0.881535 0.259267 \n 1.71214 3.47279 5.52768 7.84679 10.3946 \n 13.1387 16.0533 19.12 22.3271 25.6681 \n 0.166428 0.170459 0.167088 0.158941 0.148264 \n 0.134559 0.116655 0.0950843 0.0717676 0.0494264 \n 0.0305923 0.0166948 0.00778478 0.00291687 0.00076581 \n 9.6658e-05 8.84137e-07 0 0 0 \n 13 26.982 4.05 fcc\n -4.3767e-11 -1.6886 -2.24356 -2.61981 -2.8881 \n -3.03673 -3.07531 -3.14579 -3.15517 -3.04228 \n -2.80696 -2.44921 -1.96903 -1.36641 -0.641356 \n 0.206131 1.17605 2.2684 3.48319 4.82042 \n 0.396504 0.268377 0.182302 0.130397 0.104787 \n 0.09764 0.10114 0.10747 0.108814 0.097359 \n 0.0701286 0.0394937 0.0192524 0.00952344 0.00538008 \n 0.00357488 0.0027837 0.00202854 0.0010566 9.93586e-05 \n 0 1.45214 3.46822 4.73416 4.63266 \n 3.27018 1.42217 0.00246105 -0.578463 -0.528943 \n -0.314831 -0.217411 -0.216257 -0.18649 -0.098564 \n -0.021759 -0.000310685 0 0 0 \n 0 2016.46 1530.29 608.866 120.656 \n 8.56573 1.68568 0.0591469 -0.564815 -0.587964 \n -0.416922 -0.286477 -0.251829 -0.249993 -0.216214 \n -0.137026 -0.0706754 -0.0262716 -1.62953e-08 0 \n 0 10.7294 10.5529 7.42998 5.15814 \n 4.13394 3.26306 1.83395 0.548762 0.044061 \n -0.0987007 -0.134826 -0.151869 -0.205764 -0.215437 \n -0.169569 -0.0703696 0.0113375 0.0283944 0.00186361 \n')

    # API test: class initialization
    def test_API(self):
        cwd = os.getcwd()
        tmpd = cwd + '/eamtemp/'
        potf = tmpd + 'testpot'
        nl = md.nlist.cell()
        metal.pair.eam(file=potf, type="Alloy", nlist=nl)

    # Unit test: ensure that forces and energies compute correctly
    def test_force(self):
        cwd = os.getcwd()
        tmpd = cwd + '/eamtemp/'
        potf = tmpd + 'testpot'
        nl = md.nlist.cell()
        eam = metal.pair.eam(file=potf, type="Alloy", nlist=nl)
        all = group.all()
        md.integrate.mode_standard(dt=0.2)
        md.integrate.nve(group=all)
        run(1)

        F = numpy.array([x.force for x in eam.forces])
        U = numpy.array([x.energy for x in eam.forces])

        F_ref = numpy.array([[0.49554526, 1.10342697, -2.7692858],
                             [0.70281927, -1.43558566, 3.87260803],
                             [-2.00473055, 1.53052375, -0.60778632],
                             [0.80636601, -1.19836506, -0.49553591]])
        U_ref = numpy.array([-0.93424631, -1.23440579, -1.71025268, -1.4023109])

        numpy.testing.assert_allclose(F, F_ref, rtol=1e-5)
        numpy.testing.assert_allclose(U, U_ref, rtol=1e-6)

        os.system('rm -rf ' + tmpd)

    # tearDown is called at the end of every test method
    def tearDown(self):
        context.initialize()

if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])