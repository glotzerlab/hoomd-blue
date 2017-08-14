from hoomd import *
from hoomd import md

import math
import numpy as np

import unittest

context.initialize()


def rotate(q,v):
    q = np.array(q)
    v = np.array(v)
    return np.array((q[0]*q[0]-np.dot(q[1:],q[1:]))*v+2*q[0]*np.cross(q[1:],v)+2*np.dot(q[1:],v)*q[1:])

class bd_angular_test(unittest.TestCase):
    def setUp(self):
        context.initialize()

        context.initialize()
        snap = data.make_snapshot(N=1,particle_types=['A'], box = data.boxdim(L=10))
        if comm.get_rank() == 0:
            snap.particles.position[0] = (0,0,0)
            snap.particles.orientation[0] = (1,0,0,0)
            snap.particles.moment_inertia[0] = (0,1,1)
        self.s = init.read_snapshot(snap)

    def test_dipole_orientation(self):
        # test the probability distribution of a dipole orientation in an homogenous electric field

        # dipole moment
        d = 1
        self.s.particles.types.add('+')
        self.s.particles.types.add('-')

        rigid = md.constrain.rigid()
        rigid.set_param('A', types=['+','-'], positions=[(-d/2,0,0),(d/2,0,0)], charges=[1,-1])
        rigid.create_bodies()

        # field along z direction
        ez = (1,0,0)
        E = 5  # field strength
        field = md.external.e_field(E*np.array(ez))

        md.integrate.mode_standard(dt=0.005)
        bd = md.integrate.brownian(group=group.rigid_center(),kT=1.0,seed=1234)
        bd.set_gamma('A',1)
        bd.set_gamma_r('A',1)

        cphi = []
        def get_polar_angle(timestep):
            ex = (1,0,0)
            q = self.s.particles[0].orientation
            n = rotate(q,ex)
            cphi.append(np.dot(n,ez))


        run(1e6,callback=get_polar_angle,callback_period=1)

        # analyze results
        hist, bins = np.histogram(cphi,bins=100,density=True)

        # expected normalization constant of P(theta) = 1/Z*exp(-beta*E*cos(theta))*sin(theta)
        Z = 29.6813

        # due to numerical issues it is hard to get the overall normalization (== integral)
        # exactly right

        # verify if all it is a constant factor
        log_factor = math.log(hist[0])+E*0.5*(bins[1]+bins[0])+math.log(Z)

        # we could get better than one significant digit by running longer
        # but we're veryfing this for the entire curve
        sigfigs = 1

        for i, h in enumerate(hist):
            b = 0.5*(bins[i+1]+bins[i])
            logP = math.log(h)
            logP_expected = -E*b-math.log(Z)+log_factor

            # the distribution is accurate down to ~exp(-6) with 1e6 time steps
            if logP < -6.0: continue

            np.testing.assert_allclose(logP,logP_expected,rtol=math.pow(10,-sigfigs+1))


if __name__ == '__main__':
    unittest.main(argv = ['test.py', '-v'])
