# Python implementation of math
import hoomd
from rowan import conjugate, rotate, vector_vector_rotation, from_axis_angle
from numpy import cross, dot, exp, square, sqrt
import numpy as np

# the pointing direction of the particle when initialized with the unit quaternion
HOOMD_REST_ORIENTATION = np.array([1,0,0])

def normalize(a):
    return a / sqrt(np.sum(square(a), axis=0))

class Particle:
    def __init__(self, position, global_to_particle_quat = None, pointing_direction = None):
        self.position = np.array(position)
        if global_to_particle_quat is None:
            # from global to particle frame
            self.orientation = vector_vector_rotation(HOOMD_REST_ORIENTATION, pointing_direction)
        else:
            self.orientation = global_to_particle_quat
        # as hoomd defines it, from particle to global frame
        self.q = self.orientation #conjugate(self.orientation)

    @property
    def pointing(self):
        ex = [1,0,0]
        ey = [0,1,0]
        ez = [0,0,1]
        return rotate(self.q, [ex, ey, ez])


class Yukawa:
    def __init__(self, epsilon, sigma, kappa):
        self.epsilon = epsilon
        self.sigma = sigma
        self.kappa = kappa

    def energy(self, r):
        return self.epsilon * exp(-self.kappa * r) / r

    def force(self, r):
        return self.epsilon / (exp(self.kappa*r) * r**2) + (self.epsilon*self.kappa) / (exp(self.kappa*r)*r)


class LJ:
    def __init__(self, epsilon, sigma):
        self.epsilon = epsilon
        self.sigma = sigma

    def energy(self, r):
        assert np.size(r) == 1
        return 4*self.epsilon*(-(self.sigma**6/r**6) + self.sigma**12/r**12)

    def force(self, r):
        assert np.size(r) == 1
        return -4*self.epsilon*((6*self.sigma**6)/r**7 - (12*self.sigma**12)/r**13)


class Patch:
    def __init__(self, alpha, omega, ni, nj):
        self.alpha = alpha
        self.cosalpha = np.cos(alpha)
        self.omega = omega
        self.ni = normalize(np.array(ni))
        self.nj = normalize(np.array(nj))

    def _costhetai(self, dr, ni_world):
        return -dot(normalize(dr), ni_world)

    def _costhetaj(self, dr, nj_world):
        return dot(normalize(dr), nj_world)

    def fi(self, dr, ni_world):
        return 1 / (1 + exp(-self.omega * (self._costhetai(dr, ni_world) - self.cosalpha) ) )

    def fj(self, dr, nj_world):
        return 1 / (1 + exp(-self.omega * (self._costhetaj(dr, nj_world) - self.cosalpha) ) )

    def dfi_dni(self, dr, ni_world):
        """Derivative of fi with respect to ni. Note the negative sign."""
        return -self.omega * exp(-self.omega * (self._costhetai(dr, ni_world) - self.cosalpha)) *  self.fi(dr, ni_world)**2

    def dfj_dnj(self, dr, nj_world):
        return self.omega *  exp(-self.omega * (self._costhetaj(dr, nj_world) - self.cosalpha)) * self.fj(dr, nj_world)**2


class PatchPair:
    def __init__(self, i, j, isotropic_potential, patch):
        self.i = i
        self.j = j
        self.iso = isotropic_potential
        self.patch = patch

    @property
    def dr(self):
        return self.i.position - self.j.position

    @property
    def magdr(self):
        """The length of the dr vector"""
        magdr = sqrt(np.sum(square(self.dr),axis=0))
        return magdr

    @property
    def ni_world(self):
        return rotate(self.i.q, self.patch.ni)

    @property
    def nj_world(self):
        return rotate(self.j.q, self.patch.nj)
        
    def energy(self):
        dr = self.dr
        return self.iso.energy(self.magdr) * self.patch.fi(dr, self.ni_world) * self.patch.fj(dr, self.nj_world)

    def torque_i(self):
        const = -self.iso.energy(self.magdr) * self.patch.fj(self.dr, self.nj_world) / self.magdr
        terms = np.array([0,0,0], dtype='float64')
        a = self.i.pointing
        for i in range(3):
            # print("ni is", self.patch.ni[i])
            # print("a i is", a[i])
            # print("deriv is", self.patch.dfi_dni(self.dr, self.ni_world))
            rhat = normalize(self.dr)
            new_term = self.patch.ni[i] * cross(a[i],  rhat * self.patch.dfi_dni(self.dr, self.ni_world))
            # print("torque i term ", i, " ", new_term)
            terms += new_term
        return const * terms

    def torque_j(self):
        # print("fi is", self.patch.fi(self.dr, self.ni_world))
        const = -self.iso.energy(self.magdr) * self.patch.fi(self.dr, self.ni_world) / self.magdr
        terms = np.array([0,0,0], dtype='float64')
        b = self.j.pointing
        for i in range(3):
            # print("nj", self.patch.nj[i])
            # print("b i is", b[i])
            # print("deriv is", self.patch.dfj_dnj(self.dr, self.nj_world))
            rhat = normalize(self.dr)
            new_term = self.patch.nj[i] * cross(b[i], rhat * self.patch.dfj_dnj(self.dr, self.nj_world))
            # print("torque j term ", i, " ", new_term)
            terms += new_term
        # print("torque j:")
        # print("terms ", terms)
        # print("const ", const)
        return const * terms

    def force(self):
        """On second particle"""
        dr = self.dr
        magdr = self.magdr

        term1 = self.iso.force(magdr)/magdr * dr * self.patch.fi(dr, self.ni_world) * self.patch.fj(dr, self.nj_world)

        # dfi/du
        dfi_dui = -self.patch.dfi_dni(dr, self.ni_world)

        # dui/dr
        lo = magdr
        dlo = normalize(dr)
        #hi = dot(dr, self.patch.ni)
        #dhi = self.patch.ni
        hi = dot(dr, self.ni_world)
        dhi = self.ni_world
        dui_dr = (lo*dhi - hi*dlo) / (lo*lo)

        # dfj/du
        dfj_duj = self.patch.dfj_dnj(dr, self.nj_world)

        # duj/dr
        lo = magdr
        dlo = normalize(dr)
        # hi = dot(dr, self.patch.nj)
        # dhi = self.patch.nj
        hi = dot(dr, self.nj_world)
        dhi = self.nj_world
        duj_dr = (lo*dhi - hi*dlo) / (lo*lo)

        term2 = (
            dfj_duj * duj_dr * self.patch.fi(dr, self.ni_world) + dfi_dui * dui_dr * self.patch.fj(dr, self.nj_world)
        )
        
        return -(term1 + term2)

    def print_values(self):
        patch_dict = vars(self.patch).copy()
        del patch_dict["cosalpha"]
        for k in ["ni", "nj"]:
            patch_dict[k] = list(patch_dict[k])
        params = {"pair_params": vars(self.iso),
             "envelope_params": patch_dict}
        return (getattr(hoomd.md.pair.aniso, "Patchy" + type(self.iso).__name__),
                {},
                params,
                [list(self.i.position), list(self.j.position)],
                [list(self.i.q), list(self.j.q)],
                list(-self.force()),
                self.energy(),
                [list(self.torque_i()), list(self.torque_j())]
                )




if __name__ == '__main__':
   pass
