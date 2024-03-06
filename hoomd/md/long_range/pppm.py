# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Long-range potentials evaluated using the PPPM method."""

import hoomd
from hoomd.md.force import Force
import math
import numpy


def make_pppm_coulomb_forces(nlist, resolution, order, r_cut, alpha=0):
    """Long range Coulomb interactions evaluated using the PPPM method.

    Args:
        nlist (hoomd.md.nlist.NeighborList): Neighbor list.
        resolution (tuple[int, int, int]): Number of grid points in the x, y,
          and z directions :math:`\\mathrm{[dimensionless]}`.
        order (int): Number of grid points in each direction to assign charges
          to :math:`\\mathrm{[dimensionless]}`.
        r_cut (float): Cutoff distance between the real space and reciprocal
          space terms :math:`\\mathrm{[length]}`.
        alpha (float): Debye screening parameter
          :math:`\\mathrm{[length^{-1}]}`.

    Evaluate the potential energy :math:`U_\\mathrm{coulomb}` and apply
    the corresponding forces to the particles in the simulation.

    .. math::

        U_\\mathrm{coulomb} = \\frac{1}{2} \\sum_\\vec{n} \\sum_{i=0}^{N-1}
          \\sum_{j=0}^{N-1} u_\\mathrm{coulomb}(\\vec{r}_j - \\vec{r}_i +
          n_1 \\cdot \\vec{a}_1 + n_2 \\cdot \\vec{a}_2 +
          n_3 \\cdot \\vec{a}_3, q_i, q_j)

    .. math::

        u_\\mathrm{coulomb}(\\vec{r}, q_i, q_j) = \\frac{q_i q_j}{r}
          e^{-\\alpha r}

    where the infinite sum includes all periodic images :math:`\\vec{n}`,
    :math:`N` is the number of particles, :math:`\\vec{r}_i` is the position of
    particle :math:`i`, :math:`q_i` is the charge of particle :math:`i`,
    :math:`\\alpha` is the Debye screening parameter, and :math:`\\vec{a}_k` are
    the periodic simulation box lattice vectors.

    Note:
        In HOOMD-blue, the :math:`\\frac{1}{4\\pi\\epsilon_0}` factor is
        included in the `units of charge <units>`.

    The particle particle particle mesh (PPPM) method splits this computation
    into real space and reciprocal space components.

    .. math::

        U_\\mathrm{coulomb} = U_\\mathrm{real\\ space}
          + U_\\mathrm{reciprocal\\ space}

    `md.pair.Ewald` to computes the real space term directly and
    `md.long_range.pppm.Coulomb` computes  the reciprocal space term using fast
    Fourier transforms performed on a charge density grid. The accuracy of the
    method is sensitive to the cutoff for the real space part, the order of
    interpolation and grid resolution.

    * `J. W. Eastwood, R. W. Hockney, and D. N. Lawrence 1980`_ describes the
      algorithm.
    * `D. LeBard et. al. 2012`_ describes the implementation in
      HOOMD-blue. Please cite it if you utilize this functionality in your work.

    .. rubric:: Exclusions

    When ``nlist`` contains exclusions, `md.pair.Ewald` skips the computation of
    the excluded real space particle-particle interactions.
    `md.long_range.pppm.Coulomb` must correct the reciprocal space computation
    for this. The full energy (``Coulomb.energy + Ewald.energy``) is the sum of
    the following terms:

    * :math:`U_\\mathrm{coulomb,additional}` (``Coulomb.additional_energy``):
      Energy from the reciprocal space calculation plus any correction due to
      ``'body'`` exclusions in the neighbor list.
    * :math:`U_{\\mathrm{coulomb},i}` (``Coulomb.energies``): Energies from the
      non-body neighbor list exclusions.
    * :math:`U_\\mathrm{ewald,additional}`
      (``Ewald.additional_energy``): 0.
    * :math:`U_\\mathrm{ewald,i}` (``Ewald.additional_energy``):
      Energies from the real space calculation for non-excluded particle pairs.

    Warning:
        Do not apply bonds, angles, dihedrals, or impropers between particles
        in the same rigid body. Doing so will cause the exclusions to be double
        counted (once in :math:`U_\\mathrm{coulomb,additional}` and again in
        :math:`U_{\\mathrm{coulomb},i}`).

    .. rubric:: Screening

    The Debye screening parameter :math:`\\alpha` enables the screening of
    electrostatic interactions with the same functional form as the short range
    `md.pair.Yukawa` potential. Use `md.long_range.pppm.Coulomb` with a non-zeo
    :math:`\\alpha` to compute screened electrostatic interactions when the
    cutoff is so large that the short ranged interactions are inefficient. See
    `Salin, G and Caillol, J. 2000`_ for details.

    Important:
        In MPI simulations with multiple ranks, the grid resolution must be a
        power of two in each dimension.

    Returns:
        ``real_space_force``, ``reciprocal_space_force``

        Add both of these forces to the integrator.

    Warning:
        `make_pppm_coulomb_forces` sets all parameters for the returned `Force`
        objects given the input resolution and order. Do not change the
        parameters of the returned objects directly.

    .. _J. W. Eastwood, R. W. Hockney, and D. N. Lawrence 1980:
      https://doi.org/10.1063/1.464397

    .. _D. LeBard et. al. 2012: http://dx.doi.org/10.1039/c1sm06787g

    .. _Salin, G and Caillol, J. 2000: http://dx.doi.org/10.1063/1.1326477
    """
    real_space_force = hoomd.md.pair.Ewald(nlist)

    # the real space force may be attached before the reciprocal space one
    # set default parameters to avoid errors in this case
    real_space_force.params.default = dict(kappa=0, alpha=0)
    real_space_force.r_cut.default = r_cut

    reciprocal_space_force = Coulomb(nlist=nlist,
                                     resolution=resolution,
                                     order=order,
                                     r_cut=r_cut,
                                     alpha=0,
                                     pair_force=real_space_force)

    return real_space_force, reciprocal_space_force


class Coulomb(Force):
    """Reciprocal space part of the PPPM Coulomb forces.

    Note:
        Use `make_pppm_coulomb_forces` to create a connected pair of
        `md.pair.Ewald` and `md.long_range.pppm.Coulomb` instances that together
        implement the PPPM method for electrostatics.

    Attributes:
        resolution (tuple[int, int, int]): Number of grid points in the x, y,
          and z directions :math:`\\mathrm{[dimensionless]}`.
        order (int): Number of grid points in each direction to assign charges
          to :math:`\\mathrm{[dimensionless]}`.
        r_cut (float): Cutoff distance between the real space and reciprocal
          space terms :math:`\\mathrm{[length]}`.
        alpha (float): Debye screening parameter
          :math:`\\mathrm{[length^{-1}]}`.
    """

    def __init__(self, nlist, resolution, order, r_cut, alpha, pair_force):
        super().__init__()
        self._nlist = hoomd.data.typeconverter.OnlyTypes(
            hoomd.md.nlist.NeighborList)(nlist)
        self._param_dict.update(
            hoomd.data.parameterdicts.ParameterDict(resolution=(int, int, int),
                                                    order=int,
                                                    r_cut=float,
                                                    alpha=float))

        self.resolution = resolution
        self.order = order
        self.r_cut = r_cut
        self.alpha = alpha
        self._pair_force = pair_force

    def _attach_hook(self):
        self.nlist._attach(self._simulation)

        if isinstance(self._simulation.device, hoomd.device.CPU):
            cls = hoomd.md._md.PPPMForceCompute
        else:
            cls = hoomd.md._md.PPPMForceComputeGPU

        # Access set parameters before attaching. These values are needed to
        # compute derived parameters before all paramters are given to the
        # _cpp_obj at the end.
        Nx, Ny, Nz = self.resolution
        order = self.order
        rcut = self.r_cut
        alpha = self.alpha

        group = self._simulation.state._get_group(hoomd.filter.All())
        self._cpp_obj = cls(self._simulation.state._cpp_sys_def,
                            self.nlist._cpp_obj, group)

        # compute the kappa parameter
        q2 = self._cpp_obj.getQ2Sum()
        N = self._simulation.state.N_particles
        box = self._simulation.state.box
        Lx = box.Lx
        Ly = box.Ly
        Lz = box.Lz

        hx = Lx / Nx
        hy = Ly / Ny
        hz = Lz / Nz

        gew1 = 0.0
        kappa = gew1
        f = _diffpr(hx, hy, hz, Lx, Ly, Lz, N, order, kappa, q2, rcut)
        hmin = min(hx, hy, hz)
        gew2 = 10.0 / hmin
        kappa = gew2
        fmid = _diffpr(hx, hy, hz, Lx, Ly, Lz, N, order, kappa, q2, rcut)

        if f * fmid >= 0.0:
            raise RuntimeError("Cannot compute PPPM Coloumb forces,\n"
                               "f*fmid >= 0.0")

        if f < 0.0:
            dgew = gew2 - gew1
            rtb = gew1
        else:
            dgew = gew1 - gew2
            rtb = gew2

        ncount = 0

        # iteratively compute kappa to minimize the error
        while math.fabs(dgew) > 0.00001 and fmid != 0.0:
            dgew *= 0.5
            kappa = rtb + dgew
            fmid = _diffpr(hx, hy, hz, Lx, Ly, Lz, N, order, kappa, q2, rcut)
            if fmid <= 0.0:
                rtb = kappa
            ncount += 1
            if ncount > 10000.0:
                raise RuntimeError("Cannot compute PPPM\n"
                                   "kappa is not converging")

        # set parameters
        particle_types = self._simulation.state.particle_types

        # this doesn't work: #1068
        # self._pair_force.params[(particle_types,
        #                          particle_types)] = dict(kappa=kappa,
        #                                                  alpha=alpha)
        # self._pair_force.r_cut[(particle_types, particle_types)] = rcut

        # workaround
        for a in particle_types:
            for b in particle_types:
                self._pair_force.params[(a, b)] = dict(kappa=kappa, alpha=alpha)
                self._pair_force.r_cut[(a, b)] = rcut

        self._cpp_obj.setParams(Nx, Ny, Nz, order, kappa, rcut, alpha)

    @property
    def nlist(self):
        """Neighbor list used to compute the real space term."""
        return self._nlist

    @nlist.setter
    def nlist(self, value):
        if self._attached:
            raise RuntimeError("nlist cannot be set after scheduling.")
        else:
            self._nlist = hoomd.data.typeconverter.OnlyTypes(
                hoomd.md.nlist.NeighborList)(value)

            # ensure that the pair force uses the same neighbor list
            self._pair_force.nlist = value


def _diffpr(hx, hy, hz, xprd, yprd, zprd, N, order, kappa, q2, rcut):
    """Part of the algorithm that computes the estimated error of the method."""
    lprx = _rms(hx, xprd, N, order, kappa, q2)
    lpry = _rms(hy, yprd, N, order, kappa, q2)
    lprz = _rms(hz, zprd, N, order, kappa, q2)
    kspace_prec = math.sqrt(lprx * lprx + lpry * lpry
                            + lprz * lprz) / math.sqrt(3.0)
    real_prec = 2.0 * q2 * math.exp(-kappa * kappa * rcut * rcut) / math.sqrt(
        N * rcut * xprd * yprd * zprd)
    value = kspace_prec - real_prec
    return value


def _rms(h, prd, N, order, kappa, q2):
    """Part of the algorithm that computes the estimated error of the method."""
    acons = numpy.zeros((8, 8))

    acons[1][0] = 2.0 / 3.0
    acons[2][0] = 1.0 / 50.0
    acons[2][1] = 5.0 / 294.0
    acons[3][0] = 1.0 / 588.0
    acons[3][1] = 7.0 / 1440.0
    acons[3][2] = 21.0 / 3872.0
    acons[4][0] = 1.0 / 4320.0
    acons[4][1] = 3.0 / 1936.0
    acons[4][2] = 7601.0 / 2271360.0
    acons[4][3] = 143.0 / 28800.0
    acons[5][0] = 1.0 / 23232.0
    acons[5][1] = 7601.0 / 13628160.0
    acons[5][2] = 143.0 / 69120.0
    acons[5][3] = 517231.0 / 106536960.0
    acons[5][4] = 106640677.0 / 11737571328.0
    acons[6][0] = 691.0 / 68140800.0
    acons[6][1] = 13.0 / 57600.0
    acons[6][2] = 47021.0 / 35512320.0
    acons[6][3] = 9694607.0 / 2095994880.0
    acons[6][4] = 733191589.0 / 59609088000.0
    acons[6][5] = 326190917.0 / 11700633600.0
    acons[7][0] = 1.0 / 345600.0
    acons[7][1] = 3617.0 / 35512320.0
    acons[7][2] = 745739.0 / 838397952.0
    acons[7][3] = 56399353.0 / 12773376000.0
    acons[7][4] = 25091609.0 / 1560084480.0
    acons[7][5] = 1755948832039.0 / 36229939200000.0
    acons[7][6] = 4887769399.0 / 37838389248.0

    sum = 0.0
    for m in range(0, order):
        sum += acons[order][m] * pow(h * kappa, 2.0 * m)
    value = q2 * pow(h * kappa, order) * math.sqrt(
        kappa * prd * math.sqrt(2.0 * math.pi) * sum / N) / prd / prd
    return value
