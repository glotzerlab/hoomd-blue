# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Long-range potentials evaluated using the PPPM method."""

import hoomd
from hoomd.md.force import Force


class Coulomb(Force):
    """Long range Coulomb interactions evaluated using the PPPM method.

    Evaluate the potential energy :math:`U_\\mathrm{couloumb}` and apply
    the corresponding forces to the particles in the simulation.

    Args:
        nlist (`hoomd.md.nlist.NList`): Neighbor list.
        resolution (tuple[int, int, int]): Number of grid points in the x, y,
          and z directions :math:`\\mathrm{[dimensionless]}`.
        order (int): Number of grid points in each direction to assign charges
          to :math:`\\mathrm{[dimensionless]}`.
        r_cut (float): Cutoff distance between the real space and reciprocal
          space terms :math:`\\mathrm{[length]}`.
        alpha (float): Debye screening parameter
          :math:`\\mathrm{[length^{-1}]}`.

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

    Use `md.pair.Ewald` to compute the real space term.
    `md.long_range.pppm.Coulomb` computes  the reciprocal space term using fast
    Fourier transforms performed on a charge density grid. The accuracy of the
    method is sensitive to the cutoff for the real space part, the order of
    interpolation and grid resolution.

    `J. W. Eastwood, R. W. Hockney, and D. N. Lawrence 1980`_ describes the
    algorithm. `D. LeBard et. al. 2012`_ describes the implementation in
    HOOMD-blue. Please cite it if you utilize this functionality in your work.

    The Debye screening parameter :math:`\\alpha` enables the screening of
    electrostatic interactions following the `md.pair.Yukawa` potential. Use
    `md.long_range.pppm.Coulomb` with a non-zeo :math:`\\alpha` to compute
    screened electrostatic interactions when the cutoff is so large that the
    short ranged interactions are inefficient. See `Salin, G and Caillol, J.
    2000`_ for details.

    Hint:
        Use TODO to create the pair potential to ensure that the parameters
        match.

    Warning:
        In MPI simulations with multiple ranks, the grid resolution must be a
        power of two in each dimension.

    Attributes:
        resolution (tuple[int, int, int]): Number of grid points in the x, y,
          and z directions :math:`\\mathrm{[dimensionless]}`.
        order (int): Number of grid points in each direction to assign charges
          to :math:`\\mathrm{[dimensionless]}`.
        r_cut (float): Cutoff distance between the real space and reciprocal
          space terms :math:`\\mathrm{[length]}`.
        alpha (float): Debye screening parameter
          :math:`\\mathrm{[length^{-1}]}`.

    .. _J. W. Eastwood, R. W. Hockney, and D. N. Lawrence 1980:
      https://doi.org/10.1063/1.464397

    .. _D. LeBard et. al. 2012: http://dx.doi.org/10.1039/c1sm06787g

    .. _Salin, G and Caillol, J. 2000: http://dx.doi.org/10.1063/1.1326477
    """

    def __init__(self, nlist, resolution, order, r_cut, alpha=0):
        self._nlist = hoomd.data.typeconverter.OnlyTypes(hoomd.md.nlist.NList)
        self._param_dict.update(
            hoomd.data.parameterdicts.ParameterDict(resolution=(int, int, int),
                                                    order=int,
                                                    r_cut=float,
                                                    alpha=float(alpha)))

    @property
    def nlist(self):
        """Neighbor list used to compute the real space term."""
        return self._nlist

    @nlist.setter
    def nlist(self, value):
        if self._attached:
            raise RuntimeError("nlist cannot be set after scheduling.")
        else:
            self._nlist = hoomd.data.typeconverter.OnlyTypes(value)

    @property
    def _children(self):
        return [self.nlist]
