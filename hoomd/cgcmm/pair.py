# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

R""" CGCMM pair potentials.
"""

import hoomd
import hoomd.md
from hoomd.cgcmm import _cgcmm
from hoomd import _hoomd
from hoomd.md import _md
import math

class cgcmm(hoomd.md.force._force):
    R""" CMM coarse-grain model pair potential.

    Args:
        r_cut (float): Default cutoff radius (in distance units).
        nlist (:py:mod:`hoomd.md.nlist`): Neighbor list

    :py:class:`cgcmm` specifies that a special version of Lennard-Jones pair force
    should be added to every non-bonded particle pair in the simulation. This potential
    version is used in the CMM coarse grain model and uses a combination of Lennard-Jones
    potentials with different exponent pairs between different atom pairs.

    `B. Levine et. al. 2011 <http://dx.doi.org/10.1021/ct2005193>`_ describes the CGCMM implementation details in
    HOOMD-blue. Cite it if you utilize the CGCMM potential in your work.

    Multiple potential functions can be selected:

    .. math::

        V_{\mathrm{LJ}}(r) = 4 \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{12} -
                                                  \alpha \left( \frac{\sigma}{r} \right)^{6} \right]

        V_{\mathrm{LJ}}(r) = \frac{27}{4} \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{9} -
                                                             \alpha \left( \frac{\sigma}{r} \right)^{6} \right]

        V_{\mathrm{LJ}}(r) = \frac{3\sqrt{3}}{2} \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{12} -
                                                               \alpha \left( \frac{\sigma}{r} \right)^{4} \right]

    See :py:class:`hoomd.md.pair.pair` for details on how forces are calculated and the available energy shifting and smoothing modes.
    Use :py:meth:`pair_coeff.set <hoomd.md.pair.coeff.set>` to set potential coefficients.

    The following coefficients must be set per unique pair of particle types:

    - :math:`\varepsilon` - *epsilon* (in energy units)
    - :math:`\sigma` - *sigma* (in distance units)
    - :math:`\alpha` - *alpha* (unitless) - *optional*: defaults to 1.0
    - exponents, the choice of LJ-exponents, currently supported are 12-6, 9-6, and 12-4.

    We support three keyword variants 124 (native), lj12_4 (LAMMPS), LJ12-4 (MPDyn).

    Example::

        nl = nlist.cell()
        cg = pair.cgcmm(r_cut=3.0, nlist=nl)
        cg.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, alpha=1.0, exponents='LJ12-6')
        cg.pair_coeff.set('W', 'W', epsilon=3.7605, sigma=1.285588, alpha=1.0, exponents='lj12_4')
        cg.pair_coeff.set('OA', 'OA', epsilon=1.88697479, sigma=1.09205882, alpha=1.0, exponents='96')

    """
    def __init__(self, r_cut, nlist):
        hoomd.util.print_status_line();

        # Error out in MPI simulations
        if (_hoomd.is_MPI_available()):
            if hoomd.context.current.system_definition.getParticleData().getDomainDecomposition():
                hoomd.context.msg.error("pair.cgcmm is not supported in multi-processor simulations.\n\n")
                raise RuntimeError("Error setting up pair potential.")

        # initialize the base class
        hoomd.md.force._force.__init__(self);

        # this class extends force, so we need to store the r_cut explicitly as a member
        # to be used in get_rcut
        # the authors of this potential also did not incorporate pairwise cutoffs, so we just use
        # the same number for everything
        self.r_cut = r_cut

        # setup the coefficient matrix
        self.pair_coeff = hoomd.md.pair.coeff();

        self.nlist = nlist
        self.nlist.subscribe(lambda:self.get_rcut())
        self.nlist.update_rcut()

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _cgcmm.CGCMMForceCompute(hoomd.context.current.system_definition, self.nlist.cpp_nlist, r_cut);
        else:
            self.nlist.cpp_nlist.setStorageMode(_md.NeighborList.storageMode.full);
            self.cpp_force = _cgcmm.CGCMMForceComputeGPU(hoomd.context.current.system_definition, self.nlist.cpp_nlist, r_cut);
            self.cpp_force.setBlockSize(128);

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

    def get_rcut(self):
        if not self.log:
            return None

        # go through the list of only the active particle types in the sim
        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i));

        # update the rcut by pair type
        r_cut_dict = hoomd.md.nlist.rcut();
        for i in range(0,ntypes):
            for j in range(i,ntypes):
                r_cut_dict.set_pair(type_list[i],type_list[j],self.r_cut);

        return r_cut_dict;

    def update_coeffs(self):
        # check that the pair coefficients are valid
        if not self.pair_coeff.verify(["epsilon", "sigma", "alpha", "exponents"]):
            hoomd.context.msg.error("Not all pair coefficients are set in pair.cgcmm\n");
            raise RuntimeError("Error updating pair coefficients");

        # set all the params
        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i));

        for i in range(0,ntypes):
            for j in range(i,ntypes):
                epsilon = self.pair_coeff.get(type_list[i], type_list[j], "epsilon");
                sigma = self.pair_coeff.get(type_list[i], type_list[j], "sigma");
                alpha = self.pair_coeff.get(type_list[i], type_list[j], "alpha");
                exponents = self.pair_coeff.get(type_list[i], type_list[j], "exponents");
                # we support three variants 124 (native), lj12_4 (LAMMPS), LJ12-4 (MPDyn)
                if (exponents == 124) or  (exponents == 'lj12_4') or  (exponents == 'LJ12-4') :
                    prefactor = 2.59807621135332
                    lja = prefactor * epsilon * math.pow(sigma, 12.0);
                    ljb = -alpha * prefactor * epsilon * math.pow(sigma, 4.0);
                    self.cpp_force.setParams(i, j, lja, 0.0, 0.0, ljb);
                elif (exponents == 96) or  (exponents == 'lj9_6') or  (exponents == 'LJ9-6') :
                    prefactor = 6.75
                    lja = prefactor * epsilon * math.pow(sigma, 9.0);
                    ljb = -alpha * prefactor * epsilon * math.pow(sigma, 6.0);
                    self.cpp_force.setParams(i, j, 0.0, lja, ljb, 0.0);
                elif (exponents == 126) or  (exponents == 'lj12_6') or  (exponents == 'LJ12-6') :
                    prefactor = 4.0
                    lja = prefactor * epsilon * math.pow(sigma, 12.0);
                    ljb = -alpha * prefactor * epsilon * math.pow(sigma, 6.0);
                    self.cpp_force.setParams(i, j, lja, 0.0, ljb, 0.0);
                else:
                    raise RuntimeError("Unknown exponent type.  Must be one of MN, ljM_N, LJM-N with M+N in 12+4, 9+6, or 12+6");

