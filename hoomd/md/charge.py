# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander / All Developers are free to add commands for new features

R""" Electrostatic potentials.

Charged interactions are usually long ranged, and for computational efficiency this is split
into two parts, one part computed in real space and on in Fourier space. You don't need to worry about this
implementation detail, however, as charge commands in hoomd automatically initialize and configure both the long
and short range parts.

Only one method of computing charged interactions should be used at a time. Otherwise, they would add together and
produce incorrect results.
"""

from hoomd.md import force;
from hoomd import _hoomd
from hoomd.md import _md
from hoomd.md import pair;
from hoomd.md import nlist as nl # to avoid naming conflicts
import hoomd;

import math;
import sys;

from math import sqrt

class pppm(force._force):
    R""" Long-range electrostatics computed with the PPPM method.

    Args:
        group (:py:mod:`hoomd.group`): Group on which to apply long range PPPM forces. The short range part is always applied between
                                       all particles.
        nlist (:py:mod:`hoomd.md.nlist`): Neighbor list


    `D. LeBard et. al. 2012 <http://dx.doi.org/10.1039/c1sm06787g>`_ describes the PPPM implementation details in
    HOOMD-blue. Please cite it if you utilize the PPPM functionality in your work.

    :py:class:`pppm` specifies **both** the long-ranged **and** short range parts of the electrostatic
    force should be computed between all charged particles in the simulation. In other words, :py:class:`pppm`
    initializes and sets all parameters for its own :py:class:`hoomd.md.pair.ewald`, so do not specify an additional one.

    The command supports additional screening of interactions, according to the Ewald summation for Yukawa potentials.
    This is useful if one wants to compute a screened interaction (i.e. a solution to the linearized Poisson-Boltzmann
    equation), yet the cut-off radius is so large that the computation with a purely short-ranged potential would become
    inefficient. In that case, the inverse Debye screening length can be supplied using :py:meth:`set_params()`.
    Also see `Salin, G and Caillol, J. 2000, <http://dx.doi.org/10.1063/1.1326477>`.

    Parameters:

    - Nx - Number of grid points in x direction
    - Ny - Number of grid points in y direction
    - Nz - Number of grid points in z direction
    - order - Number of grid points in each direction to assign charges to
    - :math:`r_{\mathrm{cut}}` - Cutoff for the short-ranged part of the electrostatics calculation

    Parameters Nx, Ny, Nz, order, :math:`r_{\mathrm{cut}}` must be set using
    :py:meth:`set_params()` before any :py:func:`hoomd.run()` can take place.

    See :ref:`page-units` for information on the units assigned to charges in hoomd.

    Note:
          :py:class:`pppm` takes a particle group as an option. This should be the group of all charged particles
          (:py:func:`hoomd.group.charged`). However, note that this group is static and determined at the time
          :py:class:`pppm` is specified. If you are going to add charged particles at a later point in the simulation
          with the data access API, ensure that this group includes those particles as well.

    .. important::
        In MPI simulations, the number of grid point along every dimensions must be a power of two.

    Example::

        charged = group.charged();
        pppm = charge.pppm(group=charged)

    """
    def __init__(self, group, nlist):
        hoomd.util.print_status_line();

        # initialize the base class
        force._force.__init__(self);

        # register the citation
        c = hoomd.cite.article(cite_key='dnlebard2012',
                         author=['D N LeBard', 'B G Levine', 'S A Barr', 'A Jusufi', 'S Sanders', 'M L Klein', 'A Z Panagiotopoulos'],
                         title='Self-assembly of coarse-grained ionic surfactants accelerated by graphics processing units',
                         journal='Journal of Computational Physics',
                         volume=8,
                         number=8,
                         pages='2385-2397',
                         month='',
                         year='2012',
                         doi='10.1039/c1sm06787g',
                         feature='PPPM')
        hoomd.cite._ensure_global_bib().add(c)

        # create the c++ mirror class

        # PPPM itself doesn't really need a neighbor list, so subscribe call back as None
        self.nlist = nlist
        self.nlist.subscribe(lambda : None)
        self.nlist.update_rcut()

        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _md.PPPMForceCompute(hoomd.context.current.system_definition, self.nlist.cpp_nlist, group.cpp_group);
        else:
            self.cpp_force = _md.PPPMForceComputeGPU(hoomd.context.current.system_definition, self.nlist.cpp_nlist, group.cpp_group);

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        # error check flag - must be set to true by set_params in order for the run() to commence
        self.params_set = False;

        # initialize the short range part of electrostatics
        hoomd.util.quiet_status();
        self.ewald = pair.ewald(r_cut = False, nlist = self.nlist);
        hoomd.util.unquiet_status();

    # override disable and enable to work with both of the forces
    def disable(self, log=False):
        hoomd.util.print_status_line();

        hoomd.util.quiet_status();
        force._force.disable(self, log);
        self.ewald.disable(log);
        hoomd.util.unquiet_status();

    def enable(self):
        hoomd.util.print_status_line();

        hoomd.util.quiet_status();
        force._force.enable(self);
        self.ewald.enable();
        hoomd.util.unquiet_status();

    def set_params(self, Nx, Ny, Nz, order, rcut, alpha = 0.0):
        """ Sets PPPM parameters.

        Args:
            Nx (int): Number of grid points in x direction
            Ny (int): Number of grid points in y direction
            Nz (int): Number of grid points in z direction
            order (int): Number of grid points in each direction to assign charges to
            rcut  (float): Cutoff for the short-ranged part of the electrostatics calculation
            alpha (float, **optional**): Debye screening parameter (in units 1/distance)
                .. versionadded:: 2.1

        Examples::

            pppm.set_params(Nx=64, Ny=64, Nz=64, order=6, rcut=2.0)

        Note that the Fourier transforms are much faster for number of grid points of the form 2^N.
        """
        hoomd.util.print_status_line();

        if hoomd.context.current.system_definition.getNDimensions() != 3:
            hoomd.context.msg.error("System must be 3 dimensional\n");
            raise RuntimeError("Cannot compute PPPM");

        self.params_set = True;

        # get sum of charges and of squared charges
        q = self.cpp_force.getQSum();
        q2 = self.cpp_force.getQ2Sum();
        N = hoomd.context.current.system_definition.getParticleData().getNGlobal()
        box = hoomd.context.current.system_definition.getParticleData().getGlobalBox()
        Lx = box.getL().x
        Ly = box.getL().y
        Lz = box.getL().z

        hx = Lx/Nx
        hy = Ly/Ny
        hz = Lz/Nz

        gew1 = 0.0
        kappa = gew1
        f = diffpr(hx, hy, hz, Lx, Ly, Lz, N, order, kappa, q2, rcut)
        hmin = min(hx, hy, hz)
        gew2 = 10.0/hmin
        kappa = gew2
        fmid = diffpr(hx, hy, hz, Lx, Ly, Lz, N, order, kappa, q2, rcut)

        if f*fmid >= 0.0:
            hoomd.context.msg.error("f*fmid >= 0.0\n");
            raise RuntimeError("Cannot compute PPPM");

        if f < 0.0:
            dgew=gew2-gew1
            rtb = gew1
        else:
            dgew=gew1-gew2
            rtb = gew2

        ncount = 0

        while math.fabs(dgew) > 0.00001 and fmid != 0.0:
            dgew *= 0.5
            kappa = rtb + dgew
            fmid = diffpr(hx, hy, hz, Lx, Ly, Lz, N, order, kappa, q2, rcut)
            if fmid <= 0.0:
                rtb = kappa
            ncount += 1
            if ncount > 10000.0:
                hoomd.context.msg.error("kappa not converging\n");
                raise RuntimeError("Cannot compute PPPM");

        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i));

        hoomd.util.quiet_status();
        for i in range(0,ntypes):
            for j in range(0,ntypes):
                self.ewald.pair_coeff.set(type_list[i], type_list[j], kappa = kappa, alpha = alpha, r_cut=rcut)
        hoomd.util.unquiet_status();

        # set the parameters for the appropriate type
        self.cpp_force.setParams(Nx, Ny, Nz, order, kappa, rcut, alpha);

    def update_coeffs(self):
        if not self.params_set:
            hoomd.context.msg.error("Coefficients for PPPM are not set. Call set_coeff prior to run()\n");
            raise RuntimeError("Error initializing run");

        if self.nlist.cpp_nlist.getDiameterShift():
            hoomd.context.msg.warning("Neighbor diameter shifting is enabled, PPPM may not correct for all excluded interactions\n");

def diffpr(hx, hy, hz, xprd, yprd, zprd, N, order, kappa, q2, rcut):
    lprx = rms(hx, xprd, N, order, kappa, q2)
    lpry = rms(hy, yprd, N, order, kappa, q2)
    lprz = rms(hz, zprd, N, order, kappa, q2)
    kspace_prec = math.sqrt(lprx*lprx + lpry*lpry + lprz*lprz) / sqrt(3.0)
    real_prec = 2.0*q2 * math.exp(-kappa*kappa*rcut*rcut)/sqrt(N*rcut*xprd*yprd*zprd)
    value = kspace_prec - real_prec
    return value

def rms(h, prd, N, order, kappa, q2):
    acons = [[0 for _ in range(8)] for _ in range(8)]

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
    for m in range(0,order):
        sum += acons[order][m]*pow(h*kappa, 2.0*m)
    value = q2*pow(h*kappa,order)*sqrt(kappa*prd*sqrt(2.0*math.pi)*sum/N)/prd/prd
    return value
