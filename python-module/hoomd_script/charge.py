# -*- coding: iso-8859-1 -*-
#Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
#(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
#Iowa State University and The Regents of the University of Michigan All rights
#reserved.

#HOOMD-blue may contain modifications ("Contributions") provided, and to which
#copyright is held, by various Contributors who have granted The Regents of the
#University of Michigan the right to modify and/or distribute such Contributions.

#Redistribution and use of HOOMD-blue, in source and binary forms, with or
#without modification, are permitted, provided that the following conditions are
#met:

#* Redistributions of source code must retain the above copyright notice, this
#list of conditions, and the following disclaimer.

#* Redistributions in binary form must reproduce the above copyright notice, this
#list of conditions, and the following disclaimer in the documentation and/or
#other materials provided with the distribution.

#* Neither the name of the copyright holder nor the names of HOOMD-blue's
#contributors may be used to endorse or promote products derived from this
#software without specific prior written permission.

#Disclaimer

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
#ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

#IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
#INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
#OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
#ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# $Id: charge.py 3644 2011-01-25 13:52:25Z joaander $
# $URL: https://codeblue.umich.edu/hoomd-blue/svn/branches/electrostatics/python-module/hoomd_script/charge.py $
# Maintainer: joaander / All Developers are free to add commands for new features

## \package hoomd_script.charge
# \brief Commands that create forces between pairs of particles
#
# Charged interactions are usually long ranged, and for computational efficiency this is split
# into two parts, one part computed in real space and on in Fourier space. You don't need to worry about this
# implementation detail, however, as charge commands in hoomd automatically initialize and configure both the long
# and short range parts.
# 
# Only one method of computing charged interactions should be used at a time. Otherwise, they would add together and
# produce incorrect results.
#
# The following methods are available:
# - pppm
#

import globals;
import force;
import hoomd;
import util;
import tune;
import init;
import data;
import variant;
import pair;

import math;
import sys;

from math import sqrt

pppm_used = False;

## Long-range electrostatics computed with the PPPM method
#
# The command charge.pppm specifies that the \b both the long-ranged \b and short range parts of the electrostatic
# force is computed between all charged particles in the simulation. In other words, charge.pppm() initializes and
# sets all parameters for its own pair.ewald, so you do not need to specify an additional one.
#
# Coeffients:
# - Nx - Number of grid points in x direction
# - Ny - Number of grid points in y direction
# - Nz - Number of grid points in z direction
# - order - Number of grid points in each direction to assign charges to
# - \f$ r_{\mathrm{cut}} \f$ - Cutoff for the short-ranged part of the electrostatics calculation
#
# Coefficients Nx, Ny, Nz, order, \f$ r_{\mathrm{cut}} \f$ must be set using
# set_coeff() before any run() can take place.
#
# See \ref page_units for information on the units assigned to charges in hoomd.
# \note charge.pppm takes a particle group as an option. This should be the group of all charged particles
#       (group.charged). However, note that this group is static and determined at the time charge.pppm() is specified.
#       If you are going to add charged particles at a later point in the simulation with the data access API,
#       ensure that this group includes those particles as well.
class pppm(force._force):
    ## Specify long-ranged electrostatic interactions between particles
    #
    # \param group Group on which to apply long range PPPM forces. The short range part is always applied between
    #              all particles.
    #
    # \b Example:
    # \code
    # charged = group.charged();
    # pppm = charge.pppm(group=charged)
    # \endcode
    def __init__(self, group):
        global pppm_used;
        
        util.print_status_line();
       
        if pppm_used:
            print >> sys.stderr, "\n***Error: cannot have more than one pppm in a single job\n";
            raise RuntimeError("Error initializing PPPM");
        pppm_used = True;
       
        # initialize the base class
        force._force.__init__(self);
        # create the c++ mirror class

        # update the neighbor list
        neighbor_list = pair._update_global_nlist(0.0)
        neighbor_list.subscribe(lambda: self.log*0.0)
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = hoomd.PPPMForceCompute(globals.system_definition, neighbor_list.cpp_nlist, group.cpp_group);
        else:
            self.cpp_force = hoomd.PPPMForceComputeGPU(globals.system_definition, neighbor_list.cpp_nlist, group.cpp_group);
        
        globals.system.addCompute(self.cpp_force, self.force_name);
        
        # error check flag - must be set to true by set_coeff in order for the run() to commence
        self.params_set = False;
        
        # initialize the short range part of electrostatics
        util._disable_status_lines = True;
        self.ewald = pair.ewald(r_cut = 0.0);
        util._disable_status_lines = False;
    
    # overrride disable and enable to work with both of the forces
    def disable(self, log=False):
        util.print_status_line();
        
        util._disable_status_lines = True;
        force._force.disable(self, log);
        self.ewald.disable(log);
        util._disable_status_lines = False;
    
    def enable(self):
        util.print_status_line();
        
        util._disable_status_lines = True;
        force._force.enable(self);
        self.ewald.enable();
        util._disable_status_lines = False;
    
    ## Sets the PPPM parameters
    #
    # \param Nx - Number of grid points in x direction
    # \param Ny - Number of grid points in y direction
    # \param Nz - Number of grid points in z direction
    # \param order - Number of grid points in each direction to assign charges to
    # \param rcut  -  Cutoff for the short-ranged part of the electrostatics calculation
    #
    # Using set_params() requires that the specified PPPM force has been saved in a variable. i.e.
    # \code
    # pppm = charge.pppm()
    # \endcode
    #
    # \b Examples:
    # \code
    # pppm.set_params(Nx=64, Ny=64, Nz=64, order=6, rcut=2.0)
    # \endcode
    # Note that the Fourier transforms are much faster for number of grid points of the form 2^N
    # The coefficients for PPPM  must be set 
    # before the run() can be started.
    def set_params(self, Nx, Ny, Nz, order, rcut):
        util.print_status_line();

        if globals.system_definition.getNDimensions() != 3:
            print >> sys.stderr, "\n***Error: System must be 3 dimensional\n";
            raise RuntimeError("Cannot compute PPPM");

        self.params_set = True;
        q2 = 0
        N = globals.system_definition.getParticleData().getN()
        for i in xrange(0,N):
            q = globals.system_definition.getParticleData().getCharge(i)
            q2 += q*q
        box = globals.system_definition.getParticleData().getBox()
        Lx = box.xhi - box.xlo
        Ly = box.yhi - box.ylo
        Lz = box.zhi - box.zlo

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
            print >> sys.stderr, "\n***Error: f*fmid >= 0.0\n";
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
                print >> sys.stderr, "\n***Error: kappa not converging\n";
                raise RuntimeError("Cannot compute PPPM");
        
        ntypes = globals.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in xrange(0,ntypes):
            type_list.append(globals.system_definition.getParticleData().getNameByType(i));

        util._disable_status_lines = True;
        for i in xrange(0,ntypes):
            for j in xrange(0,ntypes):
                self.ewald.pair_coeff.set(type_list[i], type_list[j], kappa = kappa, r_cut=rcut)
        util._disable_status_lines = False;

        # set the parameters for the appropriate type
        self.cpp_force.setParams(Nx, Ny, Nz, order, kappa, rcut);

    def update_coeffs(self):
        if not self.params_set:
            print >> sys.stderr, "\n***Error: Coefficients for PPPM are not set. Call set_coeff prior to run()\n";
            raise RuntimeError("Error initializing run");

def diffpr(hx, hy, hz, xprd, yprd, zprd, N, order, kappa, q2, rcut):
    lprx = rms(hx, xprd, N, order, kappa, q2)
    lpry = rms(hy, yprd, N, order, kappa, q2)
    lprz = rms(hz, zprd, N, order, kappa, q2)
    kspace_prec = math.sqrt(lprx*lprx + lpry*lpry + lprz*lprz) / sqrt(3.0)
    real_prec = 2.0*q2 * math.exp(-kappa*kappa*rcut*rcut)/sqrt(N*rcut*xprd*yprd*zprd)
    value = kspace_prec - real_prec
    return value

def rms(h, prd, N, order, kappa, q2):
    acons = [[0 for _ in xrange(8)] for _ in xrange(8)]

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
    for m in xrange(0,order):
        sum += acons[order][m]*pow(h*kappa, 2.0*m)
    value = q2*pow(h*kappa,order)*sqrt(kappa*prd*sqrt(2.0*math.pi)*sum/N)/prd/prd
    return value


