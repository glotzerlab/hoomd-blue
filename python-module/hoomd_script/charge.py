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

# $Id: pair.py 3644 2011-01-25 13:52:25Z joaander $
# $URL: https://codeblue.umich.edu/hoomd-blue/svn/branches/electrostatics/python-module/hoomd_script/pair.py $
# Maintainer: joaander / All Developers are free to add commands for new features

## \package hoomd_script.charge
# \brief Commands that create forces between pairs of particles
#
# Generally, %pair forces are short range and are summed over all non-bonded particles
# within a certain cutoff radius of each particle. Any number of %pair forces
# can be defined in a single simulation. The net %force on each particle due to
# all types of %pair forces is summed.
#
# Pair forces require that parameters be set for each unique type %pair. Coefficients
# are set through the aid of the coeff class. To set this coefficients, specify 
# a %pair %force and save it in a variable
# \code
# my_force = pair.some_pair_force(arguments...)
# \endcode
# Then the coefficients can be set using the saved variable.
# \code
# my_force.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
# my_force.pair_coeff.set('A', 'B', epsilon=1.0, sigma=2.0)
# my_force.pair_coeff.set('B', 'B', epsilon=2.0, sigma=1.0)
# \endcode
# This example set the parameters \a epsilon and \a sigma 
# (which are used in pair.lj). Different %pair forces require that different
# coefficients are set. Check the documentation of each to see the definition
# of the coefficients.
#
# \sa \ref page_quick_start

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


class pppm(force._force):
    ## Specify the long-ranged part of the electrostatic calculation
    # \b Example:
    # \code
    # pppm = pair.pppm(group=charged)
    # \endcode
    def __init__(self, group):
        util.print_status_line();
       
        # initialize the base class
        force._force.__init__(self);
        # create the c++ mirror class

        # update the neighbor list
        neighbor_list = pair._update_global_nlist(0.01);
        neighbor_list.subscribe(lambda: self.log*0.01)
        if not globals.exec_conf.isCUDAEnabled():
            neighbor_list.cpp_nlist.setStorageMode(hoomd.NeighborList.storageMode.full);
            self.cpp_force = hoomd.PPPMForceCompute(globals.system_definition, neighbor_list.cpp_nlist, group.cpp_group);
        else:
            neighbor_list.cpp_nlist.setStorageMode(hoomd.NeighborList.storageMode.full);
            self.cpp_force = hoomd.PPPMForceComputeGPU(globals.system_definition, neighbor_list.cpp_nlist, group.cpp_group);

        globals.system.addCompute(self.cpp_force, self.force_name);
        
    ## Sets the PPPM coefficients
    #
    # \param Nx - Number of grid points in x direction
    # \param Ny - Number of grid points in y direction
    # \param Nz - Number of grid points in z direction
    # \param order - Number of grid points in each direction to assign charges to
    # \param kappa -  Screening parameter in erfc
    # \param rcut  -  Cutoff for the short-ranged part of the electrostatics calculation
    #
    # Using set_coeff() requires that the specified PPPM force has been saved in a variable. i.e.
    # \code
    # pppm = pair.pppm()
    # \endcode
    #
    # \b Examples:
    # \code
    # pppm.set_coeff(Nx=64, Ny=64, Nz=64, order=6, kappa=1.5, rcut=2.0)
    # \endcode
    #
    # The coefficients for PPPM  must be set 
    # before the run() can be started.
    def set_coeff(self, Nx, Ny, Nz, order, kappa, rcut):
        util.print_status_line();

        ewald = pair.ewald(r_cut = rcut)
        ntypes = globals.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in xrange(0,ntypes):
            type_list.append(globals.system_definition.getParticleData().getNameByType(i));

        for i in xrange(0,ntypes):
            for j in xrange(0,ntypes):
                ewald.pair_coeff.set(type_list[i], type_list[j], kappa = kappa)

        # set the parameters for the appropriate type
        self.cpp_force.setParams(Nx, Ny, Nz, order, kappa, rcut);

    def update_coeffs(self):
        pass
