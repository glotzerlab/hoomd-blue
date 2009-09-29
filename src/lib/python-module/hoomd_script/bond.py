# -*- coding: iso-8859-1 -*-
# Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
# Source Software License
# Copyright (c) 2008 Ames Laboratory Iowa State University
# All rights reserved.

# Redistribution and use of HOOMD, in source and binary forms, with or
# without modification, are permitted, provided that the following
# conditions are met:

# * Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names HOOMD's
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.

# Disclaimer

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
# CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 

# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.

# $Id$
# $URL$
# Maintainer: joaander / All Developers are free to add commands for new features

import force;
import globals;
import hoomd;
import util;
import tune;

import math;
import sys;

## \package hoomd_script.bond
# \brief Commands that specify %bond forces
#
# Bonds add forces between specified pairs of particles and are typically used to 
# model chemical bonds. Bonds between particles are set when an input XML file is read
# (init.read_xml) or when an another initializer creates them (like init.create_random_polymers)
#
# By themselves, bonds that have been specified in an input file do nothing. Only when you 
# specify a bond force (i.e. bond.harmonic), are forces actually calculated between the 
# listed particles.

## Harmonic %bond forces
#
# The command bond.harmonic specifies a %harmonic potential energy between every bonded %pair of particles
# in the simulation. 
# \f[ V(r) = \frac{1}{2} k \left( r - r_0 \right)^2 \f]
# where \f$ \vec{r} \f$ is the vector pointing from one particle to the other in the %pair.
#
# Coefficients \f$ k \f$ and \f$ r_0 \f$ must be set for each type of %bond in the simulation using
# set_coeff().
#
# \note Specifying the bond.harmonic command when no bonds are defined in the simulation results in an error.
class harmonic(force._force):
    ## Specify the %harmonic %bond %force
    #
    # \b Example:
    # \code
    # harmonic = bond.harmonic()
    # \endcode
    def __init__(self):
        util.print_status_line();
        
        # check that some bonds are defined
        if globals.system_definition.getBondData().getNumBonds() == 0:
            print >> sys.stderr, "\n***Error! No bonds are defined.\n";
            raise RuntimeError("Error creating bond forces");		
        
        # initialize the base class
        force._force.__init__(self);
        
        # create the c++ mirror class
        if globals.system_definition.getParticleData().getExecConf().exec_mode == hoomd.ExecutionConfiguration.executionMode.CPU:
            self.cpp_force = hoomd.HarmonicBondForceCompute(globals.system_definition);
        elif globals.system_definition.getParticleData().getExecConf().exec_mode == hoomd.ExecutionConfiguration.executionMode.GPU:
            self.cpp_force = hoomd.HarmonicBondForceComputeGPU(globals.system_definition);
            self.cpp_force.setBlockSize(tune._get_optimal_block_size('bond.harmonic'));
        else:
            print >> sys.stderr, "\n***Error! Invalid execution mode\n";
            raise RuntimeError("Error creating bond forces");

        globals.system.addCompute(self.cpp_force, self.force_name);
        
        # variable for tracking which bond type coefficients have been set
        self.bond_types_set = [];
    
    ## Sets the %harmonic %bond coefficients for a particular %bond type
    #
    # \param bond_type Bond type to set coefficients for
    # \param k Coefficient \f$ k \f$ in the %force
    # \param r0 Coefficient \f$ r_0 \f$ in the %force
    #
    # Using set_coeff() requires that the specified %bond %force has been saved in a variable. i.e.
    # \code
    # harmonic = bond.harmonic()
    # \endcode
    #
    # \b Examples:
    # \code
    # harmonic.set_coeff('polymer', k=330.0, r0=0.84)
    # harmonic.set_coeff('backbone', k=100.0, r0=1.0)
    # \endcode
    #
    # The coefficients for every %bond type in the simulation must be set 
    # before the run() can be started.
    def set_coeff(self, bond_type, k, r0):
        util.print_status_line();
        
        # set the parameters for the appropriate type
        self.cpp_force.setParams(globals.system_definition.getBondData().getTypeByName(bond_type), k, r0);
        
        # track which particle types we have set
        if not bond_type in self.bond_types_set:
            self.bond_types_set.append(bond_type);
        
    def update_coeffs(self):
        # get a list of all bond types in the simulation
        ntypes = globals.system_definition.getBondData().getNBondTypes();
        type_list = [];
        for i in xrange(0,ntypes):
            type_list.append(globals.system_definition.getBondData().getNameByType(i));
            
        # check to see if all particle types have been set
        for cur_type in type_list:
            if not cur_type in self.bond_types_set:
                print >> sys.stderr, "\n***Error:", cur_type, " coefficients missing in bond.harmonic\n";
                raise RuntimeError("Error updating coefficients");



## FENE %bond forces
#
# The command bond.fene specifies a %fene potential energy between every bonded %pair of particles
# in the simulation. 
# \f[ V(r) = - k r_0^2 \ln \left( 1 - \left( \frac{r}{r_0} \right)^2 \right) + V_{\mathrm{WCA}}(r)\f]
# where \f$ \vec{r} \f$ is the vector pointing from one particle to the other in the %pair and
# \f{eqnarray*}
#   V_{\mathrm{WCA}}(r)  = & 4 \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{12} - \left( \frac{\sigma}{r} \right)^{6} \right] & r < 2^{\frac{1}{6}}\sigma \\
#            = & 0          & r \ge 2^{\frac{1}{6}}\sigma    \\
#   \f}
#
# Coefficients \f$ k \f$, \f$ r_0 \f$, \f$ \varepsilon \f$ and \f$ \sigma \f$  must be set for 
# each type of %bond in the simulation using set_coeff().
#
# \note Specifying the bond.fene command when no bonds are defined in the simulation results in an error.
class fene(force._force):
    ## Specify the %fene %bond %force
    #
    # \b Example:
    # \code
    # fene = bond.fene()
    # \endcode
    def __init__(self):
        util.print_status_line();
        
        # check that some bonds are defined
        if globals.system_definition.getBondData().getNumBonds() == 0:
            print >> sys.stderr, "\n***Error! No bonds are defined.\n";
            raise RuntimeError("Error creating bond forces");		
        
        # initialize the base class
        force._force.__init__(self);
        
        # create the c++ mirror class
        if globals.system_definition.getParticleData().getExecConf().exec_mode == hoomd.ExecutionConfiguration.executionMode.CPU:
            self.cpp_force = hoomd.FENEBondForceCompute(globals.system_definition);
        elif globals.system_definition.getParticleData().getExecConf().exec_mode == hoomd.ExecutionConfiguration.executionMode.GPU:
            self.cpp_force = hoomd.FENEBondForceComputeGPU(globals.system_definition);
            self.cpp_force.setBlockSize(tune._get_optimal_block_size('bond.fene'));
        else:
            print >> sys.stderr, "\n***Error! Invalid execution mode\n";
            raise RuntimeError("Error creating bond forces");

        globals.system.addCompute(self.cpp_force, self.force_name);
        
        # variable for tracking which bond type coefficients have been set
        self.bond_types_set = [];
    
    ## Sets the %fene %bond coefficients for a particular %bond type
    #
    # \param bond_type Bond type to set coefficients for
    # \param k Coefficient \f$ k \f$ in the %force
    # \param r0 Coefficient \f$ r_0 \f$ in the %force
    # \param sigma Coefficient \f$ \sigma \f$ in the %force
    # \param epsilon Coefficient \f$ \epsilon \f$ in the %force
    #
    # Using set_coeff() requires that the specified %bond %force has been saved in a variable. i.e.
    # \code
    # fene = bond.fene()
    # \endcode
    #
    # \b Examples:
    # \code
    # fene.set_coeff('polymer', k=30.0, r0=1.5, sigma=1.0, epsilon= 2.0)
    # fene.set_coeff('backbone', k=100.0, r0=1.0, sigma=1.0, epsilon= 2.0)
    # \endcode
    #
    # The coefficients for every %bond type in the simulation must be set 
    # before the run() can be started.
    def set_coeff(self, bond_type, k, r0, sigma, epsilon):
        util.print_status_line();
        
        self.cpp_force.setParams(globals.system_definition.getBondData().getTypeByName(bond_type), k, r0, sigma, epsilon);
        # track which particle types we have set
        if not bond_type in self.bond_types_set:
            self.bond_types_set.append(bond_type);
        
    def update_coeffs(self):
        # get a list of all bond types in the simulation
        ntypes = globals.system_definition.getBondData().getNBondTypes();
        type_list = [];
        for i in xrange(0,ntypes):
            type_list.append(globals.system_definition.getBondData().getNameByType(i));
            
        # check to see if all particle types have been set
        for cur_type in type_list:
            if not cur_type in self.bond_types_set:
                print >> sys.stderr, "\n***Error:", cur_type, " coefficients missing in bond.fene\n";
                raise RuntimeError("Error updating coefficients");

