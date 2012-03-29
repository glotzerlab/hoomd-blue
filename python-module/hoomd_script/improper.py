# -- start license --
# Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
# (HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
# Iowa State University and The Regents of the University of Michigan All rights
# reserved.

# HOOMD-blue may contain modifications ("Contributions") provided, and to which
# copyright is held, by various Contributors who have granted The Regents of the
# University of Michigan the right to modify and/or distribute such Contributions.

# You may redistribute, use, and create derivate works of HOOMD-blue, in source
# and binary forms, provided you abide by the following conditions:

# * Redistributions of source code must retain the above copyright notice, this
# list of conditions, and the following disclaimer both in the code and
# prominently in any materials provided with the distribution.

# * Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions, and the following disclaimer in the documentation and/or
# other materials provided with the distribution.

# * All publications and presentations based on HOOMD-blue, including any reports
# or published results obtained, in whole or in part, with HOOMD-blue, will
# acknowledge its use according to the terms posted at the time of submission on:
# http://codeblue.umich.edu/hoomd-blue/citations.html

# * Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
# http://codeblue.umich.edu/hoomd-blue/

# * Apart from the above required attributions, neither the name of the copyright
# holder nor the names of HOOMD-blue's contributors may be used to endorse or
# promote products derived from this software without specific prior written
# permission.

# Disclaimer

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
# WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -- end license --

# Maintainer: joaander / All Developers are free to add commands for new features

import force;
import globals;
import hoomd;
import util;
import tune;

import math;
import sys;

## \package hoomd_script.improper
# \brief Commands that specify %improper forces
#
# Impropers add forces between specified quadruplets of particles and are typically used to 
# model rotation about chemical bonds without having bonds to connect the atoms. Their most
# common use is to keep structural elements flat, i.e. model the effect of conjugated
# double bonds, like in benzene rings and its derivatives.
# Impropers between particles are set when an input XML file is read
# (init.read_xml) or when an another initializer creates them (like init.create_random_polymers)
#
# By themselves, impropers that have been specified in an input file do nothing. Only when you 
# specify an improper force (i.e. improper.harmonic), are forces actually calculated between the 
# listed particles.

## Harmonic %improper force
#
# The command improper.harmonic specifies a %harmonic improper potential energy between every quadruplet of particles
# in the simulation. 
# \f[ V(r) = \frac{1}{2}k \left( \chi - \chi_{0}  \right )^2 \f]
# where \f$ \chi \f$ is angle between two sides of the improper
#
# Coefficients:
# - \f$ k \f$ - strength of %force (in energy units)
# - \f$ \chi_{0} \f$ - equilibrium angle (in radians)
#
# Coefficients \f$ k \f$ and \f$ \chi_0 \f$ must be set for each type of %improper in the simulation using
# set_coeff().
#
# \note Specifying the improper.harmonic command when no impropers are defined in the simulation results in an error.
class harmonic(force._force):
    ## Specify the %harmonic %improper %force
    #
    # \b Example:
    # \code
    # harmonic = improper.harmonic()
    # \endcode
    def __init__(self):
        util.print_status_line();
        # check that some impropers are defined
        if globals.system_definition.getImproperData().getNumDihedrals() == 0:
            globals.msg.error("No impropers are defined.\n");
            raise RuntimeError("Error creating improper forces");
        
        # initialize the base class
        force._force.__init__(self);
        
        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = hoomd.HarmonicImproperForceCompute(globals.system_definition);
        else:
            self.cpp_force = hoomd.HarmonicImproperForceComputeGPU(globals.system_definition);
            self.cpp_force.setBlockSize(tune._get_optimal_block_size('improper.harmonic'));

        globals.system.addCompute(self.cpp_force, self.force_name);
        
        # variable for tracking which improper type coefficients have been set
        self.improper_types_set = [];
    
    ## Sets the %harmonic %improper coefficients for a particular %improper type
    #
    # \param improper_type Improper type to set coefficients for
    # \param k Coefficient \f$ k \f$ in the %force
    # \param chi Coefficient \f$ \chi \f$ in the %force
    #
    # Using set_coeff() requires that the specified %improper %force has been saved in a variable. i.e.
    # \code
    # harmonic = improper.harmonic()
    # \endcode
    #
    # \b Examples:
    # \code
    # harmonic.set_coeff('heme-ang', k=30.0, chi=1.57)
    # harmonic.set_coeff('hdyro-bond', k=20.0, chi=1.57)
    # \endcode
    #
    # The coefficients for every %improper type in the simulation must be set 
    # before the run() can be started.
    def set_coeff(self, improper_type, k, chi):
        util.print_status_line();
        
        # set the parameters for the appropriate type
        self.cpp_force.setParams(globals.system_definition.getImproperData().getTypeByName(improper_type), k, chi);
        
        # track which particle types we have set
        if not improper_type in self.improper_types_set:
            self.improper_types_set.append(improper_type);
        
    def update_coeffs(self):
        # get a list of all improper types in the simulation
        ntypes = globals.system_definition.getImproperData().getNDihedralTypes();
        type_list = [];
        for i in xrange(0,ntypes):
            type_list.append(globals.system_definition.getImproperData().getNameByType(i));
            
        # check to see if all particle types have been set
        for cur_type in type_list:
            if not cur_type in self.improper_types_set:
                globals.msg.error(str(cur_type) + " coefficients missing in improper.harmonic\n");
                raise RuntimeError("Error updating coefficients");

