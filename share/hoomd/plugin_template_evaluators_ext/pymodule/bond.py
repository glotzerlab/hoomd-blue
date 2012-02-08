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

import _evaluators_ext_template

# Next, since we are extending a bond potential, we need to bring in the base class and some other parts from
# hoomd_script
from hoomd_script import util
from hoomd_script import globals
from hoomd_script import force
import hoomd
import math


## Composite of harmonic %bond force with a DPD pair potential
#
# The command bond.harmonic_dpd specifies a %harmonic_dpd potential energy between every bonded %pair of particles
# in the simulation.
# \f[ V(r) = \frac{1}{2} k \left( r - r_0 \right)^2 + V_{\mathrm{DPD}}(r) \f]
# where \f$ \vec{r} \f$ is the vector pointing from one particle to the other
# in the %pair and
#
# \f{eqnarray*}
# V_{\mathrm{DPD}}(r)  = & A \cdot \left( r_{\mathrm{cut}} - r \right)
#                        - \frac{1}{2} \cdot \frac{A}{r_{\mathrm{cut}}} \cdot \left(r_{\mathrm{cut}}^2 - r^2 \right)
#                               & r < r_{\mathrm{cut}} \\
#                     = & 0 & r \ge r_{\mathrm{cut}} \\
# \f}
#
# Coeffients:
# - \f$ k \f$ - %force constant (in units of energy/distance^2)
# - \f$ r_0 \f$ - %bond rest length (in distance units)
# - \f$ A \f$ - \a A (in force units)
# - \f$ r_{\mathrm{cut}} \f$ - \c r_cut (in distance units)
#
# Coefficients \f$ k \f$, \f$ r_0 \f$, \f$ A \f$ and \f$ r_{\mathrm{cut}} \f$
# must be set for each type of %bond in the simulation using
# set_coeff().
#
# \note Specifying the bond.harmonic_dpd command when no bonds are defined in the simulation results in an error.
class harmonic_dpd(force._force):
    ## Specify the %harmonic_dpd %bond %force
    #
    # \param name Name of the bond instance
    #
    # \b Example:
    # \code
    # harmonic_dpd = bond.harmonic_dpd(name="mybond")
    # \endcode
    def __init__(self,name=None):
        util.print_status_line();

        # check that some bonds are defined
        if globals.system_definition.getBondData().getNumBonds() == 0:
            print >> sys.stderr, "\n***Error! No bonds are defined.\n";
            raise RuntimeError("Error creating bond forces");

        # initialize the base class
        force._force.__init__(self,name);

        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = _evaluators_ext_template.PotentialBondHarmonicDPD(globals.system_definition,self.name);
        else:
            self.cpp_force = _evaluators_ext_template.PotentialBondHarmonicDPDGPU(globals.system_definition,self.name);
            # you can play with the block size value, set it to any multiple of 32 up to 1024. Use the
            # harmonic_dpd.benchmark() command to find out which block size performs the fastest
            self.cpp_force.setBlockSize(64);

        globals.system.addCompute(self.cpp_force, self.force_name);

        # variable for tracking which bond type coefficients have been set
        self.bond_types_set = [];

    ## Sets the %harmonic_dpd %bond coefficients for a particular %bond type
    #
    # \param bond_type Bond type to set coefficients for
    # \param k Coefficient \f$ k \f$ (in units of energy/distance^2)
    # \param r0 Coefficient \f$ r_0 \f$ (in distance units)
    # \param r_cut Coefficient \f$ r_{\mathrm{cut}} \f$ (in distance units)
    # \param A Coefficient \f$ A \f$ (in force units)
    #
    # Using set_coeff() requires that the specified %bond %force has been saved in a variable. i.e.
    # \code
    # harmonic_dpd = bond.harmonic_dpd()
    # \endcode
    #
    # \b Examples:
    # \code
    # harmonic_dpd.set_coeff('polymer', k=330.0, r0=0.84, A=1.0, r_cut=1.0)
    # \endcode
    #
    # The coefficients for every %bond type in the simulation must be set
    # before the run() can be started.
    def set_coeff(self, bond_type, k, r0, r_cut, A):
        util.print_status_line();

        # set the parameters for the appropriate type
        self.cpp_force.setParams(globals.system_definition.getBondData().getTypeByName(bond_type), hoomd.make_scalar4(k, r0, r_cut, A));

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
                print >> sys.stderr, "\n***Error:", cur_type, "coefficients missing in bond.harmonic_dpd\n";
                raise RuntimeError("Error updating coefficients");
