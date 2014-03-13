# -- start license --
# Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
# (HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
# the University of Michigan All rights reserved.

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

from hoomd_plugins.evaluators_ext_template import _evaluators_ext_template

# Next, since we are extending a bond potential, we need to bring in the base class and some other parts from
# hoomd_script
from hoomd_script import util
from hoomd_script import globals
from hoomd_script.bond import _bond
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
class harmonic_dpd(_bond):
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
        _bond.__init__(self,name);

        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = _evaluators_ext_template.PotentialBondHarmonicDPD(globals.system_definition,self.name);
        else:
            self.cpp_force = _evaluators_ext_template.PotentialBondHarmonicDPDGPU(globals.system_definition,self.name);
            # you can play with the block size value, set it to any multiple of 32 up to 1024. Use the
            # harmonic_dpd.benchmark() command to find out which block size performs the fastest
            self.cpp_force.setBlockSize(64);

        globals.system.addCompute(self.cpp_force, self.force_name);

        self.required_coeffs = ['k','r0','r_cut', 'A'];

    def process_coeff(self, coeff):
        k = coeff['k'];
        r0 = coeff['r0'];
        A = coeff['A'];
        r_cut = coeff['r_cut'];

        return hoomd.make_scalar4(k, r0, r_cut, A);
