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

# Maintainer: jglaser / All Developers are free to add commands for new features

## \package hoomd_script.ai_pair
# \brief Commands that create forces and torques between pairs of anisotropic particles
#
# Generally, anisotropic %pair forces are short range and are summed over all non-bonded particles
# within a certain cutoff radius of each particle. Any number of anisotropic %pair forces
# can be defined in a single simulation. The net %force and torque on each particle due to
# all types of anisotropic %pair forces is summed.
#
# Anisotropic pair forces require that parameters be set for each unique type %pair. This is done
# with the same syntax as for isotropic pair forces using the coeff class,
# \link hoomd_script.pair.coeff.set() pair_coeff.set()\endlink.
#
# \sa \ref page_quick_start

from hoomd_script import pair
from hoomd_script import force
from hoomd_script import util
from hoomd_script import globals
from hoomd_script import data
import hoomd
import math

## Generic anisotropic %pair potential
#
# ai_pair.ai_pair is not a command hoomd scripts should execute directly. Rather, it is a base command that
# provides common features to all anisotropic %pair forces. Rather than repeating all of that documentation in a
# dozen different places, it is collected here.
#
# All anisotropic %pair potential commands specify that a given potential energy, %force and torque be computedi
# on all particle pairs in the system within a short range cutoff distance \f$ r_{\mathrm{cut}} \f$.
# The interaction energy, forces and torque depend on the inter-particle separation
# \f$ \vec r \f$ and on the orientations \f$\vec e_i, \vec e_j\f$, of the particles.
#
# The following coefficients must be set per unique %pair of particle types. See hoomd_script.pair or 
# the \ref page_quick_start for information on how to set coefficients.
# - \f$ r_{\mathrm{cut}} \f$ - \c r_cut (in distance units)
#   - <i>optional</i>: defaults to the global r_cut specified in the %pair command
#
class ai_pair(force._force):
    ## \internal
    # \brief Initialize the pair force
    # \details
    # The derived class must set
    #  - self.cpp_class (the pair class to instantiate)
    #  - self.required_coeffs (a list of the coeff names the derived class needs)
    #  - self.process_coeffs() (a method that takes in the coeffs and spits out a param struct to use in 
    #       self.cpp_force.set_params())
    def __init__(self, r_cut, name=None):
        # initialize the base class
        force._force.__init__(self, name);

        self.global_r_cut = r_cut;

        # setup the coefficent matrix
        self.pair_coeff = pair.coeff();
        self.pair_coeff.set_default_coeff('r_cut', self.global_r_cut);
 
    ## Set parameters controlling the way forces are computed
    #
    # \param mode (if set) Set the mode with which potentials are handled at the cutoff
    #
    # valid values for \a mode are: "none" (the default) and "shift"
    #  - \b none - No shifting is performed and potentials are abruptly cut off
    #  - \b shift - A constant shift is applied to the entire potential so that it is 0 at the cutoff
    #
    # \b Examples:
    # \code
    # mypair.set_params(mode="shift")
    # mypair.set_params(mode="no_shift")
    # \endcode
    # 
    def set_params(self, mode=None):
        util.print_status_line();
        
        if mode is not None:
            if mode == "no_shift":
                self.cpp_force.setShiftMode(self.cpp_class.energyShiftMode.no_shift)
            elif mode == "shift":
                self.cpp_force.setShiftMode(self.cpp_class.energyShiftMode.shift)
            else:
                globals.msg.error("Invalid mode\n");
                raise RuntimeError("Error changing parameters in pair force");
     
    def process_coeff(self, coeff):
        globals.msg.error("Bug in hoomd_script, please report\n");
        raise RuntimeError("Error processing coefficients");
    
    def update_coeffs(self):
        coeff_list = self.required_coeffs + ["r_cut"];
        # check that the pair coefficents are valid
        if not self.pair_coeff.verify(coeff_list):
            globals.msg.error("Not all pair coefficients are set\n");
            raise RuntimeError("Error updating pair coefficients");
        
        # set all the params
        ntypes = globals.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(globals.system_definition.getParticleData().getNameByType(i));
        
        for i in range(0,ntypes):
            for j in range(i,ntypes):
                # build a dict of the coeffs to pass to process_coeff
                coeff_dict = {};
                for name in coeff_list:
                    coeff_dict[name] = self.pair_coeff.get(type_list[i], type_list[j], name);
                
                param = self.process_coeff(coeff_dict);
                self.cpp_force.setParams(i, j, param);
                self.cpp_force.setRcut(i, j, coeff_dict['r_cut']);

    ## \internal
    # \brief Get the maximum r_cut value set for any type pair
    # \pre update_coeffs must be called before get_max_rcut to verify that the coeffs are set
    def get_max_rcut(self):
        # go through the list of only the active particle types in the sim
        ntypes = globals.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(globals.system_definition.getParticleData().getNameByType(i));
        
        # find the maximum r_cut
        max_rcut = 0.0;
        
        for i in range(0,ntypes):
            for j in range(i,ntypes):
                # get the r_cut value
                r_cut = self.pair_coeff.get(type_list[i], type_list[j], 'r_cut');
                max_rcut = max(max_rcut, r_cut);
        
        return max_rcut;


## Gay-Berne anisotropic %pair potential
#
# The Gay-Berne potential computes the Lennard-Jones potential between anisotropic particles.
#
# This version of the Gay-Berne potential supports identical pairs of uniaxial ellipsoids,
# with orientation-independent energy-well depth.
#
# The interaction energy for this anisotropic pair potential is (\cite Allen2006):
#
# \f{eqnarray*}
# V_{\mathrm{GB}}(\vec r, \vec e_i, \vec e_j)  = & 4 \varepsilon \left[ \left( \zeta^{-12} - 
#                       \left( \zeta{-6} \right] & \zeta < \zeta_{\mathrm{cut}} \\
#                     = & 0 & \zeta \ge \zeta_{\mathrm{cut}} \\
# \f}, 
# where
# \f{equation*}
# \zeta = \left(\frac{r-\sigma+\sigma_{\mathrm{min}}{\sigma_{\mathrm{min}}}\right)
# \f},
#
# \f{equation*}
# \sigma^{-2} = \frac12 \hat\vec r\cdot\vec H^{-1}\cdot\hat\vec r
# \f},
#
# \f{equation*}
# \vec H = 2 \ell_\perp^2 \vec 1+ (\ell_\par^2 - \ell_\perp^2) (\vec e_i \otimes e_i + \vec e_j \otimes e_j)
# \f},
# and  \f$ \sigma_min = \min(\ell_perp, \ell_par) \f$.
#
# The following coefficients must be set per unique %pair of particle types. See hoomd_script.pair or 
# the \ref page_quick_start for information on how to set coefficients.
# - \f$ \varepsilon \f$ - \c epsilon (in energy units)
# - \f$ \ell_perp \f$ - \c lperp (perpendicular <b>semi</b>-axis length, in distance units)
# - \f$ \ell_par \f$ - \c lpar (parallel <b>semi</b>-axis length, in distance units)
# - \f$ r_{\mathrm{cut}} \f$ - \c r_cut (in distance units)
#   - <i>optional</i>: defaults to the global r_cut specified in the %pair command
#
# pair.gb is an anisotropic %pair potential and supports shifting the energy at the cut-off.
# See hoomd_script.pair.pair for how to set this option.
#
# \b Example:
# \code
# gb.pair_coeff.set('A', 'A', epsilon=1.0, lperp=1.0, lpar=1.5)
# gb.pair_coeff.set('A', 'B', epsilon=2.0, lperp=1.0, lpar=1.5, r_cut=2**(1.0/6.0));
# \endcode
#
# For more information on setting pair coefficients, including examples with <i>wildcards</i>, see
# \link hoomd_script.pair.coeff.set() pair_coeff.set()\endlink.
#
# \MPI_SUPPORTED
class gb(ai_pair):
    ## Specify the Gay-Berne %pair %force and torque
    #
    # \param r_cut Default cutoff radius (in distance units)
    # \param name Name of the force instance 
    #
    # \b Example:
    # \code
    # gb.ai_pair.gb(r_cut=2.5)
    # gb.pair_coeff.set('A', 'A', epsilon=1.0, lperp=1.0, lpar=1.5)
    # gb.pair_coeff.set('A', 'B', epsilon=2.0, lperp=1.0, lpar=1.5, r_cut=2**(1.0/6.0));
    # \endcode
    #
    # \note %Pair coefficients for all type pairs in the simulation must be
    # set before it can be started with run()
    def __init__(self, r_cut, name=None):
        util.print_status_line();
        
        # tell the base class how we operate
        
        # initialize the base class
        ai_pair.__init__(self, r_cut, name);
        
        # update the neighbor list
        neighbor_list = pair._update_global_nlist(r_cut);
        neighbor_list.subscribe(lambda: self.log*self.get_max_rcut())
        
        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = hoomd.AnisoPotentialPairGB(globals.system_definition, neighbor_list.cpp_nlist, self.name);
            self.cpp_class = hoomd.AnisoPotentialPairGB;
        else:
            neighbor_list.cpp_nlist.setStorageMode(hoomd.NeighborList.storageMode.full);
            self.cpp_force = hoomd.AnisoPotentialPairGBGPU(globals.system_definition, neighbor_list.cpp_nlist, self.name);
            self.cpp_class = hoomd.AnisoPotentialPairGBGPU;
            self.cpp_force.setBlockSize(tune._get_optimal_block_size('ai_pair.gb'));
            
        globals.system.addCompute(self.cpp_force, self.force_name);
        
        # setup the coefficent options
        self.required_coeffs = ['epsilon', 'lperp', 'lpar'];
        
    def process_coeff(self, coeff):
        epsilon = coeff['epsilon'];
        lperp = coeff['lperp'];
        lpar = coeff['lpar'];

        sigma_min = 2.0*min(lperp,lpar)
        chi = (lpar*lpar-lperp*lperp)/(lperp*lperp+lpar*lpar);
        lperpsq = lperp*lperp;

        return hoomd.make_scalar4(epsilon, sigma_min, lperpsq, chi);
