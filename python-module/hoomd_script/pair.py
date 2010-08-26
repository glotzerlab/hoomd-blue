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

# $Id$
# $URL$
# Maintainer: joaander / All Developers are free to add commands for new features

## \package hoomd_script.pair
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
# This example set the parameters \a epsilon, \a sigma, and \a alpha 
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

import math;
import sys;

## Defines %pair coefficients
# 
# All %pair forces use coeff to specify the coefficients between different
# pairs of particles indexed by type. The set of %pair coefficients is a symmetric
# matrix defined over all possible pairs of particle types.
#
# There are two ways to set the coefficients for a particular %pair %force. 
# The first way is to save the %pair %force in a variable and call set() directly.
# To see an example of this, see the documentation for the package pair
# or the \ref page_quick_start
#
# The second method is to build the coeff class first and then assign it to the
# %pair %force. There are some advantages to this method in that you could specify a
# complicated set of %pair coefficients in a separate python file and import it into
# your job script.
#
# Example (file \em force_field.py):
# \code
# from hoomd_script import *
# my_coeffs = pair.coeff();
# my_force.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
# my_force.pair_coeff.set('A', 'B', epsilon=1.0, sigma=2.0)
# my_force.pair_coeff.set('B', 'B', epsilon=2.0, sigma=1.0)
# \endcode
# Example job script:
# \code
# from hoomd_script import *
# import force_field
#
# .....
# my_force = pair.some_pair_force(arguments...)
# my_force.pair_coeff = force_field.my_coeffs
# \endcode
class coeff:
    
    ## \internal
    # \brief Initializes the class
    # \details
    # The main task to be performed during initialization is just to init some variables
    # \param self Python required class instance variable
    def __init__(self):
        self.values = {};
        self.default_coeff = {}
        
    ## \var values
    # \internal
    # \brief Contains the matrix of set values in a dictionary
    
    ## \var default_coeff
    # \internal
    # \brief default_coeff['coeff'] lists the default value for \a coeff, if it is set
    
    ## \internal
    # \brief Sets a default value for a given coefficient
    # \details 
    # \param name Name of the coefficient to for which to set the default
    # \param value Default value to set
    #
    # Some coefficients have reasonable default values and the user should not be burdened with typing them in
    # all the time. set_default_coeff() sets
    def set_default_coeff(self, name, value):
        self.default_coeff[name] = value;
    
    ## Sets parameters for one type %pair
    # \param a First particle type in the %pair
    # \param b Second particle type in the %pair
    # \param coeffs Named coefficients (see below for examples)
    #
    # Calling set() results in one or more parameters being set for a single type %pair.
    # Particle types are identified by name, and parameters are also added by name. 
    # Which parameters you need to specify depends on the %pair %force you are setting
    # these coefficients for, see the corresponding documentation.
    #
    # All possible type pairs as defined in the simulation box must be specified before
    # executing run(). You will receive an error if you fail to do so. It is not an error,
    # however, to specify coefficients for particle types that do not exist in the simulation.
    # This can be useful in defining a %force field for many different types of particles even
    # when some simulations only include a subset.
    #
    # There is no need to specify coefficients for both pairs 'A','B' and 'B','A'. Specifying
    # only one is sufficient.
    #
    # \b Examples:
    # \code
    # coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
    # coeff.set('B', 'B', epsilon=2.0, sigma=1.0)
    # coeff.set('A', 'B', epsilon=1.5, sigma=1.0)
    # \endcode
    #
    # \note Single parameters can be updated. If both epsilon and sigma have already been 
    # set for a type %pair, then executing coeff.set('A', 'B', epsilon=1.1) will %update 
    # the value of epsilon and leave sigma as it was previously set.
    #
    # Some %pair potentials assign default values to certain parameters. If the default setting for a given coefficient
    # (as documented in the respective %pair command), it does not need to be listed on the coeff.set() line at all
    # and the default value will automatically be set.
    #
    def set(self, a, b, **coeffs):
        util.print_status_line();
        
        # create the pair if it hasn't been created it
        if (not (a,b) in self.values) and (not (b,a) in self.values):
            self.values[(a,b)] = {};
            
        # Find the pair to update
        if (a,b) in self.values:
            cur_pair = (a,b);
        elif (b,a) in self.values:
            cur_pair = (b,a);
        else:
            print >> sys.stderr, "\nBug detected in pair.coeff. Please report\n"
            raise RuntimeError("Error setting pair coeff");
        
        # update each of the values provided
        if len(coeffs) == 0:
            print >> sys.stderr, "\n***Error! No coefficents specified\n";
        for name, val in coeffs.items():
            self.values[cur_pair][name] = val;
        
        # set the default values
        for name, val in self.default_coeff.items():
            # don't override a coeff if it is already set
            if not name in self.values[cur_pair]:
                self.values[cur_pair][name] = val;
    
    ## \internal
    # \brief Verifies set parameters form a full matrix with all values set
    # \details
    # \param self Python required self variable
    # \param required_coeffs list of required variables
    #
    # This can only be run after the system has been initialized
    def verify(self, required_coeffs):
        # first, check that the system has been initialized
        if not init.is_initialized():
            print >> sys.stderr, "\n***Error! Cannot verify pair coefficients before initialization\n";
            raise RuntimeError('Error verifying pair coefficients');
        
        # get a list of types from the particle data
        ntypes = globals.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in xrange(0,ntypes):
            type_list.append(globals.system_definition.getParticleData().getNameByType(i));
        
        valid = True;
        # loop over all possible pairs and verify that all required variables are set
        for i in xrange(0,ntypes):
            for j in xrange(i,ntypes):
                a = type_list[i];
                b = type_list[j];
                
                # find which half of the pair is set
                if (a,b) in self.values:
                    cur_pair = (a,b);
                elif (b,a) in self.values:
                    cur_pair = (b,a);
                else:
                    print >> sys.stderr, "\n***Error! Type pair", (a,b), "not found in pair coeff\n"
                    valid = False;
                    continue;
                
                # verify that all required values are set by counting the matches
                count = 0;
                for coeff_name in self.values[cur_pair].keys():
                    if not coeff_name in required_coeffs:
                        print "Notice: Possible typo? Pair coeff", coeff_name, "is specified for pair", (a,b), \
                              ", but is not used by the pair force";
                    else:
                        count += 1;
                
                if count != len(required_coeffs):
                    print >> sys.stderr, "\n***Error! Type pair", (a,b), "is missing required coefficients\n";
                    valid = False;
                
            
        return valid;
        
    ## \internal
    # \brief Gets the value of a single %pair coefficient
    # \detail
    # \param a First name in the type pair
    # \param b Second name in the type pair
    # \param coeff_name Coefficient to get
    def get(self, a, b, coeff_name):
        # Find the pair to update
        if (a,b) in self.values:
            cur_pair = (a,b);
        elif (b,a) in self.values:
            cur_pair = (b,a);
        else:
            print >> sys.stderr, "\nBug detected in pair.coeff. Please report\n"
            raise RuntimeError("Error setting pair coeff");
        
        return self.values[cur_pair][coeff_name];
        
        
## Interface for controlling neighbor list parameters
#
# A neighbor list should not be directly created by the user. One will be automatically
# created when the first %pair %force is specified. The cutoff radius is set to the
# maximum of that set for all defined %pair forces.
#
# Any bonds defined in the simulation are automatically used to exclude bonded particle
# pairs from appearing in the neighbor list. Use the command reset_exclusions() to change this behavior.
#
# Neighborlists are properly and efficiently calculated in 2D simulations if the z dimension of the box is small,
# but non-zero and the dimensionally of the system is set \b before the first pair force is specified.
#
class nlist:
    ## \internal
    # \brief Constructs a neighbor list
    # \details
    # \param self Python required instance variable
    # \param r_cut Cutoff radius
    def __init__(self, r_cut):
        # check if initialization has occured
        if not init.is_initialized():
            print >> sys.stderr, "\n***Error!Cannot create neighbor list before initialization\n";
            raise RuntimeError('Error creating neighbor list');
        
        # decide wether to create an all-to-all neighbor list or a binned one based on box size:
        default_r_buff = 0.8;
        
        mode = "binned";
        
        box = globals.system_definition.getParticleData().getBox();
        min_width_for_bin = (default_r_buff + r_cut)*3.0;
        
        # only check the z dimesion of the box in 3D systems
        is_small_box = (box.xhi - box.xlo) < min_width_for_bin or (box.yhi - box.ylo) < min_width_for_bin;
        if globals.system_definition.getNDimensions() == 3:
            is_small_box = is_small_box or (box.zhi - box.zlo) < min_width_for_bin;
        if  is_small_box:
            if globals.system_definition.getParticleData().getN() >= 2000:
                print "\n***Warning!: At least one simulation box dimension is less than (r_cut + r_buff)*3.0. This forces the use of an";
                print "             EXTREMELY SLOW O(N^2) calculation for the neighbor list."
            else:
                print "Notice: The system is in a very small box, forcing the use of an O(N^2) neighbor list calculation."
                
            mode = "nsq";
        
        # create the C++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            if mode == "binned":
                cl_c = hoomd.CellList(globals.system_definition);
                globals.system.addCompute(cl_c, "auto_cl")
                self.cpp_nlist = hoomd.NeighborListBinned(globals.system_definition, r_cut, default_r_buff, cl_c)
            elif mode == "nsq":
                self.cpp_nlist = hoomd.NeighborList(globals.system_definition, r_cut, default_r_buff)
            else:
                print >> sys.stderr, "\n***Error! Invalid neighbor list mode\n";
                raise RuntimeError("Error creating neighbor list");
        else:
            if mode == "binned":
                cl_g = hoomd.CellListGPU(globals.system_definition);
                globals.system.addCompute(cl_g, "auto_cl")
                self.cpp_nlist = hoomd.NeighborListGPUBinned(globals.system_definition, r_cut, default_r_buff, cl_g)
                self.cpp_nlist.setBlockSize(tune._get_optimal_block_size('nlist'));
            elif mode == "nsq":
                self.cpp_nlist = hoomd.NeighborListGPU(globals.system_definition, r_cut, default_r_buff)
            else:
                print >> sys.stderr, "\n***Error! Invalid neighbor list mode\n";
                raise RuntimeError("Error creating neighbor list");
            
        self.cpp_nlist.setEvery(1);
        self.is_exclusion_overridden = False;
        
        globals.system.addCompute(self.cpp_nlist, "auto_nlist");
        
        # save the parameters we set
        self.r_cut = r_cut;
        self.r_buff = default_r_buff;
        
        # save a list of subscribers that may have a say in determining the maximum r_cut
        self.subscriber_callbacks = [];
    
    ## \internal
    # \brief Adds a subscriber to the neighbor list
    # \param callable is a 0 argument callable object that returns the minimum r_cut needed by the subscriber
    # All \a callables will be called at the beginning of each run() to determine the maximum r_cut needed for that run.
    #
    def subscribe(self, callable):
        self.subscriber_callbacks.append(callable);
        
    ## \internal
    # \brief Updates r_cut based on the subscriber's requests
    #
    def update_rcut(self):
        r_cut_max = 0.0;
        for c in self.subscriber_callbacks:
            r_cut_max = max(r_cut_max, c());
        
        self.r_cut = r_cut_max;
        self.cpp_nlist.setRCut(self.r_cut, self.r_buff);
    
    ## \internal
    # \brief Sets the default bond exclusions, but only if the defaults have not been overridden
    def update_exclusions_defaults(self):
        if not self.is_exclusion_overridden:
            util._disable_status_lines = True;
            self.reset_exclusions(exclusions=['bond']);
            util._disable_status_lines = False;
    
    ## Change neighbor list parameters
    # 
    # \param r_buff (if set) changes the buffer radius around the cutoff
    # \param check_period (if set) changes the period (in time steps) between checks to see if the neighbor list 
    #        needs updating
    # 
    # set_params() changes one or more parameters of the neighbor list. \a r_buff and \a check_period 
    # can have a significant effect on performance. As \a r_buff is made larger, the neighbor list needs
    # to be updated less often, but more particles are included leading to slower %force computations. 
    # Smaller values of \a r_buff lead to faster %force computation, but more often neighbor list updates,
    # slowing overall performance again. The sweet spot for the best performance needs to be found by 
    # experimentation. The default of \a r_buff = 0.8 works well in practice for Lennard-Jones liquid
    # simulations.
    #
    # As \a r_buff is changed, \a check_period must be changed correspondingly. The neighbor list is updated
    # no sooner than \a check_period time steps after the last %update. If \a check_period is set too high,
    # the neighbor list may not be updated when it needs to be. 
    #
    # For safety, the default check_period is 1 to ensure that the neighbor list is always updated when it
    # needs to be. Increasing this to an appropriate value for your simulation can lead to performance gains
    # of approximately 2 percent.
    #
    # \a check_period should be set so that no particle
    # moves a distance more than \a r_buff/2.0 during a the \a check_period. If this occurs, a \b dangerous
    # \b build is counted and printed in the neighbor list statistics at the end of a run().
    #
    # A single global neighbor list is created for the entire simulation. Change parameters by using
    # the built-in variable \b %nlist.
    #
    # \b Examples:
    # \code 
    # nlist.set_params(r_buff = 0.9)
    # nlist.set_params(check_period = 11)
    # nlist.set_params(r_buff = 0.7, check_period = 4)
    # \endcode
    def set_params(self, r_buff=None, check_period=None):
        util.print_status_line();
        
        if self.cpp_nlist is None:
            print >> sys.stderr, "\nBug in hoomd_script: cpp_nlist not set, please report\n";
            raise RuntimeError('Error setting neighbor list parameters');
        
        # update the parameters
        if r_buff is not None:
            self.cpp_nlist.setRCut(self.r_cut, r_buff);
            self.r_buff = r_buff;
            
        if check_period is not None:
            self.cpp_nlist.setEvery(check_period);

    ## Resets all exclusions in the neighborlist
    #
    # \param exclusions Select which interactions should be excluded from the %pair interaction calculation.
    #
    # By default, only directly bonded particles are excluded from short range %pair interactions.
    # reset_exclusions allows that setting to be overridden to add other exclusions or to remove
    # the exclusion for bonded particles.
    #
    # Specify a list of desired types in the \a exclusions argument (or an empty list to clear all exclusions).
    # All desired exclusions have to be explicitly listed, i.e. '1-3' does \b not imply '1-2'.
    # 
    # Valid types are:
    # - \b %bond - Exclude particles that are directly bonded together
    # - \b %angle - Exclude the two outside particles in all defined angles.
    # - \b %dihedral - Exclude the two outside particles in all defined dihedrals.
    #
    # The following types are determined solely by the bond topology. Every chain of particles in the simulation 
    # connected by bonds (1-2-3-4) will be subject to the following exclusions, if enabled, whether or not explicit 
    # angles or dihedrals are defined.
    # - \b 1-2  - Same as bond
    # - \b 1-3  - Exclude particles connected with a sequence of two bonds.
    # - \b 1-4  - Exclude particles connected with a sequence of three bonds.
    #
    # \b WARNING: 
    # 1-4 exclusions currently cannot work due to a limit of 4 exclusions per
    # atom and even 1-3 exclusions can reach that limit in branched molecules.
    #
    # \b Examples:
    # \code 
    # nlist.reset_exclusions(exclusions = ['1-2'])
    # nlist.reset_exclusions(exclusions = ['1-2', '1-3', '1-4'])
    # nlist.reset_exclusions(exclusions = ['bond', 'angle'])
    # nlist.reset_exclusions(exclusions = [])
    # \endcode
    # 
    def reset_exclusions(self, exclusions = None):
        util.print_status_line();
        self.is_exclusion_overridden = True;
        
        if self.cpp_nlist is None:
            print >> sys.stderr, "\nBug in hoomd_script: cpp_nlist not set, please report\n";
            raise RuntimeError('Error resetting exclusions');
        
        # clear all of the existing exclusions
        self.cpp_nlist.clearExclusions();
        
        if exclusions is None:
            # confirm that no exclusions are left.
            self.cpp_nlist.countExclusions();
            return
        
        # exclusions given directly in bond/angle/dihedral notation
        if 'bond' in exclusions:
            self.cpp_nlist.addExclusionsFromBonds();
            exclusions.remove('bond');
        
        if 'angle' in exclusions:
            self.cpp_nlist.addExclusionsFromAngles();
            exclusions.remove('angle');
        
        if 'dihedral' in exclusions:
            self.cpp_nlist.addExclusionsFromDihedrals();
            exclusions.remove('dihedral');
        
        # exclusions given in 1-2/1-3/1-4 notation.
        if '1-2' in exclusions:
            self.cpp_nlist.addExclusionsFromBonds();
            exclusions.remove('1-2');

        if '1-3' in exclusions:
            self.cpp_nlist.addOneThreeExclusionsFromTopology();
            exclusions.remove('1-3');
            
        if '1-4' in exclusions:
            self.cpp_nlist.addOneFourExclusionsFromTopology();
            exclusions.remove('1-4');

        # if there are any items left in the exclusion list, we have an error.
        if len(exclusions) > 0:
            print >> sys.stderr, "\nExclusion type(s):", exclusions, "are not supported\n";
            raise RuntimeError('Error resetting exclusions');

        # collect and print statistics about the number of exclusions.
        self.cpp_nlist.countExclusions();

    ## Benchmarks the neighbor list computation
    # \param n Number of iterations to average the benchmark over
    #
    # \b Examples:
    # \code
    # t = nlist.benchmark(n = 100)
    # \endcode
    #
    # The value returned by benchmark() is the average time to perform the neighbor list 
    # computation, in milliseconds. The benchmark is performed by taking the current
    # positions of all particles in the simulation and repeatedly calculating the neighbor list.
    # Thus, you can benchmark different situations as you need to by simply 
    # running a simulation to achieve the desired state before running benchmark().
    #
    # \note
    # There is, however, one subtle side effect. If the benchmark() command is run 
    # directly after the particle data is initialized with an init command, then the 
    # results of the benchmark will not be typical of the time needed during the actual
    # simulation. Particles are not reordered to improve cache performance until at least
    # one time step is performed. Executing run(1) before the benchmark will solve this problem.
    #
    def benchmark(self, n):
        # check that we have been initialized properly
        if self.cpp_nlist is None:
            print >> sys.stderr, "\nBug in hoomd_script: cpp_nlist not set, please report\n";
            raise RuntimeError('Error benchmarking neighbor list');
        
        # run the benchmark
        return self.cpp_nlist.benchmark(int(n))
            
## \internal
# \brief Creates the global neighbor list
# \details
# \param r_cut Cutoff radius to set
# If no neighbor list has been created, create one. If there is one, increase its r_cut value
# to be the maximum of the current and the one specified here
def _update_global_nlist(r_cut):
    # check to see if we need to create the neighbor list
    if globals.neighbor_list is None:
        globals.neighbor_list = nlist(r_cut);
        # set the global neighbor list using the evil import __main__ trick to provide the user with a default variable
        import __main__;
        __main__.nlist = globals.neighbor_list;
        
    else:
        # otherwise, we need to update r_cut
        new_r_cut = max(r_cut, globals.neighbor_list.r_cut);
        globals.neighbor_list.r_cut = new_r_cut;
        globals.neighbor_list.cpp_nlist.setRCut(new_r_cut, globals.neighbor_list.r_buff);
    
    return globals.neighbor_list;

## Generic %pair %force
#
# pair.pair is not a command hoomd scripts should execute directly. Rather, it is a base command that provides common
# features to all standard %pair forces. Rather than repeating all of that documentation in a dozen different places,
# it is collected here.
#
# All %pair %force commands specify that a given potential energy and %force be computed on all particle pairs in the
# system within a short range cuttoff distance \f$ r_{\mathrm{cut}} \f$.
#
# The %force \f$ \vec{F}\f$ is
# \f{eqnarray*}
# \vec{F}  = & -\nabla V(r) & r < r_{\mathrm{cut}} \\
#           = & 0           & r \ge r_{\mathrm{cut}} \\
# \f}
# where \f$ \vec{r} \f$ is the vector pointing from one particle to the other in the %pair, and \f$ V(r) \f$ is
# chosen by a mode switch (see set_params())
# \f{eqnarray*}
# V(r)  = & V_{\mathrm{pair}}(r) & \mathrm{mode\ is\ no\_shift} \\
#       = & V_{\mathrm{pair}}(r) - V_{\mathrm{pair}}(r_{\mathrm{cut}}) & \mathrm{mode\ is\ shift} \\
#       = & S(r) \cdot V_{\mathrm{pair}}(r) & \mathrm{mode\ is\ xplor\ and\ } r_{\mathrm{on}} < r_{\mathrm{cut}} \\
#       = & V_{\mathrm{pair}}(r) - V_{\mathrm{pair}}(r_{\mathrm{cut}}) & \mathrm{mode\ is\ xplor\ and\ } r_{\mathrm{on}} \ge r_{\mathrm{cut}}
# \f}
# , \f$ S(r) \f$ is the XPLOR smoothing function
# \f{eqnarray*} 
# S(r) = & 1 & r < r_{\mathrm{on}} \\
#      = & \frac{(r_{\mathrm{cut}}^2 - r^2)^2 \cdot (r_{\mathrm{cut}}^2 + 2r^2 - 
#          3r_{\mathrm{on}}^2)}{(r_{\mathrm{cut}}^2 - r_{\mathrm{on}}^2)^3} 
#        & r_{\mathrm{on}} \le r \le r_{\mathrm{cut}} \\
#  = & 0 & r > r_{\mathrm{cut}} \\
# \f}
# and \f$ V_{\mathrm{pair}}(r) \f$ is the specific %pair potential chosen by the respective command.
#
# Enabling the XPLOR smoothing function \f$ S(r) \f$ results in both the potential energy and the %force going smoothly
# to 0 at \f$ r = r_{\mathrm{cut}} \f$, sometimes improving the rate of energy drift in long simulations.
# \f$ r_{\mathrm{on}} \f$ controls the point at which the smoothing starts, so it can be set to only slightly modify
# the tail of the potential. It is suggested that you plot your potentials with various values of 
# \f$ r_{\mathrm{on}} \f$ in order to find a good balance between a smooth potential function and minimal modification
# of the original \f$ V_{\mathrm{pair}}(r) \f$. A good value for the LJ potential is
# \f$ r_{\mathrm{on}} = 2 \cdot \sigma\f$
#
# The split smoothing / shifting of the potential when the mode is \c xplor is designed for use in mixed WCA / LJ
# systems. The WCA potential and it's first derivative already go smoothly to 0 at the cutoff, so there is no need
# to apply the smoothing function. In such mixed systems, set \f$ r_{\mathrm{on}} \f$ to a value greater than
# \f$ r_{\mathrm{cut}} \f$ for those pairs that interact via WCA in order to enable shifting of the WCA potential
# to 0 at the cuttoff.
#
# The following coefficients must be set per unique %pair of particle types. See hoomd_script.pair or 
# the \ref page_quick_start for information on how to set coefficients.
# - \f$ r_{\mathrm{cut}} \f$ - \c r_cut
#   - <i>optional</i>: defaults to the global r_cut specified in the %pair command
# - \f$ r_{\mathrm{on}} \f$ - \c r_on
#   - <i>optional</i>: defaults to the global r_cut specified in the %pair command
#
class pair(force._force):
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
        self.pair_coeff = coeff();
        self.pair_coeff.set_default_coeff('r_cut', self.global_r_cut);
        self.pair_coeff.set_default_coeff('r_on', self.global_r_cut);
        
    ## Set parameters controlling the way forces are computed
    #
    # \param mode (if set) Set the mode with which potentials are handled at the cutoff
    #
    # valid values for \a mode are: "none" (the default), "shift", and "xplor"
    #  - \b none - No shifting is performed and potentials are abruptly cut off
    #  - \b shift - A constant shift is applied to the entire potential so that it is 0 at the cutoff
    #  - \b xplor - A smoothing function is applied to gradually decrease both the force and potential to 0 at the 
    #               cutoff when ron < rcut, and shifts the potential to 0 ar the cutoff when ron >= rcut.
    # (see pair above for formulas and more information)
    #
    # \b Examples:
    # \code
    # mypair.set_params(mode="shift")
    # mypair.set_params(mode="no_shift")
    # mypair.set_params(mode="xplor")
    # \endcode
    # 
    def set_params(self, mode=None):
        util.print_status_line();
        
        if mode is not None:
            if mode == "no_shift":
                self.cpp_force.setShiftMode(self.cpp_class.energyShiftMode.no_shift)
            elif mode == "shift":
                self.cpp_force.setShiftMode(self.cpp_class.energyShiftMode.shift)
            elif mode == "xplor":
                self.cpp_force.setShiftMode(self.cpp_class.energyShiftMode.xplor)
            else:
                print >> sys.stderr, "\n***Error! Invalid mode\n";
                raise RuntimeError("Error changing parameters in pair force");
    
    def process_coeff(self, coeff):
        print >> sys.stderr, "\n***Error! Bug in hoomd_script, please report\n";
        raise RuntimeError("Error processing coefficients");
    
    def update_coeffs(self):
        coeff_list = self.required_coeffs + ["r_cut", "r_on"];
        # check that the pair coefficents are valid
        if not self.pair_coeff.verify(coeff_list):
            print >> sys.stderr, "\n***Error: Not all pair coefficients are set\n";
            raise RuntimeError("Error updating pair coefficients");
        
        # set all the params
        ntypes = globals.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in xrange(0,ntypes):
            type_list.append(globals.system_definition.getParticleData().getNameByType(i));
        
        for i in xrange(0,ntypes):
            for j in xrange(i,ntypes):
                # build a dict of the coeffs to pass to process_coeff
                coeff_dict = {};
                for name in coeff_list:
                    coeff_dict[name] = self.pair_coeff.get(type_list[i], type_list[j], name);
                
                # error check r_cut
                if coeff_dict['r_cut'] > self.global_r_cut:
                    print >> sys.stderr, \
                        "\n***Error: r_cut for a given particle type pair cannot be greater than the global value\n";
                    raise RuntimeError("Error updating pair coefficients");
                
                param = self.process_coeff(coeff_dict);
                self.cpp_force.setParams(i, j, param);
                self.cpp_force.setRcut(i, j, coeff_dict['r_cut']);
                self.cpp_force.setRon(i, j, coeff_dict['r_on']);

    ## \internal
    # \brief Get the maximum r_cut value set for any type pair
    # \pre update_coeffs must be called before get_max_rcut to verify that the coeffs are set
    def get_max_rcut(self):
        # go through the list of only the active particle types in the sim
        ntypes = globals.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in xrange(0,ntypes):
            type_list.append(globals.system_definition.getParticleData().getNameByType(i));
        
        # find the maximum r_cut
        max_rcut = 0.0;
        
        for i in xrange(0,ntypes):
            for j in xrange(i,ntypes):
                # get the r_cut value
                r_cut = self.pair_coeff.get(type_list[i], type_list[j], 'r_cut');
                max_rcut = max(max_rcut, r_cut);
        
        return max_rcut;

## Lennard-Jones %pair %force
#
# The command pair.lj specifies that a Lennard-Jones type %pair %force should be added to every
# non-bonded particle %pair in the simulation.
#
# \f{eqnarray*}
# V_{\mathrm{LJ}}(r)  = & 4 \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{12} - 
#                   \alpha \left( \frac{\sigma}{r} \right)^{6} \right] & r < r_{\mathrm{cut}} \\
#                     = & 0 & r \ge r_{\mathrm{cut}} \\
# \f}
#
# For an exact definition of the %force and potential calculation and how cutoff radii are handled, see pair.
#
# The following coefficients must be set per unique %pair of particle types. See hoomd_script.pair or 
# the \ref page_quick_start for information on how to set coefficients.
# - \f$ \varepsilon \f$ - \c epsilon
# - \f$ \sigma \f$ - \c sigma
# - \f$ \alpha \f$ - \c alpha
#   - <i>optional</i>: defaults to 1.0
# - \f$ r_{\mathrm{cut}} \f$ - \c r_cut
#   - <i>optional</i>: defaults to the global r_cut specified in the %pair command
# - \f$ r_{\mathrm{on}} \f$ - \c r_on
#   - <i>optional</i>: defaults to the global r_cut specified in the %pair command
#
# pair.lj is a standard %pair potential and supports a number of energy shift / smoothing modes. See pair for a full
# description of the various options.
#
# \b Example:
# \code
# lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
# lj.pair_coeff.set('A', 'B', epsilon=2.0, sigma=1.0, alpha=0.5, r_cut=3.0, r_on=2.0);
# lj.pair_coeff.set('B', 'B', epsilon=1.0, sigma=1.0, r_cut=2**(1.0/6.0), r_on=2.0);
# \endcode
#
# The cutoff radius \a r_cut passed into the initial pair.lj command sets the default \a r_cut for all %pair
# interactions. Smaller (or larger) cutoffs can be set individually per each type %pair. The cutoff distances used for
# the neighbor list will by dynamically determined from the maximum of all \a r_cut values specified among all type
# %pair parameters among all %pair potentials.
#
class lj(pair):
    ## Specify the Lennard-Jones %pair %force
    #
    # \param r_cut Default cutoff radius
    # \param name Name of the force instance 
    #
    # \b Example:
    # \code
    # lj = pair.lj(r_cut=3.0)
    # lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
    # lj.pair_coeff.set('A', 'B', epsilon=2.0, sigma=1.0, alpha=0.5, r_cut=3.0, r_on=2.0);
    # lj.pair_coeff.set('B', 'B', epsilon=1.0, sigma=1.0, r_cut=2**(1.0/6.0), r_on=2.0);
    # \endcode
    #
    # \note %Pair coefficients for all type pairs in the simulation must be
    # set before it can be started with run()
    def __init__(self, r_cut, name=None):
        util.print_status_line();
        
        # tell the base class how we operate
        
        # initialize the base class
        pair.__init__(self, r_cut, name);
        
        # update the neighbor list
        neighbor_list = _update_global_nlist(r_cut);
        neighbor_list.subscribe(lambda: self.log*self.get_max_rcut())
        
        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = hoomd.PotentialPairLJ(globals.system_definition, neighbor_list.cpp_nlist, self.name);
            self.cpp_class = hoomd.PotentialPairLJ;
        else:
            neighbor_list.cpp_nlist.setStorageMode(hoomd.NeighborList.storageMode.full);
            self.cpp_force = hoomd.PotentialPairLJGPU(globals.system_definition, neighbor_list.cpp_nlist, self.name);
            self.cpp_class = hoomd.PotentialPairLJGPU;
            self.cpp_force.setBlockSize(tune._get_optimal_block_size('pair.lj'));
            
        globals.system.addCompute(self.cpp_force, self.force_name);
        
        # setup the coefficent options
        self.required_coeffs = ['epsilon', 'sigma', 'alpha'];
        self.pair_coeff.set_default_coeff('alpha', 1.0);
        
    def process_coeff(self, coeff):
        epsilon = coeff['epsilon'];
        sigma = coeff['sigma'];
        alpha = coeff['alpha'];
        
        lj1 = 4.0 * epsilon * math.pow(sigma, 12.0);
        lj2 = alpha * 4.0 * epsilon * math.pow(sigma, 6.0);
        return hoomd.make_scalar2(lj1, lj2);
        
## Gaussian %pair %force
#
# The command pair.gauss specifies that a Gaussian %pair %force should be added to every
# non-bonded particle %pair in the simulation.
#
# \f{eqnarray*}
#  V_{\mathrm{gauss}}(r)  = & \varepsilon \exp \left[ -\frac{1}{2}\left( \frac{r}{\sigma} \right)^2 \right]
#                                         & r < r_{\mathrm{cut}} \\
#                     = & 0 & r \ge r_{\mathrm{cut}} \\
# \f}
#
# For an exact definition of the %force and potential calculation and how cutoff radii are handled, see pair.
#
# The following coefficients must be set per unique %pair of particle types. See hoomd_script.pair or 
# the \ref page_quick_start for information on how to set coefficients.
# - \f$ \varepsilon \f$ - \c epsilon
# - \f$ \sigma \f$ - \c sigma
# - \f$ r_{\mathrm{cut}} \f$ - \c r_cut
#   - <i>optional</i>: defaults to the global r_cut specified in the %pair command
# - \f$ r_{\mathrm{on}} \f$ - \c r_on
#   - <i>optional</i>: defaults to the global r_cut specified in the %pair command
#
# pair.gauss is a standard %pair potential and supports a number of energy shift / smoothing modes. See pair for a full
# description of the various options.
#
# \b Example:
# \code
# gauss.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
# gauss.pair_coeff.set('A', 'B', epsilon=2.0, sigma=1.0, r_cut=3.0, r_on=2.0);
# \endcode
#
# The cutoff radius \a r_cut passed into the initial pair.gauss command sets the default \a r_cut for all %pair
# interactions. Smaller (or larger) cutoffs can be set individually per each type %pair. The cutoff distances used for
# the neighbor list will by dynamically determined from the maximum of all \a r_cut values specified among all type
# %pair parameters among all %pair potentials.
#
class gauss(pair):
    ## Specify the Gaussian %pair %force
    #
    # \param r_cut Default cutoff radius
    # \param name Name of the force instance 
    #
    # \b Example:
    # \code
    # gauss = pair.lj(r_cut=3.0)
    # gauss.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
    # gauss.pair_coeff.set('A', 'B', epsilon=2.0, sigma=1.0, r_cut=3.0, r_on=2.0);
    # \endcode
    #
    # \note %Pair coefficients for all type pairs in the simulation must be
    # set before it can be started with run()
    def __init__(self, r_cut, name=None):
        util.print_status_line();
        
        # tell the base class how we operate
        
        # initialize the base class
        pair.__init__(self, r_cut, name);
        
        # update the neighbor list
        neighbor_list = _update_global_nlist(r_cut);
        neighbor_list.subscribe(lambda: self.log*self.get_max_rcut())
        
        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = hoomd.PotentialPairGauss(globals.system_definition, neighbor_list.cpp_nlist, self.name);
            self.cpp_class = hoomd.PotentialPairGauss;
        else:
            neighbor_list.cpp_nlist.setStorageMode(hoomd.NeighborList.storageMode.full);
            self.cpp_force = hoomd.PotentialPairGaussGPU(globals.system_definition, neighbor_list.cpp_nlist, self.name);
            self.cpp_class = hoomd.PotentialPairGaussGPU;
            self.cpp_force.setBlockSize(tune._get_optimal_block_size('pair.gauss'));
            
        globals.system.addCompute(self.cpp_force, self.force_name);
        
        # setup the coefficent options
        self.required_coeffs = ['epsilon', 'sigma'];
        
    def process_coeff(self, coeff):
        epsilon = coeff['epsilon'];
        sigma = coeff['sigma'];

        return hoomd.make_scalar2(epsilon, sigma);

## Shifted Lennard-Jones %pair %force
#
# The command pair.slj specifies that a shifted Lennard-Jones type %pair %force should be added to every
# non-bonded particle %pair in the simulation.
#
#    \f{eqnarray*}
#    V_{\mathrm{SLJ}}(r)  = & 4 \varepsilon \left[ \left( \frac{\sigma}{r - \Delta} \right)^{12} - 
#                           \left( \frac{\sigma}{r - \Delta} \right)^{6} \right] & r < (r_{\mathrm{cut}} + \Delta) \\
#                         = & 0 & r \ge (r_{\mathrm{cut}} + \Delta) \\
#    \f}
#    where \f$ \Delta = (d_i + d_j)/2 - 1 \f$ and \f$ d_i \f$ is the diameter of particle \f$ i \f$.
#
# For an exact definition of the %force and potential calculation and how cutoff radii are handled, see pair.
#
# The following coefficients must be set per unique %pair of particle types. See hoomd_script.pair or 
# the \ref page_quick_start for information on how to set coefficients.
# - \f$ \varepsilon \f$ - \c epsilon
# - \f$ \sigma \f$ - \c sigma
#   - <i>optional</i>: defaults to 1.0
# - \f$ r_{\mathrm{cut}} \f$ - \c r_cut
#   - <i>optional</i>: defaults to the global r_cut specified in the %pair command
#
# pair.slj is a standard %pair potential and supports a number of energy shift / smoothing modes. See pair for a full
# description of the various options.
#\note Due to the way that pair.slj modifies the cutoff criteria, a shift_mode of xplor is not supported.
#
# \b Example:
# \code
# slj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
# slj.pair_coeff.set('A', 'B', epsilon=2.0, sigma=1.0, r_cut=3.0);
# slj.pair_coeff.set('B', 'B', epsilon=1.0, sigma=1.0, r_cut=2**(1.0/6.0));
# \endcode
#
# The cutoff radius \a r_cut passed into the initial pair.slj command sets the default \a r_cut for all %pair
# interactions. Smaller (or larger) cutoffs can be set individually per each type %pair. The cutoff distances used for
# the neighbor list will by dynamically determined from the maximum of all \a r_cut values specified among all type
# %pair parameters among all %pair potentials. pair.slj adds an extra term to the maximum r_cut, determined by the 
# maximum diameter set by the user in \a d_max.
#
class slj(pair):
    ## Specify the Shifted Lennard-Jones %pair %force
    #
    # \param r_cut Default cutoff radius
    # \param name Name of the force instance
    # \param d_max Maximum diameter particles in the simulation will have
    #
    # The specified value of \a d_max will be used to properly determine the neighbor lists during the following
    # run() commands. If not specified, slj will set d_max to the largest diameter in particle data at the time it is initialized
    # If particle diameters change after initialization, it is \b imperative that \a d_max be the largest
    # diameter that any particle will attain at any time during the following run() command. If \a d_max is smaller
    # than it should be, some particles will effectively have a smaller value of \a r_cut then was set and the
    # simulation will be incorrect. \a d_max can be changed between runs by calling set_params().
    #
    # \b Example:
    # \code
    # slj = pair.slj(r_cut=3.0, d_max = 2.0)
    # slj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
    # slj.pair_coeff.set('A', 'B', epsilon=2.0, sigma=1.0, r_cut=3.0);
    # slj.pair_coeff.set('B', 'B', epsilon=1.0, sigma=1.0, r_cut=2**(1.0/6.0));
    # \endcode
    #
    # \note %Pair coefficients for all type pairs in the simulation must be
    # set before it can be started with run()
    def __init__(self, r_cut, d_max=None, name=None):
        util.print_status_line();
        
        # tell the base class how we operate
        
        # initialize the base class
        pair.__init__(self, r_cut, name);
        
        # update the neighbor list
        if d_max is None :
            sysdef = globals.system_definition;
            self.d_max = max([x.diameter for x in data.particle_data(sysdef.getParticleData())])
            print "slj internally setting d_max to", self.d_max
        else:    
            self.d_max = d_max;
                        
        neighbor_list = _update_global_nlist(r_cut);
        neighbor_list.subscribe(lambda: self.log*(self.get_max_rcut() + self.d_max - 1.0))
        
        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = hoomd.PotentialPairSLJ(globals.system_definition, neighbor_list.cpp_nlist, self.name);
            self.cpp_class = hoomd.PotentialPairSLJ;
        else:
            neighbor_list.cpp_nlist.setStorageMode(hoomd.NeighborList.storageMode.full);
            self.cpp_force = hoomd.PotentialPairSLJGPU(globals.system_definition, neighbor_list.cpp_nlist, self.name);
            self.cpp_class = hoomd.PotentialPairSLJGPU;
            self.cpp_force.setBlockSize(tune._get_optimal_block_size('pair.slj'));
            
        globals.system.addCompute(self.cpp_force, self.force_name);
        
        # setup the coefficent options
        self.required_coeffs = ['epsilon', 'sigma'];
        
    def process_coeff(self, coeff):
        epsilon = coeff['epsilon'];
        sigma = coeff['sigma'];
        
        lj1 = 4.0 * epsilon * math.pow(sigma, 12.0);
        lj2 = 4.0 * epsilon * math.pow(sigma, 6.0);
        return hoomd.make_scalar2(lj1, lj2);

    ## Set parameters controlling the way forces are computed
    #
    # \param mode (if set) Set the mode with which potentials are handled at the cutoff
    # \param d_max (if set) Set the new maximum particle diameter in the system
    #
    # valid values for \a mode are: "none" (the default), "shift", and "xplor"
    #  - \b none - No shifting is performed and potentials are abruptly cut off
    #  - \b shift - A constant shift is applied to the entire potential so that it is 0 at the cutoff
    #
    # (see pair above for formulas and more information)
    #
    # \b Examples:
    # \code
    # slj.set_params(mode="shift")
    # slj.set_params(mode="no_shift")
    # slj.set_params(d_max = 3.0)
    # \endcode
    # 
    def set_params(self, mode=None, d_max=None):
        util.print_status_line();
        
        if mode == "xplor":
            print >> sys.stderr, "\n***Error! XPLOR is smoothing is not supported with slj\n";
            raise RuntimeError("Error changing parameters in pair force");
        
        pair.set_params(self, mode=mode);
        
        if d_max is not None:
            self.d_max = d_max;

## Yukawa %pair %force
#
# The command pair.yukawa specifies that a Yukawa %pair %force should be added to every
# non-bonded particle %pair in the simulation.
#
# \f{eqnarray*}
#  V_{\mathrm{yukawa}}(r)  = & \varepsilon \frac{ \exp \left( -\kappa r \right) }{r} & r < r_{\mathrm{cut}} \\
#                     = & 0 & r \ge r_{\mathrm{cut}} \\
# \f}
#
# For an exact definition of the %force and potential calculation and how cutoff radii are handled, see pair.
#
# The following coefficients must be set per unique %pair of particle types. See hoomd_script.pair or 
# the \ref page_quick_start for information on how to set coefficients.
# - \f$ \varepsilon \f$ - \c epsilon
# - \f$ \kappa \f$ - \c kappa
# - \f$ r_{\mathrm{cut}} \f$ - \c r_cut
#   - <i>optional</i>: defaults to the global r_cut specified in the %pair command
# - \f$ r_{\mathrm{on}} \f$ - \c r_on
#   - <i>optional</i>: defaults to the global r_cut specified in the %pair command
#
# pair.yukawa is a standard %pair potential and supports a number of energy shift / smoothing modes. See pair for a full
# description of the various options.
#
# \b Example:
# \code
# yukawa.pair_coeff.set('A', 'A', epsilon=1.0, kappa=1.0)
# yukawa.pair_coeff.set('A', 'B', epsilon=2.0, kappa=0.5, r_cut=3.0, r_on=2.0);
# \endcode
#
# The cutoff radius \a r_cut passed into the initial pair.yukawa command sets the default \a r_cut for all %pair
# interactions. Smaller (or larger) cutoffs can be set individually per each type %pair. The cutoff distances used for
# the neighbor list will by dynamically determined from the maximum of all \a r_cut values specified among all type
# %pair parameters among all %pair potentials.
#
class yukawa(pair):
    ## Specify the Yukawa %pair %force
    #
    # \param r_cut Default cutoff radius
    # \param name Name of the force instance
    #
    # \b Example:
    # \code
    # yukawa = pair.lj(r_cut=3.0)
    # yukawa.pair_coeff.set('A', 'A', epsilon=1.0, kappa=1.0)
    # yukawa.pair_coeff.set('A', 'B', epsilon=2.0, kappa=0.5, r_cut=3.0, r_on=2.0);
    # \endcode
    #
    # \note %Pair coefficients for all type pairs in the simulation must be
    # set before it can be started with run()
    def __init__(self, r_cut, name=None):
        util.print_status_line();
        
        # tell the base class how we operate
        
        # initialize the base class
        pair.__init__(self, r_cut, name);
        
        # update the neighbor list
        neighbor_list = _update_global_nlist(r_cut);
        neighbor_list.subscribe(lambda: self.log*self.get_max_rcut())
        
        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = hoomd.PotentialPairYukawa(globals.system_definition, neighbor_list.cpp_nlist, self.name);
            self.cpp_class = hoomd.PotentialPairYukawa;
        else:
            neighbor_list.cpp_nlist.setStorageMode(hoomd.NeighborList.storageMode.full);
            self.cpp_force = hoomd.PotentialPairYukawaGPU(globals.system_definition, neighbor_list.cpp_nlist, self.name);
            self.cpp_class = hoomd.PotentialPairYukawaGPU;
            self.cpp_force.setBlockSize(tune._get_optimal_block_size('pair.yukawa'));
            
        globals.system.addCompute(self.cpp_force, self.force_name);
        
        # setup the coefficent options
        self.required_coeffs = ['epsilon', 'kappa'];
        
    def process_coeff(self, coeff):
        epsilon = coeff['epsilon'];
        kappa = coeff['kappa'];

        return hoomd.make_scalar2(epsilon, kappa);

## CMM coarse-grain model %pair %force
#
# The command pair.cgcmm specifies that a special version of Lennard-Jones type %pair %force
# should be added to every non-bonded particle %pair in the simulation. This potential
# version is used in the CMM coarse grain model and uses a combination of Lennard-Jones
# potentials with different exponent pairs between different atom pairs.
#
# The %force \f$ \vec{F}\f$ is
# \f{eqnarray*}
# \vec{F}  = & -\nabla V_{\mathrm{LJ}}(r) & r < r_{\mathrm{cut}} \\
#          = & 0                          & r \ge r_{\mathrm{cut}} \\
# \f}
# with being either
# \f[ V_{\mathrm{LJ}}(r) = 4 \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{12} - 
#                                               \alpha \left( \frac{\sigma}{r} \right)^{6} \right] \f],
# or
# \f[ V_{\mathrm{LJ}}(r) = \frac{27}{4} \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{9} - 
#                                                          \alpha \left( \frac{\sigma}{r} \right)^{6} \right] \f],
# or
# \f[ V_{\mathrm{LJ}}(r) = \frac{3\sqrt{3}}{2} \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{12} - 
#                                                            \alpha \left( \frac{\sigma}{r} \right)^{4} \right] \f],
# and \f$ \vec{r} \f$ being the vector pointing from one particle to the other in the %pair.
#
# The following coefficients must be set per unique %pair of particle types. See hoomd_script.pair or 
# the \ref page_quick_start for information on how to set coefficients.
# - \f$ \varepsilon \f$ - \c epsilon
# - \f$ \sigma \f$ - \c sigma
# - \f$ \alpha \f$ - \c alpha
# - exponents, the choice of LJ-exponents, currently supported are 12-6, 9-6, and 12-4.
# 
# We support three keyword variants 124 (native), lj12_4 (LAMMPS), LJ12-4 (MPDyn)
#
# \b Example:
# \code
# cg.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, alpha=1.0, exponents='LJ12-6')
# cg.pair_coeff.set('W', 'W', epsilon=3.7605, sigma=1.285588, alpha=1.0, exponents='lj12_4')
# cg.pair_coeff.set('OA', 'OA', epsilon=1.88697479, sigma=1.09205882, alpha=1.0, exponents='96')
# \endcode
#
# The cuttoff radius \f$ r_{\mathrm{cut}} \f$ is set once when pair.cg is specified (see __init__())
#
class cgcmm(force._force):
    ## Specify the CG-CMM Lennard-Jones %pair %force
    #
    # \param r_cut Cuttoff radius (see documentation above)
    #
    # \b Example:
    # \code
    # cg1 = pair.cgcmm(r_cut=3.0)
    # cg1.pair_coeff.set('A', 'A', epsilon=0.5, sigma=1.0, alpha=1.0, exponents='lj12_4')
    # \endcode
    #
    # \note %Pair coefficients for all type pairs in the simulation must be
    # set before it can be started with run()
    def __init__(self, r_cut):
        util.print_status_line();
        
        # initialize the base class
        force._force.__init__(self);
        
        # update the neighbor list
        neighbor_list = _update_global_nlist(r_cut);
        neighbor_list.subscribe(lambda: self.log*r_cut)
        
        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = hoomd.CGCMMForceCompute(globals.system_definition, neighbor_list.cpp_nlist, r_cut);
        else:
            neighbor_list.cpp_nlist.setStorageMode(hoomd.NeighborList.storageMode.full);
            self.cpp_force = hoomd.CGCMMForceComputeGPU(globals.system_definition, neighbor_list.cpp_nlist, r_cut);
            self.cpp_force.setBlockSize(tune._get_optimal_block_size('pair.cgcmm'));
            
        globals.system.addCompute(self.cpp_force, self.force_name);
        
        # setup the coefficent matrix
        self.pair_coeff = coeff();
        
    def update_coeffs(self):
        # check that the pair coefficents are valid
        if not self.pair_coeff.verify(["epsilon", "sigma", "alpha", "exponents"]):
            print >> sys.stderr, "\n***Error: Not all pair coefficients are set in pair.cgcmm\n";
            raise RuntimeError("Error updating pair coefficients");
        
        # set all the params
        ntypes = globals.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in xrange(0,ntypes):
            type_list.append(globals.system_definition.getParticleData().getNameByType(i));
        
        for i in xrange(0,ntypes):
            for j in xrange(i,ntypes):
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

## Tabulated %pair %force
#
# The command pair.table specifies that a tabulated  %pair %force should be added to every non-bonded particle %pair 
# in the simulation.
#
# The %force \f$ \vec{F}\f$ is
# \f{eqnarray*}
#  \vec{F}(\vec{r})     = & 0                           & r < r_{\mathrm{min}} \\
#                       = & F_{\mathrm{user}}(r)\hat{r} & r < r_{\mathrm{max}} \\
#                       = & 0                           & r \ge r_{\mathrm{max}} \\
# \f}
# and the potential \f$ V(r) \f$ is
# \f{eqnarray*}
# V(r)       = & 0                    & r < r_{\mathrm{min}} \\
#            = & V_{\mathrm{user}}(r) & r < r_{\mathrm{max}} \\
#            = & 0                    & r \ge r_{\mathrm{max}} \\
# \f}
# ,where \f$ \vec{r} \f$ is the vector pointing from one particle to the other in the %pair.
#
# \f$  F_{\mathrm{user}}(r) \f$ and \f$ V_{\mathrm{user}}(r) \f$ are evaluated on \a width grid points between 
# \f$ r_{\mathrm{min}} \f$ and \f$ r_{\mathrm{max}} \f$. Values are interpolated linearly between grid points.
# For correctness, the user must specify a force defined by: \f$ F = -\frac{\partial V}{\partial r}\f$  
#
# The following coefficients must be set per unique %pair of particle types. See hoomd_script.pair or 
# the \ref page_quick_start for information on how to set coefficients.
# - \f$ F_{\mathrm{user}}(r) \f$ and \f$ V_{\mathrm{user}}(r) \f$ - evaluated by \c func (see example)
# - coefficients passed to \c func - \c coeff (see example)
# - \f$ r_{\mathrm{min}} \f$ - \c rmin
# - \f$ r_{\mathrm{max}} \f$ - \c rmax
# 
# \b Example:
# \code
# table.pair_coeff.set('A', 'A', func=my_potential, rmin=0, rmax=10, coeff=dict(A=1.5, s=3.0))
# \endcode
#
# The table \a width is set once when pair.table is specified (see __init__())
#
class table(force._force):
    ## Specify the Tabulated %pair %force
    #
    # \param width Number of points to use to interpolate V and F (see documentation above)
    # \param name Name of the force instance
    #
    # \b Example:
    # \code
    # def lj(r, rmin, rmax, epsilon, sigma):
    #     V = 4 * epsilon * ( (sigma / r)**12 - (sigma / r)**6);
    #     F = 4 * epsilon / r * ( 12 * (sigma / r)**12 - 6 * (sigma / r)**6);
    #     return (V, F)
    #
    # table = pair.table(width=1000)
    # table.pair_coeff.set('A', 'A', func=lj, rmin=0.8, rmax=3.0, coeff=dict(epsilon=1.5, sigma=1.0))
    # table.pair_coeff.set('A', 'B', func=lj, rmin=0.8, rmax=3.0, coeff=dict(epsilon=2.0, sigma=1.2))
    # table.pair_coeff.set('B', 'B', func=lj, rmin=0.8, rmax=3.0, coeff=dict(epsilon=0.5, sigma=1.0))
    # \endcode
    #
    # \note For potentials that diverge near r=0, make sure to set \c rmin to a reasonable value. If a potential does 
    # not diverge near r=0, then a setting of \c rmin=0 is valid.
    #
    # \note %Pair coefficients for all type pairs in the simulation must be
    # set before it can be started with run()
    def __init__(self, width, name=None):
        util.print_status_line();
        
        # initialize the base class
        force._force.__init__(self, name);

        # update the neighbor list with a dummy 0 r_cut. The r_cut will be properly updated before the first run()
        neighbor_list = _update_global_nlist(0);
        neighbor_list.subscribe(lambda: self.log*self.get_max_rcut())
        
        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = hoomd.TablePotential(globals.system_definition, neighbor_list.cpp_nlist, int(width), self.name);
        else:
            neighbor_list.cpp_nlist.setStorageMode(hoomd.NeighborList.storageMode.full);
            self.cpp_force = hoomd.TablePotentialGPU(globals.system_definition, neighbor_list.cpp_nlist, int(width), self.name);
            self.cpp_force.setBlockSize(tune._get_optimal_block_size('pair.table'));
            
        globals.system.addCompute(self.cpp_force, self.force_name);
        
        # setup the coefficent matrix
        self.pair_coeff = coeff();
        
        # stash the width for later use
        self.width = width;
        
    def update_pair_table(self, typei, typej, func, rmin, rmax, coeff):
        # allocate arrays to store V and F
        Vtable = hoomd.std_vector_float();
        Ftable = hoomd.std_vector_float();
        
        # calculate dr
        dr = (rmax - rmin) / float(self.width-1);
        
        # evaluate each point of the function
        for i in xrange(0, self.width):
            r = rmin + dr * i;
            (V,F) = func(r, rmin, rmax, **coeff);
                
            # fill out the tables
            Vtable.append(V);
            Ftable.append(F);
        
        # pass the tables on to the underlying cpp compute
        self.cpp_force.setTable(typei, typej, Vtable, Ftable, rmin, rmax);
    
    def get_max_rcut(self):
        # loop only over current particle types
        ntypes = globals.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in xrange(0,ntypes):
            type_list.append(globals.system_definition.getParticleData().getNameByType(i));
        
        # find the maximum rmax to update the neighbor list with
        maxrmax = 0.0;
        
        # loop through all of the unique type pairs and find the maximum rmax
        for i in xrange(0,ntypes):
            for j in xrange(i,ntypes):
                rmax = self.pair_coeff.get(type_list[i], type_list[j], "rmax");
                maxrmax = max(maxrmax, rmax);

        return maxrmax;
                            
    def update_coeffs(self):
        # check that the pair coefficents are valid
        if not self.pair_coeff.verify(["func", "rmin", "rmax", "coeff"]):
            print >> sys.stderr, "\n***Error: Not all pair coefficients are set for pair.table\n";
            raise RuntimeError("Error updating pair coefficients");
        
        # set all the params
        ntypes = globals.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in xrange(0,ntypes):
            type_list.append(globals.system_definition.getParticleData().getNameByType(i));
        
        # loop through all of the unique type pairs and evaluate the table
        for i in xrange(0,ntypes):
            for j in xrange(i,ntypes):
                func = self.pair_coeff.get(type_list[i], type_list[j], "func");
                rmin = self.pair_coeff.get(type_list[i], type_list[j], "rmin");
                rmax = self.pair_coeff.get(type_list[i], type_list[j], "rmax");
                coeff = self.pair_coeff.get(type_list[i], type_list[j], "coeff");

                self.update_pair_table(i, j, func, rmin, rmax, coeff);
                

## Morse %pair %force
#
# The command pair.morse specifies that a Morse %pair %force should be added to every
# non-bonded particle %pair in the simulation.
#
# \f{eqnarray*}
#  V_{\mathrm{morse}}(r)  = & D_0 \left[ \exp \left(-2\alpha\left(r-r_0\right)\right) -2\exp \left(-\alpha\left(r-r_0\right)\right) \right] & r < r_{\mathrm{cut}} \\
#                     = & 0 & r \ge r_{\mathrm{cut}} \\
# \f}
#
# For an exact definition of the %force and potential calculation and how cutoff radii are handled, see pair.
#
# The following coefficients must be set per unique %pair of particle types. See hoomd_script.pair or 
# the \ref page_quick_start for information on how to set coefficients.
# - \f$ D_0 \f$ - \c D0, depth of the potential at its minimum
# - \f$ \alpha \f$ - \c alpha, controls the width of the potential well
# - \f$ r_0 \f$ - \c r0, position of the minimum
# - \f$ r_{\mathrm{cut}} \f$ - \c r_cut
#   - <i>optional</i>: defaults to the global r_cut specified in the %pair command
# - \f$ r_{\mathrm{on}} \f$ - \c r_on
#   - <i>optional</i>: defaults to the global r_cut specified in the %pair command
#
# pair.morse is a standard %pair potential and supports a number of energy shift / smoothing modes. See pair for a full
# description of the various options.
#
# \b Example:
# \code
# morse.pair_coeff.set('A', 'A', D0=1.0, alpha=3.0, r0=1.0)
# morse.pair_coeff.set('A', 'B', D0=1.0, alpha=3.0, r0=1.0, r_cut=3.0, r_on=2.0);
# \endcode
#
# The cutoff radius \a r_cut passed into the initial pair.morse command sets the default \a r_cut for all %pair
# interactions. Smaller (or larger) cutoffs can be set individually per each type %pair. The cutoff distances used for
# the neighbor list will by dynamically determined from the maximum of all \a r_cut values specified among all type
# %pair parameters among all %pair potentials.
#
class morse(pair):
    ## Specify the Morse %pair %force
    #
    # \param r_cut Default cutoff radius
    # \param name Name of the force instance
    #
    # \b Example:
    # \code
    # morse = pair.morse(r_cut=3.0)
    # morse.pair_coeff.set('A', 'A', D0=1.0, alpha=3.0, r0=1.0)
    # morse.pair_coeff.set('A', 'B', D0=1.0, alpha=3.0, r0=1.0, r_cut=3.0, r_on=2.0);
    # \endcode
    #
    # \note %Pair coefficients for all type pairs in the simulation must be
    # set before it can be started with run()
    def __init__(self, r_cut, name=None):
        util.print_status_line();
        
        # tell the base class how we operate
        
        # initialize the base class
        pair.__init__(self, r_cut, name);
        
        # update the neighbor list
        neighbor_list = _update_global_nlist(r_cut);
        neighbor_list.subscribe(lambda: self.log*self.get_max_rcut())
        
        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = hoomd.PotentialPairMorse(globals.system_definition, neighbor_list.cpp_nlist, self.name);
            self.cpp_class = hoomd.PotentialPairMorse;
        else:
            neighbor_list.cpp_nlist.setStorageMode(hoomd.NeighborList.storageMode.full);
            self.cpp_force = hoomd.PotentialPairMorseGPU(globals.system_definition, neighbor_list.cpp_nlist, self.name);
            self.cpp_class = hoomd.PotentialPairMorseGPU;
            self.cpp_force.setBlockSize(tune._get_optimal_block_size('pair.morse'));
            
        globals.system.addCompute(self.cpp_force, self.force_name);
        
        # setup the coefficent options
        self.required_coeffs = ['D0', 'alpha', 'r0'];
        
    def process_coeff(self, coeff):
        D0 = coeff['D0'];
        alpha = coeff['alpha'];
        r0 = coeff['r0']

        return hoomd.make_scalar4(D0, alpha, r0, 0.0);

