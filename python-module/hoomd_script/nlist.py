# -- start license --
# Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
# (HOOMD-blue) Open Source Software License Copyright 2009-2016 The Regents of
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

# Maintainer: joaander

from hoomd_script import globals;
from hoomd_script import init
from hoomd_script import util;
import hoomd;
import hoomd_script

## \package hoomd_script.nlist
# \brief Commands that create neighbor lists
#
# Neighbor lists accelerate %pair force calculation by maintaining a list of particles within a cutoff radius.
# Multiple %pair forces can utilize the same neighbor list. Neighbors are included using a pairwise cutoff
# \f$ r_\mathrm{cut}(i,j) \f$ that is the maximum of all \f$ r_\mathrm{cut}(i,j) \f$ set for the %pair forces attached
# to the list.
#
# Multiple neighbor lists can be created to accelerate simulations where there is significant disparity in
# \f$ r_\mathrm{cut}(i,j) \f$ between pair potentials. If one %pair force has a cutoff radius much smaller than
# another %pair force, the %pair force calculation for the short cutoff will be slowed down considerably because many
# particles in the neighbor list will have to be read and skipped because they lie outside the shorter cutoff.
#
# The simplest way to build a neighbor list is \f$ O(N^2) \f$: each particle loops over all other particles and only
# includes those within the neighbor list cutoff. This algorithm is no longer implemented in HOOMD-blue because it is
# slow and inefficient. Instead, three accelerated algorithms based on %cell lists and bounding volume hierarchy trees
# are implemented. The %cell list implementation is fastest when the cutoff radius is similar between all %pair forces
# (smaller than 2:1 ratio). The %stencil implementation is a different variant of the cell list, and is usually fastest
# when there is large disparity in the %pair cutoff radius and a high number fraction of particles with the
# bigger cutoff (at least 30%%). The %tree implementation is faster when there is large size disparity and
# the number fraction of big objects is low. Because the performance of these algorithms depends sensitively on your
# system and hardware, you should carefully test which option is fastest for your simulation.
#
# Particles can be excluded from the neighbor list based on certain criteria. Setting \f$ r_\mathrm{cut}(i,j) \le 0\f$ 
# will exclude this cross interaction from the neighbor list on build time. Particles can also be excluded by topology
# or for belonging to the same rigid body (see reset_exclusions()).
#
# In previous versions of HOOMD-blue, %pair forces automatically created and subscribed to a single global neighbor
# list that was automatically created. Backwards compatibility is maintained to this behavior if a neighbor list is
# not specified by a %pair force. This package also maintains a thin wrapper around globals.neighbor_list for
# for interfacing with this object. It takes the place of the old model for making the global neighbor list available
# as "nlist" in the __main__ namespace. Moving it into the hoomd_script namespace is backwards compatible as long as
# the user does "from hoomd_script import *" - but it also makes it much easier to reference the nlist from modules
# other than __main__. Backwards compatibility is only ensured if the script only uses the public python facing API.
# Bypassing this to get at the C++ interface should be done through globals.neighbor_list . These wrappers are
# (re-)documented below, but should \b only be used to interface with globals.neighbor_list. Otherwise, the methods
# should be called directly on the neighbor list objects themselves. These global wrappers may be deprecated in a
# future release.
#
# \b Examples:
# \code
# nl_c = nlist.cell(check_period=1)
# nl_t = nlist.tree(r_buff = 0.8)
# lj1 = pair.lj(r_cut = 3.0, nlist=nl_c)
# lj2 = pair.lj(r_cut = 10.0, nlist=nl_t)
# lj3 = pair.lj(r_cut = 1.1) # subscribe to the default global nlist
# \endcode

## \internal
# \brief Generic neighbor list object
#
# Any bonds defined in the simulation are automatically used to exclude bonded particle
# pairs from appearing in the neighbor list. Use the command reset_exclusions() to change this behavior.
#
# Neighborlists are properly and efficiently calculated in 2D simulations if the z dimension of the box is small,
# but non-zero and the dimensionally of the system is set \b before the first pair force is specified.
#
class _nlist:
    ## \internal
    # \brief Constructs a neighbor list
    # \details
    # \param self Python required instance variable
    def __init__(self):
        # check if initialization has occured
        if not init.is_initialized():
            globals.msg.error("Cannot create neighbor list before initialization\n");
            raise RuntimeError('Error creating neighbor list');
        
        # default exclusions
        self.is_exclusion_overridden = False;
        self.exclusions = None

        # save the parameters we set
        self.r_cut = rcut();
        self.r_buff = 0.0;

        # save a list of subscribers that may have a say in determining the maximum r_cut
        self.subscriber_callbacks = [];

    ## \internal
    # \brief Adds a subscriber to the neighbor list
    # \param callable is a 0 argument callable object that returns the rcut object for all cutoff pairs in potential
    # All \a callables will be called at the beginning of each run() to determine the maximum r_cut needed for that run.
    #
    def subscribe(self, callable):
        self.subscriber_callbacks.append(callable);

    ## \internal
    # \brief Updates r_cut based on the subscriber's requests
    # \details This method is triggered every time the run command is called
    #
    def update_rcut(self):
        r_cut_max = rcut();
        for c in self.subscriber_callbacks:
            rcut_obj = c();
            if rcut_obj is not None:
                r_cut_max.merge(rcut_obj);
        
        # ensure that all type pairs are filled
        r_cut_max.fill()
        self.r_cut = r_cut_max;

        # get a list of types from the particle data
        ntypes = globals.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(globals.system_definition.getParticleData().getNameByType(i));

        # loop over all possible pairs and require that a dictionary key exists for them
        for i in range(0,ntypes):
            for j in range(i,ntypes):
                a = type_list[i];
                b = type_list[j];
                self.cpp_nlist.setRCutPair(i,j, self.r_cut.values[(a,b)]);

    ## \internal
    # \brief Sets the default bond exclusions, but only if the defaults have not been overridden
    def update_exclusions_defaults(self):
        if self.cpp_nlist.wantExclusions() and self.exclusions is not None:
            util._disable_status_lines = True;
            # update exclusions using stored values
            self.reset_exclusions(exclusions=self.exclusions)
            util._disable_status_lines = False;
        elif not self.is_exclusion_overridden:
            util._disable_status_lines = True;
            self.reset_exclusions(exclusions=['body', 'bond']);
            util._disable_status_lines = False;


    ## Change neighbor list parameters
    #
    # \param r_buff (if set) changes the buffer radius around the cutoff (in distance units)
    # \param check_period (if set) changes the period (in time steps) between checks to see if the neighbor list
    #        needs updating
    # \param d_max (if set) notifies the neighbor list of the maximum diameter that a particle attain over the following
    #        run() commands. (in distance units)
    # \param dist_check When set to False, disable the distance checking logic and always regenerate the nlist every
    #        \a check_period steps
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
    # When using pair.slj, \a d_max \b MUST be set to the maximum diameter that a particle will attain at any point
    # during the following run() commands (see pair.slj for more information). When using in conjunction with pair.slj,
    # pair.slj will
    # automatically set \a d_max for the nlist.  This can be overridden (e.g. if multiple potentials using diameters are used)
    # by using nlist.set_params() after the
    # pair.slj class has been initialized.   When <i>not</i> using pair.slj (or other diameter-using potential), \a d_max
    # \b MUST be left at the default value of 1.0 or the simulation will be incorrect if d_max is less than 1.0 and slower
    # than necessary if d_max is greater than 1.0.
    #
    # \b Examples:
    # \code
    # nl.set_params(r_buff = 0.9)
    # nl.set_params(check_period = 11)
    # nl.set_params(r_buff = 0.7, check_period = 4)
    # nl.set_params(d_max = 3.0)
    # \endcode
    def set_params(self, r_buff=None, check_period=None, d_max=None, dist_check=True):
        util.print_status_line();

        if self.cpp_nlist is None:
            globals.msg.error('Bug in hoomd_script: cpp_nlist not set, please report\n');
            raise RuntimeError('Error setting neighbor list parameters');

        # update the parameters
        if r_buff is not None:
            self.cpp_nlist.setRBuff(r_buff);
            self.r_buff = r_buff;

        if check_period is not None:
            self.cpp_nlist.setEvery(check_period, dist_check);

        if d_max is not None:
            self.cpp_nlist.setMaximumDiameter(d_max);

    ## Resets all exclusions in the neighborlist
    #
    # \param exclusions Select which interactions should be excluded from the %pair interaction calculation.
    #
    # By default, the following are excluded from short range %pair interactions.
    # - Directly bonded particles
    # - Particles that are in the same rigid body
    #
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
    # - \b %body - Exclude particles that belong to the same body
    #
    # The following types are determined solely by the bond topology. Every chain of particles in the simulation
    # connected by bonds (1-2-3-4) will be subject to the following exclusions, if enabled, whether or not explicit
    # angles or dihedrals are defined.
    # - \b 1-2  - Same as bond
    # - \b 1-3  - Exclude particles connected with a sequence of two bonds.
    # - \b 1-4  - Exclude particles connected with a sequence of three bonds.
    #
    # \b Examples:
    # \code
    # nl.reset_exclusions(exclusions = ['1-2'])
    # nl.reset_exclusions(exclusions = ['1-2', '1-3', '1-4'])
    # nl.reset_exclusions(exclusions = ['bond', 'angle'])
    # nl.reset_exclusions(exclusions = [])
    # \endcode
    #
    def reset_exclusions(self, exclusions = None):
        util.print_status_line();
        self.is_exclusion_overridden = True;

        if self.cpp_nlist is None:
            globals.msg.error('Bug in hoomd_script: cpp_nlist not set, please report\n');
            raise RuntimeError('Error resetting exclusions');

        # clear all of the existing exclusions
        self.cpp_nlist.clearExclusions();
        self.cpp_nlist.setFilterBody(False);

        if exclusions is None:
            # confirm that no exclusions are left.
            self.cpp_nlist.countExclusions();
            return

        # store exclusions for later use
        self.exclusions = list(exclusions)

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

        if 'body' in exclusions:
            self.cpp_nlist.setFilterBody(True);
            exclusions.remove('body');

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
            globals.msg.error('Exclusion type(s): ' + str(exclusions) +  ' are not supported\n');
            raise RuntimeError('Error resetting exclusions');

        # collect and print statistics about the number of exclusions.
        self.cpp_nlist.countExclusions();

    ## Benchmarks the neighbor list computation
    # \param n Number of iterations to average the benchmark over
    #
    # \b Examples:
    # \code
    # t = nl.benchmark(n = 100)
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
            globals.msg.error('Bug in hoomd_script: cpp_nlist not set, please report\n');
            raise RuntimeError('Error benchmarking neighbor list');

        # run the benchmark
        return self.cpp_nlist.benchmark(int(n))

    ## Query the maximum possible check_period
    # query_update_period examines the counts of nlist rebuilds during the previous run() command.
    # It returns \c s-1, where s is the smallest update period experienced during that time.
    # Use it after a medium-length warm up run with check_period=1 to determine what check_period to set
    # for production runs.
    #
    # \note If the previous run() was short, insufficient sampling may cause the queried update period
    # to be large enough to result in dangerous builds during longer runs. Unless you use a really long
    # warm up run, subtract an additional 1 from this when you set check_period for additional safety.
    #
    def query_update_period(self):
        if self.cpp_nlist is None:
            globals.msg.error('Bug in hoomd_script: cpp_nlist not set, please report\n');
            raise RuntimeError('Error setting neighbor list parameters');

        return self.cpp_nlist.getSmallestRebuild()-1;
        
    ## Make a series of short runs to determine the fastest performing r_buff setting
    # \param warmup Number of time steps to run() to warm up the benchmark
    # \param r_min Smallest value of r_buff to test
    # \param r_max Largest value of r_buff to test
    # \param jumps Number of different r_buff values to test
    # \param steps Number of time steps to run() at each point
    # \param set_max_check_period Set to True to enable automatic setting of the maximum nlist check_period
    #
    # tune() executes \a warmup time steps. Then it sets the nlist \a r_buff value to \a r_min and runs for
    # \a steps time steps. The TPS value is recorded, and the benchmark moves on to the next \a r_buff value
    # completing at \a r_max in \a jumps jumps. Status information is printed out to the screen, and the optimal
    # \a r_buff value is left set for further runs() to continue at optimal settings.
    #
    # Each benchmark is repeated 3 times and the median value chosen. Then, \a warmup time steps are run() again
    # at the optimal r_buff in order to determine the maximum value of check_period. In total,
    # (2*warmup + 3*jump*steps) time steps are run().
    #
    # \note By default, the maximum check_period is \b not set for safety. If you wish to have it set
    # when the call completes, call with the parameter set_max_check_period=True.
    #
    # \returns (optimal_r_buff, maximum check_period)
    #
    # \MPI_SUPPORTED
    def tune(self, warmup=200000, r_min=0.05, r_max=1.0, jumps=20, steps=5000, set_max_check_period=False):
        # check if initialization has occurred
        if not init.is_initialized():
            globals.msg.error("Cannot tune r_buff before initialization\n");
        
        if self.cpp_nlist is None:
            globals.msg.error('Bug in hoomd_script: cpp_nlist not set, please report\n')
            raise RuntimeError('Error tuning neighbor list')

        # start off at a check_period of 1
        self.set_params(check_period=1)

        # make the warmup run
        hoomd_script.run(warmup);

        # initialize scan variables
        dr = (r_max - r_min) / (jumps - 1);
        r_buff_list = [];
        tps_list = [];

        # loop over all desired r_buff points
        for i in range(0,jumps):
            # set the current r_buff
            r_buff = r_min + i * dr;
            self.set_params(r_buff=r_buff);

            # run the benchmark 3 times
            tps = [];
            hoomd_script.run(steps);
            tps.append(globals.system.getLastTPS())
            hoomd_script.run(steps);
            tps.append(globals.system.getLastTPS())
            hoomd_script.run(steps);
            tps.append(globals.system.getLastTPS())

            # record the median tps of the 3
            tps.sort();
            tps_list.append(tps[1]);
            r_buff_list.append(r_buff);

        # find the fastest r_buff
        fastest = tps_list.index(max(tps_list));
        fastest_r_buff = r_buff_list[fastest];

        # set the fastest and rerun the warmup steps to identify the max check period
        self.set_params(r_buff=fastest_r_buff);
        hoomd_script.run(warmup);

        # notify the user of the benchmark results
        globals.msg.notice(2, "r_buff = " + str(r_buff_list) + '\n');
        globals.msg.notice(2, "tps = " + str(tps_list) + '\n');
        globals.msg.notice(2, "Optimal r_buff: " + str(fastest_r_buff) + '\n');
        globals.msg.notice(2, "Maximum check_period: " + str(self.query_update_period()) + '\n');

        # set the found max check period
        if set_max_check_period:
            self.set_params(check_period=self.query_update_period());

        # return the results to the script
        return (fastest_r_buff, self.query_update_period());

## \internal
# \brief %nlist r_cut matrix
# \details
# Holds the maximum cutoff radius by pair type, and gracefully updates maximum cutoffs as new pairs are added
class rcut:

    ## \internal
    # \brief Initializes the class
    def __init__(self):
        self.values = {};

    ## \var values
    # \internal
    # \brief Contains the matrix of set r_cut values in a dictionary

    ## \internal
    # \brief Ensures a pair exists for the type by creating one if it doesn't exist
    # \details
    # \param a Atom type A
    # \param b Atom type B
    def ensure_pair(self,a,b):
        # create the pair if it hasn't been created yet
        if (not (a,b) in self.values) and (not (b,a) in self.values):
            self.values[(a,b)] = -1.0; # negative means this hasn't been set yet
           
        # find the pair we seek    
        if (a,b) in self.values:
            cur_pair = (a,b);
        elif (b,a) in self.values:
            cur_pair = (b,a);
        else:
            globals.msg.error("Bug ensuring pair exists in nlist.r_cut.ensure_pair. Please report.\n");
            raise RuntimeError("Error fetching rcut(i,j) pair");
        
        return cur_pair;
            
    ## \internal
    # \brief Forces a change of a single r_cut
    # \details
    # \param a Atom type A
    # \param b Atom type B
    # \param cutoff Cutoff radius
    def set_pair(self, a, b, cutoff):
        cur_pair = self.ensure_pair(a,b);
        
        if cutoff is None or cutoff is False:
            cutoff = -1.0
        else:
            cutoff = float(cutoff);
        self.values[cur_pair] = cutoff;

    ## \internal
    # \brief Attempts to update a single r_cut
    # \details Similar to set_pair, but updates to the larger r_cut value
    # \param a Atom type A
    # \param b Atom type B
    # \param cutoff Cutoff radius
    def merge_pair(self,a,b,cutoff):
        cur_pair = self.ensure_pair(a,b);
        
        if cutoff is None or cutoff is False:
            cutoff = -1.0
        else:
            cutoff = float(cutoff);
        self.values[cur_pair] = max(cutoff,self.values[cur_pair]); 
            
    ## \internal
    # \brief Gets the value of a single %pair coefficient
    # \param a First name in the type pair
    # \param b Second name in the type pair
    def get_pair(self, a, b):
        cur_pair = self.ensure_pair(a,b);
        return self.values[cur_pair];
    
    ## \internal
    # \brief Merges two rcut objects by maximum cutoff
    # \param rcut_obj The other rcut to merge in
    def merge(self,rcut_obj):
        for pair in rcut_obj.values:
            (a,b) = pair;
            self.merge_pair(a,b,rcut_obj.values[pair]);
        
    ## \internal
    # \brief Fills out the rcut(i,j) dictionary to include default unset keys
    #
    # This can only be run after the system has been initialized
    def fill(self):
        # first, check that the system has been initialized
        if not init.is_initialized():
            globals.msg.error("Cannot fill rcut(i,j) before initialization\n");
            raise RuntimeError('Error filling nlist rcut(i,j)');

        # get a list of types from the particle data
        ntypes = globals.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(globals.system_definition.getParticleData().getNameByType(i));

        # loop over all possible pairs and require that a dictionary key exists for them
        for i in range(0,ntypes):
            for j in range(i,ntypes):
                a = type_list[i];
                b = type_list[j];
                    
                # ensure the pair
                cur_pair = self.ensure_pair(a,b);

## %Cell list-based neighbor list
#
# nlist.cell creates a %cell list-based neighbor list object to which %pair potentials can be attached for computing
# non-bonded pairwise interactions. %Cell listing allows for O(N) construction of the neighbor list. Particles are first
# spatially sorted into cells based on the largest pairwise cutoff radius attached to this instance of the neighbor
# list. Particles then query their adjacent cells, and neighbors are included based on pairwise cutoffs. This method
# is very efficient for systems with nearly monodisperse cutoffs, but performance degrades for large cutoff radius
# asymmetries due to the significantly increased number of particles per %cell. Users can create multiple neighbor
# lists, and may see significant performance increases by doing so for systems with size asymmetry, especially when
# used in conjunction with nlist.tree.
#
# \b Examples:
# \code
# nl_c = nlist.cell(check_period = 1)
# nl_c.tune()
# \endcode
#
# \MPI_SUPPORTED
class cell(_nlist):
    ## Initialize a %cell neighbor list
    #
    # \param r_buff Buffer width
    # \param check_period How often to attempt to rebuild the neighbor list
    # \param d_max The maximum diameter a particle will achieve, only used in conjunction with slj diameter shifting
    # \param dist_check Flag to enable / disable distance checking
    # \param name Optional name for this neighbor list instance
    #
    # \note \a d_max should only be set when slj diameter shifting is required by a pair potential. Currently, slj
    # is the only %pair potential requiring this shifting, and setting \a d_max for other potentials may lead to
    # significantly degraded performance or incorrect results.
    def __init__(self, r_buff=None, check_period=1, d_max=None, dist_check=True, name=None):
        util.print_status_line()

        _nlist.__init__(self)

        if name is None:
            self.name = "cell_nlist_%d" % cell.cur_id
            cell.cur_id += 1
        else:
            self.name = name

        # the r_cut will be overridden by the pair potentials attached to the neighbor list
        default_r_cut = 0.0
        # assume r_buff = 0.4 as a typical default value that the user can (and should) override
        default_r_buff = 0.4

        # create the C++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_cl = hoomd.CellList(globals.system_definition)
            globals.system.addCompute(self.cpp_cl , self.name + "_cl")
            self.cpp_nlist = hoomd.NeighborListBinned(globals.system_definition, default_r_cut, default_r_buff, self.cpp_cl )
        else:
            self.cpp_cl  = hoomd.CellListGPU(globals.system_definition)
            globals.system.addCompute(self.cpp_cl , self.name + "_cl")
            self.cpp_nlist = hoomd.NeighborListGPUBinned(globals.system_definition, default_r_cut, default_r_buff, self.cpp_cl )

        self.cpp_nlist.setEvery(check_period, dist_check)

        globals.system.addCompute(self.cpp_nlist, self.name)
        
        # register this neighbor list with the globals
        globals.neighbor_lists += [self]
        
        # save the user defined parameters
        util._disable_status_lines = True
        self.set_params(r_buff, check_period, d_max, dist_check)
        util._disable_status_lines = False

    ## Change neighbor list parameters
    #
    # \param r_buff (if set) changes the buffer radius around the cutoff (in distance units)
    # \param check_period (if set) changes the period (in time steps) between checks to see if the neighbor list
    #        needs updating
    # \param d_max (if set) notifies the neighbor list of the maximum diameter that a particle attain over the following
    #        run() commands. (in distance units)
    # \param dist_check When set to False, disable the distance checking logic and always regenerate the nlist every
    #        \a check_period steps
    # \param deterministic (if set) Enable deterministic runs on the GPU by sorting the cell list
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
    # When using pair.slj, \a d_max \b MUST be set to the maximum diameter that a particle will attain at any point
    # during the following run() commands (see pair.slj for more information). When using in conjunction with pair.slj,
    # pair.slj will
    # automatically set \a d_max for the nlist.  This can be overridden (e.g. if multiple potentials using diameters are used)
    # by using nlist.set_params() after the
    # pair.slj class has been initialized.   When <i>not</i> using pair.slj (or other diameter-using potential), \a d_max
    # \b MUST be left at the default value of 1.0 or the simulation will be incorrect if d_max is less than 1.0 and slower
    # than necessary if d_max is greater than 1.0.
    #
    # \b Examples:
    # \code
    # nl.set_params(r_buff = 0.9)
    # nl.set_params(check_period = 11)
    # nl.set_params(r_buff = 0.7, check_period = 4)
    # nl.set_params(d_max = 3.0)
    # \endcode
    #
    # \note For truly deterministic simulations, also the autotuner should be disabled.
    # This can significantly decrease performance.
    #
    # \b Example:
    # \code
    # nlist.set_params(deterministic=True)
    # option.set_autotuner_params(enable=False)
    # \endcode
    def set_params(self, r_buff=None, check_period=None, d_max=None, dist_check=True, deterministic=None):
        util.print_status_line();

        if self.cpp_nlist is None:
            globals.msg.error('Bug in hoomd_script: cpp_nlist not set, please report\n');
            raise RuntimeError('Error setting neighbor list parameters');

        # update the parameters
        if r_buff is not None:
            self.cpp_nlist.setRBuff(r_buff);
            self.r_buff = r_buff;

        if check_period is not None:
            self.cpp_nlist.setEvery(check_period, dist_check);

        if d_max is not None:
            self.cpp_nlist.setMaximumDiameter(d_max);

        if deterministic is not None:
            self.cpp_cl.setSortCellList(deterministic)
cell.cur_id = 0

## %Cell list-based neighbor list using stencils
#
#
# nlist.stencil creates a %cell list-based neighbor list object to which %pair potentials can be attached for computing
# non-bonded pairwise interactions. %Cell listing allows for O(N) construction of the neighbor list. Particles are first
# spatially sorted into cells based on the largest pairwise cutoff radius attached to this instance of the neighbor
# list.
#
# This neighbor-list style differs from nlist.cell based on how the adjacent cells are searched for particles. The cell
# list \a cell_width is set by default using the shortest active cutoff radius in the system. One "stencil" is computed
# per particle type based on the largest cutoff radius that type participates in, which defines the bins that the
# particle must search in. Distances to the bins in the stencil are precomputed so that certain particles can be
# quickly excluded from the neighbor list, leading to improved performance compared to nlist.cell when there is size
# disparity in the cutoff radius.
#
# The performance of the %stencil depends strongly on the choice of \a cell_width. The best performance is obtained
# when the cutoff radii are multiples of the \a cell_width, and when the \a cell_width covers the simulation box with
# a roughly integer number of cells. The \a cell_width can be set manually, or be automatically scanning through a range
# of possible bin widths using stencil.tune_cell_width().
#
# \b Examples:
# \code
# nl_s = nlist.stencil(check_period = 1)
# nl_s.tune()
# nl_s.tune_cell_width(min_width=1.5, max_width=3.0)
# \endcode
#
# \MPI_SUPPORTED
class stencil(_nlist):
    ## Initialize a %stencil neighbor list
    #
    # \param r_buff Buffer width
    # \param check_period How often to attempt to rebuild the neighbor list
    # \param d_max The maximum diameter a particle will achieve, only used in conjunction with slj diameter shifting
    # \param dist_check Flag to enable / disable distance checking
    # \param cell_width The underlying stencil bin width for the cell list
    # \param name Optional name for this neighbor list instance
    #
    # \note \a d_max should only be set when slj diameter shifting is required by a pair potential. Currently, slj
    # is the only %pair potential requiring this shifting, and setting \a d_max for other potentials may lead to
    # significantly degraded performance or incorrect results.
    def __init__(self, r_buff=None, check_period=1, d_max=None, dist_check=True, cell_width=None, name=None):
        util.print_status_line()

        _nlist.__init__(self)

        if name is None:
            self.name = "stencil_nlist_%d" % stencil.cur_id
            stencil.cur_id += 1
        else:
            self.name = name

        # the r_cut will be overridden by the pair potentials attached to the neighbor list
        default_r_cut = 0.0
        # assume r_buff = 0.4 as a typical default value that the user can (and should) override
        default_r_buff = 0.4

        # create the C++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_cl = hoomd.CellList(globals.system_definition)
            globals.system.addCompute(self.cpp_cl , self.name + "_cl")
            cls = hoomd.CellListStencil(globals.system_definition, self.cpp_cl)
            globals.system.addCompute(cls, self.name + "_cls")
            self.cpp_nlist = hoomd.NeighborListStencil(globals.system_definition, default_r_cut, default_r_buff, self.cpp_cl, cls)
        else:
            self.cpp_cl  = hoomd.CellListGPU(globals.system_definition)
            globals.system.addCompute(self.cpp_cl , self.name + "_cl")
            cls = hoomd.CellListStencil(globals.system_definition, self.cpp_cl)
            globals.system.addCompute(cls, self.name + "_cls")
            self.cpp_nlist = hoomd.NeighborListGPUStencil(globals.system_definition, default_r_cut, default_r_buff, self.cpp_cl, cls)

        self.cpp_nlist.setEvery(check_period, dist_check)

        globals.system.addCompute(self.cpp_nlist, self.name)
        
        # register this neighbor list with the globals
        globals.neighbor_lists += [self]
        
        # save the user defined parameters
        util._disable_status_lines = True
        self.set_params(r_buff, check_period, d_max, dist_check, cell_width)
        util._disable_status_lines = False

    ## Change neighbor list parameters
    #
    # \param r_buff (if set) changes the buffer radius around the cutoff (in distance units)
    # \param check_period (if set) changes the period (in time steps) between checks to see if the neighbor list
    #        needs updating
    # \param d_max (if set) notifies the neighbor list of the maximum diameter that a particle attain over the following
    #        run() commands. (in distance units)
    # \param dist_check When set to False, disable the distance checking logic and always regenerate the nlist every
    #        \a check_period steps
    # \param cell_width The underlying stencil bin width for the cell list
    # \param deterministic (if set) Enable deterministic runs on the GPU by sorting the cell list
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
    # When using pair.slj, \a d_max \b MUST be set to the maximum diameter that a particle will attain at any point
    # during the following run() commands (see pair.slj for more information). When using in conjunction with pair.slj,
    # pair.slj will
    # automatically set \a d_max for the nlist.  This can be overridden (e.g. if multiple potentials using diameters are used)
    # by using nlist.set_params() after the
    # pair.slj class has been initialized.   When <i>not</i> using pair.slj (or other diameter-using potential), \a d_max
    # \b MUST be left at the default value of 1.0 or the simulation will be incorrect if d_max is less than 1.0 and slower
    # than necessary if d_max is greater than 1.0.
    #
    # \b Examples:
    # \code
    # nl.set_params(r_buff = 0.9)
    # nl.set_params(check_period = 11)
    # nl.set_params(r_buff = 0.7, check_period = 4)
    # nl.set_params(d_max = 3.0)
    # \endcode
    #
    # \note For truly deterministic simulations, also the autotuner should be disabled.
    # This can significantly decrease performance.
    #
    # \b Example:
    # \code
    # nlist.set_params(deterministic=True)
    # option.set_autotuner_params(enable=False)
    # \endcode
    def set_params(self, r_buff=None, check_period=None, d_max=None, dist_check=True, cell_width=None, deterministic=None):
        util.print_status_line();

        if self.cpp_nlist is None:
            globals.msg.error('Bug in hoomd_script: cpp_nlist not set, please report\n');
            raise RuntimeError('Error setting neighbor list parameters');

        # update the parameters
        if r_buff is not None:
            self.cpp_nlist.setRBuff(r_buff);
            self.r_buff = r_buff;

        if check_period is not None:
            self.cpp_nlist.setEvery(check_period, dist_check);

        if cell_width is not None:
            self.cpp_nlist.setCellWidth(cell_width)

        if d_max is not None:
            self.cpp_nlist.setMaximumDiameter(d_max);

        if deterministic is not None:
            self.cpp_cl.setSortCellList(deterministic)

    ## Make a series of short runs to determine the fastest performing bin width
    # \param warmup Number of time steps to run() to warm up the benchmark
    # \param min_width Minimum %cell bin width to try
    # \param max_width Maximum %cell bin width to try
    # \param jumps Number of different bin width to test
    # \param steps Number of time steps to run() at each point
    #
    # tune_cell_width() executes \a warmup time steps. Then it sets the nlist \a cell_width value to \a min_width and
    # runs for \a steps time steps. The TPS value is recorded, and the benchmark moves on to the next \a cell_width
    # value completing at \a max_width in \a jumps jumps. Status information is printed out to the screen, and the
    # optimal \a cell_width value is left set for further runs() to continue at optimal settings.
    #
    # Each benchmark is repeated 3 times and the median value chosen. In total, (warmup + 3*jump*steps) time steps
    # are run().
    #
    # \returns optimal_cell_width Optimal cell width
    #
    # \MPI_SUPPORTED
    def tune_cell_width(self, warmup=200000, min_width=None, max_width=None, jumps=20, steps=5000):
        util.print_status_line()

        # check if initialization has occurred
        if not init.is_initialized():
            globals.msg.error("Cannot tune r_buff before initialization\n");

        if self.cpp_nlist is None:
            globals.msg.error('Bug in hoomd_script: cpp_nlist not set, please report\n')
            raise RuntimeError('Error tuning neighbor list')

        min_cell_width = min_width
        if min_cell_width is None:
            min_cell_width = 0.5*self.cpp_nlist.getMinRList()
        max_cell_width = max_width
        if max_cell_width is None:
            max_cell_width = self.cpp_nlist.getMaxRList()

        # make the warmup run
        hoomd_script.run(warmup);

        # initialize scan variables
        dr = (max_cell_width - min_cell_width) / (jumps - 1);
        width_list = [];
        tps_list = [];

        # loop over all desired r_buff points
        for i in range(0,jumps):
            # set the current r_buff
            cw = min_cell_width + i * dr;
            self.set_params(cell_width=cw);

            # run the benchmark 3 times
            tps = [];
            hoomd_script.run(steps);
            tps.append(globals.system.getLastTPS())
            hoomd_script.run(steps);
            tps.append(globals.system.getLastTPS())
            hoomd_script.run(steps);
            tps.append(globals.system.getLastTPS())

            # record the median tps of the 3
            tps.sort();
            tps_list.append(tps[1]);
            width_list.append(cw);

        # find the fastest r_buff
        fastest = tps_list.index(max(tps_list));
        fastest_width = width_list[fastest];

        # set the fastest and rerun the warmup steps to identify the max check period
        self.set_params(cell_width=fastest_width);

        # notify the user of the benchmark results
        globals.msg.notice(2, "cell width = " + str(width_list) + '\n');
        globals.msg.notice(2, "tps = " + str(tps_list) + '\n');
        globals.msg.notice(2, "Optimal cell width: " + str(fastest_width) + '\n');

        # return the results to the script
        return fastest_width
stencil.cur_id = 0

## Fast neighbor list for size asymmetric particles
#
# nlist.tree creates a neighbor list using bounding volume hierarchy (BVH) tree traversal. %Pair potentials are attached
# for computing non-bonded pairwise interactions. A BVH tree of axis-aligned bounding boxes is constructed per particle
# type, and each particle queries each tree to determine its neighbors. This method of searching leads to significantly
# improved performance compared to %cell listing in systems with moderate size asymmetry, but has poorer performance
# for monodisperse systems. The user should carefully benchmark neighbor list build times to select the appropriate
# neighbor list construction type.
#
# Users can create multiple neighbor lists, and may see significant performance increases by doing so for systems with
# size asymmetry, especially when used in conjunction with nlist.cell.
#
# \b Examples:
# \code
# nl_t = nlist.tree(check_period = 1)
# nl_t.tune()
# \endcode
#
# \warning BVH tree neighbor lists are currently only supported on Kepler (sm_30) architecture devices and newer.
#
# \MPI_SUPPORTED
class tree(_nlist):
    ## Initialize a %tree neighbor list
    #
    # \param r_buff Buffer width
    # \param check_period How often to attempt to rebuild the neighbor list
    # \param d_max The maximum diameter a particle will achieve, only used in conjunction with slj diameter shifting
    # \param dist_check Flag to enable / disable distance checking
    # \param name Optional name for this neighbor list instance
    #
    # \note \a d_max should only be set when slj diameter shifting is required by a pair potential. Currently, slj
    # is the only %pair potential requiring this shifting, and setting \a d_max for other potentials may lead to
    # significantly degraded performance or incorrect results.
    #
    # \warning BVH tree neighbor lists are currently only supported on Kepler (sm_30) architecture devices and newer.
    #
    def __init__(self, r_buff=None, check_period=1, d_max=None, dist_check=True, name=None):
        util.print_status_line()

        _nlist.__init__(self)

        # the r_cut will be overridden by the pair potentials attached to the neighbor list
        default_r_cut = 0.0
        # assume r_buff = 0.4 as a typical default value that the user can (and should) override
        default_r_buff = 0.4

        # create the C++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_nlist = hoomd.NeighborListTree(globals.system_definition, default_r_cut, default_r_buff)
        else:
            self.cpp_nlist = hoomd.NeighborListGPUTree(globals.system_definition, default_r_cut, default_r_buff)

        self.cpp_nlist.setEvery(check_period, dist_check)

        if name is None:
            self.name = "tree_nlist_%d" % tree.cur_id
            tree.cur_id += 1
        else:
            self.name = name

        globals.system.addCompute(self.cpp_nlist, self.name)
        
        # register this neighbor list with the globals
        globals.neighbor_lists += [self]
        
        # save the user defined parameters
        util._disable_status_lines = True
        self.set_params(r_buff, check_period, d_max, dist_check)
        util._disable_status_lines = False
tree.cur_id = 0

## \internal
# \brief Creates the global neighbor list
# \details
# \param cb Callable function passed to subscribe()
# If no neighbor list has been created, create one. If there is one, subscribe the new potential and update the rcut
def _subscribe_global_nlist(cb):
    # create a global neighbor list if it doesn't exist
    if globals.neighbor_list is None:
        globals.neighbor_list = cell();
    
    # subscribe and force an update
    globals.neighbor_list.subscribe(cb);
    globals.neighbor_list.update_rcut();

    return globals.neighbor_list;

## Thin wrapper for changing parameters for the global neighbor list
#
# \param r_buff (if set) changes the buffer radius around the cutoff (in distance units)
# \param check_period (if set) changes the period (in time steps) between checks to see if the neighbor list
#        needs updating
# \param d_max (if set) notifies the neighbor list of the maximum diameter that a particle attain over the following
#        run() commands. (in distance units)
# \param dist_check When set to False, disable the distance checking logic and always regenerate the nlist every
#        \a check_period steps
# \param deterministic (if set) Enable deterministic runs on the GPU by sorting the cell list
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
# When using pair.slj, \a d_max \b MUST be set to the maximum diameter that a particle will attain at any point
# during the following run() commands (see pair.slj for more information). When using in conjunction with pair.slj,
# pair.slj will
# automatically set \a d_max for the nlist.  This can be overridden (e.g. if multiple potentials using diameters are used)
# by using nlist.set_params() after the
# pair.slj class has been initialized.   When <i>not</i> using pair.slj (or other diameter-using potential), \a d_max
# \b MUST be left at the default value of 1.0 or the simulation will be incorrect if d_max is less than 1.0 and slower
# than necessary if d_max is greater than 1.0.
#
# \b Examples:
# \code
# nlist.set_params(r_buff = 0.9)
# nlist.set_params(check_period = 11)
# nlist.set_params(r_buff = 0.7, check_period = 4)
# nlist.set_params(d_max = 3.0)
# \endcode
#
# \note For truly deterministic simulations, also the autotuner should be disabled.
# This can significantly decrease performance.
#
# \b Example:
# \code
# nlist.set_params(deterministic=True)
# option.set_autotuner_params(enable=False)
# \endcode
def set_params(r_buff=None, check_period=None, d_max=None, dist_check=True, deterministic=True):
    util.print_status_line();
    if globals.neighbor_list is None:
        globals.msg.error('Cannot set global neighbor list parameters without creating it first\n');
        raise RuntimeError('Error modifying global neighbor list');

    util._disable_status_lines = True;
    globals.neighbor_list.set_params(r_buff, check_period, d_max, dist_check, deterministic);
    util._disable_status_lines = False;

## Thin wrapper for resetting exclusion for global neighbor list
#
# \param exclusions Select which interactions should be excluded from the %pair interaction calculation.
#
# By default, the following are excluded from short range %pair interactions.
# - Directly bonded particles
# - Particles that are in the same rigid body
#
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
# - \b %body - Exclude particles that belong to the same body
#
# The following types are determined solely by the bond topology. Every chain of particles in the simulation
# connected by bonds (1-2-3-4) will be subject to the following exclusions, if enabled, whether or not explicit
# angles or dihedrals are defined.
# - \b 1-2  - Same as bond
# - \b 1-3  - Exclude particles connected with a sequence of two bonds.
# - \b 1-4  - Exclude particles connected with a sequence of three bonds.
#
# \b Examples:
# \code
# nlist.reset_exclusions(exclusions = ['1-2'])
# nlist.reset_exclusions(exclusions = ['1-2', '1-3', '1-4'])
# nlist.reset_exclusions(exclusions = ['bond', 'angle'])
# nlist.reset_exclusions(exclusions = [])
# \endcode
#
def reset_exclusions(exclusions = None):
    util.print_status_line();
    if globals.neighbor_list is None:
        globals.msg.error('Cannot set exclusions in global neighbor list without creating it first\n');
        raise RuntimeError('Error modifying global neighbor list');

    util._disable_status_lines = True;
    globals.neighbor_list.reset_exclusions(exclusions);
    util._disable_status_lines = False;

## Thin wrapper for benchmarking the global neighbor list
#
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
def benchmark(n):
    util.print_status_line();
    if globals.neighbor_list is None:
        globals.msg.error('Cannot benchmark global neighbor list without creating it first\n');
        raise RuntimeError('Error modifying global neighbor list');

    util._disable_status_lines = True;
    globals.neighbor_list.benchmark(n);
    util._disable_status_lines = False;

## Thin wrapper for querying the update period for the global neighbor list
#
# query_update_period examines the counts of nlist rebuilds during the previous run() command.
# It returns \c s-1, where s is the smallest update period experienced during that time.
# Use it after a medium-length warm up run with check_period=1 to determine what check_period to set
# for production runs.
#
# \note If the previous run() was short, insufficient sampling may cause the queried update period
# to be large enough to result in dangerous builds during longer runs. Unless you use a really long
# warm up run, subtract an additional 1 from this when you set check_period for additional safety.
#
def query_update_period():
    util.print_status_line();
    if globals.neighbor_list is None:
        globals.msg.error('Cannot query global neighbor list without creating it first\n');
        raise RuntimeError('Error modifying global neighbor list');

    util._disable_status_lines = True;
    globals.neighbor_list.query_update_period(*args, **kwargs);
    util._disable_status_lines = False;
