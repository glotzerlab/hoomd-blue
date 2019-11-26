# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander

R""" Neighbor list acceleration structures.

Neighbor lists accelerate pair force calculation by maintaining a list of particles within a cutoff radius.
Multiple pair forces can utilize the same neighbor list. Neighbors are included using a pairwise cutoff
:math:`r_\mathrm{cut}(i,j)` that is the maximum of all :math:`r_\mathrm{cut}(i,j)` set for the pair forces attached
to the list.

Multiple neighbor lists can be created to accelerate simulations where there is significant disparity in
:math:`r_\mathrm{cut}(i,j)` between pair potentials. If one pair force has a cutoff radius much smaller than
another pair force, the pair force calculation for the short cutoff will be slowed down considerably because many
particles in the neighbor list will have to be read and skipped because they lie outside the shorter cutoff.

The simplest way to build a neighbor list is :math:`O(N^2)`: each particle loops over all other particles and only
includes those within the neighbor list cutoff. This algorithm is no longer implemented in HOOMD-blue because it is
slow and inefficient. Instead, three accelerated algorithms based on cell lists and bounding volume hierarchy trees
are implemented. The cell list implementation is usually fastest when the cutoff radius is similar between all pair forces
(smaller than 2:1 ratio). The stencil implementation is a different variant of the cell list, and its main use is when a
cell list would be faster than a tree but memory demands are too big. The tree implementation is faster when there is
large size disparity, but its performance has been improved to be only slightly slower than the cell list for many use
cases. Because the performance of these algorithms depends on your system and hardware, you should carefully test which
option is fastest for your simulation.

Particles can be excluded from the neighbor list based on certain criteria. Setting :math:`r_\mathrm{cut}(i,j) \le 0`
will exclude this cross interaction from the neighbor list on build time. Particles can also be excluded by topology
or for belonging to the same rigid body (see :py:meth:`nlist.reset_exclusions()`). To support molecular structures,
the body flag can also be used to exclude particles that are not part of a rigid structure. All particles with
positive values of the body flag are considered part of a rigid body (see :py:class:`hoomd.md.constrain.rigid`),
while the default value of -1 indicates that a particle is free. Any other negative value of the body flag indicates
that the particles are part of a floppy body; such particles are integrated
separately, but are automatically excluded from the neighbor list as well.

Examples::

    nl_c = nlist.cell(check_period=1)
    nl_t = nlist.tree(r_buff = 0.8)
    lj1 = pair.lj(r_cut = 3.0, nlist=nl_c)
    lj2 = pair.lj(r_cut = 10.0, nlist=nl_t)

"""

from hoomd import _hoomd
from hoomd.md import _md
import hoomd;

class nlist:
    R""" Base class neighbor list.

    Methods provided by this base class are available to all subclasses.
    """

    def __init__(self):
        # check if initialization has occurred
        if not hoomd.init.is_initialized():
            hoomd.context.msg.error("Cannot create neighbor list before initialization\n");
            raise RuntimeError('Error creating neighbor list');

        # default exclusions
        self.is_exclusion_overridden = False;
        self.exclusions = None  # Excluded groups
        self.exclusion_list = []  # Specific pairs to exclude

        # save the parameters we set
        self.r_cut = rcut();
        self.r_buff = 0.4;

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
        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i));

        # loop over all possible pairs and require that a dictionary key exists for them
        for i in range(0,ntypes):
            for j in range(i,ntypes):
                a = type_list[i];
                b = type_list[j];
                self.cpp_nlist.setRCutPair(i,j, self.r_cut.get_pair(a,b));

    ## \internal
    # \brief Sets the default bond exclusions, but only if the defaults have not been overridden
    def update_exclusions_defaults(self):
        if self.cpp_nlist.wantExclusions() and self.exclusions is not None:
            hoomd.util.quiet_status();
            # update exclusions using stored values
            self.reset_exclusions(exclusions=self.exclusions)
            hoomd.util.unquiet_status();
        elif not self.is_exclusion_overridden:
            hoomd.util.quiet_status();
            self.reset_exclusions(exclusions=['body', 'bond','constraint']);
            hoomd.util.unquiet_status();

        # Add any specific interparticle exclusions
        for i, j in self.exclusion_list:
            hoomd.util.quiet_status();
            self.cpp_nlist.addExclusion(i, j)
            hoomd.util.unquiet_status();

    def set_params(self, r_buff=None, check_period=None, d_max=None, dist_check=True):
        R""" Change neighbor list parameters.

        Args:

            r_buff (float): (if set) changes the buffer radius around the cutoff (in distance units)
            check_period (int): (if set) changes the period (in time steps) between checks to see if the neighbor list
              needs updating
            d_max (float): (if set) notifies the neighbor list of the maximum diameter that a particle attain over the following
              run() commands. (in distance units)
            dist_check (bool): When set to False, disable the distance checking logic and always regenerate the nlist every
              *check_period* steps

        :py:meth:`set_params()` changes one or more parameters of the neighbor list. *r_buff* and *check_period*
        can have a significant effect on performance. As *r_buff* is made larger, the neighbor list needs
        to be updated less often, but more particles are included leading to slower force computations.
        Smaller values of *r_buff* lead to faster force computation, but more often neighbor list updates,
        slowing overall performance again. The sweet spot for the best performance needs to be found by
        experimentation. The default of *r_buff = 0.4* works well in practice for Lennard-Jones liquid
        simulations.

        As *r_buff* is changed, *check_period* must be changed correspondingly. The neighbor list is updated
        no sooner than *check_period* time steps after the last update. If *check_period* is set too high,
        the neighbor list may not be updated when it needs to be.

        For safety, the default check_period is 1 to ensure that the neighbor list is always updated when it
        needs to be. Increasing this to an appropriate value for your simulation can lead to performance gains
        of approximately 2 percent.

        *check_period* should be set so that no particle
        moves a distance more than *r_buff/2.0* during a the *check_period*. If this occurs, a *dangerous*
        *build* is counted and printed in the neighbor list statistics at the end of a :py:func:`hoomd.run()`.

        When using :py:class:`hoomd.md.pair.slj`, *d_max* **MUST** be set to the maximum diameter that a particle will
        attain at any point during the following :py:func:`hoomd.run()` commands (see :py:class:`hoomd.md.pair.slj` for more
        information). When using in conjunction, :py:class:`hoomd.md.pair.slj` will
        automatically set *d_max* for the nlist.  This can be overridden (e.g. if multiple potentials using diameters are used)
        by using :py:meth:`set_params()` after the
        :py:class:`hoomd.md.pair.slj` class has been initialized.

        .. caution::
            When **not** using :py:class:`hoomd.md.pair.slj`, *d_max*
            **MUST** be left at the default value of 1.0 or the simulation will be incorrect if d_max is less than 1.0
            and slower than necessary if d_max is greater than 1.0.

        Examples::

            nl.set_params(r_buff = 0.9)
            nl.set_params(check_period = 11)
            nl.set_params(r_buff = 0.7, check_period = 4)
            nl.set_params(d_max = 3.0)
        """
        hoomd.util.print_status_line();

        if self.cpp_nlist is None:
            hoomd.context.msg.error('Bug in hoomd: cpp_nlist not set, please report\n');
            raise RuntimeError('Error setting neighbor list parameters');

        # update the parameters
        if r_buff is not None:
            self.cpp_nlist.setRBuff(r_buff);
            self.r_buff = r_buff;

        if check_period is not None:
            self.cpp_nlist.setEvery(check_period, dist_check);

        if d_max is not None:
            self.cpp_nlist.setMaximumDiameter(d_max);

    def reset_exclusions(self, exclusions = None):
        R""" Resets all exclusions in the neighborlist.

        Args:
            exclusions (list): Select which interactions should be excluded from the pair interaction calculation.

        By default, the following are excluded from short range pair interactions"

        - Directly bonded particles.
        - Directly constrained particles.
        - Particles that are in the same body (i.e. have the same body flag). Note that these bodies need not be rigid.

        reset_exclusions allows the defaults to be overridden to add other exclusions or to remove
        the exclusion for bonded or constrained particles.

        Specify a list of desired types in the *exclusions* argument (or an empty list to clear all exclusions).
        All desired exclusions have to be explicitly listed, i.e. '1-3' does **not** imply '1-2'.

        Valid types are:

        - **bond** - Exclude particles that are directly bonded together.
        - **constraint** - Exclude particles that are directly constrained.
        - **angle** - Exclude the two outside particles in all defined angles.
        - **dihedral** - Exclude the two outside particles in all defined dihedrals.
        - **pair** - Exclude particles in all defined special pairs.
        - **body** - Exclude particles that belong to the same body.

        The following types are determined solely by the bond topology. Every chain of particles in the simulation
        connected by bonds (1-2-3-4) will be subject to the following exclusions, if enabled, whether or not explicit
        angles or dihedrals are defined:

        - **1-2**  - Same as bond
        - **1-3**  - Exclude particles connected with a sequence of two bonds.
        - **1-4**  - Exclude particles connected with a sequence of three bonds.

        Examples::

            nl.reset_exclusions(exclusions = ['1-2'])
            nl.reset_exclusions(exclusions = ['1-2', '1-3', '1-4'])
            nl.reset_exclusions(exclusions = ['bond', 'angle'])
            nl.reset_exclusions(exclusions = ['bond', 'angle','constraint'])
            nl.reset_exclusions(exclusions = [])

        """
        hoomd.util.print_status_line();
        self.is_exclusion_overridden = True;

        if self.cpp_nlist is None:
            hoomd.context.msg.error('Bug in hoomd: cpp_nlist not set, please report\n');
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

        if 'constraint' in exclusions:
            self.cpp_nlist.addExclusionsFromConstraints();
            exclusions.remove('constraint');

        if 'pair' in exclusions:
            self.cpp_nlist.addExclusionsFromPairs();
            exclusions.remove('pair');

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
            hoomd.context.msg.error('Exclusion type(s): ' + str(exclusions) +  ' are not supported\n');
            raise RuntimeError('Error resetting exclusions');

        # collect and print statistics about the number of exclusions.
        self.cpp_nlist.countExclusions();

    def add_exclusion(self, i, j):
        R"""Add a specific pair of particles to the exclusion list.

        Args:
            i (int): The tag of the first particle in the pair.
            j (int): The tag of the second particle in the pair.

        Examples::

            nl.add_exclusions(system.particles[0].tag, system.particles[1].tag)
        """
        hoomd.util.print_status_line();

        if self.cpp_nlist is None:
            hoomd.context.msg.error('Bug in hoomd_script: cpp_nlist not set, please report\n');
            raise RuntimeError('Error resetting exclusions');

        # store exclusions for later use
        self.exclusion_list.append((i, j))
        self.cpp_nlist.addExclusion(i, j);

    def query_update_period(self):
        R""" Query the maximum possible check_period.

        :py:meth:`query_update_period` examines the counts of nlist rebuilds during the previous :py:func:`hoomd.run()`.
        It returns ``s-1``, where *s* is the smallest update period experienced during that time.
        Use it after a medium-length warm up run with *check_period=1* to determine what check_period to set
        for production runs.

        Warning:
            If the previous :py:func:`hoomd.run()` was short, insufficient sampling may cause the queried update period
            to be large enough to result in dangerous builds during longer runs. Unless you use a really long
            warm up run, subtract an additional 1 from this when you set check_period for additional safety.

        """
        if self.cpp_nlist is None:
            hoomd.context.msg.error('Bug in hoomd: cpp_nlist not set, please report\n');
            raise RuntimeError('Error setting neighbor list parameters');

        return self.cpp_nlist.getSmallestRebuild()-1;

    def tune(self, warmup=200000, r_min=0.05, r_max=1.0, jumps=20, steps=5000, set_max_check_period=False, quiet=False):
        R""" Make a series of short runs to determine the fastest performing r_buff setting.

        Args:
            warmup (int): Number of time steps to run() to warm up the benchmark
            r_min (float): Smallest value of r_buff to test
            r_max (float): Largest value of r_buff to test
            jumps (int): Number of different r_buff values to test
            steps (int): Number of time steps to run() at each point
            set_max_check_period (bool): Set to True to enable automatic setting of the maximum nlist check_period
            quiet (bool): Quiet the individual run() calls.

        :py:meth:`tune()` executes *warmup* time steps. Then it sets the nlist *r_buff* value to *r_min* and runs for
        *steps* time steps. The TPS value is recorded, and the benchmark moves on to the next *r_buff* value
        completing at *r_max* in *jumps* jumps. Status information is printed out to the screen, and the optimal
        *r_buff* value is left set for further :py:func:`hoomd.run()` calls to continue at optimal settings.

        Each benchmark is repeated 3 times and the median value chosen. Then, *warmup* time steps are run again
        at the optimal *r_buff* in order to determine the maximum value of check_period. In total,
        ``(2*warmup + 3*jump*steps)`` time steps are run.

        Note:
            By default, the maximum check_period is **not** set for safety. If you wish to have it set
            when the call completes, call with the parameter *set_max_check_period=True*.

        Returns:
            (optimal_r_buff, maximum check_period)
        """
        hoomd.util.print_status_line();

        # check if initialization has occurred
        if not hoomd.init.is_initialized():
            hoomd.context.msg.error("Cannot tune r_buff before initialization\n");

        if self.cpp_nlist is None:
            hoomd.context.msg.error('Bug in hoomd: cpp_nlist not set, please report\n')
            raise RuntimeError('Error tuning neighbor list')

        # quiet the tuner starting here so that the user doesn't see all of the parameter set and run calls
        hoomd.util.quiet_status();

        # start off at a check_period of 1
        self.set_params(check_period=1)

        # make the warmup run
        hoomd.run(warmup, quiet=quiet);

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
            hoomd.run(steps, quiet=quiet);
            tps.append(hoomd.context.current.system.getLastTPS())
            hoomd.run(steps, quiet=quiet);
            tps.append(hoomd.context.current.system.getLastTPS())
            hoomd.run(steps, quiet=quiet);
            tps.append(hoomd.context.current.system.getLastTPS())

            # record the median tps of the 3
            tps.sort();
            tps_list.append(tps[1]);
            r_buff_list.append(r_buff);

        # find the fastest r_buff
        fastest = tps_list.index(max(tps_list));
        fastest_r_buff = r_buff_list[fastest];

        # set the fastest and rerun the warmup steps to identify the max check period
        self.set_params(r_buff=fastest_r_buff);
        hoomd.run(warmup, quiet=quiet);

        # all done with the parameter sets and run calls (mostly)
        hoomd.util.unquiet_status();

        # notify the user of the benchmark results
        hoomd.context.msg.notice(2, "r_buff = " + str(r_buff_list) + '\n');
        hoomd.context.msg.notice(2, "tps = " + str(tps_list) + '\n');
        hoomd.context.msg.notice(2, "Optimal r_buff: " + str(fastest_r_buff) + '\n');
        hoomd.context.msg.notice(2, "Maximum check_period: " + str(self.query_update_period()) + '\n');

        # set the found max check period
        if set_max_check_period:
            hoomd.util.quiet_status();
            self.set_params(check_period=self.query_update_period());
            hoomd.util.unquiet_status();

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
            hoomd.context.msg.error("Bug ensuring pair exists in nlist.r_cut.ensure_pair. Please report.\n");
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
    # \brief Gets the value of a single pair coefficient
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
        if not hoomd.init.is_initialized():
            hoomd.context.msg.error("Cannot fill rcut(i,j) before initialization\n");
            raise RuntimeError('Error filling nlist rcut(i,j)');

        # get a list of types from the particle data
        ntypes = hoomd.context.current.system_definition.getParticleData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getParticleData().getNameByType(i));

        # loop over all possible pairs and require that a dictionary key exists for them
        for i in range(0,ntypes):
            for j in range(i,ntypes):
                a = type_list[i];
                b = type_list[j];

                # ensure the pair
                cur_pair = self.ensure_pair(a,b);

class cell(nlist):
    R""" Cell list based neighbor list

    Args:
        r_buff (float):  Buffer width.
        check_period (int): How often to attempt to rebuild the neighbor list.
        d_max (float): The maximum diameter a particle will achieve, only used in conjunction with slj diameter shifting.
        dist_check (bool): Flag to enable / disable distance checking.
        name (str): Optional name for this neighbor list instance.
        deterministic (bool): When True, enable deterministic runs on the GPU by sorting the cell list.

    :py:class:`cell` creates a cell list based neighbor list object to which pair potentials can be attached for computing
    non-bonded pairwise interactions. Cell listing allows for *O(N)* construction of the neighbor list. Particles are first
    spatially sorted into cells based on the largest pairwise cutoff radius attached to this instance of the neighbor
    list. Particles then query their adjacent cells, and neighbors are included based on pairwise cutoffs. This method
    is very efficient for systems with nearly monodisperse cutoffs, but performance degrades for large cutoff radius
    asymmetries due to the significantly increased number of particles per cell.

    Use base class methods to change parameters (:py:meth:`set_params <nlist.set_params>`), reset the exclusion list
    (:py:meth:`reset_exclusions <nlist.reset_exclusions>`) or tune *r_buff* (:py:meth:`tune <nlist.tune>`).

    Examples::

        nl_c = nlist.cell(check_period = 1)
        nl_c.set_params(r_buff=0.5)
        nl_c.reset_exclusions([]);
        nl_c.tune()

    Note:
        *d_max* should only be set when slj diameter shifting is required by a pair potential. Currently, slj
        is the only pair potential requiring this shifting, and setting *d_max* for other potentials may lead to
        significantly degraded performance or incorrect results.
    """
    def __init__(self, r_buff=0.4, check_period=1, d_max=None, dist_check=True, name=None, deterministic=False):
        hoomd.util.print_status_line()

        nlist.__init__(self)

        if name is None:
            self.name = "cell_nlist_%d" % cell.cur_id
            cell.cur_id += 1
        else:
            self.name = name

        # create the C++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_cl = _hoomd.CellList(hoomd.context.current.system_definition)
            hoomd.context.current.system.addCompute(self.cpp_cl , self.name + "_cl")
            self.cpp_nlist = _md.NeighborListBinned(hoomd.context.current.system_definition, 0.0, r_buff, self.cpp_cl )
        else:
            self.cpp_cl  = _hoomd.CellListGPU(hoomd.context.current.system_definition)
            hoomd.context.current.system.addCompute(self.cpp_cl , self.name + "_cl")
            self.cpp_nlist = _md.NeighborListGPUBinned(hoomd.context.current.system_definition, 0.0, r_buff, self.cpp_cl )

        self.cpp_nlist.setEvery(check_period, dist_check)

        hoomd.context.current.system.addCompute(self.cpp_nlist, self.name)
        self.cpp_cl.setSortCellList(deterministic)

        # register this neighbor list with the context
        hoomd.context.current.neighbor_lists += [self]

        # save the user defined parameters
        hoomd.util.quiet_status()
        self.set_params(r_buff, check_period, d_max, dist_check)
        hoomd.util.unquiet_status()

cell.cur_id = 0

class stencil(nlist):
    R""" Cell list based neighbor list using stencils

    Args:
        r_buff (float):  Buffer width.
        check_period (int): How often to attempt to rebuild the neighbor list.
        d_max (float): The maximum diameter a particle will achieve, only used in conjunction with slj diameter shifting.
        dist_check (bool): Flag to enable / disable distance checking.
        cell_width (float): The underlying stencil bin width for the cell list
        name (str): Optional name for this neighbor list instance.
        deterministic (bool): When True, enable deterministic runs on the GPU by sorting the cell list.

    :py:class:`stencil` creates a cell list based neighbor list object to which pair potentials can be attached for computing
    non-bonded pairwise interactions. Cell listing allows for O(N) construction of the neighbor list. Particles are first
    spatially sorted into cells based on the largest pairwise cutoff radius attached to this instance of the neighbor
    list.

    `M.P. Howard et al. 2016 <http://dx.doi.org/10.1016/j.cpc.2016.02.003>`_ describes this neighbor list implementation
    in HOOMD-blue. Cite it if you utilize this neighbor list style in your work.

    This neighbor-list style differs from :py:class:`cell` based on how the adjacent cells are searched for particles. The cell
    list *cell_width* is set by default using the shortest active cutoff radius in the system. One *stencil* is computed
    per particle type based on the largest cutoff radius that type participates in, which defines the bins that the
    particle must search in. Distances to the bins in the stencil are precomputed so that certain particles can be
    quickly excluded from the neighbor list, leading to improved performance compared to :py:class:`cell` when there is size
    disparity in the cutoff radius. The memory demands of :py:class:`stencil` can also be lower than :py:class:`cell` if your
    system is large and has many small cells in it; however, :py:class:`tree` is usually a better choice for these systems.

    The performance of the stencil depends strongly on the choice of *cell_width*. The best performance is obtained
    when the cutoff radii are multiples of the *cell_width*, and when the *cell_width* covers the simulation box with
    a roughly integer number of cells. The *cell_width* can be set manually, or be automatically scanning through a range
    of possible bin widths using :py:meth:`tune_cell_width()`.

    Examples::

        nl_s = nlist.stencil(check_period = 1)
        nl_s.set_params(r_buff=0.5)
        nl_s.reset_exclusions([]);
        nl_s.tune()
        nl_s.tune_cell_width(min_width=1.5, max_width=3.0)

    Note:
        *d_max* should only be set when slj diameter shifting is required by a pair potential. Currently, slj
        is the only pair potential requiring this shifting, and setting *d_max* for other potentials may lead to
        significantly degraded performance or incorrect results.
    """
    def __init__(self, r_buff=0.4, check_period=1, d_max=None, dist_check=True, cell_width=None, name=None, deterministic=False):
        hoomd.util.print_status_line()

        # register the citation
        c = hoomd.cite.article(cite_key='howard2016',
                         author=['M P Howard', 'J A Anderson', 'A Nikoubashman', 'S C Glotzer', 'A Z Panagiotopoulos'],
                         title='Efficient neighbor list calculation for molecular simulation of colloidal systems using graphics processing units',
                         journal='Computer Physics Communications',
                         volume=203,
                         pages='45--52',
                         month='Mar',
                         year='2016',
                         doi='10.1016/j.cpc.2016.02.003',
                         feature='stenciled neighbor lists')
        hoomd.cite._ensure_global_bib().add(c)

        nlist.__init__(self)

        if name is None:
            self.name = "stencil_nlist_%d" % stencil.cur_id
            stencil.cur_id += 1
        else:
            self.name = name

        # create the C++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_cl = _hoomd.CellList(hoomd.context.current.system_definition)
            hoomd.context.current.system.addCompute(self.cpp_cl , self.name + "_cl")
            cls = _hoomd.CellListStencil(hoomd.context.current.system_definition, self.cpp_cl)
            hoomd.context.current.system.addCompute(cls, self.name + "_cls")
            self.cpp_nlist = _md.NeighborListStencil(hoomd.context.current.system_definition, 0.0, r_buff, self.cpp_cl, cls)
        else:
            self.cpp_cl  = _hoomd.CellListGPU(hoomd.context.current.system_definition)
            hoomd.context.current.system.addCompute(self.cpp_cl , self.name + "_cl")
            cls = _hoomd.CellListStencil(hoomd.context.current.system_definition, self.cpp_cl)
            hoomd.context.current.system.addCompute(cls, self.name + "_cls")
            self.cpp_nlist = _md.NeighborListGPUStencil(hoomd.context.current.system_definition, 0.0, r_buff, self.cpp_cl, cls)

        self.cpp_nlist.setEvery(check_period, dist_check)

        hoomd.context.current.system.addCompute(self.cpp_nlist, self.name)
        self.cpp_cl.setSortCellList(deterministic)

        # register this neighbor list with the context
        hoomd.context.current.neighbor_lists += [self]

        # save the user defined parameters
        hoomd.util.quiet_status()
        self.set_params(r_buff, check_period, d_max, dist_check)
        self.set_cell_width(cell_width)
        hoomd.util.unquiet_status()

    def set_cell_width(self, cell_width):
        R""" Set the cell width

        Args:
            cell_width (float): New cell width.
        """
        hoomd.util.print_status_line()
        if cell_width is not None:
            self.cpp_nlist.setCellWidth(float(cell_width))

    def tune_cell_width(self, warmup=200000, min_width=None, max_width=None, jumps=20, steps=5000):
        R""" Make a series of short runs to determine the fastest performing bin width.

        Args:
            warmup (int): Number of time steps to run() to warm up the benchmark
            min_width (float): Minimum cell bin width to try
            max_width (float): Maximum cell bin width to try
            jumps (int): Number of different bin width to test
            steps (int): Number of time steps to run() at each point

        :py:class:`tune_cell_width()` executes *warmup* time steps. Then it sets the nlist *cell_width* value to *min_width* and
        runs for *steps* time steps. The TPS value is recorded, and the benchmark moves on to the next *cell_width*
        value completing at *max_width* in *jumps* jumps. Status information is printed out to the screen, and the
        optimal *cell_width* value is left set for further runs() to continue at optimal settings.

        Each benchmark is repeated 3 times and the median value chosen. In total, ``(warmup + 3*jump*steps)`` time steps
        are run.

        Returns:
            The optimal cell width.
        """
        hoomd.util.print_status_line()

        # check if initialization has occurred
        if not hoomd.init.is_initialized():
            hoomd.context.msg.error("Cannot tune r_buff before initialization\n");

        if self.cpp_nlist is None:
            hoomd.context.msg.error('Bug in hoomd: cpp_nlist not set, please report\n')
            raise RuntimeError('Error tuning neighbor list')

        min_cell_width = min_width
        if min_cell_width is None:
            min_cell_width = 0.5*self.cpp_nlist.getMinRList()
        max_cell_width = max_width
        if max_cell_width is None:
            max_cell_width = self.cpp_nlist.getMaxRList()

        # quiet the tuner starting here so that the user doesn't see all of the parameter set and run calls
        hoomd.util.quiet_status();

        # make the warmup run
        hoomd.run(warmup);

        # initialize scan variables
        dr = (max_cell_width - min_cell_width) / (jumps - 1);
        width_list = [];
        tps_list = [];

        # loop over all desired cell width points
        for i in range(0,jumps):
            # set the current cell width
            cw = min_cell_width + i * dr;
            hoomd.util.quiet_status();
            self.set_cell_width(cell_width=cw)
            hoomd.util.unquiet_status();

            # run the benchmark 3 times
            tps = [];
            hoomd.run(steps);
            tps.append(hoomd.context.current.system.getLastTPS())
            hoomd.run(steps);
            tps.append(hoomd.context.current.system.getLastTPS())
            hoomd.run(steps);
            tps.append(hoomd.context.current.system.getLastTPS())

            # record the median tps of the 3
            tps.sort();
            tps_list.append(tps[1]);
            width_list.append(cw);

        # find the fastest cell width
        fastest = tps_list.index(max(tps_list));
        fastest_width = width_list[fastest];

        # set the fastest cell width
        self.set_cell_width(cell_width=fastest_width)

        # all done with the parameter sets and run calls (mostly)
        hoomd.util.unquiet_status();

        # notify the user of the benchmark results
        hoomd.context.msg.notice(2, "cell width = " + str(width_list) + '\n');
        hoomd.context.msg.notice(2, "tps = " + str(tps_list) + '\n');
        hoomd.context.msg.notice(2, "Optimal cell width: " + str(fastest_width) + '\n');

        # return the results to the script
        return fastest_width
stencil.cur_id = 0

class tree(nlist):
    R""" Bounding volume hierarchy based neighbor list.

    Args:
        r_buff (float):  Buffer width.
        check_period (int): How often to attempt to rebuild the neighbor list.
        d_max (float): The maximum diameter a particle will achieve, only used in conjunction with slj diameter shifting.
        dist_check (bool): Flag to enable / disable distance checking.
        name (str): Optional name for this neighbor list instance.

    :py:class:`tree` creates a neighbor list using bounding volume hierarchy (BVH) tree traversal. Pair potentials are attached
    for computing non-bonded pairwise interactions. A BVH tree of axis-aligned bounding boxes is constructed per particle
    type, and each particle queries each tree to determine its neighbors. This method of searching leads to significantly
    improved performance compared to cell lists in systems with moderate size asymmetry, but has slightly poorer performance
    (10% slower) for monodisperse systems. :py:class:`tree` can also be slower than :py:class:`cell` if there are multiple
    types in the system, but the cutoffs between types are identical. (This is because one BVH is created per type.)
    The user should carefully benchmark neighbor list build times to select the appropriate neighbor list construction type.

    `M.P. Howard et al. 2016 <http://dx.doi.org/10.1016/j.cpc.2016.02.003>`_ describes the original implementation of this
    algorithm for HOOMD-blue. `M.P. Howard et al. 2019 <https://doi.org/10.1016/j.commatsci.2019.04.004>`_ describes the
    improved algorithm that is currently implemented. Cite both if you utilize this neighbor list style in your work.

    Examples::

        nl_t = nlist.tree(check_period = 1)
        nl_t.set_params(r_buff=0.5)
        nl_t.reset_exclusions([]);
        nl_t.tune()

    Note:
        *d_max* should only be set when slj diameter shifting is required by a pair potential. Currently, slj
        is the only pair potential requiring this shifting, and setting *d_max* for other potentials may lead to
        significantly degraded performance or incorrect results.

    """
    def __init__(self, r_buff=0.4, check_period=1, d_max=None, dist_check=True, name=None):
        hoomd.util.print_status_line()

        # register the citation
        c1 = hoomd.cite.article(cite_key='howard2016',
                         author=['M P Howard', 'J A Anderson', 'A Nikoubashman', 'S C Glotzer', 'A Z Panagiotopoulos'],
                         title='Efficient neighbor list calculation for molecular simulation of colloidal systems using graphics processing units',
                         journal='Computer Physics Communications',
                         volume=203,
                         pages='45--52',
                         month='Mar',
                         year='2016',
                         doi='10.1016/j.cpc.2016.02.003',
                         feature='tree neighbor lists')
        c2 = hoomd.cite.article(cite_key='howard2019',
                         author=['M P Howard', 'A Statt', 'F Madutsa', 'T M Truskett', 'A Z Panagiotopoulos'],
                         title='Quantized bounding volume hierarchies for neighbor search in molecular simulations on graphics processing units',
                         journal='Computational Materials Science',
                         volume=164,
                         pages='139--146',
                         month='Jun',
                         year='2019',
                         doi='10.1016/j.commatsci.2019.04.004',
                         feature='tree neighbor lists')
        hoomd.cite._ensure_global_bib().add((c1,c2))

        nlist.__init__(self)

        # create the C++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_nlist = _md.NeighborListTree(hoomd.context.current.system_definition, 0.0, r_buff)
        else:
            self.cpp_nlist = _md.NeighborListGPUTree(hoomd.context.current.system_definition, 0.0, r_buff)

        self.cpp_nlist.setEvery(check_period, dist_check)

        if name is None:
            self.name = "tree_nlist_%d" % tree.cur_id
            tree.cur_id += 1
        else:
            self.name = name

        hoomd.context.current.system.addCompute(self.cpp_nlist, self.name)

        # register this neighbor list with the context
        hoomd.context.current.neighbor_lists += [self]

        # save the user defined parameters
        hoomd.util.quiet_status()
        self.set_params(r_buff, check_period, d_max, dist_check)
        hoomd.util.unquiet_status()
tree.cur_id = 0
