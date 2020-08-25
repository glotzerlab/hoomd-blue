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
or for belonging to the same rigid body (see ``nlist.reset_exclusions``). To support molecular structures,
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
import hoomd
from hoomd.typeconverter import OnlyFrom
from hoomd.parameterdicts import ParameterDict
from hoomd.operation import _HOOMDBaseObject
from hoomd.logging import log


class nlist:
    pass


class _NList(_HOOMDBaseObject):
    R""" Base class neighbor list.

    Methods provided by this base class are available to all subclasses.
    """

    def __init__(self, buffer, exclusions, rebuild_check_delay,
                 diameter_shift, check_dist, max_diameter):

        validate_exclusions = OnlyFrom(
            ['bond', 'angle', 'constraint', 'dihedral', 'special_pair',
             'body', '1-3', '1-4']
        )
        # default exclusions
        params = ParameterDict(exclusions=[validate_exclusions],
                               buffer=float(buffer),
                               rebuild_check_delay=int(rebuild_check_delay),
                               check_dist=bool(check_dist),
                               diameter_shift=bool(diameter_shift),
                               max_diameter=float(max_diameter),
                               _defaults={'exclusions': exclusions}
                               )
        self._param_dict.update(params)

    @log
    def shortest_rebuild(self):
        R""" Query the maximum possible check_period.

        :py:meth:`query_update_period` examines the counts of nlist rebuilds
        during the previous ```hoomd.run```.  It returns ``s-1``, where
        *s* is the smallest update period experienced during that time.  Use it
        after a medium-length warm up run with *check_period=1* to determine
        what check_period to set for production runs.

        Warning:
            If the previous ```hoomd.run``` was short, insufficient
            sampling may cause the queried update period to be large enough to
            result in dangerous builds during longer runs. Unless you use a
            really long warm up run, subtract an additional 1 from this when you
            set check_period for additional safety.

        """
        if not self._attached():
            return None
        else:
            return self._cpp_obj.getSmallestRebuild() - 1

    # TODO need to add tuning Updater for NList


## \internal
# \brief %nlist r_cut matrix
# \details
# Holds the maximum cutoff radius by pair type, and gracefully updates maximum cutoffs as new pairs are added
class rcut:
    pass


class Cell(_NList):
    R""" Cell list based neighbor list

    Args:
        buffer (float):  Buffer width.
        exclusions (tuple): ...
        rebuild_check_delay (int): How often to attempt to rebuild the neighbor
            list.
        diameter_shift (bool): ...
        check_dist (bool): Flag to enable / disable distance checking.
        max_diameter (float): The maximum diameter a particle will achieve, only
            used in conjunction with slj diameter shifting.
        deterministic (bool): When True, enable deterministic runs on the GPU by
            sorting the cell list.

    :py:class:`Cell` creates a cell list based neighbor list object to which
    pair potentials can be attached for computing non-bonded pairwise
    interactions. Cell listing allows for *O(N)* construction of the neighbor
    list. Particles are first spatially sorted into cells based on the largest
    pairwise cutoff radius attached to this instance of the neighbor list.
    Particles then query their adjacent cells, and neighbors are included based
    on pairwise cutoffs. This method is very efficient for systems with nearly
    monodisperse cutoffs, but performance degrades for large cutoff radius
    asymmetries due to the significantly increased number of particles per cell.

    Use base class methods to change parameters ``set_params``,
    reset the exclusion list (``reset_exclusions``) or tune
    *r_buff* ``nlist.tune``).

    Examples::

        nl_c = nlist.cell(rebuild_check_delay = 1)
        nl_c.buffer = 0.5
        nl_c.exclusions = tuple();

    Note:
        *max_diameter* should only be set when slj diameter shifting is required
            by a pair potential. Currently, slj is the only pair potential
            requiring this shifting, and setting *d_max* for other potentials
            may lead to significantly degraded performance or incorrect results.
    """

    def __init__(self, buffer=0.4, exclusions=('bond',), rebuild_check_delay=1,
                 diameter_shift=False, check_dist=True, max_diameter=1.0,
                 deterministic=False):

        super().__init__(buffer, exclusions, rebuild_check_delay,
                         diameter_shift, check_dist, max_diameter)

        self._param_dict.update(
            ParameterDict(deterministic=bool(deterministic)))

    def _attach(self, simulation):
        if not simulation.device.cpp_exec_conf.isCUDAEnabled():
            cell_cls = _hoomd.CellList
            nlist_cls = _md.NeighborListBinned
        else:
            cell_cls = _hoomd.CellListGPU
            nlist_cls = _md.NeighborListGPUBinned
        self._cpp_cell = cell_cls(simulation.state._cpp_sys_def)
        # TODO remove 0.0 (r_cut) from constructor
        self._cpp_obj = nlist_cls(simulation.state._cpp_sys_def, 0.0,
                                  self.buffer, self._cpp_cell)
        super()._attach(simulation)

    def _detach(self):
        del self._cpp_cell
        super()._detach()


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

    This neighbor-list style differs from :py:class:`Cell` based on how the adjacent cells are searched for particles. The cell
    list *cell_width* is set by default using the shortest active cutoff radius in the system. One *stencil* is computed
    per particle type based on the largest cutoff radius that type participates in, which defines the bins that the
    particle must search in. Distances to the bins in the stencil are precomputed so that certain particles can be
    quickly excluded from the neighbor list, leading to improved performance compared to :py:class:`Cell` when there is size
    disparity in the cutoff radius. The memory demands of :py:class:`stencil` can also be lower than :py:class:`Cell` if your
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
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
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
        self.set_params(r_buff, check_period, d_max, dist_check)
        self.set_cell_width(cell_width)

    def set_cell_width(self, cell_width):
        R""" Set the cell width

        Args:
            cell_width (float): New cell width.
        """
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

        # check if initialization has occurred
        if not hoomd.init.is_initialized():
            hoomd.context.current.device.cpp_msg.error("Cannot tune r_buff before initialization\n");

        if self.cpp_nlist is None:
            hoomd.context.current.device.cpp_msg.error('Bug in hoomd: cpp_nlist not set, please report\n')
            raise RuntimeError('Error tuning neighbor list')

        min_cell_width = min_width
        if min_cell_width is None:
            min_cell_width = 0.5*self.cpp_nlist.getMinRList()
        max_cell_width = max_width
        if max_cell_width is None:
            max_cell_width = self.cpp_nlist.getMaxRList()

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
            self.set_cell_width(cell_width=cw)

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

        # notify the user of the benchmark results
        hoomd.context.current.device.cpp_msg.notice(2, "cell width = " + str(width_list) + '\n');
        hoomd.context.current.device.cpp_msg.notice(2, "tps = " + str(tps_list) + '\n');
        hoomd.context.current.device.cpp_msg.notice(2, "Optimal cell width: " + str(fastest_width) + '\n');

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
    (10% slower) for monodisperse systems. :py:class:`tree` can also be slower than :py:class:`Cell` if there are multiple
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
        if not hoomd.context.current.device.cpp_exec_conf.isCUDAEnabled():
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
        self.set_params(r_buff, check_period, d_max, dist_check)

tree.cur_id = 0
