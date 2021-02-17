# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

r"""Neighbor list acceleration structures.

Pair forces (`hoomd.md.pair`) use neighbor list data structures to perform
efficient calculations. HOOMD-blue provides a several types of neighbor list
construction algorithms that you can select from. Multiple pair force objects
can share a neighbor list, or use independent neighbor list objects. When
neighbor lists are shared, they find neighbors within the the maximum
:math:`r_{\mathrm{cut},i,j}` over the associated pair potentials.
"""

from hoomd import _hoomd
from hoomd.md import _md
import hoomd
from hoomd.data.typeconverter import OnlyFrom
from hoomd.data.parameterdicts import ParameterDict
from hoomd.operation import _HOOMDBaseObject
from hoomd.logging import log


class nlist:
    """Obsolete parent class for neighbor list objects.
    """
    pass

# To Do: Migrate all hoomdv2 codes still using nlist to NList

class NList(_HOOMDBaseObject):
    r"""Base class neighbor list.

    Methods and attributes provided by this base class are available to all
    subclasses.

    Attention:
        Users should instantiate the subclasses, using `NList` directly
        will result in an error.

    .. rubric:: Buffer distance

    Set the `buffer` distance to amortize the cost of the neighbor list build.
    When ``buffer > 0``, a neighbor list computed on one step can be reused on
    subsequent steps until a particle moves a distance ``buffer/2``. When
    `check_dist` is `True`, `NList` starts checking how far particles have
    moved `rebuild_check_delay` time steps after the last build and performs a
    rebuild when any particle has moved a distance ``buffer/2``. When
    `check_dist` is `False`, `NList` always rebuilds after
    `rebuild_check_delay` time steps.

    .. rubric:: Exclusions

    Neighbor lists nominally include all particles within the specified cutoff
    distances. The `exclusions` attribute defines which particles will be
    excluded from the list, even if they are within the cutoff. `exclusions`
    is a tuple of strings that enable one more more types of exclusions.
    The valid exclusion types are:

    * ``bond``: Exclude particles that are directly bonded together.
    * ``angle``: Exclude the first and third particles in each angle.
    * ``constraint``: Exclude particles that have a distance constraint applied
      between them.
    * ``dihedral``: Exclude the first and fourth particles in each dihedral.
    * ``special_pair``: Exclude particles that are part of a special pair.
    * ``body``: Exclude particles that belong to the same rigid body.
    * ``1-3``: Exclude particles *i* and *k* whenever there is a bond (i,j) and
      a bond (j,k).
    * ``1-4``: Exclude particles *i* and *m* whenever there are bonds (i,j),
      (j,k), and (k,m).

    .. rubric:: Diameter shifting

    Set `diameter_shift` to `True` when using `hoomd.md.pair.SLJ` or
    `hoomd.md.pair.DLVO` so that the neighbor list includes all particles that
    interact under the modified :math:`r_\mathrm{cut}` conditions in those
    potentials. When `diameter_shift` is `True`, set `max_diameter` to the
    largest value that any particle's diameter will achieve (where **diameter**
    is the per particle quantity stored in the `hoomd.State`).

    Attributes:
        buffer (float): Buffer width.
        check_dist (bool): Flag to enable / disable distance checking.
        diameter_shift (bool): Flag to enable / disable diameter shifting.
        exclusions (tuple[str]): Excludes pairs from the neighbor list, which
            excludes them from the pair potential calculation.
        max_diameter (float): The maximum diameter a particle will achieve.
        rebuild_check_delay (int): How often to attempt to rebuild the neighbor
            list.
    """

    _remove_for_pickling = _HOOMDBaseObject._remove_for_pickling + ('_cpp_cell',)

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
        """int: The shortest period between neighbor list rebuilds.

        `shortest_rebuild` is the smallest number of time steps between neighbor
        list rebuilds during the previous `Simulation.run`.
        """
        if not self._attached:
            return None
        else:
            return self._cpp_obj.getSmallestRebuild()

    # TODO need to add tuning Updater for NList


## \internal
# \brief %nlist r_cut matrix
# \details
# Holds the maximum cutoff radius by pair type, and gracefully updates maximum cutoffs as new pairs are added
class rcut:
    pass


class Cell(NList):
    r"""Cell list based neighbor list

    Args:
        buffer (float): Buffer width.
        check_dist (bool): Flag to enable / disable distance checking.
        deterministic (bool): When `True`, sort neighbors to help provide
            deterministic simulation runs.
        diameter_shift (bool): Flag to enable / disable diameter shifting.
        exclusions (tuple[str]): Excludes pairs from the neighbor list, which
            excludes them from the pair potential calculation.
        max_diameter (float): The maximum diameter a particle will achieve.
        rebuild_check_delay (int): How often to attempt to rebuild the neighbor
            list.

    `Cell` finds neighboring particles using a fixed width cell list, allowing
    for *O(kN)* construction of the neighbor list where *k* is the number of
    particles per cell. Cells are sized to the largest :math:`r_\mathrm{cut}`.
    This method is very efficient for systems with nearly monodisperse cutoffs,
    but performance degrades for large cutoff radius asymmetries due to the
    significantly increased number of particles per cell.

    Examples::

        cell = nlist.Cell()
        lj = md.pair.LJ(nlist=cell)

    Attributes:
        deterministic (bool): When `True`, sort neighbors to help provide
            deterministic simulation runs.
    """

    def __init__(self, buffer=0.4, exclusions=('bond',), rebuild_check_delay=1,
                 diameter_shift=False, check_dist=True, max_diameter=1.0,
                 deterministic=False):

        super().__init__(buffer, exclusions, rebuild_check_delay,
                         diameter_shift, check_dist, max_diameter)

        self._param_dict.update(
            ParameterDict(deterministic=bool(deterministic)))

    def _attach(self):
        if isinstance(self._simulation.device, hoomd.device.CPU):
            cell_cls = _hoomd.CellList
            nlist_cls = _md.NeighborListBinned
        else:
            cell_cls = _hoomd.CellListGPU
            nlist_cls = _md.NeighborListGPUBinned
        self._cpp_cell = cell_cls(self._simulation.state._cpp_sys_def)
        # TODO remove 0.0 (r_cut) from constructor
        self._cpp_obj = nlist_cls(self._simulation.state._cpp_sys_def, 0.0,
                                  self.buffer, self._cpp_cell)
        super()._attach()

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
