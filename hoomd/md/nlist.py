# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""Neighbor list acceleration structures.

Pair forces (`hoomd.md.pair`) use neighbor list data structures to perform
efficient calculations. HOOMD-blue provides a several types of neighbor list
construction algorithms that you can select from. Multiple pair force objects
can share a neighbor list, or use independent neighbor list objects. When
neighbor lists are shared, they find neighbors within the the maximum
:math:`r_{\mathrm{cut},i,j}` over the associated pair potentials.
"""

import hoomd
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyFrom
from hoomd.logging import log
from hoomd.md import _md
from hoomd.operation import _HOOMDBaseObject


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

    Attributes:
        buffer (float): Buffer width :math:`[\mathrm{length}]`.
        exclusions (tuple[str]): Defines which particles to exlclude from the
            neighbor list, see more details above.
        rebuild_check_delay (int): How often to attempt to rebuild the neighbor
            list.
        diameter_shift (bool): Flag to enable / disable diameter shifting.
        check_dist (bool): Flag to enable / disable distance checking.
        max_diameter (float): The maximum diameter a particle will achieve
            :math:`[\mathrm{length}]`.
    """

    def __init__(self, buffer, exclusions, rebuild_check_delay, diameter_shift,
                 check_dist, max_diameter):

        validate_exclusions = OnlyFrom([
            'bond', 'angle', 'constraint', 'dihedral', 'special_pair', 'body',
            '1-3', '1-4'
        ])
        # default exclusions
        params = ParameterDict(exclusions=[validate_exclusions],
                               buffer=float(buffer),
                               rebuild_check_delay=int(rebuild_check_delay),
                               check_dist=bool(check_dist),
                               diameter_shift=bool(diameter_shift),
                               max_diameter=float(max_diameter))
        params["exclusions"] = exclusions
        self._param_dict.update(params)

    @log(requires_run=True)
    def shortest_rebuild(self):
        """int: The shortest period between neighbor list rebuilds.

        `shortest_rebuild` is the smallest number of time steps between neighbor
        list rebuilds during the previous `Simulation.run`.
        """
        return self._cpp_obj.getSmallestRebuild()

    def _remove_dependent(self, obj):
        super()._remove_dependent(obj)
        if len(self._dependents) == 0:
            if self._attached:
                self._detach()
                self._remove()
                return
            if self._added:
                self._remove()

    # TODO need to add tuning Updater for NList


class Cell(NList):
    r"""Neighbor list computed via a cell list.

    Args:
        buffer (float): Buffer width :math:`[\mathrm{length}]`.
        exclusions (tuple[str]): Defines which particles to exlclude from the
            neighbor list, see more details in `NList`.
        rebuild_check_delay (int): How often to attempt to rebuild the neighbor
            list.
        diameter_shift (bool): Flag to enable / disable diameter shifting.
        check_dist (bool): Flag to enable / disable distance checking.
        max_diameter (float): The maximum diameter a particle will achieve
            :math:`[\mathrm{length}]`.
        deterministic (bool): When `True`, sort neighbors to help provide
            deterministic simulation runs.

    `Cell` finds neighboring particles using a fixed width cell list, allowing
    for *O(kN)* construction of the neighbor list where *k* is the number of
    particles per cell. Cells are sized to the largest :math:`r_\mathrm{cut}`.
    This method is very efficient for systems with nearly monodisperse cutoffs,
    but performance degrades for large cutoff radius asymmetries due to the
    significantly increased number of particles per cell.

    Examples::

        cell = nlist.Cell()

    Attributes:
        deterministic (bool): When `True`, sort neighbors to help provide
            deterministic simulation runs.
    """

    def __init__(self,
                 buffer,
                 exclusions=('bond',),
                 rebuild_check_delay=1,
                 diameter_shift=False,
                 check_dist=True,
                 max_diameter=1.0,
                 deterministic=False):

        super().__init__(buffer, exclusions, rebuild_check_delay,
                         diameter_shift, check_dist, max_diameter)

        self._param_dict.update(
            ParameterDict(deterministic=bool(deterministic)))

    def _attach(self):
        if isinstance(self._simulation.device, hoomd.device.CPU):
            nlist_cls = _md.NeighborListBinned
        else:
            nlist_cls = _md.NeighborListGPUBinned
        self._cpp_obj = nlist_cls(self._simulation.state._cpp_sys_def,
                                  self.buffer)
        super()._attach()


class Stencil(NList):
    """Cell list based neighbor list using stencils.

    Args:
        cell_width (float): The underlying stencil bin width for the cell list
            :math:`[\\mathrm{length}]`.
        buffer (float): Buffer width :math:`[\\mathrm{length}]`.
        exclusions (tuple[str]): Defines which particles to exlclude from the
            neighbor list, see more details in `NList`.
        rebuild_check_delay (int): How often to attempt to rebuild the neighbor
            list.
        diameter_shift (bool): Flag to enable / disable diameter shifting.
        check_dist (bool): Flag to enable / disable distance checking.
        max_diameter (float): The maximum diameter a particle will achieve
            :math:`[\\mathrm{length}]`.
        deterministic (bool): When `True`, sort neighbors to help provide
            deterministic simulation runs.

    `Stencil` creates a cell list based neighbor list object to which pair
    potentials can be attached for computing non-bonded pairwise interactions.
    Cell listing allows for O(N) construction of the neighbor list. Particles
    are first spatially sorted into cells with the given width `cell_width`.

    `M.P. Howard et al. 2016 <http://dx.doi.org/10.1016/j.cpc.2016.02.003>`_
    describes this neighbor list implementation in HOOMD-blue. Cite it if you
    utilize this neighbor list style in your work.

    This neighbor list style differs from `Cell` in how the adjacent cells are
    searched for particles. One stencil is computed per particle type based on
    the value of *cell_width* set by the user, which defines the bins that the
    particle must search in. Distances to the bins in the stencil are
    precomputed so that certain particles can be quickly excluded from the
    neighbor list, leading to improved performance compared to `Cell` when there
    is size disparity in the cutoff radius. The memory demands of `Stencil` can
    also be lower than `Cell` if your system is large and has many small cells
    in it; however, `Tree` is usually a better choice for these systems.

    The performance of `Stencil` depends strongly on the choice of *cell_width*.
    The best performance is obtained when the cutoff radii are multiples of the
    *cell_width*, and when the *cell_width* covers the simulation box with a
    roughly integer number of cells. The *cell_width* must be set manually.

    Examples::

        nl_s = nlist.Stencil(cell_width=1.5)

    Attributes:
        cell_width (float): The underlying stencil bin width for the cell list
            :math:`[\\mathrm{length}]`.
        deterministic (bool): When `True`, sort neighbors to help provide
            deterministic simulation runs.
    """

    def __init__(self,
                 cell_width,
                 buffer,
                 exclusions=('bond',),
                 rebuild_check_delay=1,
                 diameter_shift=False,
                 check_dist=True,
                 max_diameter=1.0,
                 deterministic=False):

        super().__init__(buffer, exclusions, rebuild_check_delay,
                         diameter_shift, check_dist, max_diameter)

        params = ParameterDict(deterministic=bool(deterministic),
                               cell_width=float(cell_width))

        self._param_dict.update(params)

    def _attach(self):
        if isinstance(self._simulation.device, hoomd.device.CPU):
            nlist_cls = _md.NeighborListStencil
        else:
            nlist_cls = _md.NeighborListGPUStencil
        self._cpp_obj = nlist_cls(self._simulation.state._cpp_sys_def,
                                  self.buffer)
        super()._attach()


class Tree(NList):
    """Bounding volume hierarchy based neighbor list.

    Args:
        buffer (float): Buffer width :math:`[\\mathrm{length}]`.
        exclusions (tuple[str]): Defines which particles to exlclude from the
            neighbor list, see more details in `NList`.
        rebuild_check_delay (int): How often to attempt to rebuild the neighbor
            list.
        diameter_shift (bool): Flag to enable / disable diameter shifting.
        check_dist (bool): Flag to enable / disable distance checking.
        max_diameter (float): The maximum diameter a particle will achieve
            :math:`[\\mathrm{length}]`.

    `Tree` creates a neighbor list using a bounding volume hierarchy (BVH) tree
    traversal. A BVH tree of axis-aligned bounding boxes is constructed per
    particle type, and each particle queries each tree to determine its
    neighbors. This method of searching leads to significantly improved
    performance compared to cell lists in systems with moderate size asymmetry,
    but has slightly poorer performance (10% slower) for monodisperse systems.
    `Tree` can also be slower than `Cell` if there are multiple types in the
    system, but the cutoffs between types are identical. (This is because one
    BVH is created per type.) The user should carefully benchmark neighbor list
    build times to select the appropriate neighbor list construction type.

    `M.P. Howard et al. 2016 <http://dx.doi.org/10.1016/j.cpc.2016.02.003>`_
    describes the original implementation of this algorithm for HOOMD-blue.
    `M.P. Howard et al. 2019 <https://doi.org/10.1016/j.commatsci.2019.04.004>`_
    describes the improved algorithm that is currently implemented. Cite both
    if you utilize this neighbor list style in your work.

    Examples::

        nl_t = nlist.Tree(check_dist=False)
    """

    def __init__(self,
                 buffer,
                 exclusions=('bond',),
                 rebuild_check_delay=1,
                 diameter_shift=False,
                 check_dist=True,
                 max_diameter=1.0):

        super().__init__(buffer, exclusions, rebuild_check_delay,
                         diameter_shift, check_dist, max_diameter)

    def _attach(self):
        if isinstance(self._simulation.device, hoomd.device.CPU):
            nlist_cls = _md.NeighborListTree
        else:
            nlist_cls = _md.NeighborListGPUTree
        self._cpp_obj = nlist_cls(self._simulation.state._cpp_sys_def,
                                  self.buffer)
        super()._attach()
