# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

r"""Neighbor list acceleration structures.

Pair forces (`hoomd.md.pair`) use neighbor list data structures to find
neighboring particle pairs (those within a distance of :math:`r_\mathrm{cut}`)
efficiently. HOOMD-blue provides a several types of neighbor list construction
algorithms that you can select from: `Cell`, `Tree`, and `Stencil`.

Multiple pair force objects can share a single neighbor list, or use independent
neighbor list objects. When neighbor lists are shared, they find neighbors
within the the maximum :math:`r_{\mathrm{cut},i,j}` over the associated pair
potentials.

.. rubric:: Buffer distance

Set the `NeighborList.buffer` distance to amortize the cost of the neighbor list
build. When ``buffer > 0``, a neighbor list computed on one step can be reused
on subsequent steps until a particle moves a distance ``buffer/2``. When
`NeighborList.check_dist` is `True`, `NeighborList` starts checking how far
particles have moved `NeighborList.rebuild_check_delay` time steps after the
last build and performs a rebuild when any particle has moved a distance
``buffer/2``. When `NeighborList.check_dist` is `False`, `NeighborList` always
rebuilds after `NeighborList.rebuild_check_delay` time steps.

Note:
    With the default settings (``check_dist=True`` and
    ``rebuild_check_delay=1``), changing `NeighborList.buffer` only impacts
    simulation performance and not correctness.

    Set the buffer too small and the neighbor list will need to be updated
    often, slowing simulation performance. Set the buffer too large, and
    `hoomd.md.pair.Pair` will need to needlessly calculate many non-interacting
    particle pairs and slow the simulation. There is an optimal value for
    `NeighborList.buffer` between the two extremes that provides the best
    performance.

.. rubric:: Base distance cutoff

The `NeighborList.r_cut` attribute can be used to set the base cutoff distance
for neighbor list queries. The actual cutoff distance is always the maximum
:math:`r_{\mathrm{cut},i,j}` of the base cutoff and associated pair potentials.

Note:
    This attribute is particularly useful for implementing custom pair forces in
    Python.

Attention:
    Users should only set this attribute when utilizing the accessor APIs,
    `pair_list`, `local_pair_list`, `cpu_local_nlist_arrays`, or
    `gpu_local_nlist_arrays`.

.. rubric:: Exclusions

Neighbor lists nominally include all particles within the chosen cutoff
distances. The `NeighborList.exclusions` attribute defines which particles will
be excluded from the list, even if they are within the cutoff.
`NeighborList.exclusions` is a tuple of strings that enable one more more types
of exclusions. The valid exclusion types are:

* ``'angle'``: Exclude the first and third particles in each angle.
* ``'body'``: Exclude particles that belong to the same rigid body.
* ``'bond'``: Exclude particles that are directly bonded together.
* ``'meshbond'``: Exclude particles that are bonded together via a mesh.
* ``'constraint'``: Exclude particles that have a distance constraint applied
  between them.
* ``'dihedral'``: Exclude the first and fourth particles in each dihedral.
* ``'special_pair'``: Exclude particles that are part of a special pair.
* ``'1-3'``: Exclude particles *i* and *k* whenever there is a bond (i,j) and
  a bond (j,k).
* ``'1-4'``: Exclude particles *i* and *m* whenever there are bonds (i,j),
  (j,k), and (k,m).
"""

import hoomd
from hoomd.md import _md
import hoomd.device
from hoomd.data.parameterdicts import ParameterDict, TypeParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.data.typeconverter import OnlyFrom, OnlyTypes, nonnegative_real
from hoomd.logging import log
from hoomd.mesh import Mesh
from hoomd.operation import Compute


class NeighborList(Compute):
    r"""Base class neighbor list.

    `NeighborList` is the base class for all neighbor lists.

    Warning:
        Users should not instantiate this class directly. The class can be used
        for `isinstance` or `issubclass` checks.

    Attributes:
        buffer (float): Buffer width :math:`[\mathrm{length}]`.
        exclusions (tuple[str]): Defines which particles to exclude from the
            neighbor list, see more details above.
        rebuild_check_delay (int): How often to attempt to rebuild the neighbor
            list.
        check_dist (bool): Flag to enable / disable distance checking.
        mesh (Mesh): mesh data structure (optional)
        default_r_cut (float): Default cutoff distance :math:`[\mathrm{length}]`
            (optional).

    .. py:attribute:: r_cut

        Base cutoff radius for neighbor list queries.
        :math:`[\mathrm{length}]`. *Optional*: defaults to the value
        ``default_r_cut`` specified on construction.

        Type: `TypeParameter` [`tuple` [``particle_type``, ``particle_type``],
        `float`])
    """

    def __init__(self, buffer, exclusions, rebuild_check_delay, check_dist,
                 mesh, default_r_cut):

        validate_exclusions = OnlyFrom([
            'bond', 'angle', 'constraint', 'dihedral', 'special_pair', 'body',
            '1-3', '1-4', 'meshbond'
        ])

        validate_mesh = OnlyTypes(Mesh, allow_none=True)

        tp_r_cut = TypeParameter(
            'r_cut', 'particle_types',
            TypeParameterDict(nonnegative_real, len_keys=2))
        tp_r_cut.default = default_r_cut
        self._add_typeparam(tp_r_cut)

        # default exclusions
        params = ParameterDict(exclusions=[validate_exclusions],
                               buffer=float(buffer),
                               rebuild_check_delay=int(rebuild_check_delay),
                               check_dist=bool(check_dist))
        params["exclusions"] = exclusions
        self._param_dict.update(params)

        self._mesh = validate_mesh(mesh)

        self._in_context_manager = False

    def _attach_hook(self):
        if self._mesh is not None:
            self._cpp_obj.addMesh(self._mesh._cpp_obj)

    def _detach_hook(self):
        if self._mesh is not None:
            self._mesh._detach_hook()

    @property
    def cpu_local_nlist_arrays(self):
        """hoomd.md.data.NeighborListLocalAccess: Expose nlist arrays on the \
        CPU.

        Provides direct acces to the neighbor list arrays on the cpu. All data
        is MPI rank-local.

        The `hoomd.md.data.NeighborListLocalAccess` object exposes the internal
        data representation to efficently iterate over the neighbor list.

        Note:
            The local arrays are read only.

        Examples::

            with self.cpu_local_nlist_arrays as arrays:
                nlist_iter = zip(arrays.head_list, arrays.n_neigh)
                for i, (head, nn) in enumerate(nlist_iter):
                    for j_idx in range(head, head + nn):
                        j = arrays.nlist[j_idx]
                        # i and j are "neighbor" indices
        """
        if not self._attached:
            raise hoomd.error.DataAccessError("cpu_local_nlist_arrays")
        if self._in_context_manager:
            raise RuntimeError("Cannot enter cpu_local_nlist_arrays context "
                               "manager inside another local_nlist_arrays "
                               "context manager")
        self._cpp_obj.compute(self._simulation.timestep)
        return hoomd.md.data.NeighborListLocalAccess(self,
                                                     self._simulation.state)

    @property
    def gpu_local_nlist_arrays(self):
        """hoomd.md.data.NeighborListLocalAccessGPU: Expose nlist arrays on \
        the GPU.

        Provides direct access to the neighbor list arrays on the gpu. All data
        is MPI rank-local.

        The `hoomd.md.data.NeighborListLocalAccessGPU` object exposes the
        internal data representation to efficently iterate over the neighbor
        list.

        Note:
            The local arrays are read only.

        See Also:
            `cpu_local_nlist_arrays`

        Examples::

            get_local_pairs = cupy.RawKernel(r'''
            extern "C" __global__
            void get_local_pairs(
                    const unsigned int N,
                    const unsigned long* heads,
                    const unsigned int* nns,
                    const unsigned int* nlist,
                    const unsigned int* tags,
                    const unsigned long* offsets,
                    unsigned long* pairs) {
                unsigned int i = (unsigned int)
                    (blockDim.x * blockIdx.x + threadIdx.x);
                if (i >= N)
                    return;
                uint2* pair = (uint2*)pairs;
                unsigned long head = heads[i];
                unsigned int nn = nns[i];
                unsigned long offset = offsets[i];
                unsigned int tag_i = tags[i];
                for (unsigned int idx = 0; idx < nn; idx++) {
                    unsigned int j = nlist[head + idx];
                    pair[offset + idx] = make_uint2(tag_i, tags[j]);
                }
            }
            ''', 'get_local_pairs')

            with nlist.gpu_local_nlist_arrays as data:
                with sim.state.gpu_local_snapshot as snap_data:
                    tags = snap_data.particles.tag_with_ghost
                    tags = tags._coerce_to_ndarray()

                    head_list = data.head_list._coerce_to_ndarray()
                    n_neigh = data.n_neigh._coerce_to_ndarray()
                    raw_nlist = data.nlist._coerce_to_ndarray()

                    N = int(head_list.size)
                    n_pairs = int(cupy.sum(n_neigh))
                    offsets = cupy.cumsum(n_neigh.astype(cupy.uint64)
                    offsets -= n_neigh[0]
                    device_local_pairs = cupy.zeros(
                        (n_pairs, 2),
                        dtype=cupy.uint32)

                    block = 256
                    n_grid = (N + 255) // 256
                    get_local_pairs(
                        (n_grid,),
                        (block,),
                        (
                            N,
                            head_list,
                            n_neigh,
                            raw_nlist,
                            tags,
                            offsets,
                            device_local_pairs
                        ))

        Note:
            GPU local nlist data is not available if the chosen device for the
            simulation is `hoomd.device.CPU`.
        """
        if not isinstance(self._simulation.device, hoomd.device.GPU):
            raise RuntimeError(
                "Cannot access gpu_local_nlist_arrays without a GPU device")
        if not self._attached:
            raise hoomd.error.DataAccessError("gpu_local_nlist_arrays")
        if self._in_context_manager:
            raise RuntimeError(
                "Cannot enter gpu_local_nlist_arrays context manager inside "
                "another local_nlist_arrays context manager")
        self._cpp_obj.compute(self._simulation.timestep)
        return hoomd.md.data.NeighborListLocalAccessGPU(self,
                                                        self._simulation.state)

    @property
    def local_pair_list(self):
        """(*N_pairs*, 2) `numpy.ndarray` of `numpy.uint32`: Local pair list.

        Note:
            The local pair list returns rank-local *indices*, not particle tags.
        """
        if not self._attached:
            raise hoomd.error.DataAccessError("local_pair_list")
        return self._cpp_obj.getLocalPairList(self._simulation.timestep)

    @property
    def pair_list(self):
        """(*N_pairs*, 2) `numpy.ndarray` of `numpy.uint32`: Global pair list.

        Note:
            The pair list returns particle *tags*, not rank-local indices.

        Attention:
            In MPI parallel execution, the array is available on rank 0 only.
            `pair_list` is `None` on ranks >= 1.
        """
        if not self._attached:
            raise hoomd.error.DataAccessError("pair_list")
        return self._cpp_obj.getPairList(self._simulation.timestep)

    @log(requires_run=True)
    def shortest_rebuild(self):
        """int: The shortest period between neighbor list rebuilds.

        `shortest_rebuild` is the smallest number of time steps between neighbor
        list rebuilds since the last call to `Simulation.run`.
        """
        return self._cpp_obj.getSmallestRebuild()

    @log(requires_run=True, default=False)
    def num_builds(self):
        """int: The number of neighbor list builds.

        `num_builds` is the number of neighbor list rebuilds performed since the
        last call to `Simulation.run`.
        """
        return self._cpp_obj.num_builds


class Cell(NeighborList):
    r"""Neighbor list computed via a cell list.

    Args:
        buffer (float): Buffer width :math:`[\mathrm{length}]`.
        exclusions (tuple[str]): Defines which particles to exclude from the
            neighbor list, see more details in `NeighborList`.
        rebuild_check_delay (int): How often to attempt to rebuild the neighbor
            list.
        check_dist (bool): Flag to enable / disable distance checking.
        deterministic (bool): When `True`, sort neighbors to help provide
            deterministic simulation runs.
        mesh (Mesh): When a mesh object is passed, the neighbor list uses the
            mesh to determine the bond exclusions in addition to all other
            set exclusions.
        default_r_cut

    `Cell` finds neighboring particles using a fixed width cell list, allowing
    for *O(kN)* construction of the neighbor list where *k* is the number of
    particles per cell. Cells are sized to the largest :math:`r_\mathrm{cut}`.
    This method is very efficient for systems with nearly monodisperse cutoffs,
    but performance degrades for large cutoff radius asymmetries due to the
    significantly increased number of particles per cell. In practice, `Cell`
    is usually the best option for most users when the asymmetry between the
    largest and smallest cutoff radius is less than 2:1.

    .. image:: cell_list.png
        :width: 250 px
        :align: center
        :alt: Cell list schematic

    Note:
        `Cell` may consume a significant amount of memory, especially on GPU
        devices. One cause of this can be non-uniform density distributions
        because the memory allocated for the cell list is proportional the
        maximum number of particles in any cell. Another common cause is large
        box volumes combined with small cutoffs, which results in a very large
        number of cells in the system. In these cases, consider using `Stencil`
        or `Tree`, which can use less memory.

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
                 check_dist=True,
                 deterministic=False,
                 mesh=None,
                 default_r_cut=0.0):

        super().__init__(buffer, exclusions, rebuild_check_delay, check_dist,
                         mesh, default_r_cut)

        self._param_dict.update(
            ParameterDict(deterministic=bool(deterministic)))

    def _attach_hook(self):
        if isinstance(self._simulation.device, hoomd.device.CPU):
            nlist_cls = _md.NeighborListBinned
        else:
            nlist_cls = _md.NeighborListGPUBinned
        self._cpp_obj = nlist_cls(self._simulation.state._cpp_sys_def,
                                  self.buffer)
        super()._attach_hook()

    @log(requires_run=True, default=False, category='sequence')
    def dimensions(self):
        """tuple[int, int, int]: Cell list dimensions.

        `dimensions` is the number of cells in the x, y, and z directions.

        See Also:
            `allocated_particles_per_cell`
        """
        dimensions = self._cpp_obj.getDim()
        return (dimensions.x, dimensions.y, dimensions.z)

    @log(requires_run=True, default=False)
    def allocated_particles_per_cell(self):
        """int: Number of particle slots allocated per cell.

        The total memory usage of `Cell` is proportional to the product of the
        three cell list `dimensions` and the `allocated_particles_per_cell`.
        """
        return self._cpp_obj.getNmax()


class Stencil(NeighborList):
    """Cell list based neighbor list using stencils.

    Args:
        cell_width (float): The underlying stencil bin width for the cell list
            :math:`[\\mathrm{length}]`.
        buffer (float): Buffer width :math:`[\\mathrm{length}]`.
        exclusions (tuple[str]): Defines which particles to exclude from the
            neighbor list, see more details in `NeighborList`.
        rebuild_check_delay (int): How often to attempt to rebuild the neighbor
            list.
        check_dist (bool): Flag to enable / disable distance checking.
        deterministic (bool): When `True`, sort neighbors to help provide
            deterministic simulation runs.
        mesh (Mesh): When a mesh object is passed, the neighbor list uses the
            mesh to determine the bond exclusions in addition to all other
            set exclusions.

    `Stencil` finds neighboring particles using a fixed width cell list, for
    *O(kN)* construction of the neighbor list where *k* is the number of
    particles per cell. In contrast with `Cell`, `Stencil` allows the user to
    choose the cell width: `cell_width` instead of fixing it to the largest
    cutoff radius (`P.J. in't Veld et al. 2008
    <https://dx.doi.org/10.1016/j.cpc.2008.03.005>`_):

    .. image:: stencil_schematic.png
        :width: 300 px
        :align: center
        :alt: Stenciled cell list schematic

    This neighbor list style differs from `Cell` in how the adjacent cells are
    searched for particles. One stencil is computed per particle type based on
    the value of `cell_width` set by the user, which defines the bins that the
    particle must search in. Distances to the bins in the stencil are
    precomputed so that certain particles can be quickly excluded from the
    neighbor list, leading to improved performance compared to `Cell` when there
    is size disparity in the cutoff radius. The memory demands of `Stencil` can
    also be lower than `Cell` if your system is large and has many small cells
    in it; however, `Tree` is usually a better choice for these systems.

    The performance of `Stencil` depends strongly on the choice of *cell_width*.
    The best performance is obtained when the cutoff radii are multiples of the
    *cell_width*, and when the *cell_width* covers the simulation box with a
    roughly integer number of cells.

    Examples::

        nl_s = nlist.Stencil(cell_width=1.5)

    Important:
        `M.P. Howard et al. 2016 <https://dx.doi.org/10.1016/j.cpc.2016.02.003>`_
        describes this neighbor list implementation. Cite it if you utilize
        `Stencil` in your research.

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
                 check_dist=True,
                 deterministic=False,
                 mesh=None,
                 default_r_cut=0.0):

        super().__init__(buffer, exclusions, rebuild_check_delay, check_dist,
                         mesh, default_r_cut)

        params = ParameterDict(deterministic=bool(deterministic),
                               cell_width=float(cell_width))

        self._param_dict.update(params)

    def _attach_hook(self):
        if isinstance(self._simulation.device, hoomd.device.CPU):
            nlist_cls = _md.NeighborListStencil
        else:
            nlist_cls = _md.NeighborListGPUStencil
        self._cpp_obj = nlist_cls(self._simulation.state._cpp_sys_def,
                                  self.buffer)
        super()._attach_hook()


class Tree(NeighborList):
    """Bounding volume hierarchy based neighbor list.

    Args:
        buffer (float): Buffer width :math:`[\\mathrm{length}]`.
        exclusions (tuple[str]): Defines which particles to exclude from the
            neighbor list, see more details in `NeighborList`.
        rebuild_check_delay (int): How often to attempt to rebuild the neighbor
            list.
        check_dist (bool): Flag to enable / disable distance checking.
        mesh (Mesh): When a mesh object is passed, the neighbor list uses the
            mesh to determine the bond exclusions in addition to all other
            set exclusions.

    `Tree` creates a neighbor list using a bounding volume hierarchy (BVH) tree
    traversal in :math:`O(N \\log N)` time. A BVH tree of axis-aligned bounding
    boxes is constructed per particle type, and each particle queries each tree
    to determine its neighbors. This method of searching leads to significantly
    improved performance compared to cell lists in systems with moderate size
    asymmetry, but has slower performance for monodisperse systems. `Tree` can
    also be slower than `Cell` if there are multiple types in the system, but
    the cutoffs between types are identical. (This is because one BVH is created
    per type.) The user should carefully benchmark neighbor list build times to
    select the appropriate neighbor list construction type.

    .. image:: tree_schematic.png
        :width: 400 px
        :align: center
        :alt: BVH tree schematic

    `Tree`'s memory requirements scale with the number of particles in the
    system rather than the box volume, which may be particularly advantageous
    for large, sparse systems.

    Important:
        `M.P. Howard et al. 2016 <https://dx.doi.org/10.1016/j.cpc.2016.02.003>`_
        describes the original implementation of this algorithm for HOOMD-blue.
        `M.P. Howard et al. 2019
        <https://doi.org/10.1016/j.commatsci.2019.04.004>`_ describes the
        improved algorithm that is currently implemented. Cite both if you
        utilize this neighbor list style in your work.

    Examples::

        nl_t = nlist.Tree(check_dist=False)
    """

    def __init__(self,
                 buffer,
                 exclusions=('bond',),
                 rebuild_check_delay=1,
                 check_dist=True,
                 mesh=None,
                 default_r_cut=0.0):

        super().__init__(buffer, exclusions, rebuild_check_delay, check_dist,
                         mesh, default_r_cut)

    def _attach_hook(self):
        if isinstance(self._simulation.device, hoomd.device.CPU):
            nlist_cls = _md.NeighborListTree
        else:
            nlist_cls = _md.NeighborListGPUTree
        self._cpp_obj = nlist_cls(self._simulation.state._cpp_sys_def,
                                  self.buffer)
        super()._attach_hook()
