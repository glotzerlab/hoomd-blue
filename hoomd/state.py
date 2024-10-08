# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Module implements the `State` class.

`State` stores and exposes a parent `hoomd.Simulation` object's data (e.g.
particle positions, system bonds).

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()
"""
import weakref
from collections import defaultdict

from . import _hoomd
from hoomd.box import Box
from hoomd.snapshot import Snapshot
from hoomd.data import LocalSnapshot, LocalSnapshotGPU
import hoomd
import math
import collections.abc


def _create_domain_decomposition(device, box, domain_decomposition):
    """Create the domain decomposition.

    Args:
        device (Device): The simulation's device
        box: The C++ global box object for the state being initialized
        domain_decomposition: See Simulation.create_state_from_* for a
          description.
    """
    if (not isinstance(domain_decomposition, collections.abc.Sequence)
            or len(domain_decomposition) != 3):
        raise TypeError("domain_decomposition must be a length 3 sequence")

    initialize_grid = False
    initialize_fractions = False

    for v in domain_decomposition:
        if v is not None:
            if isinstance(v, int):
                initialize_grid = True
            elif isinstance(v, collections.abc.Sequence):
                if not math.isclose(sum(v), 1.0, rel_tol=1e-6):
                    raise ValueError("Rank fractions must sum to 1.0.")
                initialize_fractions = True
            else:
                raise TypeError("Invalid type in domain_decomposition.")

    if initialize_grid and initialize_fractions:
        raise ValueError("Domain decomposition mixes integers and sequences.")

    if not hoomd.version.mpi_enabled:
        return None

    # if we are only running on one processor, we use optimized code paths
    # for single-GPU execution
    if device.communicator.num_ranks == 1:
        return None

    if initialize_fractions:
        fractions = [
            v[:-1] if v is not None else [] for v in domain_decomposition
        ]
        result = _hoomd.DomainDecomposition(device._cpp_exec_conf, box.getL(),
                                            *fractions)
    else:
        grid = [v if v is not None else 0 for v in domain_decomposition]
        result = _hoomd.DomainDecomposition(device._cpp_exec_conf, box.getL(),
                                            *grid, False)

    return result


class State:
    """The state of a `Simulation` object.

    Note:
        This object cannot be directly instantiated. Use
        `Simulation.create_state_from_gsd` and
        `Simulation.create_state_from_snapshot` to instantiate a `State`
        object as part of a simulation.

    .. rubric:: Overview

    `State` stores the data that describes the thermodynamic microstate of a
    `Simulation` object. This data consists of the box, particles, bonds,
    angles, dihedrals, impropers, special pairs, and constraints.

    .. rubric:: Box

    The simulation `box` describes the space that contains the particles as a
    `Box` object.

    .. rubric:: Particles

    The state contains `N_particles` particles. Each particle has a position,
    orientation, type id, body, mass, moment of inertia, charge, diameter,
    velocity, angular momentum, image, and tag:

    - :math:`\\vec{r}`: position :math:`[\\mathrm{length}]` - X,Y,Z cartesian
      coordinates defining the position of the particle in the box.

    - :math:`\\mathbf{q}`: orientation :math:`[\\mathrm{dimensionless}]` -
      Unit quaternion defining the rotation from the particle's local reference
      frame to the box reference frame. The four components are in the order
      :math:`s`, :math:`a_x`, :math:`a_y`, :math:`a_z` for the in complex
      notation :math:`s + a_x i + a_y j + a_z k`.

    - ``particle_typeid``: type id :math:`[\\mathrm{dimensionless}]` -
      An integer in the interval ``[0,len(particle_types)``) that identifies the
      particle's type. `particle_types` maps type ids to names with:
      ``name = particle_types[particle_typeid]``.

    - ``particle_body``: body id :math:`[\\mathrm{dimensionless}]` - An integer
      that identifies the particle's rigid body. A value of ``-1`` indicates
      that this particle does not belong to a body. A positive value indicates
      that the particle belongs to the body ``particle_body``. This particle is
      the central particle of a body when the body id is equal to the tag
      :math:`\\mathrm{particle\\_body} = \\mathrm{particle\\_tag}`. (used by
      `hoomd.md.constrain.Rigid`)

    - :math:`m`: mass :math:`[\\mathrm{mass}]` - The particle's mass.

    - :math:`I`: moment of inertia :math:`[\\mathrm{mass} \\cdot
      \\mathrm{length}^2]` - :math:`I_{xx}`, :math:`I_{yy}`, :math:`I_{zz}`
      elements of the diagonal moment of inertia tensor in the particle's local
      reference frame. The off-diagonal elements are 0.

    - :math:`q`: charge :math:`[\\mathrm{charge}]` - The particle's charge.

    - :math:`d`: diameter :math:`[\\mathrm{length}]` - Deprecated in v3.0.0.
      HOOMD-blue reads and writes particle diameters, but does not use them in
      any computations.

    - :math:`\\vec{v}`: velocity :math:`[\\mathrm{velocity}]` - X,Y,Z
      components of the particle's velocity in the box's reference frame.

    - :math:`\\mathbf{P_S}`: angular momentum :math:`[\\mathrm{mass} \\cdot
      \\mathrm{velocity} \\cdot \\mathrm{length}]` - Quaternion defining the
      particle's angular momentum (see note).

    - :math:`\\vec{n}` : image :math:`[\\mathrm{dimensionless}]` - Integers
      x,y,z that record how many times the particle has crossed each of the
      periodic box boundaries.

    - ``particle_tag`` : tag :math:`[\\mathrm{dimensionless}]` -
      An integer that uniquely identifies a given particle. The particles are
      stored in tag order when writing and initializing to/from a GSD file or
      snapshot: :math:`\\mathrm{particle\\_tag}_i = i`. When accessing data in
      local snapshots, particles may be in any order.

    Note:
        HOOMD stores angular momentum as a quaternion because that is the form
        used when integrating the equations of motion (see `Kamberaj 2005`_).
        The angular momentum quaternion :math:`\\mathbf{P_S}` is defined with
        respect to the orientation quaternion of the particle
        :math:`\\mathbf{q}` and the vector angular momentum of the particle,
        lifted into pure imaginary quaternion form :math:`\\mathbf{S}^{(4)}` as:

        .. math::

            \\mathbf{P_S} = 2 \\mathbf{q} \\times \\mathbf{S}^{(4)}

        . Following this, the angular momentum vector :math:`\\vec{S}` in the
        particle's local reference frame is:

        .. math::

            \\vec{S} = \\frac{1}{2}im(\\mathbf{q}^* \\times \\mathbf{P_S})

    .. rubric:: Bonded groups

    The state contains `N_bonds` bonds, `N_angles` angles, `N_dihedrals`
    dihedrals, `N_impropers` impropers, and `N_special_pairs` special pairs.
    Each of these data structures is similar, differing in the number of
    particles in the group and what operations use them. Bonds, angles,
    dihedrals, and impropers contain 2, 3, 4, and 4 particles
    per group respectively. Bonds specify the toplogy used when computing
    energies and forces in `md.bond`, angles define the same for `md.angle`,
    dihedrals for `md.dihedral` and impropers for `md.improper`. These
    collectively implement bonding potentials used in molecular dynamics
    force fields. Like bonds, special pairs define connections between two
    particles, but special pairs are intended to adjust the 1-4 pairwise
    interactions in some molecular dynamics force fields: see
    `md.special_pair`. Each bonded group is defined by a type id, the
    group members, and a tag.

    - ``bond_typeid``: type id :math:`[\\mathrm{dimensionless}]` -
      An integer in the interval ``[0,len(bond_types))`` that identifies the
      bond's type. `bond_types` maps type ids to names with:
      ``name = bond_types[bond_typeid]``. Similarly, `angle_types` lists the
      angle types, `dihedral_types` lists the dihedral types, `improper_types`
      lists the improper types, and `special_pair_types` lists the special pair
      types.
    - ``bond_group``: A list of integers in the interval
      :math:`[0, \\max(\\mathrm{particle\\_tag})]` that defines the tags of the
      particles in the bond (2), angle in ``angle_group`` (3), dihedral
      in ``dihedral_group`` (4), improper in ``improper_group`` (4), or
      special pair in ``pair_group`` (2).
    - ``bond_tag`` : tag :math:`[\\mathrm{dimensionless}]` -
      An integer that uniquely identifies a given bond. The bonds are in tag
      order when writing and initializing to/from a GSD file or snapshot
      :math:`\\mathrm{bond\\_tag}_i = i`. When accessing data in local
      snapshots, bonds may be in any order. The same applies to angles with
      ``angle_tag``, dihedrals with ``dihedral_tag``, impropers with
      ``improper_tag``, and special pairs with ``pair_tag``.

    .. rubric:: Constraints

    The state contains `N_constraints` distance constraints between particles.
    These constraints are used by `hoomd.md.constrain.Distance`. Each distance
    constraint consists of a distance value and the group members.

    - ``constraint_group``: A list of 2 integers in the interval
      :math:`[0, \\max(\\mathrm{particle\\_tag})]` that identifies the tags of
      the particles in the constraint.
    - :math:`d`: constraint value :math:`[\\mathrm{length}]` - The distance
      between particles in the constraint.

    .. rubric:: MPI domain decomposition

    When running in serial or on 1 MPI rank, the entire simulation state is
    stored in that process. When using more than 1 MPI rank, HOOMD-blue employs
    a domain decomposition approach to split the simulation box an integer
    number of times in the x, y, and z directions (`domain_decomposition`). Each
    MPI rank stores and operates on the particles local to that rank. Local
    particles are those contained within the region defined by the split planes
    (`domain_decomposition_split_fractions`). Each MPI rank communicates with
    its neighbors to obtain the properties of particles near the boundary
    between ranks (ghost particles) so that it can compute interactions across
    the boundary.

    .. rubric:: Accessing Data

    Two complementary APIs provide access to the state data: *local* snapshots
    that access data directly available on the local MPI rank (including the
    local and ghost particles) and *global* snapshots that collect the entire
    state on rank 0. See `State.cpu_local_snapshot`, `State.gpu_local_snapshot`,
    `get_snapshot`, and `set_snapshot` for information about these data access
    patterns.

    See Also:
        To write the simulation to disk, use `write.GSD`.

    .. _Kamberaj 2005: https://dx.doi.org/10.1063/1.1906216
    """

    def __init__(self, simulation, snapshot, domain_decomposition):
        self._simulation = simulation
        snapshot._broadcast_box()
        decomposition = _create_domain_decomposition(
            simulation.device, snapshot._cpp_obj._global_box,
            domain_decomposition)

        if decomposition is not None:
            self._cpp_sys_def = _hoomd.SystemDefinition(
                snapshot._cpp_obj, simulation.device._cpp_exec_conf,
                decomposition)
        else:
            self._cpp_sys_def = _hoomd.SystemDefinition(
                snapshot._cpp_obj, simulation.device._cpp_exec_conf)

        # Necessary for local snapshot API. This is used to ensure two local
        # snapshots are not contexted at once.
        self._in_context_manager = False

        # self._groups provides a cache of C++ group objects of the form:
        # {type(filter): {filter: C++ group}}
        # The first layer is to prevent user created filters with poorly
        # implemented __hash__ and __eq__ from causing cache errors.
        self._groups = defaultdict(dict)

    def get_snapshot(self):
        """Make a copy of the simulation current state.

        `State.get_snapshot` makes a copy of the simulation state and
        makes it available in a single object. `set_snapshot` resets
        the internal state to that in the given snapshot. Use these methods
        to implement techniques like hybrid MD/MC or umbrella sampling where
        entire system configurations need to be reset to a previous one after a
        rejected move.

        Note:
            Data across all MPI ranks and from GPUs is gathered on the root MPI
            rank's memory. When accessing data in MPI simulations, use a ``if
            snapshot.communicator.rank == 0:`` conditional to access data arrays
            only on the root rank.

        Note:
            `State.get_snapshot` is an order :math:`O(N_{particles} + N_{bonds}
            + \\ldots)` operation.

        See Also:
            `set_snapshot`

        Returns:
            Snapshot: The current simulation state

        .. rubric:: Example:

        .. code-block:: python

            snapshot = simulation.state.get_snapshot()
        """
        cpp_snapshot = self._cpp_sys_def.takeSnapshot_double()
        return Snapshot._from_cpp_snapshot(cpp_snapshot,
                                           self._simulation.device.communicator)

    def set_snapshot(self, snapshot):
        """Restore the state of the simulation from a snapshot.

        Also calls `update_group_dof` to count the number of degrees in the
        system with the new state.

        Args:
            snapshot (Snapshot): Snapshot of the system from
              `get_snapshot`

        Warning:
            `set_snapshot` can only make limited changes to the simulation
            state. While it can change the number of particles/bonds/etc... or
            their properties, it cannot change the number or names of the
            particle/bond/etc.. types.

        Note:
            `set_snapshot` is an order :math:`O(N_{particles} + N_{bonds} +
            \\ldots)` operation and is very expensive when the simulation device
            is a GPU.

        See Also:
            `get_snapshot`

            `Simulation.create_state_from_snapshot`

        .. rubric:: Example:

        .. code-block:: python

            simulation.state.set_snapshot(snapshot)
        """
        if self._in_context_manager:
            raise RuntimeError(
                "Cannot set state to new snapshot inside local snapshot.")
        if self._simulation.device.communicator.rank == 0:
            if snapshot.particles.types != self.particle_types:
                raise RuntimeError("Particle types must remain the same")
            if snapshot.bonds.types != self.bond_types:
                raise RuntimeError("Bond types must remain the same")
            if snapshot.angles.types != self.angle_types:
                raise RuntimeError("Angle types must remain the same")
            if snapshot.dihedrals.types != self.dihedral_types:
                raise RuntimeError("Dihedral types must remain the same")
            if snapshot.impropers.types != self.improper_types:
                raise RuntimeError("Improper types must remain the same")
            if snapshot.pairs.types != self.special_pair_types:
                raise RuntimeError("Pair types must remain the same")

        self._cpp_sys_def.initializeFromSnapshot(snapshot._cpp_obj)
        self.update_group_dof()

    @property
    def particle_types(self):
        """list[str]: List of all particle types in the simulation state.

        .. rubric:: Example:

        .. code-block:: python

            particle_types = simulation.state.particle_types
        """
        return self._cpp_sys_def.getParticleData().getTypes()

    @property
    def bond_types(self):
        """list[str]: List of all bond types in the simulation state.

        .. rubric:: Example:

        .. code-block:: python

            bond_types = simulation.state.bond_types
        """
        return self._cpp_sys_def.getBondData().getTypes()

    @property
    def angle_types(self):
        """list[str]: List of all angle types in the simulation state.

        .. rubric:: Example:

        .. code-block:: python

            angle_types = simulation.state.angle_types
        """
        return self._cpp_sys_def.getAngleData().getTypes()

    @property
    def dihedral_types(self):
        """list[str]: List of all dihedral types in the simulation state.

        .. rubric:: Example:

        .. code-block:: python

            angle_types = simulation.state.angle_types
        """
        return self._cpp_sys_def.getDihedralData().getTypes()

    @property
    def improper_types(self):
        """list[str]: List of all improper types in the simulation state.

        .. rubric:: Example:

        .. code-block:: python

            improper_types = simulation.state.improper_types
        """
        return self._cpp_sys_def.getImproperData().getTypes()

    @property
    def special_pair_types(self):
        """list[str]: List of all special pair types in the simulation state.

        .. rubric:: Example:

        .. code-block:: python

            special_pair_types = simulation.state.special_pair_types
        """
        return self._cpp_sys_def.getPairData().getTypes()

    @property
    def types(self):
        """dict[str, list[str]]: dictionary of all types in the state.

        Combines the data from `particle_types`, `bond_types`, `angle_types`,
        `dihedral_types`, `improper_types`, and `special_pair_types` into a
        dictionary with keys matching the property names.
        """
        return dict(particle_types=self.particle_types,
                    bond_types=self.bond_types,
                    angle_types=self.angle_types,
                    dihedral_types=self.dihedral_types,
                    improper_types=self.improper_types,
                    special_pair_types=self.special_pair_types)

    @property
    def N_particles(self):  # noqa: N802 - allow N in name
        """int: The number of particles in the simulation state.

        .. rubric:: Example:

        .. code-block:: python

            N_particles = simulation.state.N_particles
        """
        return self._cpp_sys_def.getParticleData().getNGlobal()

    @property
    def N_bonds(self):  # noqa: N802 - allow N in name
        """int: The number of bonds in the simulation state.

        .. rubric:: Example:

        .. code-block:: python

            N_bonds = simulation.state.N_bonds
        """
        return self._cpp_sys_def.getBondData().getNGlobal()

    @property
    def N_angles(self):  # noqa: N802 - allow N in name
        """int: The number of angles in the simulation state.

        .. rubric:: Example:

        .. code-block:: python

            N_angles = simulation.state.N_angles
        """
        return self._cpp_sys_def.getAngleData().getNGlobal()

    @property
    def N_impropers(self):  # noqa: N802 - allow N in name
        """int: The number of impropers in the simulation state.

        .. rubric:: Example:

        .. code-block:: python

            N_impropers = simulation.state.N_impropers
        """
        return self._cpp_sys_def.getImproperData().getNGlobal()

    @property
    def N_special_pairs(self):  # noqa: N802 - allow N in name
        """int: The number of special pairs in the simulation state."""
        return self._cpp_sys_def.getPairData().getNGlobal()

    @property
    def N_dihedrals(self):  # noqa: N802 - allow N in name
        """int: The number of dihedrals in the simulation state.

        .. rubric:: Example:

        .. code-block:: python

            N_dihedrals = simulation.state.N_dihedrals
        """
        return self._cpp_sys_def.getDihedralData().getNGlobal()

    @property
    def N_constraints(self):  # noqa: N802 - allow N in name
        """int: The number of constraints in the simulation state.

        .. rubric:: Example:

        .. code-block:: python

            N_constraints = simulation.state.N_constraints
        """
        return self._cpp_sys_def.getConstraintData().getNGlobal()

    @property
    def box(self):
        """hoomd.Box: A copy of the current simulation box.

        Note:
            The `box` property cannot be set. Call `set_box` to set a new
            simulation box.

        .. rubric:: Example:

        .. code-block:: python

            box = simulation.state.box
        """
        b = Box._from_cpp(self._cpp_sys_def.getParticleData().getGlobalBox())
        return Box.from_box(b)

    def set_box(self, box):
        """Set a new simulation box.

        Args:
            box (hoomd.box.box_like): New simulation box.

        Note:
            All particles must be inside the new box. `set_box` does not change
            any particle properties.

        See Also:
            `hoomd.update.BoxResize.update`

        .. rubric:: Example:

        .. code-block:: python

            simulation.state.set_box(box)
        """
        if self._in_context_manager:
            raise RuntimeError(
                "Cannot set system box within local snapshot context manager.")
        try:
            box = Box.from_box(box)
        except Exception:
            raise ValueError('{} is not convertable to hoomd.Box using '
                             'hoomd.Box.from_box'.format(box))

        if box.dimensions != self._cpp_sys_def.getNDimensions():
            self._simulation.device._cpp_msg.warning(
                "Box changing dimensions from {} to {}."
                "".format(self._cpp_sys_def.getNDimensions(), box.dimensions))
            self._cpp_sys_def.setNDimensions(box.dimensions)
        self._cpp_sys_def.getParticleData().setGlobalBox(box._cpp_obj)

    def replicate(self, nx, ny, nz=1):
        """Replicate the state of the system along the periodic box directions.

        Args:
            nx (int): Number of times to replicate in the x direction.
            ny (int): Number of times to replicate in the y direction.
            nz (int): Number of times to replicate in the z direction.

        `replicate` makes the system state ``nx * ny * nz`` times larger. In
        each of the new periodic box images, it places a copy of the initial
        state with the particle positions offset to locate them in the image and
        the bond, angle, dihedral, improper, and pair group tags
        offset to apply to the copied particles. All other particle properties
        (mass, typeid, velocity, charge, ...) are copied to the new particles
        without change.

        After placing the particles, `replicate` expands the simulation box by a
        factor of ``nx``, ``ny``, and ``nz`` in the direction of the first,
        second, and third box lattice vectors respectively and adjusts the
        particle positions to center them in the new box.

        .. rubric:: Example:

        .. code-block:: python

            simulation.state.replicate(nx=2, ny=2, nz=2)
        """
        snap = self.get_snapshot()
        snap.replicate(nx, ny, nz)
        self.set_snapshot(snap)

    def _get_group(self, filter_):
        cls = filter_.__class__
        group_cache = self._groups
        if filter_ in group_cache[cls]:
            return group_cache[cls][filter_]
        else:
            if isinstance(filter_, hoomd.filter.CustomFilter):
                group = _hoomd.ParticleGroup(
                    self._cpp_sys_def,
                    _hoomd.ParticleFilterCustom(filter_, self))
            else:
                group = _hoomd.ParticleGroup(self._cpp_sys_def, filter_)
            group_cache[cls][filter_] = group
            self._simulation._cpp_sys.group_cache.append(group)

            integrator = self._simulation.operations.integrator
            if integrator is not None and integrator._attached:
                integrator._cpp_obj.updateGroupDOF(group)

            return group

    def update_group_dof(self):
        """Schedule an update to the number of degrees of freedom in each group.

        `update_group_dof` requests that `Simulation` update the degrees of
        freedom provided to each group by the Integrator. `Simulation` will
        perform this update at the start of `Simulation.run` or at the start
        of the next timestep during an ongoing call to `Simulation.run`.

        This method is called automatically when:

        * An Integrator is assigned to the `Simulation`'s operations.
        * The `hoomd.md.Integrator.integrate_rotational_dof` parameter is set.
        * `set_snapshot` is called.
        * On timesteps where a `hoomd.update.FilterUpdater` triggers.

        Call `update_group_dof` manually to force an update, such as when
        you modify particle moments of inertia with `cpu_local_snapshot`.

        .. rubric:: Example:

        .. code-block:: python

            box = simulation.state.update_group_dof()
        """
        self._simulation._cpp_sys.updateGroupDOFOnNextStep()

    @property
    def cpu_local_snapshot(self):
        """hoomd.data.LocalSnapshot: Expose simulation data on the CPU.

        Provides access directly to the system state's particle, bond, angle,
        dihedral, improper, constaint, and pair data through a context manager.
        Data in `State.cpu_local_snapshot` is MPI rank local, and the
        `hoomd.data.LocalSnapshot` object is only usable within a context
        manager (i.e. ``with sim.state.cpu_local_snapshot as data:``). Attempts
        to assess data outside the context manager will result in errors. The
        local snapshot interface is similar to that of `Snapshot`.

        The `hoomd.data.LocalSnapshot` data access is mediated through
        `hoomd.data.array.HOOMDArray` objects. This lets us ensure memory safety
        when directly accessing HOOMD-blue's data. The interface provides
        zero-copy access (zero-copy is guaranteed on CPU, access may be
        zero-copy if running on GPU).

        Changing the data in the buffers exposed by the local snapshot will
        change the data across the HOOMD-blue simulation. For a trivial example,
        this example would set all particle z-axis positions to 0.

        .. code-block:: python

            with simulation.state.cpu_local_snapshot as local_snapshot:
                local_snapshot.particles.position[:, 2] = 0

        Note:
            The state's box and the number of particles, bonds, angles,
            dihedrals, impropers, constaints, and pairs cannot
            change within the context manager.

        Note:
            Getting a local snapshot object is order :math:`O(1)` and setting a
            single value is of order :math:`O(1)`.
        """
        if self._in_context_manager:
            raise RuntimeError(
                "Cannot enter cpu_local_snapshot context manager inside "
                "another local_snapshot context manager.")
        return LocalSnapshot(self)

    @property
    def gpu_local_snapshot(self):
        """hoomd.data.LocalSnapshotGPU: Expose simulation data on the GPU.

        Provides access directly to the system state's particle, bond, angle,
        dihedral, improper, constaint, and pair data through a context manager.
        Data in `State.gpu_local_snapshot` is GPU local, and the
        `hoomd.data.LocalSnapshotGPU` object is only usable within a context
        manager (i.e. ``with sim.state.gpu_local_snapshot as data:``). Attempts
        to assess data outside the context manager will result in errors. The
        local snapshot interface is similar to that of `Snapshot`.

        The `hoomd.data.LocalSnapshotGPU` data access is mediated through
        `hoomd.data.array.HOOMDGPUArray` objects. This helps us maintain memory
        safety when directly accessing HOOMD-blue's data. The interface provides
        zero-copy access on the GPU (assuming data was last accessed on the
        GPU).

        Changing the data in the buffers exposed by the local snapshot will
        change the data across the HOOMD-blue simulation. For a trivial example,
        this example would set all particle z-axis positions to 0.

        .. invisible-code-block: python

            if not (gpu_not_available or cupy_not_available):
                gpu=hoomd.device.GPU()
                simulation = hoomd.util.make_example_simulation(device=gpu)
                simulation.state.replicate(nx=2, ny=2, nz=2)

        .. skip: next if(gpu_not_available or cupy_not_available)

        .. code-block:: python

            with simulation.state.gpu_local_snapshot as local_snapshot:
                local_snapshot.particles.position[:, 2] = 0

        Warning:
            This property is only available when running on a GPU (or multiple
            GPUs).

        Note:
            The state's box and the number of particles, bonds, angles,
            dihedrals, impropers, constaints, and pairs cannot
            change within the context manager.

        Note:
            Getting a local snapshot object is order :math:`O(1)` and setting a
            single value is of order :math:`O(1)`.
        """
        if not isinstance(self._simulation.device, hoomd.device.GPU):
            raise RuntimeError(
                "Cannot access gpu_snapshot with a non GPU device.")
        elif self._in_context_manager:
            raise RuntimeError(
                "Cannot enter gpu_local_snapshot context manager inside "
                "another local_snapshot context manager.")
        else:
            return LocalSnapshotGPU(self)

    def thermalize_particle_momenta(self, filter, kT):
        """Assign random values to particle momenta.

        Args:
            filter (hoomd.filter.filter_like): Particles to modify
            kT (float): Thermal energy to set :math:`[\\mathrm{energy}]`

        `thermalize_particle_momenta` assigns the selected particle's velocities
        and angular momentum to random values drawn from a Gaussian distribution
        consistent with the given thermal energy *kT*.

        .. rubric:: Velocity

        `thermalize_particle_momenta` assigns random velocities to the *x* and
        *y* components of each particle's velocity. When the simulation box is
        3D, it also assigns a random velocity to the *z* component. When the
        simulation box is 2D, it sets the *z* component to 0. Finally,
        sets the center of mass velocity of the selected particles to 0.

        .. rubric:: Angular momentum

        `thermalize_particle_momenta` assigns random angular momenta to each
        rotational degree of freedom that has a non-zero moment of intertia.
        Each particle can have 0, 1, 2, or 3 rotational degrees of freedom
        as determine by its moment of inertia.

        .. seealso::
            `md.methods.thermostats.MTTK.thermalize_dof`

            `md.methods.ConstantPressure.thermalize_barostat_dof`

        .. rubric:: Example:

        .. code-block:: python

            simulation.state.thermalize_particle_momenta(
                filter=hoomd.filter.All(),
                kT=1.5)
        """
        self._simulation._warn_if_seed_unset()
        group = self._get_group(filter)
        group.thermalizeParticleMomenta(kT, self._simulation.timestep)

    @property
    def domain_decomposition_split_fractions(self):
        """tuple(list[float], list[float], list[float]): Box fractions of the \
        domain split planes in the x, y, and z directions."""
        particle_data = self._cpp_sys_def.getParticleData()

        if (not hoomd.version.mpi_enabled
                or particle_data.getDomainDecomposition() is None):
            return ([], [], [])

        return tuple([
            list(particle_data.getDomainDecomposition().getCumulativeFractions(
                dir))[1:-1] for dir in range(3)
        ])

    @property
    def domain_decomposition(self):
        """tuple(int, int, int): Number of domains in the x, y, and z \
        directions."""
        particle_data = self._cpp_sys_def.getParticleData()

        if (not hoomd.version.mpi_enabled
                or particle_data.getDomainDecomposition() is None):
            return (1, 1, 1)

        return tuple([
            len(particle_data.getDomainDecomposition().getCumulativeFractions(
                dir)) - 1 for dir in range(3)
        ])

    @property
    def _simulation(self):
        sim = self._simulation_
        if sim is not None:
            sim = sim()
            if sim is not None:
                return sim

    @_simulation.setter
    def _simulation(self, sim):
        if sim is not None:
            sim = weakref.ref(sim)
        self._simulation_ = sim
