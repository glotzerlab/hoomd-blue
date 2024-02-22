# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Newtonain Event-Chain Integrators for Hard Particle Monte Carlo."""

from hoomd.hpmc.integrate import HPMCIntegrator
from hoomd.data.parameterdicts import TypeParameterDict, ParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.data.typeconverter import OnlyTypes

from hoomd.logging import log


class HPMCNECIntegrator(HPMCIntegrator):
    """HPMC Chain Integrator base class.

    `HPMCNECIntegrator` is the base class for all HPMC Newtonian event chain
    integrators. The attributes documented here are available to all HPMC
    integrators.

    Warning:
        This class should not be instantiated by users. The class can be used
        for `isinstance` or `issubclass` checks.
    """
    _cpp_cls = None

    def __init__(self,
                 default_d=0.1,
                 default_a=0.1,
                 chain_probability=0.5,
                 chain_time=0.5,
                 update_fraction=0.5,
                 nselect=1):
        # initialize base class
        super().__init__(default_d, default_a, 0.5, nselect)

        # Set base parameter dict for hpmc chain integrators
        param_dict = ParameterDict(
            chain_probability=OnlyTypes(
                float, postprocess=self._process_chain_probability),
            chain_time=OnlyTypes(float, postprocess=self._process_chain_time),
            update_fraction=OnlyTypes(
                float, postprocess=self._process_update_fraction))
        self._param_dict.update(param_dict)
        self.chain_probability = chain_probability
        self.chain_time = chain_time
        self.update_fraction = update_fraction

    @staticmethod
    def _process_chain_probability(value):
        if 0.0 < value <= 1.0:
            return value
        else:
            raise ValueError(
                "chain_probability has to be between 0 and 1 (got {}).".format(
                    value))

    @staticmethod
    def _process_chain_time(value):
        if 0.0 <= value:
            return value
        else:
            raise ValueError(
                "chain_time has to be positive (got {}).".format(value))

    @staticmethod
    def _process_update_fraction(value):
        if 0.0 < value <= 1.0:
            return value
        else:
            raise ValueError(
                "update_fraction has to be between 0 and 1. (got {})".format(
                    value))

    @property
    def nec_counters(self):
        """Trial move counters.

        The counter object has the following attributes:

        * ``chain_start_count``:        `int` Number of chains
        * ``chain_at_collision_count``: `int` Number of collisions
        * ``chain_no_collision_count``: `int` Number of chain events that are
          no collision (i.e. no collision partner found or end of chain)
        * ``distance_queries``:         `int` Number of sweep distances
          calculated
        * ``overlap_errors``:           `int` Number of errors during sweep
          calculations

        Note:
            The counts are reset to 0 at the start of each
            `hoomd.Simulation.run`.
        """
        if self._attached:
            return self._cpp_obj.getNECCounters(1)
        else:
            return None

    @log(requires_run=True)
    def virial_pressure(self):
        """float: virial pressure.

        Note:
            The statistics are reset at every timestep.
        """
        return self._cpp_obj.virial_pressure

    @log(requires_run=True)
    def particles_per_chain(self):
        """float: particles per chain.

        Note:
            The statistics are reset at every `hoomd.Simulation.run`.
        """
        necCounts = self._cpp_obj.getNECCounters(1)
        return (necCounts.chain_at_collision_count * 1.0
                / necCounts.chain_start_count)

    @log(requires_run=True)
    def chains_in_space(self):
        """float: rate of chain events that did neither collide nor end.

        Note:
            The statistics are reset at every `hoomd.Simulation.run`.
        """
        necCounts = self._cpp_obj.getNECCounters(1)
        return (necCounts.chain_no_collision_count - necCounts.chain_start_count
                ) / (necCounts.chain_at_collision_count
                     + necCounts.chain_no_collision_count)


class Sphere(HPMCNECIntegrator):
    """HPMC chain integration for spheres (2D/3D).

    Args:
        default_d (`float`, optional): Default colission search distance
            :math:`[\\mathrm{length}]`, defaults to 0.1.
        chain_time (`float`, optional): Length of a chain
            :math:`[\\mathrm{time}]`, defaults to 0.5.
        update_fraction (`float`, optional): Number of chains to be done as
            fraction of N, defaults to 0.5.
        nselect (`int`, optional): The number of repeated updates to perform in
            each cell, defaults to 1.

    Perform Newtonian event chain Monte Carlo integration of spheres.

    .. rubric:: Wall support.

    `Sphere` supports no `hoomd.wall` geometries.

    .. rubric:: Potential support.

    `Sphere` does not support ``pair_potential`` or ``external_potential``.

    Attention:
        `Sphere` does not support execution on GPUs.

    Attention:
        `Sphere` does not support MPI parallel simulations.

    Example::

        mc = hoomd.hpmc.integrate.nec.Sphere(d=0.05, update_fraction=0.05)
        mc.chain_time = 0.05

    Attributes:
        chain_time (float): Length of a chain :math:`[\\mathrm{time}]`.

        update_fraction (float): Number of chains to be done as fraction of N.

        shape (`TypeParameter` [``particle type``, `dict`]):
            The shape parameters for each particle type. The dictionary has the
            following keys:

            * ``diameter`` (`float`, **required**) - Sphere diameter
              :math:`[\\mathrm{length}]`.
            * ``ignore_statistics`` (`bool`, **default:** `False`) - set to
              `True` to ignore tracked statistics.
            * ``orientable`` (`bool`, **default:** `False`) - set to `True` to
              allow rotation moves on this particle type.
    """

    _cpp_cls = 'IntegratorHPMCMonoNECSphere'

    def __init__(self,
                 default_d=0.1,
                 chain_time=0.5,
                 update_fraction=0.5,
                 nselect=1):
        # initialize base class
        super().__init__(default_d=default_d,
                         default_a=0.1,
                         chain_probability=1.0,
                         chain_time=chain_time,
                         update_fraction=update_fraction,
                         nselect=nselect)

        typeparam_shape = TypeParameter('shape',
                                        type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            diameter=float,
                                            ignore_statistics=False,
                                            orientable=False,
                                            len_keys=1))
        self._add_typeparam(typeparam_shape)

    @log(category='object')
    def type_shapes(self):
        """list[dict]: Description of shapes in ``type_shapes`` format.

        Examples:
            The types will be 'Sphere' regardless of dimensionality.

            >>> mc.type_shapes
            [{'type': 'Sphere', 'diameter': 1},
             {'type': 'Sphere', 'diameter': 2}]
        """
        return super()._return_type_shapes()


class ConvexPolyhedron(HPMCNECIntegrator):
    """HPMC integration for convex polyhedra (3D) with nec.

    Args:
        default_d (`float`, optional): Default colission search distance
            :math:`[\\mathrm{length}]`, defaults to 0.1.
        default_a (`float`, optional): Default maximum size of rotation trial
            moves :math:`[\\mathrm{dimensionless}]`, defaults to 0.1.
        chain_probability (`float`, optional): Probability of making a chain
            move instead of a rotation move, defaults to 0.5.
        chain_time (`float`, optional): Length of a chain
            :math:`[\\mathrm{time}]`, defaults to 0.5.
        update_fraction (`float`, optional): Number of chains to be done as
            fraction of N, defaults to 0.5.
        nselect (`int`, optional): Number of repeated updates for the
            cell/system, defaults to 1.

    Perform Newtonian event chain Monte Carlo integration of convex polyhedra.

    .. rubric:: Wall support.

    `ConvexPolyhedron` supports no `hoomd.wall` geometries.

    .. rubric:: Potential support.

    `ConvexPolyhedron` does not support ``pair_potential`` or
    ``external_potential``.

    Attention:
        `ConvexPolyhedron` does not support execution on GPUs.

    Attention:
        `ConvexPolyhedron` does not support MPI parallel simulations.

    Example::

        mc = hoomd.hpmc.nec.integrate.ConvexPolyhedron(d=1.0, a=0.05,
            chain_probability=0.1, nselect=10)
        mc.shape['A'] = dict(vertices=[[1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1],
            [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]])

    Attributes:
        chain_probability (float): Probability of making a chain move instead
            of a rotation move.

        chain_time (float): Length of a chain :math:`[\\mathrm{time}]`.

        update_fraction (float): Number of chains to be done as fraction of N.

        shape (`TypeParameter` [``particle type``, `dict`]):
            The shape parameters for each particle type. The dictionary has the
            following keys.

            * ``vertices`` (`list` [`tuple` [`float`, `float`, `float`]],
              **required**) - vertices of the polyhedron
              :math:`[\\mathrm{length}]`.

              * The origin **MUST** be contained within the polyhedron.
              * The origin centered sphere that encloses all vertices should
                be of minimal size for optimal performance.

            * ``ignore_statistics`` (`bool`, **default:** `False`) - set to
              `True` to ignore tracked statistics.
            * ``sweep_radius`` (`float`, **default:** 0.0) - Ignored, but
              present because `ConvexPolyhedron` shares data structures with
              `ConvexSpheropolyhedron` :math:`[\\mathrm{length}]`.

            Warning:
                HPMC does not check that all vertex requirements are met.
                Undefined behavior will result when they are violated.
    """
    _cpp_cls = 'IntegratorHPMCMonoNECConvexPolyhedron'

    def __init__(self,
                 default_d=0.1,
                 default_a=0.1,
                 chain_probability=0.5,
                 chain_time=0.5,
                 update_fraction=0.5,
                 nselect=1):

        super().__init__(default_d=default_d,
                         default_a=default_a,
                         chain_probability=chain_probability,
                         chain_time=chain_time,
                         update_fraction=update_fraction,
                         nselect=nselect)

        typeparam_shape = TypeParameter('shape',
                                        type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            vertices=[(float, float, float)],
                                            sweep_radius=0.0,
                                            ignore_statistics=False,
                                            len_keys=1))
        self._add_typeparam(typeparam_shape)

    @log(category='object')
    def type_shapes(self):
        """list[dict]: Description of shapes in ``type_shapes`` format.

        Example:
            >>> mc.type_shapes()
            [{'type': 'ConvexPolyhedron', 'sweep_radius': 0,
              'vertices': [[0.5, 0.5, 0.5], [0.5, -0.5, -0.5],
                           [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5]]}]
        """
        return super(ConvexPolyhedron, self)._return_type_shapes()
