# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Newtonain Event-Chain Integrators for Hard Particle Monte Carlo."""

from hoomd.hpmc.integrate import HPMCIntegrator
from hoomd.data.parameterdicts import TypeParameterDict, ParameterDict
from hoomd.data.typeparam import TypeParameter
from hoomd.data.typeconverter import OnlyTypes

from hoomd.logging import log


class HPMCNECIntegrator(HPMCIntegrator):
    """HPMC Chain Integrator base class.

    :py:class:`HPMCNECIntegrator` is the base class for all HPMC Newtonian event
    chain integrators. Users should not instantiate this class directly. The
    attributes documented here are available to all HPMC integrators.
    """
    _cpp_cls = None

    def __init__(self,
                 d=0.1,
                 a=0.1,
                 chain_probability=0.5,
                 chain_time=0.5,
                 update_fraction=0.5,
                 nselect=1):
        # initialize base class
        super().__init__(d, a, 0.5, nselect)

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

    def _attach(self):
        super()._attach()

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
        d (float): Maximum move displacement, Scalar to set for all types,
            or a dict containing {type:size} to set by type.
        chain_time (float): length of a chain in units of time.
        update_fraction (float): number of chains to be done as fraction of N.
        nselect (int): The number of repeated updates to perform in each cell.

    Hard particle Monte Carlo integration method for spheres.

    Sphere parameters: (see sphere)

    Example:
        mc = hoomd.hpmc.integrate_nec.Sphere(d=0.05, update_fraction=0.05)
        mc.chain_time = 0.05
    """

    _cpp_cls = 'IntegratorHPMCMonoNECSphere'

    def __init__(self, d=0.1, chain_time=0.5, update_fraction=0.5, nselect=1):
        # initialize base class
        super().__init__(d=d,
                         a=0.1,
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
        d (float): Maximum move displacement, Scalar to set for all types,
            or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a
            dict containing {type:size} to set by type.
        chain_probability (float): Ratio of chains to rotation moves. As
            there should be several particles in a chain it will be small.
            See example.
        chain_time (float):
        update_fraction (float):
        nselect (int): Number of repeated updates for the cell/system.

    Convex polyhedron parameters:
        see ``ConvexPolyhedron``

    Warning:
        HPMC does not check that all requirements are met. Undefined behavior
        will result if they are violated.

    Example:
        cpu = hoomd.device.CPU()
        sim = hoomd.Simulation(device=cpu)
        sim.create_state_from_gsd(filename='start.gsd')

        sim.state.thermalize_particle_momenta(hoomd.filter.All(), kT=1)

        mc = hoomd.hpmc.integrate_nec.ConvexPolyhedron(d=1.0, a=0.05,
            chain_probability=0.1, nselect=10)
        mc.shape['A'] = dict(vertices=[[1,1,1], [1,1,-1], [1,-1,1], [1,-1,-1],
            [-1,1,1], [-1,1,-1], [-1,-1,1], [-1,-1,-1]])
        mc.chain_time = 0.05
        sim.operations.integrator = mc

        triggerTune = hoomd.trigger.Periodic(50,0)
        tune_nec_d = hoomd.hpmc.tune.MoveSize.scale_solver(triggerTune,
                        moves=['d'], target=0.10, max_translation_move=0.15)
        sim.operations.tuners.append(tune_nec_d)
        tune_nec_a = hoomd.hpmc.tune.MoveSize.scale_solver(triggerTune,
                        moves=['a'], target=0.30)
        sim.operations.tuners.append(tune_nec_a)

        import hoomd.hpmc.tune.nec_chain_time
        tune_nec_ct = hoomd.hpmc.tune.ChainTime.scale_solver(
                        triggerTune, target=20, tol=1, gamma=20 )
        sim.operations.tuners.append(tune_nec_ct)

        sim.run(1000)


    """
    _cpp_cls = 'IntegratorHPMCMonoNECConvexPolyhedron'

    def __init__(self,
                 d=0.1,
                 a=0.1,
                 chain_probability=0.5,
                 chain_time=0.5,
                 update_fraction=0.5,
                 nselect=1):

        super().__init__(d=d,
                         a=a,
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
