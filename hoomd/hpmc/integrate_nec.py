# Copyright (c) 2009-2019 The Regents of the University of Michigan
#                    2019 Marco Klement and Michael Engel
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

from hoomd import _hoomd
from hoomd.hpmc import _hpmc
from hoomd.hpmc import data
from hoomd.integrate import _integrator
import hoomd
import sys

from hoomd.hpmc.integrate import HPMCIntegrator


# add HPMC-NEC article citation notice
import hoomd
_citation_nec = hoomd.cite.article(cite_key='klement2019',
                               author=['M Klement', 'M Engel'],
                               title='Efficient equilibration of hard spheres with Newtonian event chains',
                               journal='The Journal of Chemical Physics',
                               volume=150,
                               pages='174108',
                               month='May',
                               year='2019',
                               doi='10.1063/1.5090882',
                               feature='HPMC-NEC')

if hoomd.context.bib is None:
    hoomd.cite._extra_default_entries.append(_citation_nec)
else:
    hoomd.context.bib.add(_citation_nec)



class HPMCNECIntegrator(HPMCIntegrator):
	""" HPMC Chain Integrator Meta Class
	
	Insert Doc-string here.
	"""
    _cpp_cls = None

    def __init__(self,
                 seed,
                 d=0.1,
                 a=0.1,
                 chain_probability=0.5,
                 chain_time=0.5,
                 update_fraction=0.5,
                 nselect=1):

        # initialize base class
        super().__init__(seed, d, a, translation_move_probability, nselect)


        # Set base parameter dict for hpmc chain integrators
        param_dict = ParameterDict(
            chain_probability=float(chain_probability),
            chain_time=float(chain_time),
            update_fraction=float(update_fraction))
        self._param_dict.update(param_dict)

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
        * ``overlap_err_count``:        `int` Number of errors during sweep
          calculations

        Note:
            The counts are reset to 0 at the start of each
            `hoomd.Simulation.run`.
        """
        if self._attached:
            return self._cpp_obj.getNECCounters(1)
        else:
            return None

    @log
    def virial_pressure(self):
        """float: virial pressure

        Note:
            The statistics are reset at every timestep.
        """
        if self._attached:
            return self._cpp_obj.getPressure()
        else:
            return None

    @log
    def particles_per_chain(self):
        """float: particles per chain

        Note:
            The statistics are reset at every `hoomd.Simulation.run`.
        """
        if self._attached:
            necCounts = self._cpp_obj.getNECCounters(1)
            return necCounts.chain_at_collision_count * 1.0 / necCounts.chain_start_count
        else:
            return None

    @log
    def chains_in_space(self):
        """float: rate of chain events that did neither collide nor end.

        Note:
            The statistics are reset at every `hoomd.Simulation.run`.
        """
        if self._attached:
            necCounts = self._cpp_obj.getNECCounters(1)
            return (necCounts.chain_no_collision_count - necCounts.chain_start_count) / ( necCounts.chain_at_collision_count + necCounts.chain_no_collision_count )
        else:
            return None

    # Tuners... (this probably is different now as well?)
    def get_particles_per_chain(self):
        R"""
        Returns the average number of particles in a chain for the last update step. (For use in a tuner.)
        """
        return self._cpp_obj.getTunerParticlesPerChain()

    def get_chain_time(self):
        R"""
        Get the current chain_time value. (For use in a tuner.)
        """
        return self._cpp_obj.getChainTime()
    
    def set_chain_time(self,chain_time):
        R"""
        Set the chain_time parameter to a new value. (For use in a tuner.)
        """
        self._cpp_obj.setChainTime(chain_time)
        


class Sphere(HPMCNECIntegrator):
    R""" HPMC chain integration for spheres (2D/3D).

    Args:
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        chain_time (float): length of a chain in units of time.
        update_fraction (float): number of chains to be done as fraction of N.
        nselect (int): The number of trial moves to perform in each cell.
        restore_state(bool): Restore internal state from initialization file when True. See :py:class:`mode_hpmc`
                             for a description of what state data restored. (added in version 2.2)

    Hard particle Monte Carlo integration method for spheres.

    Sphere parameters: (see sphere)

    Example:

        system = hoomd.init.read_gsd( "initialFile.gsd" )
        snapshot = system.take_snapshot(all=True)
        for i in range(snapshot.particles.N):
            snapshot.particles.velocity[i] = [ random.uniform(-1,1) for d in range(3) ]
        system.restore_snapshot(snapshot)

        target_nc = 100

        mc = hoomd.hpmc.integrate_nec.sphere(
                    d=0.5,
                    chain_time=10.0,
                    update_fraction=1.0/target_nc,
                    seed=1354765,
                    );
                    
        tune_nec_d  = hpmc.util.tune(self.mc, tunables=['d'],         max_val=[4],   gamma=1, target=0.03)
        tune_nec_ct = hpmc.util.tune(self.mc, tunables=['chain_time'], max_val=[100], gamma=1, target=1.0/target_nc, tunable_map=hoomd.hpmc.integrate_nec.make_tunable_map(mc))

    """

    _cpp_cls = 'IntegratorHPMCMonoNECSphere'

    def __init__(self,
                 seed,
                 d=0.1,
                 a=0.1,
                 chain_probability=0.5,
                 chain_time=0.5,
                 update_fraction=0.5,
                 nselect=1):
        # initialize base class
        super().__init__(
                 seed=seed,
                 d=d,
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

    @log(flag='object')
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
    R""" HPMC integration for convex polyhedra (3D) with nec.

    Args:
        seed (int): Random number seed.
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of chains to rotation moves. As there should be several particles in a chain it will be small. See example.
        nselect (int): (Override the automatic choice for the number of trial moves to perform in each cell.
        restore_state(bool): Restore internal state from initialization file when True. See :py:class:`mode_hpmc`
                             for a description of what state data restored. (added in version 2.2)

    Convex polyhedron parameters:
        -- as convex_polyhedron --

    Warning:
        HPMC does not check that all requirements are met. Undefined behavior will result if they are
        violated.

    Example:

        system = hoomd.init.read_gsd( "initialFile.gsd" )
        snapshot = system.take_snapshot(all=True)
        for i in range(snapshot.particles.N):
            snapshot.particles.velocity[i] = [ random.uniform(-1,1) for d in range(3) ]
        system.restore_snapshot(snapshot)
        
        target_nc = 100
        target_mr = 0.5
        
        param_mr = target_mr/(1+target_nc*(1-target_mr))
        
        mc = hoomd.hpmc.integrate_nec.convex_polyhedron(
                    d=0.5,
                    a=0.2,
                    move_ratio=param_mr,
                    chain_time=10.0,
                    update_fraction=0.5,
                    seed=1354765,
                    );
        mc.shape_param.set('A', vertices=[(0.5, 0.5, 0.5), (0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, 0.5)]);
        
        tune_mc_a   = hpmc.util.tune(self.mc, tunables=['a'],         max_val=[0.5], gamma=1, target=0.3)
        tune_nec_d  = hpmc.util.tune(self.mc, tunables=['d'],         max_val=[4],   gamma=1, target=0.03)
        tune_nec_ct = hpmc.util.tune(self.mc, tunables=['chain_time'],max_val=[100], gamma=1, target=1.0/target_nc,  tunable_map=hoomd.hpmc.integrate_nec.make_tunable_map(mc) )

        
    """
    _cpp_cls = 'IntegratorHPMCMonoNECConvexPolyhedron'


    def __init__(self,
                 seed,
                 d=0.1,
                 a=0.1,
                 chain_probability=0.5,
                 chain_time=0.5,
                 update_fraction=0.5,
                 nselect=1):

        super().__init__(
                 seed=seed,
                 d=d,
                 a=a,
                 chain_probability=chain_probability,
                 chain_time=chain_time,
                 update_fraction=update_fraction,
                 nselect=nselect):


        typeparam_shape = TypeParameter('shape',
                                        type_kind='particle_types',
                                        param_dict=TypeParameterDict(
                                            vertices=[(float, float, float)],
                                            sweep_radius=0.0,
                                            ignore_statistics=False,
                                            len_keys=1))
        self._add_typeparam(typeparam_shape)

    @log(flag='object')
    def type_shapes(self):
        """list[dict]: Description of shapes in ``type_shapes`` format.

        Example:
            >>> mc.type_shapes()
            [{'type': 'ConvexPolyhedron', 'sweep_radius': 0,
              'vertices': [[0.5, 0.5, 0.5], [0.5, -0.5, -0.5],
                           [-0.5, 0.5, -0.5], [-0.5, -0.5, 0.5]]}]
        """
        return super(ConvexPolyhedron, self)._return_type_shapes()
    
    
def make_tunable_map(obj=None):
    R"""
    Creates a tunable map for hpmc.tune and NEC.
    
    By updating the chain time the number of chains per particle is pushed towards the target.
    We used chains-per-particle = 1.0 / particles-per-chain as that value is treated like an
    acceptance rate for 'a' and 'd'.
    
    See the examples in integrate_nec.sphere and integrate_nec.convex_polyhedron for how it is used.
    """
    return {'chain_time': {
                    'get': lambda: getattr(obj, 'get_chain_time')(),
                    'acceptance': lambda: 1.0/getattr(obj, 'get_particles_per_chain')(),
                    'set': lambda x: getattr(obj, 'set_chain_time')(x),
                    'maximum': 100.0
                    }
              }
