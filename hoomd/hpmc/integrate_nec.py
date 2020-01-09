# Copyright (c) 2009-2019 The Regents of the University of Michigan
#                    2019 Marco Klement and Michael Engel
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

from hoomd import _hoomd
from hoomd.hpmc import _hpmc
from hoomd.hpmc import data
from hoomd.integrate import _integrator
import hoomd
import sys

from hoomd.hpmc.integrate import mode_hpmc, setD, setA
from hoomd.hpmc.integrate import sphere as hpmc_sphere
from hoomd.hpmc.integrate import convex_polyhedron as hpmc_convex_polyhedron


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



class sphere(hpmc_sphere):
    R""" HPMC chain integration for spheres (2D/3D).

    Args:
        seed (int): Random number seed
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

    def __init__(self, seed, d=0.1, chain_time=1, update_fraction=0.1, nselect=1, restore_state=False):
        hoomd.util.print_status_line();

        # These have no impact but are used as arguments...
        implicit=False
        depletant_mode='circumsphere'

        # initialize base class
        mode_hpmc.__init__(self,implicit, depletant_mode);

        # initialize the reflected c++ class
        if hoomd.context.exec_conf.isCUDAEnabled() or implicit:
            raise NotImplementedError("HPMC-SphereNEC is not implemented for implicit mode or use with CUDA.")

        self.cpp_integrator = _hpmc.IntegratorHPMCMonoNECSphere(hoomd.context.current.system_definition, seed);

        # set the default parameters
        setD(self.cpp_integrator,d);
        self.cpp_integrator.setMoveRatio(1.0)
        self.cpp_integrator.setNSelect(nselect);
        self.cpp_integrator.setChainTime(chain_time);
        self.cpp_integrator.setUpdateFraction(update_fraction);

        hoomd.context.current.system.setIntegrator(self.cpp_integrator);

        self.initialize_shape_params();

        if implicit:
            self.implicit_required_params=['nR', 'depletant_type']

        if restore_state:
            self.restore_state()
            
    def get_particles_per_chain(self):
        R"""
        Returns the average number of particles in a chain for the last update step. (For use in a tuner.)
        """
        return self.cpp_integrator.getTunerParticlesPerChain()

    def get_chain_time(self):
        R"""
        Get the current chain_time value. (For use in a tuner.)
        """
        return self.cpp_integrator.getChainTime()
    
    def set_chain_time(self,chain_time):
        R"""
        Set the chain_time parameter to a new value. (For use in a tuner.)
        """
        self.cpp_integrator.setChainTime(chain_time)
        


class convex_polyhedron(hpmc_convex_polyhedron):
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
    def __init__(self, seed, d=0.1, a=0.1, move_ratio=0.1, chain_time=10, update_fraction=0.5, nselect=4, restore_state=False):
        hoomd.util.print_status_line();
        
        # These have no impact but are used as arguments...
        implicit=False
        depletant_mode='circumsphere'
        max_verts=None

        if max_verts is not None:
            hoomd.context.msg.warning("max_verts is deprecated. Ignoring.\n")

        # initialize base class
        mode_hpmc.__init__(self,implicit, depletant_mode);

        # initialize the reflected c++ class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            if(implicit):
                raise NotImplementedError("HPMC-EventChain is not implemented for implicit mode.")
            else:
                self.cpp_integrator = _hpmc.IntegratorHPMCMonoNECConvexPolyhedron(hoomd.context.current.system_definition, seed);
        else:
            raise NotImplementedError("HPMC-EventChain is not implemented for use with cuda.")
        

        # set default parameters
        setD(self.cpp_integrator,d);
        setA(self.cpp_integrator,a);
        self.cpp_integrator.setMoveRatio(move_ratio)
        self.cpp_integrator.setChainTime(chain_time)
        self.cpp_integrator.setUpdateFraction(update_fraction)
        if nselect is not None:
            self.cpp_integrator.setNSelect(nselect);

        hoomd.context.current.system.setIntegrator(self.cpp_integrator);
        self.initialize_shape_params();

        if implicit:
            self.implicit_required_params=['nR', 'depletant_type']

        if restore_state:
            self.restore_state()
            
    def get_particles_per_chain(self):
        R"""
        Returns the average number of particles in a chain for the last update step. (For use in a tuner.)
        """
        return self.cpp_integrator.getTunerParticlesPerChain()

    def get_chain_time(self):
        R"""
        Get the current chain_time value. (For use in a tuner.)
        """
        return self.cpp_integrator.getChainTime()
    
    def set_chain_time(self,chain_time):
        R"""
        Set the chain_time parameter to a new value. (For use in a tuner.)
        """
        self.cpp_integrator.setChainTime(chain_time)
        
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
