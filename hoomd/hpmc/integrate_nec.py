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

        mc = hoomd.hpmc.integrate.sphere_nec(
                    d=0.5,
                    chain_time=10.0,
                    update_fraction=0.02,
                    seed=1354765,
                    );

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

        # Error out in MPI simulations
        if (_hoomd.is_MPI_available()):
            if hoomd.context.current.system_definition.getParticleData().getDomainDecomposition():
                hoomd.context.msg.error("HPMC-SphereNEC is not supported in multi-processor simulations.\n\n")
                raise RuntimeError("Error setting up integration method.")

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


class convex_polyhedron(hpmc_convex_polyhedron):
    R""" HPMC integration for convex polyhedra (3D) with nec.

    Args:
        seed (int): Random number seed.
        d (float): Maximum move displacement, Scalar to set for all types, or a dict containing {type:size} to set by type.
        a (float): Maximum rotation move, Scalar to set for all types, or a dict containing {type:size} to set by type.
        move_ratio (float): Ratio of translation moves to rotation moves.
        nselect (int): (Override the automatic choice for the number of trial moves to perform in each cell.
        implicit (bool): Flag to enable implicit depletants.
        depletant_mode (string, only with **implicit=True**): Where to place random depletants, either 'circumsphere' or 'overlap_regions'
            (added in version 2.2)
        max_verts (int): Set the maximum number of vertices in a polyhedron. (deprecated in version 2.2)
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
            
        mc = hoomd.hpmc.integrate.convex_polyhedron_nec(
                    d=0.5,
                    a=0.2,
                    move_ratio=0.05,
                    chain_time=10.0,
                    update_fraction=0.5,
                    seed=1354765,
                    );
        mc.shape_param.set('A', vertices=[(0.5, 0.5, 0.5), (0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (-0.5, -0.5, 0.5)]);
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

        # Error out in MPI simulations
        if (_hoomd.is_MPI_available()):
            if hoomd.context.current.system_definition.getParticleData().getDomainDecomposition():
                hoomd.context.msg.error("HPMC-NEC is not supported in multi-processor simulations.\n\n")
                raise RuntimeError("Error setting up integration method.")        

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
            
