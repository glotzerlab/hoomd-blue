import hoomd
import pytest
import numpy as np
import hoomd.hpmc.pytest.conftest


def test_before_attaching():
    sdf = hoomd.hpmc.compute.SDF(xmax=0.02, dx=1e-4)

    assert sdf.xmax == 0.02
    assert sdf.dx == 1e-4
    with pytest.raises(hoomd.error.DataAccessError):
        sdf.sdf
    with pytest.raises(hoomd.error.DataAccessError):
        sdf.betaP


def test_after_attaching(valid_args, simulation_factory,
                         lattice_snapshot_factory):
    snap = lattice_snapshot_factory(particle_types=['A'])
    sim = simulation_factory(snap)

    integrator = valid_args[0]
    args = valid_args[1]
    # Need to unpack union integrators
    if isinstance(integrator, tuple):
        inner_integrator = integrator[0]
        integrator = integrator[1]
        inner_mc = inner_integrator()
        for i in range(len(args["shapes"])):
            # This will fill in default values for the inner shape objects
            inner_mc.shape["A"] = args["shapes"][i]
            args["shapes"][i] = inner_mc.shape["A"]
    mc = integrator()
    mc.shape["A"] = args
    sim.operations.add(mc)

    sdf = hoomd.hpmc.compute.SDF(xmax=0.02, dx=1e-4)

    sim.operations.add(sdf)
    assert len(sim.operations.computes) == 1
    sim.run(0)

    assert sdf.xmax == 0.02
    assert sdf.dx == 1e-4

    sim.run(10)
    assert isinstance(sdf.sdf, np.ndarray)
    assert len(sdf.sdf) > 0
    assert isinstance(sdf.betaP, float)
    assert not np.isclose(sdf.betaP, 0)
