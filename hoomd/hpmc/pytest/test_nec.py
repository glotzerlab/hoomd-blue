# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test hoomd.hpmc.nec.integrate.Sphere."""

import hoomd
import hoomd.hpmc.tune
from hoomd.conftest import operation_pickling_check

import pytest
import math

# from test_sphere_eos.py for hoomd-2.9
# see for example Guang-Wen Wu and Richard J. Sadus, doi:10.1002.aic10233
_phi_p_ref = [(0.29054, 0.1), (0.91912, 0.2), (2.2768, 0.3), (5.29102, 0.4),
              (8.06553, 0.45), (9.98979, 0.475)]
rel_err_cs = 0.0015


@pytest.mark.parametrize("betap,phi", _phi_p_ref)
@pytest.mark.validate
def test_sphere_eos_nec(betap, phi, simulation_factory,
                        lattice_snapshot_factory):
    """Test that NEC runs and computes the pressure correctly."""
    n = 7

    v_particle = 4 / 3 * math.pi * (0.5)**3
    lattice_length = (v_particle / phi)**(1. / 3)
    snap = lattice_snapshot_factory(n=n, a=lattice_length)

    sim = simulation_factory(snap)
    sim.seed = 123456
    sim.state.thermalize_particle_momenta(hoomd.filter.All(), kT=1)

    mc = hoomd.hpmc.nec.integrate.Sphere(d=0.05, update_fraction=0.05)
    mc.shape['A'] = dict(diameter=1)
    mc.chain_time = 0.05
    sim.operations.integrator = mc

    triggerTune = hoomd.trigger.Periodic(50, 0)
    tune_nec_d = hoomd.hpmc.tune.MoveSize.scale_solver(
        triggerTune,
        moves=['d'],
        target=0.10,
        tol=0.001,
        max_translation_move=0.15)
    sim.operations.tuners.append(tune_nec_d)

    tune_nec_ct = hoomd.hpmc.nec.tune.ChainTime.scale_solver(triggerTune,
                                                             target=20.0,
                                                             tol=1.0,
                                                             gamma=20.0)
    sim.operations.tuners.append(tune_nec_ct)

    # equilibrate
    sim.run(1000)
    sim.operations.tuners.clear()

    pressures = []
    mc.nselect = 50
    for i in range(100):
        sim.run(1)
        pressures.append(mc.virial_pressure)

    mean = sum(pressures) / len(pressures)
    variance = sum(p**2 for p in pressures) / len(pressures) - mean**2
    std_dev = variance**0.5
    error = std_dev / (len(pressures) - 1)**0.5

    # confidence interval, 0.95 quantile of the normal distribution
    ci = 1.96

    assert math.isclose(betap, mean, abs_tol=ci * (error + rel_err_cs * betap))
    assert error < 1

    mcCounts = mc.counters
    necCounts = mc.nec_counters

    assert mcCounts.overlap_errors == 0
    assert necCounts.overlap_errors == 0


def test_pickling(valid_args, simulation_factory,
                  two_particle_snapshot_factory):
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
    # L needs to be ridiculously large as to not be too small for the domain
    # decomposition of some of the shapes definitions in valid_args which have
    # shapes with large extent in at least one dimension.
    sim = simulation_factory(two_particle_snapshot_factory(L=1000))
    operation_pickling_check(mc, sim)
