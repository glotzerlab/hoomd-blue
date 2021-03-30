# Copyright (c) 2009-2021 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause
# License.

"""Test hoomd.hpmc.integrate_nec.Sphere."""

import hoomd
import hoomd.hpmc.integrate_nec
import hoomd.hpmc.tune.nec_chain_time

import pytest
import math

## from test_sphere_eos.py for hoomd-2.9
##phi_p_ref = {0.29054: 0.1, 0.91912: 0.2, 2.2768: 0.3, 5.29102: 0.4, 8.06553: 0.45, 9.98979: 0.475}
phi_p_ref = [(0.29054, 0.1), (0.91912, 0.2), (2.2768, 0.3), (5.29102, 0.4),
             (8.06553, 0.45), (9.98979, 0.475)]
rel_err_cs = 0.0015  # see for example Guang-Wen Wu and Richard J. Sadus, doi:10.1002.aic10233


@pytest.mark.parametrize("betap,phi", phi_p_ref)
@pytest.mark.validate
def test_sphere_eos_nec(betap, phi, simulation_factory,
                        lattice_snapshot_factory):
    """Test that QuickCompress can compress (and expand) simulation boxes."""
    n = 7

    v_particle = 4 / 3 * math.pi * (0.5)**3
    lattice_length = (v_particle / phi)**(1. / 3)
    snap = lattice_snapshot_factory(n=n, a=lattice_length)

    sim = simulation_factory(snap)
    sim.state.thermalize_particle_momenta(hoomd.filter.All(), kT=1)

    mc = hoomd.hpmc.integrate_nec.Sphere(d=0.05, update_fraction=0.05)
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

    tune_nec_ct = hoomd.hpmc.tune.nec_chain_time.ChainTime.scale_solver(
        triggerTune, target=20, tol=1, gamma=20)
    sim.operations.tuners.append(tune_nec_ct)

    #equilibrate
    sim.run(1000)
    sim.operations.tuners.clear()

    pressures = []
    mc.nselect = 50
    for i in range(100):
        sim.run(1)
        pressures.append(mc.virial_pressure())

    mean = sum(pressures) / len(pressures)
    variance = sum(p**2 for p in pressures) / len(pressures) - mean**2
    std_dev = variance**0.5
    error = std_dev / (len(pressures) - 1)**0.5

    print(betap, mean, error)

    # confidence interval, 0.95 quantile of the normal distribution
    ci = 1.96

    assert abs(betap - mean) < ci * (error + rel_err_cs * betap)
    assert error < 1

    mcCounts = mc.counters
    necCounts = mc.nec_counters

    assert mcCounts.overlap_errors == 0
    assert necCounts.overlap_errors == 0
