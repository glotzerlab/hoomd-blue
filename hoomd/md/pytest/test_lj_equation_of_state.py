# Copyright (c) 2009-2022 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import pytest
import hoomd.conftest
import numpy

# Selected state point(s) in the high density fluid. These state point(s) are in
# single phase regions of the phase diagram and suitable for NVT/NPT and MC
# cross validation.
# T_star, rho_star, mean_U_ref, sigma_U_ref, mean_P_ref, sigma_P_ref,
# log_period, equilibration_steps, run_steps
statepoints = [
    (1.4, 0.9, -4.6622, 0.0006089, 6.6462, 0.00328, 64, 2**10, 2**16),
]


@pytest.mark.validate
@pytest.mark.parametrize(
    'T_star, rho_star, mean_U_ref, sigma_U_ref, mean_P_ref, sigma_P_ref,'
    'log_period, equilibration_steps, run_steps', statepoints)
@pytest.mark.parametrize('method_name', ['Langevin', 'NVT', 'NPT', 'NVTStochastic'])
def test_lj_equation_of_state(
    T_star,
    rho_star,
    mean_U_ref,
    sigma_U_ref,
    mean_P_ref,
    sigma_P_ref,
    log_period,
    equilibration_steps,
    run_steps,
    method_name,
    fcc_snapshot_factory,
    simulation_factory,
    device,
):
    # construct the system at the given density
    n = 6
    if device.communicator.num_ranks > 1:
        # MPI tests need a box large enough to decompose
        n = 8
    N = n**3 * 4
    V = N / rho_star
    L = V**(1 / 3)
    r_cut = 2.5
    a = L / n

    snap = fcc_snapshot_factory(n=n, a=a)
    sim = simulation_factory(snap)
    sim.seed = 10

    # set the simulation parameters
    integrator = hoomd.md.Integrator(dt=0.005)
    lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(buffer=0.4),
                          default_r_cut=r_cut,
                          mode='shift')
    lj.params.default = {'sigma': 1, 'epsilon': 1}
    integrator.forces.append(lj)
    if method_name == 'NVT':
        method = hoomd.md.methods.NVT(filter=hoomd.filter.All(),
                                      kT=T_star,
                                      tau=0.1)
    elif method_name == 'Langevin':
        method = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=T_star)
    elif method_name == 'NPT':
        method = hoomd.md.methods.NPT(filter=hoomd.filter.All(),
                                      kT=T_star,
                                      tau=0.1,
                                      S=mean_P_ref,
                                      tauS=0.5,
                                      couple='xyz',
                                      gamma = 10.0)
    elif method_name == 'NVTStochastic':
        method = hoomd.md.methods.NVTStochastic(filter=hoomd.filter.All(), kT = T_star)
    integrator.methods.append(method)
    sim.operations.integrator = integrator

    # equilibrate the simulation
    sim.run(0)
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=T_star)
    if method_name == 'NVT':
        method.thermalize_thermostat_dof()
    elif method_name == 'NPT':
        method.thermalize_thermostat_and_barostat_dof()
    sim.run(equilibration_steps)

    # log energy and pressure
    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo)
    energy_log = hoomd.conftest.ListWriter(thermo, 'potential_energy')
    pressure_log = hoomd.conftest.ListWriter(thermo, 'pressure')
    volume_log = hoomd.conftest.ListWriter(thermo, 'volume')
    log_trigger = hoomd.trigger.Periodic(log_period)
    sim.operations.writers.append(
        hoomd.write.CustomWriter(action=energy_log, trigger=log_trigger))
    sim.operations.writers.append(
        hoomd.write.CustomWriter(action=pressure_log, trigger=log_trigger))
    sim.operations.writers.append(
        hoomd.write.CustomWriter(action=volume_log, trigger=log_trigger))

    sim.always_compute_pressure = True
    sim.run(run_steps)

    # compute the average and error
    energy = hoomd.conftest.BlockAverage(numpy.array(energy_log.data) / N)
    pressure = hoomd.conftest.BlockAverage(pressure_log.data)
    rho = hoomd.conftest.BlockAverage(N / numpy.array(volume_log.data))

    # Useful information to know when the test fails
    print('U_ref = ', mean_U_ref, '+/-', sigma_U_ref)
    print('U = ', energy.mean, '+/-', energy.standard_deviation, '(',
          energy.relative_error * 100, '%)')
    print('P_ref = ', mean_P_ref, '+/-', sigma_P_ref)
    print('P = ', pressure.mean, '+/-', pressure.standard_deviation,
          pressure.standard_deviation, '(', pressure.relative_error * 100, '%)')
    print('rho = ', rho.mean, '+/-', rho.standard_deviation)

    print(f'Statepoint entry: {T_star:0.4}, {rho_star:0.4}, '
          f'{energy.mean:0.5}, {energy.standard_deviation:0.4}, '
          f'{pressure.mean:0.5}, {pressure.standard_deviation:0.4}')

    energy.assert_close(mean_U_ref, sigma_U_ref)

    if method_name == 'NVT' or method_name == 'Langevin' or method_name == 'NVTStochastic':
        pressure.assert_close(mean_P_ref, sigma_P_ref)

    if method_name == 'NPT':
        rho.assert_close(rho_star, 0)
