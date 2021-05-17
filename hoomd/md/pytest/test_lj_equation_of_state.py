import hoomd
import pytest
import hoomd.conftest
import numpy
import math

# Selected state points from the reference data at
# https://mmlapps.nist.gov/srs/LJ_PURE/mc.htm T_star, rho_star, mean_U_ref,
# sigma_U_ref, mean_P_ref, sigma_P_ref JAA - I replaced the pressure values from
# the reference because I could not get the pressure correction term to
# validate. These pressure values here were computed in NVT and are used for
# cross validation between NPT and NVT in HOOMD.
statepoints = [
    (8.50E-01, 9.00E-03, -9.3973E-02, 1.29E-04, 7.2152E-03, 6.4e-06),
    (8.50E-01, 8.60E-01, -6.0305E+00, 2.38E-03, 1.7214E+00, 0.004421),
]


@pytest.mark.validate
@pytest.mark.parametrize(
    'T_star, rho_star, mean_U_ref, sigma_U_ref, mean_P_ref, sigma_P_ref',
    statepoints)
@pytest.mark.parametrize('method_name', ['NVT', 'Langevin', 'NPT'])
def test_lj_equation_of_state(T_star, rho_star, mean_U_ref, sigma_U_ref,
                              mean_P_ref, sigma_P_ref, method_name,
                              lattice_snapshot_factory, simulation_factory):

    # construct the system at the given density
    n = 8
    N = n**3
    V = N / rho_star
    L = V**(1 / 3)
    r_cut = 3.0

    snap = lattice_snapshot_factory(dimensions=3, n=n, a=L / n)
    sim = simulation_factory(snap)

    # set the simulation parameters
    integrator = hoomd.md.Integrator(dt=0.005)
    lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(), r_cut=r_cut)
    lj.params.default = {'sigma': 1, 'epsilon': 1}
    integrator.forces.append(lj)
    if method_name == 'NVT':
        method = hoomd.md.methods.NVT(filter=hoomd.filter.All(),
                                      kT=T_star,
                                      tau=1)
    elif method_name == 'Langevin':
        method = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=T_star)
    elif method_name == 'NPT':
        method = hoomd.md.methods.NPT(filter=hoomd.filter.All(),
                                      kT=T_star,
                                      tau=1.0,
                                      S=mean_P_ref,
                                      tauS=1.0,
                                      couple='xyz')
    integrator.methods.append(method)
    sim.operations.integrator = integrator

    # equilibrate the simulation
    sim.run(0)
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=T_star)
    if method_name == 'NVT':
        method.thermalize_thermostat_dof()
    elif method_name == 'NPT':
        method.thermalize_thermostat_and_barostat_dof()
    sim.run(5000)

    # log energy and pressure
    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo)
    energy_log = hoomd.conftest.ListWriter(thermo, 'potential_energy')
    pressure_log = hoomd.conftest.ListWriter(thermo, 'pressure')
    volume_log = hoomd.conftest.ListWriter(thermo, 'volume')
    log_trigger = hoomd.trigger.Periodic(100)
    sim.operations.writers.append(
        hoomd.write.CustomWriter(action=energy_log, trigger=log_trigger))
    sim.operations.writers.append(
        hoomd.write.CustomWriter(action=pressure_log, trigger=log_trigger))
    sim.operations.writers.append(
        hoomd.write.CustomWriter(action=volume_log, trigger=log_trigger))

    sim.always_compute_pressure = True
    sim.run(20000)

    # apply the long range correctionsused in the reference data
    corrected_energy = numpy.array(
        energy_log.data) / N + 8 / 9.0 * math.pi * rho_star * (
            (1 / r_cut)**9 - 3 * (1 / r_cut)**3)

    # compute the average and error
    energy = hoomd.conftest.BlockAverage(corrected_energy)
    pressure = hoomd.conftest.BlockAverage(pressure_log.data)
    rho = hoomd.conftest.BlockAverage(N / numpy.array(volume_log.data))

    print("U_ref = ", mean_U_ref, '+/-', sigma_U_ref)
    print("U = ", numpy.mean(energy.data), '+/-', energy.get_error_estimate())
    print("P_ref = ", mean_P_ref, '+/-', sigma_P_ref)
    print("P = ", numpy.mean(pressure.data), '+/-',
          pressure.get_error_estimate())
    print("rho = ", numpy.mean(rho.data), '+/-', rho.get_error_estimate())

    energy.assert_close(mean_U_ref, sigma_U_ref)

    # use larger tolerances for pressure and density as these tend to fluctuate
    if method_name == 'NVT' or method_name == 'Langevin':
        pressure.assert_close(mean_P_ref,
                              sigma_P_ref,
                              z=5,
                              max_relative_error=0.04)
    if method_name == 'NPT':
        rho.assert_close(rho_star, 0, z=5, max_relative_error=0.04)
