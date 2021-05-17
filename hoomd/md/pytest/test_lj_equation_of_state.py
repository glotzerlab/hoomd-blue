import hoomd
import pytest
import hoomd.conftest
import numpy
import math

# Selected state points from the reference data at
# https://mmlapps.nist.gov/srs/LJ_PURE/mc.htm
# T_star, rho_star, mean_U_ref, sigma_U_ref, mean_P_ref, sigma_P_ref
statepoints = [
    (8.50E-01, 5.00E-03, -5.1901E-02, 7.53E-05, 4.1003E-03, 5.05E-07),
    #(9.00E-01, 7.76E-01, -5.4689E+00, 4.20E-04, 2.4056E-01, 2.74E-03)
]


@pytest.mark.validate
@pytest.mark.parametrize(
    "T_star, rho_star, mean_U_ref, sigma_U_ref, mean_P_ref, sigma_P_ref",
    statepoints)
def test_lj_nvt(T_star, rho_star, mean_U_ref, sigma_U_ref, mean_P_ref,
                sigma_P_ref, lattice_snapshot_factory, simulation_factory):

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
    method = hoomd.md.methods.NVT(hoomd.filter.All(), kT=T_star, tau=1)
    integrator.methods.append(method)
    sim.operations.integrator = integrator

    # equilibrate the simulation
    sim.run(0)
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=T_star)
    method.thermalize_thermostat_dof()
    sim.run(1000)

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

    # apply the long range correction to the energy used in thereference
    corrected_energy = numpy.array(energy_log.data) / N + 8/9.0 * math.pi * rho_star * ((1/r_cut)**9-3*(1/r_cut)**3)

    # compute the average and error
    energy = hoomd.conftest.BlockAverage(corrected_energy)
    pressure = hoomd.conftest.BlockAverage(pressure_log.data)
    rho = hoomd.conftest.BlockAverage(N / numpy.array(volume_log.data))

    print("U = ", numpy.mean(energy.data), '+/-', energy.get_error_estimate())
    print("P = ", numpy.mean(pressure.data), '+/-', pressure.get_error_estimate())
    print("rho = ", numpy.mean(rho.data), '+/-', rho.get_error_estimate())

    energy.assert_close(mean_U_ref, sigma_U_ref)
    pressure.assert_close(mean_P_ref, sigma_P_ref)
    # rho.assert_close(rho_star, 0)
