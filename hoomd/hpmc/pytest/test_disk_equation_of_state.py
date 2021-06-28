import hoomd
import pytest
import hoomd.conftest
import numpy

# mean_phi_p_ref, sigma_phi_p_ref, mean_betaP_ref, sigma_betaP_ref, n_particles
statepoints = [
    (0.698, 0.0001, 9.1709, 0.0002),
]


@pytest.mark.validate
@pytest.mark.parametrize(
    'mean_phi_p_ref, sigma_phi_p_ref, mean_betaP_ref, sigma_betaP_ref', statepoints)
def test_disk_equation_of_state(
    mean_phi_p_ref,
    sigma_phi_p_ref,
    mean_betaP_ref,
    sigma_betaP_ref,
    lattice_snapshot_factory,
    simulation_factory,
    device,
):
    # construct the system at the given density
    n = 256
    N = n**2
    a = numpy.sqrt(numpy.pi / (4 * mean_phi_p_ref))

    snap = lattice_snapshot_factory(dimensions=2, a=a, n=n)
    sim = simulation_factory(snap)
    sim.seed = 0

    mc = hoomd.hpmc.integrate.Sphere(d=0.2)
    mc.shape["A"] = dict(diameter=1.0)
    sim.operations.add(mc)

    sim.run(1e3)

    sdf = hoomd.hpmc.compute.SDF(xmax=0.02, dx=1e-4)
    sim.operations.add(sdf)
    betaP_log = hoomd.conftest.ListWriter(sdf, 'betaP')
    sim.operations.writers.append(hoomd.write.CustomWriter(action=betaP_log, trigger=hoomd.trigger.Periodic(100)))

    sim.run(1e4)

    betaP = hoomd.conftest.BlockAverage(betaP_log.data)

    # Useful information to know when the test fails
    print('betaP_ref = ', mean_betaP_ref, '+/-', sigma_betaP_ref)
    print('betaP = ', betaP.mean, '+/-', betaP.standard_deviation, '(',
          betaP.relative_error * 100, '%)')

    betaP.assert_close(mean_betaP_ref, sigma_betaP_ref)
