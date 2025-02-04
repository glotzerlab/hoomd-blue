# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import pytest

import hoomd
import hoomd.conftest
import hoomd.md as md
import numpy


@pytest.fixture
def make_simulation(simulation_factory, two_particle_snapshot_factory):

    def sim_factory(particle_types=['A'], dimensions=3, d=1, L=20):
        snap = two_particle_snapshot_factory()
        if snap.communicator.rank == 0:
            snap.constraints.N = 1
            snap.constraints.value[0] = 1.0
            snap.constraints.group[0] = [0, 1]
        return simulation_factory(snap)

    return sim_factory


@pytest.fixture
def integrator_elements():
    nlist = md.nlist.Cell(buffer=0.4)
    lj = md.pair.LJ(nlist=nlist, default_r_cut=2.5)
    gauss = md.pair.Gaussian(nlist, default_r_cut=3.0)
    lj.params[("A", "A")] = {"epsilon": 1.0, "sigma": 1.0}
    gauss.params[("A", "A")] = {"epsilon": 1.0, "sigma": 1.0}
    return {
        "methods": [md.methods.ConstantVolume(hoomd.filter.All())],
        "forces": [lj, gauss],
        "constraints": [md.constrain.Distance()]
    }


def test_attaching(make_simulation, integrator_elements):
    sim = make_simulation()
    integrator = hoomd.md.Integrator(0.005, **integrator_elements)
    sim.operations.integrator = integrator
    sim.run(0)
    assert integrator._attached
    assert integrator._forces._synced
    assert integrator._methods._synced
    assert integrator._constraints._synced


def test_detaching(make_simulation, integrator_elements):
    sim = make_simulation()
    integrator = hoomd.md.Integrator(0.005, **integrator_elements)
    sim.operations.integrator = integrator
    sim.run(0)
    sim.operations._unschedule()
    assert not integrator._attached
    assert not integrator._forces._synced
    assert not integrator._methods._synced
    assert not integrator._constraints._synced


def test_validate_groups(simulation_factory, two_particle_snapshot_factory):
    snapshot = two_particle_snapshot_factory(particle_types=['R', 'A'])
    if snapshot.communicator.rank == 0:
        snapshot.particles.body[:] = [0, 1]
    CUBE_VERTS = [
        (-0.5, -0.5, -0.5),
        (-0.5, -0.5, 0.5),
        (-0.5, 0.5, -0.5),
        (-0.5, 0.5, 0.5),
        (0.5, -0.5, -0.5),
        (0.5, -0.5, 0.5),
        (0.5, 0.5, -0.5),
        (0.5, 0.5, 0.5),
    ]

    rigid = hoomd.md.constrain.Rigid()
    rigid.body['R'] = {
        "constituent_types": ['A'] * 8,
        "positions": CUBE_VERTS,
        "orientations": [(1.0, 0.0, 0.0, 0.0)] * 8,
    }

    nve1 = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    integrator = hoomd.md.Integrator(dt=0,
                                     methods=[nve1],
                                     integrate_rotational_dof=True)
    integrator.rigid = rigid
    sim = simulation_factory(snapshot)
    sim.operations.integrator = integrator

    rigid.create_bodies(sim.state)

    # Confirm that 1) Attaching calls `validate_groups` and 2) That
    # rigid constituent particles trigger an error.
    with pytest.raises(RuntimeError):
        sim.run(1)


def test_overlapping_filters(simulation_factory, lattice_snapshot_factory):
    snapshot = lattice_snapshot_factory()

    integrator = hoomd.md.Integrator(dt=0, integrate_rotational_dof=True)

    sim = simulation_factory(snapshot)
    sim.operations.integrator = integrator

    # Attach the integrator. No methods are set, so no error.
    sim.run(0)

    nve1 = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.Tags([0, 1]))
    nve2 = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.Tags([0, 1]))
    # Setting invalid methods does not trigger an error.
    integrator.methods = [nve1, nve2]

    # Running does not trigger an error, `validate_groups` is only called on
    # attach.
    sim.run(0)

    # Check that 1) Users can call `validate_groups` and 2) That overlapping
    # groups result in an error.
    with pytest.raises(RuntimeError):
        integrator.validate_groups()


def test_linear_momentum(simulation_factory, lattice_snapshot_factory):
    snapshot = lattice_snapshot_factory()
    if snapshot.communicator.rank == 0:
        snapshot.particles.mass[:] = numpy.linspace(1, 5, snapshot.particles.N)
        snapshot.particles.velocity[:,
                                    0] = numpy.linspace(-5, 5,
                                                        snapshot.particles.N)
        snapshot.particles.velocity[:,
                                    1] = numpy.linspace(1, 10,
                                                        snapshot.particles.N)
        snapshot.particles.velocity[:,
                                    2] = numpy.linspace(5, 20,
                                                        snapshot.particles.N)

    sim = simulation_factory(snapshot)
    integrator = hoomd.md.Integrator(dt=0.005)
    sim.operations.integrator = integrator
    sim.run(0)

    linear_momentum = integrator.linear_momentum

    if snapshot.communicator.rank == 0:
        reference = numpy.sum(snapshot.particles.mass[numpy.newaxis, :].T
                              * snapshot.particles.velocity,
                              axis=0)
        numpy.testing.assert_allclose(linear_momentum, reference)


def test_pickling(make_simulation, integrator_elements):
    sim = make_simulation()
    integrator = hoomd.md.Integrator(0.005, **integrator_elements)
    hoomd.conftest.operation_pickling_check(integrator, sim)


def test_logging():
    hoomd.conftest.logging_check(hoomd.md.Integrator, ("md",), {
        "linear_momentum": {
            "category": hoomd.logging.LoggerCategories.sequence
        }
    })
