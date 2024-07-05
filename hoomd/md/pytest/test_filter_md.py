# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import pytest
from hoomd.filter import CustomFilter, Rigid
import hoomd.md as md
import numpy as np
from hoomd import Snapshot


@pytest.fixture(scope="function")
def make_filter_snapshot(device):

    def filter_snapshot(n=10, particle_types=['A']):
        s = Snapshot(device.communicator)
        if s.communicator.rank == 0:
            s.configuration.box = [20, 20, 20, 0, 0, 0]
            s.particles.N = n
            s.particles.position[:] = np.random.uniform(-10, 10, size=(n, 3))
            s.particles.types = particle_types
        return s

    return filter_snapshot


def test_rigid_filter(make_filter_snapshot, simulation_factory):
    MPI = pytest.importorskip("mpi4py.MPI")

    rigid = md.constrain.Rigid()
    rigid.body["A"] = {
        "constituent_types": ["B", "B", "B", "B"],
        "positions": [
            [1, 0, -1 / (2**(1. / 2.))],
            [-1, 0, -1 / (2**(1. / 2.))],
            [0, -1, 1 / (2**(1. / 2.))],
            [0, 1, 1 / (2**(1. / 2.))],
        ],
        "orientations": [(1.0, 0.0, 0.0, 0.0)] * 4,
    }

    snapshot = make_filter_snapshot(n=100, particle_types=["A", "B", "C"])
    if snapshot.communicator.rank == 0:
        snapshot.particles.typeid[50:100] = 2
        snapshot.particles.body[50:100] = -1
    sim = simulation_factory(snapshot)
    rigid.create_bodies(sim.state)

    def check_tags(filter_, state, expected_tags):
        mpi_communicator = MPI.COMM_WORLD

        local_tags = filter_(state)
        all_tags = mpi_communicator.gather(local_tags, root=0)
        if mpi_communicator.rank == 0:
            # unique automatically sorts the items
            all_tags = np.unique(np.concatenate(all_tags))
            assert np.all(all_tags == expected_tags)

    only_centers = Rigid()
    check_tags(only_centers, sim.state, np.arange(50))

    only_free = Rigid(('free',))
    check_tags(only_free, sim.state, np.arange(50, 100))

    only_constituent = Rigid(('constituent',))
    check_tags(only_constituent, sim.state, np.arange(100, 300))

    free_and_centers = Rigid(('free', 'center'))
    check_tags(free_and_centers, sim.state, np.arange(0, 100))

    constituent_and_centers = Rigid(('constituent', 'center'))
    check_tags(constituent_and_centers, sim.state,
               np.concatenate((np.arange(0, 50), np.arange(100, 300))))

    constituent_and_free = Rigid(('free', 'constituent'))
    check_tags(constituent_and_free, sim.state, np.arange(50, 300))

    all_ = Rigid(('free', 'constituent', 'center'))
    check_tags(all_, sim.state, np.arange(0, 300))


def test_custom_filter(make_filter_snapshot, simulation_factory):
    """Tests that custom particle filters work on simulations.

    Specifically we test that using the Langevin integrator method, that only
    particles selected by the custom filter move. Since the Langevin method uses
    random movements we don't need to initialize velocities or have any forces
    to test this.
    """

    class NegativeCharge(CustomFilter):
        """Grab all particles with a negative charge."""

        def __call__(self, state):
            with state.cpu_local_snapshot as snap:
                return snap.particles.tag[snap.particles.charge < 0]

        def __hash__(self):
            return hash(self.__class__.__name__)

        def __eq__(self, other):
            return isinstance(other, self.__class__)

    charge_filter = NegativeCharge()
    sim = simulation_factory(make_filter_snapshot())
    # grabs tags on individual MPI ranks
    with sim.state.cpu_local_snapshot as snap:
        # Grab half of all particles on an MPI rank, 1 particle, or no particles
        # depending on how many particles are local to the MPI ranks.
        local_Np = snap.particles.charge.shape[0]
        N_negative_charge = max(0, max(1, int(local_Np * 0.5)))
        negative_charge_ind = np.random.choice(local_Np,
                                               N_negative_charge,
                                               replace=False)
        # Get the expected tags returned by the custom filter and the positions
        # that should vary and remain static for testing after running.
        snap.particles.charge[negative_charge_ind] = -1.0
        expected_tags = snap.particles.tag[negative_charge_ind]
        positive_charge_tags = snap.particles.tag[snap.particles.charge > 0]
        positive_charge_ind = snap.particles.rtag[positive_charge_tags]
        original_positions = snap.particles.position[negative_charge_ind]
        static_positions = snap.particles.position[positive_charge_ind]

    # Test that the filter merely works as expected and that tags are correctly
    # grabbed on local MPI ranks
    assert all(np.sort(charge_filter(sim.state)) == np.sort(expected_tags))

    # Test that the filter works when used in a simulation
    langevin = md.methods.Langevin(charge_filter, 1.0)
    sim.operations += md.Integrator(0.005, methods=[langevin])
    sim.run(100)
    snap = sim.state.get_snapshot()
    if snap.communicator.rank == 0:
        assert not np.allclose(snap.particles.position[negative_charge_ind],
                               original_positions)
        assert np.allclose(snap.particles.position[positive_charge_tags],
                           static_positions)
