# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import numpy as np
import pytest

from hoomd.conftest import operation_pickling_check
import hoomd
import hoomd.write

try:
    import h5py
except ImportError:
    pytestmark = pytest.skip("h5py required to test this feature.")


def lj_integrator():
    integrator = hoomd.md.Integrator(dt=0.005)
    lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(buffer=0.4),
                          default_r_cut=2.5)
    lj.params.default = {'sigma': 1, 'epsilon': 1}
    integrator.forces.append(lj)
    langevin = hoomd.md.methods.Langevin(hoomd.filter.All(), kT=1)
    integrator.methods.append(langevin)

    return integrator


@pytest.fixture(scope='function')
def create_md_sim(simulation_factory, device, two_particle_snapshot_factory):
    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = lj_integrator()

    return sim


@pytest.fixture
def logger():
    logger = hoomd.logging.Logger(categories=['scalar'])
    return logger


def test_invalid_attrs(tmp_path, logger):
    h5_writer = hoomd.write.HDF5Logger(1, tmp_path / "eg.h5", logger)
    with pytest.raises(AttributeError):
        h5_writer.action
    with pytest.raises(AttributeError):
        h5_writer.detach
    with pytest.raises(AttributeError):
        h5_writer.attach


def test_only_error_on_strings(tmp_path):
    logger = hoomd.logging.Logger(categories=["strings"])
    with pytest.raises(ValueError):
        hoomd.write.HDF5Logger(1, tmp_path / "eg.h5", logger)
    logger = hoomd.logging.Logger(categories=['string'])
    with pytest.raises(ValueError):
        hoomd.write.HDF5Logger(1, tmp_path / "eg.h5", logger)


def test_pickling(simulation_factory, two_particle_snapshot_factory, tmp_path,
                  logger):
    sim = simulation_factory(two_particle_snapshot_factory())
    h5_writer = hoomd.write.HDF5Logger(1, tmp_path / "eg.h5", logger)
    operation_pickling_check(h5_writer, sim)


def test_write(create_md_sim, tmp_path):

    filename = tmp_path / "temporary_test_file.h5"

    sim = create_md_sim
    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo)

    logger = hoomd.logging.Logger(["scalar", "particle", "sequence"])
    logger.add(thermo)

    h5_writer = hoomd.write.HDF5Logger(filename=filename,
                                       trigger=hoomd.trigger.Periodic(1),
                                       mode='w',
                                       logger=logger)
    sim.operations.writers.append(h5_writer)

    kinetic_energy_list = []
    for _ in range(5):
        sim.run(1)
        kinetic_energy_list.append(thermo.kinetic_energy)

    h5_writer.flush()

    if sim.device.communicator.rank == 0:
        key = 'hoomd-data/md/compute/ThermodynamicQuantities/kinetic_energy'
        with h5py.File(filename, mode='r') as fh:
            assert np.allclose(fh[key], kinetic_energy_list)


def test_mode(tmp_path, logger, create_md_sim):
    sim = create_md_sim
    fn = tmp_path / "eg.py"
    logger[("foo", "bar")] = (lambda: 42, "scalar")
    h5_writer = hoomd.write.HDF5Logger(1, fn, logger, mode="w")
    sim.operations.writers.append(h5_writer)
    sim.run(2)
    h5_writer.flush()
    sim.operations.writers.clear()
    del h5_writer
    with h5py.File(fn, "r") as fh:
        assert len(fh["hoomd-data/foo/bar"]) == 2

    h5_writer = hoomd.write.HDF5Logger(1, fn, logger, mode="a")
    sim.operations.writers.append(h5_writer)
    sim.run(2)
    h5_writer.flush()
    sim.operations.writers.clear()
    del h5_writer
    with h5py.File(fn, "r") as fh:
        assert len(fh["hoomd-data/foo/bar"]) == 4

    h5_writer = hoomd.write.HDF5Logger(1, fn, logger, mode="w")
    sim.operations.writers.append(h5_writer)
    sim.run(2)
    h5_writer.flush()
    sim.operations.writers.clear()
    del h5_writer
    with h5py.File(fn, "r") as fh:
        assert len(fh["hoomd-data/foo/bar"]) == 2
