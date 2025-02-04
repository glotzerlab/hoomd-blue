# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import numpy as np
import pytest

from hoomd.conftest import operation_pickling_check
import hoomd
import hoomd.write

h5py = pytest.importorskip("h5py")


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


def test_invalid_attrs(tmp_path):
    logger = hoomd.logging.Logger(categories=['scalar'])
    hdf5_writer = hoomd.write.HDF5Log(1, tmp_path / "eg.h5", logger)
    with pytest.raises(AttributeError):
        hdf5_writer.action
    with pytest.raises(AttributeError):
        hdf5_writer.detach
    with pytest.raises(AttributeError):
        hdf5_writer.attach


def test_only_error_on_strings(tmp_path):
    logger = hoomd.logging.Logger(categories=["strings"])
    with pytest.raises(ValueError):
        hoomd.write.HDF5Log(1, tmp_path / "eg.h5", logger)
    logger = hoomd.logging.Logger(categories=['string'])
    with pytest.raises(ValueError):
        hoomd.write.HDF5Log(1, tmp_path / "eg.h5", logger)


def test_pickling(simulation_factory, two_particle_snapshot_factory, tmp_path):
    logger = hoomd.logging.Logger(categories=['scalar'])
    sim = simulation_factory(two_particle_snapshot_factory())
    hdf5_writer = hoomd.write.HDF5Log(1, tmp_path / "eg.h5", logger)
    operation_pickling_check(hdf5_writer, sim)


def test_write(create_md_sim, tmp_path):

    filename = tmp_path / "temporary_test_file.h5"

    sim = create_md_sim
    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo)

    logger = hoomd.logging.Logger(["scalar", "particle", "sequence"])
    logger.add(thermo)

    hdf5_writer = hoomd.write.HDF5Log(filename=filename,
                                      trigger=hoomd.trigger.Periodic(1),
                                      mode='w',
                                      logger=logger)
    sim.operations.writers.append(hdf5_writer)

    kinetic_energy_list = []
    for _ in range(5):
        sim.run(1)
        kinetic_energy_list.append(thermo.kinetic_energy)

    hdf5_writer.flush()

    if sim.device.communicator.rank == 0:
        key = 'hoomd-data/md/compute/ThermodynamicQuantities/kinetic_energy'
        with h5py.File(filename, mode='r') as fh:
            assert np.allclose(fh[key], kinetic_energy_list)


def test_write_method(create_md_sim, tmp_path):
    filename = tmp_path / "temporary_test_file.h5"

    sim = create_md_sim
    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo)

    logger = hoomd.logging.Logger(["scalar", "particle", "sequence"])
    logger.add(thermo)

    # Writer should never run on its own.
    hdf5_writer = hoomd.write.HDF5Log(filename=filename,
                                      trigger=hoomd.trigger.Periodic(100),
                                      mode='w',
                                      logger=logger)
    sim.operations.writers.append(hdf5_writer)

    kinetic_energy_list = []
    for _ in range(5):
        sim.run(1)
        kinetic_energy_list.append(thermo.kinetic_energy)
        hdf5_writer.write()

    hdf5_writer.flush()

    if sim.device.communicator.rank == 0:
        key = 'hoomd-data/md/compute/ThermodynamicQuantities/kinetic_energy'
        with h5py.File(filename, mode='r') as fh:
            assert np.allclose(fh[key], kinetic_energy_list)


def test_mode(tmp_path, create_md_sim):
    logger = hoomd.logging.Logger(categories=['scalar'])
    sim = create_md_sim
    fn = tmp_path / "eg.py"
    logger[("foo", "bar")] = (lambda: 42, "scalar")
    hdf5_writer = hoomd.write.HDF5Log(1, fn, logger, mode="w")
    sim.operations.writers.append(hdf5_writer)
    sim.run(2)
    hdf5_writer.flush()
    sim.operations.writers.clear()
    del hdf5_writer
    if sim.device.communicator.rank == 0:
        with h5py.File(fn, "r") as fh:
            assert len(fh["hoomd-data/foo/bar"]) == 2

    hdf5_writer = hoomd.write.HDF5Log(1, fn, logger, mode="a")
    sim.operations.writers.append(hdf5_writer)
    sim.run(2)
    hdf5_writer.flush()
    sim.operations.writers.clear()
    del hdf5_writer
    if sim.device.communicator.rank == 0:
        with h5py.File(fn, "r") as fh:
            assert len(fh["hoomd-data/foo/bar"]) == 4

    hdf5_writer = hoomd.write.HDF5Log(1, fn, logger, mode="w")
    sim.operations.writers.append(hdf5_writer)
    sim.run(2)
    hdf5_writer.flush()
    sim.operations.writers.clear()
    del hdf5_writer
    if sim.device.communicator.rank == 0:
        with h5py.File(fn, "r") as fh:
            assert len(fh["hoomd-data/foo/bar"]) == 2


def test_type_handling(tmp_path, create_md_sim):
    logger = hoomd.logging.Logger(categories=['scalar'])
    sim = create_md_sim
    fn = tmp_path / "types.h5"
    loggables = {
        int: lambda: 42,
        float: lambda: 0.0,
        bool: lambda: True,
        np.uint32: lambda: np.uint32(42),
        np.float32: lambda: np.float32(3.1415),
        np.bool_: lambda: np.bool_(True)
    }
    for key, value in loggables.items():
        logger[str(key)] = (value, "scalar")
    hdf5_writer = hoomd.write.HDF5Log(1, fn, logger, mode="w")
    sim.operations.writers.append(hdf5_writer)
    sim.run(1)

    rank = sim.device.communicator.rank
    del sim

    if rank == 0:
        with h5py.File(fn, "r") as fh:
            for key in loggables:
                type_ = key if key not in (float, int, bool) else np.dtype(key)
                assert fh[f"hoomd-data/{str(key)}"].dtype == type_
