from hoomd import *
import hoomd
import numpy as np
import pytest
import os
import tempfile
import gsd


def _make_gsd_snapshot(hoomd_snapshot):
    s = gsd.hoomd.Snapshot()
    for attr in dir(hoomd_snapshot):
        if attr[0] != '_' and attr not in ['exists', 'replicate']:
            for prop in dir(getattr(hoomd_snapshot, attr)):
                if prop[0] != '_':
                    # s.attr.prop = hoomd_snapshot.attr.prop
                    setattr(getattr(s, attr), prop,
                            getattr(getattr(hoomd_snapshot, attr), prop))
    return s


@pytest.fixture(scope='function')
def gsd_snapshot(lattice_snapshot_factory):
    snap = lattice_snapshot_factory(particle_types=particle_types,
                                    n=10, a=2.0, r=0.01)
    gsd_snap = _make_gsd_snapshot(snap)
    positions = np.random.rand(len(gsd_snap.particles), 3) * 2 - 1
    positions *= 20
    velocities = np.random.rand(len(gsd_snap.particles), 3) * 2 - 1
    accelerations = np.random.rand(len(gsd_snap.particles), 3) * 2 - 1

    gsd_snap.particles.position[:] = positions
    gsd_snap.particles.velocities[:] = velocities
    gsd_snap.particles.accelerations[:] = accelerations
    gsd_snap.particles.mass[:] = np.random.rand(len(gsd_snap.particles))
    gsd_snap.particles.charge[:] = np.random.rand(len(gsd_snap.particles))
    gsd_snap.particles.diameter[:] = np.random.rand(len(gsd_snap.particles))
    gsd_snap.particles.image[:] = np.random.randint(1, 100,
                                                    (len(gsd_snap.particles), 3))
    gsd_snap.particles.types = ['p1', 'p2']
    gsd_snap.particles.typeid = np.random.randint(0, 2, len(gsd_snap.particles))

    # bonds
    gsd_snap.bonds.types = ['b1', 'b2']
    gsd_snap.bonds.resize(2)
    gsd_snap.bonds.typeid[:] = [0, 1]
    gsd_snap.bonds.group[0] = [0, 1]
    gsd_snap.bonds.group[1] = [2, 3]

    # angles
    gsd_snap.angles.types = ['a1', 'a2']
    gsd_snap.angles.resize(2)
    gsd_snap.angles.typeid[:] = [1, 0]
    gsd_snap.angles.group[0] = [0, 1, 2]
    gsd_snap.angles.group[1] = [2, 3, 0]

    # dihedrals
    gsd_snap.dihedrals.types = ['d1']
    gsd_snap.dihedrals.resize(1)
    gsd_snap.dihedrals.typeid[:] = [0]
    gsd_snap.dihedrals.group[0] = [0, 1, 2, 3]

    # impropers
    gsd_snap.impropers.types = ['i1']
    gsd_snap.impropers.resize(1)
    gsd_snap.impropers.typeid[:] = [0]
    gsd_snap.impropers.group[0] = [3, 2, 1, 0]

    # constraints
    gsd_snap.constraints.resize(1)
    gsd_snap.constraints.group[0] = [0, 1]
    gsd_snap.constraints.value[0] = 2.5

    # special pairs
    gsd_snap.pairs.types = ['p1', 'p2']
    gsd_snap.pairs.resize(2)
    gsd_snap.pairs.typeid[:] = [0, 1]
    gsd_snap.pairs.group[0] = [0, 1]
    gsd_snap.pairs.group[1] = [2, 3]
    return gsd_snap


# tests basic creation of the dump
def test_dump(simulation_factory, lattice_snapshot_factory, tmp_path):
    sim = simulation_factory(lattice_snapshot_factory())
    d = tmp_path / "sub"
    d.mkdir()
    filename = d / "temporary_test_file.gsd"
    trigger = hoomd.trigger.Periodic(1)
    gsd_dump = hoomd.dump.GSD(filename, trigger)
    sim.operations.add(gsd_dump)
    sim.operations.schedule()
    sim.run(1)
