from hoomd import *
import hoomd
import numpy as np
import pytest
import os
import tempfile
import gsd.hoomd


def _make_gsd_snapshot(snap):
    s = gsd.hoomd.Snapshot()
    for attr in dir(snap):
        if attr[0] != '_' and attr not in ['exists', 'replicate']:
            for prop in dir(getattr(snap, attr)):
                if prop[0] != '_':
                    # s.attr.prop = snap.attr.prop
                    setattr(getattr(s, attr), prop,
                            getattr(getattr(snap, attr), prop))
    return s


def _assert_equivalent_snapshots(gsd_snap, hoomd_snap):
    for attr in dir(hoomd_snap):
        if attr[0] == '_' or attr in ['exists', 'replicate']:
            continue
        for prop in dir(getattr(hoomd_snap, attr)):
            if prop[0] == '_':
                continue
            elif prop == 'types':
                # if hoomd_snap.exists:
                assert getattr(getattr(gsd_snap, attr), prop) == \
                    getattr(getattr(hoomd_snap, attr), prop)
            else:
                # if hoomd_snap.exists:
                np.testing.assert_allclose(
                    getattr(getattr(gsd_snap, attr), prop),
                    getattr(getattr(hoomd_snap, attr), prop)
                )


@pytest.fixture(scope='function')
def hoomd_snapshot(lattice_snapshot_factory):
    snap = lattice_snapshot_factory(particle_types=['p1', 'p2'],
                                    n=10, a=2.0, r=0.01)
    positions = np.random.rand(len(snap.particles.mass), 3) * 2 - 1
    # positions *= 20
    velocities = np.random.rand(len(snap.particles.mass), 3) * 2 - 1
    accelerations = np.random.rand(len(snap.particles.mass), 3) * 2 - 1

    snap.particles.typeid[:] = np.random.randint(0, 2, len(snap.particles.mass))
    snap.particles.position[:] = positions
    snap.particles.velocity[:] = velocities
    snap.particles.acceleration[:] = accelerations
    snap.particles.mass[:] = np.random.rand(len(snap.particles.mass))
    snap.particles.charge[:] = np.random.rand(len(snap.particles.mass))
    snap.particles.diameter[:] = np.random.rand(len(snap.particles.mass))
    snap.particles.image[:] = np.random.randint(1, 100,
                                                (len(snap.particles.mass), 3))
    snap.particles.types = ['p1', 'p2']
    # print(np.random.randint(0, 2, len(snap.particles.mass)))
    # for i in np.random.randint(0, 2, len(snap.particles.mass)):
    #     s.particles.typeid[i] = particle_types.index(particle_type)

    # bonds
    snap.bonds.types = ['b1', 'b2']
    snap.bonds.N = 2
    snap.bonds.typeid[:] = [0, 1]
    snap.bonds.group[0] = [0, 1]
    snap.bonds.group[1] = [2, 3]

    # angles
    snap.angles.types = ['a1', 'a2']
    snap.angles.N = 2
    snap.angles.typeid[:] = [1, 0]
    snap.angles.group[0] = [0, 1, 2]
    snap.angles.group[1] = [2, 3, 0]

    # dihedrals
    snap.dihedrals.types = ['d1']
    snap.dihedrals.N = 1
    snap.dihedrals.typeid[:] = [0]
    snap.dihedrals.group[0] = [0, 1, 2, 3]

    # impropers
    snap.impropers.types = ['i1']
    snap.impropers.N = 1
    snap.impropers.typeid[:] = [0]
    snap.impropers.group[0] = [3, 2, 1, 0]

    # constraints
    snap.constraints.N = 1
    snap.constraints.group[0] = [0, 1]
    snap.constraints.value[0] = 2.5

    # special pairs
    snap.pairs.types = ['p1', 'p2']
    snap.pairs.N = 2
    snap.pairs.typeid[:] = [0, 1]
    snap.pairs.group[0] = [0, 1]
    snap.pairs.group[1] = [2, 3]
    return snap


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


# tests data.gsd_snapshot
def test_gsd_snapshot(hoomd_snapshot, device, tmp_path):
    sim = hoomd.Simulation(device)
    snap = hoomd_snapshot
    sim.create_state_from_snapshot(snap)
    d = tmp_path / "sub"
    d.mkdir()
    filename = d / "temporary_test_file.gsd"
    lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(), r_cut=2.5)
    lj.params[('p1', 'p1')] = {'sigma': 1, 'epsilon': 5e-200}
    lj.params[('p1', 'p2')] = {'sigma': 1, 'epsilon': 5e-200}
    lj.params[('p2', 'p2')] = {'sigma': 1, 'epsilon': 5e-200}
    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(lj)
    integrator.methods.append(hoomd.md.methods.Langevin(hoomd.filter.All(),
                                                        kT=1, seed=1))
    sim.operations.integrator = integrator
    trigger = hoomd.trigger.Periodic(1)
    gsd_dump = hoomd.dump.GSD(filename, trigger, overwrite=True)
    sim.operations.add(gsd_dump)
    sim.operations.schedule()
    sim.run(1)
    with gsd.hoomd.open(name=filename, mode='rb') as file:
        dumped_gsd_snap = file[0]
    undumped_gsd_snap = _make_gsd_snapshot(snap)

    def conditional(attr):
        if attr[0] != '_' and 'accel' not in attr and attr != 'validate':
            return True
        else:
            return False
    outer_attributes = ['angles', 'bonds', 'configuration',
                        'constraints', 'dihedrals',
                        'impropers', 'pairs', 'particles']
    for outer_attr in outer_attributes:
        snap1 = getattr(undumped_gsd_snap, outer_attr)
        snap2 = getattr(dumped_gsd_snap, outer_attr)
        for inner_attr in dir(snap1):
            if conditional(inner_attr):
                if inner_attr in ['types', 'N']:
                    assert getattr(snap1, inner_attr) == getattr(snap2,
                                                                 inner_attr)
                elif inner_attr == 'type_shapes':
                    assert getattr(snap1, inner_attr) is None
                    assert getattr(snap2, inner_attr) == [{}]
                elif inner_attr == 'step':
                    assert getattr(snap1, inner_attr) is None
                    assert getattr(snap2, inner_attr) == 0
                else:
                    att1 = getattr(snap1, inner_attr)
                    att2 = getattr(snap2, inner_attr)
                    np.testing.assert_allclose(att1,
                                               att2)
