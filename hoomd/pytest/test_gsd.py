from hoomd import *
import hoomd
import numpy as np
import pytest
import os
import tempfile
import gsd.hoomd
from copy import deepcopy
from itertools import combinations


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

defaults = {}
non_defaults = {}

defaults['angmom'] = [[0, 0, 0, 0]] * 8
non_defaults['angmom'] = np.random.randint(0, 5, (8, 4)).tolist()

defaults['charge'] = [0] * 8
non_defaults['charge'] = np.random.randint(-3, 3, 8).tolist()

defaults['diameter'] = [1] * 8
non_defaults['diameter'] = np.random.randint(0, 4, 8).tolist()

defaults['image'] = [[0, 0, 0]] * 8
non_defaults['image'] = np.random.randint(0, 5, (8, 3)).tolist()

defaults['mass'] = [1] * 8
non_defaults['mass'] = np.random.randint(1, 4, 8).tolist()

defaults['moment_inertia'] = [[0, 0, 0]] * 8
non_defaults['moment_inertia'] = np.random.randint(0, 5, (8, 3)).tolist()

defaults['orientation'] = [[1, 0, 0, 0]] * 8
non_defaults['orientation'] = np.random.randint(0, 4, (8, 4)).tolist()
non_defaults['orientation'] /= np.linalg.norm(non_defaults['orientation'],
                                              axis=1).reshape(8, 1)

defaults['typeid'] = [0] * 8
non_defaults['typeid'] = np.random.randint(0, 3, 8).tolist()

defaults['velocity'] = [[0, 0, 0]] * 8
non_defaults['velocity'] = np.random.randint(0, 4, (8, 3)).tolist()

_default_properties = []
for key in defaults.keys():
    _default_properties.append((key, defaults[key], non_defaults[key]))


@pytest.fixture(scope="function",
                params=_default_properties,
                ids=(lambda x: x[0]))
def default_properties(request):
    return deepcopy(request.param)


def test_default_vals(default_properties, lattice_snapshot_factory,
                      simulation_factory, tmp_path):
    prop, defaults, non_defaults = default_properties
    particle_types = ['A', 'B', 'C']
    snap = lattice_snapshot_factory(particle_types=particle_types,
                                    n=2, a=4.0, r=0.01)
    sim = simulation_factory(snap)

    d = tmp_path / "sub"
    d.mkdir()
    filename = d / "temporary_test_file.gsd"
    lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(), r_cut=2.5)
    lj_args = {'sigma': 1, 'epsilon': 5e-200}
    for t in particle_types:
        lj.params[(t, t)] = lj_args
    for combo in combinations(particle_types, 2):
        lj.params[combo] = lj_args
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
        gsd_snap = file[0]
    np.testing.assert_allclose(getattr(gsd_snap.particles, prop), defaults)

    sim = hoomd.Simulation(device)
    particle_types = ['A', 'B', 'C']
    snap = lattice_snapshot_factory(particle_types=particle_types,
                                    n=2, a=4.0, r=0.01)
    getattr(snap.particles, prop)[:] = non_defaults
    sim = simulation_factory(snap)
    lj = hoomd.md.pair.LJ(nlist=hoomd.md.nlist.Cell(), r_cut=2.5)
    lj_args = {'sigma': 1, 'epsilon': 5e-200}
    for t in particle_types:
        lj.params[(t, t)] = lj_args
    for combo in combinations(particle_types, 2):
        lj.params[combo] = lj_args
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
        gsd_snap = file[0]
    np.testing.assert_allclose(getattr(gsd_snap.particles, prop), non_defaults)
