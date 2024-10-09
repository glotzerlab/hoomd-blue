# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import pytest
import numpy as np

_harmonic_args = {'k': [30.0, 25.0, 20.0], 'r0': [1.6, 1.7, 1.8]}
_harmonic_arg_list = [(hoomd.md.mesh.bond.Harmonic,
                       dict(zip(_harmonic_args, val)))
                      for val in zip(*_harmonic_args.values())]

_FENE_args = {
    'k': [30.0, 25.0, 20.0],
    'r0': [1.6, 1.7, 1.8],
    'epsilon': [0.9, 1.0, 1.1],
    'sigma': [1.1, 1.0, 0.9],
    'delta': [0, 0, 0]
}
_FENE_arg_list = [(hoomd.md.mesh.bond.FENEWCA, dict(zip(_FENE_args, val)))
                  for val in zip(*_FENE_args.values())]

_Tether_args = {
    'k_b': [5.0, 6.0, 7.0],
    'l_min': [0.7, 0.8, 0.9],
    'l_c1': [0.9, 1.05, 1.1],
    'l_c0': [1.1, 1.1, 1.3],
    'l_max': [1.3, 1.3, 1.5]
}
_Tether_arg_list = [(hoomd.md.mesh.bond.Tether, dict(zip(_Tether_args, val)))
                    for val in zip(*_Tether_args.values())]

_Volume_args = {'k': [20.0, 50.0, 100.0], 'V0': [0.107227, 1, 0.01]}
_Volume_arg_list = [(hoomd.md.mesh.conservation.Volume,
                     dict(zip(_Volume_args, val)))
                    for val in zip(*_Volume_args.values())]
_BendingRigidity_args = {
    'k': [2.0, 10.0, 300.0],
}
_BendingRigidity_arg_list = [(hoomd.md.mesh.bending.BendingRigidity,
                              dict(zip(_BendingRigidity_args, val)))
                             for val in zip(*_BendingRigidity_args.values())]


def get_mesh_potential_and_args():
    return (_harmonic_arg_list + _FENE_arg_list + _Tether_arg_list
            + _Volume_arg_list + _BendingRigidity_arg_list)


def get_mesh_potential_args_forces_and_energies():
    harmonic_forces = [[[37.86, 0., -26.771063], [-37.86, 0., -26.771063],
                        [0., 37.86, 26.771063], [0., -37.86, 26.771063]],
                       [[36.55, 0., -25.844753], [-36.55, 0., -25.844753],
                        [0., 36.55, 25.844753], [0., -36.55, 25.844753]],
                       [[33.24, 0., -23.504229], [-33.24, 0., -23.504229],
                        [0., 33.24, 23.504229], [0., -33.24, 23.504229]]]
    harmonic_energies = [35.83449, 40.077075, 41.43366]
    FENE_forces = [[[221.113071, 0.,
                     -156.350552], [-221.113071, 0., -156.350552],
                    [0., 221.113071, 156.350552], [0., -221.113071,
                                                   156.350552]],
                   [[12.959825, 0., -9.16398], [-12.959825, 0., -9.16398],
                    [0., 12.959825, 9.16398], [0., -12.959825, 9.16398]],
                   [[-44.644347, 0., 31.568321], [44.644347, 0., 31.568321],
                    [0., -44.644347, -31.568321], [0., 44.644347, -31.568321]]]
    FENE_energies = [163.374213, 97.189301, 67.058202]
    Tether_forces = [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                     [[0.048888, 0., -0.034569], [-0.048888, 0., -0.034569],
                      [0., 0.048888, 0.034569], [0., -0.048888, 0.034569]],
                     [[7.144518, 0., -5.051937], [-7.144518, 0., -5.051937],
                      [0., 7.144518, 5.051937], [0., -7.144518, 5.051937]]]
    Tether_energies = [0, 0.000926, 0.294561]

    Volume_forces = [[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                     [[4.93960528, 0,
                       -3.49282839], [-4.93960528, 0, -3.49282839],
                      [0, 4.93960528, 3.49282839], [0, -4.93960528,
                                                    3.49282839]],
                     [[-107.5893328, 0, 76.0771468],
                      [107.5893328, 0, 76.0771468],
                      [0, -107.5893328, -76.0771468],
                      [0, 107.5893328, -76.0771468]]]
    Volume_energies = [0, 19.92608051621174, 47.2656702899458]
    BendingRigidity_forces = [[[0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
                               [0., 0., 0.]],
                              [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
                               [0., 0., 0.]],
                              [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.],
                               [0., 0., 0.]]]
    BendingRigidity_energies = [8, 40, 1200]

    harmonic_args_and_vals = []
    FENE_args_and_vals = []
    Tether_args_and_vals = []
    Volume_args_and_vals = []
    BendingRigidity_args_and_vals = []
    for i in range(3):
        harmonic_args_and_vals.append(
            (*_harmonic_arg_list[i], harmonic_forces[i], harmonic_energies[i]))
        FENE_args_and_vals.append(
            (*_FENE_arg_list[i], FENE_forces[i], FENE_energies[i]))
        Tether_args_and_vals.append(
            (*_Tether_arg_list[i], Tether_forces[i], Tether_energies[i]))
        Volume_args_and_vals.append(
            (*_Volume_arg_list[i], Volume_forces[i], Volume_energies[i]))
        BendingRigidity_args_and_vals.append(
            (*_BendingRigidity_arg_list[i], BendingRigidity_forces[i],
             BendingRigidity_energies[i]))
    return (harmonic_args_and_vals + FENE_args_and_vals + Tether_args_and_vals
            + Volume_args_and_vals + BendingRigidity_args_and_vals)


@pytest.fixture(scope='session')
def tetrahedron_snapshot_factory(device):

    def make_snapshot(d=1.0, particle_types=['A'], L=20):
        s = hoomd.Snapshot(device.communicator)
        N = 4
        if s.communicator.rank == 0:
            box = [L, L, L, 0, 0, 0]
            s.configuration.box = box
            s.particles.N = N

            base_positions = np.array([[1.0, 0.0, -1.0 / np.sqrt(2.0)],
                                       [-1.0, 0.0, -1.0 / np.sqrt(2.0)],
                                       [0.0, 1.0, 1.0 / np.sqrt(2.0)],
                                       [0.0, -1.0, 1.0 / np.sqrt(2.0)]])
            # move particles slightly in direction of MPI decomposition which
            # varies by simulation dimension
            s.particles.position[:] = 0.5 * d * base_positions
            s.particles.types = particle_types
        return s

    return make_snapshot


@pytest.mark.parametrize("mesh_potential_cls, potential_kwargs",
                         get_mesh_potential_and_args())
def test_before_attaching(mesh_potential_cls, potential_kwargs):
    mesh = hoomd.mesh.Mesh()
    mesh_potential = mesh_potential_cls(mesh)
    mesh_potential.params["mesh"] = potential_kwargs

    assert mesh is mesh_potential.mesh
    for key in potential_kwargs:
        np.testing.assert_allclose(mesh_potential.params["mesh"][key],
                                   potential_kwargs[key],
                                   rtol=1e-6)

    mesh1 = hoomd.mesh.Mesh()
    mesh_potential.mesh = mesh1
    assert mesh1 is mesh_potential.mesh


@pytest.mark.parametrize("mesh_potential_cls, potential_kwargs",
                         get_mesh_potential_and_args())
def test_after_attaching(tetrahedron_snapshot_factory, simulation_factory,
                         mesh_potential_cls, potential_kwargs):
    snap = tetrahedron_snapshot_factory(d=0.969, L=5)
    sim = simulation_factory(snap)

    mesh = hoomd.mesh.Mesh()
    type_ids = [0, 0, 0, 0]
    triangles = [[2, 1, 0], [0, 1, 3], [2, 0, 3], [1, 2, 3]]
    mesh.triangulation = dict(type_ids=type_ids, triangles=triangles)

    mesh_potential = mesh_potential_cls(mesh)
    mesh_potential.params["mesh"] = potential_kwargs

    integrator = hoomd.md.Integrator(dt=0.005)

    integrator.forces.append(mesh_potential)

    langevin = hoomd.md.methods.Langevin(kT=1,
                                         filter=hoomd.filter.All(),
                                         default_gamma=0.1)
    integrator.methods.append(langevin)
    sim.operations.integrator = integrator

    sim.run(0)
    for key in potential_kwargs:
        np.testing.assert_allclose(mesh_potential.params["mesh"][key],
                                   potential_kwargs[key],
                                   rtol=1e-6)

    mesh1 = hoomd.mesh.Mesh()
    with pytest.raises(RuntimeError):
        mesh_potential.mesh = mesh1


@pytest.mark.parametrize("mesh_potential_cls, potential_kwargs",
                         get_mesh_potential_and_args())
def test_multiple_types(tetrahedron_snapshot_factory, simulation_factory,
                        mesh_potential_cls, potential_kwargs):

    snap = tetrahedron_snapshot_factory(d=0.969, L=5)
    sim = simulation_factory(snap)

    mesh = hoomd.mesh.Mesh()
    mesh.types = ["mesh", "patch"]
    type_ids = [0, 0, 0, 1]
    triangles = [[2, 1, 0], [0, 1, 3], [2, 0, 3], [1, 2, 3]]
    mesh.triangulation = dict(type_ids=type_ids, triangles=triangles)

    mesh_potential = mesh_potential_cls(mesh)
    mesh_potential.params.default = potential_kwargs

    integrator = hoomd.md.Integrator(dt=0.005)

    integrator.forces.append(mesh_potential)

    langevin = hoomd.md.methods.Langevin(kT=1,
                                         filter=hoomd.filter.All(),
                                         default_gamma=0.1)
    integrator.methods.append(langevin)
    sim.operations.integrator = integrator

    sim.run(0)
    for key in potential_kwargs:
        np.testing.assert_allclose(mesh_potential.params["mesh"][key],
                                   potential_kwargs[key],
                                   rtol=1e-6)
        np.testing.assert_allclose(mesh_potential.params["patch"][key],
                                   potential_kwargs[key],
                                   rtol=1e-6)

    mesh1 = hoomd.mesh.Mesh()
    with pytest.raises(RuntimeError):
        mesh_potential.mesh = mesh1


@pytest.mark.parametrize("mesh_potential_cls, potential_kwargs, force, energy",
                         get_mesh_potential_args_forces_and_energies())
def test_forces_and_energies(tetrahedron_snapshot_factory, simulation_factory,
                             mesh_potential_cls, potential_kwargs, force,
                             energy):

    snap = tetrahedron_snapshot_factory(d=0.969, L=5)
    sim = simulation_factory(snap)

    mesh = hoomd.mesh.Mesh()
    mesh.types = ["mesh", "patch"]
    type_ids = [1, 1, 1, 1]
    triangles = [[2, 1, 0], [0, 1, 3], [2, 0, 3], [1, 2, 3]]
    mesh.triangulation = dict(type_ids=type_ids, triangles=triangles)

    mesh_potential = mesh_potential_cls(mesh)
    mesh_potential.params["mesh"] = potential_kwargs
    mesh_potential.params["patch"] = potential_kwargs

    integrator = hoomd.md.Integrator(dt=0.005)

    integrator.forces.append(mesh_potential)

    langevin = hoomd.md.methods.Langevin(kT=1,
                                         filter=hoomd.filter.All(),
                                         default_gamma=0.1)
    integrator.methods.append(langevin)
    sim.operations.integrator = integrator

    sim.run(0)

    sim_energies = sim.operations.integrator.forces[0].energies
    sim_forces = sim.operations.integrator.forces[0].forces
    if sim.device.communicator.rank == 0:
        np.testing.assert_allclose(sum(sim_energies),
                                   energy,
                                   rtol=1e-2,
                                   atol=1e-5)
        np.testing.assert_allclose(sim_forces, force, rtol=1e-2, atol=1e-5)


def test_volume(simulation_factory, tetrahedron_snapshot_factory):
    snap = tetrahedron_snapshot_factory(d=0.969, L=5)
    sim = simulation_factory(snap)

    mesh = hoomd.mesh.Mesh()
    mesh.types = ["mesh", "patch"]
    type_ids = [0, 0, 1, 0]
    triangles = [[2, 1, 0], [0, 1, 3], [2, 0, 3], [1, 2, 3]]
    mesh.triangulation = dict(type_ids=type_ids, triangles=triangles)

    mesh_potential = hoomd.md.mesh.conservation.Volume(mesh)
    mesh_potential.params.default = dict(k=1, V0=1)

    integrator = hoomd.md.Integrator(dt=0.005)

    integrator.forces.append(mesh_potential)

    langevin = hoomd.md.methods.Langevin(kT=1,
                                         filter=hoomd.filter.All(),
                                         default_gamma=0.1)
    integrator.methods.append(langevin)
    sim.operations.integrator = integrator

    sim.run(0)

    np.testing.assert_allclose(mesh_potential.volume, [0.08042, 0.026807],
                               rtol=1e-2,
                               atol=1e-5)


def test_volume_ignore_type(simulation_factory, tetrahedron_snapshot_factory):
    snap = tetrahedron_snapshot_factory(d=0.969, L=5)
    sim = simulation_factory(snap)

    mesh = hoomd.mesh.Mesh()
    mesh.types = ["mesh", "patch"]
    type_ids = [0, 0, 1, 0]
    triangles = [[2, 1, 0], [0, 1, 3], [2, 0, 3], [1, 2, 3]]
    mesh.triangulation = dict(type_ids=type_ids, triangles=triangles)

    mesh_potential = hoomd.md.mesh.conservation.Volume(mesh, ignore_type=True)
    mesh_potential.params.default = dict(k=1, V0=1)

    integrator = hoomd.md.Integrator(dt=0.005)

    integrator.forces.append(mesh_potential)

    langevin = hoomd.md.methods.Langevin(kT=1,
                                         filter=hoomd.filter.All(),
                                         default_gamma=0.1)
    integrator.methods.append(langevin)
    sim.operations.integrator = integrator

    sim.run(0)

    np.testing.assert_allclose(mesh_potential.volume, [0.107227, 0.0],
                               rtol=1e-2,
                               atol=1e-5)


def test_auto_detach_simulation(simulation_factory,
                                tetrahedron_snapshot_factory):
    sim = simulation_factory(tetrahedron_snapshot_factory(d=0.969, L=5))
    mesh = hoomd.mesh.Mesh()
    mesh.triangulation = dict(type_ids=[0, 0], triangles=[[0, 1, 2], [0, 2, 3]])

    harmonic = hoomd.md.mesh.bond.Harmonic(mesh)
    harmonic.params["mesh"] = dict(k=1, r0=1)

    harmonic_2 = hoomd.md.mesh.bond.Harmonic(mesh)
    harmonic_2.params["mesh"] = dict(k=5, r0=1.1)

    integrator = hoomd.md.Integrator(dt=0.005, forces=[harmonic, harmonic_2])

    integrator.methods.append(
        hoomd.md.methods.Langevin(kT=1, filter=hoomd.filter.All()))
    sim.operations.integrator = integrator

    sim.run(0)
    del integrator.forces[1]
    assert mesh._attached
    assert hasattr(mesh, "_cpp_obj")
    del integrator.forces[0]
    assert not mesh._attached
    assert mesh._cpp_obj is None
