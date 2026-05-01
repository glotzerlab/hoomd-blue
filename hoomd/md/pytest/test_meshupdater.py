# Copyright (c) 2009-2026 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import pytest
import numpy as np


def test_before_attaching():
    mesh = hoomd.mesh.Mesh()
    mesh_potential1 = hoomd.md.mesh.bond.Harmonic(mesh)
    mesh_potential1.params.default = dict(k=100, r0=1)

    mesh_potential2 = hoomd.md.mesh.conservation.TriangleArea(mesh)
    mesh_potential2.params.default = dict(k=100, A0=1)

    mdb = hoomd.md.update.MeshDynamicalBonding(10, mesh, 1, forces=[mesh_potential1])

    assert mdb.trigger == hoomd.trigger.Periodic(10)
    assert mdb.kT == 1
    assert mdb.mesh == mesh
    assert mdb.forces == [mesh_potential1]

    mdb.trigger = hoomd.trigger.Periodic(100)
    mdb.kT = 0.5
    mdb.forces.append(mesh_potential2)

    tree = hoomd.md.nlist.Cell(buffer=0.5)

    lj = hoomd.md.pair.LJ(tree, default_r_cut=2)
    lj.params.default = dict(epsilon=1, sigma=1)

    with pytest.raises(ValueError):
        mdb.forces.append(lj)

    assert mdb.trigger == hoomd.trigger.Periodic(100)
    assert mdb.kT == 0.5
    assert mdb.forces == [mesh_potential1, mesh_potential2]


@pytest.fixture(scope="session")
def local_snapshot_factory(device):
    def make_snapshot(d=0.0, particle_types=["A"], L=20):
        s = hoomd.Snapshot(device.communicator)
        N = 64
        if s.communicator.rank == 0:
            Lx = 8
            Ly = 4 * np.sqrt(3)
            box = [Lx, Ly, L, 0, 0, 0]
            s.configuration.box = box
            s.particles.N = N

            x = np.arange(-Lx / 2, Lx / 2, Lx / 8)
            y = np.arange(-Ly / 2, Ly / 2, Ly / 8)
            X, Y = np.meshgrid(x, y)

            # Offset every second row to create a triangular pattern
            X[1::2, :] += 0.5

            # Combine X and Y coordinates to define vertex positions
            base_positions = np.column_stack([X.flatten(), Y.flatten(), np.zeros(N)])

            base_positions[12, 2] = d
            base_positions[13, 2] = d

            # move particles slightly in direction of MPI decomposition which
            # varies by simulation dimension
            s.particles.position[:] = base_positions
            s.particles.types = particle_types
        return s

    return make_snapshot


def make_mesh(flip=True):
    row = 8
    N = row**2

    v_index = np.arange(N)
    vr_index = v_index + 1
    vr_index[row - 1 :: row] -= row

    e_index = np.repeat(v_index, 2).reshape(-1, 2)
    e_index[:, 0] += row
    e_index[:, 1] -= row
    e_index = e_index.reshape(row, -1, 2)
    e_index[1::2] += 1
    e_index[1::2, -1] -= row
    e_index = e_index.reshape(-1) % N

    v_index = np.repeat(v_index, 2)
    vr_index = np.repeat(vr_index, 2)
    vr_index[1::2] = e_index[1::2]
    e_index[1::2] = vr_index[0::2]

    triangles = np.column_stack([v_index, vr_index, e_index])

    if flip:
        triangles[24] = [12, 5, 21]
        triangles[25] = [13, 21, 5]
    else:
        triangles[24] = [5, 13, 12]
        triangles[25] = [21, 12, 13]

    return triangles


def test_after_attaching(local_snapshot_factory, simulation_factory):
    snap = local_snapshot_factory(d=0, L=20)
    sim = simulation_factory(snap)

    mesh = hoomd.mesh.Mesh()

    mesh_potential1 = hoomd.md.mesh.bond.Harmonic(mesh)
    mesh_potential1.params.default = dict(k=100, r0=1)

    mesh_potential2 = hoomd.md.mesh.conservation.TriangleArea(mesh)
    mesh_potential2.params.default = dict(k=100, A0=1)

    mdb = hoomd.md.update.MeshDynamicalBonding(10, mesh, 1, forces=[mesh_potential1])
    sim.operations.updaters.append(mdb)

    sim.operations.integrator = hoomd.md.Integrator(dt=0.005)

    if sim.device.communicator.num_ranks > 1:
        with pytest.raises(NotImplementedError):
            sim.run(0)
    else:
        sim.run(0)

        assert mdb.trigger == hoomd.trigger.Periodic(10)
        assert mdb.kT == 1
        assert mdb.mesh == mesh
        assert mdb.forces == [mesh_potential1]

        mdb.trigger = hoomd.trigger.Periodic(100)
        mdb.kT = 0.5
        mdb.forces.append(mesh_potential2)

        with pytest.raises(hoomd.error.MutabilityError):
            mdb.mesh = hoomd.mesh.Mesh()

        assert mdb.trigger == hoomd.trigger.Periodic(100)
        assert mdb.kT == 0.5
        assert mdb.forces == [mesh_potential1, mesh_potential2]


def test_updating(local_snapshot_factory, simulation_factory):
    snap = local_snapshot_factory(d=0.0, L=20)
    sim = simulation_factory(snap)

    if sim.device.communicator.num_ranks > 1:
        pytest.skip("Cannot run MeshDynamicalBonding with MPI")

    mesh_triangle = make_mesh()

    mesh = hoomd.mesh.Mesh()
    mesh.types = ["mesh"]
    mesh.triangulation = dict(
        type_ids=[0] * len(mesh_triangle), triangles=mesh_triangle
    )

    mesh_potential = hoomd.md.mesh.bond.Harmonic(mesh)
    mesh_potential.params.default = dict(k=1, r0=1)

    integrator = hoomd.md.Integrator(dt=0.01)
    integrator.forces.append(mesh_potential)
    sim.operations.integrator = integrator

    sim.run(0)

    assert not np.isclose(mesh_potential.energy, 0)

    mdb = hoomd.md.update.MeshDynamicalBonding(
        hoomd.trigger.Periodic(1), mesh, kT=0.001, forces=[mesh_potential]
    )

    sim.operations += mdb

    sim.run(1)

    mesh_triangle = make_mesh(False)

    assert np.isclose(mesh_potential.energy, 0)

    assert np.array_equal(mesh.triangles, make_mesh(False))


_harmonic_arg_list = [(hoomd.md.mesh.bond.Harmonic, dict(k=30.0, r0=1.0))]
_FENE_arg_list = [
    (
        hoomd.md.mesh.bond.FENEWCA,
        dict(k=30.0, r0=2.0, epsilon=1.0, sigma=1.0, delta=0.0),
    )
]
_Tether_arg_list = [
    (hoomd.md.mesh.bond.Tether, dict(k_b=10, l_min=0.5, l_c1=0.9, l_c0=1.1, l_max=2.0))
]
_BendingRigidity_arg_list = [(hoomd.md.mesh.bending.BendingRigidity, dict(k=300.0))]
_Helfrich_arg_list = [(hoomd.md.mesh.bending.Helfrich, dict(k=100.0))]
_AreaConservation_arg_list = [
    (hoomd.md.mesh.conservation.Area, dict(k=100.0, A0=8 * np.sqrt(3)))
]
_TriangleAreaConservation_arg_list = [
    (hoomd.md.mesh.conservation.TriangleArea, dict(k=100.0, A0=np.sqrt(3) / 4))
]


def get_mesh_potential_and_args():
    return (
        _harmonic_arg_list
        + _FENE_arg_list
        + _Tether_arg_list
        + _AreaConservation_arg_list
        + _TriangleAreaConservation_arg_list
        + _BendingRigidity_arg_list
        + _Helfrich_arg_list
    )


@pytest.mark.parametrize(
    "mesh_potential_cls, potential_kwargs", get_mesh_potential_and_args()
)
def test_reduce_energy(
    local_snapshot_factory, simulation_factory, mesh_potential_cls, potential_kwargs
):
    snap = local_snapshot_factory(d=0.5, L=20)
    sim = simulation_factory(snap)

    if sim.device.communicator.num_ranks > 1:
        pytest.skip("Cannot run MeshDynamicalBonding with MPI")

    mesh_triangle = make_mesh()

    mesh = hoomd.mesh.Mesh()
    mesh.types = ["mesh"]
    mesh.triangulation = dict(
        type_ids=[0] * len(mesh_triangle), triangles=mesh_triangle
    )

    mesh_potential = mesh_potential_cls(mesh)
    mesh_potential.params["mesh"] = potential_kwargs

    integrator = hoomd.md.Integrator(dt=0.01)
    integrator.forces.append(mesh_potential)
    sim.operations.integrator = integrator

    sim.run(0)

    energy = mesh_potential.energy

    mdb = hoomd.md.update.MeshDynamicalBonding(
        hoomd.trigger.Periodic(1), mesh, kT=0.001, forces=[mesh_potential]
    )

    sim.operations += mdb

    sim.run(1)

    assert mesh_potential.energy <= energy


def test_pickling(local_snapshot_factory, simulation_factory):
    # don't add the rd_updater since operation_pickling_check will deal with
    # that.
    snap = local_snapshot_factory(d=0, L=20)
    sim = simulation_factory(snap)

    if sim.device.communicator.num_ranks > 1:
        pytest.skip("Cannot run MeshDynamicalBonding with MPI")

    mesh = hoomd.mesh.Mesh()

    mdb = hoomd.md.update.MeshDynamicalBonding(1, mesh, kT=0.001)

    hoomd.conftest.operation_pickling_check(mdb, sim)
