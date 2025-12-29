# Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
import pytest
import numpy as np

def get_mesh_forces_and_energies():
    forces = [
        [
            [0.0, 0.0, -758.6739957/3],
            [0.0, 0.0, -758.6739957/3],
            [0.0, 0.0, -758.6739957/3],
            [0.0, 0.0,  758.6739957],
        ],
        [
            [-287.99962387/2, -498.83/2, -4032.00288006/2],
            [-287.99962387/2, -498.83/2, -4032.00288006/2],
            [0.0, 0.0, 0.0],
            [287.99962387, 498.83, 4032.00288006],
        ],
        [
            [-287.99962387/4, 498.83/4, -4032.00288006/4],
            [0.0, 0.0, 0.0],
            [-3*287.99962387/4, 3*498.83/4, -3*4032.00288006/4],
            [287.99962387, -498.83, 4032.00288006],
        ],
        [
            [0.0, 0.0, 0.0],
            [ 12*np.sqrt(0.5), 0.0, -12*np.sqrt(0.5)],
            [ 12*np.sqrt(0.5), 0.0, -12*np.sqrt(0.5)],
            [-24*np.sqrt(0.5), 0.0,  24*np.sqrt(0.5)],
        ],
        [
            [ -24*np.sqrt(0.5), 0.0, -24*np.sqrt(0.5)],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [24*np.sqrt(0.5), 0.0,  24*np.sqrt(0.5)],
        ],
        [
            [0.0, 0.0, 0.0],
            [287.99962387, -498.83, 4032.00288006],
            [0.0, 0.0, 0.0],
            [-287.99962387, 498.83, -4032.00288006],
        ],
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [379.336998, 657.030953, 0.0],
            [-379.336998, -657.030953, 0.0],
        ],
    ]
    energies = [42.94887,224,224,0,0,224,42.94887]

    height = [0.8,0.7,0.7,np.sqrt(0.5),np.sqrt(0.5),-0.7,0.0]

    angle = [0,np.pi/3,5*np.pi/3+0.6666,np.pi, 0,2*np.pi/3,4*np.pi/3]
    
    radius = [0,1.1,1.4,1+np.sqrt(0.5),2+np.sqrt(0.5),2.1,2.8]

    args_and_vals = []

    for i in range(7):
        args_and_vals.append(
            (forces[i], energies[i], height[i], angle[i], radius[i])
        )
    return args_and_vals
    

@pytest.fixture(scope="session")
def triangle_particle_snapshot_factory(device):
    def make_snapshot(d=1.0, h = 1, c=0, a=0, particle_types=["A"], L=20):
        s = hoomd.Snapshot(device.communicator)
        N = 4
        if s.communicator.rank == 0:
            box = [L, L, L, 0, 0, 0]
            s.configuration.box = box
            s.particles.N = N

            base_positions = np.array(
                [
                    [d, 0.0, 0.0],
                    [d*np.cos(2/3*np.pi), d*np.sin(2/3*np.pi), 0.0],
                    [d*np.cos(4/3*np.pi), d*np.sin(4/3*np.pi), 0.0],
                    [c*np.cos(a), c*np.sin(a), h],
                ]
            )
            # move particles slightly in direction of MPI decomposition which
            # varies by simulation dimension
            s.particles.position[:] = base_positions
            s.particles.types = particle_types
        return s

    return make_snapshot

def test_before_attaching():
    mesh = hoomd.mesh.Mesh()
    tree = hoomd.md.nlist.Cell(buffer=0.5)

    tlj = hoomd.md.mesh.triangle_pair.WCA(nlist=tree,mesh=mesh,default_r_cut=2**(1/6),default_nlist_r_cut = 5)


    assert tlj.mesh == mesh
    assert tlj.nlist == tree
    assert tlj.r_cut.default == 2**(1/6)
    assert tlj.nlist_r_cut.default == 5

    tlj.params['A'] = dict(epsilon=2, sigma=1)
    tlj.r_cut['A'] = 0
    tlj.r_cut['A'] = 2**(1/6)

    tlj.nlist_r_cut[('A','A')] = 0
    tlj.nlist_r_cut[('A','A')] = 6

    assert tlj.params['A']['epsilon'] == 2
    assert tlj.params['A']['sigma'] == 1
    assert tlj.r_cut["A"] == 2**(1/6)
    assert tlj.nlist_r_cut[("A","A")] == 6

def test_after_attaching(
    triangle_particle_snapshot_factory,
    simulation_factory,
):
    snap = triangle_particle_snapshot_factory(d=2, L=20)
    sim = simulation_factory(snap)

    mesh = hoomd.mesh.Mesh()
    type_ids = [0]
    triangles = [[2, 1, 0]]
    mesh.triangulation = dict(type_ids=type_ids, triangles=triangles)
    tree = hoomd.md.nlist.Tree(buffer=0.5,mesh=mesh,exclusions=("meshbond",))

    tlj = hoomd.md.mesh.triangle_pair.WCA(nlist=tree,mesh=mesh,default_r_cut=0,default_nlist_r_cut = 5)
    tlj.params['A'] = dict(epsilon=2, sigma=1)

    integrator = hoomd.md.Integrator(dt=0.005)

    integrator.forces.append(tlj)

    langevin = hoomd.md.methods.Langevin(
        kT=1, filter=hoomd.filter.All(), default_gamma=0.1
    )

    integrator.methods.append(langevin)
    sim.operations.integrator = integrator

    sim.run(0)

    assert tlj.mesh == mesh
    assert tlj.nlist == tree
    assert tlj.r_cut.default == 0
    assert tlj.nlist_r_cut.default == 5
    assert tlj.params['A']['epsilon'] == 2
    assert tlj.params['A']['sigma'] == 1

    tlj.params['A'] = dict(epsilon=3, sigma=2)
    tlj.r_cut['A'] = 2**(1/6)
    tlj.nlist_r_cut[('A','A')] = 6

    assert tlj.params['A']['epsilon'] == 3
    assert tlj.params['A']['sigma'] == 2
    assert tlj.r_cut["A"] == 2**(1/6)
    assert tlj.nlist_r_cut[("A","A")] == 6

    mesh1 = hoomd.mesh.Mesh()
    with pytest.raises(RuntimeError):
        tlj.mesh = mesh1

    tree1 = hoomd.md.nlist.Tree(buffer=0.5,mesh=mesh1,exclusions=("meshbond",))
    with pytest.raises(RuntimeError):
        tlj.nlist = tree1


@pytest.mark.parametrize(
    "force, energy, height, angle, radius",
    get_mesh_forces_and_energies(),
)
def test_forces_and_energies(
    triangle_particle_snapshot_factory,
    simulation_factory,
    force,
    energy,
    height,
    angle,
    radius,
):
    snap = triangle_particle_snapshot_factory(d=2, h=height, a=angle, c=radius, L=20)
    sim = simulation_factory(snap)

    mesh = hoomd.mesh.Mesh()
    type_ids = [0]
    triangles = [[2, 1, 0]]
    mesh.triangulation = dict(type_ids=type_ids, triangles=triangles)
    tree = hoomd.md.nlist.Tree(buffer=0.5,mesh=mesh,exclusions=("meshbond",))

    tlj = hoomd.md.mesh.triangle_pair.WCA(nlist=tree,mesh=mesh,default_r_cut=2**(1/6),default_nlist_r_cut = 5)
    tlj.params['A'] = dict(epsilon=1, sigma=1)

    integrator = hoomd.md.Integrator(dt=0.005)

    integrator.forces.append(tlj)

    langevin = hoomd.md.methods.Langevin(
        kT=1, filter=hoomd.filter.All(), default_gamma=0.1
    )

    integrator.methods.append(langevin)
    sim.operations.integrator = integrator

    sim.run(0)

    if sim.device.communicator.rank == 0:
        assert sum(tlj.energies) ==  pytest.approx(energy, rel=1e-2, abs=1e-5)
        np.testing.assert_allclose(tlj.forces, force, rtol=1e-2, atol=1e-5)
