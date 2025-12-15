#Copyright (c) 2009-2025 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
from hoomd.mesh3D import Mesh3D
import pytest
import numpy
from hoomd.snapshot import Snapshot
from pathlib import Path
import gsd.hoomd
import gsd


def test_elastic(device, simulation_factory):
    reference_positions = numpy.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
    )
    positions = reference_positions.copy()
    tetrahedra = numpy.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    types = ["A", "B"]
    type_ids = [0, 1]
    reference_tags = [0, 1, 2, 3, 4]

    # mesh = Mesh3D(reference_positions, tetrahedra, types, type_ids, reference_tags)
    mesh = Mesh3D(reference_positions, tetrahedra, types, type_ids)

    snapshot = Snapshot(device.communicator)
    if snapshot.communicator.rank == 0:
        snapshot.particles.N = len(positions)
        snapshot.particles.position[:] = positions
        snapshot.particles.types = ["A"]
        snapshot.configuration.box = [10, 10, 10, 0, 0, 0]
    sim = simulation_factory(snapshot)

    elastic = hoomd.md.mesh.Elastic(mesh, reference_tags)
    elastic.params["A"] = {"C_xxxx": 1.0, "C_xxyy": 0.5, "C_xyxy": 0.25}
    elastic.params["B"] = {"C_xxxx": 2.0, "C_xxyy": 1.0, "C_xyxy": 0.5}

    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(elastic)

    sim.operations.integrator = integrator
    sim.run(0)

    assert elastic.energy == pytest.approx(0, abs=1e-4)


def test_sphere_stretch(device, simulation_factory):
    path = Path(__file__).parent / "sphere.mesh"

    with open(path, "r") as f:
        mesh_data = f.read()
    mesh_list = mesh_data.split("\n")
    verts_start = mesh_list.index("vertices")
    verts = []
    i = 2
    while True:
        if mesh_list[verts_start + i] == "":
            break
        else:
            verts.append(mesh_list[verts_start + i])
        i += 1
    # converting the list of strings to position node data
    node_data = numpy.zeros((len(verts), 3))
    for i, vert in enumerate(verts):
        node_data[i, :] = numpy.fromstring(vert, dtype=float, sep=" ")[1:]

    tets_start = mesh_list.index("volumes")
    tets = []
    i = 2
    while True:
        if mesh_list[tets_start + i] == "":
            break
        else:
            tets.append(mesh_list[tets_start + i])
        i += 1
    elem_data = numpy.empty((len(tets), 4), dtype=int)
    for i, elem in enumerate(tets):
        elem_data[i, :] = numpy.fromstring(elem, dtype=int, sep=" ")[1:]

    elem_data -= 1

    p = numpy.copy(node_data)
    pref = numpy.copy(node_data)

    # stretching the colloid slightly by ~5%
    stretch = 1.2
    p[:, 0] = p[:, 0] * stretch
    p[:, 1] = p[:, 1] * 1.0 / numpy.sqrt(stretch)
    p[:, 2] = p[:, 2] * 1.0 / numpy.sqrt(stretch)

    types = ["A"]
    type_ids = [0] * len(elem_data)
    reference_tags = numpy.arange(len(p))

    mesh = Mesh3D(pref, elem_data, types, type_ids)
    snapshot = Snapshot(device.communicator)
    if snapshot.communicator.rank == 0:
        snapshot.particles.N = len(p)
        snapshot.particles.position[:] = p
        snapshot.particles.types = ["A"]
        snapshot.configuration.box = [20, 20, 20, 0, 0, 0]
    sim = simulation_factory(snapshot)

    elastic = hoomd.md.mesh.Elastic(mesh, reference_tags)
    elastic.params["A"] = {"C_xxxx": 29.1, "C_xxyy": 28.0, "C_xyxy": 0.57}

    integrator = hoomd.md.Integrator(dt=0.00005)

    filter_all = hoomd.filter.All()
    nve = hoomd.md.methods.ConstantVolume(filter=filter_all)
    integrator.forces.append(elastic)
    integrator.methods.append(nve)

    sim.operations.integrator = integrator

    sim.run(0)

    elastic.energy > 0

# test across periodic boundaries
def test_periodic_boundaries(device, simulation_factory):
    reference_positions = numpy.array(
        [[0., 0, 0.5/(numpy.sqrt(3))], [1, 1/2, 0], [1, -1/2, 0], [1, 0, 1]]
    )
    box_L = 10
    stretch = 1
    positions0 = reference_positions.copy()
    positions0[0] -= numpy.array([stretch,0.0,0.0])

    positions1 = positions0.copy() - numpy.array([1.1*box_L/2,0,0])
    positions1[0] += numpy.array([box_L,0.0,0.0])

    positions2 = positions1.copy() + numpy.array([0,0,box_L/2])
    positions2[3] -= numpy.array([0.0, 0.0, box_L])
    positions2[0] -= numpy.array([0.0, 0.0, box_L])

    tetrahedra = numpy.array([[0, 1, 2, 3],[4, 5, 6, 7],[8,9,10,11]]) 
    types = ["A"]
    type_ids = [0]
    reference_tags = [[0, 1, 2, 3], [4, 5, 6, 7], [8,9,10,11]]

    # For comparison
    mesh0 = Mesh3D(reference_positions, tetrahedra[0], types, type_ids)
    # Across one boundary:
    mesh1 = Mesh3D(reference_positions, tetrahedra[1], types, type_ids)
    # Across edge boundary:
    mesh2 = Mesh3D(reference_positions, tetrahedra[2], types, type_ids)

    snapshot = Snapshot(device.communicator)
    if snapshot.communicator.rank == 0:
        snapshot.particles.N = len(positions0)*3
        snapshot.particles.position[:] = list(positions0) + list(positions1) + list(positions2)
        snapshot.particles.types = types
        snapshot.configuration.box = [box_L, box_L, box_L, 0, 0, 0]
    sim = simulation_factory(snapshot)

    elastic0 = hoomd.md.mesh.Elastic(mesh0, reference_tags[0])
    elastic1 = hoomd.md.mesh.Elastic(mesh1, reference_tags[1])
    elastic2 = hoomd.md.mesh.Elastic(mesh2, reference_tags[2])
    elastic0.params["A"] = {"C_xxxx": 0.1, "C_xxyy": 0.05, "C_xyxy": 0.025}
    elastic1.params["A"] = {"C_xxxx": 0.1, "C_xxyy": 0.05, "C_xyxy": 0.025}
    elastic2.params["A"] = {"C_xxxx": 0.1, "C_xxyy": 0.05, "C_xyxy": 0.025}
    
    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.extend([elastic0, elastic1, elastic2])

    filter_all = hoomd.filter.All() 
    nve = hoomd.md.methods.ConstantVolume(filter=filter_all)
    integrator.methods.append(nve)
    sim.operations.integrator = integrator
    
    sim.run(200)
    
    assert elastic0.energy == pytest.approx(elastic1.energy, abs=1e-4) and elastic0.energy == pytest.approx(elastic2.energy, abs=1e-4)

# particles not in mesh
def test_other_particles_and_potentials(device, simulation_factory):
    reference_positions = numpy.array(
        [[0., 0, 0.5/(numpy.sqrt(3))], [1, 1/2, 0], [1, -1/2, 0], [1, 0, 1]]
    )
    p_pos = [-1, 0, 0.5/(numpy.sqrt(3))]
    positions = reference_positions.copy()
    tetrahedra = numpy.array([0, 1, 2, 3])
    mesh_reference_tags = [0, 1, 2, 3]

    mesh = Mesh3D(reference_positions, tetrahedra, ["mesh"], [0])

    snapshot = Snapshot(device.communicator)
    if snapshot.communicator.rank == 0:
        snapshot.particles.N = len(positions) + 1
        snapshot.particles.position[:] =numpy.vstack((positions,p_pos))
        snapshot.particles.types = ["mesh","particle"]
        snapshot.particles.typeid[:] = [0]*4 +[1]
        snapshot.configuration.box = [10, 10, 10, 0, 0, 0]
    sim = simulation_factory(snapshot)

    elastic = hoomd.md.mesh.Elastic(mesh, mesh_reference_tags)
    elastic.params["mesh"] = {"C_xxxx": 1.0, "C_xxyy": 0.5, "C_xyxy": 0.25}

    cell = hoomd.md.nlist.Cell(buffer=0.5)
    LJ = hoomd.md.pair.LJ(nlist=cell,default_r_cut=0)
    LJ.params.default = dict(epsilon=0, sigma=1)
    LJ.params[('mesh','particle')] = dict(epsilon=1, sigma=1)
    LJ.r_cut[('mesh','particle')] = 3

    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.extend([elastic,LJ])

    sim.operations.integrator = integrator
    filter_all = hoomd.filter.All() 
    nve = hoomd.md.methods.ConstantVolume(filter=filter_all)
    integrator.methods.append(nve)
    sim.operations.integrator = integrator
    

    sim.run(500)

    assert elastic.energy > 0 #pytest.approx(0, abs=1e-4)


