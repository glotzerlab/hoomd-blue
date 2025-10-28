import hoomd
from hoomd.mesh3D import Mesh3D
import pytest
import numpy
from hoomd.snapshot import Snapshot
from hoomd import Simulation
from pathlib import Path

def test_elastic(device, simulation_factory):

    reference_positions = numpy.array([[0, 0, 0],
                             [1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1],
                             [1, 1, 1]])
    positions = reference_positions.copy()
    tetrahedra = numpy.array([[0, 1, 2, 3],
                              [1, 2, 3, 4]])
    types = ["A", "B"]
    type_ids = [0, 1]
    reference_tags = [0,1,2,3,4]

    #mesh = Mesh3D(reference_positions, tetrahedra, types, type_ids, reference_tags)  
    mesh = Mesh3D(reference_positions, tetrahedra, types, type_ids)  

    snapshot = Snapshot(device.communicator)
    if snapshot.communicator.rank == 0:
        snapshot.particles.N = len(positions)
        snapshot.particles.position[:] = positions
        snapshot.particles.types = ["A"]
        snapshot.configuration.box = [10,10,10,0,0,0]
    sim = simulation_factory(snapshot)

    elastic = hoomd.md.mesh.Elastic(mesh, reference_tags)
    elastic.params["A"] = {"C_xxxx": 1.0, "C_xxyy": 1.0, "C_xyxy": 1.0}
    elastic.params["B"] = {"C_xxxx": 2.0, "C_xxyy": 2.0, "C_xyxy": 2.0}

    integrator = hoomd.md.Integrator(dt=0.005)
    integrator.forces.append(elastic)

    sim.operations.integrator = integrator
    sim.run(0)

    assert elastic.energy == pytest.approx(0, abs = 1e-4)


def test_sphere_stretch(device, simulation_factory):
    path = Path(__file__).parent / "sphere.mesh" 

    with open(path,"r") as f:
        mesh_data = f.read()
    mesh_list = mesh_data.split("\n")
    verts_start = mesh_list.index("vertices")
    verts = []
    i = 2
    while True:
        if mesh_list[verts_start + i] == '':
            break
        else:
            verts.append(mesh_list[verts_start + i])
        i += 1
    # converting the list of strings to position node data
    node_data = numpy.zeros((len(verts),3))
    for i,vert in enumerate(verts):
        node_data[i,:] = numpy.fromstring(vert,dtype=float,sep=' ')[1:]

    tets_start = mesh_list.index("volumes")
    tets = []
    i = 2
    while True:
        if mesh_list[tets_start + i] == '':
            break
        else:
            tets.append(mesh_list[tets_start + i])
        i += 1
    elem_data = numpy.empty((len(tets),4),dtype=int)
    for i,elem in enumerate(tets):
        elem_data[i,:] = numpy.fromstring(elem,dtype=int,sep=' ')[1:]
    
    elem_data -= 1

    p = numpy.copy(node_data)
    pref = numpy.copy(node_data)

    # stretching the colloid slightly by ~5%
    stretch = 1.2
    p[:,0] = p[:,0]*stretch
    p[:,1] = p[:,1]*1.0/numpy.sqrt(stretch)
    p[:,2] = p[:,2]*1.0/numpy.sqrt(stretch)

    types = ["A"]
    type_ids = [0]*len(elem_data)
    reference_tags = numpy.arange(len(p))

    mesh = Mesh3D(pref, elem_data, types, type_ids)
    snapshot = Snapshot(device.communicator)
    if snapshot.communicator.rank == 0:
        snapshot.particles.N = len(p)
        snapshot.particles.position[:] = p
        snapshot.particles.types = ["A"]
        snapshot.configuration.box = [20,20,20,0,0,0]
    sim = simulation_factory(snapshot)

    elastic = hoomd.md.mesh.Elastic(mesh, reference_tags)
    elastic.params["A"] = {"C_xxxx": 29.1, "C_xxyy": 28.0, "C_xyxy": 0.57}

    integrator = hoomd.md.Integrator(dt=0.00005)

    filter_all  = hoomd.filter.All()
    nve = hoomd.md.methods.ConstantVolume(filter=filter_all)
    integrator.forces.append(elastic)
    integrator.methods.append(nve)

    sim.operations.integrator = integrator

    sim.run(0)

    elastic.energy > 0
