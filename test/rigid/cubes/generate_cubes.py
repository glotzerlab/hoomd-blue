from bbb import basic_blocks, packmol
import copy

cube = basic_blocks.Cuboid(Nw=4, w=2, type='A');
n_cubes = 300

box = packmol.ConstraintSimulationBoxHOOMD(1, 1, 1)
box.scaleToVolume(n_cubes * 100)

g = packmol.GeneratorXML(simulation_box=box, seed=1, tolerance=1.2)
g.addBuildingBlock(cube, n_cubes)

g.writeOutput(open('cubes.xml', 'w'));

