from bbb import basic_blocks, packmol
import copy

cube = basic_blocks.Cuboid(4, 4, 4, 2, 2, 2, type='A');
n_cubes = 30

box = packmol.ConstraintSimulationBoxHOOMD(1, 1, 1)
box.scaleToVolume(n_cubes * 100)

g = packmol.GeneratorXML(simulation_box=box, seed=1, tolerance=1.2)
g.addBuildingBlock(cube, 30)

g.writeOutput(open('cubes.xml', 'w'));
