from bbb import basic_blocks, packmol
import copy

rod = basic_blocks.Rod(N=5, r0=0.5, type='A');

box = packmol.ConstraintSimulationBoxHOOMD(1, 1, 1)
box.scaleToVolume(10000)

g = packmol.GeneratorXML(simulation_box=box, seed=1, tolerance=1.2)
g.addBuildingBlock(rod, 300)

g.writeOutput(open('rods.xml', 'w'));
