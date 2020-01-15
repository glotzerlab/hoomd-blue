import hoomd
import numpy as np
from hoomd import hpmc

def test_simple_polygon():

    verts = [(-1, 1), (1, -1), (1, 1), (-1, -1)]
    poly = hpmc.integrate.SimplePolygon(23456)
    poly.shape['A'] = dict(vertices=verts)
    assert not poly.shape['A']['ignore_statistics']
    np.testing.assert_allclose(poly.shape['A']['vertices'], verts)

def test_after_attaching(device, dummy_simulation_factory):

    verts = [(-1, 1), (1, -1), (1, 1), (-1, -1)]
    verts2 = [(-1, 1), (1, -1), (1, 1)]
    poly = hpmc.integrate.SimplePolygon(23456)
    poly.shape['A'] = dict(vertices=verts)
    poly.shape['B'] = dict(vertices=verts2, ignore_statistics=True)

    sim = dummy_simulation_factory(particle_types=['A', 'B'])
    sim.operations.add(poly)
    sim.operations.schedule()

    assert not poly.shape['A']['ignore_statistics']
    assert poly.shape['B']['ignore_statistics']
    np.testing.assert_allclose(poly.shape['A']['vertices'], verts)
    np.testing.assert_allclose(poly.shape['B']['vertices'], verts2)



def test_overlaps(device, dummy_simulation_check_overlaps):
    hoomd.context.initialize("--mode=cpu");
    mc = hoomd.hpmc.integrate.SimplePolygon(23456)
    mc.shape['A'] = dict(vertices=[(0,(0.75**0.5)/2),
                                   (0, -0.2),
                                   (-0.5,-(0.75**0.5)/2),
                                   (0.5, -(0.75**0.5)/2)])
    
    sim = dummy_simulation_check_overlaps()
    sim.operations.add(mc)
    # gsd_dumper = hoomd.dump.GSD(filename='/Users/danevans/hoomd/test_dump_polygon.gsd', trigger=1, overwrite=True)
    # gsd_logger = hoomd.logger.Logger()
    # gsd_logger += mc
    # gsd_dumper.log = gsd_logger
    # sim.operations.add(gsd_dumper)
    sim.operations.schedule()
    sim.run(100)
    overlaps = sim.operations.integrator.overlaps
    assert overlaps > 0
    
def test_shape_moves(device, dummy_simulation_check_moves):
    hoomd.context.initialize("--mode=cpu");
    mc = hoomd.hpmc.integrate.SimplePolygon(23456)
    mc.shape['A'] = dict(vertices=[(0,(0.75**0.5)/2),
                                   (0, 0),
                                   (-0.5,-(0.75**0.5)/2),
                                   (0.5, -(0.75**0.5)/2)])
    sim = dummy_simulation_check_moves()
    sim.operations.add(mc)
    sim.operations.schedule()
    sim.run(100)
    accepted_and_rejected_rotations = sum(sim.operations.integrator.rotate_moves)
    #print(sim.operations.integrator.rotate_moves)
    #print(sim.operations.integrator.translate_moves)
    #assert accepted_and_rejected_rotations > 0
    accepted_and_rejected_translations = sum(sim.operations.integrator.translate_moves)
    assert accepted_and_rejected_translations > 0
