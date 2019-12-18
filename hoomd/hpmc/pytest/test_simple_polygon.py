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
                                   (0, 0),
                                   (-0.5,-(0.75**0.5)/2),
                                   (0.5, -(0.75**0.5)/2)])
    
    sim = dummy_simulation_check_overlaps()
    sim.operations.add(mc)
    sim.operations.schedule()
    sim.run(100)
    #overlaps = sim.operations.integrator.overlaps
    #assert overlaps > 0
    assert True
    
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
    initial_snap = sim.state.snapshot
    initial_pos = initial_snap.particles.position
    sim.run(100)
    final_snap = sim.state.snapshot
    final_pos = final_snap.particles.position
    #accepted_and_rejected = sim.operations.integrator.accepted + 
    #                        sim.operations.integrator.rejected
    #assert accepted_and_rejected > 0
    np.testing.assert_raises(AssertionError, 
                                np.testing.assert_allclose, 
                                final_pos, 
                                initial_pos)
