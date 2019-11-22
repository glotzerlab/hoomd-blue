import hoomd
import numpy as np
from hoomd import hpmc

def test_simple_polygon():

    verts = [(-1, 1), (1, -1), (1, 1), (-1, -1)] 
    poly = hpmc.integrate.SimplePolygon(23456)
    poly.shape['A'] = dict(vertices=verts)
    assert not poly.shape['A']['ignore_statistics']
    np.testing.assert_allclose(poly.shape['A']['vertices'], verts)

def test_after_attaching(device, dummy_simulation):
    
    verts = [(-1, 1), (1, -1), (1, 1), (-1, -1)] 
    verts2 = [(-1, 1), (1, -1), (1, 1)] 
    poly = hpmc.integrate.SimplePolygon(23456)
    poly.shape['A'] = dict(vertices=verts)
    poly.shape['B'] = dict(vertices=verts2, ignore_statistics=True)
    
    sim = dummy_simulation
    sim.operations.add(poly)
    sim.operations.schedule()
    
    assert not poly.shape['A']['ignore_statistics']
    assert poly.shape['B']['ignore_statistics']
    np.testing.assert_allclose(poly.shape['A']['vertices'], verts)    
    np.testing.assert_allclose(poly.shape['B']['vertices'], verts2)    
