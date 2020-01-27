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


def test_overlaps(device, lattice_simulation_factory):

    mc = hoomd.hpmc.integrate.SimplePolygon(23456, d=0, a=0)
    mc.shape['A'] = dict(vertices=[(0, (0.75**0.5) / 2),
                                   (0, -0.2),
                                   (-0.5, -(0.75**0.5) / 2),
                                   (0.5, -(0.75**0.5) / 2)])

    sim = lattice_simulation_factory(dimensions=2, n=(2, 1), a=0.25)
    sim.operations.add(mc)
    sim.operations.schedule()
    sim.run(1)
    assert mc.overlaps > 0

    s = sim.state.snapshot
    if s.exists:
        s.particles.position[0] = (0, 0, 0)
        s.particles.position[1] = (0, 8, 0)
    sim.state.snapshot = s
    assert mc.overlaps == 0

    s = sim.state.snapshot
    if s.exists:
        s.particles.position[0] = (0, 0, 0)
        s.particles.position[1] = (0, 0.85, 0)
    sim.state.snapshot = s
    assert mc.overlaps == 1


def test_shape_moves(device, lattice_simulation_factory):

    mc = hoomd.hpmc.integrate.SimplePolygon(23456)
    mc.shape['A'] = dict(vertices=[(0, (0.75**0.5) / 2),
                                   (0, 0),
                                   (-0.5, -(0.75**0.5) / 2),
                                   (0.5, -(0.75**0.5) / 2)])
    sim = lattice_simulation_factory(dimensions=2)
    sim.operations.add(mc)
    sim.operations.schedule()
    sim.run(100)
    accepted_rejected_rot = sum(sim.operations.integrator.rotate_moves)
    assert accepted_rejected_rot > 0
    accepted_rejected_trans = sum(sim.operations.integrator.translate_moves)
    assert accepted_rejected_trans > 0
