import numpy as np
from hoomd import hpmc
import pytest
import hoomd


def test_convex_spheropolyhedron_python():

    verts = [(-1, 1, 0), (1, -1, 0), (1, 1, 0), (-1, -1, 0)]
    poly = hpmc.integrate.ConvexSpheropolyhedron(23456)
    poly.shape['A'] = dict(vertices=verts)
    assert not poly.shape['A']['ignore_statistics']
    np.testing.assert_allclose(poly.shape['A']['vertices'], verts)


def test_convex_spheropolyhedron_params():

    verts1 = [(-1, 1, 0), (1, -1, 0), (1, 1, 0), (-1, -1, 0)]
    verts2 = [(-1, 1, 0), (1, -1, 1), (1, 1, 1)]
    verts3 = [(-1, 1, 0), (1, -1, 0), (1, 1, 1), (0, 0, 0)]
    verts4 = [(0, 1, 0), (1, 0, 0), (1, 0, 1), (0, -1, 0), (-1, 0, 0)]
    verts5 = [(-1, 1, 0),
              (-1, -1, 0),
              (1, 1, 1),
              (0, 0, 0),
              (1, 0, 0),
              (0, 0, 1)]

    poly = hpmc.integrate.ConvexSpheropolyhedron(23456)

    poly.shape['A'] = dict()
    assert poly.shape['A']['vertices'] is None
    assert poly.shape['A']['ignore_statistics'] is False

    poly.shape['B'] = dict(vertices=verts1, ignore_statistics=True)
    assert poly.shape['B']['vertices'] == verts1
    assert poly.shape['B']['ignore_statistics'] is True

    poly.shape['C'] = dict(vertices=verts1)
    poly.shape['D'] = dict(vertices=verts2, ignore_statistics=True)
    poly.shape['E'] = dict(vertices=verts3)
    poly.shape['F'] = dict(vertices=verts4, ignore_statistics=True)
    poly.shape['G'] = dict(vertices=verts5)
    '''
    np.testing.assert_allclose(poly.shape['C']['vertices'], verts1)
    np.testing.assert_allclose(poly.shape['D']['vertices'], verts2)
    np.testing.assert_allclose(poly.shape['E']['vertices'], verts3)
    np.testing.assert_allclose(poly.shape['F']['vertices'], verts4)
    np.testing.assert_allclose(poly.shape['G']['vertices'], verts5)
    '''
    assert poly.shape['C']['vertices'] == verts1
    assert poly.shape['D']['vertices'] == verts2
    assert poly.shape['E']['vertices'] == verts3
    assert poly.shape['F']['vertices'] == verts4
    assert poly.shape['G']['vertices'] == verts5


def test_convex_poly_after_attaching(device, dummy_simulation_factory):

    verts1 = [(-1, 1, 0), (1, -1, 0), (1, 1, 0), (-1, -1, 0)]
    verts2 = [(-1, 1, 0), (1, -1, 1), (1, 1, 1)]
    verts3 = [(-1, 1, 0), (1, -1, 0), (1, 1, 1), (0, 0, 0)]
    verts4 = [(0, 1, 0), (1, 0, 0), (1, 0, 1), (0, -1, 0), (-1, 0, 0)]
    verts5 = [(-1, 1, 0),
              (-1, -1, 0),
              (1, 1, 1),
              (0, 0, 0),
              (1, 0, 0),
              (0, 0, 1)]

    poly = hpmc.integrate.ConvexSpheropolyhedron(23456)
    poly.shape['A'] = dict(vertices=verts1)
    poly.shape['B'] = dict(vertices=verts2, ignore_statistics=True)
    poly.shape['C'] = dict(vertices=verts3)
    poly.shape['D'] = dict(vertices=verts4, ignore_statistics=True)
    poly.shape['E'] = dict(vertices=verts5)

    sim = dummy_simulation_factory(particle_types=['A', 'B', 'C', 'D', 'E'])
    sim.operations.add(poly)
    sim.operations.schedule()

    assert not poly.shape['A']['ignore_statistics']
    assert poly.shape['B']['ignore_statistics']
    assert not poly.shape['C']['ignore_statistics']
    assert poly.shape['D']['ignore_statistics']
    assert not poly.shape['E']['ignore_statistics']

    np.testing.assert_allclose(poly.shape['A']['vertices'], verts1)
    np.testing.assert_allclose(poly.shape['B']['vertices'], verts2)
    np.testing.assert_allclose(poly.shape['C']['vertices'], verts3)
    np.testing.assert_allclose(poly.shape['D']['vertices'], verts4)
    np.testing.assert_allclose(poly.shape['E']['vertices'], verts5)

    # check for errors on invalid input
    with pytest.raises(RuntimeError):
        poly.shape['A'] = dict(vertices='invalid')

    with pytest.raises(TypeError):
        poly.shape['A'] = dict(vertices=[1, 2, 3, 4])

    with pytest.raises(RuntimeError):
        poly.shape['A'] = dict(vertices=verts1, ignore_statistics='invalid')


def test_overlaps(device, dummy_simulation_check_overlaps):

    mc = hoomd.hpmc.integrate.ConvexSpheropolyhedron(23456, d=0, a=0)
    mc.shape['A'] = dict(vertices=[(0.25, 0, 0), (-0.25, 0, 0)],
                         sweep_radius=0.25)

    sim = dummy_simulation_check_overlaps()
    sim.operations.add(mc)
    # gsd_dumper = hoomd.dump.GSD(filename='/Users/danevans/hoomd/
    # test_dump_convex_spheropolyhedron.gsd', trigger=1, overwrite=True)
    # gsd_logger = hoomd.logger.Logger()
    # gsd_logger += mc
    # gsd_dumper.log = gsd_logger
    # sim.operations.add(gsd_dumper)
    sim.operations.schedule()
    sim.run(1)
    overlaps = sim.operations.integrator.overlaps
    assert overlaps > 0

    s = sim.state.snapshot
    s.particles.position[0] = (0, 0, 0)
    s.particles.position[1] = (0, 8, 0)
    sim.state.snapshot = s
    sim.operations.add(mc)
    # gsd_dumper = hoomd.dump.GSD(filename='/Users/danevans/hoomd/
    # test_dump_convex_spheropolyhedron.gsd', trigger=1, overwrite=True)
    # gsd_logger = hoomd.logger.Logger()
    # gsd_logger += mc
    # gsd_dumper.log = gsd_logger
    # sim.operations.add(gsd_dumper)
    sim.operations.schedule()
    sim.run(1)
    overlaps = sim.operations.integrator.overlaps
    assert overlaps == 0

    s = sim.state.snapshot
    s.particles.position[0] = (0, 0, 0)
    s.particles.position[1] = (0, 0.5, 0)
    sim.state.snapshot = s
    sim.operations.add(mc)
    # gsd_dumper = hoomd.dump.GSD(filename='/Users/danevans/hoomd/
    # test_dump_convex_spheropolyhedron.gsd', trigger=1, overwrite=True)
    # gsd_logger = hoomd.logger.Logger()
    # gsd_logger += mc
    # gsd_dumper.log = gsd_logger
    # sim.operations.add(gsd_dumper)
    sim.operations.schedule()
    sim.run(1)
    overlaps = sim.operations.integrator.overlaps
    assert overlaps > 0


def test_shape_moves(device, dummy_simulation_check_moves):

    mc = hoomd.hpmc.integrate.ConvexSpheropolyhedron(23456)
    mc.shape['A'] = dict(vertices=[(0.25, 0, 0), (-0.25, 0, 0)],
                         sweep_radius=0.25)
    sim = dummy_simulation_check_moves()
    sim.operations.add(mc)
    sim.operations.schedule()
    sim.run(100)
    accepted_rejected_rot = sum(sim.operations.integrator.rotate_moves)
    assert accepted_rejected_rot > 0
    accepted_rejected_trans = sum(sim.operations.integrator.translate_moves)
    assert accepted_rejected_trans > 0
