import hoomd
import hoomd.hpmc
import hoomd.hpmc._hpmc as hpmc
import math
import pytest
import numpy

def test_ellipsoid():

    args_1 = {'a': 1, 'b': 2, 'c': 3, 'ignore_statistics':1}
    test_ellipsoid1 = hpmc.EllipsoidParams(args_1)
    test_dict1 = test_ellipsoid1.asDict()
    assert test_dict1 == args_1

    args_2 = {'a': 4, 'b': 1, 'c': 30, 'ignore_statistics':1}
    test_ellipsoid2 = hpmc.EllipsoidParams(args_2)
    test_dict2 = test_ellipsoid2.asDict()
    assert test_dict2 == args_2

    args_3 = {'a': 10, 'b': 5, 'c': 6, 'ignore_statistics':0}
    test_ellipsoid3 = hpmc.EllipsoidParams(args_3)
    test_dict3 = test_ellipsoid3.asDict()
    assert test_dict3 == args_3


def test_shape_params():

    mc = hoomd.hpmc.integrate.Ellipsoid(23456)

    mc.shape['A'] = dict()
    assert mc.shape['A']['a'] is None
    assert mc.shape['A']['b'] is None
    assert mc.shape['A']['c'] is None
    assert mc.shape['A']['ignore_statistics'] is False

    mc.shape['B'] = dict(a=2.5, b=1, c=3)
    assert mc.shape['B']['a'] == 2.5
    assert mc.shape['B']['b'] == 1
    assert mc.shape['B']['c'] == 3
    assert mc.shape['B']['ignore_statistics'] is False

    mc.shape['C'] = dict(ignore_statistics=True)
    assert mc.shape['C']['a'] is None
    assert mc.shape['C']['b'] is None
    assert mc.shape['C']['c'] is None
    assert mc.shape['C']['ignore_statistics'] is True


def test_shape_params_attached(device, dummy_simulation_factory):

    mc = hoomd.hpmc.integrate.Ellipsoid(23456)

    mc.shape['A'] = dict(a=5.1, b=1.7, c=2.2)
    mc.shape['B'] = dict(a=1.1, b=4.6, c=6.3, ignore_statistics=True)
    mc.shape['C'] = dict(a=0.4, b=25, c=3.1)
    mc.shape['D'] = dict(a=4.2, b=4.2, c=4.2)

    sim = dummy_simulation_factory(particle_types=['A', 'B', 'C', 'D'])
    sim.operations.add(mc)
    sim.operations.schedule()
    #isclose(a, b, rel_tol = 1e-09, abs_tol 0.0)
    assert math.isclose(mc.shape['A']['a'], 5.1, rel_tol = 1e-06)
    assert math.isclose(mc.shape['A']['b'], 1.7, rel_tol = 1e-06)
    assert math.isclose(mc.shape['A']['c'], 2.2, rel_tol = 1e-06)
    assert mc.shape['A']['ignore_statistics'] == False

    assert math.isclose(mc.shape['B']['a'], 1.1, rel_tol = 1e-06)
    assert math.isclose(mc.shape['B']['b'], 4.6, rel_tol = 1e-06)
    assert math.isclose(mc.shape['B']['c'], 6.3, rel_tol = 1e-06)
    assert mc.shape['B']['ignore_statistics'] == True

    assert math.isclose(mc.shape['C']['a'], 0.4, rel_tol = 1e-06)
    assert math.isclose(mc.shape['C']['b'], 25, rel_tol = 1e-06)
    assert math.isclose(mc.shape['C']['c'], 3.1, rel_tol = 1e-06)
    assert mc.shape['C']['ignore_statistics'] == False

    assert math.isclose(mc.shape['D']['a'], 4.2, rel_tol = 1e-06)
    assert math.isclose(mc.shape['D']['b'], 4.2, rel_tol = 1e-06)
    assert math.isclose(mc.shape['D']['c'], 4.2, rel_tol = 1e-06)
    assert mc.shape['D']['ignore_statistics'] == False

    # check for errors on invalid input
    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(a='invalid', b='invalid', c='invalid')

    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(a=1.3, b='invalid', c='invalid')

    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(a='invalid', b=4.1, c='invalid')

    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(a='invalid', b='invalid', c=3.6)

    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(a=5.5, b=2.7, c='invalid')

    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(a=1.8, b='invalid', c=8.3)

    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(a='invalid', b=4.7, c=5.8)

    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(a=[1, 2, 3], b=[4, 5, 6], c=[7, 8, 9])

    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(a=3.1, b=[4, 5, 6], c=[7, 8, 9])

    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(a=[1, 2, 3], b=4.2, c=[7, 8, 9])

    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(a=[1, 2, 3], b=[4, 5, 6], c=7.1)

    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(a=1.2, b=3.1, c=[7, 8, 9])

    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(a=4.4, b=[4, 5, 6], c=9.1)

    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(a=[1, 2, 3], b=4.4, c=2.9)

    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(a=2.1, b=5.2, c=1.4, ignore_statistics='invalid')
        
        
        
        
def test_overlaps(device, dummy_simulation_check_overlaps):
    hoomd.context.initialize("--mode=cpu");
    mc = hoomd.hpmc.integrate.Ellipsoid(23456, d=0, a=0)
    mc.shape['A'] = dict(a=0.75,b=1, c=0.5)
    sim = dummy_simulation_check_overlaps()
    sim.operations.add(mc)
    # gsd_dumper = hoomd.dump.GSD(filename='/Users/danevans/hoomd/test_dump_ellipsoid.gsd', trigger=1, overwrite=True)
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
    # gsd_dumper = hoomd.dump.GSD(filename='/Users/danevans/hoomd/test_dump_ellipsoid.gsd', trigger=1, overwrite=True)
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
    s.particles.position[1] = (0, 1.99, 0)
    sim.state.snapshot = s
    sim.operations.add(mc)
    gsd_dumper = hoomd.dump.GSD(filename='/Users/danevans/hoomd/test_dump_ellipsoid.gsd', trigger=1, overwrite=True)
    gsd_logger = hoomd.logger.Logger()
    gsd_logger += mc
    gsd_dumper.log = gsd_logger
    sim.operations.add(gsd_dumper)
    sim.operations.schedule()
    sim.run(1)
    overlaps = sim.operations.integrator.overlaps
    assert overlaps > 0
    
def test_shape_moves(device, dummy_simulation_check_moves):
    hoomd.context.initialize("--mode=cpu");
    mc = hoomd.hpmc.integrate.Ellipsoid(23456)
    mc.shape['A'] = dict(a=0.75,b=1, c=0.5)
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
