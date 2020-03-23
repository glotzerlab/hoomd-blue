import hoomd
import hoomd.hpmc
from hoomd.hpmc import _hpmc
import numpy as np
import pytest
from hoomd.hpmc.pytest.conftest import *


def check_dict(shape_dict, args):
    """
    check_dict: Function to check that two dictionaries are equivalent

    Arguments: shape_dict and args - dictionaries to test

    Ex: mc = hoomd.hpmc.integrate.Sphere(23456)
        mc.shape["A"] = {"diameter": 1}
        check_dict(mc.shape["A"], {"diameter": 1})

    Useful for more complex nested dictionaries (like the shape key in unions)
    Used to test that the dictionary passed in is what gets passed out
    """
    for key, val in args.items():
        if isinstance(shape_dict[key], list) and len(shape_dict[key]) > 0 \
           and key != 'shapes':
            np.testing.assert_allclose(shape_dict[key], val)
        elif key == 'shapes':
            for shape_args in shape_dict[key]:
                for shape_key, shape_val in shape_args.items():
                    if isinstance(shape_args[shape_key], list) \
                       and len(shape_args[shape_key]) > 0:
                        np.testing.assert_allclose(shape_args[shape_key],
                                                   shape_val)
                    else:
                        assert shape_args[shape_key] == shape_val
        else:
            np.testing.assert_almost_equal(shape_dict[key], val)


def test_dict_conversion(shape_dict_conversion_args):
    for shape_params, args_list in shape_dict_conversion_args():
        for args in args_list:
            test_shape = shape_params(args)
            test_dict = test_shape.asDict()
            check_dict(test_dict, args)


def test_shape_params(integrator_args):
    for shape_integrator, valid_args, invalid_args in integrator_args():
        mc = shape_integrator(23456)
        for args in valid_args:
            mc.shape["A"] = args
            check_dict(mc.shape["A"], args)
        for args in invalid_args:
            with pytest.raises(Exception):
                mc.shape["A"] = args


def test_shape_attached(dummy_simulation_factory, integrator_args):
    for shape_integrator, valid_args, invalid_args in integrator_args():
        mc = shape_integrator(23456)
        for args in valid_args:
            mc.shape["A"] = args
            sim = dummy_simulation_factory()
            sim.operations.add(mc)
            sim.operations.schedule()
            check_dict(mc.shape["A"], args)


def test_moves(device, lattice_simulation_factory, integrator_args):
    dims = 3
    for shape_integrator, valid_args, invalid_args in integrator_args():
        if 'union' not in str(shape_integrator).lower():
            args = valid_args[0]
            if 'polygon' in str(shape_integrator).lower():
                dims = 2
            mc = shape_integrator(23456)
            mc.shape['A'] = args
            sim = lattice_simulation_factory(dimensions=dims)
            sim.operations.add(mc)
            sim.operations.schedule()
            sim.run(100)
            accepted_rejected_trans = sum(sim.operations.integrator.translate_moves)
            assert accepted_rejected_trans > 0
            if 'sphere' not in str(shape_integrator).lower():
                accepted_rejected_rot = sum(sim.operations.integrator.rotate_moves)
                assert accepted_rejected_rot > 0


def test_overlaps_sphere(device, lattice_simulation_factory):
    mc = hoomd.hpmc.integrate.Sphere(23456)
    mc.shape["A"] = {'diameter': 1}
    diameter = mc.shape["A"]["diameter"]
    
    # Should overlap when spheres are less than one diameter apart
    sim = lattice_simulation_factory(dimensions=2,
                                     n=(2, 1),
                                     a=diameter * 0.9)
    sim.operations.add(mc)
    sim.operations.schedule()
    assert mc.overlaps > 0
    
    # Should not overlap when spheres are larger than one diameter apart
    s = sim.state.snapshot
    if s.exists:
        s.particles.position[0] = (0, 0, 0)
        s.particles.position[1] = (0, diameter * 1.1, 0)
    sim.state.snapshot = s
    assert mc.overlaps == 0
    
    # Should barely overlap when spheres are exactly than one diameter apart
    s = sim.state.snapshot
    if s.exists:
        s.particles.position[0] = (0, 0, 0)
        s.particles.position[1] = (0, diameter * 0.9999, 0)
    sim.state.snapshot = s
    assert mc.overlaps == 1


def test_overlaps_ellipsoid(device, lattice_simulation_factory):
    a = 1/4
    b = 1/2
    c = 1
    mc = hoomd.hpmc.integrate.Ellipsoid(23456)
    mc.shape["A"] = {'a': a, 'b': b, 'c': c}
    
    sim = lattice_simulation_factory(dimensions=2,
                                     n=(2, 1),
                                     a=10)
    sim.operations.add(mc)
    gsd_dumper = hoomd.dump.GSD(filename='/Users/dan/danevans/Michigan/Glotzer_Lab/hoomd-dev/test_dump_ellipsoid.gsd', trigger=1, overwrite=True)
    sim.operations.schedule()
    assert mc.overlaps == 0
    
    abc_list = [(0, 0, c), (0, b, 0), (a, 0, 0)]
    for i in range(len(abc_list)):
        # Should barely overlap when ellipsoids are exactly than one diameter apart
        s = sim.state.snapshot
        if s.exists:
            s.particles.position[0] = (0, 0, 0)
            s.particles.position[1] = (abc_list[i][0]*0.9*2,
                                       abc_list[i][1]*0.9*2,
                                       abc_list[i][2]*0.9*2)
        sim.state.snapshot = s
        assert mc.overlaps == 1
        
        # Should not overlap when ellipsoids are larger than one diameter apart
        s = sim.state.snapshot
        if s.exists:
            s.particles.position[0] = (0, 0, 0)
            s.particles.position[1] = (abc_list[i][0]*1.15*2,
                                       abc_list[i][1]*1.15*2,
                                       abc_list[i][2]*1.15*2)
        sim.state.snapshot = s
        assert mc.overlaps == 0
    
    # Line up ellipsoids where they aren't overlapped, and then rotate one so 
    # they overlap
    s = sim.state.snapshot
    if s.exists:
        s.particles.position[0] = (0, 0, 0)
        s.particles.position[1] = (a*1.1*2, 0, 0)
        s.particles.orientation[1] = tuple(np.array([1, 0, 0.45, 0])/(1.2025**0.5))
    sim.state.snapshot = s 
    assert mc.overlaps > 0


def test_overlaps_convex_polygon(device, lattice_simulation_factory):
    triangle = {'vertices': [(0, (0.75**0.5) / 2),
                             (-0.5, -(0.75**0.5) / 2),
                             (0.5, -(0.75**0.5) / 2)]}
    mc = hoomd.hpmc.integrate.ConvexPolygon(23456)
    mc.shape['A'] = triangle

    sim = lattice_simulation_factory(dimensions=2, n=(2, 1), a=10)
    sim.operations.add(mc)
    sim.operations.schedule()
    assert mc.overlaps == 0
    
    # Place center of shape 2 on each of shape 1's vertices
    for vert in mc.shape["A"]["vertices"]:
        s = sim.state.snapshot
        if s.exists:
            s.particles.position[0] = (0, 0, 0)
            s.particles.position[1] = (vert[0], vert[1], 0)
        sim.state.snapshot = s
        assert mc.overlaps > 0
    
    s = sim.state.snapshot
    if s.exists:
        s.particles.position[0] = (0, 0, 0)
        s.particles.position[1] = (0, 1.05, 0)
    sim.state.snapshot = s
    assert mc.overlaps == 0
    
    # Rotate one of the triangles so they will overlap
    s = sim.state.snapshot
    if s.exists:
        s.particles.orientation[1] = tuple(np.array([1, 0, 0, 0.45])/(1.2025**0.5))
    sim.state.snapshot = s
    assert mc.overlaps > 0


def test_overlaps_convex_polyhedron(device, lattice_simulation_factory):
    tetrahedron = {"vertices": np.array([(1, 1, 1),
                                         (-1, -1, 1),
                                         (1, -1, -1),
                                         (-1, 1, -1)])/2}
    mc = hoomd.hpmc.integrate.ConvexPolyhedron(23456)
    mc.shape['A'] = tetrahedron

    sim = lattice_simulation_factory(dimensions=2, n=(2, 1), a=10)
    sim.operations.add(mc)
    sim.operations.schedule()
    assert mc.overlaps == 0
    
    # Place center of shape 2 on each of shape 1's vertices
    for vert in mc.shape["A"]["vertices"]:
        s = sim.state.snapshot
        if s.exists:
            s.particles.position[0] = (0, 0, 0)
            s.particles.position[1] = vert
        sim.state.snapshot = s
        assert mc.overlaps > 0
    
    s = sim.state.snapshot
    if s.exists:
        s.particles.position[0] = (0, 0, 0)
        s.particles.position[1] = (0, 0.9, 0)
    sim.state.snapshot = s
    assert mc.overlaps > 0
    
    s = sim.state.snapshot
    if s.exists:
        s.particles.position[0] = (0, 0, 0)
        s.particles.position[1] = (0, 1.1, 0)
    sim.state.snapshot = s
    assert mc.overlaps == 0
    
    # Rotate one of the tetrahedra so they will overlap
    s = sim.state.snapshot
    if s.exists:
        s.particles.orientation[1] = tuple(np.array([1, 1, 1, 0])/(3**0.5))
    sim.state.snapshot = s
    assert mc.overlaps > 0       

