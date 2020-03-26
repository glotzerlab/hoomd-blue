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
            for i in range(len(shape_dict[key])):
                shape_args = shape_dict[key][i]
                val_args = val[i]
                for shape_key in shape_args:
                    if isinstance(shape_args[shape_key], list) \
                       and len(shape_args[shape_key]) > 0:
                        np.testing.assert_allclose(val_args[shape_key],
                                                   shape_args[shape_key])
                    else:
                        assert shape_args[shape_key] == val_args[shape_key]
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
            assert sim.operations.integrator is None
            sim.operations.add(mc)
            sim.operations.schedule()
            check_dict(mc.shape["A"], args)


def test_moves(device, lattice_simulation_factory, integrator_args):
    dims = 3
    for shape_integrator, valid_args, invalid_args in integrator_args():
        args = valid_args[0]
        if 'polygon' in str(shape_integrator).lower():
            dims = 2
        mc = shape_integrator(23456)
        mc.shape['A'] = args
        
        sim = lattice_simulation_factory(dimensions=dims)
        sim.operations.add(mc)
        with pytest.raises(AttributeError):
            sim.operations.integrator.translate_moves
        with pytest.raises(AttributeError):
            sim.operations.integrator.rotate_moves
        sim.operations.schedule()
        
        assert sum(sim.operations.integrator.translate_moves) == 0
        assert sum(sim.operations.integrator.rotate_moves) == 0
        
        sim.run(100)
        accepted_rejected_trans = sum(sim.operations.integrator.translate_moves)
        assert accepted_rejected_trans > 0
        if 'sphere' not in str(shape_integrator).lower():
            accepted_rejected_rot = sum(sim.operations.integrator.rotate_moves)
            assert accepted_rejected_rot > 0


def test_overlaps_sphere(device, lattice_simulation_factory):
    
    # A spheropolyhedron with a single vertex should be a sphere
    # A sphinx where the indenting sphere is negligible should also be a sphere
    shapes = [({'diameter': 1},
               hoomd.hpmc.integrate.Sphere),
              ({'vertices': [(0, 0, 0)], 'sweep_radius': 0.5},
               hoomd.hpmc.integrate.ConvexSpheropolyhedron),
              ({'diameters': [1, -0.0001], 'centers': [(0, 0, 0), (0, 0, 0.5)]},
               hoomd.hpmc.integrate.Sphinx)]

    for args, integrator in shapes:
        mc = integrator(23456)
        mc.shape["A"] = args
        diameter = 1
        
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
    sim.operations.schedule()
    assert mc.overlaps == 0
    
    abc_list = [(0, 0, c), (0, b, 0), (a, 0, 0)]
    for i in range(len(abc_list)):
        # Should barely overlap when ellipsoids are exactly than one diameter apart
        s = sim.state.snapshot
        if s.exists:
            s.particles.position[0] = (0, 0, 0)
            s.particles.position[1] = (abc_list[i][0] * 0.9 * 2,
                                       abc_list[i][1] * 0.9 * 2,
                                       abc_list[i][2] * 0.9 * 2)
        sim.state.snapshot = s
        assert mc.overlaps == 1
        
        # Should not overlap when ellipsoids are larger than one diameter apart
        s = sim.state.snapshot
        if s.exists:
            s.particles.position[0] = (0, 0, 0)
            s.particles.position[1] = (abc_list[i][0] * 1.15 * 2,
                                       abc_list[i][1] * 1.15 * 2,
                                       abc_list[i][2] * 1.15 * 2)
        sim.state.snapshot = s
        assert mc.overlaps == 0
    
    # Line up ellipsoids where they aren't overlapped, and then rotate one so 
    # they overlap
    s = sim.state.snapshot
    if s.exists:
        s.particles.position[0] = (0, 0, 0)
        s.particles.position[1] = (a * 1.1 * 2, 0, 0)
        s.particles.orientation[1] = tuple(np.array([1, 0, 0.45, 0]) / (1.2025**0.5))
    sim.state.snapshot = s 
    assert mc.overlaps > 0


def test_overlaps_polygons(device, lattice_simulation_factory):
    triangle = {'vertices': [(0, (0.75**0.5) / 2),
                             (-0.5, -(0.75**0.5) / 2),
                             (0.5, -(0.75**0.5) / 2)]}
    
    square = {"vertices": np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)])/2}
    
    # Args should work for ConvexPolygon, SimplePolygon, and ConvexSpheropolygon
    shapes = [(triangle, hoomd.hpmc.integrate.ConvexPolygon),
              (triangle, hoomd.hpmc.integrate.SimplePolygon),
              (triangle, hoomd.hpmc.integrate.ConvexSpheropolygon),
              (square, hoomd.hpmc.integrate.ConvexPolygon),
              (square, hoomd.hpmc.integrate.SimplePolygon),
              (square, hoomd.hpmc.integrate.ConvexSpheropolygon)]
    
    for args, integrator in shapes:
        mc = integrator(23456)
        mc.shape['A'] = args

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
        
        # Rotate one of the shapes so they will overlap
        s = sim.state.snapshot
        if s.exists:
            s.particles.orientation[1] = tuple(np.array([1, 0, 0, 0.45]) / (1.2025**0.5))
        sim.state.snapshot = s
        assert mc.overlaps > 0


def test_overlaps_polyhedra(device, lattice_simulation_factory):
    tetrahedron_verts = np.array([(1, 1, 1),
                                  (-1, -1, 1),
                                  (1, -1, -1),
                                  (-1, 1, -1)])/2
    
    tetrahedron_faces = [[1, 3, 2], [3, 0, 2], [1, 0, 3], [1, 2, 0]]
    
    cube_verts = [(-0.5, -0.5, -0.5),
                  (-0.5, -0.5, 0.5),
                  (-0.5, 0.5, -0.5),
                  (-0.5, 0.5, 0.5),
                  (0.5, -0.5, -0.5),
                  (0.5, -0.5, 0.5),
                  (0.5, 0.5, -0.5),
                  (0.5, 0.5, 0.5)]

    cube_faces = [[0, 2, 6],
                  [6, 4, 0],
                  [5, 0, 4],
                  [5, 1, 0],
                  [5, 4, 6],
                  [5, 6, 7],
                  [3, 2, 0],
                  [3, 0, 1],
                  [3, 6, 2],
                  [3, 7, 6],
                  [3, 1, 5],
                  [3, 5, 7]]
    
    # Test args with ConvexPolyhedron, ConvexSpheropolyhedron, and Polyhedron      
    shapes = [({"vertices": tetrahedron_verts},
               hoomd.hpmc.integrate.ConvexPolyhedron),
              ({"vertices": tetrahedron_verts},
               hoomd.hpmc.integrate.ConvexSpheropolyhedron),
              ({"vertices": tetrahedron_verts,
                "faces": tetrahedron_faces,
                "overlap": [True, True, True, True]},
               hoomd.hpmc.integrate.Polyhedron),
              ({"vertices": cube_verts},
               hoomd.hpmc.integrate.ConvexPolyhedron),
              ({"vertices": cube_verts},
               hoomd.hpmc.integrate.ConvexSpheropolyhedron),
              ({"vertices": cube_verts,
                "faces": cube_faces,
                "overlap": [True, True, True, True, True, True,
                            True, True, True, True, True, True]},
               hoomd.hpmc.integrate.Polyhedron)]
    
    for args, integrator in shapes:
        mc = integrator(23456)
        mc.shape['A'] = args

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
        
        # Rotate one of the polyhedra so they will overlap
        s = sim.state.snapshot
        if s.exists:
            s.particles.orientation[1] = tuple(np.array([1, 1, 1, 0]) / (3**0.5))
        sim.state.snapshot = s
        assert mc.overlaps > 0          


def test_overlaps_spheropolygon(device, lattice_simulation_factory):

    triangle = {'vertices': [(0, (0.75**0.5) / 2),
                             (-0.5, -(0.75**0.5) / 2),
                             (0.5, -(0.75**0.5) / 2)],
                'sweep_radius': 0.2}
    
    square = {"vertices": np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)]) / 2,
              "sweep_radius": 0.1}
    
    for args in [triangle, square]:
        mc = hoomd.hpmc.integrate.ConvexSpheropolygon(23456)
        mc.shape['A'] = args

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
        
        # Place shapes where they wouldn't overlap w/o sweep radius
        s = sim.state.snapshot
        if s.exists:
            s.particles.position[0] = (0, 0, 0)
            s.particles.position[1] = (0, 1.2, 0)
        sim.state.snapshot = s
        assert mc.overlaps > 0
             
        s = sim.state.snapshot
        if s.exists:
            s.particles.position[0] = (0, 0, 0)
            s.particles.position[1] = (0, 1.3, 0)
        sim.state.snapshot = s
        assert mc.overlaps == 0
        
        # Rotate one of the shapes so they will overlap
        s = sim.state.snapshot
        if s.exists:
            s.particles.orientation[1] = tuple(np.array([1, 0, 0, 0.45]) / (1.2025**0.5))
        sim.state.snapshot = s
        assert mc.overlaps > 0


def test_overlaps_spheropolyhedron(device, lattice_simulation_factory):

    tetrahedron = {"vertices": np.array([(1, 1, 1),
                                         (-1, -1, 1),
                                         (1, -1, -1),
                                         (-1, 1, -1)]) / 2,
                   "sweep_radius": 0.2}
    
    cube = {"vertices": [(-0.5, -0.5, -0.5),
                         (-0.5, -0.5, 0.5),
                         (-0.5, 0.5, -0.5),
                         (-0.5, 0.5, 0.5),
                         (0.5, -0.5, -0.5),
                         (0.5, -0.5, 0.5),
                         (0.5, 0.5, -0.5),
                         (0.5, 0.5, 0.5)],
            "sweep_radius": 0.2}
    
    for args in [tetrahedron, cube]:
        mc = hoomd.hpmc.integrate.ConvexSpheropolyhedron(23456)
        mc.shape['A'] = args

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
            s.particles.position[1] = (0, 1.2, 0)
        sim.state.snapshot = s
        assert mc.overlaps > 0
        
        s = sim.state.snapshot
        if s.exists:
            s.particles.position[0] = (0, 0, 0)
            s.particles.position[1] = (0, 1.5, 0)
        sim.state.snapshot = s
        assert mc.overlaps == 0
        
        # Rotate one of the polyhedra so they will overlap
        s = sim.state.snapshot
        if s.exists:
            s.particles.orientation[1] = tuple(np.array([1, 1, 1, 0]) / (3**0.5))
        sim.state.snapshot = s
        assert mc.overlaps > 0


def test_overlaps_union(device, lattice_simulation_factory):
    sphere_mc = hoomd.hpmc.integrate.Sphere(23456)
    sphere_mc.shape['A'] = {'diameter': 1}

    spheropolyhedron_mc = hoomd.hpmc.integrate.ConvexSpheropolyhedron(23456)
    spheropolyhedron_mc.shape['A'] = {"vertices": np.array([(1, 1, 1),
                                                            (-1, -1, 1),
                                                            (1, -1, -1),
                                                            (-1, 1, -1)]) / 2}

    faceted_ell_mc = hoomd.hpmc.integrate.FacetedEllipsoid(23456)
    faceted_ell_mc.shape['A'] = {"normals": [(0, 0, 1)],
                                 "a": 0.5,
                                 "b": 0.5,
                                 "c": 1,
                                 "vertices": [],
                                 "origin": (0, 0, 0),
                                 "offsets": [0]}

    shapes = [(sphere_mc, 
               hoomd.hpmc.integrate.SphereUnion),
              (spheropolyhedron_mc, 
               hoomd.hpmc.integrate.ConvexSpheropolyhedronUnion),
              (faceted_ell_mc,
               hoomd.hpmc.integrate.FacetedEllipsoidUnion)]
    
    for inner_mc, integrator in shapes:
        args = {'shapes': [inner_mc.shape['A'], inner_mc.shape['A']],
                'positions': [(0, 0, 0), (0, 0, 1)],
                'orientations': [(1, 0, 0, 0), (1, 0, 0, 0)],
                'overlap': [1, 1]}
        mc = integrator(23456)
        mc.shape['A'] = args
        
        sim = lattice_simulation_factory(dimensions=2, n=(2, 1), a=10)
        sim.operations.add(mc)
        sim.operations.schedule()
        
        assert mc.overlaps == 0
        test_positions = [(1.1, 0, 0), (0, 1.1, 0)]
        test_orientations = np.array([[1, 0, -0.06, 0], [1, 0.06, 0, 0]])
        test_orientations = test_orientations.T/np.linalg.norm(test_orientations, 
                                                               axis=1)
        test_orientations = test_orientations.T
        # Shapes are stacked in z direction
        ang = 0
        for i in range(len(test_positions)):
            s = sim.state.snapshot
            if s.exists:
                s.particles.position[0] = (0, 0, 0)
                s.particles.position[1] = test_positions[i]
                s.particles.orientation[1] = (1, 0, 0, 0)
            sim.state.snapshot = s
            assert mc.overlaps == 0
            
            # Slightly rotate union about x or y axis so they overlap
            if s.exists:
                s.particles.orientation[1] = test_orientations[i]
            sim.state.snapshot = s
                
            assert mc.overlaps > 0

        
        for pos in [(0.9, 0, 0), (0, 0.9, 0), (0, 0, 1.1)]:
            s = sim.state.snapshot
            if s.exists:
                s.particles.position[0] = (0, 0, 0)
                s.particles.position[1] = pos
            sim.state.snapshot = s
            assert mc.overlaps > 0
