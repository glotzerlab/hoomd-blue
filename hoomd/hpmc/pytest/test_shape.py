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
            assert shape_dict[key] == val


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


def test_overlaps(device, lattice_simulation_factory, integrator_args):
    for shape_integrator, valid_args, invalid_args in integrator_args():
        if 'union' not in str(shape_integrator).lower():
            args = valid_args[0]
            mc = shape_integrator(23456)
            mc.shape['A'] = args

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
                s.particles.position[1] = (0, 0.5, 0)
            sim.state.snapshot = s
            assert mc.overlaps == 1


def test_overlaps_sphere(device, lattice_simulation_factory, sphere_valid_args):
    for args in sphere_valid_args():
        mc = hoomd.hpmc.integrate.Sphere(23456)
        mc.shape["A"] = args
        diameter = mc.shape["A"]["diameter"]

        # Should overlap when spheres are less than one diameter apart
        sim = lattice_simulation_factory(dimensions=2,
                                         n=(2, 1),
                                         a=diameter * 0.9)
        sim.operations.add(mc)
        sim.operations.schedule()
        sim.run(1)
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
        print(args)
        print(s.particles.position)
        assert mc.overlaps == 1
