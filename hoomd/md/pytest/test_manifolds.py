# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
from hoomd.conftest import pickling_check
import pytest
from copy import deepcopy
from collections import namedtuple

paramtuple = namedtuple(
    'paramtuple', ['setup_params', 'extra_params', 'changed_params', 'surface'])


def _manifold_base_params():
    manifold_base_params_list = []
    # Start with valid parameters to get the keys and placeholder values

    cylinder_setup_params = {'r': 5}
    cylinder_extra_params = {'P': (0, 0, 0)}
    cylinder_changed_params = {'r': 4, 'P': (1.0, 0, 0)}

    manifold_base_params_list.extend([
        paramtuple(cylinder_setup_params, cylinder_extra_params,
                   cylinder_changed_params, hoomd.md.manifold.Cylinder)
    ])

    diamond_setup_params = {'N': (1, 1, 1)}
    diamond_extra_params = {'epsilon': 0}
    diamond_changed_params = {'N': (1, 2, 2), 'epsilon': 0.1}

    manifold_base_params_list.extend([
        paramtuple(diamond_setup_params, diamond_extra_params,
                   diamond_changed_params, hoomd.md.manifold.Diamond)
    ])

    ellipsoid_setup_params = {'a': 3.3, 'b': 5, 'c': 4.1}
    ellipsoid_extra_params = {'P': (0, 0, 0)}
    ellipsoid_changed_params = {'a': 4, 'b': 2, 'c': 5.2, 'P': (1.0, 0, 0)}

    manifold_base_params_list.extend([
        paramtuple(ellipsoid_setup_params, ellipsoid_extra_params,
                   ellipsoid_changed_params, hoomd.md.manifold.Ellipsoid)
    ])

    gyroid_setup_params = {'N': (1, 2, 1)}
    gyroid_extra_params = {'epsilon': 0}
    gyroid_changed_params = {'N': (2, 1, 1), 'epsilon': 0.1}

    manifold_base_params_list.extend([
        paramtuple(gyroid_setup_params, gyroid_extra_params,
                   gyroid_changed_params, hoomd.md.manifold.Gyroid)
    ])

    primitive_setup_params = {'N': (1, 1, 1)}
    primitive_extra_params = {'epsilon': 0}
    primitive_changed_params = {'N': (2, 2, 2), 'epsilon': -0.1}

    manifold_base_params_list.extend([
        paramtuple(primitive_setup_params, primitive_extra_params,
                   primitive_changed_params, hoomd.md.manifold.Primitive)
    ])

    sphere_setup_params = {'r': 5}
    sphere_extra_params = {'P': (0, 0, 0)}
    sphere_changed_params = {'r': 4, 'P': (1.0, 0, 0)}

    manifold_base_params_list.extend([
        paramtuple(sphere_setup_params, sphere_extra_params,
                   sphere_changed_params, hoomd.md.manifold.Sphere)
    ])

    xyplane_setup_params = {}
    xyplane_extra_params = {'shift': 0}
    xyplane_changed_params = {'shift': 0.5}

    manifold_base_params_list.extend([
        paramtuple(xyplane_setup_params, xyplane_extra_params,
                   xyplane_changed_params, hoomd.md.manifold.Plane)
    ])

    return manifold_base_params_list


@pytest.fixture(scope="function",
                params=_manifold_base_params(),
                ids=(lambda x: x[3].__name__))
def manifold_base_params(request):
    return deepcopy(request.param)


def check_instance_attrs(instance, attr_dict, set_attrs=False):
    for attr, value in attr_dict.items():
        if set_attrs:
            with pytest.raises(AttributeError):
                setattr(instance, attr, value)
        else:
            assert getattr(instance, attr) == value


def test_attributes(manifold_base_params):
    surface = manifold_base_params.surface(**manifold_base_params.setup_params)

    check_instance_attrs(surface, manifold_base_params.setup_params)
    check_instance_attrs(surface, manifold_base_params.extra_params)

    check_instance_attrs(surface, manifold_base_params.changed_params, True)

    check_instance_attrs(surface, manifold_base_params.setup_params)
    check_instance_attrs(surface, manifold_base_params.extra_params)


def test_attributes_attached(simulation_factory, two_particle_snapshot_factory,
                             manifold_base_params):

    all_ = hoomd.filter.All()
    surface = manifold_base_params.surface(**manifold_base_params.setup_params)
    method = hoomd.md.methods.rattle.NVE(filter=all_,
                                         manifold_constraint=surface)

    sim = simulation_factory(two_particle_snapshot_factory())
    sim.operations.integrator = hoomd.md.Integrator(0.005, methods=[method])
    sim.run(0)

    check_instance_attrs(surface, manifold_base_params.setup_params)
    check_instance_attrs(surface, manifold_base_params.extra_params)

    check_instance_attrs(surface, manifold_base_params.changed_params, True)

    check_instance_attrs(surface, manifold_base_params.setup_params)
    check_instance_attrs(surface, manifold_base_params.extra_params)


def test_pickling(manifold_base_params, simulation_factory,
                  two_particle_snapshot_factory):
    sim = simulation_factory(two_particle_snapshot_factory())
    manifold = manifold_base_params.surface(**manifold_base_params.setup_params)
    nve = hoomd.md.methods.rattle.NVE(filter=hoomd.filter.All(),
                                      manifold_constraint=manifold)
    integrator = hoomd.md.Integrator(0.005, methods=[nve])
    sim.operations += integrator
    pickling_check(manifold)
    sim.run(0)
    pickling_check(manifold)
