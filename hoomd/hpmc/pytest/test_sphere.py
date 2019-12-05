import hoomd
import hoomd.hpmc
from hoomd.hpmc import _hpmc
import pytest


def test_sphere():

    args_1 = {'diameter': 1, 'orientable': 0, 'ignore_statistics': 1}
    test_sphere1 = _hpmc.sph_params(args_1)
    test_dict1 = test_sphere1.asDict()
    assert test_dict1 == args_1

    args_2 = {'diameter': 9, 'orientable': 1, 'ignore_statistics': 1}
    test_sphere2 = _hpmc.sph_params(args_2)
    test_dict2 = test_sphere2.asDict()
    assert test_dict2 == args_2

    args_3 = {'diameter': 4, 'orientable': 0, 'ignore_statistics': 0}
    test_sphere3 = _hpmc.sph_params(args_3)
    test_dict3 = test_sphere3.asDict()
    assert test_dict3 == args_3


def test_shape_params():

    mc = hoomd.hpmc.integrate.Sphere(23456)

    mc.shape['A'] = dict()
    assert mc.shape['A']['diameter'] is None
    assert mc.shape['A']['ignore_statistics'] is False
    assert mc.shape['A']['orientable'] is False

    mc.shape['B'] = dict(diameter=2.5)
    assert mc.shape['B']['diameter'] == 2.5
    assert mc.shape['B']['ignore_statistics'] is False
    assert mc.shape['B']['orientable'] is False

    mc.shape['B'] = dict(ignore_statistics=True, orientable=True)
    assert mc.shape['B']['diameter'] is None
    assert mc.shape['B']['ignore_statistics'] is True
    assert mc.shape['B']['orientable'] is True


def test_shape_params_attached(device, dummy_simulation_factory):

    mc = hoomd.hpmc.integrate.Sphere(23456)
    mc.shape['A'] = dict(diameter=1.25)
    mc.shape['B'] = dict(diameter=4.125,
                         ignore_statistics=True,
                         orientable=False)
    mc.shape['C'] = dict(diameter=2.5,
                         ignore_statistics=False,
                         orientable=True)

    sim = dummy_simulation_factory(particle_types=['A', 'B', 'C'])
    sim.operations.add(mc)
    sim.operations.schedule()

    assert mc.shape['A']['diameter'] == 1.25
    assert mc.shape['A']['ignore_statistics'] is False
    assert mc.shape['A']['orientable'] is False

    assert mc.shape['B']['diameter'] == 4.125
    assert mc.shape['B']['ignore_statistics'] is True
    assert mc.shape['B']['orientable'] is False

    assert mc.shape['C']['diameter'] == 2.5
    assert mc.shape['C']['ignore_statistics'] is False
    assert mc.shape['C']['orientable'] is True

    # check for errors on invalid input
    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(diameter='invalid')

    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(diameter=[1, 2, 3])

    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(diameter=1, ignore_statistics='invalid')

    with pytest.raises(RuntimeError):
        mc.shape['A'] = dict(diameter=1, orientable='invalid')
