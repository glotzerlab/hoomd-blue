# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import hoomd
from hoomd.conftest import pickling_check
from hoomd.data.typeparam import TypeParameter
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeconverter import RequiredArg
from hoomd.pytest.dummy import DummyCppObj, DummySimulation
from hoomd.pytest.dummy import DummyOperation
from copy import deepcopy
import pytest


def identity(x):
    return x


@pytest.fixture()
def base_op():
    return DummyOperation()


@pytest.fixture()
def params():
    return {"param1": 2, "param2": "foo"}


def test_adding_params(base_op, params):
    base_op._param_dict.update(params)
    assert base_op.param1 == 2
    assert base_op.param2 == "foo"
    with pytest.raises(ValueError):
        base_op.param2 = 3.0


@pytest.fixture()
def type_param():
    return TypeParameter(name='type_param',
                         type_kind='particle_types',
                         param_dict=TypeParameterDict(foo=1,
                                                      bar=identity,
                                                      len_keys=1))


def test_adding_typeparams(type_param, base_op):
    base_op._add_typeparam(type_param)
    assert 'type_param' in base_op._typeparam_dict.keys()
    expected_dict = {"foo": 1, "bar": RequiredArg}
    assert base_op._typeparam_dict['type_param']['A'] == expected_dict


def test_extending_typeparams(base_op):
    type_params = (TypeParameter('foo', 'particle', {"a": int}),
                   TypeParameter('bar', 'particle', {"a": str}),
                   TypeParameter('baz', 'particle', {"a": 2.5}))

    base_op._extend_typeparam(type_params)
    keys = set(base_op._typeparam_dict.keys())
    expected_keys = {'foo', 'bar', 'baz'}
    # That keys are the same
    assert keys.union(expected_keys) == keys and keys - expected_keys == set()
    # That each value is the same
    for tp in type_params:
        assert base_op._typeparam_dict[tp.name].param_dict == tp.param_dict
        assert base_op._typeparam_dict[tp.name].type_kind == tp.type_kind


@pytest.fixture()
def full_op(base_op, params, type_param):
    base_op._param_dict.update({"param1": 2, "param2": "foo"})
    base_op._add_typeparam(type_param)
    return base_op


def test_getattr(full_op, params, type_param):
    assert type(full_op.type_param) is TypeParameter
    assert full_op.type_param['A'] == type_param["A"]
    for key, param in params.items():
        assert getattr(full_op, key) == param


def test_setattr_type_param(full_op):
    new_dict = {"foo": 2, "bar": 3}
    full_op.type_param = {"A": new_dict, "B": new_dict}
    assert full_op.type_param['A'] == new_dict
    assert full_op.type_param['B'] == new_dict
    new_new_dict = {"foo": 3, "bar": None}
    full_op.type_param['A'] = new_new_dict
    assert full_op.type_param['A'] == new_new_dict


@pytest.fixture()
def type_param_non_default():
    return {
        "A": {
            "bar": "world"
        },
        "B": {
            "bar": "hello"
        },
        "C": {
            "bar": "hello world"
        }
    }


def test_apply_typeparam_dict(full_op, type_param_non_default):
    """Tests _apply_typeparam_dict and by necessity getattr."""
    for key, value in type_param_non_default.items():
        full_op.type_param[key] = value
    cpp_obj = DummyCppObj()
    full_op._cpp_obj = cpp_obj
    full_op._apply_typeparam_dict(cpp_obj, DummySimulation())
    for key, value in type_param_non_default.items():
        expected_dict = {"foo": 1, **value}
        assert cpp_obj.getTypeParam(key) == expected_dict


def test_apply_param_dict(full_op, params):
    """Tests _apply_param_dict and by necessity getattr."""
    cpp_obj = DummyCppObj()
    full_op._cpp_obj = cpp_obj
    full_op._apply_param_dict()
    for key, param in params.items():
        assert getattr(full_op._cpp_obj, key) == param


@pytest.fixture()
def attached(full_op, type_param_non_default):
    op = deepcopy(full_op)
    dummy_sim = DummySimulation()
    cpp_obj = DummyCppObj()
    op._cpp_obj = cpp_obj
    op.type_param.update(type_param_non_default)
    op._apply_typeparam_dict(cpp_obj, dummy_sim)
    op._apply_param_dict()
    # Keep sim around despite weak reference
    op.__simulation = dummy_sim
    return op


def test_attached_setattr(attached):
    attached.type_param['A'] = dict(foo=5., bar='world')
    assert attached._cpp_obj.getTypeParam('A') == dict(foo=5., bar='world')
    attached.param1 = 4
    assert attached._cpp_obj.param1 == 4


def test_attached(full_op, attached):
    assert not full_op._attached
    assert attached._attached


def test_detach(attached, params, type_param_non_default):
    detached = attached._detach()
    for key, value in type_param_non_default.items():
        expected_dict = {"foo": 1, **value}
        assert detached.type_param[key] == expected_dict
    for key, param in params.items():
        assert getattr(detached, key) == param


def test_pickling(full_op, attached):
    pickling_check(full_op)
    sim = attached.__simulation  # noqa: F841
    del attached.__simulation
    pickling_check(attached)


def test_operation_lifetime(simulation_factory, two_particle_snapshot_factory):

    def drop_sim(attach=False):
        sim = simulation_factory(two_particle_snapshot_factory())
        # Use operation available regardless of build
        box_resize = hoomd.update.BoxResize(
            10,
            hoomd.variant.box.Interpolate(hoomd.Box.cube(4), hoomd.Box.cube(5),
                                          hoomd.variant.Ramp(0, 1, 0, 10_000)))
        sim.operations.updaters.append(box_resize)
        if attach:
            sim.run(0)
        return box_resize

    box_resize = drop_sim()
    assert box_resize._simulation is None

    box_resize = drop_sim(True)
    assert box_resize._simulation is None
    assert box_resize._cpp_obj is None
