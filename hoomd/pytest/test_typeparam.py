# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

from hoomd.conftest import pickling_check
from hoomd.data.typeparam import TypeParameter
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.pytest.dummy import DummyCppObj, DummySimulation
from pytest import fixture, raises


@fixture()
def default_value():
    return {"foo": 1, "bar": 4}


@fixture()
def typedict(default_value):
    return TypeParameterDict(**default_value, len_keys=1)


@fixture(scope='function')
def typeparam(typedict):
    return TypeParameter(name='type_param',
                         type_kind='particle_types',
                         param_dict=typedict)


@fixture()
def attached(typeparam):
    typeparam._attach(DummyCppObj(), DummySimulation().state)
    return typeparam


@fixture()
def detached(attached):
    attached._detach()
    return attached


@fixture(scope='function', params=['typeparam', 'attached', 'detached'])
def all_(request, typeparam, attached, detached):
    if request.param == 'typeparam':
        return typeparam
    elif request.param == 'attached':
        return attached
    else:
        return detached


def test_set_get_item(all_, default_value):
    all_['A'] = {"bar": 2}
    all_['B'] = {"bar": 5}
    assert all_['A'] == {**default_value, "bar": 2}
    assert all_['B'] == {**default_value, "bar": 5}
    assert all_['A']['bar'] == 2
    assert all_['B']['bar'] == 5


def test_setitem_attached(attached, default_value):
    new_value = {"bar": 2}
    attached['A'] = new_value
    assert attached._cpp_obj.getTypeParam("A") == {**default_value, **new_value}


def test_default(all_):
    assert all_.default == all_.param_dict.default
    all_.default = dict(bar=10.)
    assert all_.default == all_.param_dict.default
    assert all_.default['bar'] == 10.
    assert all_.default['foo'] == 1


def test_type_checking(all_):
    bad_inputs = [dict(), dict(A=4), ['A', 4]]
    for input_ in bad_inputs:
        with raises(KeyError):
            all_[input_]


def test_attached_type_checking(attached):
    with raises(KeyError):
        _ = attached['D']
    with raises(KeyError):
        attached['D'] = dict(bar=2)


def test_pickling(all_):
    pickling_check(all_)
