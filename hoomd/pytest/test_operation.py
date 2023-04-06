# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

from hoomd.conftest import pickling_check
from hoomd.data.typeparam import TypeParameter
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.data.typeconverter import RequiredArg
from hoomd.pytest.dummy import DummyCppObj, DummySimulation
from hoomd.pytest.dummy import DummyOperation
from copy import deepcopy
from pytest import fixture


def identity(x):
    return x


@fixture(scope='function')
def typeparam():
    return TypeParameter(name='type_param',
                         type_kind='particle_types',
                         param_dict=TypeParameterDict(foo=1,
                                                      bar=identity,
                                                      len_keys=1))


@fixture(scope='function')
def base_op():
    return DummyOperation()


def test_adding_typeparams(typeparam, base_op):
    base_op._add_typeparam(typeparam)
    assert 'type_param' in base_op._typeparam_dict.keys()
    assert base_op._typeparam_dict['type_param']['A'] == \
        dict(foo=1, bar=RequiredArg)
    return base_op


def test_adding_params(base_op):
    base_op._param_dict['param1'] = 1
    base_op._param_dict['param2'] = 2
    return base_op


def test_extending_typeparams(base_op):
    type_params = [
        TypeParameter('1', 'fake1', dict(a=1)),
        TypeParameter('2', 'fake2', dict(a=2)),
        TypeParameter('3', 'fake3', dict(a=3))
    ]
    base_op._extend_typeparam(type_params)
    keys = set(base_op._typeparam_dict.keys())
    expected_keys = set(['1', '2', '3'])
    # That keys are the same
    assert keys.union(expected_keys) == keys and keys - expected_keys == set()
    # That each value is the same
    for tp in type_params:
        assert base_op._typeparam_dict[tp.name].param_dict == tp.param_dict
        assert base_op._typeparam_dict[tp.name].type_kind == tp.type_kind


@fixture(scope='function')
def full_op(typeparam, base_op):
    return test_adding_params(test_adding_typeparams(typeparam, base_op))


def test_resetting_params(base_op):
    base_op._param_dict = dict(param1=1, param2=2)
    assert base_op.param1 == 1
    assert base_op.param2 == 2


def test_getattr(full_op):
    assert type(full_op.type_param) == TypeParameter
    assert full_op.type_param['A'] == dict(foo=1, bar=RequiredArg)
    assert full_op.param1 == 1
    assert full_op.param2 == 2


def test_setattr(full_op):
    new_dict = dict(foo=2, bar=3)
    full_op.type_param = dict(A=new_dict, B=new_dict)
    assert full_op.type_param['A'] == new_dict
    assert full_op.type_param['B'] == new_dict
    full_op.type_param['A'] = dict(foo=3, bar=None)
    assert full_op.type_param['A'] == dict(foo=3, bar=None)
    full_op.param1 = 4.
    assert full_op.param1 == 4.


def test_apply_typeparam_dict(full_op):
    """Tests _apply_typeparam_dict and by necessity getattr."""
    full_op.type_param['A'] = dict(bar='world')
    full_op.type_param['B'] = dict(bar='hello')
    full_op.type_param['C'] = dict(bar='hello world')
    cpp_obj = DummyCppObj()
    full_op._cpp_obj = cpp_obj
    full_op._apply_typeparam_dict(cpp_obj, DummySimulation())
    assert cpp_obj.getTypeParam('A') == dict(foo=1, bar='world')
    assert cpp_obj.getTypeParam('B') == dict(foo=1, bar='hello')
    return full_op


def test_apply_param_dict(full_op):
    """Tests _apply_param_dict and by necessity getattr."""
    full_op = test_apply_typeparam_dict(full_op)
    full_op._apply_param_dict()
    assert full_op._cpp_obj.param1 == 1
    assert full_op._cpp_obj.param2 == 2
    return full_op


@fixture(scope='function')
def attached(full_op):
    cp = deepcopy(full_op)
    op = test_apply_param_dict(cp)
    op._simulation = 1
    return op


def test_attached_setattr(attached):
    attached.type_param['A'] = dict(foo=5., bar='world')
    assert attached._cpp_obj.getTypeParam('A') == dict(foo=5., bar='world')
    attached.param1 = 4
    assert attached._cpp_obj.param1 == 4


def test_attached(full_op, attached):
    assert not full_op._attached
    assert attached._attached


def test_detach(attached):
    detached = attached._detach()
    assert detached.type_param['A'] == dict(foo=1, bar='world')
    assert detached.type_param['B'] == dict(foo=1, bar='hello')
    assert detached.param1 == 1
    assert detached.param2 == 2


def test_pickling(full_op, attached):
    pickling_check(full_op)
    pickling_check(attached)
