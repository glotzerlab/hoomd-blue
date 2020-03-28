from hoomd.typeparam import TypeParameter
from hoomd.parameterdicts import TypeParameterDict
from hoomd.pytest.dummy import DummyCppObj, DummySimulation
from pytest import fixture, raises


@fixture(scope='function')
def typedict():
    return TypeParameterDict(foo=1, bar=4, len_keys=1)


def test_instantiation(typedict):
    return TypeParameter(name='type_param',
                         type_kind='particle_types',
                         param_dict=typedict
                         )


@fixture(scope='function')
def typeparam(typedict):
    return test_instantiation(typedict)


def test_attach(typeparam):
    return typeparam.attach(DummyCppObj(), DummySimulation())


@fixture(scope='function')
def attached(typeparam):
    return test_attach(typeparam)


def test_detaching(attached):
    return attached.detach()


@fixture(scope='function')
def detached(attached):
    return test_detaching(attached)


@fixture(scope='function', params=['typeparam', 'attached', 'detached'])
def all_(request, typeparam, attached, detached):
    if request.param == 'typeparam':
        return typeparam
    elif request.param == 'attached':
        return attached
    else:
        return detached


def test_set_get_item(all_):
    all_['A'] = dict(bar=2)
    all_['B'] = dict(bar=5)
    assert all_['A'] == all_.param_dict['A']
    assert all_['B'] == all_.param_dict['B']
    assert all_['A']['bar'] == 2
    assert all_['B']['bar'] == 5


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
        _ = attached['C']
    with raises(KeyError):
        attached['C'] = dict(bar=2)
