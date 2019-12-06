from pytest import raises, fixture
from hoomd.pytest.dummy import DummyLoggedObj
from hoomd.logger import LoggerQuantity, NamespaceDict, Logger, dict_map

# Test LoggerQuantity


def test_yield_names():
    namespace = ('baz', 'boo')
    private_namespace = 'bar'
    name = 'foo'
    quantity = LoggerQuantity(name=name, private_namespace=private_namespace,
                              namespace=namespace)
    for i, given_namespace in enumerate(quantity.yield_names()):
        if i == 0:
            assert given_namespace == namespace + (private_namespace, name)
        elif i < 100:
            assert given_namespace[-2].endswith('_' + str(i)) and \
                given_namespace[:-2] == namespace and \
                given_namespace[-2].split('_')[0] == private_namespace and \
                given_namespace[-1] == name
        else:
            break

# Test NamespaceDict class
@fixture
def base_dict():
    return dict(a=dict(b=dict(c=None), d=None), e=None, f=dict(g=None))


@fixture
def expected_mapped_dict():
    return dict(a=dict(b=dict(c=1), d=1), e=1, f=dict(g=1))


def test_dict_map(base_dict, expected_mapped_dict):
    def func(x):
        return 1

    mapped_dict = dict_map(base_dict, func)
    assert mapped_dict == expected_mapped_dict


@fixture
def namespace_dict(base_dict):
    dict_ = NamespaceDict()
    dict_._dict = base_dict
    return dict_


def test_key_exists(namespace_dict):
    good_keys = [('a',), ('a', 'b'), ('a', 'b', 'c'), ('a', 'd'), ('e'),
                 ('f'), ('f', 'g'), 'a', 'e', 'f']
    bad_keys = [('z', 'q'), dict(), ('f', 'g', 'h')]
    for key in good_keys:
        assert namespace_dict.key_exists(key)
    for key in bad_keys:
        assert not namespace_dict.key_exists(key)


@fixture
def blank_namespace_dict():
    return NamespaceDict()


def test_setitem(blank_namespace_dict):
    nsdict = blank_namespace_dict
    nsdict['a'] = 5
    nsdict[('b', 'c')] = None
    assert nsdict._dict['a'] == 5
    assert isinstance(nsdict._dict['b'], dict)
    assert nsdict._dict['b']['c'] is None


# Test Logger
@fixture
def blank_logger():
    return Logger()


def test_logger_setitem(blank_logger):
    logger = blank_logger
    logger['a'] = (5,)
    logger[('b', 'c')] = (5,)
    for value in [dict(), list(), None, 5]:
        with raises(ValueError):
            logger[('c', 'd')] = value
