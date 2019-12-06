from pytest import raises, fixture
from hoomd.logger import LoggerQuantity, SafeNamespaceDict, Logger, dict_map
from hoomd.logger import generate_namespace
from hoomd.logger import Loggable


# ------- Test LoggerQuantity
class TestLoggerQuantity:
    def test_initialization(self):
        logquant = LoggerQuantity('foo', ('bar', 'boo' 'baz'), flag='particle')
        assert logquant.flag == 'particle'
        assert logquant.name == 'foo'
        assert logquant.namespace == ('bar', 'boo' 'baz')

    def test_yield_names(self):
        namespace = ('bar', 'boo', 'baz')
        name = 'foo'
        quantity = LoggerQuantity(name=name, namespace=namespace)
        for i, given_namespace in enumerate(quantity.yield_names()):
            if i == 0:
                assert given_namespace == namespace + (name,)
            elif i < 100:
                assert given_namespace[-2].endswith('_' + str(i)) and \
                    given_namespace[:-2] == namespace[:-1] and \
                    given_namespace[-2].split('_')[0] == namespace[-1] and \
                    given_namespace[-1] == name
            else:
                break


# ------- Test generate_logger_decorator and associated functions
def test_generate_namespace():
    assert generate_namespace(TestLoggerQuantity) == \
        ('hoomd', 'pytest', 'test_logger', 'TestLoggerQuantity')

class TestLoggableMetaclass():

    class DummyLoggable(metaclass=Loggable):

        @Loggable.log
        def prop(self):
            return 1

        @Loggable.log(flag='array')
        def proplist(self):
            return [1, 2, 3]

        @Loggable.log(flag='dict')
        def propdict(self):
            return dict(a=(2, 'scalar'), b=([2, 3, 4], 'array'))

    dummy_loggable = DummyLoggable

    class InherentedDummyLoggable(DummyLoggable):

        @Loggable.log
        def propinherented(self):
            return None

    dummy_loggable_inher = InherentedDummyLoggable

    def test_logger_functor_application(self):
        loggable_list = ['prop', 'proplist', 'propdict']
        assert set(self.dummy_loggable._export_dict.keys()) == set(loggable_list)
        expected_namespace = generate_namespace(self.dummy_loggable)
        expected_flags = ['scalar', 'array', 'dict']
        for loggable, flag in zip(loggable_list, expected_flags):
            log_quantity = self.dummy_loggable._export_dict[loggable]
            assert log_quantity.namespace == expected_namespace
            assert log_quantity.flag == flag
            assert log_quantity.name == loggable

    def test_loggable_inherentence(self):
        assert 'propinherented' not in self.dummy_loggable._export_dict.keys()
        assert all([p in self.dummy_loggable_inher._export_dict.keys()
                    for p in ['prop', 'proplist', 'propdict', 'propinherented']
                    ])


# ------- Test dict_map function
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
    dict_ = SafeNamespaceDict()
    dict_._dict = base_dict
    return dict_


@fixture
def blank_namespace_dict():
    return SafeNamespaceDict()


class TestSafeNamespaceDict:

    def test_key_exists(self, namespace_dict):
        good_keys = [('a',), ('a', 'b'), ('a', 'b', 'c'), ('a', 'd'), ('e'),
                     ('f'), ('f', 'g'), 'a', 'e', 'f']
        bad_keys = [('z', 'q'), dict(), ('f', 'g', 'h')]
        for key in good_keys:
            assert namespace_dict.key_exists(key)
        for key in bad_keys:
            assert not namespace_dict.key_exists(key)

    def test_setitem(self, blank_namespace_dict):
        nsdict = blank_namespace_dict
        nsdict['a'] = 5
        nsdict[('b', 'c')] = None
        assert nsdict._dict['a'] == 5
        assert isinstance(nsdict._dict['b'], dict)
        assert nsdict._dict['b']['c'] is None


# ------ Test Logger
@fixture
def blank_logger():
    return Logger()


class TestLogger:
    def test_logger_setitem(self, blank_logger):
        logger = blank_logger
        logger['a'] = (5,)
        logger[('b', 'c')] = (5,)
        for value in [dict(), list(), None, 5]:
            with raises(ValueError):
                logger[('c', 'd')] = value
