from pytest import raises, fixture
from hoomd.logger import LoggerQuantity, SafeNamespaceDict, Logger, dict_map
from hoomd.logger import generate_namespace
from hoomd.logger import Loggable


class DummyNamespace:
    pass


@fixture(scope='module')
def dummy_namespace():
    return ('pytest', 'test_logger', 'DummyNamespace')


# ------- Test LoggerQuantity
class TestLoggerQuantity:
    def test_initialization(self, dummy_namespace):
        logquant = LoggerQuantity('foo', DummyNamespace, flag='particle')
        assert logquant.flag == 'particle'
        assert logquant.name == 'foo'
        assert logquant.namespace == dummy_namespace

    def test_yield_names(self, dummy_namespace):
        name = 'foo'
        quantity = LoggerQuantity(name=name, cls=DummyNamespace)
        for i, given_namespace in enumerate(quantity.yield_names()):
            if i == 0:
                assert given_namespace == dummy_namespace + (name,)
            elif i < 100:
                assert given_namespace[-2].endswith('_' + str(i)) and \
                    given_namespace[:-2] == dummy_namespace[:-1] and \
                    given_namespace[-2].split('_')[0] == \
                    dummy_namespace[-1] and given_namespace[-1] == name
            else:
                break


# ------- Test generate_logger_decorator and associated functions
def test_generate_namespace():
    assert generate_namespace(TestLoggerQuantity) == \
        ('pytest', 'test_logger', 'TestLoggerQuantity')


class DummyLoggable(metaclass=Loggable):

    @Loggable.log
    def prop(self):
        return 1

    @Loggable.log(flag='multi')
    def proplist(self):
        return [1, 2, 3]

    @Loggable.log(flag='dict')
    def propdict(self):
        return dict(a=(2, 'scalar'), b=([2, 3, 4], 'multi'))


class TestLoggableMetaclass():

    dummy_loggable = DummyLoggable

    class InherentedDummyLoggable(DummyLoggable):

        @Loggable.log
        def propinherented(self):
            return None

    dummy_loggable_inher = InherentedDummyLoggable

    class NotInherentedDummy(metaclass=Loggable):

        @Loggable.log
        def propnotinherented(self):
            return True

    not_dummy_loggable_inher = NotInherentedDummy

    def test_logger_functor_application(self):
        loggable_list = ['prop', 'proplist', 'propdict']
        assert set(self.dummy_loggable._export_dict.keys()
                   ) == set(loggable_list)
        expected_namespace = generate_namespace(self.dummy_loggable)
        expected_flags = ['scalar', 'multi', 'dict']
        for loggable, flag in zip(loggable_list, expected_flags):
            log_quantity = self.dummy_loggable._export_dict[loggable]
            assert log_quantity.namespace == expected_namespace
            assert log_quantity.flag == flag
            assert log_quantity.name == loggable

    def test_loggable_inherentence(self):
        inherented_list = ['prop', 'proplist', 'propdict', 'propinherented']
        assert 'propinherented' not in self.dummy_loggable._export_dict.keys()
        assert all([p in self.dummy_loggable_inher._export_dict.keys()
                    for p in inherented_list])
        assert all([p not in self.not_dummy_loggable_inher._export_dict.keys()
                    for p in inherented_list])
        assert 'propnotinherented' in \
            self.not_dummy_loggable_inher._export_dict.keys()


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
    dict_ = SafeNamespaceDict(base_dict)
    return dict_


@fixture
def blank_namespace_dict():
    return SafeNamespaceDict()


@fixture
def good_keys():
    return [('a',), ('a', 'b'), ('a', 'b', 'c'), ('a', 'd'), ('e'),
            ('f'), ('f', 'g'), 'a', 'e', 'f']


class TestSafeNamespaceDict:

    def test_key_exists(self, namespace_dict, good_keys):
        bad_keys = [('z', 'q'), dict(), ('f', 'g', 'h')]
        for key in good_keys:
            assert namespace_dict.key_exists(key)
            assert key in namespace_dict
        for key in bad_keys:
            assert not namespace_dict.key_exists(key)
            assert key not in namespace_dict

    def test_setitem(self, blank_namespace_dict):
        nsdict = blank_namespace_dict
        nsdict['a'] = 5
        nsdict[('b', 'c')] = None
        assert nsdict._dict['a'] == 5
        assert isinstance(nsdict._dict['b'], dict)
        assert nsdict._dict['b']['c'] is None

    def test_delitem(self, namespace_dict):
        keys = [('a', 'b', 'c'), 'a']
        keys_left = [('e',), ('f',), ('f', 'g'), 'e', 'f']
        for key in keys:
            assert key in namespace_dict
            del namespace_dict[key]
            assert key not in namespace_dict
        assert ('a', 'b') not in namespace_dict
        for key in keys_left:
            assert key in namespace_dict

    def test_len(self, namespace_dict, blank_namespace_dict):
        assert len(namespace_dict) == 4
        assert len(blank_namespace_dict) == 0
        blank_namespace_dict['a'] = 1
        assert len(blank_namespace_dict) == 1


# ------ Test Logger
@fixture
def blank_logger():
    return Logger()


@fixture
def log_quantity():
    return LoggerQuantity('example', DummyNamespace)


@fixture
def logged_obj():
    return DummyLoggable()


@fixture
def base_namespace():
    return ('pytest', 'test_logger', 'DummyLoggable')


class TestLogger:
    def test_setitem(self, blank_logger):
        logger = blank_logger
        logger['a'] = (5, 2, 1)
        logger[('b', 'c')] = (5, 2, 1)
        for value in [dict(), list(), None, 5, (5, 2)]:
            with raises(ValueError):
                logger[('c', 'd')] = value

    def test_add_single_quantity(self, blank_logger, log_quantity):
        blank_logger._add_single_quantity(None, log_quantity)
        namespace = log_quantity.namespace + (log_quantity.name,)
        assert namespace in blank_logger
        log_value = blank_logger[namespace]
        assert log_value[0] is None
        assert log_value[1] == log_quantity.name
        assert log_value[2] == log_quantity.flag
        blank_logger._add_single_quantity([], log_quantity)
        namespace = log_quantity.namespace[:-1] + \
            (log_quantity.namespace[-1] + '_1', log_quantity.name)
        assert namespace in blank_logger

    def test_grab_log_quantities_from_names(self, blank_logger, logged_obj):

        # Check when quantities is None
        log_quanities = blank_logger._grab_log_quantities_from_names(
            logged_obj, None)
        logged_names = ['prop', 'propdict', 'proplist']
        assert all([log_quantity.name in logged_names
                    for log_quantity in log_quanities])

        # Check when quantities is given
        accepted_quantities = ['prop', 'proplist']
        log_quanities = blank_logger._grab_log_quantities_from_names(
            logged_obj, accepted_quantities)
        assert all([log_quantity.name in accepted_quantities
                    for log_quantity in log_quanities])

        # Check when quantities has a bad value
        bad_quantities = ['bad', 'quant']
        with raises(KeyError):
            blank_logger._grab_log_quantities_from_names(
                logged_obj, bad_quantities)

    def test_flags_checks(self, blank_logger):
        scalar = LoggerQuantity('name', DummyNamespace, flag='scalar')
        multi = LoggerQuantity('name', DummyNamespace, flag='multi')
        particle = LoggerQuantity('name', DummyNamespace, flag='particle')
        assert all([blank_logger.flag_checks(log_quantity)
                    for log_quantity in [scalar, multi, particle]])
        blank_logger._flags.append('scalar')
        assert blank_logger.flag_checks(scalar)
        assert not blank_logger.flag_checks(multi) and \
            not blank_logger.flag_checks(particle)
        blank_logger._flags.append('particle')
        assert blank_logger.flag_checks(scalar) and \
            blank_logger.flag_checks(particle)
        assert not blank_logger.flag_checks(multi)

    def test_add(self, blank_logger, logged_obj, base_namespace):

        # Test adding everything
        namespaces = blank_logger.add(logged_obj)
        expected_namespaces = [base_namespace + ('prop',),
                               base_namespace + ('proplist',),
                               base_namespace + ('propdict',)]
        assert set(namespaces) == set(expected_namespaces)
        assert all([ens in blank_logger for ens in expected_namespaces])
        assert len(blank_logger) == 3

        # Test adding specific quantity
        blank_logger._dict = dict()
        namespaces = blank_logger.add(logged_obj, 'prop')
        expected_namespace = base_namespace + ('prop',)
        assert set(namespaces) == set([expected_namespace])
        assert expected_namespace in blank_logger
        assert len(blank_logger) == 1

        # Test multiple quantities
        blank_logger._dict = dict()
        namespaces = blank_logger.add(logged_obj, ['prop', 'proplist'])
        expected_namespaces = [base_namespace + ('prop',),
                               base_namespace + ('proplist',)]
        assert set(namespaces) == set(expected_namespaces)
        assert all([ens in blank_logger for ens in expected_namespaces])
        assert len(blank_logger) == 2

        # Test with flag
        blank_logger._dict = dict()
        blank_logger._flags.append('scalar')
        namespaces = blank_logger.add(logged_obj)
        expected_namespace = base_namespace + ('prop',)
        assert set(namespaces) == set([expected_namespace])
        assert expected_namespace in blank_logger
        assert len(blank_logger) == 1

    def test_add_with_flags(self, blank_logger, logged_obj, base_namespace):
        blank_logger._flags.append('scalar')
        # Test adding everything should filter non-scalar
        namespaces = blank_logger.add(logged_obj)
        expected_namespace = base_namespace + ('prop',)
        assert set(namespaces) == set([expected_namespace])
        blank_logger._flags.append('multi')
        expected_namespace = base_namespace + ('proplist',)
        namespaces = blank_logger.add(logged_obj)
        assert expected_namespace in blank_logger
        assert expected_namespace in namespaces
        assert len(blank_logger) == 2

    def test_remove(self, logged_obj, base_namespace):

        # Test removing all properties
        prop_namespace = base_namespace + ('prop',)
        list_namespace = base_namespace + ('proplist',)
        dict_namespace = base_namespace + ('propdict',)
        log = Logger()
        log.add(logged_obj)
        log.remove(logged_obj)
        assert len(log) == 0
        assert prop_namespace not in log
        assert list_namespace not in log
        assert dict_namespace not in log

        # Test removing select properties
        log = Logger()
        log.add(logged_obj)
        log.remove(logged_obj, 'prop')
        assert len(log) == 2
        assert prop_namespace not in log
        assert list_namespace in log
        assert dict_namespace in log

        log = Logger()
        log.add(logged_obj)
        log.remove(logged_obj, ['prop', 'proplist'])
        assert len(log) == 1
        assert prop_namespace not in log
        assert list_namespace not in log
        assert dict_namespace in log

        # Test remove just given namespaces
        prop_namespace = ('pytest', 'test_logger',
                          'DummyLoggable', 'prop')
        log = Logger()
        log.add(logged_obj)
        log.remove(quantities=[prop_namespace])
        assert len(log) == 2
        assert prop_namespace not in log
        assert list_namespace in log
        assert dict_namespace in log

        # Test remove when not in initial namespace
        log = Logger()
        log[prop_namespace] = (None, 'eg', 'scalar')
        log.add(logged_obj)
        assert len(log) == 4
        log.remove(logged_obj, 'prop')
        assert len(log) == 3
        assert prop_namespace in log
        assert list_namespace in log
        assert dict_namespace in log
        assert prop_namespace[:-2] + (prop_namespace[-2] + '_1',
                                      prop_namespace[-1]) not in log

    def test_iadd(self, blank_logger, logged_obj):
        blank_logger.add(logged_obj)
        add_log = blank_logger._dict
        blank_logger._dict = dict()
        blank_logger += logged_obj
        assert add_log == blank_logger._dict
        assert len(blank_logger) == 3

    def test_isub(self, logged_obj, base_namespace):

        # Test when given string
        log = Logger()
        log += logged_obj
        log -= 'pytest'
        assert len(log) == 0
        log += logged_obj
        log -= 'eg'
        assert len(log) == 3

        # Test when given a namespace
        log -= base_namespace + ('prop',)
        assert base_namespace + ('prop',) not in log
        assert len(log) == 2

        # Test with list of namespaces
        log += logged_obj
        rm_nsp = [base_namespace + (name,) for name in ['prop', 'proplist']]
        log -= rm_nsp
        for nsp in rm_nsp:
            assert nsp not in log
        assert len(log) == 1

        # Test with obj
        log += logged_obj
        log -= logged_obj
        assert len(log) == 0

    def test_log(self, logged_obj):
        log = Logger()
        log += logged_obj
        logged = log.log()
        inner_dict = logged['pytest']['test_logger']['DummyLoggable']
        assert inner_dict['prop'] == (logged_obj.prop, 'scalar')
        assert inner_dict['proplist'] == (logged_obj.proplist, 'multi')
        assert inner_dict['propdict'] == logged_obj.propdict
