from hoomd.conftest import pickling_check
from pytest import raises, fixture
from hoomd.logging import (
    _LoggerQuantity, SafeNamespaceDict, Logger, dict_map, Loggable, LoggerCategories,
    log)


class DummyNamespace:
    pass


@fixture(scope='module')
def dummy_namespace():
    return ('pytest', 'test_logging', 'DummyNamespace')


# ------- Test _LoggerQuantity
class TestLoggerQuantity:
    def test_initialization(self, dummy_namespace):
        logquant = _LoggerQuantity('foo', DummyNamespace, category='particle')
        assert logquant.category == LoggerCategories['particle']
        assert logquant.name == 'foo'
        assert logquant.namespace == dummy_namespace

    def test_yield_names(self, dummy_namespace):
        name = 'foo'
        quantity = _LoggerQuantity(name=name, cls=DummyNamespace)
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
        user_defined_namespace = next(quantity.yield_names('USER'))
        assert user_defined_namespace == dummy_namespace[:-1] + ('USER', name)

    def test_generate_namespace(self):
        assert _LoggerQuantity._generate_namespace(TestLoggerQuantity) == \
            ('pytest', 'test_logging', 'TestLoggerQuantity')


class DummyLoggable(metaclass=Loggable):

    @log
    def prop(self):
        return 1

    @log(category='sequence')
    def proplist(self):
        return [1, 2, 3]

    def __eq__(self, other):
        return isinstance(other, type(self))


class TestLoggableMetaclass():

    dummy_loggable = DummyLoggable

    class InherentedDummyLoggable(DummyLoggable):

        @log
        def propinherented(self):
            return None

    dummy_loggable_inher = InherentedDummyLoggable

    class NotInherentedDummy(metaclass=Loggable):

        @log
        def propnotinherented(self):
            return True

    not_dummy_loggable_inher = NotInherentedDummy

    def test_logger_functor_application(self):
        loggable_list = ['prop', 'proplist']
        assert set(self.dummy_loggable._export_dict.keys()
                   ) == set(loggable_list)
        expected_namespace = _LoggerQuantity._generate_namespace(
            self.dummy_loggable)
        expected_categories = ['scalar', 'sequence']
        for loggable, category in zip(loggable_list, expected_categories):
            log_quantity = self.dummy_loggable._export_dict[loggable]
            assert log_quantity.namespace == expected_namespace
            assert log_quantity.category == LoggerCategories[category]
            assert log_quantity.name == loggable

    def test_loggable_inherentence(self):
        inherented_list = ['prop', 'proplist', 'propinherented']
        assert 'propinherented' not in self.dummy_loggable._export_dict.keys()
        assert all([p in self.dummy_loggable_inher._export_dict.keys()
                    for p in inherented_list])
        assert all([p not in self.not_dummy_loggable_inher._export_dict.keys()
                    for p in inherented_list])
        assert 'propnotinherented' in \
            self.not_dummy_loggable_inher._export_dict.keys()

    def test_loggables(self):
        dummy_obj = self.dummy_loggable()
        assert dummy_obj.loggables == {'prop': 'scalar', 'proplist': 'sequence'}


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
    return _LoggerQuantity('example', DummyNamespace)


@fixture
def logged_obj():
    return DummyLoggable()


@fixture
def base_namespace():
    return ('pytest', 'test_logging', 'DummyLoggable')


class TestLogger:
    def test_setitem(self, blank_logger):
        logger = blank_logger
        logger['a'] = (5, '__eq__', 'scalar')
        logger[('b', 'c')] = (5, '__eq__', 'scalar')
        logger['c'] = (lambda: [1, 2, 3], 'sequence')
        for value in [dict(), list(), None, 5, (5, 2), (5, 2, 1)]:
            with raises(ValueError):
                logger[('c', 'd')] = value
        with raises(KeyError):
            logger['a'] = (lambda: [1, 2, 3], 'sequence')

    def test_add_single_quantity(self, blank_logger, log_quantity):
        blank_logger._add_single_quantity(None, log_quantity, None)
        namespace = log_quantity.namespace + (log_quantity.name,)
        assert namespace in blank_logger
        log_value = blank_logger[namespace]
        assert log_value.obj is None
        assert log_value.attr == log_quantity.name
        assert log_value.category == log_quantity.category
        blank_logger._add_single_quantity([], log_quantity, None)
        namespace = log_quantity.namespace[:-1] + \
            (log_quantity.namespace[-1] + '_1', log_quantity.name)
        assert namespace in blank_logger

    def test_get_loggables_by_names(self, blank_logger, logged_obj):
        # Check when quantities is None
        log_quanities = blank_logger._get_loggables_by_name(
            logged_obj, None)
        logged_names = ['prop', 'proplist']
        assert all([log_quantity.name in logged_names
                    for log_quantity in log_quanities])

        # Check when quantities is given
        accepted_quantities = ['prop', 'proplist']
        log_quanities = blank_logger._get_loggables_by_name(
            logged_obj, accepted_quantities)
        assert all([log_quantity.name in accepted_quantities
                    for log_quantity in log_quanities])

        # Check when quantities has a bad value
        bad_quantities = ['bad', 'quant']
        with raises(ValueError):
            a = blank_logger._get_loggables_by_name(logged_obj, bad_quantities)
            list(a)

    def test_add(self, blank_logger, logged_obj, base_namespace):

        # Test adding everything
        blank_logger.add(logged_obj)
        expected_namespaces = [base_namespace + ('prop',),
                               base_namespace + ('proplist',)]
        assert all(ns in blank_logger for ns in expected_namespaces)
        assert len(blank_logger) == 2

        # Test adding specific quantity
        blank_logger._dict = dict()
        blank_logger.add(logged_obj, 'prop')
        expected_namespace = base_namespace + ('prop',)
        assert expected_namespace in blank_logger
        assert len(blank_logger) == 1

        # Test multiple quantities
        blank_logger._dict = dict()
        blank_logger.add(logged_obj, ['prop', 'proplist'])
        expected_namespaces = [base_namespace + ('prop',),
                               base_namespace + ('proplist',)]
        assert all([ns in blank_logger for ns in expected_namespaces])
        assert len(blank_logger) == 2

        # Test with category
        blank_logger._dict = dict()
        blank_logger._categories = LoggerCategories['scalar']
        blank_logger.add(logged_obj)
        expected_namespace = base_namespace + ('prop',)
        assert expected_namespace in blank_logger
        assert len(blank_logger) == 1

    def test_add_with_user_names(
            self, blank_logger, logged_obj, base_namespace):
        # Test adding a user specified identifier into the namespace
        user_name = 'UserName'
        blank_logger.add(logged_obj, user_name=user_name)
        assert base_namespace[:-1] + (user_name, 'prop') in blank_logger
        assert base_namespace[:-1] + (user_name, 'proplist') in blank_logger

    def test_add_with_categories(self, blank_logger, logged_obj, base_namespace):
        blank_logger._categories = LoggerCategories['scalar']
        # Test adding everything should filter non-scalar
        blank_logger.add(logged_obj)
        expected_namespace = base_namespace + ('prop',)
        assert expected_namespace in blank_logger
        blank_logger._categories = LoggerCategories['sequence']
        expected_namespace = base_namespace + ('proplist',)
        blank_logger.add(logged_obj)
        assert expected_namespace in blank_logger
        assert len(blank_logger) == 2

    def test_remove(self, logged_obj, base_namespace):

        # Test removing all properties
        prop_namespace = base_namespace + ('prop',)
        list_namespace = base_namespace + ('proplist',)
        log = Logger()
        log.add(logged_obj)
        log.remove(logged_obj)
        assert len(log) == 0
        assert prop_namespace not in log
        assert list_namespace not in log

        # Test removing select properties
        log = Logger()
        log.add(logged_obj)
        log.remove(logged_obj, 'prop')
        assert len(log) == 1
        assert prop_namespace not in log
        assert list_namespace in log

        log = Logger()
        log.add(logged_obj)
        log.remove(logged_obj, ['prop', 'proplist'])
        assert len(log) == 0
        assert prop_namespace not in log
        assert list_namespace not in log

        # Test remove just given namespaces
        prop_namespace = base_namespace + ('prop',)
        log = Logger()
        log.add(logged_obj)
        log.remove(quantities=[prop_namespace])
        assert len(log) == 1
        assert prop_namespace not in log
        assert list_namespace in log

        # Test remove when not in initial namespace
        log = Logger()
        log[prop_namespace] = (lambda: None, '__call__', 'scalar')
        log.add(logged_obj)
        assert len(log) == 3
        log.remove(logged_obj, 'prop')
        assert len(log) == 2
        assert prop_namespace in log
        assert list_namespace in log
        assert prop_namespace[:-2] + (prop_namespace[-2] + '_1',
                                      prop_namespace[-1]) not in log

    def test_remove_with_user_name(
            self, blank_logger, logged_obj, base_namespace):
        # Test remove using a user specified namespace identifier
        user_name = 'UserName'
        blank_logger.add(logged_obj, user_name=user_name)
        assert base_namespace[:-1] + (user_name, 'prop') in blank_logger
        assert base_namespace[:-1] + (user_name, 'proplist') in blank_logger

    def test_iadd(self, blank_logger, logged_obj):
        blank_logger.add(logged_obj)
        add_log = blank_logger._dict
        blank_logger._dict = dict()
        blank_logger += logged_obj
        assert add_log == blank_logger._dict
        assert len(blank_logger) == 2

    def test_isub(self, logged_obj, base_namespace):

        # Test when given string
        log = Logger()
        log += logged_obj
        log -= 'pytest'
        assert len(log) == 0
        log += logged_obj
        log -= 'eg'
        assert len(log) == 2

        # Test when given a namespace
        log -= base_namespace + ('prop',)
        assert base_namespace + ('prop',) not in log
        assert len(log) == 1

        # Test with list of namespaces
        log += logged_obj
        rm_nsp = [base_namespace + (name,) for name in ['prop', 'proplist']]
        log -= rm_nsp
        for nsp in rm_nsp:
            assert nsp not in log
        assert len(log) == 0

        # Test with obj
        log += logged_obj
        log -= logged_obj
        assert len(log) == 0

    def test_log(self, logged_obj):
        log = Logger()
        log += logged_obj
        logged = log.log()
        inner_dict = logged['pytest']['test_logging']['DummyLoggable']
        assert inner_dict['prop'] == (logged_obj.prop, 'scalar')
        assert inner_dict['proplist'] == (logged_obj.proplist, 'sequence')

    def test_pickling(self, blank_logger, logged_obj):
        blank_logger.add(logged_obj)
        pickling_check(blank_logger)
