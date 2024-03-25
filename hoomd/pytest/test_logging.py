# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

from hoomd.conftest import pickling_check
from pytest import raises, fixture, mark
from hoomd.logging import (_LoggerQuantity, _NamespaceFilter,
                           _SafeNamespaceDict, Logger, Loggable,
                           LoggerCategories, log)
from hoomd.util import _dict_map


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

    @log(category="string", default=False)
    def prop_nondefault(self):
        return "foo"

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
        loggable_list = ['prop', 'proplist', "prop_nondefault"]
        assert set(
            self.dummy_loggable._export_dict.keys()) == set(loggable_list)
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
        assert all([
            p in self.dummy_loggable_inher._export_dict.keys()
            for p in inherented_list
        ])
        assert all([
            p not in self.not_dummy_loggable_inher._export_dict.keys()
            for p in inherented_list
        ])
        assert 'propnotinherented' in \
            self.not_dummy_loggable_inher._export_dict.keys()

    def test_loggables(self):
        dummy_obj = self.dummy_loggable()
        assert dummy_obj.loggables == {
            'prop': 'scalar',
            'proplist': 'sequence',
            'prop_nondefault': 'string'
        }


class TestNamespaceFilter:

    def test_remove_name(self):
        filter_ = _NamespaceFilter(remove_names={"foo", "bar"})
        assert ("baz",) == tuple(filter_(("foo", "bar", "baz")))
        assert () == tuple(filter_(("foo", "bar")))
        assert ("a", "c") == tuple(filter_(("a", "bar", "c")))

    def test_base_names(self):
        filter_ = _NamespaceFilter(base_names={"foo", "bar"})
        assert ("foo", "bar") == tuple(filter_(("foo", "baz", "bar")))
        assert ("foo",) == tuple(filter_(("foo", "bar")))
        assert ("a", "bar") == tuple(filter_(("a", "bar", "c")))

    def test_skip_duplicates(self):
        filter_ = _NamespaceFilter()
        assert ("a",) == tuple(filter_(("a", "a", "a", "a")))
        assert ("a", "b", "a") == tuple(filter_(("a", "a", "b", "a")))
        assert ("a", "b", "a") == tuple(filter_(("a", "b", "a", "a")))

    def test_non_native_remove(self):
        filter_ = _NamespaceFilter(non_native_remove={"foo", "bar"})
        assert (
            "foo",
            "bar",
            "baz",
        ) == tuple(filter_(("foo", "bar", "baz")))
        assert ("baz",) == tuple(filter_(("foo", "bar", "baz"), False))
        assert () == tuple(filter_(("foo", "bar"), False))
        assert ("a", "c") == tuple(filter_(("a", "bar", "c"), False))


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

    mapped_dict = _dict_map(base_dict, func)
    assert mapped_dict == expected_mapped_dict


@fixture
def namespace_dict(base_dict):
    dict_ = _SafeNamespaceDict(base_dict)
    return dict_


@fixture
def blank_namespace_dict():
    return _SafeNamespaceDict()


@fixture
def good_keys():
    return [('a',), ('a', 'b'), ('a', 'b', 'c'), ('a', 'd'), ('e'), ('f'),
            ('f', 'g'), 'a', 'e', 'f']


class TestSafeNamespaceDict:

    def test_contains(self, namespace_dict, good_keys):
        bad_keys = [('z', 'q'), dict(), ('f', 'g', 'h')]
        for key in good_keys:
            assert key in namespace_dict
            assert key in namespace_dict
        for key in bad_keys:
            assert key not in namespace_dict
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
@fixture(params=(
    {},
    {
        "only_default": False
    },
    {
        "categories": LoggerCategories.scalar | LoggerCategories.string
    },
    {
        "only_default": False,
        "categories": ("scalar",)
    },
))
def blank_logger(request):
    return Logger(**request.param)


@fixture
def log_quantity():
    return _LoggerQuantity('example', DummyNamespace)


@fixture
def logged_obj():
    return DummyLoggable()


@fixture
def base_namespace():
    return ('pytest', 'test_logging', 'DummyLoggable')


def nested_getitem(obj, namespace):
    for k in namespace:
        obj = obj[k]
    return obj


class TestLogger:

    def get_filter(self, logger, overwrite_default=False):

        def filter(loggable):
            with_default = not logger.only_default or loggable.default
            return (loggable.category in logger.categories
                    and (with_default or overwrite_default))

        return filter

    def test_setitem(self, blank_logger):

        def check(logger, namespace, loggable):
            if LoggerCategories[loggable[-1]] not in logger.categories:
                with raises(ValueError):
                    logger[namespace] = loggable
                return
            logger[namespace] = loggable
            assert namespace in logger
            log_quantity = nested_getitem(logger, namespace)
            assert log_quantity.obj == loggable[0]
            if len(loggable) == 3:
                assert log_quantity.attr == loggable[1]
            assert log_quantity.category == LoggerCategories[loggable[-1]]

        # Valid values with potentially incompatible categories
        check(blank_logger, 'a', (5, '__eq__', 'scalar'))
        check(blank_logger, ('b', 'c'), (5, '__eq__', 'scalar'))
        check(blank_logger, 'c', (lambda: [1, 2, 3], 'sequence'))
        # Invalid values
        for value in [dict(), list(), None, 5, (5, 2), (5, 2, 1)]:
            with raises(ValueError):
                blank_logger[('c', 'd')] = value
        # Existent key
        extant_key = next(iter(blank_logger.keys()))
        # Requires that scalar category accepted or raises a ValueError
        with raises(KeyError):
            blank_logger[extant_key] = (lambda: 1, 'scalar')

    def test_add_single_quantity(self, blank_logger, log_quantity):
        # Assumes "scalar" is always accepted
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
        log_quantities = list(
            blank_logger._get_loggables_by_name(logged_obj, None))
        log_filter = self.get_filter(blank_logger)
        logged_names = [
            loggable.name
            for loggable in logged_obj._export_dict.values()
            if log_filter(loggable)
        ]
        assert len(log_quantities) == len(logged_names)
        assert all([
            log_quantity.name in logged_names for log_quantity in log_quantities
        ])

        # Check when quantities is given
        accepted_quantities = ['proplist', "prop_nondefault"]
        log_filter = self.get_filter(blank_logger, overwrite_default=True)
        log_quantities = list(
            blank_logger._get_loggables_by_name(logged_obj,
                                                accepted_quantities))
        logged_names = [
            loggable.name
            for loggable in logged_obj._export_dict.values()
            if loggable.name in accepted_quantities and log_filter(loggable)
        ]
        assert len(log_quantities) == len(logged_names)
        assert all([
            log_quantity.name in logged_names for log_quantity in log_quantities
        ])

        # Check when quantities has a bad value
        bad_quantities = ['bad', 'quant']
        with raises(ValueError):
            a = blank_logger._get_loggables_by_name(logged_obj, bad_quantities)
            list(a)

    @mark.parametrize("quantities", ([], [
        "prop",
    ], ['prop', 'proplist', "prop_nondefault"]))
    def test_add(self, blank_logger, logged_obj, base_namespace, quantities):

        if len(quantities) != 0:
            blank_logger.add(logged_obj, quantities)
            log_filter = self.get_filter(blank_logger, overwrite_default=True)
        else:
            blank_logger.add(logged_obj)
            log_filter = self.get_filter(blank_logger)

        expected_namespaces = [
            base_namespace + (loggable.name,)
            for loggable in logged_obj._export_dict.values()
            if log_filter(loggable)
        ]
        if len(quantities) != 0:
            expected_namespaces = [
                name for name in expected_namespaces
                if any(name[-1] in q for q in quantities)
            ]
        assert all(ns in blank_logger for ns in expected_namespaces)
        assert len(blank_logger) == len(expected_namespaces)

    def test_add_with_user_names(self, logged_obj, base_namespace):
        logger = Logger()
        # Test adding a user specified identifier into the namespace
        user_name = 'UserName'
        logger.add(logged_obj, user_name=user_name)
        assert base_namespace[:-1] + (user_name, 'prop') in logger
        assert base_namespace[:-1] + (user_name, 'proplist') in logger

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

    def test_remove_with_user_name(self, logged_obj, base_namespace):
        # Test remove using a user specified namespace identifier
        logger = Logger()
        user_name = 'UserName'
        logger.add(logged_obj, user_name=user_name)
        assert base_namespace[:-1] + (user_name, 'prop') in logger
        assert base_namespace[:-1] + (user_name, 'proplist') in logger

    def test_iadd(self, blank_logger, logged_obj):
        blank_logger.add(logged_obj)
        add_log = blank_logger._dict
        blank_logger._dict = dict()
        blank_logger += logged_obj
        assert add_log == blank_logger._dict
        log_filter = self.get_filter(blank_logger)
        expected_loggables = [
            loggable.name
            for loggable in logged_obj._export_dict.values()
            if log_filter(loggable)
        ]
        assert len(blank_logger) == len(expected_loggables)

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

    def test_iter(self):
        logger = Logger()
        logger[("a", "b", "c")] = (lambda: 4, "scalar")
        logger[("b", "c")] = (lambda: "hello", "string")
        logger[("a", "a")] = (lambda: 17, "scalar")
        keys = list(logger)
        assert len(keys) == 3
        assert ("a",) not in keys
        assert ("a", "b") not in keys
        assert ("b",) not in keys
        assert ("a", "b", "c") in keys
        assert ("a", "a") in keys
        assert ("b", "c") in keys
