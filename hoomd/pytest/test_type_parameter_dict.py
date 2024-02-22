# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import itertools

import pytest

from hoomd.conftest import BaseMappingTest, Either, pickling_check
from hoomd.data.parameterdicts import TypeParameterDict
from hoomd.pytest.dummy import DummyCppObj
from hoomd.data.collections import _HOOMDSyncedCollection
from hoomd.data.typeconverter import RequiredArg
from hoomd.error import TypeConversionError, IncompleteSpecificationError


def identity(x):
    return x


class TestTypeParameterDict(BaseMappingTest):
    _deletion_error = NotImplementedError
    _has_default = True

    @pytest.fixture(params=(1, 2, 3), ids=lambda x: f"n={x}")
    def n(self, request):
        return request.param

    @pytest.fixture(autouse=True, params=(1, 2), ids=lambda x: f"len_keys={x}")
    def len_keys(self, request):
        self._len_keys = request.param
        return request.param

    @pytest.fixture(autouse=True, params=("dict", "int"))
    def spec(self, request):
        self._spec = request.param
        return request.param

    def _generate_keys(self, n):
        if self._len_keys == 1:
            yield from self.alphabet[:n]
        else:
            yield from set(
                tuple(sorted(key))
                for key in itertools.combinations_with_replacement(
                    self.alphabet[:n], 2))

    def _generate_value(self):
        if self._spec == "int":
            return self.generator.int()
        value = {}
        for key in ("foo", "bar", "baz"):
            if self.generator.rng.random() > 0.5:
                continue
            if key == "foo":
                value["foo"] = self.generator.int()
            elif key == "bar":
                value["bar"] = self.generator(Either(int, None, float, str))
            elif key == "baz":
                value["baz"] = self.generator.str()
            else:
                value["gar"] = self.generator([int])
        return value

    @pytest.fixture
    def generate_plain_collection(self, len_keys):

        def generate(n):
            return {
                key: self._generate_value() for key in self._generate_keys(n)
            }

        return generate

    def is_equal(self, test_type_param, mapping):
        """Assert that the keys in mapping have equivalent values in type param.

        We don't care if their are keys in test_type_param that exist but don't
        in mapping since we have defaults that take effect.
        """
        if isinstance(test_type_param, (str, tuple)) or self._spec == "int":
            return test_type_param == mapping
        return all(
            test_type_param[key] == value for key, value in mapping.items())

    @pytest.fixture
    def empty_collection(self, len_keys, spec):
        """Return an empty type parameter."""
        validator = int
        if spec == "dict":
            validator = {
                "foo": 1,
                "bar": identity,
                "baz": "hello",
                "gar": [int]
            }
        return TypeParameterDict(validator, len_keys=len_keys)

    def check_equivalent(self, test_mapping, other):
        # We overwrite check_equivalent to enable us to check exact equality
        # using the default.
        assert set(test_mapping) == other.keys()
        default = test_mapping.default
        for key, value in other.items():
            if self._spec == "dict":
                value = {**default, **value}
            assert test_mapping[key] == value

    def random_keys(self):
        if self._len_keys == 1:
            yield from super().random_keys()
        else:
            yield from (tuple(sorted(k)) for k in zip(super().random_keys(),
                                                      super().random_keys()))

    @pytest.fixture(params=(True, False),
                    ids=lambda x: "in_map" if x else "out_map")
    def setitem_key_value(self, n, request):
        keys = list(self._generate_keys(n))
        value = self._generate_value()
        if request.param:
            return keys[self.generator.int(len(keys))], value
        key = next(filter(lambda x: x not in keys, self.random_keys()))
        return key, value

    @pytest.fixture
    def setdefault_key_value(self, setitem_key_value):
        return setitem_key_value

    def test_pickling(self, populated_collection):
        pickling_check(populated_collection[0])

    @pytest.fixture
    def valid_key_specs(self, n):
        types = list(self.alphabet[:n])

        def yield_key_pairs(len_keys):
            if len_keys == 1:
                yield types, set(types)
                return
            yield (types[0], types[1:]), {(types[0], t) for t in types[1:]}
            keys = [(types[-1], t) for t in types[:2]]
            yield keys, set(tuple(sorted(k)) for k in keys)
            if len(types) > 4:
                mid_point = len(types) // 2
                key_spec = (types[:mid_point], types[mid_point:])
                keys = {(t1, t2)
                        for t2 in types[mid_point:]
                        for t1 in types[mid_point:]}
                yield key_spec, keys

        return yield_key_pairs

    def _invalid_key_specs(self, n):
        types = list(self.alphabet[:n])

        def yield_invalid_key(len_keys):
            if len_keys == 1:
                yield None
                yield 2
                yield {}
                yield 1.5
                return
            yield (types[0], types[0], types[0])
            yield [types[0], types[0]]
            yield (types[0], None)
            yield ({}, types[0])
            yield [(types[0], types[0]), ({}, types[0])]

        return yield_invalid_key

    @pytest.fixture
    def invalid_key_specs(self, n):
        return self._invalid_key_specs(n)

    # we use populated_collection to ensure that types are correctly captured
    # for TestTypeParameterDictAttached.
    def test_invalid_keys(self, populated_collection, valid_key_specs,
                          invalid_key_specs):
        test_mapping, _ = populated_collection
        for key_spec, expected_keys in valid_key_specs(self._len_keys):
            assert set(test_mapping._indexer(key_spec)) == expected_keys
        for key_spec in invalid_key_specs(self._len_keys):
            with pytest.raises(KeyError):
                test_mapping[key_spec]
        if self._len_keys == 1:
            return
        for key_spec, expected_keys in valid_key_specs(1):
            with pytest.raises(KeyError):
                test_mapping[key_spec]

    def test_defaults(self, populated_collection):
        test_mapping, _ = populated_collection
        initial_default = test_mapping.default
        # hard code initial default values
        if self._spec == "dict":
            assert initial_default == {
                "foo": 1,
                "bar": RequiredArg,
                "baz": "hello",
                "gar": [RequiredArg],
            }
            new_default = self._generate_value()
            initial_default.update(new_default)
            test_mapping.default = new_default
            assert initial_default == test_mapping.default
        else:
            assert initial_default == RequiredArg
            new_default = self._generate_value()
            test_mapping.default = new_default
            assert new_default == test_mapping.default

    def _generate_invalid_values(self):
        if self._spec == "int":
            yield None
            yield "foo"
            yield []
            return
        yield {"foo": None}
        yield {"baz": 10}
        valid_value = self._generate_value()
        yield {**valid_value, "nonexistent": 10.0}

    def test_invalid_setting(self, populated_collection):
        test_mapping, _ = populated_collection
        key = self.choose_random_key(test_mapping)
        for invalid_value in self._generate_invalid_values():
            with pytest.raises((TypeConversionError, KeyError)):
                test_mapping[key] = invalid_value


class TestTypeParameterDictAttached(TestTypeParameterDict):
    _allow_new_keys = False

    # While it techically has defaults still, get, __getitem__, etc. don't use
    # it when attached
    _has_default = False

    @pytest.fixture
    def populated_collection(self, empty_collection, plain_collection, n):
        empty_collection.update(plain_collection)
        empty_collection._attach(DummyCppObj(),
                                 param_name="type_param",
                                 types=self.alphabet[:n])
        return empty_collection, plain_collection

    def _generate_value(self):
        if self._spec == "int":
            return self.generator.int()
        value = {}
        for key in ("foo", "baz"):
            if self.generator.rng.random() > 0.5:
                continue
            if key == "foo":
                value["foo"] = self.generator.int()
            else:
                value["baz"] = self.generator.str()
        value["bar"] = self.generator(Either(int, None, str, float))
        value["gar"] = self.generator([int])
        return value

    def test_detach(self, populated_collection):
        test_mapping, plain_mapping = populated_collection
        test_mapping._detach()
        assert not test_mapping._attached
        assert test_mapping._cpp_obj is None
        self.check_equivalent(test_mapping, plain_mapping)

    def test_premature_attaching(self, empty_collection, plain_collection, n):
        for key, value in plain_collection.items():
            with pytest.raises(IncompleteSpecificationError):
                empty_collection._attach(DummyCppObj(),
                                         param_name="type_param",
                                         types=self.alphabet[:n])
            empty_collection[key] = value
        empty_collection._attach(DummyCppObj(),
                                 param_name="type_param",
                                 types=self.alphabet[:n])

    def test_unspecified_sequence_errors(self, empty_collection,
                                         plain_collection, n):
        if self._spec != "dict":
            return
        last_key, last_value = plain_collection.popitem()
        for key, value in plain_collection.items():
            empty_collection[key] = value
        last_value.pop("gar")
        empty_collection[last_key] = last_value
        with pytest.raises(IncompleteSpecificationError):
            empty_collection._attach(DummyCppObj(),
                                     param_name="type_param",
                                     types=self.alphabet[:n])

    def _invalid_key_specs(self, n):
        invalid_types = list(self.alphabet[n:n + n])

        def yield_invalid_key(len_keys):
            yield from super(type(self), self)._invalid_key_specs(n)(len_keys)
            if len_keys == 1:
                yield invalid_types
                yield invalid_types[0]
                return
            yield (invalid_types[0], invalid_types[0])
            yield (invalid_types[0], invalid_types[:])
            yield [(invalid_types[-1], t) for t in invalid_types[:2]]

        return yield_invalid_key

    def test_introspection(self, populated_collection):
        test_mapping, _ = populated_collection
        cpp_obj = test_mapping._cpp_obj
        key = self.choose_random_key(test_mapping)
        for _ in range(3):
            v = self._generate_value()
            cpp_obj._dict[key] = v
            assert test_mapping[key] == v

    def test_isolation(self, populated_collection):
        test_mapping, _ = populated_collection
        key = self.choose_random_key(test_mapping)
        for _ in range(3):
            old_v = test_mapping[key]
            v = self._generate_value()
            test_mapping[key] = v
            if self._spec == "int":
                assert test_mapping[key] == v
            else:
                assert test_mapping[key] == {**test_mapping.default, **v}
            if isinstance(old_v, _HOOMDSyncedCollection):
                assert old_v._isolated
