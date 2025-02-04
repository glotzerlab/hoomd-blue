# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import warnings

import numpy as np
import pytest

from hoomd.conftest import BaseListTest, BaseMappingTest, BaseSequenceTest
from hoomd.data import collections, typeconverter
from hoomd.error import IsolationWarning


class MockRoot:

    def __init__(self, schema, data):
        self._data = data
        validator = typeconverter.to_type_converter(schema)
        self._sync_data = {}
        for key in data:
            self._sync_data[key] = collections._to_hoomd_data(
                root=self,
                schema=validator[key],
                parent=None,
                identity=key,
                data=data[key],
            )

    def _write(self, obj):
        with obj._parent._suspend_read:
            self._data[obj._identity] = obj._parent.to_base()

    def _read(self, obj):
        key = obj._identity
        new_value = self._data[key]
        obj._parent._update(new_value)


class TestHoomdList(BaseListTest):

    @pytest.fixture(autouse=True, params=("ints", "floats", "strs"))
    def current_list(self, request):
        self._current_list = request.param

    @pytest.fixture
    def generate_plain_collection(self):
        if self._current_list == "ints":
            generate_one = self.generator.int
        elif self._current_list == "floats":
            generate_one = self.generator.float
        elif self._current_list == "strs":

            def generate_one():
                return [
                    self.generator.str()
                    for _ in range(3 + self.generator.int(10))
                ]

        def generate(n):
            return [generate_one() for _ in range(n)]

        return generate

    def is_equal(self, a, b):
        return a == b

    def final_check(self, test_list):
        assert test_list.to_base() == self._data._data["lists"][
            self._current_list]
        if self._current_list == "strs":
            assert all(
                isinstance(t_item, collections._HOOMDList)
                for t_item in test_list)

    @pytest.fixture
    def empty_collection(self):
        self._data = MockRoot(
            {"lists": {
                "ints": [int],
                "floats": [float],
                "strs": [[str]]
            }},
            {"lists": {
                "ints": [],
                "floats": [],
                "strs": []
            }},
        )
        with self._data._sync_data["lists"]._suspend_read_and_write:
            return self._data._sync_data["lists"][self._current_list]

    def test_isolation(self, populated_collection, generate_plain_collection):
        if self._current_list != "strs":
            return
        test_list, _ = populated_collection
        item = test_list.pop()
        assert item._isolated
        item = test_list[0]
        test_list[0] = generate_plain_collection(1)[0]
        assert item._isolated
        item = test_list[0]
        del test_list[0]
        assert item._isolated
        remaining_items = test_list[:]
        test_list.clear()
        assert all(item._isolated for item in remaining_items)

    def test_iadd(self, populated_collection):
        test_list, plain_collection = populated_collection
        data = self._data._sync_data["lists"]
        data[self._current_list] += [plain_collection[0]]
        assert len(data[self._current_list]) == len(plain_collection) + 1
        assert data[self._current_list][-1] == plain_collection[0]
        self.final_check(data[self._current_list])


class TestHoomdTuple(BaseSequenceTest):

    @pytest.fixture
    def generate_plain_collection(self):

        def generate(n):
            strings = [
                self.generator.str() for _ in range(self.generator.int(10))
            ]
            return (self.generator.int(), strings, self.generator.float(),
                    self.generator.ndarray((None, 3)))

        return generate

    def is_equal(self, a, b):
        if isinstance(a, np.ndarray):
            return np.array_equal(a, b)
        if isinstance(b, np.ndarray):
            return False
        return a == b

    def final_check(self, test_tuple):
        assert test_tuple == self._data._data["tuple"]
        assert all(isinstance(s, collections._HOOMDList) for s in test_tuple[1])

    @pytest.fixture
    def empty_collection(self):
        return

    @pytest.fixture
    def populated_collection(self, plain_collection):

        self._data = MockRoot(
            {
                "tuple": (int, [str], float,
                          typeconverter.NDArrayValidator("float64", (None, 3)))
            }, {"tuple": plain_collection})
        return self._data._sync_data["tuple"], plain_collection


class TestHoomdDict(BaseMappingTest):
    _allow_new_keys = False

    @pytest.fixture
    def generate_plain_collection(self):

        def generate(n):
            tuples = [(self.generator.int(), [
                self.generator.str() for _ in range(3 + self.generator.int(10))
            ]) for _ in range(self.generator.int(10) + 3)]
            data = {
                "sigma": self.generator.float(),
                "epsilon": self.generator.float(),
                "notes": tuples
            }
            return data

        return generate

    def is_equal(self, a, b):
        return a == b

    def final_check(self, test_mapping):
        assert test_mapping.to_base() == self._data._data["params"]
        if "notes" not in test_mapping:
            return
        assert isinstance(test_mapping["notes"], collections._HOOMDList)
        assert all(
            isinstance(t, collections._HOOMDTuple)
            for t in test_mapping["notes"])
        assert all(
            isinstance(str_list, collections._HOOMDList)
            for i, str_list in test_mapping["notes"])

    @pytest.fixture
    def empty_collection(self):
        self._data = MockRoot(
            {
                "params": {
                    "epsilon": float,
                    "sigma": float,
                    "notes": [(int, [str])]
                }
            }, {"params": {}})
        return self._data._sync_data["params"]

    @pytest.fixture(params=(True, False))
    def setitem_key_value(self, generate_plain_collection, request):
        mapping = generate_plain_collection(None)
        if request.param:
            key = self.choose_random_key(mapping)
            return key, mapping[key]
        _, value = mapping.popitem()
        key = next(filter(lambda k: k not in mapping, self.random_keys()))
        return key, value

    @pytest.fixture
    def setdefault_key_value(self, setitem_key_value):
        return setitem_key_value

    def test_isolation(self, populated_collection, generate_plain_collection):
        """Test that data structures are properly isolated when overwritten.

        This also serves as the master test since it deals with a fairly nested
        structure.
        """

        def check_isolation(obj):
            # Check list of tuples of (int, [str])
            assert obj._isolated
            assert len(obj._children) == 0
            with pytest.warns(IsolationWarning):
                _ = obj[0]

            with pytest.warns(IsolationWarning):
                list(obj)

            with pytest.warns(IsolationWarning):
                obj[0] = obj._data[0]

            with warnings.catch_warnings(record=True) as warning_record:
                obj.to_base()
            assert len(warning_record) == 0

            # Check tuples of (int, [str])
            assert all(item._isolated for item in obj._data)
            assert all(len(item._children) == 0 for item in obj._data)
            with pytest.warns(IsolationWarning):
                obj._data[0][1]

            with pytest.warns(IsolationWarning):
                list(obj._data[0])

            with warnings.catch_warnings(record=True) as warning_record:
                obj._data[0].to_base()
            assert len(warning_record) == 0

            # Check list of str
            assert all(item._data[1]._isolated for item in obj._data)
            assert all(len(item._data[1]._children) == 0 for item in obj._data)
            with pytest.warns(IsolationWarning):
                _ = obj._data[0][1][0]

            with pytest.warns(IsolationWarning):
                list(obj._data[0][1])

            with pytest.warns(IsolationWarning):
                obj._data[0][1][0] = obj._data[0][1]._data[0]

            with warnings.catch_warnings(record=True) as warning_record:
                obj._data[0]._data[1].to_base()
            assert len(warning_record) == 0

        test_mapping, _ = populated_collection
        item = test_mapping.pop("notes")
        check_isolation(item)
        test_mapping["notes"] = generate_plain_collection(None)["notes"]
        item = test_mapping["notes"]
        del test_mapping["notes"]
        check_isolation(item)
        test_mapping["notes"] = generate_plain_collection(None)["notes"]
        item = test_mapping["notes"]
        test_mapping.clear()
        check_isolation(item)
