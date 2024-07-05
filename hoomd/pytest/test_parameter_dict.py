# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import pytest
import numpy as np

import hoomd
from hoomd.conftest import BaseMappingTest, Options, pickling_check
from hoomd.data.collections import _HOOMDSyncedCollection
from hoomd.data.parameterdicts import ParameterDict


def identity(x):
    return x


class DummyCppObj:

    def __init__(self):
        self._dict = dict()

    def __getattr__(self, attr):
        return self._dict[attr]

    def __setattr__(self, attr, value):
        if attr == "_dict":
            super().__setattr__(attr, value)
            return
        self._dict[attr] = value

    def notifyDetach(self):  # noqa: N802
        pass

    def __getstate__(self):
        raise RuntimeError("Mimic lack of pickling for C++ objects.")


class TestParameterDict(BaseMappingTest):
    _has_default = False

    @pytest.fixture(params=(1, 2, 3), ids=lambda x: f"n={x}")
    def n(self, request):
        return request.param

    @pytest.fixture(autouse=True)
    def spec(self, n):
        self._n = n
        if n == 1:
            spec = {
                "int": int,
                "list[int]": [int],
                "(float, str)": (float, str)
            }
        elif n == 2:
            spec = {"dict": {"str": str}, "filter": hoomd.filter.ParticleFilter}
        else:
            spec = {
                "(float, float, float)": (float, float, float),
                "list[dict[str, int]]": [{
                    "foo": int,
                    "bar": int
                }],
                "(float, str)": (float, str)
            }
        self._spec = spec
        return spec

    def filter(self):
        return self.generator(
            Options(hoomd.filter.All(), hoomd.filter.Type(("A", "B")),
                    hoomd.filter.Tags([1, 2, 25])))

    def _generate_value(self):
        if self._n == 2:
            filter_ = self._spec.pop("filter")
            value = self.generator(self._spec)
            value["filter"] = self.filter()
            self._spec["filter"] = filter_
        else:
            value = self.generator(self._spec)
        return value

    @pytest.fixture
    def generate_plain_collection(self):

        def generate(n):
            return self._generate_value()

        return generate

    def is_equal(self, test_param, mapping):
        """Assert that the keys in mapping have equivalent values in type param.

        This is looking at the keys of the ParameterDict, and they should be
        equivalent.
        """
        return test_param == mapping

    @pytest.fixture
    def empty_collection(self, spec):
        """Return an empty type parameter."""
        return ParameterDict(**spec)

    def check_equivalent(self, test_mapping, other):
        assert set(test_mapping) == other.keys()
        for key, value in other.items():
            assert test_mapping[key] == value

    @pytest.fixture(params=(True, False),
                    ids=lambda x: "in_map" if x else "out_map")
    def setitem_key_value(self, n, request):
        value = self._generate_value()
        keys = list(value)
        if request.param:
            key = self.generator.rng.choice(keys)
            return key, value[key]
        key = next(filter(lambda x: x not in keys, self.random_keys()))
        return key, value

    @pytest.fixture
    def setdefault_key_value(self, setitem_key_value):
        return setitem_key_value

    def test_pickling(self, populated_collection):
        pickling_check(populated_collection[0])


class TestParameterDictAttached(TestParameterDict):
    _allow_new_keys = False
    _deletion_error = RuntimeError

    def final_check(self, test_mapping):
        for attr, value in test_mapping.items():
            assert self.is_equal(value, getattr(test_mapping._cpp_obj, attr))

    @pytest.fixture
    def populated_collection(self, empty_collection, plain_collection, n):
        empty_collection.update(plain_collection)
        empty_collection._attach(DummyCppObj())
        return empty_collection, plain_collection

    def test_detach(self, populated_collection):
        test_mapping, plain_mapping = populated_collection
        test_mapping._detach()
        assert not test_mapping._attached
        assert test_mapping._cpp_obj is None
        self.check_equivalent(test_mapping, plain_mapping)

    def test_introspection(self, populated_collection, n):
        test_mapping, _ = populated_collection
        cpp_obj = test_mapping._cpp_obj
        new_value = self._generate_value()
        for k, v in new_value.items():
            # setting through C++ object allows for data to update and not
            # isolate.
            setattr(cpp_obj, k, v)
            assert test_mapping[k] == v

    def test_isolation(self, populated_collection, n):
        test_mapping, _ = populated_collection
        new_value = self._generate_value()
        for k, v in new_value.items():
            # setting through ParameterDict isolates composed objects
            old_v = test_mapping[k]
            test_mapping[k] = v
            assert test_mapping[k] == v
            if isinstance(old_v, _HOOMDSyncedCollection):
                assert old_v._isolated


class TestSpecialTypes:

    def test_variants(self):
        mapping = ParameterDict(variant=hoomd.variant.Variant)
        mapping["variant"] = 4.0
        assert mapping["variant"] == hoomd.variant.Constant(4.0)
        ramp = hoomd.variant.Ramp(0, 1, 0, 10_000)
        mapping["variant"] = ramp
        assert mapping["variant"] == ramp

    def test_triggers(self):
        mapping = ParameterDict(trigger=hoomd.trigger.Trigger)
        mapping["trigger"] = 1
        assert mapping["trigger"] == hoomd.trigger.Periodic(1)
        mapping["trigger"] = 1.5
        assert mapping["trigger"] == hoomd.trigger.Periodic(1)
        after = hoomd.trigger.After(100)
        mapping["trigger"] = after
        assert mapping["trigger"] == after

    def test_filters(self):
        mapping = ParameterDict(filter=hoomd.filter.ParticleFilter)
        mapping["filter"] = hoomd.filter.All()
        assert mapping["filter"] == hoomd.filter.All()
        tag_100 = hoomd.filter.Tags([100])
        mapping["filter"] = tag_100
        assert mapping["filter"] == tag_100

        class NewFilter(hoomd.filter.CustomFilter):

            def __call__(self, state):
                return np.array([], dtype=np.uint64)

            def __eq__(self, other):
                return self is other

            def __hash__(self):
                return hash(self.__class__.__name__)

        new_filter = NewFilter()
        mapping["filter"] = new_filter
        assert mapping["filter"] == new_filter

    def test_ndarray(self):
        mapping = ParameterDict(ndarray=np.ndarray)
        mapping["ndarray"] = np.array([], dtype=bool)
        assert mapping["ndarray"].dtype == float

        int_array = np.array([1, 2], dtype=int)
        mapping["ndarray"] = int_array
        assert mapping["ndarray"].dtype == float
        assert np.allclose(mapping["ndarray"], int_array)

        float_array = np.array([[3.15], [4.10]], dtype=float)
        with pytest.raises(hoomd.error.TypeConversionError):
            mapping["ndarray"] = float_array
        float_array = float_array.flatten()
        mapping["ndarray"] = float_array
        assert np.allclose(mapping["ndarray"], float_array)

    def test_str(self):
        mapping = ParameterDict(str=str)
        with pytest.raises(hoomd.error.TypeConversionError):
            mapping["str"] = 4.0
        with pytest.raises(hoomd.error.TypeConversionError):
            mapping["str"] = {}
        mapping["str"] = "abc"
        assert mapping["str"] == "abc"

    @pytest.mark.parametrize("box", ([10, 15, 25, 0, -0.5, 2], {
        "Lx": 10,
        "Ly": 15,
        "Lz": 25,
        "xy": 0,
        "xz": -0.5,
        "yz": 2
    }, {
        "Lx": 10,
        "Ly": 15,
        "Lz": 25
    }, {
        "Lx": 10,
        "Ly": 15
    }, {
        "Lx": 10,
        "Ly": 15,
        "xy": 0
    }, [10, 15]))
    def test_box_valid(self, box):
        mapping = ParameterDict(box=hoomd.Box)
        mapping["box"] = box
        assert mapping["box"] == hoomd.Box.from_box(box)

    @pytest.mark.parametrize("box", ([10, 15, 25, 0, -0.5], {
        "Ly": 15,
        "Lz": 25,
        "xy": 0,
        "xz": -0.5,
        "yz": 2
    }, {
        "Lx": 10,
        "Ly": 15,
        "xz": 1
    }, [10]))
    def test_box_invalid(self, box):
        mapping = ParameterDict(box=hoomd.Box)
        with pytest.raises(hoomd.error.TypeConversionError):
            mapping["box"] = box
