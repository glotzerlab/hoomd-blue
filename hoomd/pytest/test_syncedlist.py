# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

import numpy as np
import pytest

import hoomd
from hoomd.conftest import (BaseListTest, pickling_check)
from hoomd.pytest.dummy import DummyOperation, DummySimulation
from hoomd.data.syncedlist import SyncedList


class OpInt(int):
    """Used to test SyncedList where item equality checks are needed."""
    _cpp_obj = False

    def _attach(self, simulation):
        self._simulation = simulation
        self._cpp_obj = True

    def _detach(self):
        self._simulation = None
        self._cpp_obj = False

    @property
    def _attached(self):
        return self._cpp_obj


class TestSyncedList(BaseListTest):
    _rng = np.random.default_rng(12564)

    @property
    def rng(self):
        return self._rng

    @pytest.fixture(autouse=True, params=(DummyOperation, OpInt))
    def item_cls(self, request):
        return request.param

    @pytest.fixture(autouse=True, params=(True, False))
    def attached(self, request):
        return request.param

    @pytest.fixture(autouse=True, params=(True, False))
    def attach_items(self, request):
        return request.param

    @pytest.fixture
    def generate_plain_collection(self, item_cls):
        if item_cls == DummyOperation:

            def generate(n):
                return [DummyOperation() for _ in range(n)]

            return generate
        else:

            def generate(n):
                return [
                    OpInt(self.generator.int(100_000_000)) for _ in range(n)
                ]

            return generate

    @pytest.fixture
    def empty_collection(self, item_cls, attached, attach_items):
        list_ = SyncedList(validation=item_cls, attach_members=attach_items)
        if attached:
            self._synced_list = []
            # have to store reference since SyncedList only stores weak ref.
            self.__simulation = DummySimulation()
            list_._sync(self.__simulation, self._synced_list)
        return list_

    def is_equal(self, a, b):
        if isinstance(a, DummyOperation):
            return a is b
        return a == b

    def final_check(self, test_list):
        if test_list._synced:
            if test_list._attach_members:
                assert all(item._attached for item in test_list)
            for item, synced_item in zip(test_list, self._synced_list):
                assert self.is_equal(item, synced_item)
            assert self._synced_list is test_list._synced_list
        if not test_list._attach_members:
            assert not any(
                getattr(item, "_attached", False) for item in test_list)

    def test_init(self, generate_plain_collection, item_cls):

        # Test automatic to_synced_list function generation
        synced_list = SyncedList(validation=item_cls)
        assert item_cls in synced_list._validate.types
        op = item_cls()
        assert synced_list._to_synced_list_conversion(op) is op

        # Test specified to_synced_list
        def cpp_identity(x):
            return x._cpp_obj

        # Test full initialziation
        plain_list = generate_plain_collection(5)
        synced_list = SyncedList(validation=item_cls,
                                 to_synced_list=cpp_identity,
                                 iterable=plain_list)
        assert synced_list._to_synced_list_conversion == cpp_identity
        op._cpp_obj = 2
        assert synced_list._to_synced_list_conversion(op) == 2
        self.check_equivalent(plain_list, synced_list)

    def test_synced(self):
        test_list = SyncedList(lambda x: x)
        assert not test_list._synced
        test_list._sync(None, [])
        assert test_list._synced
        test_list._unsync()
        assert not test_list._synced

    def test_register_item(self, empty_collection, item_cls):
        op = item_cls()
        empty_collection._register_item(op)
        assert op._attached == (empty_collection._synced
                                and empty_collection._attach_members)

    def test_validate_or_error(self, empty_collection, item_cls):
        with pytest.raises(ValueError):
            empty_collection._validate_or_error({})
        with pytest.raises(ValueError):
            empty_collection._validate_or_error(None)
        with pytest.raises(ValueError):
            empty_collection._validate_or_error("hello")
        empty_collection._validate_or_error(item_cls())

    def test_syncing(self, populated_collection):
        test_list, plain_list = populated_collection
        self.final_check(test_list)

    def test_unsync(self, populated_collection):
        test_list, plain_list = populated_collection
        test_list._unsync()
        assert not hasattr(test_list, "_synced_list")
        self.final_check(test_list)

    def test_synced_iter(self, empty_collection):
        empty_collection._sync(None, [3, 2, 1])
        empty_collection._synced_list = [1, 2, 3]
        assert all([
            i == j for i, j in zip(range(1, 4), empty_collection._synced_iter())
        ])

    def test_pickling(self, populated_collection):
        test_list, _ = populated_collection
        pickling_check(test_list)

    def test_sim_weakref(self, simulation_factory,
                         two_particle_snapshot_factory):

        def drop_sim(attach=False):
            sim = simulation_factory(two_particle_snapshot_factory())
            # Use operation available regardless of build
            box_resize = hoomd.update.BoxResize(
                10,
                hoomd.variant.box.Interpolate(
                    hoomd.Box.cube(4), hoomd.Box.cube(5),
                    hoomd.variant.Ramp(0, 1, 0, 10_000)))
            sim.operations.updaters.append(box_resize)
            if attach:
                sim.run(0)
            return sim.operations.updaters

        synced_list = drop_sim()
        assert synced_list._simulation is None

        synced_list = drop_sim(True)
        assert synced_list._simulation is None
        assert not synced_list._synced
