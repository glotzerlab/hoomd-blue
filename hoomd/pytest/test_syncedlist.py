from pytest import fixture, raises
from hoomd.conftest import pickling_check
from hoomd.pytest.dummy import DummyOperation, DummySimulation
from hoomd.operation import Operation
from hoomd.data.syncedlist import SyncedList, _PartialIsInstance


@fixture
def op_list():
    return [DummyOperation(), DummyOperation(), DummyOperation()]


def test_init(op_list):

    def validate(x):
        return isinstance(x, DummyOperation)

    # Test automatic to_synced_list function generation
    synced_list = SyncedList(validation=validate)
    assert synced_list._validate == validate
    op = DummyOperation()
    assert synced_list._to_synced_list_conversion(op) is op

    # Test specified to_synced_list
    def cpp_identity(x):
        return x._cpp_obj

    synced_list = SyncedList(validation=validate, to_synced_list=cpp_identity)
    assert synced_list._to_synced_list_conversion == cpp_identity
    op._cpp_obj = 2
    assert synced_list._to_synced_list_conversion(op) == 2

    # Test full initialziation
    synced_list = SyncedList(validation=validate,
                             to_synced_list=cpp_identity,
                             iterable=op_list)
    assert len(synced_list._list) == 3
    assert all(op._added for op in synced_list)


@fixture
def synced_list_empty():
    return SyncedList(_PartialIsInstance(Operation))


@fixture
def synced_list(synced_list_empty, op_list):
    synced_list_empty.extend(op_list)
    return synced_list_empty


def test_contains(synced_list_empty, op_list):
    for op in op_list:
        synced_list_empty._list.append(op)
        assert op in synced_list_empty
        new_op = DummyOperation()
        print(id(new_op), [id(op) for op in synced_list_empty])
        assert new_op not in synced_list_empty


def test_len(synced_list_empty, op_list):
    synced_list_empty._list.extend(op_list)
    assert len(synced_list_empty) == 3
    del synced_list_empty._list[1]
    assert len(synced_list_empty) == 2


def test_iter(synced_list, op_list):
    for op, op2 in zip(synced_list, synced_list._list):
        assert op is op2


def test_getitem(synced_list):
    assert all([op is synced_list[i] for i, op in enumerate(synced_list)])
    assert synced_list[:] == synced_list._list
    assert synced_list[1:] == synced_list._list[1:]


def test_synced(synced_list):
    assert not synced_list._synced
    synced_list._synced_list = None
    assert synced_list._synced


def test_attach_value(synced_list):
    op = DummyOperation()
    synced_list._attach_value(op)
    assert not op._attached
    assert op._added
    synced_list._synced_list = []
    synced_list._simulation = DummySimulation()
    op = DummyOperation()
    synced_list._attach_value(op)
    assert op._attached
    assert op._added


def test_validate_or_error(synced_list):
    with raises(ValueError):
        synced_list._validate_or_error(3)
    with raises(ValueError):
        synced_list._validate_or_error(None)
    with raises(ValueError):
        synced_list._validate_or_error("hello")
    assert synced_list._validate_or_error(DummyOperation())


def test_syncing(synced_list, op_list):
    sync_list = []
    synced_list._sync(None, sync_list)
    assert len(sync_list) == 3
    assert all([op is op2 for op, op2 in zip(synced_list, sync_list)])
    assert all([op._attached for op in synced_list])


def test_unsync(synced_list, op_list):
    sync_list = []
    synced_list._sync(None, sync_list)
    synced_list._unsync()
    assert all([not op._attached for op in synced_list])
    assert not hasattr(synced_list, "_synced_list")


def test_delitem(synced_list):
    old_op = synced_list[2]
    del synced_list[2]
    assert len(synced_list) == 2
    assert old_op not in synced_list
    assert not old_op._added
    synced_list.append(old_op)
    old_ops = synced_list[1:]
    del synced_list[1:]
    assert len(synced_list) == 1
    assert all(old_op not in synced_list for old_op in old_ops)
    assert all(not old_op._added for old_op in old_ops)
    synced_list.extend(old_ops)

    # Tested attached
    sync_list = []
    synced_list._sync(None, sync_list)
    old_op = synced_list[1]
    del synced_list[1]
    assert len(synced_list) == 2
    assert len(sync_list) == 2
    assert old_op not in synced_list
    assert all(old_op is not op for op in sync_list)
    assert not old_op._attached
    assert not old_op._added
    old_ops = synced_list[1:]
    del synced_list[1:]
    assert len(synced_list) == 1
    assert all(old_op not in synced_list for old_op in old_ops)
    assert all(not (old_op._added or old_op._attached) for old_op in old_ops)


def test_setitem(synced_list, op_list):
    with raises(IndexError):
        synced_list[3]
    with raises(IndexError):
        synced_list[-4]
    new_op = DummyOperation()
    synced_list[1] = new_op
    assert new_op is synced_list[1]
    assert new_op._added

    # Check when attached
    sync_list = []
    synced_list._sync(None, sync_list)
    new_op = DummyOperation()
    old_op = synced_list[1]
    synced_list[1] = new_op
    assert not (old_op._attached or old_op._added)
    assert new_op._attached and new_op._added
    assert sync_list[1] is new_op


def test_synced_iter(synced_list):
    synced_list._simulation = None
    synced_list._synced_list = [1, 2, 3]
    assert all([i == j for i, j in zip(range(1, 4), synced_list.synced_iter())])


class OpInt(int):
    """Used to test SyncedList where item equality checks are needed."""

    def _attach(self):
        self._cpp_obj = None

    @property
    def _attached(self):
        return hasattr(self, '_cpp_obj')

    def _detach(self):
        del self._cpp_obj

    def _add(self, simulation):
        self._simulation = simulation

    def _remove(self):
        del self._simulation

    @property
    def _added(self):
        return hasattr(self, '_simulation')


@fixture
def int_synced_list(synced_list_empty):
    return SyncedList(_PartialIsInstance(int),
                      iterable=[OpInt(i) for i in [1, 2, 3]])


def test_sync(int_synced_list):
    int_synced_list.append(OpInt(4))
    assert len(int_synced_list) == 4
    assert int_synced_list[-1] == 4

    # Test attached
    sync_list = []
    int_synced_list._sync(None, sync_list)
    int_synced_list.append(OpInt(5))
    assert len(int_synced_list) == 5
    assert len(sync_list) == 5
    assert int_synced_list[-1] == 5


def test_insert(int_synced_list):
    index = 1
    int_synced_list.insert(1, OpInt(4))
    assert len(int_synced_list) == 4
    assert int_synced_list[index] == 4

    # Test attached
    sync_list = []
    int_synced_list._sync(None, sync_list)
    int_synced_list.insert(index, OpInt(5))
    assert len(int_synced_list) == 5
    assert len(sync_list) == 5
    assert int_synced_list[index] == 5


def test_extend(int_synced_list):
    oplist = [OpInt(i) for i in range(4, 7)]
    int_synced_list.extend(oplist)
    assert len(int_synced_list) == 6
    assert int_synced_list[3:] == oplist

    # Test attached
    oplist = [OpInt(i) for i in range(7, 10)]
    sync_list = []
    int_synced_list._sync(None, sync_list)
    int_synced_list.extend(oplist)
    assert len(int_synced_list) == 9
    assert len(sync_list) == 9
    assert sync_list[6:] == oplist
    assert int_synced_list[6:] == oplist


def test_clear(int_synced_list):
    int_synced_list.clear()
    assert len(int_synced_list) == 0
    oplist = [OpInt(i) for i in range(1, 4)]
    int_synced_list.extend(oplist)

    # Test attached
    sync_list = []
    int_synced_list._sync(None, sync_list)
    int_synced_list.clear()
    assert len(int_synced_list) == 0
    assert len(sync_list) == 0
    assert all([not op._attached for op in oplist])


def test_remove(int_synced_list):
    int_synced_list.clear()
    oplist = [OpInt(i) for i in range(1, 4)]
    int_synced_list.extend(oplist)
    int_synced_list.remove(oplist[1])
    assert len(int_synced_list) == 2
    assert oplist[1] not in int_synced_list

    # Test attached
    sync_list = []
    int_synced_list._sync(None, sync_list)
    int_synced_list.remove(oplist[0])
    assert len(int_synced_list) == 1
    assert len(sync_list) == 1
    assert not oplist[0]._attached
    assert oplist[0] not in int_synced_list
    assert oplist[0] not in sync_list


def test_without_attaching():
    synced_list = SyncedList(_PartialIsInstance(int),
                             iterable=[OpInt(i) for i in [1, 2, 3]],
                             attach_members=False)
    synced_list.append(OpInt(4))
    assert len(synced_list) == 4
    assert synced_list[-1] == 4

    # Test attached
    sync_list = []
    synced_list._sync(None, sync_list)
    synced_list.append(OpInt(5))
    assert len(synced_list) == 5
    assert len(sync_list) == 5
    assert synced_list[-1] == 5
    assert all(not op._added for op in synced_list)
    assert all(not op._attached for op in synced_list)


def test_pickling(synced_list):
    pickling_check(synced_list)
