"""Test that `hoomd.local_access.LocalSnapshot` and
`hoomd.local_access_gpu.LocalSnapshotGPu` work.
"""
from copy import deepcopy
import hoomd
from hoomd.array import HOOMDGPUArray
import numpy as np
import pytest
try:
    from mpi4py import MPI
except ImportError:
    skip_mpi4py = True
else:
    skip_mpi4py = False

skip_mpi4py = pytest.mark.skipif(
    skip_mpi4py, reason='mpi4py could not be imported.')

"""
_N and _types are distinct in that the local snapshot does not know about them.
We use the underscore to signify this. Those keys are skipped when testing the
local snapshots, though are still used to define the state.
"""

Np = 5
_particle_data = dict(
    _N=Np,
    position=dict(
        np_typecode_key='Float',
        value=[[-1, -1, -1], [-1, -1, 0], [-1, 0, 0], [1, 1, 1], [1, 0, 0]],
        new_value=[[5, 5, 5]] * Np,
        shape=(Np, 3)),
    velocity=dict(
        np_typecode_key='Float',
        value=np.linspace(-4, 4, Np * 3).reshape((Np, 3)),
        new_value=np.linspace(4, 8, Np * 3).reshape((Np, 3)),
        shape=(Np, 3)),
    acceleration=dict(
        np_typecode_key='Float',
        value=np.linspace(-4, 4, Np * 3).reshape((Np, 3)),
        new_value=np.linspace(4, 8, Np * 3).reshape((Np, 3)),
        shape=(Np, 3)),
    typeid=dict(
        np_typecode_key='Integer',
        value=[0, 0, 0, 1, 1],
        new_value=[1, 1, 1, 0, 0],
        shape=(Np,)),
    mass=dict(np_typecode_key='Float', value=[5, 4, 3, 2, 1],
              new_value=[1, 2, 3, 4, 5], shape=(Np,)),
    charge=dict(np_typecode_key='Float', value=[1, 2, 3, 2, 1],
                new_value=[-1, -1, -3, -2, -1], shape=(Np,)),
    diameter=dict(np_typecode_key='Float', value=[5, 2, 3, 2, 5],
                  new_value=[2, 1, 0.5, 1, 2], shape=(Np,)),
    image=dict(
        np_typecode_key='Integer',
        value=np.linspace(-10, 20, Np * 3, dtype=np.int).reshape(Np, 3),
        new_value=np.linspace(-20, 10, Np * 3, dtype=np.int).reshape(Np, 3),
        shape=(Np, 3)),
    tag=dict(np_typecode_key='UnsignedInteger', value=None, shape=(Np,)),
    rtag=dict(np_typecode_key='UnsignedInteger', value=None, shape=(Np,)),
    _types=['p1', 'p2']
)

Nb = 2
_bond_data = dict(
    _N=Nb,
    typeid=dict(np_typecode_key='UnsignedInteger',
                value=[0, 1], new_value=[1, 0], shape=(Nb,)),
    group=dict(
        np_typecode_key='UnsignedInteger',
        value=[[0, 1], [2, 3]],
        new_value=[[1, 0], [3, 2]],
        shape=(Nb, 2)),
    tag=dict(np_typecode_key='UnsignedInteger', value=None, shape=(Nb,)),
    rtag=dict(np_typecode_key='UnsignedInteger', value=None, shape=(Nb,)),
    _types=['b1', 'b2']
)

Na = 2
_angle_data = dict(
    _N=Na,
    typeid=dict(np_typecode_key='UnsignedInteger',
                value=[1, 0], new_value=[0, 1], shape=(Na,)),
    group=dict(
        np_typecode_key='UnsignedInteger',
        value=[[0, 1, 2], [2, 3, 4]],
        new_value=[[1, 3, 4], [0, 2, 4]],
        shape=(Na, 3)),
    tag=dict(np_typecode_key='UnsignedInteger', value=None, shape=(Na,)),
    rtag=dict(np_typecode_key='UnsignedInteger', value=None, shape=(Na,)),
    _types=['a1', 'a2']
)

Nd = 2
_dihedral_data = dict(
    _N=Nd,
    typeid=dict(np_typecode_key='UnsignedInteger',
                value=[1, 0], new_value=[0, 1], shape=(Nd,)),
    group=dict(
        np_typecode_key='UnsignedInteger',
        value=[[0, 1, 2, 3], [1, 2, 3, 4]],
        new_value=[[4, 3, 2, 1], [2, 4, 0, 1]],
        shape=(Nd, 4)),
    tag=dict(np_typecode_key='UnsignedInteger', value=None, shape=(Nd,)),
    rtag=dict(np_typecode_key='UnsignedInteger', value=None, shape=(Nd,)),
    _types=['d1', 'd2']
)

Ni = 2
_improper_data = dict(
    _N=Ni,
    typeid=dict(
        np_typecode_key='UnsignedInteger', value=[0, 0], shape=(Ni,)),
    group=dict(
        np_typecode_key='UnsignedInteger',
        value=[[3, 2, 1, 0], [1, 2, 3, 4]],
        new_value=[[1, 2, 3, 0], [4, 2, 3, 1]],
        shape=(Ni, 4)),
    tag=dict(
        np_typecode_key='UnsignedInteger', value=None, shape=(Ni,)),
    rtag=dict(
        np_typecode_key='UnsignedInteger', value=None, shape=(Ni,)),
    _types=['i1']
)

Nc = 3
_constraint_data = dict(
    _N=Nc,
    value=dict(np_typecode_key='Float', value=[2.5, 0.5, 2.],
               new_value=[3., 1.5, 1.], shape=(Nc,)),
    group=dict(
        np_typecode_key='UnsignedInteger',
        value=[[0, 1], [2, 3], [1, 3]],
        new_value=[[4, 1], [3, 1], [2, 4]],
        shape=(Nc, 2)),
    tag=dict(np_typecode_key='UnsignedInteger', value=None, shape=(Nc,)),
    rtag=dict(np_typecode_key='UnsignedInteger', value=None, shape=(Nc,)),
)

Npa = 2
_pair_data = dict(
    _N=Npa,
    typeid=dict(np_typecode_key='UnsignedInteger',
                value=[0, 1], new_value=[1, 0], shape=(Npa,)),
    group=dict(
        np_typecode_key='UnsignedInteger',
        value=[[0, 1], [2, 3]],
        new_value=[[4, 1], [0, 3]],
        shape=(Npa, 2)),
    tag=dict(np_typecode_key='UnsignedInteger', value=None, shape=(Npa,)),
    rtag=dict(np_typecode_key='UnsignedInteger', value=None, shape=(Npa,)),
    _types=['p1', 'p2']
)


@pytest.fixture(scope='session')
def base_snapshot(device):
    """Defines a snapshot using the data given above."""
    def set_snapshot(snap, data, base):
        """Sets individual sections of snapshot (e.g. particles)."""
        snap_section = getattr(snap, base)
        for k in data:
            if k.startswith('_'):
                setattr(snap_section, k[1:], data[k])
                continue
            elif data[k]['value'] is None:
                continue
            try:
                array = getattr(snap_section, k)
                array[:] = data[k]['value']
            except TypeError:
                setattr(snap_section, k, data[k]['value'])

    snapshot = hoomd.Snapshot(device.comm)

    if snapshot.exists:
        snapshot.configuration.box = [2.1, 2.1, 2.1, 0, 0, 0]
        set_snapshot(snapshot, _particle_data, 'particles')
        set_snapshot(snapshot, _bond_data, 'bonds')
        set_snapshot(snapshot, _angle_data, 'angles')
        set_snapshot(snapshot, _dihedral_data, 'dihedrals')
        set_snapshot(snapshot, _improper_data, 'impropers')
        set_snapshot(snapshot, _constraint_data, 'constraints')
        set_snapshot(snapshot, _pair_data, 'pairs')
    return snapshot


@pytest.fixture(scope='function')
def simulation(simulation_factory, base_snapshot):
    """Creates the simulation from the base_snapshot.

    It is recreated every function to ensure that changing a simulation's state
    does not change it for all test.
    """
    return simulation_factory(base_snapshot)


def check_boxes(sim, snap):
    """Checks that local and global boxes behave as expected in local snapshot.
    """
    assert type(snap.box) == hoomd.Box
    assert type(snap.local_box) == hoomd.Box

    if sim.device.num_ranks == 1:
        assert snap.local_box == sim.state.box
        assert snap.box == sim.state.box
    else:
        assert snap.local_box != sim.state.box
        assert snap.box == sim.state.box


def test_box_cpu(simulation):
    """General check that ``box`` and ``local_box`` properties work."""
    with simulation.state.cpu_local_snapshot as data:
        check_boxes(simulation, data)


@pytest.mark.gpu(True)
def test_box_gpu(simulation):
    """General check that ``box`` and ``local_box`` properties work."""
    with simulation.state.gpu_local_snapshot as data:
        check_boxes(simulation, data)


def check_tag_size(snapshot, local_snapshot, comm, device):
    """Checks that tags are the appropriate size from local snapshots.

    tags are used for checking other shapes so this is necessary to validate
    those tests.
    """
    def get_N(snapshot, group, comm):
        if snapshot.exists:
            N = getattr(snapshot, group).N
        else:
            N = None
        return comm.bcast(N, root=0)

        # check particles tag size
        Np = get_N(snapshot, 'particles', comm)
        total_len = comm.allreduce(
            len(local_snapshot.particles.tag), op=MPI.SUM)
        assert total_len == Np

        # bonds and others can have a tag stored on multiple ranks
        for group in ['bonds', 'angles', 'dihedrals', 'impropers',
                      'constraints', 'pairs']:
            data_section = getattr(local_snapshot, group)
            N = get_N(snapshot, group, comm)
            if device.num_ranks > 1:
                assert len(data_section.tag) <= N
            else:
                assert len(data_section.tag) == N


@skip_mpi4py
def test_cpu_local_snapshot_tags_shape(simulation):
    mpi_comm = MPI.COMM_WORLD
    snapshot = simulation.state.snapshot
    with simulation.state.cpu_local_snapshot as data:
        check_tag_size(snapshot, data, mpi_comm, simulation.device)


@pytest.mark.gpu(True)
@skip_mpi4py
def test_gpu_local_snapshot_tags_shape(simulation):
    mpi_comm = MPI.COMM_WORLD
    snapshot = simulation.state.snapshot
    with simulation.state.gpu_local_snapshot as data:
        check_tag_size(snapshot, data, mpi_comm, simulation.device)


def check_type(data, prop_dict, tags):
    """Check that the expected dtype is found for local snapshots."""
    expected_typecodes = np.typecodes[prop_dict['np_typecode_key']]
    if hasattr(data, 'dtype'):
        given_typecode = data.dtype.char
    elif isinstance(data, HOOMDGPUArray):
        given_typecode = data.__cuda_array_interface__['typestr']
    else:
        raise RuntimeError("Array expected to have dtype attribute.")
    assert given_typecode in expected_typecodes


def check_shape(data, prop_dict, tags):
    """Check shape of properties in the snapshot."""
    # checks size of prop_dict values and tags.
    assert data.shape == (len(tags),) + prop_dict['shape'][1:]


def check_getting(data, prop_dict, tags):
    """Checks getting properties of the state through a local snapshot."""
    if prop_dict['value'] is not None:
        if hasattr(data, '__getitem__'):
            expected_values = np.array(prop_dict['value'])
            if prop_dict['np_typecode_key'] == 'Float':
                assert np.allclose(expected_values[tags], data[:])
            else:
                assert all(expected_values[tags].ravel() == data[:].ravel())
        else:
            # Ensure that only HOOMDGPUArray can not have __getitem__
            assert isinstance(data, HOOMDGPUArray)


def check_setting(data, prop_dict, tags):
    """Checks setting properties of the state through a local snapshot.

    Also tests error raising for read only arrays."""
    if 'new_value' in prop_dict:
        if hasattr(data, '__setitem__'):
            new_values = np.array(prop_dict['new_value'])[tags]
            if data.read_only:
                with pytest.raises(ValueError):
                    data[:] = new_values
            else:
                data[:] = new_values
                if prop_dict['np_typecode_key'] == 'Float':
                    assert np.allclose(new_values, data[:])
                else:
                    assert all(new_values.ravel() == data[:].ravel())
        else:
            # Ensure that only HOOMDGPUArray can not have __setitem__
            assert isinstance(data, HOOMDGPUArray)


@pytest.fixture(scope='function',
                params=[check_type, check_shape, check_getting, check_setting])
def property_check(request):
    return request.param


@pytest.fixture(
    scope='function',
    params=[
        ('particles', _particle_data), ('bonds', _bond_data),
        ('angles', _angle_data), ('dihedrals', _dihedral_data),
        ('impropers', _improper_data), ('constraints', _constraint_data),
        ('pairs', _pair_data)], ids=lambda x: x[0] + '-data')
def data_name_pair(request):
    return deepcopy(request.param)


@pytest.fixture(scope='function', params=['', 'ghost_', '_with_ghosts'],
                ids=lambda x: x.strip('_'))
def affix(request):
    return request.param


def test_cpu_local_snapshot_data(
        simulation, data_name_pair, affix, property_check):
    name, data_dict = data_name_pair
    with simulation.state.cpu_local_snapshot as snapshot:
        # gets the particle, bond, etc data
        snap_data = getattr(snapshot, name)
        # Checks each property of data from passed dictionary
        for property_name, property_dict in data_dict.items():
            if property_name.startswith('_'):
                continue
            if affix.startswith('_'):
                hoomd_buffer = getattr(snap_data, property_name + affix)
                tags = getattr(snap_data, 'tag' + affix)
            elif affix.endswith('_'):
                hoomd_buffer = getattr(snap_data, affix + property_name)
                tags = getattr(snap_data, affix + 'tag')
            else:
                hoomd_buffer = getattr(snap_data, property_name)
                tags = snap_data.tag
            property_check(hoomd_buffer, property_dict, tags)


@pytest.mark.gpu(True)
def test_gpu_local_snapshot_data(simulation, data_name_pair):
    name, data_dict = data_name_pair
    with simulation.state.gpu_local_snapshot as snapshot:
        # gets the particle, bond, etc data
        snap_data = getattr(snapshot, name)
        # Checks each property of data from passed dictionary
        for property_name, property_dict in data_dict.items():
            if property_name.startswith('_'):
                continue
            hoomd_buffer = getattr(snap_data, property_name)
            check_property(
                simulation.device, hoomd_buffer, property_dict, snap_data.tag)
