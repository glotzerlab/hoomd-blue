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

try:
    import cupy
    CUPY_IMPORTED=True
except ImportError:
    CUPY_IMPORTED=False


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


@pytest.fixture(scope='session')
def cpu_simulation_factory(device_cpu):
    """Creates the simulation from the base_snapshot."""

    def make_simulation(snapshot):
        sim = hoomd.Simulation(device_cpu)

        # reduce sorter grid to avoid Hilbert curve overhead in unit tests
        for tuner in sim.operations.tuners:
            if isinstance(tuner, hoomd.tuner.ParticleSorter):
                tuner.grid = 8

        sim.create_state_from_snapshot(snapshot)
        return sim

    return make_simulation


@pytest.fixture(scope='session')
def gpu_simulation_factory(device_gpu):
    """Creates the simulation from the base_snapshot."""

    def make_simulation(snapshot):
        sim = hoomd.Simulation(device_gpu)

        # reduce sorter grid to avoid Hilbert curve overhead in unit tests
        for tuner in sim.operations.tuners:
            if isinstance(tuner, hoomd.tuner.ParticleSorter):
                tuner.grid = 8

        sim.create_state_from_snapshot(snapshot)
        return sim

    return make_simulation


def check_box(local_snapshot, global_box, ranks):
    """General check that ``box`` and ``local_box`` properties work."""
    assert type(local_snapshot.box) == hoomd.Box
    assert type(local_snapshot.local_box) == hoomd.Box

    if ranks == 1:
        assert local_snapshot.local_box == global_box
        assert local_snapshot.box == global_box
    else:
        assert local_snapshot.local_box != global_box
        assert local_snapshot.box == global_box


def test_box_cpu(cpu_simulation_factory, base_snapshot):
    sim = cpu_simulation_factory(base_snapshot)
    with sim.state.cpu_local_snapshot as data:
        check_box(data, sim.state.box, sim.device.num_ranks)


def test_box_gpu(gpu_simulation_factory, base_snapshot):
    sim = gpu_simulation_factory(base_snapshot)

    with sim.state.cpu_local_snapshot as data:
        check_box(data, sim.state.box, sim.device.num_ranks)

    with sim.state.gpu_local_snapshot as data:
        check_box(data, sim.state.box, sim.device.num_ranks)


def check_tag_shape(base_snapshot, local_snapshot, group, ranks):
    mpi_comm = MPI.COMM_WORLD

    if base_snapshot.exists:
        N = getattr(base_snapshot, group).N
    else:
        N = None
    return mpi_comm.bcast(N, root=0)

    # check particles tag size
    if group == 'particles':
        total_len = mpi_comm.allreduce(
            len(local_snapshot.particles.tag), op=MPI.SUM)
        assert total_len == N
    else:
        local_snapshot_section = getattr(local_snapshot, group)
        if ranks > 1:
            assert len(local_snapshot_section.tag) <= N
        else:
            assert len(local_snapshot_section.tag) == N


@pytest.fixture(
    params=['particles', 'bonds', 'angles',
            'dihedrals', 'impropers', 'constraints', 'pairs'])
def snapshot_section(request):
    return request.param


@skip_mpi4py
def test_cpu_tags_shape(
        cpu_simulation_factory, base_snapshot, snapshot_section):
    """Checks that tags are the appropriate size from local snapshots.

    tags are used for checking other shapes so this is necessary to validate
    those tests.
    """
    sim = cpu_simulation_factory(base_snapshot)
    with sim.state.cpu_local_snapshot as data:
        check_tag_shape(
            base_snapshot, data, snapshot_section, sim.device.num_ranks)


@skip_mpi4py
def test_gpu_tags_shape(
        gpu_simulation_factory, base_snapshot, snapshot_section):
    """Checks that tags are the appropriate size from local snapshots.

    tags are used for checking other shapes so this is necessary to validate
    those tests.
    """
    sim = gpu_simulation_factory(base_snapshot)
    with sim.state.cpu_local_snapshot as data:
        check_tag_shape(
            base_snapshot, data, snapshot_section, sim.device.num_ranks)

    with sim.state.gpu_local_snapshot as data:
        check_tag_shape(
            base_snapshot, data, snapshot_section, sim.device.num_ranks)


# Testing local snapshot array properties

@pytest.fixture(
    scope='function',
    params=[(name, prop_name, prop_dict)
            for name, section_dict in
            [('particles', _particle_data), ('bonds', _bond_data),
             ('angles', _angle_data), ('dihedrals', _dihedral_data),
             ('impropers', _improper_data), ('constraints', _constraint_data),
             ('pairs', _pair_data)]
            for prop_name, prop_dict in section_dict.items()
            if not prop_name.startswith('_')
            ],
    ids=lambda x: x[0] + '-' + x[1])
def section_name_dict(request):
    """Parameterization of expected values for local_snapshot properties.

    Examples include ``('particles', 'position', position_dict)`` where
    ``position_dict`` is the dictionary with the expected typecodes, shape, and
    value of particle positions.
    """
    return deepcopy(request.param)


@pytest.fixture(scope='function', params=['', 'ghost_', '_with_ghosts'],
                ids=lambda x: x.strip('_'))
def affix(request):
    """Parameterizes over the different variations of a local_snapshot property.

    These include ``property``, ``ghost_property``, and
    ``property_with_ghosts``.
    """
    return request.param


def get_property_name_from_affix(name, affix):
    if affix.startswith('_'):
        return name + affix
    elif affix.endswith('_'):
        return affix + name
    else:
        return name


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
    if isinstance(data, HOOMDGPUArray):
        if len(tags) == 0:
            assert data.shape == (0,)
        else:
            assert data.shape == (len(tags),) + prop_dict['shape'][1:]
    else:
        assert data.shape == (len(tags),) + prop_dict['shape'][1:]


def general_array_equality(arr1, arr2):
    if any(np.issubdtype(a.dtype, np.floating) for a in (arr1, arr2)):
        if any(isinstance(a, HOOMDGPUArray) for a in (arr1, arr2)):
            return cupy.allclose(arr1, arr2)
        else:
            return np.allclose(arr1, arr2)
    else:
        return all(arr1.ravel() == arr2.ravel())


def check_getting(data, prop_dict, tags):
    """Checks getting properties of the state through a local snapshot."""
    # Check to end test early
    if isinstance(data, HOOMDGPUArray) and not CUPY_IMPORTED:
        pytest.skip("Not available for HOOMDGPUArray without CuPy.")
    if len(tags) == 0 or prop_dict['value'] is None:
        return None

    if isinstance(data, HOOMDGPUArray):
        expected_values = cupy.array(prop_dict['value'])
    else:
        expected_values = np.array(prop_dict['value'])
    assert general_array_equality(data, expected_values[tags.tolist()])


def check_setting(data, prop_dict, tags):
    """Checks setting properties of the state through a local snapshot.

    Also tests error raising for read only arrays."""
    # Test if test should be skipped or just return
    if isinstance(data, HOOMDGPUArray) and not CUPY_IMPORTED:
        pytest.skip("Not available for HOOMDGPUArray without CuPy.")
    if 'new_value' not in prop_dict:
        return None

    if isinstance(data, HOOMDGPUArray):
        new_values = cupy.array(prop_dict['new_value'])[tags.tolist()]
    else:
        new_values = np.array(prop_dict['new_value'])[tags]

    if data.read_only:
        with pytest.raises(ValueError):
            data[:] = new_values
    else:
        data[:] = new_values
        assert general_array_equality(data, new_values)


@pytest.fixture(scope='function',
                params=[check_type, check_shape, check_getting, check_setting])
def property_check(request):
    """Parameterizes differnt types of checks on local_snapshot properties."""
    return request.param


def test_cpu_arrays_properties(cpu_simulation_factory, base_snapshot,
                           section_name_dict, affix, property_check):
    """This test makes extensive use of parameterizing in pytest.

    This test tests the type, shape, getting, and setting of array values in the
    local snapshot. We test all properties including ghost and both ghost and
    normal particles, bonds, etc.
    """
    name, property_name, property_dict = section_name_dict
    property_name = get_property_name_from_affix(property_name, affix)
    tag_name = get_property_name_from_affix('tag', affix)

    sim = cpu_simulation_factory(base_snapshot)
    with sim.state.cpu_local_snapshot as data:
        # gets the particle, bond, etc data
        snapshot_section = getattr(data, name)
        hoomd_buffer = getattr(snapshot_section, property_name)
        tags = getattr(snapshot_section, tag_name)
        property_check(hoomd_buffer, property_dict, tags)


def test_cpu_arrays_properties_with_gpu_device(
        gpu_simulation_factory, base_snapshot,
        section_name_dict, affix, property_check):
    """This test makes extensive use of parameterizing in pytest.

    This test tests the type, shape, getting, and setting of array values in the
    local snapshot. We test all properties including ghost and both ghost and
    normal particles, bonds, etc.
    """
    name, property_name, property_dict = section_name_dict
    property_name = get_property_name_from_affix(property_name, affix)
    tag_name = get_property_name_from_affix('tag', affix)

    sim = gpu_simulation_factory(base_snapshot)
    with sim.state.cpu_local_snapshot as data:
        # gets the particle, bond, etc data
        snapshot_section = getattr(data, name)
        hoomd_buffer = getattr(snapshot_section, property_name)
        tags = getattr(snapshot_section, tag_name)
        property_check(hoomd_buffer, property_dict, tags)


def test_gpu_arrays_properties(gpu_simulation_factory, base_snapshot,
                               section_name_dict, affix, property_check):
    """This test makes extensive use of parameterizing in pytest.

    This test tests the type, shape, getting, and setting of array values in the
    local snapshot. We test all properties including ghost and both ghost and
    normal particles, bonds, etc.
    """
    name, property_name, property_dict = section_name_dict
    property_name = get_property_name_from_affix(property_name, affix)
    tag_name = get_property_name_from_affix('tag', affix)

    sim = gpu_simulation_factory(base_snapshot)
    with sim.state.gpu_local_snapshot as data:
        # gets the particle, bond, etc data
        snapshot_section = getattr(data, name)
        hoomd_buffer = getattr(snapshot_section, property_name)
        tags = getattr(snapshot_section, tag_name)
        property_check(hoomd_buffer, property_dict, tags)
