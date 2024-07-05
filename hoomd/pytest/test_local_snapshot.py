# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Test that `LocalSnapshot` and `LocalSnapshotGPU` work."""

from copy import deepcopy
import hoomd
from hoomd.data.array import HOOMDGPUArray
import numpy as np
import pytest
try:
    # This try block is purely to allow testing locally without mpi4py. We could
    # require it for testing, and simplify the logic here. The CI containers all
    # have mpi4py.
    from mpi4py import MPI
except ImportError:
    skip_mpi4py = True
else:
    skip_mpi4py = False

skip_mpi4py = pytest.mark.skipif(skip_mpi4py,
                                 reason='mpi4py could not be imported.')

try:
    # We use the CUPY_IMPORTED variable to allow for local GPU testing without
    # CuPy installed. This code could be simplified to only work with CuPy, by
    # requiring its installation for testing. The CI containers already have
    # CuPy installed when build for the GPU.
    import cupy
    CUPY_IMPORTED = True
except ImportError:
    CUPY_IMPORTED = False
"""
_N and _types are distinct in that the local snapshot does not know about them.
We use the underscore to signify this. Those keys are skipped when testing the
local snapshots, though are still used to define the state.
"""

Np = 5
_particle_data = dict(
    _N=Np,
    position=dict(np_type=np.floating,
                  value=[[-1, -1, -1], [-1, -1, 0], [-1, 0, 0], [1, 1, 1],
                         [1, 0, 0]],
                  new_value=[[5, 5, 5]] * Np,
                  shape=(Np, 3)),
    velocity=dict(np_type=np.floating,
                  value=np.linspace(-4, 4, Np * 3).reshape((Np, 3)),
                  new_value=np.linspace(4, 8, Np * 3).reshape((Np, 3)),
                  shape=(Np, 3)),
    acceleration=dict(np_type=np.floating,
                      value=np.linspace(-4, 4, Np * 3).reshape((Np, 3)),
                      new_value=np.linspace(4, 8, Np * 3).reshape((Np, 3)),
                      shape=(Np, 3)),
    angmom=dict(np_type=np.floating,
                value=np.linspace(-3, 6, Np * 4).reshape((Np, 4)),
                new_value=np.linspace(1, 3, Np * 4).reshape((Np, 4)),
                shape=(Np, 4)),
    moment_inertia=dict(np_type=np.floating,
                        value=np.linspace(3, 12, Np * 3).reshape((Np, 3)),
                        new_value=np.linspace(0, 20, Np * 3).reshape((Np, 3)),
                        shape=(Np, 3)),
    # We don't care about a valid body specification here just that we can
    # retrieve and set it correctly.
    body=dict(np_type=np.uint32,
              value=np.linspace(4294967295, 10, Np, dtype=np.uint32),
              new_value=np.linspace(1, 20, Np, dtype=np.uint32),
              shape=(Np,)),
    # typeid is a signed integer in C++ despite always being nonnegative
    typeid=dict(np_type=np.int32,
                value=[0, 0, 0, 1, 1],
                new_value=[1, 1, 1, 0, 0],
                shape=(Np,)),
    mass=dict(np_type=np.floating,
              value=[5, 4, 3, 2, 1],
              new_value=[1, 2, 3, 4, 5],
              shape=(Np,)),
    charge=dict(np_type=np.floating,
                value=[1, 2, 3, 2, 1],
                new_value=[-1, -1, -3, -2, -1],
                shape=(Np,)),
    diameter=dict(np_type=np.floating,
                  value=[5, 2, 3, 2, 5],
                  new_value=[2, 1, 0.5, 1, 2],
                  shape=(Np,)),
    image=dict(np_type=np.int32,
               value=np.linspace(-10, 20, Np * 3,
                                 dtype=np.int32).reshape(Np, 3),
               new_value=np.linspace(-20, 10, Np * 3,
                                     dtype=np.int32).reshape(Np, 3),
               shape=(Np, 3)),
    tag=dict(np_type=np.uint32, value=None, shape=(Np,)),
    _types=['p1', 'p2'])

_particle_local_data = dict(
    net_force=dict(np_type=np.floating,
                   value=np.linspace(0.5, 4.5, Np * 3).reshape((Np, 3)),
                   new_value=np.linspace(6, 12, Np * 3).reshape((Np, 3)),
                   shape=(Np, 3)),
    net_torque=dict(np_type=np.floating,
                    value=np.linspace(-0.5, 2.5, Np * 3).reshape((Np, 3)),
                    new_value=np.linspace(12.75, 25, Np * 3).reshape((Np, 3)),
                    shape=(Np, 3)),
    net_virial=dict(np_type=np.floating,
                    value=np.linspace(-1.5, 6.5, Np * 6).reshape((Np, 6)),
                    new_value=np.linspace(9.75, 13.12, Np * 6).reshape((Np, 6)),
                    shape=(Np, 6)),
    net_energy=dict(np_type=np.floating,
                    value=np.linspace(0.5, 3.5, Np),
                    new_value=np.linspace(0, 4.2, Np),
                    shape=(Np,)),
)

Nb = 2
_bond_data = dict(_N=Nb,
                  typeid=dict(np_type=np.unsignedinteger,
                              value=[0, 1],
                              new_value=[1, 0],
                              shape=(Nb,)),
                  group=dict(np_type=np.unsignedinteger,
                             value=[[0, 1], [2, 3]],
                             new_value=[[1, 0], [3, 2]],
                             shape=(Nb, 2)),
                  tag=dict(np_type=np.unsignedinteger, value=None, shape=(Nb,)),
                  _types=['b1', 'b2'])

Na = 2
_angle_data = dict(_N=Na,
                   typeid=dict(np_type=np.unsignedinteger,
                               value=[1, 0],
                               new_value=[0, 1],
                               shape=(Na,)),
                   group=dict(np_type=np.unsignedinteger,
                              value=[[0, 1, 2], [2, 3, 4]],
                              new_value=[[1, 3, 4], [0, 2, 4]],
                              shape=(Na, 3)),
                   tag=dict(np_type=np.unsignedinteger, value=None,
                            shape=(Na,)),
                   _types=['a1', 'a2'])

Nd = 2
_dihedral_data = dict(_N=Nd,
                      typeid=dict(np_type=np.unsignedinteger,
                                  value=[1, 0],
                                  new_value=[0, 1],
                                  shape=(Nd,)),
                      group=dict(np_type=np.unsignedinteger,
                                 value=[[0, 1, 2, 3], [1, 2, 3, 4]],
                                 new_value=[[4, 3, 2, 1], [2, 4, 0, 1]],
                                 shape=(Nd, 4)),
                      tag=dict(np_type=np.unsignedinteger,
                               value=None,
                               shape=(Nd,)),
                      _types=['d1', 'd2'])

Ni = 2
_improper_data = dict(_N=Ni,
                      typeid=dict(np_type=np.unsignedinteger,
                                  value=[0, 0],
                                  shape=(Ni,)),
                      group=dict(np_type=np.unsignedinteger,
                                 value=[[3, 2, 1, 0], [1, 2, 3, 4]],
                                 new_value=[[1, 2, 3, 0], [4, 2, 3, 1]],
                                 shape=(Ni, 4)),
                      tag=dict(np_type=np.unsignedinteger,
                               value=None,
                               shape=(Ni,)),
                      _types=['i1'])

Nc = 3
_constraint_data = dict(
    _N=Nc,
    value=dict(np_type=np.floating,
               value=[2.5, 0.5, 2.],
               new_value=[3., 1.5, 1.],
               shape=(Nc,)),
    group=dict(np_type=np.unsignedinteger,
               value=[[0, 1], [2, 3], [1, 3]],
               new_value=[[4, 1], [3, 1], [2, 4]],
               shape=(Nc, 2)),
    tag=dict(np_type=np.unsignedinteger, value=None, shape=(Nc,)),
)

Npa = 2
_pair_data = dict(_N=Npa,
                  typeid=dict(np_type=np.unsignedinteger,
                              value=[0, 1],
                              new_value=[1, 0],
                              shape=(Npa,)),
                  group=dict(np_type=np.unsignedinteger,
                             value=[[0, 1], [2, 3]],
                             new_value=[[4, 1], [0, 3]],
                             shape=(Npa, 2)),
                  tag=dict(np_type=np.unsignedinteger, value=None,
                           shape=(Npa,)),
                  _types=['p1', 'p2'])

_global_dict = dict(rtag=dict(
    particles=dict(np_type=np.unsignedinteger, value=None, shape=(Np,)),
    bonds=dict(np_type=np.unsignedinteger, value=None, shape=(Nb,)),
    angles=dict(np_type=np.unsignedinteger, value=None, shape=(Na,)),
    dihedrals=dict(np_type=np.unsignedinteger, value=None, shape=(Nd,)),
    impropers=dict(np_type=np.unsignedinteger, value=None, shape=(Ni,)),
    constraints=dict(np_type=np.unsignedinteger, value=None, shape=(Nc,)),
    pairs=dict(np_type=np.unsignedinteger, value=None, shape=(Npa,)),
))


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

    snapshot = hoomd.Snapshot(device.communicator)

    if snapshot.communicator.rank == 0:
        snapshot.configuration.box = [2.1, 2.1, 2.1, 0, 0, 0]
        set_snapshot(snapshot, _particle_data, 'particles')
        set_snapshot(snapshot, _bond_data, 'bonds')
        set_snapshot(snapshot, _angle_data, 'angles')
        set_snapshot(snapshot, _dihedral_data, 'dihedrals')
        set_snapshot(snapshot, _improper_data, 'impropers')
        set_snapshot(snapshot, _constraint_data, 'constraints')
        set_snapshot(snapshot, _pair_data, 'pairs')
    return snapshot


@pytest.fixture(params=[
    'particles', 'bonds', 'angles', 'dihedrals', 'impropers', 'constraints',
    'pairs'
])
def snapshot_section(request):
    return request.param


@pytest.fixture(scope="function",
                params=[(section_name, prop_name, prop_dict)
                        for prop_name, global_prop_dict in _global_dict.items()
                        for section_name, prop_dict in global_prop_dict.items()
                        ],
                ids=lambda x: x[0] + '-' + x[1])
def global_property(request):
    return request.param


@pytest.fixture(
    scope='function',
    params=[(name, prop_name, prop_dict)
            for name, section_dict in [('particles', {
                **_particle_data,
                **_particle_local_data
            }), ('bonds', _bond_data), (
                'angles', _angle_data), (
                    'dihedrals',
                    _dihedral_data), (
                        'impropers',
                        _improper_data), (
                            'constraints',
                            _constraint_data), ('pairs', _pair_data)]
            for prop_name, prop_dict in section_dict.items()
            if not prop_name.startswith('_')],
    ids=lambda x: x[0] + '-' + x[1])
def section_name_dict(request):
    """Parameterization of expected values for local_snapshot properties.

    Examples include ``('particles', 'position', position_dict)`` where
    ``position_dict`` is the dictionary with the expected typecodes, shape, and
    value of particle positions.
    """
    return deepcopy(request.param)


@pytest.fixture(scope='function',
                params=['', 'ghost_', '_with_ghost'],
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


def general_array_equality(arr1, arr2):
    """Allows checking of equality with both HOOMDArrays and HOOMDGPUArrays."""
    if arr1.shape != arr2.shape:
        return False
    if any(np.issubdtype(a.dtype, np.floating) for a in (arr1, arr2)):
        if any(isinstance(a, HOOMDGPUArray) for a in (arr1, arr2)):
            return cupy.allclose(arr1, arr2)
        else:
            return np.allclose(arr1, arr2)
    else:
        return all(arr1.ravel() == arr2.ravel())


def check_type(data, prop_dict, tags):
    """Check that the expected dtype is found for local snapshots."""
    assert np.issubdtype(data.dtype, prop_dict['np_type'])


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

    Also tests error raising for read only arrays.
    """
    if len(tags) == 0:
        return
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


class TestLocalSnapshots:
    """Base class for CPU and GPU based localsnapshot tests."""

    @staticmethod
    def check_box(local_snapshot, global_box, ranks):
        """General check that ``box`` and ``local_box`` properties work."""
        assert type(local_snapshot.global_box) is hoomd.Box
        assert type(local_snapshot.local_box) is hoomd.Box

        assert local_snapshot.global_box == global_box
        # The local box and global box are equal if and only if
        # we run on a single rank.
        assert (local_snapshot.local_box == global_box) == (ranks == 1)

    def test_box(self, base_simulation, base_snapshot):
        sim = base_simulation()
        for lcl_snapshot_attr in self.get_snapshot_attr(sim):
            with getattr(sim.state, lcl_snapshot_attr) as data:
                self.check_box(data, sim.state.box,
                               sim.device.communicator.num_ranks)

    @staticmethod
    def check_tag_shape(base_snapshot, local_snapshot, group, ranks):
        mpi_comm = MPI.COMM_WORLD

        if base_snapshot.communicator.rank == 0:
            N = getattr(base_snapshot, group).N
        else:
            N = None
        N = mpi_comm.bcast(N, root=0)

        # check particles tag size
        if group == 'particles':
            total_len = mpi_comm.allreduce(len(local_snapshot.particles.tag),
                                           op=MPI.SUM)
            assert total_len == N
        else:
            local_snapshot_section = getattr(local_snapshot, group)
            if ranks > 1:
                assert len(local_snapshot_section.tag) <= N
            else:
                assert len(local_snapshot_section.tag) == N

    @skip_mpi4py
    @pytest.mark.cupy_optional
    def test_tags_shape(self, base_simulation, base_snapshot, snapshot_section):
        """Checks that tags are the appropriate size from local snapshots.

        tags are used for checking other shapes so this is necessary to validate
        those tests.
        """
        sim = base_simulation()
        for lcl_snapshot_attr in self.get_snapshot_attr(sim):
            with getattr(sim.state, lcl_snapshot_attr) as data:
                self.check_tag_shape(base_snapshot, data, snapshot_section,
                                     sim.device.communicator.num_ranks)

    @staticmethod
    def check_global_properties(prop, global_property_dict, N):
        assert prop.shape == global_property_dict['shape']
        assert np.issubdtype(prop.dtype, global_property_dict['np_type'])
        if isinstance(prop, HOOMDGPUArray) and not CUPY_IMPORTED:
            return
        else:
            if global_property_dict['value'] is not None:
                general_array_equality(prop, global_property_dict['value'])
            with pytest.raises(ValueError):
                prop[:] = 1

    @skip_mpi4py
    @pytest.mark.cupy_optional
    def test_cpu_global_properties(self, base_simulation, base_snapshot,
                                   global_property):
        section_name, prop_name, prop_dict = global_property
        sim = base_simulation()
        snapshot = sim.state.get_snapshot()

        mpi_comm = MPI.COMM_WORLD

        if snapshot.communicator.rank == 0:
            N = getattr(snapshot, section_name).N
        else:
            N = None
        N = mpi_comm.bcast(N, root=0)
        with sim.state.cpu_local_snapshot as data:
            self.check_global_properties(
                getattr(getattr(data, section_name), prop_name), prop_dict, N)

    @pytest.mark.cupy_optional
    def test_arrays_properties(self, base_simulation, section_name_dict, affix,
                               property_check):
        """This test makes extensive use of parameterizing in pytest.

        This test tests the type, shape, getting, and setting of array values in
        the local snapshot. We test all properties including ghost and both
        ghost and normal particles, bonds, etc.
        """
        name, property_name, property_dict = section_name_dict
        property_name = get_property_name_from_affix(property_name, affix)
        tag_name = get_property_name_from_affix('tag', affix)

        sim = base_simulation()
        for lcl_snapshot_attr in self.get_snapshot_attr(sim):
            with getattr(sim.state, lcl_snapshot_attr) as data:
                # gets the particle, bond, etc data
                snapshot_section = getattr(data, name)
                hoomd_buffer = getattr(snapshot_section, property_name)
                tags = getattr(snapshot_section, tag_name)
                property_check(hoomd_buffer, property_dict, tags)

    def test_run_failure(self, base_simulation):
        sim = base_simulation()
        for lcl_snapshot_attr in self.get_snapshot_attr(sim):
            with getattr(sim.state, lcl_snapshot_attr):
                with pytest.raises(RuntimeError):
                    sim.run(1)

    def test_setting_snapshot_failure(self, base_simulation, base_snapshot):
        sim = base_simulation()
        for lcl_snapshot_attr in self.get_snapshot_attr(sim):
            with getattr(sim.state, lcl_snapshot_attr):
                with pytest.raises(RuntimeError):
                    sim.state.set_snapshot(base_snapshot)

    @pytest.fixture
    def base_simulation(self, simulation_factory, base_snapshot):
        """Creates the simulation from the base_snapshot."""

        def factory():
            sim = simulation_factory(base_snapshot)
            with sim.state.cpu_local_snapshot as snap:
                particle_data = getattr(snap, 'particles')
                tags = snap.particles.tag
                for attr, inner_dict in _particle_local_data.items():
                    arr_values = np.array(inner_dict['value'])[tags]
                    getattr(particle_data, attr)[:] = arr_values
            return sim

        return factory

    def get_snapshot_attr(self, sim):
        if isinstance(sim.device, hoomd.device.CPU):
            yield 'cpu_local_snapshot'
        else:
            yield 'cpu_local_snapshot'
            yield 'gpu_local_snapshot'
