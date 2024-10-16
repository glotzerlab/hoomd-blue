# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Code to support unit and validation tests.

``conftest`` is not part of HOOMD-blue's public API.
"""

from collections.abc import Mapping
import logging
import pickle
import pytest
import hoomd
import atexit
import os
import numpy
import math
try:
    import sybil
    import sybil.parsers.rest
except ImportError:
    sybil = None

from hoomd.logging import LoggerCategories
from hoomd.snapshot import Snapshot
from hoomd import Simulation

logger = logging.getLogger()

pytest_plugins = ("hoomd.pytest_plugin_validate",)

devices = [hoomd.device.CPU]
_n_available_gpu = len(hoomd.device.GPU.get_available_devices())
_require_gpu_tests = (os.environ.get('_HOOMD_REQUIRE_GPU_TESTS_IN_GPU_BUILDS_')
                      is not None)
if hoomd.version.gpu_enabled and (_n_available_gpu > 0 or _require_gpu_tests):

    if os.environ.get('_HOOMD_SKIP_CPU_TESTS_WHEN_GPUS_PRESENT_') is not None:
        devices.pop(0)

    devices.append(hoomd.device.GPU)


def setup_sybil_tests(namespace):
    """Sybil setup function."""
    # Common imports.
    namespace['numpy'] = numpy
    namespace['hoomd'] = hoomd
    namespace['math'] = math

    namespace['gpu_not_available'] = _n_available_gpu == 0

    try:
        import cupy
    except ImportError:
        cupy = None

    namespace['cupy_not_available'] = cupy is None

    namespace['llvm_not_available'] = not hoomd.version.llvm_enabled


if sybil is not None:
    pytest_collect_file = sybil.Sybil(
        parsers=[
            sybil.parsers.rest.PythonCodeBlockParser(),
            sybil.parsers.rest.SkipParser(),
        ],
        pattern='*.py',
        # exclude files not yet tested with sybil
        excludes=[
            'hpmc/pair/user.py',
        ],
        setup=setup_sybil_tests,
        fixtures=['tmp_path']).pytest()


@pytest.fixture(scope='session', params=devices)
def device(request):
    """Parameterized Device fixture.

    Tests that use `device` will run once on the CPU and once on the GPU. The
    device object is session scoped to avoid device creation overhead when
    running tests.
    """
    d = request.param()

    # enable GPU error checking
    if isinstance(d, hoomd.device.GPU):
        d.gpu_error_checking = True

    return d


@pytest.fixture(scope='session')
def simulation_factory(device):
    """Make a Simulation object from a snapshot.

    TODO: duck type this to allow it to create state from GSD files as well
    """

    def make_simulation(snapshot=None, domain_decomposition=None):
        sim = Simulation(device)

        # reduce sorter grid to avoid Hilbert curve overhead in unit tests
        for tuner in sim.operations.tuners:
            if isinstance(tuner, hoomd.tune.ParticleSorter):
                tuner.grid = 8

        if snapshot is not None:
            if domain_decomposition is None:
                sim.create_state_from_snapshot(snapshot)
            else:
                sim.create_state_from_snapshot(snapshot, domain_decomposition)
        sim.seed = 22765
        return sim

    return make_simulation


@pytest.fixture(scope='session')
def one_particle_snapshot_factory(device):
    """Make a snapshot with a single particle."""

    def make_snapshot(particle_types=['A'],
                      dimensions=3,
                      position=(0, 0, 0),
                      orientation=(1, 0, 0, 0),
                      L=20):
        """Make the snapshot.

        Args:
            particle_types: List of particle type names
            dimensions: Number of dimensions (2 or 3)
            position: Position to place the particle
            orientation: Orientation quaternion to assign to the particle
            L: Box length

        The arguments position and orientation define the position and
        orientation of the particle.  When dimensions==3, the box is a cubic box
        with dimensions L by L by L. When dimensions==2, the box is a square box
        with dimensions L by L by 0.
        """
        s = Snapshot(device.communicator)
        N = 1

        if dimensions == 2 and position[2] != 0:
            raise ValueError(
                'z component of position must be zero for 2D simulation.')

        if s.communicator.rank == 0:
            box = [L, L, L, 0, 0, 0]
            if dimensions == 2:
                box[2] = 0
            s.configuration.box = box
            s.particles.N = N
            # shift particle positions slightly in z so MPI tests pass
            s.particles.position[0] = position
            s.particles.orientation[0] = orientation
            s.particles.types = particle_types
        return s

    return make_snapshot


@pytest.fixture(scope='session')
def two_particle_snapshot_factory(device):
    """Make a snapshot with two particles."""

    def make_snapshot(particle_types=['A'], dimensions=3, d=1, L=20):
        """Make the snapshot.

        Args:
            particle_types: List of particle type names
            dimensions: Number of dimensions (2 or 3)
            d: Distance apart to place particles
            L: Box length

        The two particles are placed at (-d/2, 0, 0) and (d/2,0,0). When,
        dimensions==3, the box is L by L by L. When dimensions==2, the box is
        L by L by 0.
        """
        s = Snapshot(device.communicator)
        N = 2

        if s.communicator.rank == 0:
            box = [L, L, L, 0, 0, 0]
            if dimensions == 2:
                box[2] = 0
            s.configuration.box = box
            s.particles.N = N
            # shift particle positions slightly in z so MPI tests pass
            s.particles.position[:] = [[-d / 2, 0, .1], [d / 2, 0, .1]]
            s.particles.types = particle_types
            if dimensions == 2:
                box[2] = 0
                s.particles.position[:] = [[-d / 2, 0.1, 0], [d / 2, 0.1, 0]]

        return s

    return make_snapshot


@pytest.fixture(scope='session')
def lattice_snapshot_factory(device):
    """Make a snapshot with particles on a cubic/square lattice."""

    def make_snapshot(particle_types=['A'], dimensions=3, a=1, n=7, r=0):
        """Make the snapshot.

        Args:
            particle_types: List of particle type names
            dimensions: Number of dimensions (2 or 3)
            a: Lattice constant
            n: Number of particles along each box edge. Pass a tuple for
                different lengths in each dimension.
            r: Fraction of `a` to randomly perturb particles

        Place particles on a simple cubic (dimensions==3) or square
        (dimensions==2) lattice. The box is cubic (or square) with a side length
        of `n * a`.

        Set `r` to randomly perturb particles a small amount off their lattice
        positions. This is useful in MD simulation testing so that forces do not
        cancel out by symmetry.
        """
        if isinstance(n, int):
            n = (n,) * dimensions
            if dimensions == 2:
                n += (1,)

        s = Snapshot(device.communicator)

        if s.communicator.rank == 0:
            box = [n[0] * a, n[1] * a, n[2] * a, 0, 0, 0]
            if dimensions == 2:
                box[2] = 0
            s.configuration.box = box

            s.particles.N = numpy.prod(n)
            s.particles.types = particle_types

            if any(nx == 0 for nx in n):
                return s

            # create the lattice
            ranges = [numpy.arange(-nx / 2, nx / 2) for nx in n]
            x, y, z = numpy.meshgrid(*ranges)
            lattice_position = numpy.vstack(
                (x.flatten(), y.flatten(), z.flatten())).T
            pos = (lattice_position + 0.5) * a
            if dimensions == 2:
                pos[:, 2] = 0
            # perturb the positions
            if r > 0:
                shift = numpy.random.uniform(-r, r, size=(s.particles.N, 3))
                if dimensions == 2:
                    shift[:, 2] = 0
                pos += shift
            s.particles.position[:] = pos

        return s

    return make_snapshot


@pytest.fixture(scope='session')
def fcc_snapshot_factory(device):
    """Make a snapshot with particles in a fcc structure."""

    def make_snapshot(particle_types=['A'], a=1, n=7, r=0):
        """Make a snapshot with particles in a fcc structure.

        Args:
            particle_types: List of particle type names
            a: Lattice constant
            n: Number of unit cells along each box edge
            r: Amount to randomly perturb particles in x,y,z

        Place particles in a fcc structure. The box is cubic with a side length
        of ``n * a``. There will be ``4 * n**3`` particles in the snapshot.
        """
        s = Snapshot(device.communicator)

        if s.communicator.rank == 0:
            # make one unit cell
            s.configuration.box = [a, a, a, 0, 0, 0]
            s.particles.N = 4
            s.particles.types = particle_types
            s.particles.position[:] = [
                [0, 0, 0],
                [0, a / 2, a / 2],
                [a / 2, 0, a / 2],
                [a / 2, a / 2, 0],
            ]
            # and replicate it
            s.replicate(n, n, n)

        # perturb the positions
        if r > 0:
            shift = numpy.random.uniform(-r, r, size=(s.particles.N, 3))
            s.particles.position[:] += shift

        return s

    return make_snapshot


@pytest.fixture(autouse=True)
def skip_mpi(request):
    """Skip tests marked ``serial`` when running with MPI."""
    if request.node.get_closest_marker('serial'):
        if 'device' in request.fixturenames:
            if request.getfixturevalue('device').communicator.num_ranks > 1:
                pytest.skip('Test does not support MPI execution')
        else:
            raise ValueError('skip_mpi requires the *device* fixture')


@pytest.fixture(autouse=True)
def only_gpu(request):
    """Skip CPU tests marked ``gpu``."""
    if request.node.get_closest_marker('gpu'):
        if 'device' in request.fixturenames:
            if not isinstance(request.getfixturevalue('device'),
                              hoomd.device.GPU):
                pytest.skip('Test is run only on GPU(s).')
        else:
            raise ValueError('only_gpu requires the *device* fixture')


@pytest.fixture(autouse=True)
def only_cpu(request):
    """Skip GPU tests marked ``cpu``."""
    if request.node.get_closest_marker('cpu'):
        if 'device' in request.fixturenames:
            if not isinstance(request.getfixturevalue('device'),
                              hoomd.device.CPU):
                pytest.skip('Test is run only on CPU(s).')
        else:
            raise ValueError('only_cpu requires the *device* fixture')


@pytest.fixture(scope='function', autouse=True)
def numpy_random_seed():
    """Seed the numpy random number generator.

    Automatically reset the numpy random seed at the start of each function
    for reproducible tests.
    """
    numpy.random.seed(42)


@pytest.fixture(scope="module")
def rng():
    """Return a NumPy random generator."""
    return numpy.random.default_rng(564)


def pytest_configure(config):
    """Add markers to pytest configuration."""
    config.addinivalue_line(
        "markers",
        "serial: Tests that will not execute with more than 1 MPI process")
    config.addinivalue_line("markers",
                            "gpu: Tests that should only run on the gpu.")
    config.addinivalue_line(
        "markers",
        "cupy_optional: tests that should pass with and without CuPy.")
    config.addinivalue_line("markers", "cpu: Tests that only run on the CPU.")
    config.addinivalue_line("markers", "gpu: Tests that only run on the GPU.")


def abort(exitstatus):
    """Call MPI_Abort when pytest tests fail."""
    # get a default mpi communicator
    communicator = hoomd.communicator.Communicator()
    # abort the deadlocked ranks
    hoomd._hoomd.abort_mpi(communicator.cpp_mpi_conf, exitstatus)


def pytest_sessionfinish(session, exitstatus):
    """Finalize pytest session.

    MPI tests may fail on one rank but not others. To prevent deadlocks in these
    situations, this code calls ``MPI_Abort`` when pytest is exiting with a
    non-zero exit code. **pytest** should be run with the ``-x`` option so that
    it exits on the first error.
    """
    if exitstatus != 0 and hoomd.version.mpi_enabled:
        atexit.register(abort, exitstatus)


expected_loggable_params = {
    'energy': {
        'category': LoggerCategories.scalar,
        'default': True
    },
    'energies': {
        'category': LoggerCategories.particle,
        'default': True
    },
    'forces': {
        'category': LoggerCategories.particle,
        'default': True
    },
    'torques': {
        'category': LoggerCategories.particle,
        'default': True
    },
    'virials': {
        'category': LoggerCategories.particle,
        'default': True
    },
    'additional_energy': {
        'category': LoggerCategories.scalar,
        'default': True
    },
    'additional_virial': {
        'category': LoggerCategories.sequence,
        'default': True
    }
}


def logging_check(cls, expected_namespace, expected_loggables):
    """Function for testing object logging specification.

    Args:
        cls (object): The loggable class to test for the correct logging
            specfication.
        expected_namespace (tuple[str]): A tuple of strings that indicate the
            expected namespace minus the class name.
        expected_loggables (dict[str, dict[str, Any]]): A dict with string keys
            representing the expected loggable quantities. If the value for a
            key is ``None`` then, only check for the existence of the loggable
            quantity. Otherwise, the inner `dict` should consist of some
            combination of the keys ``default`` and ``category`` indicating the
            expected value of each for the loggable.
    """
    # Check namespace
    assert all(log_quantity.namespace == expected_namespace + (cls.__name__,)
               for log_quantity in cls._export_dict.values())

    # Check specific loggables
    def check_loggable(cls, name, properties):
        assert name in cls._export_dict
        if properties is None:
            return None
        log_quantity = cls._export_dict[name]
        for name, prop in properties.items():
            assert getattr(log_quantity, name) == prop

    for name, properties in expected_loggables.items():
        check_loggable(cls, name, properties)


def _check_obj_attr_compatibility(a, b):
    """Check key compatibility."""
    a_keys = set(a.__dict__.keys())
    b_keys = set(b.__dict__.keys())
    different_keys = a_keys.symmetric_difference(b_keys) - a._skip_for_equality
    if different_keys == {}:
        return True
    # Check through reserved attributes with defaults to ensure that the
    # difference isn't an initialized default.
    compatible = True
    filtered_differences = set(different_keys)
    for key in different_keys:
        if key in a._reserved_default_attrs:
            default = a._reserved_default_attrs[key]
            if callable(default):
                default = default()
            if getattr(a, key, default) == getattr(b, key, default):
                filtered_differences.remove(key)
                continue
        else:
            compatible = False

    if compatible:
        return True

    logger.debug(f"In equality check, incompatible attrs found "
                 f"{filtered_differences}.")
    return False


def equality_check(a, b):
    """Check equality between to instances of _HOOMDBaseObject."""

    def check_item(x, y, attr):
        if isinstance(x, hoomd.operation._HOOMDGetSetAttrBase):
            equality_check(x, y)
            return
        if isinstance(x, Mapping):
            for k, v in x.items():
                assert k in y, f"For attr {attr}, key difference {k}"
                check_item(v, y[k], ".".join((attr, str(k))))
            return
        if not isinstance(x, str) and hasattr(x, "__len__"):
            assert len(x) == len(y)
            for i, (v_x, v_y) in enumerate(zip(x, y)):
                check_item(v_x, v_y, attr + f"[{i}]")
            return
        if isinstance(x, float):
            assert numpy.isclose(x, y), f"attr '{attr}' not equal:"
            return
        assert x == y, f"attr '{attr}' not equal:"

    if not isinstance(a, hoomd.operation._HOOMDGetSetAttrBase):
        return a == b
    assert type(a) is type(b)

    _check_obj_attr_compatibility(a, b)

    for attr in a.__dict__:
        if attr in a._skip_for_equality:
            continue

        if attr == "_param_dict":
            param_keys = a._param_dict.keys()
            b_param_keys = b._param_dict.keys()
            # Check key equality
            assert param_keys == b_param_keys, "Incompatible param_dict keys:"
            # Check item equality
            for key in param_keys:
                check_item(a._param_dict[key], b._param_dict[key], key)
            continue

        if attr == "_typeparam_dict":
            keys = a._typeparam_dict.keys()
            b_keys = b._typeparam_dict.keys()
            # Check key equality
            assert keys == b_keys, "Incompatible _typeparam_dict:"
            # Check item equality
            for key in keys:
                for type_, value in a._typeparam_dict[key].items():
                    check_item(value, b._typeparam_dict[key][type_], ".".join(
                        (key, str(type_))))
            continue

        check_item(a.__dict__[attr], b.__dict__[attr], attr)


def pickling_check(instance):
    """Test that an instance can be pickled and unpickled."""
    pkled_instance = pickle.loads(pickle.dumps(instance))
    equality_check(instance, pkled_instance)


def operation_pickling_check(instance, sim):
    """Test that an operation can be pickled and unpickled."""
    pickling_check(instance)
    sim.operations += instance
    sim.run(0)
    pickling_check(instance)


def autotuned_kernel_parameter_check(instance, activate, all_optional=False):
    """Check that an AutotunedObject behaves as expected."""
    instance.tune_kernel_parameters()

    initial_kernel_parameters = instance.kernel_parameters

    if isinstance(instance._simulation.device, hoomd.device.CPU):
        # CPU instances have no parameters and are always complete.
        assert initial_kernel_parameters == {}
        assert instance.is_tuning_complete
    else:
        # GPU instances have parameters and start incomplete.
        assert initial_kernel_parameters != {}

        # is_tuning_complete is True when all tuners are optional.
        if not all_optional:
            assert not instance.is_tuning_complete

        activate()

        assert instance.kernel_parameters != initial_kernel_parameters

        # Note: It is not practical to automatically test that
        # `is_tuning_complete` is eventually achieved as failure results in an
        # infinite loop. Also, some objects (like neighbor lists) require
        # realistic simulation conditions to test adequately. `hoomd-benchmarks`
        # tests that tuning completes in all benchmarks.

        # Ensure that we can set parameters.
        instance.kernel_parameters = initial_kernel_parameters
        activate()
        assert instance.kernel_parameters == initial_kernel_parameters


class ListWriter(hoomd.custom.Action):
    """Log a single quantity to a list.

    On each triggered timestep, access the given attribute and add the value
    to `data`.

    Args:
        operation: Operation to log
        attribute: Name of the attribute to log

    Attributes:
        data (list): Saved data
    """

    def __init__(self, operation, attribute):
        self._operation = operation
        self._attribute = attribute
        self.data = []

    def act(self, timestep):
        """Add the attribute value to the list."""
        self.data.append(getattr(self._operation, self._attribute))


class ManyListWriter(hoomd.custom.Action):
    """Log many quantities to a list.

    On each triggered timestep, access the attributes given to the constructor
    and append the data to lists.

    Args:
        list_tuples (list(tuple)):
            List of pairs (operation, attribute) similar to the two arguments
            given to the ListWriter constructor.
    """

    def __init__(self, list_tuples):
        self._listwriters = [ListWriter(op, attr) for op, attr in list_tuples]

    def act(self, timestep):
        """Add each attribute value to the listwriter for that attribute."""
        for listwriter in self._listwriters:
            listwriter.act(timestep)

    @property
    def data(self):
        """tuple(list): Data for each attribute specified in the constructor."""
        return tuple([w.data for w in self._listwriters])


def index_id(i):
    """Used for pytest fixture ids of indices."""
    return f"(i={i})"


class Options:
    """Item should be one of a set number of values.

    For use with `Generator`.
    """

    def __init__(self, *options):
        self.options = options


class Either:
    """Item should be a value from a set number of specs.

    For use with `Generator`.
    """

    def __init__(self, *options):
        self.options = options


class Generator:
    """Generates random values of various specifications based on method.

    The purpose is similar to property testing libraries like hypothesis in that
    it enables automatic testing with a variety of values. This implementation
    is nowhere near as sophicisticated as those packages. However, for general
    purpose testing of property setting and manipulation, this is sufficient.

    Note:
        Developers should use this over adding ad-hoc values to tests. This
        should not be used when testing the behavior of an object in a
        simulation where manual specified values is often important.

    Note:
        If more flexibility is needed small classes like `Options` would work
        well for instance a ``Float`` class which specified the range of values
        to assume would be quite simple to add.
    """
    alphabet = [
        char for char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ]

    def __init__(self, rng, max_float=1e9, max_int=1_000_000):
        self.rng = rng
        self.max_float = max_float
        self.max_int = max_int

    def __call__(self, spec):
        """Return a random valid value from the specification."""
        if isinstance(spec, dict):
            return self.dict(spec)
        if isinstance(spec, tuple):
            return self.tuple(spec)
        if isinstance(spec, list):
            return self.list(spec)
        if isinstance(spec, Either):
            return self.either(spec)
        if isinstance(spec, Options):
            return self.option(spec)
        return {
            str: self.str,
            float: self.float,
            int: self.int,
            bool: self.bool,
            numpy.ndarray: self.ndarray,
            hoomd.variant.Variant: self.variant,
            None: self.none
        }[spec]()

    def tuple(self, spec):
        """Return an appropriately structured tuple."""
        return tuple(self(inner_spec) for inner_spec in spec)

    def list(self, spec, max_size=20):
        """Return an appropriately structured list."""
        return [self(spec[0]) for _ in range(self.rng.integers(max_size))]

    def dict(self, spec):
        """Return an appropriately structured dict."""
        return {k: self(inner_spec) for k, inner_spec in spec.items()}

    def none(self):
        """Return ``None``."""
        return None

    def int(self, max_=None):
        """Return a random integer."""
        max_ = self.max_int if max_ is None else max_
        return self.rng.integers(max_).item()

    def float(self, max_=None):
        """Return a random float."""
        max_ = self.max_float if max_ is None else max_
        return max_ * (self.rng.random() - 0.5)

    def bool(self):
        """Return a random Boolean."""
        return bool(self.int(2))

    def str(self, max_length=20):
        """Return a random string."""
        length = self.int(max_length) + 1
        characters = [
            self.rng.choice(self.alphabet)
            for _ in range(self.rng.integers(length))
        ]
        return "".join(characters)

    def ndarray(self, shape=(None,), dtype="float64"):
        """Return a ndarray of specified shape and dtype.

        A value of None in shape means any length.
        """
        shape = tuple(i if i is not None else self.int(20) for i in shape)
        return (100 * self.rng.random(numpy.prod(shape))
                - 50).reshape(shape).astype(dtype)

    def variant(self):
        """Return a random `hoomd.variant.Variant` or `float`."""
        classes = ((hoomd.variant.Constant, (float,)),
                   (hoomd.variant.Cycle, (float, float, int, int, int, int,
                                          int)), (hoomd.variant.Ramp,
                                                  (float, float, int, int)),
                   (hoomd.variant.Power, (float, float, int, int,
                                          int)), (float, (float,)))
        cls, spec = classes[self.rng.integers(len(classes))]
        return cls(*self(spec))

    def option(self, spec):
        """Return one of the specified options."""
        return spec.options[self.rng.integers(len(spec.options))]

    def either(self, spec):
        """Return a random value from one of the specified specifications."""
        return self(spec.options[self.rng.integers(len(spec.options))])


class ClassDefinition:
    """Provides a class interface for working with classes with `Generator`.

    See methods for usage.

    Note:
        For further development, methods for dealing with type_parameters would
        be helpful for testing.
    """

    def __init__(
        self,
        cls,
        constructor_spec,
        attribute_spec=None,
        generator=None,
    ):
        self.cls = cls
        self.constructor_spec = constructor_spec
        if attribute_spec is None:
            attribute_spec = constructor_spec
        self.attribute_spec = attribute_spec
        if generator is None:
            generator = Generator(numpy.random.default_rng())
        self.generator = generator

    def generate_init_args(self):
        """Get arguments necessary for constructing the object."""
        return self.generator(self.constructor_spec)

    def generate_all_attr_change(self):
        """Get arguments to test setting attributes."""
        return {
            k: self.generator(spec) for k, spec in self.attribute_spec.items()
        }


class BaseCollectionsTest:
    """Basic extensible test suite for collection classes.

    This class and subclasses allow for extensive testing of list, tuple, dict,
    and set like objects. Given that different data structure classes require
    different specific in testing (see `to_base` for an example) these classes
    can have class specific accommodations. However, this code smell is worth
    the increase in testing and reduction in code.

    For usage of this and subclasses see ``hoomd.pytest.test_collections.py``,
    and the documentation of the provided methods.

    Note:
        This test suite isn't meant to contain class specific tests, merely
        those of the given data structure. Class specific tests should be added
        to the speceific test class for the tested class.

    Note:
        Not using `abc.ABC` was a a conscious decision. ``pytest`` fails when
        test classes inherit from `abc.ABC`
    """

    alphabet = [char for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]

    def to_base(self, obj):
        """Use to convert an item removed from a data structure.

        This is only necessary for things like _HoomdDict objects since they
        will error when isolated. Having something this niche isn't ideal, but
        reduces the amount of testing code signifcantly.
        """
        if hasattr(obj, "to_base"):
            return obj.to_base()
        return obj

    @pytest.fixture
    def generate_plain_collection(self):
        """Return a function that generates plain collections for tests.

        For a list this would be a plain list, a mapping a dict, etc. The
        returned function should take in an integer and return a data structure
        of that size.

        Note:
            For cases where the data structure size is not variable the function
            can ignore the passed in argument.
        """
        raise NotImplementedError

    def check_equivalent(self, a, b):
        """Assert whether two collections are equivalent for test purposes.

        This takes in the "plain" data structure and an instance of the tested
        class. In general this does not need to be overwritten by subclasses,
        but `is_equal` does.
        """
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert self.is_equal(x, y)

    def is_equal(self, a, b):
        """Return whether two collection items are equal.

        Default to the safest assumption which is identity equality. For more
        specific checks overwrite this. This is the main work horse for checks
        in the suite.

        Note:
            For mapping types `is_equal` has to deal with the key values as
            well.
        """
        return a is b

    def final_check(self, test_collection):
        """Perform any final assert on the collection like object.

        For test that modify a collection this is called at the end to perform
        any specific tests relevant to the currently tested class. For instance,
        this can test that a synced list is kept up to date with modification.
        """
        assert True

    _generator = Generator(numpy.random.default_rng(15656456))

    @property
    def generator(self):
        """Return the value generator.

        Many test rely on the generation of random numbers. To ensure
        reproducible this should have a constant seed.
        """
        return self._generator

    @pytest.fixture(autouse=True, params=(5, 10, 20))
    def n(self, request):
        """Fixture that controls tested collection sizes.

        Note:
            This can also be used to control the number of examples each test
            gets, making this function is useful even when data structure size
            does not change.
        """
        return request.param

    @pytest.fixture(scope="function")
    def plain_collection(self, n, generate_plain_collection):
        """Return a plain collection with specified items.

        Used by `populated_collection`.
        """
        return generate_plain_collection(n)

    @pytest.fixture(scope="function")
    def empty_collection(self):
        """Return an empty test class collection.

        This is required by `populated_collection`.
        """
        raise NotImplementedError

    @pytest.fixture(scope="function")
    def populated_collection(self, empty_collection, plain_collection):
        """Return a test collection and the plain data the collection uses.

        This is implemented by subclasses and in general is not required to be
        overwritten. The exception is immutable classes.
        """
        raise NotImplementedError

    def test_contains(self, populated_collection, generate_plain_collection):
        """Test __contains__."""
        test_collection, plain_collection = populated_collection
        for item in plain_collection:
            assert item in test_collection
        # This does not guarentee that items that do not exist in the collection
        # will be tested for inclusion, but with a suffiently broad random
        # collection generation this is all but guaranteed.
        new_collection = generate_plain_collection(5)
        for item in new_collection:
            # Having a NumPy array anywhere in another collection reeks havoc
            # because of NumPy's use of == as a elementwise operator.
            if isinstance(item, numpy.ndarray):
                contains = any(
                    test_collection._numpy_equality(item, item2)
                    for item2 in plain_collection)
            else:
                if any(isinstance(a, numpy.ndarray) for a in plain_collection):
                    contains = False
                    for a in plain_collection:
                        if isinstance(a, numpy.ndarray):
                            contains |= test_collection._numpy_equality(a, item)
                        else:
                            contains |= a == item
                        if contains:
                            break
                else:
                    contains = item in plain_collection
            if contains:
                assert item in test_collection
            else:
                assert item not in test_collection

    def test_len(self, populated_collection):
        """Test __len__."""
        test_collection, plain_collection = populated_collection
        assert len(test_collection) == len(plain_collection)

    def test_iter(self, populated_collection):
        """Test __iter__."""
        test_collection, plain_collection = populated_collection
        for t_item, p_item in zip(test_collection, plain_collection):
            assert self.is_equal(t_item, p_item)


class BaseSequenceTest(BaseCollectionsTest):
    """Basic extensible test suite for tuple-like classes."""
    _negative_indexing = True
    _allow_slices = True

    def test_getitem(self, populated_collection):
        """Test __getitem__."""
        test_collection, plain_collection = populated_collection
        with pytest.raises(IndexError):
            _ = test_collection[len(test_collection)]
        for i, p_item in enumerate(plain_collection):
            assert self.is_equal(test_collection[i], p_item)
        if self._allow_slices:
            assert all(
                self.is_equal(t, p)
                for t, p in zip(test_collection[:], plain_collection))
            assert all(
                self.is_equal(t, p)
                for t, p in zip(test_collection[1:], plain_collection[1:]))
        if self._negative_indexing:
            for i in range(-1, -len(plain_collection), -1):
                assert self.is_equal(test_collection[i], plain_collection[i])


class BaseListTest(BaseSequenceTest):
    """Basic extensible test suite for list-like classes."""

    @pytest.fixture
    def populated_collection(self, empty_collection, plain_collection):
        """Return a test collection and the plain data the collection uses."""
        empty_collection.extend(plain_collection)
        return empty_collection, plain_collection

    @pytest.fixture(params=(3, 6, 11, -2, -15), ids=index_id)
    def delete_index(self, request):
        """Determines the indices used for test_delitem.

        Note:
            At least one index should be out of range and one negative to test
            proper behavior.
        """
        return request.param

    def test_delitem(self, delete_index, populated_collection):
        """Test __delitem__."""
        if not self._negative_indexing and delete_index < 0:
            return
        test_list, plain_list = populated_collection
        # out of bounds test
        if delete_index >= len(test_list) or delete_index < -len(test_list):
            with pytest.raises(IndexError):
                del test_list[delete_index]
            return
        # single index test
        old_item = test_list[delete_index]
        del test_list[delete_index]
        del plain_list[delete_index]
        self.check_equivalent(test_list, plain_list)
        assert self.to_base(old_item) not in test_list
        # test slice deletion
        if not self._allow_slices:
            return
        old_items = test_list[1:]
        del test_list[1:]
        assert len(test_list) == 1
        assert all(
            self.to_base(old_item) not in test_list for old_item in old_items)
        self.final_check(test_list)

    def test_append(self, empty_collection, plain_collection):
        """Test append."""
        for i, item in enumerate(plain_collection, start=1):
            empty_collection.append(item)
            assert len(empty_collection) == i
            assert self.is_equal(item, empty_collection[i - 1])
        self.check_equivalent(empty_collection, plain_collection)
        self.final_check(empty_collection)

    @pytest.fixture(params=(3, 6, 11, -1, -10), ids=index_id)
    def insert_index(self, request):
        """Determines the indices used for test_insert.

        Note:
            At least one index should be greater than the list size to test
            proper performance. At least one should be negative too.
        """
        return request.param

    def test_insert(self, insert_index, empty_collection, plain_collection):
        """Test insert."""
        if not self._negative_indexing and insert_index < 0:
            return
        check_collection = []
        empty_collection.extend(plain_collection[:-1])
        check_collection.extend(plain_collection[:-1])
        empty_collection.insert(insert_index, plain_collection[-1])
        check_collection.insert(insert_index, plain_collection[-1])
        N = len(plain_collection) - 1
        # The fancy indexing is just the insert_index or the last item in
        # the list with a positive index or the beginning of the list or
        # insert_index away from the end of the list when negative which is the
        # expected behavior for insert.
        if insert_index >= 0:
            expected_index = min(N, insert_index)
        else:
            expected_index = max(0, N + insert_index)
        assert len(empty_collection) == len(plain_collection)
        assert self.is_equal(
            empty_collection[expected_index],
            plain_collection[-1],
        )
        self.check_equivalent(empty_collection, check_collection)
        self.final_check(empty_collection)

    def test_extend(self, empty_collection, plain_collection):
        """Test extend."""
        empty_collection.extend(plain_collection)
        self.check_equivalent(empty_collection, plain_collection)
        self.final_check(empty_collection)

    def test_clear(self, populated_collection):
        """Test clear."""
        test_list, plain_list = populated_collection
        test_list.clear()
        assert len(test_list) == 0
        self.final_check(test_list)

    @pytest.fixture(params=(3, 6, 11, -3, -10), ids=index_id)
    def setitem_index(self, request):
        """Determines the indices used for test_setitem.

        Note:
            At least one index should be larger than the list size and another
            negatiev to test proper behavior.
        """
        return request.param

    def test_setitem(self, setitem_index, populated_collection,
                     generate_plain_collection):
        """Test __setitem__."""
        if not self._negative_indexing and setitem_index < 0:
            return
        test_list, plain_list = populated_collection
        item = generate_plain_collection(1)[0]
        # Test out of bounds setting
        if setitem_index >= len(test_list) or setitem_index < -len(test_list):
            with pytest.raises(IndexError):
                test_list[setitem_index] = item
            return
        # Basic test
        test_list[setitem_index] = item
        assert self.is_equal(test_list[setitem_index], item)
        assert len(test_list) == len(plain_list)
        self.final_check(test_list)

    @pytest.fixture(params=(3, 6, 11, -1, -10), ids=index_id)
    def pop_index(self, request):
        """Determines the indices used for test_pop.

        Note:
            At least one index should be larger than the list size and another
            negative to test proper behavior.
        """
        return request.param

    def test_pop(self, pop_index, populated_collection):
        """Test pop."""
        if not self._negative_indexing and pop_index < 0:
            return
        test_list, plain_list = populated_collection
        if pop_index >= len(test_list) or pop_index < -len(test_list):
            with pytest.raises(IndexError):
                test_list.pop(pop_index)
            return

        item = test_list.pop(pop_index)
        assert self.is_equal(self.to_base(item), plain_list[pop_index])
        plain_list.pop(pop_index)
        self.check_equivalent(test_list, plain_list)
        self.final_check(test_list)

    def test_empty_pop(self, populated_collection):
        """Test pop without argument."""
        test_list, plain_list = populated_collection
        item = test_list.pop()
        assert self.is_equal(self.to_base(item), plain_list[-1])
        plain_list.pop()
        self.check_equivalent(test_list, plain_list)
        self.final_check(test_list)

    def test_remove(self, populated_collection):
        """Test remove."""
        test_list, plain_list = populated_collection
        remove_index = self.generator.int(len(plain_list))
        test_list.remove(plain_list[remove_index])
        assert plain_list[remove_index] not in test_list
        plain_list.remove(plain_list[remove_index])
        self.check_equivalent(test_list, plain_list)


class BaseMappingTest(BaseCollectionsTest):
    """Basic extensible test suite for mapping classes."""

    # Some mapping classes do not allow new keys. This enables branched testing
    # for those that do and don't.
    _allow_new_keys = True
    # Classes that do not allow for the removal of keys should set this to the
    # appropriate error type.
    _deletion_error = None
    # Whether the class has a default system.
    _has_default = False

    @pytest.fixture
    def populated_collection(self, empty_collection, plain_collection):
        """Return a test mapping and the plain data the collection uses."""
        empty_collection.update(plain_collection)
        return empty_collection, plain_collection

    def check_equivalent(self, a, b):
        """Assert whether two collections are equivalent for test purposes."""
        assert set(a) == set(b)
        for key in a:
            assert self.is_equal(a[key], b[key])

    def random_keys(self):
        """Generate random string keys.

        Note:
            This can be used to geneate random strings too.

        Warning:
            This is an infinite generator.
        """
        while True:
            yield self.generator.str()

    def choose_random_key(self, mapping):
        """Pick a random existing key from mapping.

        Fails on an empty mapping.
        """
        return list(mapping)[self.generator.int(len(mapping))]

    def test_iter(self, populated_collection):
        """Test __iter__."""
        test_mapping, plain_mapping = populated_collection
        cnt = 0
        for _ in test_mapping:
            cnt += 1
        assert cnt == len(plain_mapping)
        assert set(test_mapping) == plain_mapping.keys()

    def test_contains(self, populated_collection):
        """Test __contains__."""
        test_collection, plain_collection = populated_collection
        for key in plain_collection:
            assert key in test_collection
        # Test non-existent keys
        cnt = 0
        for key in self.random_keys():
            if key in plain_collection:
                assert key in test_collection
            else:
                assert key not in test_collection
                cnt += 1
                if cnt == 5:
                    break

    def test_getitem(self, populated_collection):
        """Test __getitem__."""
        test_mapping, plain_mapping = populated_collection
        for key, value in plain_mapping.items():
            assert self.is_equal(test_mapping[key], value)
        # Test non-existent keys. With a default this will not error otherwise
        # it will. Currently we expect the default to be named default.
        if self._has_default:
            for key in self.random_keys():
                if key not in test_mapping:
                    value = test_mapping[key]
                    assert self.is_equal(value, test_mapping.default)
                    return
        with pytest.raises(KeyError):
            for key in self.random_keys():
                if key not in test_mapping:
                    _ = test_mapping[key]
                    break

    def test_delitem(self, populated_collection):
        """Test __delitem__."""
        test_mapping, plain_mapping = populated_collection
        # Test that non-existent keys error appropriately.
        expected_error = (KeyError,)
        if self._deletion_error is not None:
            expected_error = expected_error + (self._deletion_error,)
        with pytest.raises(expected_error):
            for key in self.random_keys():
                if key not in test_mapping:
                    del test_mapping[key]
        # base test
        key = self.choose_random_key(test_mapping)
        if self._deletion_error is not None:
            with pytest.raises(self._deletion_error):
                del test_mapping[key]
            return

        del test_mapping[key]
        del plain_mapping[key]
        self.check_equivalent(test_mapping, plain_mapping)
        assert key not in test_mapping
        self.final_check(test_mapping)

    def test_clear(self, populated_collection):
        """Test clear."""
        test_mapping, plain_mapping = populated_collection
        if self._deletion_error is not None:
            with pytest.raises(self._deletion_error):
                test_mapping.clear()
            return
        test_mapping.clear()
        assert len(test_mapping) == 0
        self.final_check(test_mapping)

    @pytest.fixture
    def setitem_key_value(self):
        """Determines the indices used for test_setitem.

        Required for all subclasses.

        Note:
            A non-existent key should be included in this.
        """
        raise NotImplementedError

    def test_setitem(self, setitem_key_value, populated_collection):
        """Test __setitem__."""
        test_mapping, plain_mapping = populated_collection
        key, value = setitem_key_value

        if not self._allow_new_keys and key not in test_mapping:
            with pytest.raises(KeyError):
                test_mapping[key] = value
            return

        test_mapping[key] = value
        assert self.is_equal(test_mapping[key], value)
        if key in plain_mapping:
            assert len(test_mapping) == len(plain_mapping)
        else:
            assert len(test_mapping) == len(plain_mapping) + 1
        self.final_check(test_mapping)

    def test_pop(self, populated_collection):
        """Test pop."""
        test_mapping, plain_mapping = populated_collection
        # Test for error with non-existent keys.
        expected_error = (KeyError,)
        if self._deletion_error is not None:
            expected_error = expected_error + (self._deletion_error,)
        with pytest.raises(expected_error):
            for key in self.random_keys():
                if key not in test_mapping:
                    test_mapping.pop(key)
                    break
        # base test
        key = self.choose_random_key(test_mapping)
        if self._deletion_error is not None:
            with pytest.raises(self._deletion_error):
                item = test_mapping.pop(key)
            return

        item = test_mapping.pop(key)
        assert self.is_equal(item, plain_mapping[key])
        plain_mapping.pop(key)
        self.check_equivalent(test_mapping, plain_mapping)
        self.final_check(test_mapping)
        test_mapping.pop(key, None)

    def test_keys(self, populated_collection):
        """Test keys."""
        test_mapping, plain_mapping = populated_collection
        assert set(test_mapping.keys()) == plain_mapping.keys()

    def test_values(self, populated_collection):
        """Test __iter__."""
        test_mapping, plain_mapping = populated_collection
        # We rely on keys() and values() using the same ordering
        for key, item in zip(test_mapping.keys(), test_mapping.values()):
            assert self.is_equal(item, plain_mapping[key])

    def test_items(self, populated_collection):
        """Test __iter__."""
        test_mapping, plain_mapping = populated_collection
        for key, value in test_mapping.items():
            assert self.is_equal(value, plain_mapping[key])

    def test_update(self, populated_collection, generate_plain_collection, n):
        """Test update."""
        test_mapping, plain_mapping = populated_collection
        new_mapping = generate_plain_collection(max(n - 1, 1))
        test_mapping.update(new_mapping)
        for key in new_mapping:
            assert key in test_mapping
            assert self.is_equal(test_mapping[key], new_mapping[key])
        for key in test_mapping:
            if key not in new_mapping:
                assert self.is_equal(test_mapping[key], plain_mapping[key])
        self.final_check(test_mapping)

    @pytest.fixture
    def setdefault_key_value(self):
        """Determines the indices used for test_setdefault.

        Subclasses are required to implement this.

        Note:
            A non-existent key should be included for proper testing.
        """
        raise NotImplementedError

    def test_setdefault(self, setdefault_key_value, populated_collection):
        """Test update."""
        test_mapping, plain_mapping = populated_collection
        key, value = setdefault_key_value
        if not self._allow_new_keys and key not in test_mapping:
            with pytest.raises(KeyError):
                test_mapping.setdefault(key, value)
            return

        test_mapping.setdefault(key, value)
        if key in plain_mapping:
            assert self.is_equal(test_mapping[key], plain_mapping[key])
            assert len(test_mapping) == len(plain_mapping)
        else:
            assert self.is_equal(test_mapping[key], value)
            assert len(test_mapping) == len(plain_mapping) + 1
        self.final_check(test_mapping)

    def test_popitem(self, populated_collection):
        """Test popitem."""
        test_mapping, plain_mapping = populated_collection
        if self._deletion_error is not None:
            with pytest.raises(self._deletion_error):
                test_mapping.popitem()
            return

        for length in range(len(test_mapping) - 1, -1, -1):
            key, item = test_mapping.popitem()
            assert key not in test_mapping
            assert len(test_mapping) == length
        self.final_check(test_mapping)

    def test_get(self, populated_collection):
        """Test get."""
        test_mapping, plain_mapping = populated_collection
        for key, value in plain_mapping.items():
            assert self.is_equal(test_mapping.get(key), value)
        for key in self.random_keys():
            if key not in test_mapping:
                assert test_mapping.get(key) is None
                assert test_mapping.get(key, 1) == 1
                break
