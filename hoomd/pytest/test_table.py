# Copyright (c) 2009-2023 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

from io import StringIO
from math import isclose
import pytest

from hoomd.conftest import operation_pickling_check
import hoomd
import hoomd.write
import sys

try:
    from mpi4py import MPI
    skip_mpi = False
except ImportError:
    skip_mpi = True

skip_mpi = pytest.mark.skipif(skip_mpi, reason="MPI4py is not importable.")


class Identity:

    def __init__(self, x):
        self.x = x

    def __call__(self):
        return self.x

    def __eq__(self, other):
        return self.x == other.x


@pytest.fixture
def logger():
    logger = hoomd.logging.Logger(categories=['scalar', "string"])
    logger[('dummy', 'loggable', 'int')] = (Identity(42000000), 'scalar')
    logger[('dummy', 'loggable', 'float')] = (Identity(3.1415), 'scalar')
    logger[('dummy', 'loggable', 'string')] = (Identity("foobarbaz"), 'string')
    return logger


@pytest.fixture
def expected_values():
    return {
        'dummy.loggable.int': 42000000,
        'dummy.loggable.float': 3.1415,
        'dummy.loggable.string': "foobarbaz"
    }


# TODO: Clean this up


def get_output_from_table_writer(writer_function, output):
    writer_function()
    return output.getvalue()


def write_table_lines(table_writer):

    def _write_table_lines():
        for i in range(10):
            table_writer.write()

    return _write_table_lines


def get_stdout(writer_function, output):
    stdout = sys.stdout
    output_IO = StringIO()
    sys.stdout = output_IO
    writer_function()
    sys.stdout = stdout
    return output_IO.getvalue()


def get_table_writer_string_io(device, logger, table_function=None):

    if table_function is None:

        def table_function(output):
            return hoomd.write.Table(1, logger, output)

    output = StringIO("")
    table_writer = table_function(output)
    table_writer._comm = device.communicator
    return table_writer, output


def get_table_writer_output(device, logger, table_function=None):
    if table_function is None:

        def table_function(output):
            return hoomd.write.Table(1, logger, output)

    output = "notice"
    table_writer = table_function(output)
    table_writer._comm = device.communicator
    table_writer._notice = device.notice
    return table_writer, output


table_writer_functions = [
    (get_table_writer_string_io, lambda table_writer_function, output:
     get_output_from_table_writer(table_writer_function, output)),
    (get_table_writer_output, lambda table_writer_function, output: get_stdout(
        table_writer_function, output))
]

# Clean this up


def test_invalid_attrs(logger):
    output = StringIO("")
    table_writer = hoomd.write.Table(1, logger, output)
    with pytest.raises(AttributeError):
        table_writer.action
    with pytest.raises(AttributeError):
        table_writer.detach
    with pytest.raises(AttributeError):
        table_writer.attach


@pytest.mark.parametrize("get_table_writer,get_output_str",
                         table_writer_functions)
@pytest.mark.serial
def test_header_generation(device, logger, get_table_writer, get_output_str):

    table_writer, output = get_table_writer(device, logger)
    output_str = get_output_str(write_table_lines(table_writer), output)
    lines = output_str.split('\n')
    headers = lines[0].split()
    expected_headers = [
        'dummy.loggable.int', 'dummy.loggable.float', 'dummy.loggable.string'
    ]
    assert all(hdr in headers for hdr in expected_headers)
    for i in range(1, 10):
        values = lines[i].split()
        assert not any(v in expected_headers for v in values)
    table_writer.logger[('new', 'quantity')] = (lambda: 53, 'scalar')
    output_str = get_output_str(table_writer.write, output)
    lines = output_str.split('\n')
    headers = lines[-3].split()
    expected_headers.append('new.quantity')
    assert all(hdr in headers for hdr in expected_headers)


@pytest.mark.parametrize("get_table_writer,get_output_str",
                         table_writer_functions)
@pytest.mark.serial
def test_values(device, logger, get_table_writer, get_output_str,
                expected_values):

    table_writer, output = get_table_writer(device, logger)
    output_str = get_output_str(write_table_lines(table_writer), output)
    lines = output_str.split('\n')
    headers = lines[0].split()

    def test_equality(expected, given):
        """Used to test the accuracy and ordering of Table rows."""
        type_ = expected.__class__
        try:
            if issubclass(type_, (int, float)):
                # the float conversion is necessary for scientific notation
                # conversion to int
                return isclose(expected, type_(float(given)))
            else:
                return expected == type_(given)
        except Exception:
            return False

    for row in lines[1:]:
        values = row.split()
        assert all(
            test_equality(expected_values[hdr], v)
            for hdr, v in zip(headers, values))


@skip_mpi
@pytest.mark.parametrize("get_table_writer,get_output_str",
                         table_writer_functions)
def test_mpi_write_only(device, logger, get_output_str, get_table_writer):

    table_writer, output = get_table_writer(device, logger)
    output_str = get_output_str(table_writer.write, output)
    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        assert output_str != ''
    else:
        assert output_str == ''


@pytest.mark.serial
@pytest.mark.parametrize("get_table_writer,get_output_str",
                         table_writer_functions)
def test_header_attributes(device, logger, get_output_str, get_table_writer):

    def table_function(output):
        return hoomd.write.Table(1,
                                 logger,
                                 output,
                                 header_sep='-',
                                 max_header_len=13)

    table_writer, output = get_table_writer(device, logger, table_function)
    output_str = get_output_str(table_writer.write, output)
    lines = output_str.split('\n')
    headers = lines[0].split()
    expected_headers = ['loggable-int', 'loggable-float', 'string']
    assert all(hdr in headers for hdr in expected_headers)


@pytest.mark.serial
@pytest.mark.parametrize("get_table_writer,get_output_str",
                         table_writer_functions)
def test_delimiter(device, logger, get_output_str, get_table_writer):

    def table_function(output):
        return hoomd.write.Table(1, logger, output, delimiter=',')

    table_writer, output = get_table_writer(device, logger, table_function)
    output_str = get_output_str(table_writer.write, output)
    lines = output_str.split('\n')
    assert all(len(row.split(',')) == 3 for row in lines[:-1])


@pytest.mark.serial
@pytest.mark.parametrize("get_table_writer,get_output_str",
                         table_writer_functions)
def test_max_precision(device, logger, get_output_str, get_table_writer):

    def table_function(max_precision):

        def _table_function(output):
            return hoomd.write.Table(1,
                                     logger,
                                     output,
                                     pretty=False,
                                     max_precision=max_precision)

        return _table_function

    table_writer, output = get_table_writer(device, logger, table_function(5))
    output_str = get_output_str(write_table_lines(table_writer), output)
    smaller_lines = output_str.split('\n')

    table_writer, output = get_table_writer(device, logger, table_function(15))
    output_str = get_output_str(write_table_lines(table_writer), output)
    longer_lines = output_str.split('\n')

    for long_row, short_row in zip(longer_lines[1:-1], smaller_lines[1:-1]):
        assert all(
            len(long_) >= len(short)
            for long_, short in zip(long_row.split(), short_row.split()))

        assert any(
            len(long_) > len(short)
            for long_, short in zip(long_row.split(), short_row.split()))


@pytest.mark.parametrize("get_table_writer",
                         (get_table_writer_string_io, get_table_writer_output))
def test_only_string_and_scalar_quantities(device, get_table_writer):
    logger = hoomd.logging.Logger()
    with pytest.raises(ValueError):
        table_writer, output = get_table_writer(device, logger)
    logger = hoomd.logging.Logger(categories=['sequence'])
    with pytest.raises(ValueError):
        table_writer, output = get_table_writer(device, logger)


def test_pickling(simulation_factory, two_particle_snapshot_factory, logger):
    sim = simulation_factory(two_particle_snapshot_factory())
    table = hoomd.write.Table(1, logger)
    operation_pickling_check(table, sim)
