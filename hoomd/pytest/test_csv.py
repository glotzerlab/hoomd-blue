from io import StringIO
from math import isclose
import pytest

import hoomd
import hoomd.output

try:
    from mpi4py import MPI
    skip_mpi = False
except ImportError:
    skip_mpi = True

skip_mpi = pytest.mark.skipif(skip_mpi, reason="MPI4py is not importable.")


@pytest.fixture
def logger():
    logger = hoomd.logging.Logger(flags=['scalar'])
    logger[('dummy', 'loggable', 'int')] = (lambda: 42000000, 'scalar')
    logger[('dummy', 'loggable', 'float')] = (lambda: 3.1415, 'scalar')
    logger[('dummy', 'loggable', 'string')] = (lambda: "foobarbaz", 'string')
    return logger


@pytest.fixture
def expected_values():
    return {'dummy.loggable.int': 42000000,
            'dummy.loggable.float': 3.1415,
            'dummy.loggable.string': "foobarbaz"}


@pytest.mark.serial
def test_header_generation(device, logger):
    output = StringIO("")
    csv_writer = hoomd.output.CSV(0, logger, output)
    csv_writer._comm = device.comm
    for i in range(10):
        csv_writer.act()
    output_str = output.getvalue()
    lines = output_str.split('\n')
    headers = lines[0].split()
    expected_headers = [
        'dummy.loggable.int', 'dummy.loggable.float', 'dummy.loggable.string']
    assert all(hdr in headers for hdr in expected_headers)
    for i in range(1, 10):
        values = lines[i].split()
        assert not any(v in expected_headers for v in values)
    csv_writer._logger[('new', 'quantity')] = (lambda: 53, 'scalar')
    csv_writer.act()
    output_str = output.getvalue()
    lines = output_str.split('\n')
    headers = lines[-3].split()
    expected_headers.append('new.quantity')
    assert all(hdr in headers for hdr in expected_headers)


@pytest.mark.serial
def test_values(device, logger, expected_values):
    output = StringIO("")
    csv_writer = hoomd.output.CSV(0, logger, output)
    csv_writer._comm = device.comm
    for i in range(10):
        csv_writer.act()
    lines = output.getvalue().split('\n')
    headers = lines[0].split()

    def test_equality(expected, given):
        """Used to test the accuracy and ordering of CSV rows."""
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
        assert all(test_equality(expected_values[hdr], v)
                   for hdr, v in zip(headers, values))


@skip_mpi
def test_mpi_write_only(device, logger):
    output = StringIO("")
    csv_writer = hoomd.output.CSV(0, logger, output)
    csv_writer._comm = device.comm
    csv_writer.act()

    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        assert output.getvalue() != ''
    else:
        assert output.getvalue() == ''


@pytest.mark.serial
def test_header_attributes(device, logger):
    output = StringIO("")
    csv_writer = hoomd.output.CSV(
        0, logger, output, header_sep='-', max_header_len=13)
    csv_writer._comm = device.comm
    csv_writer.act()
    lines = output.getvalue().split('\n')
    headers = lines[0].split()
    expected_headers = ['loggable-int', 'loggable-float', 'string']
    assert all(hdr in headers for hdr in expected_headers)


@pytest.mark.serial
def test_delimiter(device, logger):
    output = StringIO("")
    csv_writer = hoomd.output.CSV(
        0, logger, output, delimiter=',')
    csv_writer._comm = device.comm
    csv_writer.act()
    lines = output.getvalue().split('\n')
    assert all(len(row.split(',')) == 3 for row in lines[:-1])


@pytest.mark.serial
def test_max_precision(device, logger):
    output = StringIO("")
    csv_writer = hoomd.output.CSV(0, logger, output, pretty=False,
                                  max_precision=5)
    csv_writer._comm = device.comm
    for i in range(10):
        csv_writer.act()

    smaller_lines = output.getvalue().split('\n')

    output = StringIO("")
    csv_writer = hoomd.output.CSV(0, logger, output, pretty=False,
                                  max_precision=15)
    csv_writer._comm = device.comm
    for i in range(10):
        csv_writer.act()

    longer_lines = output.getvalue().split('\n')

    for long_row, short_row in zip(longer_lines[1:-1], smaller_lines[1:-1]):
        assert all(len(long_) >= len(short)
                   for long_, short in zip(long_row.split(), short_row.split()))

        assert any(len(long_) > len(short)
                   for long_, short in zip(long_row.split(), short_row.split()))


def test_only_string_and_scalar_quantities(device):
    logger = hoomd.logging.Logger()
    output = StringIO("")
    with pytest.raises(ValueError):
        _ = hoomd.output.CSV(0, logger, output)
    logger = hoomd.logging.Logger(flags=['sequence'])
    with pytest.raises(ValueError):
        _ = hoomd.output.CSV(0, logger, output)
