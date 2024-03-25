# Copyright (c) 2009-2024 The Regents of the University of Michigan.
# Part of HOOMD-blue, released under the BSD 3-Clause License.

"""Implement Table.

.. invisible-code-block: python

    simulation = hoomd.util.make_example_simulation()
"""

from abc import ABCMeta, abstractmethod
import copy
from numbers import Integral
from math import log10
from sys import stdout

from hoomd.write.custom_writer import _InternalCustomWriter
from hoomd.custom.custom_action import _InternalAction
from hoomd.logging import LoggerCategories, Logger
from hoomd.data.parameterdicts import ParameterDict
from hoomd.data.typeconverter import OnlyTypes
from hoomd.util import _dict_flatten
from hoomd.custom import Action


class _OutputWriter(metaclass=ABCMeta):
    """Represents the necessary functions for writing out data.

    We use this to ensure the output object passed to Table will support the
    necessary functions.
    """

    @abstractmethod
    def flush(self):
        pass

    @abstractmethod
    def write(self):
        pass

    @abstractmethod
    def writable(self):
        pass

    @classmethod
    def __subclasshook__(cls, C):
        if cls is _OutputWriter:
            return all(hasattr(C, method) for method in cls.__abstractmethods__)
        else:
            return NotImplemented


def _ensure_writable(fh):
    if not fh.writable():
        raise ValueError("file-like object must be writable.")
    return fh


class _Formatter:
    """Internal class for number and string formatting for Table object.

    Main method is ``__call__``. It takes a value with the corresponding column
    width and outputs the string to use for that column. Some of these
    parameters are not currently used in the _InternalTable class, but are
    available in the _Formatter class, meaning that adding these features later
    would be fairly simple. I (Brandon Butler) did not think they were worth
    complicating the Table Logger any more than it currently is though, so they
    are not used now.

    Args:
        pretty (bool): whether to attempt to make output pretty (more readable).
        max_precision (int): The max length for formatting a number or string.
        max_decimals_pretty (int): The maximum number of decimals. This is
            required to ensure that the decimals don't eat up all space in a
            pretty print.
        pad (str, optional): What to pad extra column space with, defaults to
            space.
        align (str, optional): What type of alignment to use, defaults to
            centered ('^').
    """

    def __init__(self,
                 pretty=True,
                 max_precision=15,
                 max_decimals_pretty=5,
                 pad=" ",
                 align="^"):
        self.generate_fmt_strings(pad, align)
        self.pretty = pretty
        self.precision = max_precision - 1
        self.max_decimals_pretty = max_decimals_pretty

    def generate_fmt_strings(self, pad, align):
        base = "{:" + pad + align
        self._num_format = base + "{width}{type}}"
        self._str_format = base + "{width}}"

    def __call__(self, value, column_width):
        if isinstance(value, str):
            return self.format_str(value, column_width)
        else:
            return self.format_num(value, column_width)

    @staticmethod
    def _digits_from_decimal(num):
        """Return digits to represent to the first significant digit."""
        if num == 0.0:
            return 1

        digits = int(log10(abs(num)))
        # Positive exponents require an extra space (10^0 == 1)
        digits = 1 + digits if digits >= 0 else -digits
        if num < 0:
            digits += 1  # - (negative symbol)
        return digits + 1  # decimal point

    def format_num(self, value, column_width):
        # Always output full integer values
        if isinstance(value, Integral):
            return self._num_format.format(value, width=column_width, type="d")
        # For floating point numbers
        else:
            # The minimum length representation if greater than one than the
            # smallest representation is to write the number without any
            # information past the decimal point. For values less than 1 the
            # smallest is 0.xxx. The plus one is for the decimal point. We
            # already attempt to print out as many decimal points as possible so
            # we only need to determine the minimum size to from the decimal
            # point including the decimal point.
            min_len_repr = self._digits_from_decimal(value) + 1
            # Use scientific formatting
            if not min_len_repr < 6 or min_len_repr > column_width:
                # Determine the number of decimals to use
                if self.pretty:
                    decimals = min(max(column_width - 6, 1),
                                   self.max_decimals_pretty)
                else:
                    decimals = max(self.precision, 0)
                type_fmt = "." + str(decimals) + "e"
            # Use regular formatting
            else:
                # Determine the number of decimals to use
                if self.pretty:
                    decimals = min(max(column_width - min_len_repr - 2, 1),
                                   self.max_decimals_pretty)
                else:
                    decimals = max(self.precision - min_len_repr + 1, 0)
                type_fmt = "." + str(decimals) + "f"

            return self._num_format.format(value,
                                           width=column_width,
                                           type=type_fmt)

    def format_str(self, value, column_width):
        if self.pretty and len(value) > column_width:
            truncate_to = max(1, column_width - 2)
            return self._str_format.format(value[-truncate_to:],
                                           width=column_width)
        else:
            return self._str_format.format(value, width=column_width)

    def __eq__(self, other):
        if not isinstance(other, _Formatter):
            return NotImplemented
        return (self.pretty == other.pretty
                and self.precision == other.precision
                and self.max_decimals_pretty == other.max_decimals_pretty
                and self._num_format == other._num_format
                and self._str_format == other._str_format)


class _TableInternal(_InternalAction):
    """Implements the logic for a simple text based logger backend.

    This currently has to check the logged quantities every time to ensure it
    has not changed since the last run of `~.act`. Performance could be
    improved by allowing for writing of data without checking for a change in
    logged quantities, but would be more fragile.
    """

    _invalid_logger_categories = LoggerCategories.any([
        'sequence', 'object', 'particle', 'bond', 'angle', 'dihedral',
        'improper', 'pair', 'constraint', 'strings'
    ])

    flags = [
        Action.Flags.ROTATIONAL_KINETIC_ENERGY, Action.Flags.PRESSURE_TENSOR,
        Action.Flags.EXTERNAL_FIELD_VIRIAL
    ]

    _skip_for_equality = {"_comm"}

    def __init__(self,
                 logger,
                 output=stdout,
                 header_sep='.',
                 delimiter=' ',
                 pretty=True,
                 max_precision=10,
                 max_header_len=None):

        param_dict = ParameterDict(header_sep=str,
                                   delimiter=str,
                                   min_column_width=int,
                                   max_header_len=OnlyTypes(int,
                                                            allow_none=True),
                                   pretty=bool,
                                   max_precision=int,
                                   output=OnlyTypes(
                                       _OutputWriter,
                                       postprocess=_ensure_writable),
                                   logger=Logger)

        param_dict.update(
            dict(header_sep=header_sep,
                 delimiter=delimiter,
                 min_column_width=max(10, max_precision + 6),
                 max_header_len=max_header_len,
                 max_precision=max_precision,
                 pretty=pretty,
                 output=output,
                 logger=logger))
        self._param_dict = param_dict
        # internal variables that are not part of the state.

        # Generate LoggerCategories for valid and invalid categories
        _valid_categories = LoggerCategories.any(
            [LoggerCategories.scalar, LoggerCategories.string])
        _invalid_inputs = logger.categories & self._invalid_logger_categories

        # Ensure that only scalar and string categories are set for the logger
        if logger.categories == LoggerCategories.NONE:
            pass
        elif (_valid_categories ^ LoggerCategories.ALL
              ) & logger.categories == LoggerCategories.NONE:
            pass
        else:
            raise ValueError(
                "Table Logger may only have scalar or string categories set. \
                    Use hoomd.write.GSD for {}.".format(_invalid_inputs))

        self._cur_headers_with_width = dict()
        self._fmt = _Formatter(pretty, max_precision)
        self._comm = None

    def _setattr_param(self, attr, value):
        """Makes self._param_dict attributes read only."""
        raise ValueError("Attribute {} is read-only.".format(attr))

    def attach(self, simulation):
        self._comm = simulation.device._comm

    def detach(self):
        self._comm = None

    def _get_log_dict(self):
        """Get a flattened dict for writing to output."""
        return {
            key: value[0]
            for key, value in _dict_flatten(self.logger.log()).items()
        }

    def _update_headers(self, new_keys):
        """Update headers and write the current headers to output.

        This function could be made simpler and faster by moving some of the
        transformation to act. Since we don't expect the headers to change often
        however, this would likely slow the writer down. The design is to
        off-load any potentially unnecessary calculations to this function even
        if that means more overall computation when headers change.
        """
        header_output_list = []
        header_dict = {}
        for namespace in new_keys:
            header = self._determine_header(namespace, self.header_sep,
                                            self.max_header_len)
            column_size = max(len(header), self.min_column_width)
            header_dict[namespace] = column_size
            header_output_list.append((header, column_size))
        self._cur_headers_with_width = header_dict
        self.output.write(
            self.delimiter.join((self._fmt.format_str(hdr, width)
                                 for hdr, width in header_output_list)))
        self.output.write('\n')

    @staticmethod
    def _determine_header(namespace, sep, max_len):
        if max_len is None:
            return sep.join(namespace)
        else:
            index = -1
            char_count = len(namespace[-1])
            for name in reversed(namespace[:-1]):
                char_count += len(name)
                if char_count > max_len:
                    break
                index -= 1
            return sep.join(namespace[index:])

    def _write_row(self, data):
        """Write a row of data to output."""
        headers = self._cur_headers_with_width
        self.output.write(
            self.delimiter.join(
                (self._fmt(data[k], headers[k]) for k in headers)))
        self.output.write('\n')

    def act(self, timestep=None):
        """Write row to designated output.

        Will also write header when logged quantities are determined to have
        changed.
        """
        output_dict = self._get_log_dict()
        if self._comm is not None and self._comm.rank == 0:
            # determine if a header needs to be written. This is always the case
            # for the first call of act, and if the logged quantities change
            # within a run.
            new_keys = output_dict.keys()
            if new_keys != self._cur_headers_with_width.keys():
                self._update_headers(new_keys)

            # Write the data and flush. We must flush to ensure that the data
            # isn't merely stored in Python ready to be written later.
            self._write_row(output_dict)
            self.output.flush()

    def __getstate__(self):
        state = copy.copy(self.__dict__)
        state.pop('_comm', None)
        # This is to handle when the output specified is just stdout. By default
        # file objects like this are not picklable, so we need to handle it
        # differently. We let `None` represent stdout in the state dictionary.
        # Most other file like objects will simply fail to be pickled here.
        if self.output == stdout:
            param_dict = ParameterDict()
            param_dict.update(state['_param_dict'])
            state['_param_dict'] = param_dict
            state['_param_dict']._dict['output'] = None
            state['_param_dict']['output']
            return state
        else:
            return super().__getstate__()

    def __setstate__(self, state):
        if state['_param_dict']['output'] is None:
            del state['_param_dict']['output']
            state['_param_dict']['output'] = stdout
            state['_param_dict']._type_converter['output'] = OnlyTypes(
                _OutputWriter, postprocess=_ensure_writable),
        self.__dict__ = state


class Table(_InternalCustomWriter):
    """Write delimiter separated values to a stream.

    Args:
        trigger (hoomd.trigger.trigger_like): The trigger to determine when to
            run the Table backend.
        logger (hoomd.logging.Logger): The logger to query for output. `Table`
            supports only ``'scalar'`` and ``'string'`` logger categories.
        output (``file-like`` object , optional): A file-like object to output
            the data from, defaults to standard out. The object must have
            ``write`` and ``flush`` methods and a ``mode`` attribute. Examples
            include `sys.stdout`, `sys.stderr` and the return value of
            :py:func:`open`.
        header_sep (`str`, optional): String to use to separate names in
            the logger's namespace, defaults to ``'.'``. For example, if logging
            the total energy of an `hoomd.md.pair.LJ` pair force object, the
            default header would be ``md.pair.LJ.energy``.
        delimiter (`str`, optional): String used to separate elements in
            the space delimited file, defaults to ``' '``.
        pretty (`bool`, optional): Flags whether to attempt to make output
            prettier and easier to read (defaults to `True`).
        max_precision (`int`, optional): If pretty is not set, then this
            controls the maximum precision to use when outputing numerical
            values, defaults to 10.
        max_header_len (`int`, optional): If not `None` (the default), limit
            the outputted header names to length ``max_header_len``. When not
            `None`, names are grabbed from the most specific to the least. For
            example, if set to 7 the namespace 'hoomd.md.pair.LJ.energy' would
            be set to 'energy'.

            Note:
                At a minimum, the complete most specific part of the namespace
                will be used regardless of this setting.

    Use `Table` to write scalar and string `hoomd.logging.Logger` quantities to
    standard out or to a file.

    Warning:
        When logger quantities include strings with spaces, the default space
        delimiter will result in files that are not machine readable.

    .. rubric:: Example:

    .. code-block:: python

        logger = hoomd.logging.Logger(categories=['scalar', 'string'])
        table = hoomd.write.Table(trigger=hoomd.trigger.Periodic(10_000),
                                  logger=logger)

    Attributes:
        logger (hoomd.logging.Logger): The logger to query for output. `Table`
            supports only ``'scalar'`` and ``'string'`` logger categories
            (*read-only*).

            .. rubric:: Example:

            .. code-block:: python

                logger = table.logger

        output (``file-like`` object): A file-like object to output
            the data from. The object must have ``write`` and ``flush`` methods
            and a ``mode`` attribute (*read-only*).

            .. rubric:: Example:

            .. code-block:: python

                output = table.output

        header_sep (str): String to use to separate names in
            the logger's namespace (*read-only*).

            .. rubric:: Example:

            .. code-block:: python

                header_sep = table.header_sep

        delimiter (str): String used to separate elements in the space
            delimited file (*read-only*).

            .. rubric:: Example:

            .. code-block:: python

                delimiter = table.delimiter

        pretty (bool): Flags whether to attempt to make output
            prettier and easier to read (*read-only*).

            .. rubric:: Example:

            .. code-block:: python

                pretty = table.pretty

        max_precision (`int`, optional): If pretty is not set, then this
            controls the maximum precision to use when outputing numerical
            values (*read-only*).

            .. rubric:: Example:

            .. code-block:: python

                max_precision = table.max_precision

        max_header_len (int): Limits the outputted header names to length
            ``max_header_len`` when not ``None`` (*read-only*).

            .. rubric:: Example:

            .. code-block:: python

                max_header_len = table.max_header_len

        min_column_width (int): The minimum allowed column width (*read-only*).

            .. rubric:: Example:

            .. code-block:: python

                min_column_width = table.min_column_width
    """
    _internal_class = _TableInternal

    def write(self):
        """Write out data to ``self.output``.

        Writes a row from given ``hoomd.logging.Logger`` object data.

        .. rubric:: Example:

        .. code-block:: python

            table.write()
        """
        self._action.act()
