from sys import stdout
from math import log10
from hoomd.util import dict_flatten
from hoomd.custom_action import _InternalCustomAction
from hoomd.analyze.custom_analyzer import _InternalCustomAnalyzer
from hoomd.operation import _Analyzer


class Formatter:
    def __init__(self, pretty=True,
                 max_precision=15, max_decimals_pretty=5,
                 pad=" ", align="^"):
        self._generate_fmt_strings(pad, align)
        self.pretty = pretty
        self.precision = max_precision - 1
        self.max_decimals_pretty = max_decimals_pretty

    def _generate_fmt_strings(self, pad, align):
        base = "{:" + pad + align
        self._num_format = base + "{width}{type}}"
        self._str_format = base + "{width}}"

    def __call__(self, value, column_width):
        if isinstance(value, str):
            return self.format_str(value, column_width)
        else:
            return self.format_num(value, column_width)

    def format_num(self, value, column_width):
        digit_guess = int(log10(abs(value))) + 1
        if value < 0:
            digit_guess += 1
        # Use scientific formatting
        if not (-5 < digit_guess < 6) or digit_guess > column_width:
            # Determine the number of decimals to use
            if self.pretty:
                decimals = min(max(column_width - 6, 1),
                               self.max_decimals_pretty)
            else:
                decimals = max(self.precision, 0)
            type_fmt = "." + str(decimals) + "e"
            return self._num_format.format(value,
                                           width=column_width,
                                           type=type_fmt)
        # Use regular formatting
        else:
            if isinstance(value, int):
                return self._num_format.format(value,
                                               width=column_width,
                                               type="d")
            else:
                # Determine the number of decimals to use
                if self.pretty:
                    decimals = min(max(column_width - digit_guess - 2, 1),
                                   self.max_decimals_pretty)
                else:
                    decimals = max(self.precision - digit_guess + 1, 0)
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


class _CSVInternal(_InternalCustomAction):
    """Implements the logic for a simple text based logger backend."""

    def __init__(self, logger, output=stdout, sep='.',
                 pretty=True, max_precision=10, max_len_namespace=None):
        # Only accept loggers with scalar and string quantities
        if set(logger.flags).difference(['scalar', 'string']) != set():
            raise ValueError("Given Logger must have the scalar flag set.")
        else:
            self._logger = logger
            self.sep = sep
            # Ensure that columns are always at least ten characters
            self._min_width = max(10, 10 if pretty else max_precision + 6)
            # Used to truncate namespaces
            self._max_len_namespace = max_len_namespace
            # Records the current keys and their lengths
            self._cur_headers = dict()
            self._fmt = Formatter(pretty, max_precision)
            # the output file or stdout
            self._output = output

    def attach(self, simulation):
        pass

    def detach(self):
        pass

    def log(self):
        # Get logger output in {"module.class": value} form
        if self._max_len_namespace is None:
            output_dict = {
                self._sep.join(key): value[0]
                for key, value in dict_flatten(self._logger.log()).items()
                if value[1] in {'string', 'scalar'}
            }
        else:
            output_dict = {
                self._sep.join(key[-self._max_len_namespace:]): value[0]
                for key, value in dict_flatten(self._logger.log()).items()
                if value[1] in {'string', 'scalar'}
            }
        new_keys = output_dict.keys()
        if new_keys != self._cur_headers.keys():
            self._cur_headers = {key: max(len(key), self._min_width)
                                 for key in new_keys}
            self._write_header()

        self._write_row(output_dict)

    def _write_header(self):
        headers = self._cur_headers
        self._output.write(
            ' '.join((self._fmt.format_str(k, headers[k]) for k in headers))
            )
        self._output.write('\n')

    def _write_row(self, data):
        headers = self._cur_headers
        self._output.write(' '.join((self._fmt(data[k], headers[k])
                                     for k in headers))
                           )
        self._output.write('\n')

    def act(self, timestep):
        self.log()


class CSV(_InternalPythonAnalyzer, _Analyzer):
    _internal_class = _CSVInternal
