# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

# Maintainer: joaander

R""" Set global options.

Options may be set on the command line or from a job script using :py:func:`hoomd.context.initialize()`.
The option.set_* commands override any settings made previously.

"""

from optparse import OptionParser;

from hoomd import _hoomd
import hoomd;
import shlex;

## \internal
# \brief Storage for all option values
#
# options stores values in a similar way to the output of optparse for compatibility with existing
# code.
class options:
    def __init__(self):
        self.user = [];
        self.autotuner_enable = True;
        self.autotuner_period = 100000;

        # band aid fields
        self.mode = "auto"
        self.gpu_error_checking = False

    def __repr__(self):
        tmp = dict(user=self.user)
        return str(tmp);

## Parses command line options
#
# \internal
# Parses all hoomd command line options into the module variable cmd_options
def _parse_command_line(arg_string=None):
    parser = OptionParser();
    parser.add_option("--user", dest="user", help="User options");

    # these options are a temporary band-aid solution to make the unit tests work with the device objects
    # while reworking the api, we will also rework the unit tests, and then these options can be removed
    parser.add_option("--mode", dest="mode", default="auto")
    parser.add_option("--gpu_error_checking", dest="gpu_error_checking", action="store_true", default=False)

    input_args = None;
    if arg_string is not None:
        input_args = shlex.split(arg_string);

    (cmd_options, args) = parser.parse_args(args=input_args);

    if cmd_options.user is not None:
        hoomd.context.options.user = shlex.split(cmd_options.user);

    # copy band aid options to the global options variable
    hoomd.context.options.mode = cmd_options.mode
    hoomd.context.options.gpu_error_checking = cmd_options.gpu_error_checking

def get_user():
    R""" Get user options.

    Return:
        List of user options passed in via --user="arg1 arg2 ..."
    """
    _verify_init();
    return hoomd.context.options.user;


def set_autotuner_params(enable=True, period=100000):
    R""" Set autotuner parameters.

    Args:
        enable (bool). Set to True to enable autotuning. Set to False to disable.
        period (int): Approximate period in time steps between retuning.

    TODO: reference autotuner page here.

    """
    _verify_init();

    hoomd.context.options.autotuner_period = period;
    hoomd.context.options.autotuner_enable = enable;


## \internal
# \brief Throw an error if the context is not initialized
def _verify_init():
    if hoomd.context.current is None:
        raise RuntimeError("call context.initialize() before any other method in hoomd.")
