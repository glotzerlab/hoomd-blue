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
        self.nx = None;
        self.ny = None;
        self.nz = None;
        self.linear = None;
        self.onelevel = None;
        self.autotuner_enable = True;
        self.autotuner_period = 100000;
        self.single_mpi = False;

    def __repr__(self):
        tmp = dict(user=self.user,
                   nx=self.nx,
                   ny=self.ny,
                   nz=self.nz,
                   linear=self.linear,
                   onelevel=self.onelevel,
                   single_mpi=self.single_mpi)
        return str(tmp);

## Parses command line options
#
# \internal
# Parses all hoomd command line options into the module variable cmd_options
def _parse_command_line(arg_string=None):
    parser = OptionParser();
    parser.add_option("--nx", dest="nx", help="(MPI) Number of domains along the x-direction");
    parser.add_option("--ny", dest="ny", help="(MPI) Number of domains along the y-direction");
    parser.add_option("--nz", dest="nz", help="(MPI) Number of domains along the z-direction");
    parser.add_option("--linear", dest="linear", action="store_true", default=False, help="(MPI only) Force a slab (1D) decomposition along the z-direction");
    parser.add_option("--onelevel", dest="onelevel", action="store_true", default=False, help="(MPI only) Disable two-level (node-local) decomposition");
    parser.add_option("--single-mpi", dest="single_mpi", action="store_true", help="Allow single-threaded HOOMD builds in MPI jobs");
    parser.add_option("--user", dest="user", help="User options");

    input_args = None;
    if arg_string is not None:
        input_args = shlex.split(arg_string);

    (cmd_options, args) = parser.parse_args(args=input_args);

    # Convert nx to an integer
    if cmd_options.nx is not None:
        if not _hoomd.is_MPI_available():
            parser.error("The --nx option is only available in MPI builds.\n");
            raise RuntimeError('Error setting option');
        try:
            cmd_options.nx = int(cmd_options.nx);
        except ValueError:
            parser.error('--nx must be an integer')

    # Convert ny to an integer
    if cmd_options.ny is not None:
        if not _hoomd.is_MPI_available():
            parser.error("The --ny option is only available in MPI builds.\n");
            raise RuntimeError('Error setting option');
        try:
            cmd_options.ny = int(cmd_options.ny);
        except ValueError:
            parser.error('--ny must be an integer')

    # Convert nz to an integer
    if cmd_options.nz is not None:
       if not _hoomd.is_MPI_available():
            parser.error("The --nz option is only available in MPI builds.\n");
            raise RuntimeError('Error setting option');
       try:
            cmd_options.nz = int(cmd_options.nz);
       except ValueError:
            parser.error('--nz must be an integer')

    # copy command line options over to global options
    hoomd.context.options.nx = cmd_options.nx;
    hoomd.context.options.ny = cmd_options.ny;
    hoomd.context.options.nz = cmd_options.nz;
    hoomd.context.options.linear = cmd_options.linear
    hoomd.context.options.onelevel = cmd_options.onelevel
    hoomd.context.options.single_mpi = cmd_options.single_mpi

    if cmd_options.user is not None:
        hoomd.context.options.user = shlex.split(cmd_options.user);

def get_user():
    R""" Get user options.

    Return:
        List of user options passed in via --user="arg1 arg2 ..."
    """
    _verify_init();
    return hoomd.context.options.user;

def set_notice_level(notice_level):
    R""" Set the notice level.

    Args:
        notice_level (int). The maximum notice level to print.

    The notice level may be changed before or after initialization, and may be changed
    many times during a job script.

    Note:
        Overrides ``--notice-level`` on the command line.

    """
    _verify_init();

    try:
        notice_level = int(notice_level);
    except ValueError:
        hoomd.context.current.device.cpp_msg.error("notice-level must be an integer\n");
        raise RuntimeError('Error setting option');

    hoomd.context.current.device.cpp_msg.setNoticeLevel(notice_level);
    hoomd.context.options.notice_level = notice_level;

def set_msg_file(fname):
    R""" Set the message file.

    Args:
        fname (str): Specifies the name of the file to write. The file will be overwritten.
                     Set to None to direct messages back to stdout/stderr.

    The message file may be changed before or after initialization, and may be changed many times during a job script.
    Changing the message file will only affect messages sent after the change.

    Note:
        Overrides ``--msg-file`` on the command line.

    """
    _verify_init();

    if fname is not None:
        hoomd.context.current.device.cpp_msg.openFile(fname);
    else:
        hoomd.context.current.device.cpp_msg.openStd();

    hoomd.context.options.msg_file = fname;

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

def set_num_threads(num_threads):
    R""" Set the number of CPU (TBB) threads HOOMD uses

    Args:
        num_threads (int): The number of threads

    Note:
        Overrides ``--nthreads`` on the command line.

    """

    if not _hoomd.is_TBB_available():
        hoomd.context.current.device.cpp_msg.warning("HOOMD was compiled without thread support, ignoring request to set number of threads.\n");
    else:
        hoomd.context.current.device.cpp_exec_conf.setNumThreads(int(num_threads));


## \internal
# \brief Throw an error if the context is not initialized
def _verify_init():
    if hoomd.context.options is None:
        hoomd.context.current.device.cpp_msg.error("call context.initialize() before any other method in _hoomd.")
        raise RuntimeError("hoomd execution context is not available")
