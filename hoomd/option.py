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
import sys;
import shlex;
import os;

## \internal
# \brief Storage for all option values
#
# options stores values in a similar way to the output of optparse for compatibility with existing
# code.
class options:
    def __init__(self):
        self.mode = None;
        self.gpu = None;
        self.gpu_error_checking = None;
        self.min_cpu = None;
        self.ignore_display = None;
        self.user = [];
        self.notice_level = 2;
        self.msg_file = None;
        self.shared_msg_file = None;
        self.nrank = None;
        self.nx = None;
        self.ny = None;
        self.nz = None;
        self.linear = None;
        self.onelevel = None;
        self.autotuner_enable = True;
        self.autotuner_period = 100000;
        self.single_mpi = False;
        self.nthreads = None;

    def __repr__(self):
        tmp = dict(mode=self.mode,
                   gpu=self.gpu,
                   gpu_error_checking=self.gpu_error_checking,
                   min_cpu=self.min_cpu,
                   ignore_display=self.ignore_display,
                   user=self.user,
                   notice_level=self.notice_level,
                   msg_file=self.msg_file,
                   shared_msg_file=self.shared_msg_file,
                   nrank=self.nrank,
                   nx=self.nx,
                   ny=self.ny,
                   nz=self.nz,
                   linear=self.linear,
                   onelevel=self.onelevel,
                   single_mpi=self.single_mpi,
                   nthreads=self.nthreads)
        return str(tmp);

## Parses command line options
#
# \internal
# Parses all hoomd command line options into the module variable cmd_options
def _parse_command_line(arg_string=None):
    parser = OptionParser();
    parser.add_option("--mode", dest="mode", help="Execution mode (cpu or gpu)", default='auto');
    parser.add_option("--gpu", dest="gpu", help="GPU or comma-separated list of GPUs on which to execute");
    parser.add_option("--gpu_error_checking", dest="gpu_error_checking", action="store_true", default=False, help="Enable error checking on the GPU");
    parser.add_option("--minimize-cpu-usage", dest="min_cpu", action="store_true", default=False, help="Enable to keep the CPU usage of HOOMD to a bare minimum (will degrade overall performance somewhat)");
    parser.add_option("--ignore-display-gpu", dest="ignore_display", action="store_true", default=False, help="Attempt to avoid running on the display GPU");
    parser.add_option("--notice-level", dest="notice_level", help="Minimum level of notice messages to print");
    parser.add_option("--msg-file", dest="msg_file", help="Name of file to write messages to");
    parser.add_option("--shared-msg-file", dest="shared_msg_file", help="(MPI only) Name of shared file to write message to (append partition #)");
    parser.add_option("--nrank", dest="nrank", help="(MPI) Number of ranks to include in a partition");
    parser.add_option("--nx", dest="nx", help="(MPI) Number of domains along the x-direction");
    parser.add_option("--ny", dest="ny", help="(MPI) Number of domains along the y-direction");
    parser.add_option("--nz", dest="nz", help="(MPI) Number of domains along the z-direction");
    parser.add_option("--linear", dest="linear", action="store_true", default=False, help="(MPI only) Force a slab (1D) decomposition along the z-direction");
    parser.add_option("--onelevel", dest="onelevel", action="store_true", default=False, help="(MPI only) Disable two-level (node-local) decomposition");
    parser.add_option("--single-mpi", dest="single_mpi", action="store_true", help="Allow single-threaded HOOMD builds in MPI jobs");
    parser.add_option("--user", dest="user", help="User options");
    parser.add_option("--nthreads", dest="nthreads", help="Number of TBB threads");

    input_args = None;
    if arg_string is not None:
        input_args = shlex.split(arg_string);

    (cmd_options, args) = parser.parse_args(args=input_args);

    # check for valid mode setting
    if cmd_options.mode is not None:
        if not (cmd_options.mode == "cpu" or cmd_options.mode == "gpu" or cmd_options.mode == "auto"):
            parser.error("--mode must be either cpu, gpu, or auto");

    # check for sane options
    if cmd_options.mode == "cpu" and (cmd_options.gpu is not None):
        parser.error("--mode=cpu cannot be specified along with --gpu")

    # set the mode to gpu if the gpu # was set
    if cmd_options.gpu is not None and cmd_options.mode == 'auto':
        cmd_options.mode = "gpu"

    # convert gpu to an integer
    if cmd_options.gpu is not None:
        try:
            cmd_options.gpu = [int(gpu) for gpu in str(cmd_options.gpu).split(',')]
        except ValueError:
            parser.error('--gpu must be an integer or comma-separated list of integers')

    # convert notice_level to an integer
    if cmd_options.notice_level is not None:
        try:
            cmd_options.notice_level = int(cmd_options.notice_level);
        except ValueError:
            parser.error('--notice-level must be an integer')

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

    # Convert nthreads to an integer
    if cmd_options.nthreads is not None:
       if not _hoomd.is_TBB_available():
            parser.error("The --nthreads option is only available in TBB-enabled builds.\n");
            raise RuntimeError('Error setting option');
       try:
            cmd_options.nthreads = int(cmd_options.nthreads);
       except ValueError:
            parser.error('--nthreads must be an integer')


    # copy command line options over to global options
    hoomd.context.options.mode = cmd_options.mode;
    hoomd.context.options.gpu = cmd_options.gpu;
    hoomd.context.options.gpu_error_checking = cmd_options.gpu_error_checking;
    hoomd.context.options.min_cpu = cmd_options.min_cpu;
    hoomd.context.options.ignore_display = cmd_options.ignore_display;

    hoomd.context.options.nx = cmd_options.nx;
    hoomd.context.options.ny = cmd_options.ny;
    hoomd.context.options.nz = cmd_options.nz;
    hoomd.context.options.linear = cmd_options.linear
    hoomd.context.options.onelevel = cmd_options.onelevel
    hoomd.context.options.single_mpi = cmd_options.single_mpi
    hoomd.context.options.nthreads = cmd_options.nthreads

    hoomd.context.options.notice_level = cmd_options.notice_level;
    hoomd.context.options.msg_file = cmd_options.msg_file;

    if cmd_options.shared_msg_file is not None:
        if not _hoomd.is_MPI_available():
            parser.error("Shared log files are only available in MPI builds.\n");
            raise RuntimeError('Error setting option');
        hoomd.context.options.shared_msg_file = cmd_options.shared_msg_file;

    if cmd_options.nrank is not None:
        if not _hoomd.is_MPI_available():
            hoomd.context.msg.error("The --nrank option is only available in MPI builds.\n");
            raise RuntimeError('Error setting option');
        hoomd.context.options.nrank = int(cmd_options.nrank)

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
        hoomd.context.msg.error("notice-level must be an integer\n");
        raise RuntimeError('Error setting option');

    hoomd.context.msg.setNoticeLevel(notice_level);
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
        hoomd.context.msg.openFile(fname);
    else:
        hoomd.context.msg.openStd();

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
        msg.warning("HOOMD was compiled without thread support, ignoring request to set number of threads.\n");
    else:
        hoomd.context.exec_conf.setNumThreads(int(num_threads));


## \internal
# \brief Throw an error if the context is not initialized
def _verify_init():
    if hoomd.context.options is None:
        hoomd.context.msg.error("call context.initialize() before any other method in _hoomd.")
        raise RuntimeError("hoomd execution context is not available")
