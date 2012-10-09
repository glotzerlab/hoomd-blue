# -- start license --
# Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
# (HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
# Iowa State University and The Regents of the University of Michigan All rights
# reserved.

# HOOMD-blue may contain modifications ("Contributions") provided, and to which
# copyright is held, by various Contributors who have granted The Regents of the
# University of Michigan the right to modify and/or distribute such Contributions.

# You may redistribute, use, and create derivate works of HOOMD-blue, in source
# and binary forms, provided you abide by the following conditions:

# * Redistributions of source code must retain the above copyright notice, this
# list of conditions, and the following disclaimer both in the code and
# prominently in any materials provided with the distribution.

# * Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions, and the following disclaimer in the documentation and/or
# other materials provided with the distribution.

# * All publications and presentations based on HOOMD-blue, including any reports
# or published results obtained, in whole or in part, with HOOMD-blue, will
# acknowledge its use according to the terms posted at the time of submission on:
# http://codeblue.umich.edu/hoomd-blue/citations.html

# * Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
# http://codeblue.umich.edu/hoomd-blue/

# * Apart from the above required attributions, neither the name of the copyright
# holder nor the names of HOOMD-blue's contributors may be used to endorse or
# promote products derived from this software without specific prior written
# permission.

# Disclaimer

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
# WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -- end license --

# Maintainer: joaander

## \package hoomd_script.option
# \brief Commands set global options
#
# Options may be set on the command line (\ref page_command_line_options) or from a job script. The option.set_* commands
# override any settings made on the command line.
#

from optparse import OptionParser;

import hoomd;
from hoomd_script import globals;
from hoomd_script import init;
import sys;
import shlex;

## \internal
# \brief Storage for all option values
#
# options stores values in a similar way to the output of optparse for compatibility with existing
# code.
class options:
    def __init__(self):
        self.mode = None;
        self.gpu = None;
        self.ncpu = None;
        self.gpu_error_checking = None;
        self.min_cpu = None;
        self.ignore_display = None;
        self.user = [];
        self.notice_level = 2;
        self.msg_file = None;
        self.shared_msg_file = None;
        self.nrank = None;
        self.num_worker_threads = 1;

    def __repr__(self):
        tmp = dict(mode=self.mode,
                   gpu=self.gpu,
                   ncpu=self.ncpu,
                   gpu_error_checking=self.gpu_error_checking,
                   min_cpu=self.min_cpu,
                   ignore_display=self.ignore_display,
                   user=self.user,
                   notice_level=self.notice_level,
                   msg_file=self.msg_file,
                   shared_msg_file=self.shared_msg_file,
                   nrank=self.nrank,
                   num_worker_threads=self.num_worker_threads)
        return str(tmp);

## Parses command line options
#
# \internal
# Parses all hoomd_script command line options into the module variable cmd_options
def _parse_command_line():    
    parser = OptionParser();
    parser.add_option("--mode", dest="mode", help="Execution mode (cpu or gpu)");
    parser.add_option("--gpu", dest="gpu", help="GPU on which to execute");
    parser.add_option("--ncpu", dest="ncpu", help="Number of CPU cores on which to execute");
    parser.add_option("--gpu_error_checking", dest="gpu_error_checking", action="store_true", default=False, help="Enable error checking on the GPU");
    parser.add_option("--minimize-cpu-usage", dest="min_cpu", action="store_true", default=False, help="Enable to keep the CPU usage of HOOMD to a bare minimum (will degrade overall performance somewhat)");
    parser.add_option("--ignore-display-gpu", dest="ignore_display", action="store_true", default=False, help="Attempt to avoid running on the display GPU");
    parser.add_option("--notice-level", dest="notice_level", help="Minimum level of notice messages to print");
    parser.add_option("--msg-file", dest="msg_file", help="Name of file to write messages to");
    parser.add_option("--shared-msg-file", dest="shared_msg_file", help="(MPI only) Name of shared file to write message to (append partition #)");
    parser.add_option("--nrank", dest="nrank", help="(MPI only) Number of ranks to include in a partition");
    parser.add_option("--num-threads", dest="num_worker_threads", help="Number of worker threads used for force calculation");
    parser.add_option("--user", dest="user", help="User options");

    (cmd_options, args) = parser.parse_args();
    
    # chedk for valid mode setting
    if cmd_options.mode is not None:
        if not (cmd_options.mode == "cpu" or cmd_options.mode == "gpu"):
            parser.error("--mode must be either cpu or gpu");
    
    # check for sane options
    if cmd_options.mode == "cpu" and (cmd_options.gpu is not None):
        parser.error("--mode=cpu cannot be specified along with --gpu")

    if cmd_options.mode == "gpu" and (cmd_options.ncpu is not None):
        parser.error("--mode=gpu cannot be specified along with --ncpu")

    # set the mode to gpu if the gpu # was set
    if cmd_options.gpu is not None and cmd_options.mode is None:
        cmd_options.mode = "gpu"

    # set the mode to cpu if the ncpu was set
    if cmd_options.ncpu is not None and cmd_options.mode is None:
        cmd_options.mode = "cpu"
    
    # convert ncpu to an integer
    if cmd_options.ncpu is not None:
        try:
            cmd_options.ncpu = int(cmd_options.ncpu);
        except ValueError:
            parser.error('--ncpu must be an integer')
    
    # convert gpu to an integer
    if cmd_options.gpu:
        try:
            cmd_options.gpu = int(cmd_options.gpu);
        except ValueError:
            parser.error('--gpu must be an integer')

    # convert notice_level to an integer
    if cmd_options.notice_level is not None:
        try:
            cmd_options.notice_level = int(cmd_options.notice_level);
        except ValueError:
            parser.error('--notice-level must be an integer')

    # copy command line options over to global options
    globals.options.mode = cmd_options.mode;
    globals.options.gpu = cmd_options.gpu;
    globals.options.ncpu = cmd_options.ncpu;
    globals.options.gpu_error_checking = cmd_options.gpu_error_checking;
    globals.options.min_cpu = cmd_options.min_cpu;
    globals.options.ignore_display = cmd_options.ignore_display;

    if cmd_options.notice_level is not None:
        globals.options.notice_level = cmd_options.notice_level;
        globals.msg.setNoticeLevel(globals.options.notice_level);

    if cmd_options.msg_file is not None:
        globals.options.msg_file = cmd_options.msg_file;
        globals.msg.openFile(globals.options.msg_file);

    if cmd_options.shared_msg_file is not None:
        if not hoomd.is_MPI_available():
            globals.msg.error("Shared log files are only available in MPI builds.\n");
            raise RuntimeError('Error setting option');
        globals.options.shared_msg_file = cmd_options.shared_msg_file;
        globals.msg.setSharedFile(globals.options.shared_msg_file);

    if cmd_options.nrank is not None:
        if not hoomd.is_MPI_available():
            globals.msg.error("The --nrank option is only avaible in MPI builds.\n");
            raise RuntimeError('Error setting option');
        globals.options.nrank = int(cmd_options.nrank);

    globals.options.num_worker_threads = int(cmd_options.num_worker_threads);

    if cmd_options.user is not None:
        globals.options.user = shlex.split(cmd_options.user);

## Set the execution mode
#
# \param mode Specifies the hardware on which to execute. Must be either "cpu", "gpu" or None.
# \note When set to None, the mode is automatically chosen.
# \note Overrides --mode on the command line.
# \sa \ref page_command_line_options
#
def set_mode(mode):
    if init.is_initialized():
            globals.msg.error("Cannot change mode after initialization\n");
            raise RuntimeError('Error setting option');
    
    if mode is not None:
        if not (mode == "cpu" or mode == "gpu"):
            globals.msg.error("Invalid mode setting\n");
            raise RuntimeError('Error setting option');
        
    globals.options.mode = mode;

## Set the gpu
#
# \param gpu Specifies the identifier of the GPU on which to execute. Must be an integer.
# \note When set to None, the GPU is automatically chosen.
# \note When not None, implies \a mode = "gpu"
# \note Overrides --gpu on the command line.
# \sa \ref page_command_line_options
#
def set_gpu(gpu):
    if init.is_initialized():
            globals.msg.error("Cannot change gpu after initialization\n");
            raise RuntimeError('Error setting option');
    
    if gpu is not None:
        try:
            gpu = int(gpu);
        except ValueError:
            globals.msg.error("gpu must be an integer\n");
            raise RuntimeError('Error setting option');
        
        # imply mode=gpu
        globals.options.mode = "gpu";
        
    globals.options.gpu = gpu;

## Set the number of CPU threads
#
# \param ncpu Specifies the number of CPU cores on which to execute (OpenMP). Must be an integer.
# \note When set to None, the number of threads is automatically chosen.
# \note When not None, implies \a mode = "cpu"
# \note Overrides --ncpu on the command line.
# \sa \ref page_command_line_options
#
def set_ncpu(ncpu):
    if init.is_initialized():
            globals.msg.error("Cannot change number of threads after initialization\n");
            raise RuntimeError('Error setting option');
    
    if ncpu is not None:
        try:
            ncpu = int(ncpu);
        except ValueError:
            globals.msg.error("ncpu must be an integer\n");
            raise RuntimeError('Error setting option');
        
        # imply mode=cpu
        globals.options.mode = "cpu";
        
    globals.options.ncpu = ncpu;

## Set the error checking flag
#
# \param gpu_error_checking Specifies whether error checks are made after every GPU call. (True or False)
# \note Overrides --gpu_error_checking on the command line.
# \sa \ref page_command_line_options
#
def set_gpu_error_checking(gpu_error_checking):
    if init.is_initialized():
            globals.msg.error("Cannot change error checking flag after initialization\n");
            raise RuntimeError('Error setting option');
            
    globals.options.gpu_error_checking = gpu_error_checking;

## Set the minimize CPU usage flag
#
# \param min_cpu Specifies whether GPU synchronization blocks to minimize CPU usage. (True or False)
# \note Overrides --minimize-cpu-usage on the command line.
# \sa \ref page_command_line_options
#
def set_min_cpu(min_cpu):
    if init.is_initialized():
            globals.msg.error("Cannot change minimize cpu usage flag after initialization\n");
            raise RuntimeError('Error setting option');
            
    globals.options.min_cpu = min_cpu;

## Set the ignore display GPU flag
#
# \param ignore_display Specifies whether the display GPU should be ignored. (True or False)
# \note Overrides --ignore-display-gpu on the command line.
# \sa \ref page_command_line_options
#
def set_ignore_display(ignore_display):
    if init.is_initialized():
            globals.msg.error("Cannot change ignore display GPU flag after initialization\n");
            raise RuntimeError('Error setting option');
            
    globals.options.ignore_display = ignore_display;

## Get user options
#
# \return List of user options passed in via --user="arg1 arg2 ..."
# \sa \ref page_command_line_options
#
def get_user():
    return globals.options.user;

## Set the notice level
#
# \param notice_level Specifies the maximum notice level to print (an integer)
#
# The notice level may be changed before or after initialization, and may be changed many times during a job script.
#
# \note Overrides --notice-level on the command line.
# \sa \ref page_command_line_options
#
def set_notice_level(notice_level):
    try:
        notice_level = int(notice_level);
    except ValueError:
        globals.msg.error("ncpu must be an integer\n");
        raise RuntimeError('Error setting option');

    globals.msg.setNoticeLevel(notice_level);
    globals.options.notice_level = notice_level;

## Set the message file
#
# \param fname Specifies the name of the file to write. The file will be overwritten. Set to None to direct messages
#              back to stdout/stderr.
#
# The message file may be changed before or after initialization, and may be changed many times during a job script.
# Changing the message file will only affect messages sent after the change.
#
# \note Overrides --msg-file on the command line.
# \sa \ref page_command_line_options
#
def set_msg_file(fname):
    if fname is not None:
        globals.msg.openFile(fname);
    else:
        globals.msg.openStd();

    globals.options.msg_file = fname;

################### Parse command line on load
globals.options = options();
_parse_command_line();

