# -- start license --
# Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
# (HOOMD-blue) Open Source Software License Copyright 2009-2016 The Regents of
# the University of Michigan All rights reserved.

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

R""" Utilities.
"""

import sys;
import traceback;
import os.path;
import linecache;
import re;
import hoomd;
from hoomd import _hoomd;

## \internal
# \brief Internal flag tracking if status lines should be quieted
_status_quiet_count = 0;

def quiet_status():
    R""" Quiet the status line output.

    After calling :py:func:`hoomd.util.quiet_status()`, hoomd will no longer print out the line of
    code that executes each hoomd script command. Call :py:func:`hoomd.util.unquiet_status()` to
    enable the status messages again. Messages are only enabled after a number of
    :py:func:`hoomd.util.unquiet_status()` calls equal to the number of prior
    :py:func:`hoomd.quiet_status()` calls.
    """
    global _status_quiet_count;
    _status_quiet_count = _status_quiet_count+1;

def unquiet_status():
    R""" Resume the status line output.

    See Also:
        :py:func:`hoomd.util.quiet_status()`
    """
    global _status_quiet_count;
    _status_quiet_count = max(0, _status_quiet_count-1);

## \internal
# \brief Prints a status line tracking the execution of the current hoomd script
def print_status_line():
    if _status_quiet_count > 0:
        return;

    # get the traceback info first
    stack = traceback.extract_stack();
    if len(stack) < 3:
        hoomd.context.msg.notice(2, "hoomd_script executing unknown command\n");

    if sys.version_info[:3] != (3, 5, 0):
        frame = -3
    else:
        frame = -4

    file_name, line, module, code = stack[frame];

    # if we are in interactive mode, there is no need to print anything: the
    # interpreter loop does it for us. We can make that check by testing if
    # sys.ps1 is defined (this is not a hack, the python documentation states
    # that ps1 is _only_ defined in interactive mode
    if 'ps1' in sys.__dict__:
        return

    # piped input from stdin doesn't provide a code line, handle the situation
    if not code:
        message = os.path.basename(file_name) + ":" + str(line).zfill(3) + "  |  <unknown code>";
        hoomd.context.msg.notice(1, message + '\n');
    else:
        # build and print the message line
        # Go upwards in the source until you match the closing paren
        # dequote ensures we ignore literal parens
        dequote = lambda x: re.sub(r'[\'"].*?[\'"]','',x)
        balance = lambda y: y.count('(') - y.count(')')
        message = []
        while True:
            message.insert(0,linecache.getline(file_name,line))
            if sum(balance(dequote(x)) for x in message) == 0 or line == 0:
                break
            line = line - 1

        message.insert(0,os.path.basename(file_name) + ":" + str(line).zfill(3) + "  |  ")
        hoomd.context.msg.notice(1, ''.join(message).rstrip('\n') + '\n');
        linecache.clearcache()

def cuda_profile_start():
    """ Start CUDA profiling.

    When using nvvp to profile CUDA kernels in hoomd jobs, you usually don't care about all the initialization and
    startup. cuda_profile_start() allows you to not even record that. To use, uncheck the box "start profiling on
    application start" in your nvvp session configuration. Then, call cuda_profile_start() in your hoomd script when
    you want nvvp to start collecting information.

    Example::

        from hoomd_script import *
        init.read_xml("init.xml");
        # setup....
        run(30000);  # warm up and auto-tune kernel block sizes
        option.set_autotuner_params(enable=False);  # prevent block sizes from further autotuning
        cuda_profile_start();
        run(100);

    """
    _hoomd.cuda_profile_start();

def cuda_profile_stop():
    """ Stop CUDA profiling.

        See Also:
            :py:func:`cuda_profile_start()`.
    """

    _hoomd.cuda_profile_stop();
