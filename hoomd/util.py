# Copyright (c) 2009-2019 The Regents of the University of Michigan
# This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

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
# \brief Compatibility definition of a basestring for python 2/3
try:
    _basestring = basestring
except NameError:
    _basestring = str

## \internal
# \brief Checks if a variable is an instance of a string and always returns a list.
# \param s Variable to turn into a list
# \returns A list
def listify(s):
    if isinstance(s, _basestring):
        return [s]
    else:
        return list(s)

## \internal
# \brief Internal flag tracking if status lines should be quieted
_status_quiet_count = 0;

def quiet_status():
    R""" Quiet the status line output.

    After calling :py:func:`hoomd.util.quiet_status()`, hoomd will no longer print out the line of
    code that executes each hoomd script command. Call :py:func:`hoomd.util.unquiet_status()` to
    enable the status messages again. Messages are only enabled after a number of
    :py:func:`hoomd.util.unquiet_status()` calls equal to the number of prior
    :py:func:`hoomd.util.quiet_status()` calls.
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
        hoomd.context.msg.notice(2, "hoomd executing unknown command\n");

    if sys.version_info[:3] != (3, 5, 0):
        frame = -3
    else:
        frame = -4

    try:
        file_name, line, module, code = stack[frame];
    except IndexError:
        # No traceback information is available.
        return

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

        from hoomd import *
        init.read_xml("init.xml");
        # setup....
        run(30000);  # warm up and auto-tune kernel block sizes
        option.set_autotuner_params(enable=False);  # prevent block sizes from further autotuning
        cuda_profile_start();
        run(100);

    """
    # check if initialization has occurred
    if not hoomd.init.is_initialized():
        hoomd.context.msg.error("Cannot start profiling before initialization\n");
        raise RuntimeError('Error starting profile');

    if hoomd.context.exec_conf.isCUDAEnabled():
        hoomd.context.exec_conf.cudaProfileStart();

def cuda_profile_stop():
    """ Stop CUDA profiling.

        See Also:
            :py:func:`cuda_profile_start()`.
    """
    # check if initialization has occurred
    if not hoomd.init.is_initialized():
        hoomd.context.msg.error("Cannot stop profiling before initialization\n");
        raise RuntimeError('Error stopping profile');

    if hoomd.context.exec_conf.isCUDAEnabled():
        hoomd.context.exec_conf.cudaProfileStop();
