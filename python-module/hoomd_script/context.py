# -- start license --
# Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
# (HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
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

# Maintainer: csadorf / All Developers are free to add commands for new features

## \package hoomd_script.context
# \brief Gather information about the execution context
#
# As much data from the environment is gathered as possible.

import hoomd;
from hoomd_script import meta;

# The following global variables keep track of the walltime and processing time since the import of hoomd_script
import time
TIME_START = time.time()
CLOCK_START = time.clock()

## \internal
# \brief Gather context from the environment
class ExecutionContext(meta._metadata):
    ## \internal
    # \brief Constructs the context object
    def __init__(self):
        meta._metadata.__init__(self)
        self.metadata_fields = [
            'hostname', 'num_cpu', 'gpu', 'num_ranks',
            'hoomd_version', 'hoomd_git_sha1', 'hoomd_git_refspec',
            'cuda_version', 'compiler_version',
            'username', 'wallclocktime', 'cputime',
            ]

    ## \internal
    # \brief Return the execution configuration if initialized or raise exception.
    def _get_exec_conf(self):
        from hoomd_script import globals
        if globals.exec_conf is None:
            raise RuntimeError("Not initialized.")
        else:
            return globals.exec_conf

    # \brief Return the network hostname.
    @property
    def hostname(self):
        import socket
        return socket.gethostname()

    # \brief Return the number of CPUs used.
    @property
    def num_cpu(self):
        return self._get_exec_conf().n_cpu

    # \brief Return the name of the GPU used in GPU mode.
    @property
    def gpu(self):
        return self._get_exec_conf().getGPUName()

    # \brief Return the number of ranks.
    @property
    def num_ranks(self):
        try:
            return self._get_exec_conf().getNRanks()
        except AttributeError:
            return 1

    # \brief Return the hoomd version.
    @property
    def hoomd_version(self):
        from hoomd import __version__
        return __version__

    # \brief Return the hoomd git hash
    @property
    def hoomd_git_sha1(self):
        from hoomd import __git_sha1__
        return __git_sha1__

    # \brief Return the hoomd git refspec
    @property
    def hoomd_git_refspec(self):
        from hoomd import __git_refspec__
        return __git_refspec__

    # \brief Return the cuda version
    @property
    def cuda_version(self):
        from hoomd import __cuda_version__
        return __cuda_version__

    # \brief Return the compiler version
    @property
    def compiler_version(self):
        from hoomd import __compiler_version__
        return __compiler_version__

    # \brief Return the username.
    @property
    def username(self):
        import getpass
        return getpass.getuser()

    # \brief Return the wallclock time since the import of hoomd_script
    @property
    def wallclocktime(self):
        return time.time() - TIME_START

    # \brief Return the CPU clock time since the import of hoomd_script
    @property
    def cputime(self):
        return time.clock() - CLOCK_START
