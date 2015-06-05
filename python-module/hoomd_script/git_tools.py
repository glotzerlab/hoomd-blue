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

import subprocess
import os

BIN_GIT = 'git'

class StageDirtyWarning(RuntimeWarning):
    pass

def call_dev_null(cmd):
    FNULL = open(os.devnull, 'w')
    return subprocess.call(cmd.split(), stdout = FNULL, stderr = subprocess.STDOUT)

def check_output_dev_null(cmd):
    FNULL = open(os.devnull, 'w')
    return subprocess.check_output(cmd.split(), stderr = FNULL)

def found_git():
    try:
        clean_stage()
    except OSError:
        return False
    else:
        return True

def current_sha1():
    cmd = BIN_GIT + ' show -s --pretty=format:%H HEAD'
    return check_output_dev_null(cmd).strip().decode('utf-8')

def local_changes():
    cmd = BIN_GIT + ' diff --exit-code'
    return call_dev_null(cmd)

def cached_changes():
    cmd = BIN_GIT + ' diff --cached --exit-code'
    return call_dev_null(cmd)

def clean_stage():
    return not (local_changes() or cached_changes())

def sha1_if_clean_stage():
    if clean_stage():
        return current_sha1()
    else:
        raise StageDirtyWarning()

def main():
    print('sha1', current_sha1())
    print('diff', local_changes())
    print('cached', cached_changes())
    print('sha1 if clean', sha1_if_clean_stage())
    return 0

if __name__ == '__main__':
    main()
