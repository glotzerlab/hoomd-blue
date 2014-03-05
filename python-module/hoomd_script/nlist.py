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

# Maintainer: joaander

from hoomd_script import globals;
from hoomd_script import util;

## \package hoomd_script.nlist
# \brief Wrapper for "global" nlist commands
#
# This package is a thin wrapper around globals.neighbor_list. It takes the place of the old model for making the
# global neighbor list available as "nlist" in the __main__ namespace. Moving it into the hoomd_script namespace
# is backwards compatible as long as the user does "from hoomd_script import *" - but it also makes it much easier
# to reference the nlist from modules other than __main__.
#
# Backwards compatibility is only ensured if the script only uses the public python facing API. Bypassing this to get
# at the C++ interface should be done through globals.neighbor_list
#
# Future expansions to enable multiple neighbor lists could be enabled through this same mechanism.

## \internal
# \brief Thin wrapper for set_params
def set_params(*args, **kwargs):
    util.print_status_line();
    util._disable_status_lines = True;
    globals.neighbor_list.set_params(*args, **kwargs);
    util._disable_status_lines = False;

## \internal
# \brief Thin wrapper for reset_exclusions
def reset_exclusions(*args, **kwargs):
    util.print_status_line();
    util._disable_status_lines = True;
    globals.neighbor_list.reset_exclusions(*args, **kwargs);
    util._disable_status_lines = False;

## \internal
# \brief Thin wrapper for benchmark
def benchmark(*args, **kwargs):
    util.print_status_line();
    util._disable_status_lines = True;
    globals.neighbor_list.benchmark(*args, **kwargs);
    util._disable_status_lines = False;

## \internal
# \brief Thin wrapper for query_update_period
def query_update_period(*args, **kwargs):
    util.print_status_line();
    util._disable_status_lines = True;
    globals.neighbor_list.query_update_period(*args, **kwargs);
    util._disable_status_lines = False;
