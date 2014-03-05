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

import hoomd

from hoomd_script import globals
from hoomd_script import init
from hoomd_script import util
import hoomd_script

import math
import os
import sys

##
# \package hoomd_script.tune
# \brief Commands for tuning the performance of HOOMD

## Make a series of short runs to determine the fastest performing r_buff setting
# \param warmup Number of time steps to run() to warm up the benchmark
# \param r_min Smallest value of r_buff to test
# \param r_max Largest value of r_buff to test
# \param jumps Number of different r_buff values to test
# \param steps Number of time steps to run() at each point
# \param set_max_check_period Set to True to enable automatic setting of the maximum nlist check_period
#
# tune.r_buff() executes \a warmup time steps. Then it sets the nlist \a r_buff value to \a r_min and runs for
# \a steps time steps. The TPS value is recorded, and the benchmark moves on to the next \a r_buff value
# completing at \a r_max in \a jumps jumps. Status information is printed out to the screen, and the optimal
# \a r_buff value is left set for further runs() to continue at optimal settings.
#
# Each benchmark is repeated 3 times and the median value chosen. Then, \a warmup time steps are run() again
# at the optimal r_buff in order to determine the maximum value of check_period. In total,
# (2*warmup + 3*jump*steps) time steps are run().
#
# \note By default, the maximum check_period is \b not set in tune.r_buff() for safety. If you wish to have it set
# when the call completes, call with the parameter set_max_check_period=True.
#
# \returns (optimal_r_buff, maximum check_period)
#
# \MPI_SUPPORTED
def r_buff(warmup=200000, r_min=0.05, r_max=1.0, jumps=20, steps=5000, set_max_check_period=False):
    # check if initialization has occurred
    if not init.is_initialized():
        globals.msg.error("Cannot tune r_buff before initialization\n");

    # check that there is a nlist
    if globals.neighbor_list is None:
        globals.msg.error("Cannot tune r_buff when there is no neighbor list\n");

    # start off at a check_period of 1
    globals.neighbor_list.set_params(check_period=1);

    # make the warmup run
    hoomd_script.run(warmup);

    # initialize scan variables
    dr = (r_max - r_min) / (jumps - 1);
    r_buff_list = [];
    tps_list = [];

    # loop over all desired r_buff points
    for i in range(0,jumps):
        # set the current r_buff
        r_buff = r_min + i * dr;
        globals.neighbor_list.set_params(r_buff=r_buff);

        # run the benchmark 3 times
        tps = [];
        hoomd_script.run(steps);
        tps.append(globals.system.getLastTPS())
        hoomd_script.run(steps);
        tps.append(globals.system.getLastTPS())
        hoomd_script.run(steps);
        tps.append(globals.system.getLastTPS())

        # record the median tps of the 3
        tps.sort();
        tps_list.append(tps[1]);
        r_buff_list.append(r_buff);

    # find the fastest r_buff
    fastest = tps_list.index(max(tps_list));
    fastest_r_buff = r_buff_list[fastest];

    # set the fastest and rerun the warmup steps to identify the max check period
    globals.neighbor_list.set_params(r_buff=fastest_r_buff);
    hoomd_script.run(warmup);

    # notify the user of the benchmark results
    globals.msg.notice(2, "r_buff = " + str(r_buff_list) + '\n');
    globals.msg.notice(2, "tps = " + str(tps_list) + '\n');
    globals.msg.notice(2, "Optimal r_buff: " + str(fastest_r_buff) + '\n');
    globals.msg.notice(2, "Maximum check_period: " + str(globals.neighbor_list.query_update_period()) + '\n');

    # set the found max check period
    if set_max_check_period:
        globals.neighbor_list.set_params(check_period=globals.neighbor_list.query_update_period());

    # return the results to the script
    return (fastest_r_buff, globals.neighbor_list.query_update_period());
