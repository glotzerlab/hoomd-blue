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

from hoomd_script import nlist
from hoomd_script import util
import hoomd_script

##
# \package hoomd_script.tune
# \brief Commands for tuning the performance of HOOMD

## Thin wrapper to tune the global neighbor list parameters
#
# \param warmup Number of time steps to run() to warm up the benchmark
# \param r_min Smallest value of r_buff to test
# \param r_max Largest value of r_buff to test
# \param jumps Number of different r_buff values to test
# \param steps Number of time steps to run() at each point
# \param set_max_check_period Set to True to enable automatic setting of the maximum nlist check_period
#
# tune() executes \a warmup time steps. Then it sets the nlist \a r_buff value to \a r_min and runs for
# \a steps time steps. The TPS value is recorded, and the benchmark moves on to the next \a r_buff value
# completing at \a r_max in \a jumps jumps. Status information is printed out to the screen, and the optimal
# \a r_buff value is left set for further runs() to continue at optimal settings.
#
# Each benchmark is repeated 3 times and the median value chosen. Then, \a warmup time steps are run() again
# at the optimal r_buff in order to determine the maximum value of check_period. In total,
# (2*warmup + 3*jump*steps) time steps are run().
#
# \note By default, the maximum check_period is \b not set for safety. If you wish to have it set
# when the call completes, call with the parameter set_max_check_period=True.
#
# \returns (optimal_r_buff, maximum_check_period)
#
# \note This wrapper is maintained for backwards compatibility with the global neighbor list, but may be removed in
# future versions.
#
# \MPI_SUPPORTED
def r_buff(warmup=200000, r_min=0.05, r_max=1.0, jumps=20, steps=5000, set_max_check_period=False):
    util.print_status_line();
    util._disable_status_lines = True;
    tuner_output = hoomd_script.context.current.neighbor_list.tune(warmup, r_min, r_max, jumps, steps, set_max_check_period)
    util._disable_status_lines = False;
    return tuner_output
