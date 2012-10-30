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

# Maintainer: joaander / All Developers are free to add commands for new features

from hoomd_script import globals
from hoomd_script import init
import hoomd_script
import hoomd

try:
    import numpy;
except ImportError:
    numpy = None;

##
# \package hoomd_script.benchmark
# \brief Commands for benchmarking the performance of HOOMD


## Perform a series of short runs to benchmark overall simulation performance
# \param warmup Number of time steps to run() to warm up the benchmark
# \param repeat Number of times to repeat the benchmark \a steps
# \param steps Number of time steps to run() at each benchmark point
#
# series_run() executes \a warmup time steps. After that, it simply
# calls run(steps), \a repeat times and returns a list containing the average TPS for each of those runs.
#
# If numpy is available, a brief summary of the benchmark results will be printed to the screen 
def series(warmup=100000, repeat=20, steps=10000):
    # check if initialization has occurred
    if not init.is_initialized():
        globals.msg.error("Cannot tune r_buff before initialization\n");

    tps_list = [];
    
    hoomd_script.run(warmup);
    for i in range(0,repeat):
        hoomd_script.run(steps);
        tps_list.append(globals.system.getLastTPS());
    
    if numpy is not None:
        globals.msg.notice(1, "**Notice: Series average TPS: %4.2f\n" % numpy.average(tps_list));
        globals.msg.notice(1, "          Series median TPS : %4.2f\n" % numpy.median(tps_list));
        globals.msg.notice(1, "          Series TPS std dev: %4.2f" % numpy.std(tps_list));
    
    return tps_list;

