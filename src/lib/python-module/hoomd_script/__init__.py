# Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
# Source Software License
# Copyright (c) 2008 Ames Laboratory Iowa State University
# All rights reserved.

# Redistribution and use of HOOMD, in source and binary forms, with or
# without modification, are permitted, provided that the following
# conditions are met:

# * Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names HOOMD's
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.

# Disclaimer

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
# CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 

# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.

# $Id$
# $URL$


from hoomd import *

## \package hoomd_script
# \brief Base module for the user-level scripting API
# 
# hoomd_script provides a very high level user interface for executing 
# simulations using HOOMD. This python module is designed to be imported
# into python with "from hoomd_script import *"
#
# More details to add later...

## Internal python variable 
__all__ = ["analyze", "bond", "dump", "force", "globals", "init", 
			"integreate", "pair", "update", "run"];

## \brief Runs the simulation for a given number of time steps
#
# \param tsteps Number of timesteps to advance the simulation by
# 
# \b Examples:<br>
# run(1000)<br>
# run(10e6)<br>
#
# Execute the run() command to advance the simulation forward in time. 
# During the run, all previously specified \ref analyze "analyzers", 
# \ref dump "dumps", \ref update "updaters" and the \ref integrate "integrators"
# are executed every so many time steps as specified by their individual periods.
# 
# After run() completes, you may change parameters of the simulation (i.e. temperature)
# and continue the simulation by executing run() again. Time steps are added
# cumulatively, so calling run(1000) and then run(2000) would run the simulation
# up to time step 3000.
#
# run() cannot be executed before the system is \ref init "initialized". In most 
# cases, it also doesn't make sense to execute run() until after pair forces, bond forces,
# and an \ref integrate "integrator" have been created.
def run(tsteps):
	# check if initialization has occured
	if (globals.system == None):
		print "Error: Cannot run before initialization";
		raise RuntimeError('Error running');
	
	globals.system.run(int(tsteps));

	
	