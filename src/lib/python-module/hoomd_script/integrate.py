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

## \package hoomd_script.integrate
# \brief Commands that integrate the equations of motion
#
# Commands beginning with integrate. specify the integrator to use when
# advancing particles forward in time. By default, no integrator is
# specified. An integrator can specified anywhere before executing the 
# run() command, which will use the last integrator set in advancing
# particles forward in time. If a number of integrators are created,
# the last one is the only one to take effect. For example:
# \code
# integrate.nvt(T=1.2, tau=0.5)
# integrate.nve() 
# run(100)
# \endcode
# In this example, the nvt integration is ignored as the creation of the
# nve integrator overwrote it.
#
# However, it is valid to run() a number of time steps with one integrator
# and then switch to another for the next run().
#
# Some integrators provide parameters that can be changed between runs.
# In order to access the integrator to change it, it needs to be saved
# in a variable. For example:
# \code
# integrator = integrate.nvt(T=1.2, tau=0.5)
# run(100)
# integrator.set_params(T=1.0)
# run(100)
# \endcode
# This code snippet runs the first 100 time steps with T=1.2 and the next 100 with T=1.0

from hoomd import *
import globals

## \internal
# \brief Base class for integrators
#
# Details
class _integrator:
	## Constructs the integrator
	# \param self Python-required class variable
	# This doesn't really do much bet set some member variables to None
	def __init__(self):
		# check if initialization has occured
		if (globals.system == None):
			print "Error: Cannot create integrator before initialization";
			raise RuntimeError('Error creating integrator');
		
		self.cpp_integrator = None;
		
	## \var cpp_integrator
	# Stores the C++ side Integrator managed by this class
		
	
	
## NVT Integration via the Nos&eacute;-Hoover thermostat
#
# Details
class nvt(_integrator):
	pass
	
## NVE Integration via Velocity-Verlet
#
# Details
class nve(_integrator):
	pass
	

