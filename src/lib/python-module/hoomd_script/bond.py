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

import force;
import globals;
import hoomd;

## \package hoomd_script.bond
# \brief Commands that create bond forces
#
# Details

## Harmonic bond forces
#
# TODO: document
# This is a TEMPORARY HACK class. The interface \b WILL change in the near future
class harmonic(force._force):
	## Specify the harmonic bond force
	#
	# TODO: document me
	def __init__(self, K, r0):
		print "bond.harmonic(K =", K, ", r0 =", r0, ")";
		
		# if there is no initializer that deals with bonds, error out
		if not globals.initializer:
			print "Cannot create bonds without an initializer that sets them!"
			raise RuntimeError("Error creating bond forces");
		
		# initialize the base class
		force._force.__init__(self);
		
		# create the c++ mirror class
		if globals.particle_data.getExecConf().exec_mode == hoomd.ExecutionConfiguration.executionMode.CPU:
			self.cpp_force = hoomd.BondForceCompute(globals.particle_data, K, r0);
		elif globals.particle_data.getExecConf().exec_mode == hoomd.ExecutionConfiguration.executionMode.GPU:
			self.cpp_force = hoomd.BondForceComputeGPU(globals.particle_data, K, r0);
		else:
			print "Invalid execution mode";
			raise RuntimeError("Error creating bond forces");

		globals.bond_compute = self.cpp_force;

		globals.system.addCompute(self.cpp_force, self.force_name);
		
		# add the bonds
		globals.initializer.setupBonds(self.cpp_force);
		
	def update_coeffs(self):
		pass
