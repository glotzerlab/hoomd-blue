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
# run() command, which will then use the last integrator set. 
# If a number of integrators are created in the script,
# the last one is the only one to take effect. For example:
# \code
# integrate.nvt(dt=0.005, T=1.2, tau=0.5)
# integrate.nve(dt=0.005) 
# run(100)
# \endcode
# In this example, the nvt integration is ignored as the creation of the
# nve integrator overwrote it.
#
# However, it is valid to run() a number of time steps with one integrator
# and then replace it with another before the next run().
#
# Some integrators provide parameters that can be changed between runs.
# In order to access the integrator to change it, it needs to be saved
# in a variable. For example:
# \code
# integrator = integrate.nvt(dt=0.005, T=1.2, tau=0.5)
# run(100)
# integrator.set_params(T=1.0)
# run(100)
# \endcode
# This code snippet runs the first 100 time steps with T=1.2 and the next 100 with T=1.0

import hoomd;
import globals;

## \internal
# \brief Base class for integrators
#
# An integrator in hoomd_script reflects an Integrator in c++. It is responsible
# for all high-level management that happens behind the scenes for hoomd_script
# writers. 1) The instance of the c++ integrator itself is tracked 2) All
# forces created so far in the simulation are updated in the cpp_integrator
# whenever run() is called.
class _integrator:
	## \internal
	# \brief Constructs the integrator
	#
	# \param self Python-required class instance variable
	#
	# This doesn't really do much bet set some member variables to None
	def __init__(self):
		# check if initialization has occured
		if globals.system == None:
			print "Error: Cannot create integrator before initialization";
			raise RuntimeError('Error creating integrator');
		
		self.cpp_integrator = None;
		
		# save ourselves in the global variable
		globals.integrator = self;
		
	## \var cpp_integrator
	# \internal 
	# \brief Stores the C++ side Integrator managed by this class
	
	## \internal
	# \brief Updates the integrators in the reflected c++ class
	#
	# \param self Python-required class instance variable
	def update_forces(self):
		# check that proper initialization has occured
		if self.cpp_integrator == None:
			print "Bug in hoomd_script: cpp_integrator not set, please report";
			raise RuntimeError('Error updating forces');		
		
		# set the forces
		self.cpp_integrator.removeForceComputes();
		for f in globals.forces:
			if f.enabled:
				if f.cpp_force == None:
					print "Bug in hoomd_script: cpp_integrator not set, please report";
					raise RuntimeError('Error updating forces');		
				else:
					self.cpp_integrator.addForceCompute(f.cpp_force);



## NVT Integration via the Nos&eacute;-Hoover thermostat
#
# integrate.nvt performs constant volume, constant temperature simulations using the standard
# Nos&eacute;-Hoover thermostat. 
class nvt(_integrator):
	## Specifies the NVT integrator
	# \param self Python-required class instance variable
	# \param dt Each time step of the simulation run() will advance the real time of the system forward by \a dt
	# \param T Temperature set point for the Nos&eacute;-Hoover thermostat
	# \param tau Coupling constant for the Nos&eacute;-Hoover thermostat. It is related to the Nos&eacute; mass Q by TODO
	#
	# \b Examples:<br>
	# integrate.nvt(dt=0.005, T=1.0, tau=0.5)<br>
	# integrator = integrate.nvt(tau=1.0, dt=5e-3, T=0.65)<br>
	def __init__(self, dt, T, tau):
		print "integrate.nvt(dt=", dt, ", T=", T, ", tau=", tau, ")";
		
		# initialize base class
		_integrator.__init__(self);
		
		# initialize the reflected c++ class
		self.cpp_integrator = hoomd.NVTUpdater(globals.particle_data, dt, tau, T);
		globals.system.setIntegrator(self.cpp_integrator);
	
	## Changes parameters of an existing integrator
	# \param self Python-required class instance variable
	# \param dt New time step delta (if set)
	# \param T New temperature (if set)
	# \param tau New coupling constant (if set)
	#
	# \b Examples (assuming the integrator was saved in the variable \a integrator):<br>
	# integrator.set_params(dt=0.007)<br>
	# integrator.set_params(tau=0.6)<br>
	# integrator.set_params(dt=3e-3, T=2.0)<br>
	def set_params(self, dt=None, T=None, tau=None):
		print "nvt.set_params(dt=", dt, ", T=", T, ", tau=", tau, ")";
		# check that proper initialization has occured
		if self.cpp_integrator == None:
			print "Bug in hoomd_script: cpp_integrator not set, please report";
			raise RuntimeError('Error updating forces');
		
		# change the parameters
		if dt != None:
			self.cpp_integrator.setDeltaT(dt);
		if T != None:
			self.cpp_integrator.setT(T);
		if tau != None:
			self.cpp_integrator.setTau(tau);


## NVE Integration via Velocity-Verlet
#
# integrate.nve performs constant volume, constant energy simulations using the standard
# Velocity-Verlet method. For poor initial conditions that include overlapping atoms, a 
# limit can be specified to the movement a particle is allowed to make in one time step. 
# After a few thousand time steps with the limit set, the system should be in a safe state 
# to continue with unconstrained integration.
class nve(_integrator):
	## Specifies the NVE integrator
	# \param self Python-required class instance variable
	# \param dt Each time step of the simulation run() will advance the real time of the system forward by \a dt
	# \param limit (optional) Enforce that no particle moves more than a distance of \a limit in a single time step
	#
	# \b Examples:<br>
	# integrate.nve(dt=0.005)<br>
	# integrator = integrate.nvt(dt=5e-3)<br>
	# integrate.nve(dt=0.005, limit=0.01)<br>
	def __init__(self, dt, limit=None):
		print "integrate.nve(dt=", dt, ", limit=", limit, ")";
		
		# initialize base class
		_integrator.__init__(self);
		
		# initialize the reflected c++ class
		self.cpp_integrator = hoomd.NVEUpdater(globals.particle_data, dt);
		
		# set the limit
		if limit != None:
			self.cpp_integrator.setLimit(limit);
		
		globals.system.setIntegrator(self.cpp_integrator);
	
	## Changes parameters of an existing integrator
	# \param self Python-required class instance variable
	# \param dt New time step (if set)
	#
	# \b Examples (assuming the integrator was saved in the variable \a integrator):<br>
	# integrator.set_params(dt=0.007)<br>
	# integrator.set_params(dt=3e-3)<br>
	def set_params(self, dt=None):
		print "nvt.set_params(dt=", dt, ")";
		# check that proper initialization has occured
		if self.cpp_integrator == None:
			print "Bug in hoomd_script: cpp_integrator not set, please report";
			raise RuntimeError('Error updating forces');
		
		# change the parameters
		if dt != None:
			self.cpp_integrator.setDeltaT(dt);
	

