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

## \package hoomd_script.pair
# \brief Commands that create forces between pairs of particles
#
# Generally, %pair forces are short range and are summed over all non-bonded particles
# within a certain cutoff radius of each particle. Any number of %pair forces
# can be defined in a single simulation. The net %force on each particle due to
# all types of %pair forces is summed.
#
# Pair forces require that parameters be set for each unique type %pair. Coefficients
# are set through the aid of the coeff class. To set this coefficients, specify 
# a %pair %force and save it in a variable
# \code
# my_force = pair.some_pair_force(arguments...)
# \endcode
# Then the coefficients can be set using the saved variable.
# \code
# my_force.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, alpha=0.0)
# my_force.pair_coeff.set('A', 'B', epsilon=1.0, sigma=1.0, alpha=0.0)
# my_force.pair_coeff.set('B', 'B', epsilon=1.0, sigma=1.0, alpha=1.0)
# \endcode
# This example set the parameters \a epsilon, \a sigma, and \a alpha 
# (which are used in pair.lj). Different %pair forces require that different
# coefficients are set. Check the documentation of each to see the definition
# of the coefficients.
#
# \sa \ref page_quick_start

import globals;
import force;
import hoomd;
import math;
import sys;
import util;

## Defines %pair coefficients
# 
# All %pair forces use coeff to specify the coefficients between different
# pairs of particles indexed by type. The set of %pair coefficients is a symmetric
# matrix defined over all possible pairs of particle types.
#
# There are two ways to set the coefficients for a particular %pair %force. 
# The first way is to save the %pair %force in a variable and call set() directly.
# To see an example of this, see the documentation for the package pair
# or the \ref page_quick_start
#
# The second method is to build the coeff class first and then assign it to the
# %pair %force. There are some advantages to this method in that you could specify a
# complicated set of %pair coefficients in a separate python file and import it into
# your job script.
#
# Example (file \em force_field.py):
# \code
# from hoomd_script import *
# my_coeffs = pair.coeff();
# my_coeffs.set('A', 'A', epsilon=1.0, sigma=1.0, alpha=0.0)
# my_coeffs.set('A', 'B', epsilon=1.0, sigma=1.0, alpha=0.0)
# my_coeffs.set('B', 'B', epsilon=1.0, sigma=1.0, alpha=1.0)
# \endcode
# Example job script:
# \code
# from hoomd_script import *
# import force_field
#
# .....
# my_force = pair.some_pair_force(arguments...)
# my_force.pair_coeff = force_field.my_coeffs
# \endcode
class coeff:
	
	## \internal
	# \brief Initializes the class
	# \details
	# The main task to be performed during initialization is just to init some variables
	# \param self Python required class instance variable
	def __init__(self):
		self.values = {};
		
	## \var values
	# \internal
	# \brief Contains the matrix of set values in a dictionary
	
	## Sets parameters for one type %pair
	# \param a First particle type in the %pair
	# \param b Second particle type in the %pair
	# \param coeffs Named coefficients (see below for examples)
	#
	# Calling set() results in one or more parameters being set for a single type %pair.
	# Particle types are identified by name, and parameters are also added by name. 
	# Which parameters you need to specify depends on the %pair %force you are setting
	# these coefficients for, see the corresponding documentation.
	#
	# All possible type pairs as defined in the simulation box must be specified before
	# executing run(). You will receive an error if you fail to do so. It is not an error,
	# however, to specify coefficients for particle types that do not exist in the simulation.
	# This can be useful in defining a %force field for many different types of particles even
	# when some simulations only include a subset.
	#
	# There is no need to specify coefficients for both pairs 'A','B' and 'B','A'. Specifying
	# only one is sufficient.
	#
	# \b Examples:
	# \code
	# coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
	# coeff.set('B', 'B', epsilon=2.0, sigma=1.0)
	# coeff.set('A', 'B', epsilon=1.5, sigma=1.0)
	# \endcode
	#
	# \note Single parameters can be updated. If both epsilon and sigma have already been 
	# set for a type %pair, then executing coeff.set('A', 'B', epsilon=1.1) will %update 
	# the value of epsilon and leave sigma as it was previously set.
	def set(self, a, b, **coeffs):
		util.print_status_line();
		
		# create the pair if it hasn't been created it
		if (not (a,b) in self.values) and (not (b,a) in self.values):
			self.values[(a,b)] = {};
			
		# Find the pair to update
		if (a,b) in self.values:
			cur_pair = (a,b);
		elif (b,a) in self.values:
			cur_pair = (b,a);
		else:
			print >> sys.stderr, "\nBug detected in pair.coeff. Please report\n"
			raise RuntimeError("Error setting pair coeff");
		
		# update each of the values provided
		if len(coeffs) == 0:
			print >> sys.stderr, "\n***Error! No coefficents specified\n";
		for name, val in coeffs.items():
			self.values[cur_pair][name] = val;
	
	## \internal
	# \brief Verifies set parameters form a full matrix with all values set
	# \details
	# \param self Python required self variable
	# \param required_coeffs list of required variables
	#
	# This can only be run after the system has been initialized
	def verify(self, *required_coeffs):
		# first, check that the system has been initialized
		if globals.system == None:
			print >> sys.stderr, "\n***Error! Cannot verify pair coefficients before initialization\n";
			raise RuntimeError('Error verifying pair coefficients');
		
		# get a list of types from the particle data
		ntypes = globals.particle_data.getNTypes();
		type_list = [];
		for i in xrange(0,ntypes):
			type_list.append(globals.particle_data.getNameByType(i));
		
		valid = True;
		# loop over all possible pairs and verify that all required variables are set
		for i in xrange(0,ntypes):
			for j in xrange(i,ntypes):
				a = type_list[i];
				b = type_list[j];
				
				# find which half of the pair is set
				if (a,b) in self.values:
					cur_pair = (a,b);
				elif (b,a) in self.values:
					cur_pair = (b,a);
				else:
					print >> sys.stderr, "\n***Error! Type pair", (a,b), "not found in pair coeff\n"
					valid = False;
					continue;
				
				# verify that all required values are set by counting the matches
				count = 0;
				for coeff_name in self.values[cur_pair].keys():
					if not coeff_name in required_coeffs:
						print "Notice: Possible typo? Pair coeff", coeff_name, "is specified for pair", (a,b), ", but is not used by the pair force";
					else:
						count += 1;
				
				if count != len(required_coeffs):
					print >> sys.stderr, "\n***Error! Type pair", (a,b), "is missing required coefficients\n";
					valid = False;
				
			
		return valid;
		
	## \internal
	# \brief Gets the value of a single pair coefficient
	# \detail
	# \param a First name in the type pair
	# \param b Second name in the type pair
	# \param coeff_name Coefficient to get
	def get(self, a, b, coeff_name):
		# Find the pair to update
		if (a,b) in self.values:
			cur_pair = (a,b);
		elif (b,a) in self.values:
			cur_pair = (b,a);
		else:
			print >> sys.stderr, "\nBug detected in pair.coeff. Please report\n"
			raise RuntimeError("Error setting pair coeff");
		
		return self.values[cur_pair][coeff_name];
		
		
## Interface for controlling neighbor list parameters
#
# A neighbor list should not be directly created by you. One will be automatically
# created whenever a %pair %force is specified. The cutoff radius is set to the
# maximum of that set for all defined %pair forces.
#
# Any bonds defined in the simulation are automatically used to exclude bonded particle
# pairs from appearing in the neighbor list.
class nlist:
	## \internal
	# \brief Constructs a neighbor list
	# \details
	# \param self Python required instance variable
	# \param r_cut Cutoff radius
	def __init__(self, r_cut):
		# check if initialization has occured
		if globals.system == None:
			print >> sys.stderr, "\n***Error!Cannot create neighbor list before initialization\n";
			raise RuntimeError('Error creating neighbor list');
		
		# decide wether to create an all-to-all neighbor list or a binned one based on box size:
		default_r_buff = 0.8;
		
		mode = "binned";
		
		box = globals.particle_data.getBox();
		min_width_for_bin = (default_r_buff + r_cut)*3.0;
		if (box.xhi - box.xlo) < min_width_for_bin or (box.yhi - box.ylo) < min_width_for_bin or (box.zhi - box.zlo) < min_width_for_bin:
			print "Notice: Forcing use of O(N^2) neighbor list due to small box dimensions";
			mode = "nsq";
		
		# create the C++ mirror class
		if globals.particle_data.getExecConf().exec_mode == hoomd.ExecutionConfiguration.executionMode.CPU:
			if mode == "binned":
				self.cpp_nlist = hoomd.BinnedNeighborList(globals.particle_data, r_cut, default_r_buff)
			elif mode == "nsq":
				self.cpp_nlist = hoomd.NeighborList(globals.particle_data, r_cut, default_r_buff)
			else:
				print >> sys.stderr, "\n***Error! Invalid neighbor list mode\n";
				raise RuntimeError("Error creating neighbor list");
		elif globals.particle_data.getExecConf().exec_mode == hoomd.ExecutionConfiguration.executionMode.GPU:
			if mode == "binned":
				self.cpp_nlist = hoomd.BinnedNeighborListGPU(globals.particle_data, r_cut, default_r_buff)
			elif mode == "nsq":
				self.cpp_nlist = hoomd.NeighborListNsqGPU(globals.particle_data, r_cut, default_r_buff)
			else:
				print >> sys.stderr, "\n***Error! Invalid neighbor list mode\n";
				raise RuntimeError("Error creating neighbor list");
		else:
			print >> sys.stderr, "\n***Error! Invalid execution mode\n";
			raise RuntimeError("Error creating neighbor list");
			
		self.cpp_nlist.setEvery(1);
		self.cpp_nlist.copyExclusionsFromBonds();
		
		globals.system.addCompute(self.cpp_nlist, "auto_nlist");
		
		# save the parameters we set
		self.r_cut = r_cut;
		self.r_buff = default_r_buff;
		
	## Change neighbor list parameters
	# 
	# \param r_buff (if set) changes the buffer radius around the cutoff
	# \param check_period (if set) changes the period (in time steps) between checks to see if the neighbor list needs updating
	# 
	# set_params() changes one or more parameters of the neighbor list. \a r_buff and \a check_period 
	# can have a significant effect on performance. As \a r_buff is made larger, the neighbor list needs
	# to be updated less often, but more particles are included leading to slower %force computations. 
	# Smaller values of \a r_buff lead to faster %force computation, but more often neighbor list updates,
	# slowing overall performance again. The sweet spot for the best performance needs to be found by 
	# experimentation. The default of \a r_buff = 0.8 works well in practice for Lennard-Jones liquid
	# simulations.
	#
	# As \a r_buff is changed, \a check_period must be changed correspondingly. The neighbor list is updated
	# no sooner than \a check_period time steps after the last %update. If \a check_period is set too high,
	# the neighbor list may not be updated when it needs to be. 
	#
	# For safety, the default check_period is 1 to ensure that the neighbor list is always updated when it
	# needs to be. Increasing this to an appropriate value for your simulation can lead to performance gains
	# of approximately 2 percent.
	#
	# \a check_period should be set so that no particle
	# moves a distance more than \a r_buff/2.0 during a the \a check_period. If this occurs, a \b dangerous
	# \b build is counted and printed in the neighbor list statistics at the end of a run().
	#
	# A single global neighbor list is created for the entire simulation. Change parameters by using
	# the built-in variable \b %nlist.
	#
	# \b Examples:
	# \code 
	# nlist.set_params(r_buff = 0.9)
	# nlist.set_params(check_period = 11)
	# nlist.set_params(r_buff = 0.7, check_period = 4)
	# \endcode
	def set_params(self, r_buff=None, check_period=None):
		util.print_status_line();
		
		if self.cpp_nlist == None:
			print >> sys.stderr, "\nBug in hoomd_script: cpp_nlist not set, please report\n";
			raise RuntimeError('Error setting neighbor list parameters');
		
		# update the parameters
		if r_buff != None:
			self.cpp_nlist.setRCut(self.r_cut, r_buff);
			self.r_buff = r_buff;
			
		if check_period != None:
			self.cpp_nlist.setEvery(check_period);
			
	## Benchmarks the neighbor list computation
	# \param n Number of iterations to average the benchmark over
	#
	# \b Examples:
	# \code
	# t = nlist.benchmark(n = 100)
	# \endcode
	#
	# The value returned by benchmark() is the average time to perform the neighbor list 
	# computation, in milliseconds. The benchmark is performed by taking the current
	# positions of all particles in the simulation and repeatedly calculating the neighbor list.
	# Thus, you can benchmark different situations as you need to by simply 
	# running a simulation to achieve the desired state before running benchmark().
	#
	# \note
	# There is, however, one subtle side effect. If the benchmark() command is run 
	# directly after the particle data is intialized with an init command, then the 
	# results of the benchmark will not be typical of the time needed during the actual
	# simulation. Particles are not reorederd to improve cache performance until at least
	# one time step is performed. Executing run(1) before the benchmark will solve this problem.
	#
	def benchmark(self, n):
		# check that we have been initialized properly
		if self.cpp_nlist == None:
			print >> sys.stderr, "\nBug in hoomd_script: cpp_nlist not set, please report\n";
			raise RuntimeError('Error benchmarking neighbor list');
		
		# run the benchmark
		return self.cpp_nlist.benchmark(int(n))
			
## \internal
# \brief Creates the global neighbor list
# \details
# \param r_cut Cutoff radius to set
# If no neighbor list has been created, create one. If there is one, increase its r_cut value
# to be the maximum of the current and the one specified here
def _update_global_nlist(r_cut):
	# check to see if we need to create the neighbor list
	if globals.neighbor_list == None:
		globals.neighbor_list = nlist(r_cut);
		# set the global neighbor list using the evil import __main__ trick to provide the user with a default variable
		import __main__;
		__main__.nlist = globals.neighbor_list;
		
	else:
		# otherwise, we need to update r_cut
		new_r_cut = max(r_cut, globals.neighbor_list.r_cut);
		globals.neighbor_list.r_cut = new_r_cut;
		globals.neighbor_list.cpp_nlist.setRCut(new_r_cut, globals.neighbor_list.r_buff);
	
	return globals.neighbor_list;
	
	
## Lennard-Jones %pair %force
#
# The command pair.lj specifies that a Lennard-Jones type %pair %force should be added to every
# non-bonded particle %pair in the simulation.
#
# The %force \f$ \vec{F}\f$ is
# \f{eqnarray*}
#	\vec{F}  = & -\nabla V(r) & r < r_{\mathrm{cut}}		\\
#			 = & 0 			& r \ge r_{\mathrm{cut}}	\\
#	\f}
# where \f$ V(r) \f$ is chosen by a mode switch (see set_params())
# \f{eqnarray*}
#	V(r)  = & V_{\mathrm{LJ}}(r) & \mathrm{mode\ is\ no\_shift} \\
#			 = & V_{\mathrm{LJ}}(r) - V_{\mathrm{LJ}}(r_{\mathrm{cut}}) & \mathrm{mode\ is\ shift}	\\
#			 = & S(r) \cdot V_{\mathrm{LJ}}(r) & \mathrm{mode\ is\ xplor}	\\
#	\f}
# , \f$ S(r) \f$ is the XPLOR smoothing function
# \f{eqnarray*} 
#	S(r) = & 1 & r < r_{\mathrm{on}} \\
#	= & \frac{(r_{\mathrm{cut}}^2 - r^2)^2 \cdot (r_{\mathrm{cut}}^2 + 2r^2 - 3r_{\mathrm{on}}^2)}{(r_{\mathrm{cut}}^2 - r_{\mathrm{on}}^2)^3} & r_{\mathrm{on}} \le r \le r_{\mathrm{cut}} \\
#  = & 0 & r > r_{\mathrm{cut}} \\
# \f}
# , with \f$ r_{\mathrm{on}} = \lambda \cdot r_{\mathrm{cut}} \f$,
# \f[ V_{\mathrm{LJ}}(r) = 4 \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{12} - 
# 									\alpha \left( \frac{\sigma}{r} \right)^{6} \right] \f]
# ,
# and \f$ \vec{r} \f$ is the vector pointing from one particle to the other in the %pair.
#
# The following coefficients must be set per unique %pair of particle types. See pair or 
# the \ref page_quick_start for information on how to set coefficients.
# - \f$ \varepsilon \f$ - \c epsilon
# - \f$ \sigma \f$ - \c sigma
# - \f$ \alpha \f$ - \c alpha
#
# The following parameters are set globally via set_params()
# - mode - mode (default = "no_shift)
# - \f$ \lambda \f$ - \c xplor_factor (default = 2.0/3.0)
#
# \b Example:
# \code
# lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, alpha=1.0)
# \endcode
#
# The cuttoff radius \f$ r_{\mathrm{cut}} \f$ is set once when pair.lj is specified (see __init__())
class lj(force._force):
	## Specify the Lennard-Jones %pair %force
	#
	# \param r_cut Cutoff radius (see documentation above)
	#
	# \b Example:
	# \code
	# lj = pair.lj(r_cut=3.0)
	# lj.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0, alpha=1.0)
	# \endcode
	#
	# \note Pair coefficients for all type pairs in the simulation must be
	# set before it can be started with run()
	def __init__(self, r_cut):
		util.print_status_line();
		
		# initialize the base class
		force._force.__init__(self);
		
		# update the neighbor list
		neighbor_list = _update_global_nlist(r_cut);
		
		# create the c++ mirror class
		if globals.particle_data.getExecConf().exec_mode == hoomd.ExecutionConfiguration.executionMode.CPU:
			self.cpp_force = hoomd.LJForceCompute(globals.particle_data, neighbor_list.cpp_nlist, r_cut);
		elif globals.particle_data.getExecConf().exec_mode == hoomd.ExecutionConfiguration.executionMode.GPU:
			neighbor_list.cpp_nlist.setStorageMode(hoomd.NeighborList.storageMode.full);
			self.cpp_force = hoomd.LJForceComputeGPU(globals.particle_data, neighbor_list.cpp_nlist, r_cut);
		else:
			print >> sys.stderr, "\n***Error! Invalid execution mode\n";
			raise RuntimeError("Error creating lj pair force");
			
			
		globals.system.addCompute(self.cpp_force, self.force_name);
		
		# setup the coefficent matrix
		self.pair_coeff = coeff();
		
	def update_coeffs(self):
		# check that the pair coefficents are valid
		if not self.pair_coeff.verify("epsilon", "sigma", "alpha"):
			print >> sys.stderr, "\n***Error: Not all pair coefficients are set in pair.lj\n";
			raise RuntimeError("Error updating pair coefficients");
		
		# set all the params
		ntypes = globals.particle_data.getNTypes();
		type_list = [];
		for i in xrange(0,ntypes):
			type_list.append(globals.particle_data.getNameByType(i));
		
		for i in xrange(0,ntypes):
			for j in xrange(i,ntypes):
				epsilon = self.pair_coeff.get(type_list[i], type_list[j], "epsilon");
				sigma = self.pair_coeff.get(type_list[i], type_list[j], "sigma");
				alpha = self.pair_coeff.get(type_list[i], type_list[j], "alpha");
				
				lj1 = 4.0 * epsilon * math.pow(sigma, 12.0);
				lj2 = alpha * 4.0 * epsilon * math.pow(sigma, 6.0);
				self.cpp_force.setParams(i, j, lj1, lj2);
				
	## Set parameters controlling the way forces are computed
	#
	# \param mode (if set) Set the mode with which potentials are handled at the cutoff
	# \param fraction (if set) Change the fraction of \f$ r_{\mathrm{cut}} \f$ at which the XPLOR smoothing starts (default is 2.0/3.0). Only applies of \a mode is set to "xplor"
	#
	# valid values for \a mode are: "none" (the default), "shift", and "xplor"
	#  - \b none - No shifting is performed and potentials are abruptly cut off
	#  - \b shift - A constant shift is applied to the entire potential so that it is 0 at the cutoff
	#  - \b xplor - A smoothing function is applied to gradually decrease both the force and potential to 0 at the cutoff
	# (see above for formulas and more information)
	#
	# \b Examples:
	# \code
	# lj.set_params(mode="shift")
	# lj.set_params(mode="no_shift")
	# lj.set_params(mode="xplor", xplor_factor = 0.5)
	# \endcode	
	# 
	def set_params(self, mode=None, fraction=None):
		util.print_status_line();
		
		if mode != None:
			if mode == "no_shift":
				self.cpp_force.setShiftMode(hoomd.LJForceCompute.energyShiftMode.no_shift)
			elif mode == "shift":
				self.cpp_force.setShiftMode(hoomd.LJForceCompute.energyShiftMode.shift)
			elif mode == "xplor":
				self.cpp_force.setShiftMode(hoomd.LJForceCompute.energyShiftMode.xplor)
			else:
				print >> sys.stderr, "\n***Error: invalid mode", mode, "\n";
				raise RuntimeError("Error setting lj parameters");
			
		if fraction != None:
			self.cpp_force.setXplorFraction(fraction);

## Gaussian %pair %force
#
# The command pair.gauss specifies that a Gaussian type %pair %force should be added to every
# non-bonded particle %pair in the simulation.
#
# The %force \f$ \vec{F}\f$ is
# \f{eqnarray*}
#	\vec{F}  = & -\nabla V(r) & r < r_{\mathrm{cut}}		\\
#			 = & 0 			& r \ge r_{\mathrm{cut}}	\\
#	\f}
# where \f$ V(r) \f$ is chosen by a mode switch (see set_params())
# \f{eqnarray*}
#	V(r)  = & V_{\mathrm{gauss}}(r) & \mathrm{mode\ is\ no\_shift} \\
#			 = & V_{\mathrm{gauss}}(r) - V_{\mathrm{gauss}}(r_{\mathrm{cut}}) & \mathrm{mode\ is\ shift}	\\
#	\f}
# ,
# \f[ V_{\mathrm{gauss}}(r) = \varepsilon \exp \left[ -\frac{1}{2}\left( \frac{r}{\sigma}^2 \right) \right] \f]
# ,
# and \f$ \vec{r} \f$ is the vector pointing from one particle to the other in the %pair.
#
# The following coefficients must be set per unique %pair of particle types. See pair or 
# the \ref page_quick_start for information on how to set coefficients.
# - \f$ \varepsilon \f$ - \c epsilon
# - \f$ \sigma \f$ - \c sigma
#
# The following parameters are set globally via set_params()
# - mode - mode (default = "no_shift")
#
# \b Example:
# \code
# gauss.pair_coeff.set('A', 'A', epsilon=1.0, sigma=0.5)
# \endcode
#
# The cutoff radius \f$ r_{\mathrm{cut}} \f$ is set once when pair.gauss is specified (see __init__())
class gauss(force._force):
	## Specify the Gaussian %pair %force
	#
	# \param r_cut Cutoff radius (see documentation above)
	#
	# \b Example:
	# \code
	# gauss = pair.gauss(r_cut=3.0)
	# gauss.pair_coeff.set('A', 'A', epsilon=1.0, sigma=0.5)
	# \endcode
	#
	# \note Pair coefficients for all type pairs in the simulation must be
	# set before it can be started with run()
	def __init__(self, r_cut):
		util.print_status_line();
		
		# initialize the base class
		force._force.__init__(self);
		
		# update the neighbor list
		neighbor_list = _update_global_nlist(r_cut);
		
		# create the c++ mirror class
		if globals.particle_data.getExecConf().exec_mode == hoomd.ExecutionConfiguration.executionMode.CPU:
			self.cpp_force = hoomd.GaussianForceCompute(globals.particle_data, neighbor_list.cpp_nlist, r_cut);
		elif globals.particle_data.getExecConf().exec_mode == hoomd.ExecutionConfiguration.executionMode.GPU:
			neighbor_list.cpp_nlist.setStorageMode(hoomd.NeighborList.storageMode.full);
			self.cpp_force = hoomd.GaussianForceGPU(globals.particle_data, neighbor_list.cpp_nlist, r_cut);
		else:
			print >> sys.stderr, "\n***Error! Invalid execution mode\n";
			raise RuntimeError("Error creating gauss pair force");
			
			
		globals.system.addCompute(self.cpp_force, self.force_name);
		
		# setup the coefficent matrix
		self.pair_coeff = coeff();
		
	def update_coeffs(self):
		# check that the pair coefficents are valid
		if not self.pair_coeff.verify("epsilon", "sigma"):
			print >> sys.stderr, "\n***Error: Not all pair coefficients are set in pair.gauss\n";
			raise RuntimeError("Error updating pair coefficients");
		
		# set all the params
		ntypes = globals.particle_data.getNTypes();
		type_list = [];
		for i in xrange(0,ntypes):
			type_list.append(globals.particle_data.getNameByType(i));
		
		for i in xrange(0,ntypes):
			for j in xrange(i,ntypes):
				epsilon = self.pair_coeff.get(type_list[i], type_list[j], "epsilon");
				sigma = self.pair_coeff.get(type_list[i], type_list[j], "sigma");
				
				self.cpp_force.setParams(i, j, epsilon, sigma);
				
	## Set parameters controlling the way forces are computed
	#
	# \param mode (if set) Set the mode with which potentials are handled at the cutoff
	#
	# valid values for \a mode are: "none" (the default), and "shift"
	#  - \b none - No shifting is performed and potentials are abruptly cut off
	#  - \b shift - A constant shift is applied to the entire potential so that it is 0 at the cutoff
	# (see above for formulas and more information)
	#
	# \b Examples:
	# \code
	# gauss.set_params(mode="shift")
	# gauss.set_params(mode="no_shift")
	# \endcode	
	#
	def set_params(self, mode=None):
		util.print_status_line();
		
		if mode != None:
			if mode == "no_shift":
				self.cpp_force.setShiftMode(hoomd.GaussianForceCompute.energyShiftMode.no_shift)
			elif mode == "shift":
				self.cpp_force.setShiftMode(hoomd.GaussianForceCompute.energyShiftMode.shift)
			else:
				print >> sys.stderr, "\n***Error: invalid mode", mode, "\n";
				raise RuntimeError("Error setting gauss parameters");

		
## Yukawa %pair %force
#
# The command pair.yukawa specifies that a Yukawa type %pair %force should be added to every
# non-bonded particle %pair in the simulation.
#
# The %force \f$ \vec{F}\f$ is
# \f{eqnarray*}
#	\vec{F}  = & -\nabla V(r) & r < r_{\mathrm{cut}}		\\
#			 = & 0 			& r \ge r_{\mathrm{cut}}	\\
#	\f}
# where
# \f[ V(r) = 4 \varepsilon \frac{ e^{\kappa r}}{r} \f]
# and \f$ \vec{r} \f$ is the vector pointing from one particle to the other in the %pair.
#
# The following coefficient must be set per unique %pair of particle types. See pair or 
# the \ref page_quick_start for information on how to set coefficients.
# - \f$ \varepsilon \f$ - \c epsilon
#
# \b Example:
# \code
# yukawa.pair_coeff.set('A', 'A', epsilon=1.0 )
# \endcode
#
# The cutoff radius \f$ r_{\mathrm{cut}} \f$ is set once when pair.yukawa is specified (see __init__())
# class yukawa(force._force):
# 	## Specify the Yukawa %pair %force
# 	#
# 	# \param r_cut Cutoff radius (see documentation above)
# 	# \param kappa Screening Length
# 	#
# 	# \b Example:
# 	# \code
# 	# myforce = pair.yukawa(r_cut=3.0, kappa = 5.0)
# 	# myforce.pair_coeff.set('A', 'A', epsilon=1.0)
# 	# \endcode
# 	#
# 	# \note Pair coefficients for all type pairs in the simulation must be
# 	# set before it can be started with run()
# 	def __init__(self, r_cut, kappa):
# 		util.print_status_line();
# 		
# 		# initialize the base class
# 		force._force.__init__(self);
# 		
# 		# update the neighbor list
# 		neighbor_list = _update_global_nlist(r_cut);
# 		
# 		# create the c++ mirror class
# 		if globals.particle_data.getExecConf().exec_mode == hoomd.ExecutionConfiguration.executionMode.CPU:
# 			self.cpp_force = hoomd.YukawaForceCompute(globals.particle_data, neighbor_list.cpp_nlist, r_cut, kappa);
# 		elif globals.particle_data.getExecConf().exec_mode == hoomd.ExecutionConfiguration.executionMode.GPU:
# 			neighbor_list.cpp_nlist.setStorageMode(hoomd.NeighborList.storageMode.full);
# 			self.cpp_force = hoomd.YukawaForceComputeGPU(globals.particle_data, neighbor_list.cpp_nlist, r_cut, kappa);
# 		else:
# 			print >> sys.stderr, "\n***Error! Invalid execution mode\n";
# 			raise RuntimeError("Error creating yukawa pair force");
# 			
# 			
# 		globals.system.addCompute(self.cpp_force, self.force_name);
# 		
# 		# setup the coefficent matrix
# 		self.pair_coeff = coeff();
# 		
# 	def update_coeffs(self):
# 		# check that the pair coefficents are valid
# 		if not self.pair_coeff.verify("epsilon"):
# 			print >> sys.stderr, "\n***Error: Not all pair coefficients are set in pair.yukawa\n";
# 			raise RuntimeError("Error updating pair coefficients");
# 		
# 		# set all the params
# 		ntypes = globals.particle_data.getNTypes();
# 		type_list = [];
# 		for i in xrange(0,ntypes):
# 			type_list.append(globals.particle_data.getNameByType(i));
# 		
# 		for i in xrange(0,ntypes):
# 			for j in xrange(i,ntypes):
# 				epsilon = self.pair_coeff.get(type_list[i], type_list[j], "epsilon");
# 				
# 				self.cpp_force.setParams(i, j, epsilon);


