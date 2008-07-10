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
# \brief Commands that create forces between particle pairs
#
# Details

import globals;
import force;
import hoomd;
import math;

## Defines pair coefficients
# 
# All pair forces use coeff to specify the coefficients between different
# pairs of particles indexed by type. The set of pair coefficients is a symmetric
# matrix defined over all possible pairs of particle types.
#
# TODO: more documentation
# The simplest way is to set coefficients on the pre-defined matrix in the
# respective pair class.
#
# A coeff matrix can be created and all parameters set even without having the
# system initialized yet. This means that you could specify all of your pair 
# coefficients in one file and load them with the python import command.
# TODO: example
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
	
	## Sets parameters for one type pair
	# \param a First particle type in the pair
	# \param b Second particle type in the pair
	# \param coeffs Named coefficients (see below for examples)
	#
	# Calling set results in one or more parameters being set for a single type pair.
	# Particle types are identified by name, and parameters are also added by name. 
	# Which parameters you need to specify depends on the pair force you are setting
	# these coefficients for, see the corresponding documentation.
	#
	# All possible type pairs as defined in the simulation box must be specified before
	# executing run(). You will recieve an error if you fail to do so. It is not an error,
	# however, to specify coefficients for particle types that do not exist in the simulation.
	# This can be useful in defining a force field for many different types of particles even
	# if some simulations only include a subset.
	#
	# There is no need to specify coeffiencs for both pairs 'A','B' and 'B','A'. Specifying
	# only one is sufficient.
	#
	# \b Examples:<br>
	# coeff.set('A', 'A', epsilon=1.0, sigma=1.0)\n
	# coeff.set('B', 'B', epsilon=2.0, sigma=1.0)\n
	# coeff.set('A', 'B', epsilon=1.5, sigma=1.0)
	#
	# \note Single parameters can be updated. If both epsilon and sigma have already been 
	# set for a type pair, then executing coeff.set('A', 'B', epsilon=1.1) will update 
	# the value of epsilon and leave sigma as it was previously set.
	def set(self, a, b, **coeffs):
		print "coeff.set(", a, ",", b, ",", coeffs, ")";
		
		# create the pair if it hasn't been created it
		if (not (a,b) in self.values) and (not (b,a) in self.values):
			self.values[(a,b)] = {};
			
		# Find the pair to update
		if (a,b) in self.values:
			cur_pair = (a,b);
		elif (b,a) in self.values:
			cur_pair = (b,a);
		else:
			print "Bug detected in pair.coeff. Please report"
			raise RuntimeError("Error setting pair coeff");
		
		# update each of the values provided
		if len(coeffs) == 0:
			print "No coefficents specified";
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
			print "Error: Cannot verify pair coefficients before initialization";
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
					print "Type pair", (a,b), "not found in pair coeff!"
					valid = False;
					continue;
				
				# verify that all required values are set by counting the matches
				count = 0;
				for coeff_name in self.values[cur_pair].keys():
					if not coeff_name in required_coeffs:
						print "Possible typo? Pair coeff", coeff_name, "is specified for pair", (a,b), ", but is not used by the pair force";
					else:
						count += 1;
				
				if count != len(required_coeffs):
					print "Type pair", (a,b), "is missing required coefficients";
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
			print "Bug detected in pair.coeff. Please report"
			raise RuntimeError("Error setting pair coeff");	
		
		return self.values[cur_pair][coeff_name];
		
		
## Interface for controlling neighbor list parameters
#
# A neighbor list does not need to be created by you. One will be automatically
# created whenever a pair force is specified. The cuttoff radius is set to the
# maximum of that set for all defined pair forces. 
class nlist:
	## \internal
	# \brief Constructs a neighbor list
	# \details
	# \param self Python required instance variable
	# \param r_cut Cutoff radius
	def __init__(self, r_cut):
		# check if initialization has occured
		if globals.system == None:
			print "Error: Cannot create neighbor list before initialization";
			raise RuntimeError('Error creating neighbor list');
		
		# create the C++ mirror class
		self.cpp_nlist = hoomd.BinnedNeighborList(globals.particle_data, r_cut, 0.8)
		self.cpp_nlist.setEvery(10);
		
		# set the exclusions (TEMPORARY HACK for bonds)
		if globals.initializer:
			globals.initializer.setupNeighborListExclusions(self.cpp_nlist);
		
		# save the parameters we set
		self.r_cut = r_cut;
		self.r_buff = 0.8;
		
	## Change neighbor list parameters
	# 
	# \param r_buff (if set) changes the buffer radius around the cuttof
	# \param check_period (if set) changes the period (in time steps) between checks to see if the neighbor list needs updating
	# 
	# set_params changes one or more parameters of the neighbor list. \a r_buff and \a check_period 
	# can have a significant effect on performance. As \a r_buff is made larger, the neighbor list needs
	# to be updated less often, but more particles are included leading to slower force computations. 
	# Smaller values of \a r_buff lead to faster force computation, but more often neighbor list updates,
	# slowing overall perforamnce again. The sweet spot for the best performance needs to be found by 
	# experimentation. The default of \a r_buff = 0.8 works well in practice for Lennard-Jones liquid
	# simulations.
	#
	# As # r_buff is changed, \a check_period must be changed correspondingly. The neighbor list is updated
	# no sooner than \a check_period time steps after the last update. If \a check_period is set too high,
	# the neighbor list may not be updated when it needs to be. The default of 10 is appropriate for 
	# Lennard-Jones liquid simulations at reduced T=1.2. \a check_period should be set so that no particle
	# moves a distance no more than \a r_buff/2.0 during a the \a check_period. If this occurs, a \b dangerous
	# \b build is counted and printed in the neighbor list statistics at the end of a run().
	#
	# A single global neighbor list is created for the entire simulation. Change parameters by using
	# the built-in variable \b nlist.
	#
	# \b Examples: <br>
	# nlist.set_params(r_buff = 0.9)<br>
	# nlist.set_params(check_period = 11)<br>
	# nlist.set_params(r_buff = 0.7, check_period = 4)
	def set_params(r_buff=None, check_period=None):
		print "nlist.set_params(r_buff =", r_buff, " , check_period =", check_period, ")";
		
		if self.cpp_nlist == None:
			print "Bug in hoomd_script: cpp_nlist not set, please report";
			raise RuntimeError('Error setting neighbor list parameters');
		
		# update the parameters
		if r_buff != None:
			self.cpp_nlist.setRCut(self.r_cut, r_buff);
			self.r_buff = r_buff;
			
		if check_period != None:
			self.cpp_nlist.setEvery(check_period);
			
## \internal
# \brief Creates the global neighbor list
# \details
# \param r_cut Cuttoff radius to set
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
	
	
## Lennard-Jones pair forces
#
# TODO: document
class lj(force._force):
	## Specify the Lennard-Jones pair force
	#
	# TODO: document me
	def __init__(self, r_cut):
		print "pair.lj(r_cut =", r_cut, ")";
		
		# initialize the base class
		force._force.__init__(self);
		
		# update the neighbor list
		neighbor_list = _update_global_nlist(r_cut);
		
		# create the c++ mirror class
		self.cpp_force = hoomd.LJForceCompute(globals.particle_data, neighbor_list.cpp_nlist, r_cut);
		globals.system.addCompute(self.cpp_force, self.force_name);
		
		# setup the coefficent matrix
		self.pair_coeff = coeff();
		
	def update_coeffs(self):
		print "TODO: set default coeffs";
		
		# check that the pair coefficents are valid
		if not self.pair_coeff.verify("epsilon", "sigma", "alpha"):
			print "***Error: Not all pair coefficients are set in pair.lj\n";
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
				
				lj1 = 48.0 * epsilon * math.pow(sigma, 12.0);
				lj2 = alpha * 24.0 * epsilon * math.pow(sigma, 6.0);
				self.cpp_force.setParams(i, j, lj1, lj2);
		
