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
	def get(a, b, coeff_name):
		# Find the pair to update
		if (a,b) in self.values:
			cur_pair = (a,b);
		elif (b,a) in self.values:
			cur_pair = (b,a);
		else:
			print "Bug detected in pair.coeff. Please report"
			raise RuntimeError("Error setting pair coeff");	
		
		return self.values[cur_pair][coeff_name];
		
		
