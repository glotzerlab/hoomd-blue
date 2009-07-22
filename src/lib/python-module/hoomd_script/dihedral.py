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
import util;
import tune;

import math;
import sys;

## \package hoomd_script.dihedral
# \brief Commands that specify %dihedral forces
#
# Dihedrals add forces between specified quadruplets of particles and are typically used to 
# model rotation about chemical bonds. Dihedrals between particles are set when an input XML file is read
# (init.read_xml) or when an another initializer creates them (like init.create_random_polymers)
#
# By themselves, dihedrals that have been specified in an input file do nothing. Only when you 
# specify an dihedral force (i.e. dihedral.harmonic), are forces actually calculated between the 
# listed particles.

## Harmonic %dihedral forces
#
# The command dihedral.harmonic specifies a %harmonic dihedral potential energy between every defined 
# quadruplet of particles in the simulation. 
# \f[ V(r) = \frac{1}{2}k \left( 1 + d cos\left(n * \phi(r) \right) \right) \f]
# where \f$ \phi \f$ is angle between two sides of the dihedral
#
# Coefficients \f$ k \f$, \f$ d \f$, \f$ n \f$ and  must be set for each type of %dihedral in the simulation using
# set_coeff().
#
# \note Specifying the dihedral.harmonic command when no dihedrals are defined in the simulation results in an error.
class harmonic(force._force):
	## Specify the %harmonic %dihedral %force
	#
	# \b Example:
	# \code
	# harmonic = dihedral.harmonic()
	# \endcode
	def __init__(self):
		util.print_status_line();
		# check that some dihedrals are defined
		if globals.system_definition.getDihedralData().getNumDihedrals() == 0:
			print >> sys.stderr, "\n***Error! No dihedrals are defined.\n";
			raise RuntimeError("Error creating dihedral forces");		
		
		# initialize the base class
		force._force.__init__(self);
		
		# create the c++ mirror class
		if globals.system_definition.getParticleData().getExecConf().exec_mode == hoomd.ExecutionConfiguration.executionMode.CPU:
			self.cpp_force = hoomd.HarmonicDihedralForceCompute(globals.system_definition);
		elif globals.system_definition.getParticleData().getExecConf().exec_mode == hoomd.ExecutionConfiguration.executionMode.GPU:
			self.cpp_force = hoomd.HarmonicDihedralForceComputeGPU(globals.system_definition);
			self.cpp_force.setBlockSize(tune._get_optimal_block_size('dihedral.harmonic'));
		else:
			print >> sys.stderr, "\n***Error! Invalid execution mode\n";
			raise RuntimeError("Error creating dihedral forces");

		globals.dihedral_compute = self.cpp_force;

		globals.system.addCompute(self.cpp_force, self.force_name);
		
		# variable for tracking which dihedral type coefficients have been set
		self.dihedral_types_set = [];
	
	## Sets the %harmonic %dihedral coefficients for a particular %dihedral type
	#
	# \param dihedral_type Dihedral type to set coefficients for
	# \param k Coefficient \f$ k \f$ in the %force
	# \param d Coefficient \f$ d \f$ in the %force, and must be either -1 or 1
	# \param n Coefficient \f$ n \f$ in the %force
        #
	# Using set_coeff() requires that the specified %dihedral %force has been saved in a variable. i.e.
	# \code
	# harmonic = dihedral.harmonic()
	# \endcode
	#
	# \b Examples:
	# \code
	# harmonic.set_coeff('phi-ang', k=30.0, d=-1, n=3)
	# harmonic.set_coeff('psi-ang', k=100.0, d=1, n=4)
	# \endcode
	#
	# The coefficients for every %dihedral type in the simulation must be set 
	# before the run() can be started.
	def set_coeff(self, dihedral_type, k, d, n):
		util.print_status_line();
		
		# set the parameters for the appropriate type
		self.cpp_force.setParams(globals.system_definition.getDihedralData().getTypeByName(dihedral_type), k, d, n);
		
		# track which particle types we have set
		if not dihedral_type in self.dihedral_types_set:
			self.dihedral_types_set.append(dihedral_type);
		
	def update_coeffs(self):
		# get a list of all dihedral types in the simulation
		ntypes = globals.system_definition.getDihedralData().getNDihedralTypes();
		type_list = [];
		for i in xrange(0,ntypes):
			type_list.append(globals.system_definition.getDihedralData().getNameByType(i));
			
		# check to see if all particle types have been set
		for cur_type in type_list:
			if not cur_type in self.dihedral_types_set:
				print >> sys.stderr, "\n***Error:", cur_type, " coefficients missing in dihedral.harmonic\n";
				raise RuntimeError("Error updating coefficients");

