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

# $Id: bond.py 1117 2008-08-04 15:44:35Z joaander $
# $URL: https://svn2.assembla.com/svn/hoomd/trunk/src/lib/python-module/hoomd_script/bond.py $

import force;
import globals;
import hoomd;
import sys;
import math;
import util;

## \package hoomd_script.wall
# \brief Commands that specify %wall forces
#
# Walls can add forces to any particles within a certain distance of the wall. Walls are created
# when an input XML file is read (read.xml).
#
# By themselves, walls that have been specified in an input file do nothing. Only when you 
# specify a wall force (i.e. wall.lj), are forces actually applied between the wall and the
# particle.

## Lennard-Jones %wall %force
#
# The command wall.lj specifies that a Lennard-Jones type %wall %force should be added to every
# particle in the simulation.
#
# The %force \f$ \vec{F}\f$ is
# \f{eqnarray*}
#	\vec{F}  = & -\nabla V(r) & r < r_{\mathrm{cut}}		\\
#			 = & 0 			& r \ge r_{\mathrm{cut}}	\\
#	\f}
# where
# \f[ V(r) = 4 \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{12} - 
# 									\alpha \left( \frac{\sigma}{r} \right)^{6} \right] \f]
# and \f$ \vec{r} \f$ is the vector pointing from the %wall to the particle parallel to the wall's normal.
#
# The following coefficients must be set for each particle type using set_coeff(). 
# - \f$ \varepsilon \f$ - \c epsilon
# - \f$ \sigma \f$ - \c sigma
# - \f$ \alpha \f$ - \c alpha
#
# \b Example:
# \code
# lj.set_coeff('A', epsilon=1.0, sigma=1.0, alpha=1.0)
# \endcode
#
# The cutoff radius \f$ r_{\mathrm{cut}} \f$ is set once when wall.lj is specified (see __init__())
class lj(force._force):
	## Specify the Lennard-Jones %wall %force
	#
	# \param r_cut Cutoff radius
	#
	# \b Example:
	# \code
	# lj_wall = wall.lj(r_cut=3.0);
	# \endcode
	#
	# \note Coefficients must be set with set_coeff() before the simulation can be run().
	def __init__(self, r_cut):
		util.print_status_line();
		
		# initialize the base class
		force._force.__init__(self);
		
		# create the c++ mirror class
		#if globals.particle_data.getExecConf().exec_mode == hoomd.ExecutionConfiguration.executionMode.CPU:
		self.cpp_force = hoomd.LJWallForceCompute(globals.particle_data, r_cut);
		#elif globals.particle_data.getExecConf().exec_mode == hoomd.ExecutionConfiguration.executionMode.GPU:
		#	self.cpp_force = hoomd.LJWallForceComputeGPU(globals.particle_data, f_cut);
		#else:
		#	print >> sys.stderr, "\n***Error! Invalid execution mode\n";
		#	raise RuntimeError("Error creating wall.lj forces");
		
		# variable for tracking which particle type coefficients have been set
		self.particle_types_set = [];
		
		globals.system.addCompute(self.cpp_force, self.force_name);
		
	## Sets the particle-wall interaction coefficients for a particular particle type
	#
	# \param particle_type Particle type to set coefficients for
	# \param epsilon Coefficient \f$ \varepsilon \f$ in the %force
	# \param sigma Coefficient \f$ \sigma \f$ in the %force
	# \param alpha Coefficient \f$ \alpha \f$ in the %force
	#
	# Using set_coeff() requires that the specified %wall %force has been saved in a variable. i.e.
	# \code
	# lj_wall = wall.lj(r_cut=3.0)
	# \endcode
	#
	# \b Examples:
	# \code
	# lj_wall.set_coeff('A', epsilon=1.0, sigma=1.0, alpha=1.0)
	# lj_wall.set_coeff('B', epsilon=1.0, sigma=2.0, alpha=0.0)
	# \endcode
	#
	# The coefficients for every particle type in the simulation must be set 
	# before the run() can be started.
	def set_coeff(self, particle_type, epsilon, sigma, alpha):
		util.print_status_line();
		
		# calculate the parameters
		lj1 = 4.0 * epsilon * math.pow(sigma, 12.0);
		lj2 = alpha * 4.0 * epsilon * math.pow(sigma, 6.0);
		# set the parameters for the appropriate type
		self.cpp_force.setParams(globals.particle_data.getTypeByName(particle_type), lj1, lj2);
		
		# track which particle types we have set
		if not particle_type in self.particle_types_set:
			self.particle_types_set.append(particle_type);
		
	def update_coeffs(self):
		# get a list of all particle types in the simulation
		ntypes = globals.particle_data.getNTypes();
		type_list = [];
		for i in xrange(0,ntypes):
			type_list.append(globals.particle_data.getNameByType(i));
			
		# check to see if all particle types have been set
		for cur_type in type_list:
			if not cur_type in self.particle_types_set:
				print >> sys.stderr, "\n***Error:", cur_type, " coefficients missing in wall.lj\n";
				raise RuntimeError("Error updating coefficients");