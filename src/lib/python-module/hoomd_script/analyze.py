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

import hoomd;
import globals;
import sys;
import util;

## \package hoomd_script.analyze
# \brief Commands that %analyze the system and provide some output
#
# An analyzer examines the system state in some way every \a period time steps and generates
# some form of output based on the analysis. Check the documentation for individual analyzers 
# to see what they do.

## \internal
# \brief Base class for analyzers
#
# An analyzer in hoomd_script reflects an Analyzer in c++. It is responsible
# for all high-level management that happens behind the scenes for hoomd_script
# writers. 1) The instance of the c++ analyzer itself is tracked and added to the
# System 2) methods are provided for disabling the analyzer and changing the 
# period which the system calls it
class _analyzer:
	## \internal
	# \brief Constructs the analyzer
	#
	# Initializes the cpp_analyzer to None.
	# Assigns a name to the analyzer in analyzer_name;
	def __init__(self):
		# check if initialization has occured
		if globals.system == None:
			print >> sys.stderr, "\n***Error! Cannot create analyzer before initialization\n";
			raise RuntimeError('Error creating analyzer');
		
		self.cpp_analyzer = None;

		# increment the id counter
		id = _analyzer.cur_id;
		_analyzer.cur_id += 1;
		
		self.analyzer_name = "analyzer%d" % (id);
		self.enabled = True;

	## \var enabled
	# \internal
	# \brief True if the analyzer is enabled

	## \var cpp_analyzer
	# \internal
	# \brief Stores the C++ side Analyzer managed by this class
	
	## \var analyzer_name
	# \internal
	# \brief The Analyzer's name as it is assigned to the System

	## \var prev_period
	# \internal
	# \brief Saved period retrived when an analyzer is disabled: used to set the period when re-enabled

	## Disables the analyzer
	#
	# \b Examples:
	# \code
	# analyzer.disable()
	# \endcode
	#
	# Executing the disable command will remove the analyzer from the system.
	# Any run() command executed after disabling an analyzer will not use that 
	# analyzer during the simulation. A disabled analyzer can be re-enabled
	# with enable()
	#
	# To use this command, you must have saved the analyzer in a variable, as 
	# shown in this example:
	# \code
	# analyzer = analyzer.some_analyzer()
	# # ... later in the script
	# analyzer.disable()
	# \endcode
	def disable(self):
		util.print_status_line();
		
		# check that we have been initialized properly
		if self.cpp_analyzer == None:
			print >> sys.stderr, "\nBug in hoomd_script: cpp_analyzer not set, please report\n";
			raise RuntimeError('Error disabling analyzer');
			
		# check if we are already disabled
		if not self.enabled:
			print "***Warning! Ignoring command to disable an analyzer that is already disabled";
			return;
		
		self.prev_period = globals.system.getAnalyzerPeriod(self.analyzer_name);
		globals.system.removeAnalyzer(self.analyzer_name);
		self.enabled = False;

	## Enables the analyzer
	#
	# \b Examples:
	# \code
	# analyzer.enable()
	# \endcode
	#
	# See disable() for a detailed description.
	def enable(self):
		util.print_status_line();
		
		# check that we have been initialized properly
		if self.cpp_analyzer == None:
			print >> sys.stderr, "\nBug in hoomd_script: cpp_analyzer not set, please report\n";
			raise RuntimeError('Error disabling analyzer');
			
		# check if we are already disabled
		if self.enabled:
			print "***Warning! Ignoring command to enable an analyzer that is already enabled";
			return;
			
		globals.system.addAnalyzer(self.cpp_analyzer, self.analyzer_name, self.prev_period);
		self.enabled = True;
		
	## Changes the period between analyzer executions
	#
	# \param period New period to set
	#
	# \b Examples:
	# \code
	# analyzer.set_period(100);
	# analyzer.set_period(1);
	# \endcode
	#
	# While the simulation is \ref run() "running", the action of each analyzer
	# is executed every \a period time steps.
	#
	# To use this command, you must have saved the analyzer in a variable, as 
	# shown in this example:
	# \code
	# analyzer = analyze.some_analyzer()
	# # ... later in the script
	# analyzer.set_period(10)
	# \endcode
	def set_period(self, period):
		util.print_status_line();
		
		if self.enabled:
			globals.system.setAnalyzerPeriod(self.analyzer_name, period);
		else:
			self.prev_period = period;

# set default counter
_analyzer.cur_id = 0;


## Sends simulation snapshots to VMD in real-time
#
# analyze.imd listens on a specified TCP/IP port for connections from VMD.
# Once that connection is established, it begins transmitting simulation snapshots
# to VMD every \a period time steps.
#
# To connect to a simulation running on the local host, issue the command
# \code
# imd connect localhost 54321
# \endcode
# in the VMD command window (where 54321 is replaced with the port number you specify for
# analyze.imd
#
# \sa \ref page_example_scripts
class imd(_analyzer):
	## Initialize the IMD interface
	#
	# \param port TCP/IP port to listen on
	# \param period Number of time steps between file dumps
	# 
	# \b Examples:
	# \code
	# analyze.imd(port=54321, period=100)
	# imd = analyze.imd(port=12345, period=1000)
	# \endcode
	def __init__(self, port, period):
		util.print_status_line();
		
		# initialize base class
		_analyzer.__init__(self);
		
		# create the c++ mirror class
		self.cpp_analyzer = hoomd.IMDInterface(globals.particle_data, port);
		globals.system.addAnalyzer(self.cpp_analyzer, self.analyzer_name, period);


## Logs a number of calculated quanties to a file
#
# analyze.log can read a variety of calculated values, like energy and temperature, from 
# specified forces, integrators, and updaters. It writes a single line to the specified
# output file every \a period time steps. The resulting file is suitable for direct import
# into a spreadsheet, MATLAB, or other software that can handle simple delimited files.
#
#
# Quantities that can be logged at any time:
# - \b num_particles - Number of particles in the system
# - \b volume - Volume of the simulation box
# - \b temperature - Temperature of the system
# - \b pressure - Pressure of the system
# - \b kinetic_energy - Total kinetic energy of the system
# - \b potential_energy - Total potential energy of the system
# - \b momentum - Magnitude of the total system momentum
# - \b conserved_quantity - Conserved quantity for the current integrator (the actual definition of this value
# depends on which integrator is being used in the current run()
# - \b time - Wall-clock running time from the start of the log in seconds
#
# The following quantities are only available of certain forces have been specified (as noted in the 
# parantheses)
# - \b pair_lj_energy (pair.lj) - Total Lennard-Jones potential energy
# - \b bond_fene_energy (bond.fene) - Total fene bond potential energy
# - \b bond_harmonic_energy (bond.harmonic) - Total harmonic bond potential energy
# - \b wall_lj_energy (wall.lj) - Total Lennard-Jones wall energy
# - \b nvt_xi (integrate.nvt) - \f$ \xi \f$ value in the NVT integrator
# - \b nvt_eta (integrate.nvt) - \f$ \eta \f$ value in the NVT integrator
#
class log(_analyzer):
	## Initialize the log
	#
	# \param filename File to write the log to
	# \param quantities List of quantities to log
	# \param period Quantities are logged every \a period time steps
	# \param header_prefix (optional) Specify a string to print before the header
	#
	# \b Examples:
	# \code
	# logger = analyze.log(filename='mylog.log', period=100,
	#                      quantities=['pair_lj_energy'])
	#
	# analyze.log(quantities=['pair_lj_energy', 'bond_harmonic_energy', 
	#             'kinetic_energy'], period=1000, filename='full.log')
	#
	# analyze.log(filename='mylog.log', quantities=['pair_lj_energy'], 
	#             period=100, header_prefix='#')
	#
	# analyze.log(filename='mylog.log', quantities=['bond_harmonic_energy'], 
	#             period=10, header_prefix='Log of harmonic energy, run 5\n')
	# \endcode
	#
	# By default, columns in the log file are separated by tabs, suitable for importing as a 
	# tab-delimited spreadsheet. The delimiter can be changed to any string using set_params()
	# 
	# The \a header_prefix can be used in a number of ways. It specifies a simple string that
	# will be printed before the header line of the output file. One handy way to use this
	# is to specify header_prefix='#' so that \c gnuplot will ignore the header line
	# automatically. Another use-case would be to specify a descriptive line containing
	# details of the current run. Examples of each of these cases are given above.
	def __init__(self, filename, quantities, period, header_prefix=''):
		util.print_status_line();
		
		# initialize base class
		_analyzer.__init__(self);
		
		# create the c++ mirror class
		self.cpp_analyzer = hoomd.Logger(globals.particle_data, filename, header_prefix);
		globals.system.addAnalyzer(self.cpp_analyzer, self.analyzer_name, period);
		
		# set the logged quantities
		quantity_list = hoomd.std_vector_string();
		for item in quantities:
			quantity_list.append(item);
		self.cpp_analyzer.setLoggedQuantities(quantity_list);
		
		# add the logger to the list of loggers
		globals.loggers.append(self);
	
	## Change the parameters of the log
	#
	# \param quantities New list of quantities to log (if specified)
	# \param delimiter New delimiter between columns in the output file (if specified)
	#
	# Using set_params() requires that the specified logger was saved in a variable when created.
	# i.e. 
	# \code
	# logger = analyze.log(quantities=['pair_lj_energy', 
	#                      'bond_harmonic_energy', 'nve_kinetic_energy'], 
	#                      period=1000, filename="'full.log')
	# \endcode
	#
	# \b Examples:
	# \code
	# logger.set_params(quantities=['bond_harmonic_energy'])
	# logger.set_params(delimiter=',');
	# logger.set_params(quantities=['bond_harmonic_energy'], delimiter=',');
	# \endcode
	def set_params(self, quantities=None, delimiter=None):
		util.print_status_line();
		
		if quantities != None:
			# set the logged quantities
			quantity_list = hoomd.std_vector_string();
			for item in quantities:
				quantity_list.append(item);
			self.cpp_analyzer.setLoggedQuantities(quantity_list);
			
		if delimiter:
			self.cpp_analyzer.setDelimiter(delimiter);
		
	## \internal
	# \brief Re-registers all computes and updaters with the logger
	def update_quantities(self):
		# remove all registered quantities
		self.cpp_analyzer.removeAll();
		
		# re-register all computes and updatesr
		globals.system.registerLogger(self.cpp_analyzer);


## Calculates the mean-squared displacement of groups of particles and logs the values to a file
#
# analyze.msd can be given any number of groups of particles. Every \a period time steps, it calculates the mean squared 
# displacement of each group (referenced to the particle positions at the time step the command is issued at) and prints
# the calculated values out to a file.
# 
# The mean squared displacement (MSD) for each group is calculated as:
# \f[ \langle |\vec{r} - \vec{r}_0|^2 \rangle \f]
#
# The file format is the same convient delimited format used by analyze.log 
class msd(_analyzer):
	## Initialize the msd calculator
	#
	# \param filename File to write the data to
	# \param groups List of groups to calculate the MSDs of
	# \param period Quantities are logged every \a period time steps
	# \param header_prefix (optional) Specify a string to print before the header
	#
	# \b Examples:
	# \code
	# msd = analyze.msd(filename='msd.log', groups=[group1, group2], 
	#                   period=100)
	#
	# analyze.log(groups=[group1, group2, group3], period=1000, 
	#             filename='msd.log', header_prefix='#')
	# 
	# analyze.log(filename='msd.log', groups=[group1], period=10, 
	#             header_prefix='Log of group1 msd, run 5\n')
	# \endcode
	#
	# A group variable (\c groupN above) can be created by any number of group creation functions.
	# see group for a list.
	#
	# By default, columns in the file are separated by tabs, suitable for importing as a 
	# tab-delimited spreadsheet. The delimiter can be changed to any string using set_params()
	# 
	# The \a header_prefix can be used in a number of ways. It specifies a simple string that
	# will be printed before the header line of the output file. One handy way to use this
	# is to specify header_prefix='#' so that \c gnuplot will ignore the header line
	# automatically. Another use-case would be to specify a descriptive line containing
	# details of the current run. Examples of each of these cases are given above.
	def __init__(self, filename, groups, period, header_prefix=''):
		util.print_status_line();
		
		# initialize base class
		_analyzer.__init__(self);
		
		# create the c++ mirror class
		self.cpp_analyzer = hoomd.MSDAnalyzer(globals.particle_data, filename, header_prefix);
		globals.system.addAnalyzer(self.cpp_analyzer, self.analyzer_name, period);
	
		# it is an error to specify no groups
		if len(groups) == 0:
			print >> sys.stderr, "\nAt least one group must be specified to analyze.msd\n";
			raise RuntimeError('Error creating analyzer');

		# set the group columns
		for cur_group in groups:
			self.cpp_analyzer.addColumn(cur_group.cpp_group, cur_group.name);
		
	## Change the parameters of the msd analysis
	#
	# \param delimiter New delimiter between columns in the output file (if specified)
	#
	# Using set_params() requires that the specified msd was saved in a variable when created.
	# i.e. 
	# \code
	# msd = analyze.msd(filename='msd.log', groups=[group1, group2], period=100)
	# \endcode
	#
	# \b Examples:
	# \code
	# msd.set_params(delimiter=',');
	# \endcode
	def set_params(self, delimiter=None):
		util.print_status_line();
		
		if delimiter:
			self.cpp_analyzer.setDelimiter(delimiter);
		
