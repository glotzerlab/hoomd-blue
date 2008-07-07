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

## \package hoomd_script.analyze
# \brief Commands that analyze the particles
#
# Details

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
			print "Error: Cannot create analyzer before initialization";
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
	# \b Examples:<br>
	# analyzer.disable()
	#
	# Executing the disable command will remove the analyzer from the system.
	# Any run() command exected after disabling an analyzer will not use that 
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
		print "analyzer.disable()";
		
		# check that we have been initialized properly
		if self.cpp_analyzer == None:
			"Bug in hoomd_script: cpp_analyzer not set, please report";
			raise RuntimeError('Error disabling analyzer');
			
		# check if we are already disabled
		if not self.enabled:
			print "Warning: Ignoring command to disable an analyzer that is already disabled";
			return;
		
		self.prev_period = globals.system.getAnalyzerPeriod(self.analyzer_name);
		globals.system.removeAnalyzer(self.analyzer_name);
		self.enabled = False;

	## Enables the analyzer
	#
	# \b Examples:<br>
	# analyzer.enable()
	#
	# See disable() for a detailed description.
	def enable(self):
		print "analyzer.enable()";
		
		# check that we have been initialized properly
		if self.cpp_analyzer == None:
			"Bug in hoomd_script: cpp_analyzer not set, please report";
			raise RuntimeError('Error disabling analyzer');
			
		# check if we are already disabled
		if self.enabled:
			print "Warning: Ignoring command to enable an analyzer that is already enabled";
			return;
			
		globals.system.addAnalyzer(self.cpp_analyzer, self.analyzer_name, self.prev_period);
		self.enabled = True;
		
	## Changes the period between analyzer executions
	#
	# \param period New period to set
	#
	# \b Examples:<br>
	# analyzer.set_period(100);<br>
	# analyzer.set_period(1);<br>
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
		print "analyzer.set_period(", period, ")";
		globals.system.setAnalyzerPeriod(self.analyzer_name, period);

# set default counter
_analyzer.cur_id = 0;


## Sends simulation snapshots to VMD in real-time
#
# analyze.imd listens on a specified TCP/IP port for connections from VMD.
# Once that connection is established, it begins transmitting simulation snapshots
# to VMD every \a period time steps.
class imd(_analyzer):
	## Initialize the IMD interface
	#
	# \param port TCP/IP port to listen on
	# \param period Number of time steps between file dumps
	# 
	# \b Examples:<br>
	# analyze.imd(port=54321, period=100)<br>
	# imd = analyze.imd(port=12345, period=1000)<br>
	def __init__(self, port, period):
		print "analyze.imd(port =", port, ")";
		
		# initialize base class
		_analyzer.__init__(self);
		
		# create the c++ mirror class
		self.cpp_analyzer = hoomd.IMDInterface(globals.particle_data, port);
		globals.system.addAnalyzer(self.cpp_analyzer, self.analyzer_name, period);
