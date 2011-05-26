# -*- coding: iso-8859-1 -*-
#Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
#(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
#Iowa State University and The Regents of the University of Michigan All rights
#reserved.

#HOOMD-blue may contain modifications ("Contributions") provided, and to which
#copyright is held, by various Contributors who have granted The Regents of the
#University of Michigan the right to modify and/or distribute such Contributions.

#Redistribution and use of HOOMD-blue, in source and binary forms, with or
#without modification, are permitted, provided that the following conditions are
#met:

#* Redistributions of source code must retain the above copyright notice, this
#list of conditions, and the following disclaimer.

#* Redistributions in binary form must reproduce the above copyright notice, this
#list of conditions, and the following disclaimer in the documentation and/or
#other materials provided with the distribution.

#* Neither the name of the copyright holder nor the names of HOOMD-blue's
#contributors may be used to endorse or promote products derived from this
#software without specific prior written permission.

#Disclaimer

#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
#ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

#IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
#INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
#OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
#ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# $Id$
# $URL$
# Maintainer: joaander / All Developers are free to add commands for new features

import hoomd;
import globals;
import sys;
import util;
import init;

## \package hoomd_script.analyze
# \brief Commands that %analyze the system and provide some output
#
# An analyzer examines the system state in some way every \a period time steps and generates
# some form of output based on the analysis. Check the documentation for individual analyzers 
# to see what they do.

## \page variable_period_docs Variable period specification
#
# If, for any reason, a constant period for a command is not to your liking, you can make it any 
# function you please! Just specify a function taking a single argument to the period parameter.
# Any analyze, update, or dump command in hoomd can be given such a variable period. 
# dump.xml is used as an example here, but the same works with \b any update, dump, 
# or analyze command
#
# For example, lets say we want to dump xml files at time steps 1, 10, 100, 1000, ...
# The following command will do the job.
#
# \code
# dump.xml(filename="dump", period = lambda n: 10**n)
# \endcode
#
# It is that simple. Any mathematical expression that can be represented in python can be used
# in place of the 10**n.
#
# <b>More examples:</b>
# \code
# dump.xml(filename="dump", period = lambda n: n**2)
# dump.xml(filename="dump", period = lambda n: 2**n)
# dump.xml(filename="dump", period = lambda n: 1005 + 0.5 * 10**n)
# \endcode
#
# The only requirement is that the object passed into period is callable, accepts one argument, and returns 
# a floating point number or integer. The function also had better be monotonically increasing or the output
# might not make any sense.
#
# <b>How does it work, exactly?</b>
# - First, the current time step of the simulation is saved when the analyzer is created
# - \a n is also set to 1 when the analyzer is created
# - Every time the analyzer performs it's output, it evaluates the given function at the current value of \a n
#   and records that as the next time to perform the analysis. \a n is then incremented by 1
#
# Here is a final example of how variable periods behave in simulations where analyzers are not created on time step 0.
# The following
# \code
# ... initialize ...
# run(4000)
# dump.xml(filename="dump", period = lambda n: 2**n)
# run(513)
# \endcode
# will result in dump files at time steps 4000, 4002, 4004, 4008, 4016, 4032, 4064, 4128, 4256, and 4512.
#
# In other words, the function specified for the period starts counting at the time step <b>when the analyzer is created</b>.
# Consequently, any analyze, dump, or update command given a variable period becomes ill-defined if it is disabled and then re-enabled.
# If this is done, it will then re-enable with a constant period of 1000 as a default case.
#

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
        # check if initialization has occurred
        if not init.is_initialized():
            print >> sys.stderr, "\n***Error! Cannot create analyzer before initialization\n";
            raise RuntimeError('Error creating analyzer');
        
        self.cpp_analyzer = None;

        # increment the id counter
        id = _analyzer.cur_id;
        _analyzer.cur_id += 1;
        
        self.analyzer_name = "analyzer%d" % (id);
        self.enabled = True;

    ## \internal
    # \brief Helper function to setup analyzer period
    #
    # \param period An integer or callable function period
    #
    # If an integer is specified, then that is set as the period for the analyzer.
    # If a callable is passed in as a period, then a default period of 1000 is set 
    # to the integer period and the variable period is enabled
    #
    def setupAnalyzer(self, period):
        if type(period) == type(1.0):
            globals.system.addAnalyzer(self.cpp_analyzer, self.analyzer_name, int(period));
        elif type(period) == type(1):
            globals.system.addAnalyzer(self.cpp_analyzer, self.analyzer_name, period);
        elif type(period) == type(lambda n: n*2):
            globals.system.addAnalyzer(self.cpp_analyzer, self.analyzer_name, 1000);
            globals.system.setAnalyzerPeriodVariable(self.analyzer_name, period);
        else:
            print >> sys.stderr, "\n***Error! I don't know what to do with a period of type", type(period), "expecting an int or a function\n";
            raise RuntimeError('Error creating analyzer');
            
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
    # \brief Saved period retrieved when an analyzer is disabled: used to set the period when re-enabled

    ## \internal
    # \brief Checks that proper initialization has completed
    def check_initialization(self):
        # check that we have been initialized properly
        if self.cpp_analyzer is None:
            print >> sys.stderr, "\nBug in hoomd_script: cpp_analyzer not set, please report\n";
            raise RuntimeError();

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
        self.check_initialization();
        
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
        self.check_initialization();
        
        # check if we are already disabled
        if self.enabled:
            print "***Warning! Ignoring command to enable an analyzer that is already enabled";
            return;
            
        globals.system.addAnalyzer(self.cpp_analyzer, self.analyzer_name, self.prev_period);
        self.enabled = True;
        
    ## Changes the period between analyzer executions
    #
    # \param period New period to set (in time steps)
    #
    # \b Examples:
    # \code
    # analyzer.set_period(100)
    # analyzer.set_period(1)
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
        
        if type(period) == type(1):
            if self.enabled:
                globals.system.setAnalyzerPeriod(self.analyzer_name, period);
            else:
                self.prev_period = period;
        elif type(period) == type(lambda n: n*2):
            print "***Warning! A period cannot be changed to a variable one";
        else:
            print "***Warning! I don't know what to do with a period of type", type(period), "expecting an int or a function";
        
# set default counter
_analyzer.cur_id = 0;


## Sends simulation snapshots to VMD in real-time
#
# analyze.imd listens on a specified TCP/IP port for connections from VMD.
# Once that connection is established, it begins transmitting simulation snapshots
# to VMD every \a rate time steps.
#
# To connect to a simulation running on the local host, issue the command
# \code
# imd connect localhost 54321
# \endcode
# in the VMD command window (where 54321 is replaced with the port number you specify for
# analyze.imd
#
# \note If a period larger than 1 is set, the actual rate at which time steps are transmitted is \a rate * \a period.
#
# \sa \ref page_example_scripts
class imd(_analyzer):
    ## Initialize the IMD interface
    #
    # \param port TCP/IP port to listen on
    # \param period Number of time steps to run before checking for new IMD messages
    # \param rate Number of \a periods between coordinate data transmissions.
    # \param pause Set to True to \b pause the simulation at the first time step until an imd connection is made
    # \param force Give a saved force.constant to analyze.imd to apply forces received from VMD
    # \param force_scale Factor by which to scale all forces received from VMD
    #
    # \b Examples:
    # \code
    # analyze.imd(port=54321, rate=100)
    # analyze.imd(port=54321, rate=100, pause=True)
    # imd = analyze.imd(port=12345, rate=1000)
    # \endcode
    #
    # \a period can be a function: see \ref variable_period_docs for details
    def __init__(self, port, period=1, rate=1, pause=False, force=None, force_scale=0.1):
        util.print_status_line();
        
        # initialize base class
        _analyzer.__init__(self);
        
        # get the cpp force
        if force is not None:
            cpp_force = force.cpp_force;
        else:
            cpp_force = None;
        
        # create the c++ mirror class
        self.cpp_analyzer = hoomd.IMDInterface(globals.system_definition, port, pause, rate, cpp_force);
        self.setupAnalyzer(period);


## Logs a number of calculated quantities to a file
#
# analyze.log can read a variety of calculated values, like energy and temperature, from 
# specified forces, integrators, and updaters. It writes a single line to the specified
# output file every \a period time steps. The resulting file is suitable for direct import
# into a spreadsheet, MATLAB, or other software that can handle simple delimited files.
#
#
# Quantities that can be logged at any time:
# - \b volume - Volume of the simulation box (in volume units)
# - \b momentum - Magnitude of the average momentum of all particles (in momentum units)
# - \b time - Wall-clock running time from the start of the log (in seconds)
#
# Thermodynamic properties
# - The following quantities are always available and computed over all particles in the system
#   (see compute.thermo for detailed definitions):
#   - \b num_particles
#   - \b ndof
#   - \b potential_energy (in energy units)
#   - \b kinetic_energy (in energy units)
#   - \b temperature (in thermal energy units)
#   - \b pressure (in pressure units)
# - The above quantities, tagged with a <i>_groupname</i> suffix are automatically available for any group passed to
#   an integrate command 
# - Specify a compute.thermo directly to enable additional quantities for user-specified groups.
#
# The following quantities are only available only if the command is parentheses has been specified and is active
# for logging. 
# - \b pair_gauss_energy (pair.gauss) - Total Gaussian potential energy (in energy units)
# - \b pair_lj_energy (pair.lj) - Total Lennard-Jones potential energy (in energy units)
# - \b pair_morse_energy (pair.yukawa) - Total Morse potential energy (in energy units)
# - \b pair_table_energy (pair.table) - Total potential energy from Tabulated potentials (in energy units)
# - \b pair_slj_energy (pair.slj) - Total Shifted Lennard-Jones potential energy (in energy units)
# - \b pair_yukawa_energy (pair.yukawa) - Total Yukawa potential energy (in energy units)
# - \b pair_ewald_energy (pair.ewald) - Short ranged part of the electrostatic energy (in energy units)
# - \b pppm_energy (charge.pppm) -  Long ranged part of the electrostatic energy (in energy units)
#
# - \b bond_fene_energy (bond.fene) - Total fene bond potential energy (in energy units)
# - \b bond_harmonic_energy (bond.harmonic) - Total harmonic bond potential energy (in energy units)
# - \b wall_lj_energy (wall.lj) - Total Lennard-Jones wall energy (in energy units)
#
# - <b>bdnvt_reservoir_energy<i>_groupname</i></b> (integrate.bdnvt) - Energy reservoir for the BD thermostat (in energy units)
# - <b>nvt_reservoir_energy<i>_groupname</i></b> (integrate.nvt) - Energy reservoir for the NVT thermostat (in energy units)
# 
# Additionally, the following commands can be provided user-defined names that are appended as suffixes to the 
# logged quantitiy (e.g. with \c pair.lj(r_cut=2.5, \c name="alpha"), the logged quantity would be pair_lj_energy_alpha).
# - pair.gauss
# - pair.lj
# - pair.morse
# - pair.table
# - pair.slj
# - pair.yukawa
#
# - bond.fene
# - bond.harmonic
#
# By specifying a force, disabling it with the \a log=True option, and then logging it, different energy terms can
# be computed while only a subset of them actually drive the simulation. Common use-cases of this capability
# include separating out pair energy of given types (shown below) and free energy calculations. Be aware that the
# globally chosen \a r_cut value is the largest of all active pair potentials and those with \a log=True, so you will
# observe performance degradation if you \a disable(log=True) a potential with a large \a r_cut.
#
# \b Examples:
# \code
# lj1 = pair.lj(r_cut=3.0, name="lj1")
# lj1.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
# lj1.pair_coeff.set('A', 'B', epsilon=1.0, sigma=1.0)
# lj1.pair_coeff.set('B', 'B', epsilon=1.0, sigma=1.0)
#
# lj2 = pair.lj(r_cut=3.0, name="lj2")
# lj2.pair_coeff.set('A', 'A', epsilon=1.0, sigma=1.0)
# lj2.pair_coeff.set('A', 'B', epsilon=0.0, sigma=1.0)
# lj2.pair_coeff.set('B', 'B', epsilon=0.0, sigma=1.0)
# lj2.disable(log=True)
#
# analyze.log(filename='mylog.log', quantities=['pair_lj_energy_lj1', 'pair_lj_energy_lj2'],
#             period=100, header_prefix='#')
# \endcode
#
# \sa \ref page_units
class log(_analyzer):
    ## Initialize the log
    #
    # \param filename File to write the log to
    # \param quantities List of quantities to log
    # \param period Quantities are logged every \a period time steps
    # \param header_prefix (optional) Specify a string to print before the header
    # \param overwrite When False (the default) an existing log will be appended to. 
    #                  If True, an existing log file will be overwritten instead.
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
    # logger = analyze.log(filename='mylog.log', period=100,
    #                      quantities=['pair_lj_energy'], overwrite=True)
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
    #
    # \warning When an existing log is appended to, the header is not printed. For the log to 
    # remain consistent with the header already in the file, you must specify the same quantities
    # to log and in the same order for all runs of hoomd that append to the same log.
    #
    # \a period can be a function: see \ref variable_period_docs for details
    def __init__(self, filename, quantities, period, header_prefix='', overwrite=False):
        util.print_status_line();
        
        # initialize base class
        _analyzer.__init__(self);
        
        # create the c++ mirror class
        self.cpp_analyzer = hoomd.Logger(globals.system_definition, filename, header_prefix, overwrite);
        self.setupAnalyzer(period);
        
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
    #                      'bond_harmonic_energy', 'kinetic_energy'], 
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
        
        if quantities is not None:
            # set the logged quantities
            quantity_list = hoomd.std_vector_string();
            for item in quantities:
                quantity_list.append(item);
            self.cpp_analyzer.setLoggedQuantities(quantity_list);
            
        if delimiter:
            self.cpp_analyzer.setDelimiter(delimiter);
        
    ## Retrieve a cached value of a monitored quantity from the last update of the logger.
    # \param quantity Name of the quantity to return.
    #
    # Using query() requires that the specified logger was saved in a variable when created.
    # i.e. 
    # \code
    # logger = analyze.log(quantities=['pair_lj_energy', 
    #                      'bond_harmonic_energy', 'kinetic_energy'], 
    #                      period=1000, filename="'full.log')
    # \endcode
    #
    # \b Examples:
    # \code
    # logdata = logger.query('timestep')
    # \endcode
    def query(self, quantity):
        # retrieve data from internal cache.
        return self.cpp_analyzer.getCachedQuantity(quantity);
        
    ## \internal
    # \brief Re-registers all computes and updaters with the logger
    def update_quantities(self):
        # remove all registered quantities
        self.cpp_analyzer.removeAll();
        
        # re-register all computes and updater
        globals.system.registerLogger(self.cpp_analyzer);


## Calculates the mean-squared displacement of groups of particles and logs the values to a file
#
# analyze.msd can be given any number of groups of particles. Every \a period time steps, it calculates the mean squared 
# displacement of each group (referenced to the particle positions at the time step the command is issued at) and prints
# the calculated values out to a file.
# 
# The mean squared displacement (MSD) for each group is calculated as:
# \f[ \langle |\vec{r} - \vec{r}_0|^2 \rangle \f]
# and values are correspondingly written in units of distance squared.
#
# The file format is the same convenient delimited format used by analyze.log 
#
# analyze.msd is capable of appending to an existing msd file (the default setting) for use in restarting in long jobs.
# To generate a correct msd that does not reset to 0 at the start of each run, save the initial state of the system
# in a hoomd_xml file, including position and image data at a minimum. In the continuation job, specify this file
# in the \a r0_file argument to analyze.msd.
class msd(_analyzer):
    ## Initialize the msd calculator
    #
    # \param filename File to write the %data to
    # \param groups List of groups to calculate the MSDs of
    # \param period Quantities are logged every \a period time steps
    # \param header_prefix (optional) Specify a string to print before the header
    # \param r0_file hoomd_xml file specifying the positions (and images) to use for \f$ \vec{r}_0 \f$
    # \param overwrite set to True to overwrite the file \a filename if it exists
    #
    # \b Examples:
    # \code
    # msd = analyze.msd(filename='msd.log', groups=[group1, group2], 
    #                   period=100)
    #
    # analyze.msd(groups=[group1, group2, group3], period=1000, 
    #             filename='msd.log', header_prefix='#')
    # 
    # analyze.msd(filename='msd.log', groups=[group1], period=10, 
    #             header_prefix='Log of group1 msd, run 5\n')
    # \endcode
    #
    # A group variable (\c groupN above) can be created by any number of group creation functions.
    # See group for a list.
    #
    # By default, columns in the file are separated by tabs, suitable for importing as a 
    # tab-delimited spreadsheet. The delimiter can be changed to any string using set_params()
    # 
    # The \a header_prefix can be used in a number of ways. It specifies a simple string that
    # will be printed before the header line of the output file. One handy way to use this
    # is to specify header_prefix='#' so that \c gnuplot will ignore the header line
    # automatically. Another use-case would be to specify a descriptive line containing
    # details of the current run. Examples of each of these cases are given above.
    #
    # If \a r0_file is left at the default of None, then the current state of the system at the execution of the
    # analyze.msd command is used to initialize \f$ \vec{r}_0 \f$.
    #
    # \a period can be a function: see \ref variable_period_docs for details
    def __init__(self, filename, groups, period, header_prefix='', r0_file=None, overwrite=False):
        util.print_status_line();
        
        # initialize base class
        _analyzer.__init__(self);
        
        # create the c++ mirror class
        self.cpp_analyzer = hoomd.MSDAnalyzer(globals.system_definition, filename, header_prefix, overwrite);
        self.setupAnalyzer(period);
    
        # it is an error to specify no groups
        if len(groups) == 0:
            print >> sys.stderr, "\nAt least one group must be specified to analyze.msd\n";
            raise RuntimeError('Error creating analyzer');

        # set the group columns
        for cur_group in groups:
            self.cpp_analyzer.addColumn(cur_group.cpp_group, cur_group.name);
        
        if r0_file is not None:
            self.cpp_analyzer.setR0(r0_file);
        
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

