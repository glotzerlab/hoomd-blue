# -- start license --
# Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
# (HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
# Iowa State University and The Regents of the University of Michigan All rights
# reserved.

# HOOMD-blue may contain modifications ("Contributions") provided, and to which
# copyright is held, by various Contributors who have granted The Regents of the
# University of Michigan the right to modify and/or distribute such Contributions.

# You may redistribute, use, and create derivate works of HOOMD-blue, in source
# and binary forms, provided you abide by the following conditions:

# * Redistributions of source code must retain the above copyright notice, this
# list of conditions, and the following disclaimer both in the code and
# prominently in any materials provided with the distribution.

# * Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions, and the following disclaimer in the documentation and/or
# other materials provided with the distribution.

# * All publications and presentations based on HOOMD-blue, including any reports
# or published results obtained, in whole or in part, with HOOMD-blue, will
# acknowledge its use according to the terms posted at the time of submission on:
# http://codeblue.umich.edu/hoomd-blue/citations.html

# * Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
# http://codeblue.umich.edu/hoomd-blue/

# * Apart from the above required attributions, neither the name of the copyright
# holder nor the names of HOOMD-blue's contributors may be used to endorse or
# promote products derived from this software without specific prior written
# permission.

# Disclaimer

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
# WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# -- end license --

# Maintainer: joaander / All Developers are free to add commands for new features

import force;
import globals;
import hoomd;
import util;
import tune;
import data;
import init;

import math;
import sys;

## \package hoomd_script.bond
# \brief Commands that specify %bond forces
#
# Bonds add forces between specified pairs of particles and are typically used to 
# model chemical bonds. Bonds between particles are set when an input XML file is read
# (init.read_xml) or when an another initializer creates them (like init.create_random_polymers)
#
# By themselves, bonds that have been specified in an input file do nothing. Only when you 
# specify a bond force (i.e. bond.harmonic), are forces actually calculated between the 
# listed particles.

# \brief Defines bond potential coefficients
# The coefficients for all %bond force are specified using this class. Coefficients are
# specified per bond type.
#
# There are two ways to set the coefficients for a particular %bond %force.
# The first way is to save the %bond %force in a variable and call set() directly.
# See below for an example of this.
#
# The second method is to build the coeff class first and then assign it to the
# %bond %force. There are some advantages to this method in that you could specify a
# complicated set of %bond %force coefficients in a separate python file and import
# it into your job script.
#
# Example:
# \code
# my_coeffs = bond.coeff();
# my_bond_force.bond_coeff.set('polymer', k=330.0, r=0.84)
# my_bond_force.bond_coeff.set('backbone', k=330.0, r=0.84)
# \endcode
class coeff:
    ## \internal
    # \brief Initializes the class
    # \details
    # The main task to be performed during initialization is just to init some variables
    # \param self Python required class instance variable
    def __init__(self):
        self.values = {};
        self.default_coeff = {}

    ## \var values
    # \internal
    # \brief Contains the vector of set values in a dictionary

    ## \var default_coeff
    # \internal
    # \brief default_coeff['coeff'] lists the default value for \a coeff, if it is set

    ## \internal
    # \brief Sets a default value for a given coefficient
    # \details
    # \param name Name of the coefficient to for which to set the default
    # \param value Default value to set
    #
    # Some coefficients have reasonable default values and the user should not be burdened with typing them in
    # all the time. set_default_coeff() sets
    def set_default_coeff(self, name, value):
        self.default_coeff[name] = value;

    ## Sets parameters for one bond type
    # \param type Type of bond
    # \param coeff Named coefficients (see below for examples)
    #
    # Calling set() results in one or more parameters being set for a bond type. Types are identified
    # by name, and parameters are also added by name. Which parameters you need to specify depends on the %bond
    # %force you are setting these coefficients for, see the corresponding documentation.
    #
    # All possible bond types as defined in the simulation box must be specified before executing run().
    # You will receive an error if you fail to do so. It is not an error, however, to specify coefficients for
    # bond types that do not exist in the simulation. This can be useful in defining a %force field for many
    # different types of bonds even when some simulations only include a subset.
    #
    # To set the same coefficients between many particle types, provide a list of type names instead of a single
    # one. All types in the list will be set to the same parameters. A convenient wildcard that lists all types
    # of particles in the simulation can be gotten from a saved \c system from the init command.
    #
    # \b Examples:
    # \code
    # coeff.set('polymer', k=330.0, r0=0.84)
    # coeff.set('backbone', k=1000.0, r0=1.0)
    # coeff.set(['bondA','bondB'], k=100, r0=0.0)
    # \endcode
    #
    # \note Single parameters can be updated. If both k and r0 have already been set for a particle type,
    # then executing coeff.set('polymer', r0=1.0) will %update the value of polymer bonds and leave the other
    # parameters as they were previously set.
    #
    def set(self, type, **coeffs):
        util.print_status_line();

        # listify the input
        if isinstance(type, str):
            type = [type];

        for typei in type:
            self.set_single(typei, coeffs);

    ## \internal
    # \brief Sets a single parameter
    def set_single(self, type, coeffs):
        # create the type identifier if it hasn't been created yet
        if (not type in self.values):
            self.values[type] = {};

        # update each of the values provided
        if len(coeffs) == 0:
            print >> sys.stderr, "\n***Error! No coefficents specified\n";
        for name, val in coeffs.items():
            self.values[type][name] = val;

        # set the default values
        for name, val in self.default_coeff.items():
            # don't override a coeff if it is already set
            if not name in self.values[type]:
                self.values[type][name] = val;

    ## \internal
    # \brief Verifies that all values are set
    # \details
    # \param self Python required self variable
    # \param required_coeffs list of required variables
    #
    # This can only be run after the system has been initialized
    def verify(self, required_coeffs):
        # first, check that the system has been initialized
        if not init.is_initialized():
            print >> sys.stderr, "\n***Error! Cannot verify bond coefficients before initialization\n";
            raise RuntimeError('Error verifying force coefficients');

        # get a list of types from the particle data
        ntypes = globals.system_definition.getBondData().getNBondTypes();
        type_list = [];
        for i in xrange(0,ntypes):
            type_list.append(globals.system_definition.getBondData().getNameByType(i));

        valid = True;
        # loop over all possible types and verify that all required variables are set
        for i in xrange(0,ntypes):
            type = type_list[i];

            if type not in self.values.keys():
                print >> sys.stderr, "\n***Error! Bond type", type, "not found in bond coeff\n"
                valid = False;
                continue;

            # verify that all required values are set by counting the matches
            count = 0;
            for coeff_name in self.values[type].keys():
                if not coeff_name in required_coeffs:
                    print "Notice: Possible typo? Force coeff", coeff_name, "is specified for type", type, \
                          ", but is not used by the bond force";
                else:
                    count += 1;

            if count != len(required_coeffs):
                print >> sys.stderr, "\n***Error! Bonde type", type, "is missing required coefficients\n";
                valid = False;

        return valid;

    ## \internal
    # \brief Gets the value of a single %bond %force coefficient
    # \detail
    # \param type Name of bond type
    # \param coeff_name Coefficient to get
    def get(self, type, coeff_name):
        if type not in self.values.keys():
            print >> sys.stderr, "\nBug detected in force.coeff. Please report\n"
            raise RuntimeError("Error setting bond coeff");

        return self.values[type][coeff_name];

## \internal
# \brief Base class for bond potentials
#
# A bond in hoomd_script reflects a PotentialBond in c++. It is responsible
# for all high-level management that happens behind the scenes for hoomd_script
# writers. 1) The instance of the c++ bond force itself is tracked and added to the
# System 2) methods are provided for disabling the force from being added to the
# net force on each particle
class _bond(force._force):
    ## \internal
    # \brief Constructs the bond potential
    #
    # \param name name of the bond potential instance
    #
    # Initializes the cpp_force to None.
    # If specified, assigns a name to the instance
    # Assigns a name to the force in force_name;
    def __init__(self, name=None):
        # initialize the base class
        force._force.__init__(self, name);

        self.cpp_force = None;

        # setup the coefficient vector
        self.bond_coeff = coeff();

        self.enabled = True;

        # create force data iterator
        self.forces = data.force_data(self);

    def update_coeffs(self):
        coeff_list = self.required_coeffs;
        # check that the force coefficients are valid
        if not self.bond_coeff.verify(coeff_list):
           print >> sys.stderr, "\n***Error: Not all force coefficients are set\n";
           raise RuntimeError("Error updating force coefficients");

        # set all the params
        ntypes = globals.system_definition.getBondData().getNBondTypes();
        type_list = [];
        for i in xrange(0,ntypes):
            type_list.append(globals.system_definition.getBondData().getNameByType(i));

        for i in xrange(0,ntypes):
            # build a dict of the coeffs to pass to proces_coeff
            coeff_dict = {};
            for name in coeff_list:
                coeff_dict[name] = self.bond_coeff.get(type_list[i], name);

            param = self.process_coeff(coeff_dict);
            self.cpp_force.setParams(i, param);

    ## \var enabled
    # \internal
    # \brief True if the force is enabled

    ## \var cpp_force
    # \internal
    # \brief Stores the C++ side ForceCompute managed by this class

    ## \var force_name
    # \internal
    # \brief The Force's name as it is assigned to the System

    ## \internal
    # \brief Checks that proper initialization has completed
    def check_initialization(self):
        # check that we have been initialized properly
        if self.cpp_force is None:
            print >> sys.stderr, "\nBug in hoomd_script: cpp_force not set, please report\n";
            raise RuntimeError();

    ## Disables the bond force
    #
    # \b Examples:
    # \code
    # bond_force.disable()
    # \endcode
    #
    # Executing the disable command will remove the force from the simulation.
    # Any run() command executed after disabling a force will not calculate or
    # use the force during the simulation. A disabled force can be re-enabled
    # with enable()
    #
    # To use this command, you must have saved the force in a variable, as
    # shown in this example:
    # \code
    # force = bond.some_force()
    # # ... later in the script
    # force.disable()
    # \endcode
    def disable(self):
        util.print_status_line();
        self.check_initialization();

        # check if we are already disabled
        if not self.enabled:
            print "***Warning! Ignoring command to disable a force that is already disabled";
            return;

        self.enabled = False;

        # remove the compute from the system
        globals.system.removeCompute(self.force_name);

    ## Benchmarks the force computation
    # \param n Number of iterations to average the benchmark over
    #
    # \b Examples:
    # \code
    # t = force.benchmark(n = 100)
    # \endcode
    #
    # The value returned by benchmark() is the average time to perform the force
    # computation, in milliseconds. The benchmark is performed by taking the current
    # positions of all particles in the simulation and repeatedly calculating the forces
    # on them. Thus, you can benchmark different situations as you need to by simply
    # running a simulation to achieve the desired state before running benchmark().
    #
    # \note
    # There is, however, one subtle side effect. If the benchmark() command is run
    # directly after the particle data is initialized with an init command, then the
    # results of the benchmark will not be typical of the time needed during the actual
    # simulation. Particles are not reordered to improve cache performance until at least
    # one time step is performed. Executing run(1) before the benchmark will solve this problem.
    #
    # To use this command, you must have saved the force in a variable, as
    # shown in this example:
    # \code
    # force = bond.some_force()
    # # ... later in the script
    # t = force.benchmark(n = 100)
    # \endcode
    def benchmark(self, n):
        self.check_initialization();

        # run the benchmark
        return self.cpp_force.benchmark(int(n))


    ## Enables the force
    #
    # \b Examples:
    # \code
    # force.enable()
    # \endcode
    #
    # See disable() for a detailed description.
    def enable(self):
        util.print_status_line();
        self.check_initialization();

        # check if we are already disabled
        if self.enabled:
            print "***Warning! Ignoring command to enable a force that is already enabled";
            return;

        # add the compute back to the system
        globals.system.addCompute(self.cpp_force, self.force_name);

        self.enabled = True;


### Harmonic Harmonic %bond force
#
# The command bond.harmonic specifies a %harmonic potential energy between every bonded %pair of particles
# in the simulation. 
# \f[ V(r) = \frac{1}{2} k \left( r - r_0 \right)^2 \f]
# where \f$ \vec{r} \f$ is the vector pointing from one particle to the other in the %pair.
#
# Coeffients:
# - \f$ k \f$ - %force constant (in units of energy/distance^2)
# - \f$ r_0 \f$ - %bond rest length (in distance units)
#
# Coefficients \f$ k \f$ and \f$ r_0 \f$ must be set for each type of %bond in the simulation using
# bond_coeff.set().
# \note For compatibility with older versions of HOOMD-blue, the syntax set_coeff() is also supported.
#
# \note Specifying the bond.harmonic command when no bonds are defined in the simulation results in an error.
class harmonic(_bond):
    ## Specify the %harmonic %bond %force
    #
    # \param name Name of the bond instance
    #
    # \b Example:
    # \code
    # harmonic = bond.harmonic(name="mybond")
    # harmonic.bond_coeff.set('polymer', k=330.0, r0=0.84)
    # \endcode
    def __init__(self,name=None):
        util.print_status_line();

        # initiailize the base class
        _bond.__init__(self);

        # check that some bonds are defined
        if globals.system_definition.getBondData().getNumBonds() == 0:
            print >> sys.stderr, "\n***Error! No bonds are defined.\n";
            raise RuntimeError("Error creating bond forces");
        
        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = hoomd.PotentialBondHarmonic(globals.system_definition,self.name);
        else:
            self.cpp_force = hoomd.PotentialBondHarmonicGPU(globals.system_definition,self.name);
            self.cpp_force.setBlockSize(tune._get_optimal_block_size('bond.harmonic'));

        globals.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficient options
        self.required_coeffs = ['k','r0'];

    ## Set parameters for %harmonic %bond %force (deprecated)
    # \param type bond type
    # \param k %force constant (in units of energy/distance^2)
    # \param r0 rest length (in distance units)
    def set_coeff(self, type, **coeffs):
        print "*** Warning: Syntax bond.harmonic.set_coeff deprecated."
        self.bond_coeff.set(type,**coeffs)
        
    def process_coeff(self, coeff):
        k = coeff['k'];
        r0 = coeff['r0'];

        # set the parameters for the appropriate type
        return hoomd.make_scalar2(k, r0);
        

## FENE %bond force
#
# The command bond.fene specifies a %fene potential energy between every bonded %pair of particles
# in the simulation. 
# \f[ V(r) = - \frac{1}{2} k r_0^2 \ln \left( 1 - \left( \frac{r - \Delta}{r_0} \right)^2 \right) + V_{\mathrm{WCA}}(r)\f]
# where \f$ \vec{r} \f$ is the vector pointing from one particle to the other in the %pair,
# \f$ \Delta = (d_i + d_j)/2 - 1 \f$, \f$ d_i \f$ is the diameter of particle \f$ i \f$, and
# \f{eqnarray*}
#   V_{\mathrm{WCA}}(r)  = & 4 \varepsilon \left[ \left( \frac{\sigma}{r - \Delta} \right)^{12} - \left( \frac{\sigma}{r - \Delta} \right)^{6} \right]  + \varepsilon & r-\Delta < 2^{\frac{1}{6}}\sigma\\
#            = & 0          & r-\Delta \ge 2^{\frac{1}{6}}\sigma    \\
#   \f}
#
# Coefficients:
# - \f$ k \f$ - attractive %force strength (in units of energy/distance^2)
# - \f$ r_0 \f$ - size parameter (in distance units)
# - \f$ \varepsilon \f$ - repulsive %force strength (in energy units)
# - \f$ \sigma \f$ - repulsive %force interaction distance (in distance units)
#
# Coefficients \f$ k \f$, \f$ r_0 \f$, \f$ \varepsilon \f$ and \f$ \sigma \f$  must be set for 
# each type of %bond in the simulation using bond_coeff.set().
# \note For compatibility with older versions of HOOMD-blue, the syntax set_coeff() is also supported.
#
# \note Specifying the bond.fene command when no bonds are defined in the simulation results in an error.
class fene(_bond):
    ## Specify the %fene %bond %force
    #
    # \param name Name of the bond instance
    #
    # \b Example:
    # \code
    # fene = bond.fene()
    # \endcode
    def __init__(self, name=None):
        util.print_status_line();
        
        # check that some bonds are defined
        if globals.system_definition.getBondData().getNumBonds() == 0:
            print >> sys.stderr, "\n***Error! No bonds are defined.\n";
            raise RuntimeError("Error creating bond forces");
        
        # initialize the base class
        _bond.__init__(self, name);
        
        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = hoomd.PotentialBondFENE(globals.system_definition,self.name);
        else:
            self.cpp_force = hoomd.PotentialBondFENEGPU(globals.system_definition,self.name);
            self.cpp_force.setBlockSize(tune._get_optimal_block_size('bond.fene'));

        globals.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficient options
        self.required_coeffs = ['k','r0','epsilon','sigma'];

    ## Set parameters for %fene %bond %force (deprecated)
    #
    # \param type bond type
    # \param k attractive %force strength (in units of energy/distance^2)
    # \param r0 size parameter (in distance units)
    # \param epsilon pulsive %force strength (in energy units)
    # \param sigma repulsive %force interaction distance (in distance units)
    def set_coeff(self, type, **coeffs):
        print "*** Warning: Syntax bond.fene.set_coeff deprecated."
        self.bond_coeff.set(type, **coeffs)

    def process_coeff(self, coeff):
        k = coeff['k'];
        r0 = coeff['r0'];
        lj1 = 4.0 * coeff['epsilon'] * math.pow(coeff['sigma'], 12.0);
        lj2 = 4.0 * coeff['epsilon'] * math.pow(coeff['sigma'], 6.0);
        return hoomd.make_scalar4(k, r0, lj1, lj2);

