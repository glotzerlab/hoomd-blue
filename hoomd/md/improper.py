# -- start license --
# Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
# (HOOMD-blue) Open Source Software License Copyright 2009-2016 The Regents of
# the University of Michigan All rights reserved.

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

from hoomd.md import _md
from hoomd.md import force;
import hoomd;

import math;
import sys;

## \package hoomd.improper
# \brief Commands that specify %improper forces
#
# Impropers add forces between specified quadruplets of particles and are typically used to
# model rotation about chemical bonds without having bonds to connect the atoms. Their most
# common use is to keep structural elements flat, i.e. model the effect of conjugated
# double bonds, like in benzene rings and its derivatives.
# Impropers between particles are set when an input XML file is read
# (init.read_xml) or when an another initializer creates them (like init.create_random_polymers)
#
# By themselves, impropers that have been specified in an input file do nothing. Only when you
# specify an improper force (i.e. improper.harmonic), are forces actually calculated between the
# listed particles.

## Defines %improper coefficients
# \brief Defines improper potential coefficients
# The coefficients for all %improper force are specified using this class. Coefficients are
# specified per improper type.
#
# There are two ways to set the coefficients for a particular %improper %force.
# The first way is to save the %improper %force in a variable and call set() directly.
# See below for an example of this.
#
# The second method is to build the coeff class first and then assign it to the
# %improper %force. There are some advantages to this method in that you could specify a
# complicated set of %improper %force coefficients in a separate python file and import
# it into your job script.
#
# Example:
# \code
# my_coeffs = improper.coeff();
# my_improper_force.improper_coeff.set('polymer', k=330.0, r=0.84)
# my_improper_force.improper_coeff.set('backbone', k=330.0, r=0.84)
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

    ## Sets parameters for one improper type
    # \param type Type of improper
    # \param coeffs Named coefficients (see below for examples)
    #
    # Calling set() results in one or more parameters being set for a improper type. Types are identified
    # by name, and parameters are also added by name. Which parameters you need to specify depends on the %improper
    # %force you are setting these coefficients for, see the corresponding documentation.
    #
    # All possible improper types as defined in the simulation box must be specified before executing run().
    # You will receive an error if you fail to do so. It is not an error, however, to specify coefficients for
    # improper types that do not exist in the simulation. This can be useful in defining a %force field for many
    # different types of impropers even when some simulations only include a subset.
    #
    # To set the same coefficients between many particle types, provide a list of type names instead of a single
    # one. All types in the list will be set to the same parameters. A convenient wildcard that lists all types
    # of particles in the simulation can be gotten from a saved \c system from the init command.
    #
    # \b Examples:
    # \code
    # my_improper_force.improper_coeff.set('polymer', k=330.0, r0=0.84)
    # my_improper_force.improper_coeff.set('backbone', k=1000.0, r0=1.0)
    # my_improper_force.improper_coeff.set(['improperA','improperB'], k=100, r0=0.0)
    # \endcode
    #
    # \note Single parameters can be updated. If both k and r0 have already been set for a particle type,
    # then executing coeff.set('polymer', r0=1.0) will %update the value of polymer impropers and leave the other
    # parameters as they were previously set.
    #
    def set(self, type, **coeffs):
        hoomd.util.print_status_line();

        # listify the input
        if isinstance(type, str):
            type = [type];

        for typei in type:
            self.set_single(typei, coeffs);

    ## \internal
    # \brief Sets a single parameter
    def set_single(self, type, coeffs):
        type = str(type);

        # create the type identifier if it hasn't been created yet
        if (not type in self.values):
            self.values[type] = {};

        # update each of the values provided
        if len(coeffs) == 0:
            hoomd.context.msg.error("No coefficents specified\n");
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
        if not hoomd.init.is_initialized():
            hoomd.context.msg.error("Cannot verify improper coefficients before initialization\n");
            raise RuntimeError('Error verifying force coefficients');

        # get a list of types from the particle data
        ntypes = hoomd.context.current.system_definition.getImproperData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getImproperData().getNameByType(i));

        valid = True;
        # loop over all possible types and verify that all required variables are set
        for i in range(0,ntypes):
            type = type_list[i];

            if type not in self.values.keys():
                hoomd.context.msg.error("Improper type " +str(type) + " not found in improper coeff\n");
                valid = False;
                continue;

            # verify that all required values are set by counting the matches
            count = 0;
            for coeff_name in self.values[type].keys():
                if not coeff_name in required_coeffs:
                    hoomd.context.msg.notice(2, "Notice: Possible typo? Force coeff " + str(coeff_name) + " is specified for type " + str(type) + \
                          ", but is not used by the improper force\n");
                else:
                    count += 1;

            if count != len(required_coeffs):
                hoomd.context.msg.error("Improper type " + str(type) + " is missing required coefficients\n");
                valid = False;

        return valid;

    ## \internal
    # \brief Gets the value of a single %improper %force coefficient
    # \detail
    # \param type Name of improper type
    # \param coeff_name Coefficient to get
    def get(self, type, coeff_name):
        if type not in self.values.keys():
            hoomd.context.msg.error("Bug detected in force.coeff. Please report\n");
            raise RuntimeError("Error setting improper coeff");

        return self.values[type][coeff_name];

    ## \internal
    # \brief Return metadata
    def get_metadata(self):
        return self.values

## Harmonic %improper force
#
# The command improper.harmonic specifies a %harmonic improper potential energy between every quadruplet of particles
# in the simulation.
# \f[ V(r) = \frac{1}{2}k \left( \chi - \chi_{0}  \right )^2 \f]
# where \f$ \chi \f$ is angle between two sides of the improper
#
# Coefficients:
# - \f$ k \f$ - strength of %force (in energy units)
# - \f$ \chi_{0} \f$ - equilibrium angle (in radians)
#
# Coefficients \f$ k \f$ and \f$ \chi_0 \f$ must be set for each type of %improper in the simulation using
# improper_coeff.set().
#
# \b Examples:
# \code
# harmonic.improper_coeff.set('heme-ang', k=30.0, chi=1.57)
# harmonic.improper_coeff.set('hdyro-bond', k=20.0, chi=1.57)
# \endcode
#
# \note Specifying the improper.harmonic command when no impropers are defined in the simulation results in an error.
#
# \MPI_SUPPORTED
class harmonic(force._force):
    ## Specify the %harmonic %improper %force
    #
    # \b Example:
    # \code
    # harmonic = improper.harmonic()
    # \endcode
    def __init__(self):
        hoomd.util.print_status_line();
        # check that some impropers are defined
        if hoomd.context.current.system_definition.getImproperData().getNGlobal() == 0:
            hoomd.context.msg.error("No impropers are defined.\n");
            raise RuntimeError("Error creating improper forces");

        # initialize the base class
        force._force.__init__(self);

        self.improper_coeff = coeff();

        # create the c++ mirror class
        if not hoomd.context.exec_conf.isCUDAEnabled():
            self.cpp_force = _md.HarmonicImproperForceCompute(hoomd.context.current.system_definition);
        else:
            self.cpp_force = _md.HarmonicImproperForceComputeGPU(hoomd.context.current.system_definition);

        hoomd.context.current.system.addCompute(self.cpp_force, self.force_name);

        self.required_coeffs = ['k', 'chi'];

    ## \internal
    # \brief Update coefficients in C++
    def update_coeffs(self):
        coeff_list = self.required_coeffs;
        # check that the force coefficients are valid
        if not self.improper_coeff.verify(coeff_list):
           hoomd.context.msg.error("Not all force coefficients are set\n");
           raise RuntimeError("Error updating force coefficients");

        # set all the params
        ntypes = hoomd.context.current.system_definition.getImproperData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(hoomd.context.current.system_definition.getImproperData().getNameByType(i));

        for i in range(0,ntypes):
            # build a dict of the coeffs to pass to proces_coeff
            coeff_dict = {};
            for name in coeff_list:
                coeff_dict[name] = self.improper_coeff.get(type_list[i], name);

            self.cpp_force.setParams(i, coeff_dict['k'], coeff_dict['chi']);

    ## \internal
    # \brief Get metadata
    def get_metadata(self):
        data = force._force.get_metadata(self)

        # make sure coefficients are up-to-date
        self.update_coeffs()

        data['improper_coeff'] = self.improper_coeff
        return data
