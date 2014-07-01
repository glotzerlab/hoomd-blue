# -- start license --
# Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
# (HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
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

from hoomd_script import force;
from hoomd_script import globals;
import hoomd;
from hoomd_script import util;
from hoomd_script import tune;
from hoomd_script import init;

import math;
import sys;

## Defines %dihedral coefficients
# \brief Defines dihedral potential coefficients
# The coefficients for all %dihedral force are specified using this class. Coefficients are
# specified per dihedral type.
#
# There are two ways to set the coefficients for a particular %dihedral %force.
# The first way is to save the %dihedral %force in a variable and call set() directly.
# See below for an example of this.
#
# The second method is to build the coeff class first and then assign it to the
# %dihedral %force. There are some advantages to this method in that you could specify a
# complicated set of %dihedral %force coefficients in a separate python file and import
# it into your job script.
#
# Example:
# \code
# my_coeffs = dihedral.coeff();
# my_dihedral_force.dihedral_coeff.set('polymer', k=330.0, r=0.84)
# my_dihedral_force.dihedral_coeff.set('backbone', k=330.0, r=0.84)
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

    ## Sets parameters for one dihedral type
    # \param type Type of dihedral
    # \param coeffs Named coefficients (see below for examples)
    #
    # Calling set() results in one or more parameters being set for a dihedral type. Types are identified
    # by name, and parameters are also added by name. Which parameters you need to specify depends on the %dihedral
    # %force you are setting these coefficients for, see the corresponding documentation.
    #
    # All possible dihedral types as defined in the simulation box must be specified before executing run().
    # You will receive an error if you fail to do so. It is not an error, however, to specify coefficients for
    # dihedral types that do not exist in the simulation. This can be useful in defining a %force field for many
    # different types of dihedrals even when some simulations only include a subset.
    #
    # To set the same coefficients between many particle types, provide a list of type names instead of a single
    # one. All types in the list will be set to the same parameters. A convenient wildcard that lists all types
    # of particles in the simulation can be gotten from a saved \c system from the init command.
    #
    # \b Examples:
    # \code
    # my_dihedral_force.dihedral_coeff.set('polymer', k=330.0, r0=0.84)
    # my_dihedral_force.dihedral_coeff.set('backbone', k=1000.0, r0=1.0)
    # my_dihedral_force.dihedral_coeff.set(['dihedralA','dihedralB'], k=100, r0=0.0)
    # \endcode
    #
    # \note Single parameters can be updated. If both k and r0 have already been set for a particle type,
    # then executing coeff.set('polymer', r0=1.0) will %update the value of polymer dihedrals and leave the other
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
            globals.msg.error("No coefficents specified\n");
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
            globals.msg.error("Cannot verify dihedral coefficients before initialization\n");
            raise RuntimeError('Error verifying force coefficients');

        # get a list of types from the particle data
        ntypes = globals.system_definition.getDihedralData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(globals.system_definition.getDihedralData().getNameByType(i));

        valid = True;
        # loop over all possible types and verify that all required variables are set
        for i in range(0,ntypes):
            type = type_list[i];

            if type not in self.values.keys():
                globals.msg.error("Dihedral type " +str(type) + " not found in dihedral coeff\n");
                valid = False;
                continue;

            # verify that all required values are set by counting the matches
            count = 0;
            for coeff_name in self.values[type].keys():
                if not coeff_name in required_coeffs:
                    globals.msg.notice(2, "Notice: Possible typo? Force coeff " + str(coeff_name) + " is specified for type " + str(type) + \
                          ", but is not used by the dihedral force\n");
                else:
                    count += 1;

            if count != len(required_coeffs):
                globals.msg.error("Dihedral type " + str(type) + " is missing required coefficients\n");
                valid = False;

        return valid;

    ## \internal
    # \brief Gets the value of a single %dihedral %force coefficient
    # \detail
    # \param type Name of dihedral type
    # \param coeff_name Coefficient to get
    def get(self, type, coeff_name):
        if type not in self.values.keys():
            globals.msg.error("Bug detected in force.coeff. Please report\n");
            raise RuntimeError("Error setting dihedral coeff");

        return self.values[type][coeff_name];

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

## Harmonic %dihedral force
#
# The command dihedral.harmonic specifies a %harmonic dihedral potential energy between every defined
# quadruplet of particles in the simulation.
# \f[ V(r) = \frac{1}{2}k \left( 1 + d \cos\left(n * \phi(r) \right) \right) \f]
# where \f$ \phi \f$ is angle between two sides of the dihedral
#
# Coefficients:
# - \f$ k \f$ - strength of %force (in energy units)
# - \f$ d \f$ - sign factor (unitless)
# - \f$ n \f$ - angle scaling factor (unitless)
#
# Coefficients \f$ k \f$, \f$ d \f$, \f$ n \f$ and  must be set for each type of %dihedral in the simulation using
# set_coeff().
#
# \note Specifying the dihedral.harmonic command when no dihedrals are defined in the simulation results in an error.
# \MPI_SUPPORTED
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
        if globals.system_definition.getDihedralData().getNGlobal() == 0:
            globals.msg.error("No dihedrals are defined.\n");
            raise RuntimeError("Error creating dihedral forces");

        # initialize the base class
        force._force.__init__(self);

        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = hoomd.HarmonicDihedralForceCompute(globals.system_definition);
        else:
            self.cpp_force = hoomd.HarmonicDihedralForceComputeGPU(globals.system_definition);

        globals.system.addCompute(self.cpp_force, self.force_name);

        # variable for tracking which dihedral type coefficients have been set
        self.dihedral_types_set = [];

    ## Sets the %harmonic %dihedral coefficients for a particular %dihedral type
    #
    # \param dihedral_type Dihedral type to set coefficients for
    # \param k Coefficient \f$ k \f$ in the %force (in energy units)
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
        ntypes = globals.system_definition.getDihedralData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(globals.system_definition.getDihedralData().getNameByType(i));

        # check to see if all particle types have been set
        for cur_type in type_list:
            if not cur_type in self.dihedral_types_set:
                globals.msg.error(str(cur_type) + " coefficients missing in dihedral.harmonic\n");
                raise RuntimeError("Error updating coefficients");


## Tabulated %dihedral %force
#
# The command dihedral.table specifies that a tabulated  %dihedral %force should be added to every bonded triple of particles
# in the simulation.
#
# \f$  T_{\mathrm{user}}(\theta) \f$ and \f$ V_{\mathrm{user}}(\theta) \f$ are evaluated on *width* grid points between
# \f$ -\pi \f$ and \f$ \pi \f$. Values are interpolated linearly between grid points.
# For correctness, you must specify the derivative of the potential with respect to the dihedral angle,
# defined by: \f$ T = -\frac{\partial V}{\partial \theta} \f$
#
# The following coefficients must be set per unique %pair of particle types.
# - \f$ T_{\mathrm{user}}(\theta) \f$ and \f$ V_{\mathrm{user}} (\theta) \f$ - evaluated by `func` (see example)
# - coefficients passed to `func` - `coeff` (see example)
#
# The table *width* is set once when dihedral.table is specified (see table.__init__())
# There are two ways to specify the other parameters.
#
# \par Example: Set table from a given function
# When you have a functional form for V and T, you can enter that
# directly into python. dihedral.table will evaluate the given function over \a width points between \f$ -\pi \f$ and \f$ \pi \f$
# and use the resulting values in the table.
# ~~~~~~~~~~~~~
#def harmonic(theta, kappa, theta0):
#    V = 0.5 * kappa * (theta-theta0)**2;
#    F = -kappa*(theta-theta0);
#    return (V, F)
#
# dtable = dihedral.table(width=1000)
# dtable.dihedral_coeff.set('dihedral1', func=harmonic, coeff=dict(kappa=330, theta_0=0.0))
# dtable.dihedral_coeff.set('dihedral2', func=harmonic,coeff=dict(kappa=30, theta_0=1.0))
# ~~~~~~~~~~~~~
#
# \par Example: Set a table from a file
# When you have no function for for *V* or *T*, or you otherwise have the data listed in a file, dihedral.table can use the given
# values direcly. You must first specify the number of rows in your tables when initializing dihedral.table. Then use
# table.set_from_file() to read the file.
# ~~~~~~~~~~~~~
# dtable = dihedral.table(width=1000)
# dtable.set_from_file('polymer', 'dihedral.dat')
# ~~~~~~~~~~~~~
#
# \par Example: Mix functions and files
# ~~~~~~~~~~~~~
# dtable.dihedral_coeff.set('dihedral1', func=harmonic, coeff=dict(kappa=330, theta_0=0.0))
# dtable.set_from_file('dihedral2', 'dihedral.dat')
# ~~~~~~~~~~~~~
#
# \note %Dihedral coefficients for all type dihedrals in the simulation must be
# set before it can be started with run().
# \MPI_SUPPORTED
class table(force._force):
    ## Specify the Tabulated %dihedral %force
    #
    # \param width Number of points to use to interpolate V and T (see documentation above)
    # \param name Name of the force instance
    #
    # \b Example:
    # \code
    # def har(theta, kappa, theta_0):
    #   V = 0.5 * kappa * (theta-theta_0)**2;
    #   T = -kappa*(theta-theta_0);
    #   return (V, T)
    #
    # dtable = dihedral.table(width=1000)
    # dtable.dihedral_coeff.set('polymer', func=har, coeff=dict(kappa=330, theta_0=1.0))
    # \endcode
    #
    # \note coefficients for all type dihedrals in the simulation must be
    # set before it can be started with run()
    def __init__(self, width, name=None):
        util.print_status_line();

        # initialize the base class
        force._force.__init__(self, name);


        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = hoomd.TableDihedralForceCompute(globals.system_definition, int(width), self.name);
        else:
            self.cpp_force = hoomd.TableDihedralForceComputeGPU(globals.system_definition, int(width), self.name);

        globals.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficent matrix
        self.dihedral_coeff = coeff();

        # stash the width for later use
        self.width = width;

    def update_dihedral_table(self, atype, func, coeff):
        # allocate arrays to store V and F
        Vtable = hoomd.std_vector_scalar();
        Ttable = hoomd.std_vector_scalar();

        # calculate dth
        dth = 2.0*math.pi / float(self.width-1);

        # evaluate each point of the function
        for i in range(0, self.width):
            theta = -math.pi+dth * i;
            (V,T) = func(theta, **coeff);

            # fill out the tables
            Vtable.append(V);
            Ttable.append(T);

        # pass the tables on to the underlying cpp compute
        self.cpp_force.setTable(atype, Vtable, Ttable);


    def update_coeffs(self):
        # check that the dihedral coefficents are valid
        if not self.dihedral_coeff.verify(["func", "coeff"]):
            globals.msg.error("Not all dihedral coefficients are set for dihedral.table\n");
            raise RuntimeError("Error updating dihedral coefficients");

        # set all the params
        ntypes = globals.system_definition.getDihedralData().getNTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(globals.system_definition.getDihedralData().getNameByType(i));


        # loop through all of the unique type dihedrals and evaluate the table
        for i in range(0,ntypes):
            func = self.dihedral_coeff.get(type_list[i], "func");
            coeff = self.dihedral_coeff.get(type_list[i], "coeff");

            self.update_dihedral_table(i, func, coeff);

    ## Set a dihedral pair interaction from a file
    # \param dihedralname Name of dihedral
    # \param filename Name of the file to read
    #
    # The provided file specifies V and F at equally spaced theta values.
    # Example:
    # \code
    # #t  V    T
    # -3.1414 2.0 -3.0
    # 1.5707 3.0 - 4.0
    # 0.0 2.0 -3.0
    # 1.5707 3.0 -4.0
    # 3.1414 2.0 -3.0
    #\endcode
    #
    # Note: The theta values are not used by the code.  It is assumed that a table that has N rows will start at \f$ -\pi \f$, end at \f$ \pi \f$
    # and that \f$ \delta \theta = 2\pi/(N-1) \f$. The table is read
    # directly into the grid points used to evaluate \f$  T_{\mathrm{user}}(\theta) \f$ and \f$ V_{\mathrm{user}}(\theta) \f$.
    #
    def set_from_file(self, dihedralname, filename):
          util.print_status_line();

          # open the file
          f = open(filename);

          theta_table = [];
          V_table = [];
          T_table = [];

          # read in lines from the file
          for line in f.readlines():
              line = line.strip();

              # skip comment lines
              if line[0] == '#':
                  continue;

              # split out the columns
              cols = line.split();
              values = [float(f) for f in cols];

              # validate the input
              if len(values) != 3:
                  globals.msg.error("dihedral.table: file must have exactly 3 columns\n");
                  raise RuntimeError("Error reading table file");

              # append to the tables
              theta_table.append(values[0]);
              V_table.append(values[1]);
              T_table.append(values[2]);

          # validate input
          if self.width != len(r_table):
              globals.msg.error("dihedral.table: file must have exactly " + str(self.width) + " rows\n");
              raise RuntimeError("Error reading table file");


          # check for even spacing
          dth = 2.0*math.pi / float(self.width-1);
          for i in range(0,self.width):
              theta =  -math.pi+dnth * i;
              if math.fabs(theta - theta_table[i]) > 1e-3:
                  globals.msg.error("dihedral.table: theta must be monotonically increasing and evenly spaced, going from -pi to pi");
                  raise RuntimeError("Error reading table file");

          util._disable_status_lines = True;
          self.dihedral_coeff.set(dihedralname, func=_table_eval, coeff=dict(V=V_table, T=T_table, width=self.width))
          util._disable_status_lines = True;
