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

from hoomd_script import force;
from hoomd_script import globals;
import hoomd;
from hoomd_script import util;
from hoomd_script import tune;
from hoomd_script import data;
from hoomd_script import init;
from hoomd_script import pair;

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

## Defines %bond coefficients
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
    # \param coeffs Named coefficients (see below for examples)
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
    # my_bond_force.bond_coeff.set('polymer', k=330.0, r0=0.84)
    # my_bond_force.bond_coeff.set('backbone', k=1000.0, r0=1.0)
    # my_bond_force.bond_coeff.set(['bondA','bondB'], k=100, r0=0.0)
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
            globals.msg.error("Cannot verify bond coefficients before initialization\n");
            raise RuntimeError('Error verifying force coefficients');

        # get a list of types from the particle data
        ntypes = globals.system_definition.getBondData().getNBondTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(globals.system_definition.getBondData().getNameByType(i));

        valid = True;
        # loop over all possible types and verify that all required variables are set
        for i in range(0,ntypes):
            type = type_list[i];

            if type not in self.values.keys():
                globals.msg.error("Bond type " +str(type) + " not found in bond coeff\n");
                valid = False;
                continue;

            # verify that all required values are set by counting the matches
            count = 0;
            for coeff_name in self.values[type].keys():
                if not coeff_name in required_coeffs:
                    globals.msg.notice(2, "Notice: Possible typo? Force coeff " + str(coeff_name) + " is specified for type " + str(type) + \
                          ", but is not used by the bond force\n");
                else:
                    count += 1;

            if count != len(required_coeffs):
                globals.msg.error("Bond type " + str(type) + " is missing required coefficients\n");
                valid = False;

        return valid;

    ## \internal
    # \brief Gets the value of a single %bond %force coefficient
    # \detail
    # \param type Name of bond type
    # \param coeff_name Coefficient to get
    def get(self, type, coeff_name):
        if type not in self.values.keys():
            globals.msg.error("Bug detected in force.coeff. Please report\n");
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
           globals.msg.error("Not all force coefficients are set\n");
           raise RuntimeError("Error updating force coefficients");

        # set all the params
        ntypes = globals.system_definition.getBondData().getNBondTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(globals.system_definition.getBondData().getNameByType(i));

        for i in range(0,ntypes):
            # build a dict of the coeffs to pass to proces_coeff
            coeff_dict = {};
            for name in coeff_list:
                coeff_dict[name] = self.bond_coeff.get(type_list[i], name);

            param = self.process_coeff(coeff_dict);
            self.cpp_force.setParams(i, param);


## Harmonic %bond force
#
# The command bond.harmonic specifies a %harmonic potential energy between every bonded %bond of particles
# in the simulation. 
# \f[ V(r) = \frac{1}{2} k \left( r - r_0 \right)^2 \f]
# where \f$ \vec{r} \f$ is the vector pointing from one particle to the other in the %bond.
#
# Coeffients:
# - \f$ k \f$ - %force constant (in units of energy/distance^2)
# - \f$ r_0 \f$ - %bond rest length (in distance units)
#
# Coefficients \f$ k \f$ and \f$ r_0 \f$ must be set for each type of %bond in the simulation using
# \link hoomd_script.bond.coeff.set bond_coeff.set()\endlink
# \note For compatibility with older versions of HOOMD-blue, the syntax set_coeff() is also supported.
#
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
            globals.msg.error("No bonds are defined.\n");
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

    ## Set parameters for %harmonic %bond %force  (\b deprecated)
    # \param type bond type
    # \param coeffs named bond coefficients
    def set_coeff(self, type, **coeffs):
        globals.msg.warning("Syntax bond.harmonic.set_coeff deprecated.\n");
        self.bond_coeff.set(type,**coeffs)
        
    def process_coeff(self, coeff):
        k = coeff['k'];
        r0 = coeff['r0'];

        # set the parameters for the appropriate type
        return hoomd.make_scalar2(k, r0);


## FENE %bond force
#
# The command bond.fene specifies a %fene potential energy between every bonded %bond of particles
# in the simulation. 
# \f[ V(r) = - \frac{1}{2} k r_0^2 \ln \left( 1 - \left( \frac{r - \Delta}{r_0} \right)^2 \right) + V_{\mathrm{WCA}}(r)\f]
# where \f$ \vec{r} \f$ is the vector pointing from one particle to the other in the %bond,
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
# each type of %bond in the simulation using
# \link bond.coeff.set bond_coeff.set()\endlink.
# \note For compatibility with older versions of HOOMD-blue, the syntax set_coeff() is also supported.
class fene(_bond):
    ## Specify the %fene %bond %force
    #
    # \param name Name of the bond instance
    #
    # \b Example:
    # \code
    # fene = bond.fene()
    # fene.bond_coeff.set('polymer', k=30.0, r0=1.5, sigma=1.0, epsilon= 2.0)
    # fene.bond_coeff.set('backbone', k=100.0, r0=1.0, sigma=1.0, epsilon= 2.0)
    # \endcode
    def __init__(self, name=None):
        util.print_status_line();
        
        # check that some bonds are defined
        if globals.system_definition.getBondData().getNumBonds() == 0:
            globals.msg.error("No bonds are defined.\n");
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

    ## Set parameters for %fene %bond %force (\b deprecated)
    # \param type bond type
    # \param coeffs named bond coefficients
    def set_coeff(self, type, **coeffs):
        globals.msg.warning("Syntax bond.fene.set_coeff deprecated.\n");
        self.bond_coeff.set(type, **coeffs)

    def process_coeff(self, coeff):
        k = coeff['k'];
        r0 = coeff['r0'];
        lj1 = 4.0 * coeff['epsilon'] * math.pow(coeff['sigma'], 12.0);
        lj2 = 4.0 * coeff['epsilon'] * math.pow(coeff['sigma'], 6.0);
        return hoomd.make_scalar4(k, r0, lj1, lj2);




def _table_eval(r, rmin, rmax, V, F, width):
      dr = (rmax - rmin) / float(width-1);
      i = int(round((r - rmin)/dr))
      return (V[i], F[i])


## Tabulated %bond %force
#
# The command bond.table specifies that a tabulated  %bond %force should be added to everybonded %bond of particles 
# in the simulation.
#
# The %force \f$ \vec{F}\f$ is (in force units)
# \f{eqnarray*}
#  \vec{F}(\vec{r})     = & F_{\mathrm{user}}(r)\hat{r} & r \le r_{\mathrm{max}} and  r \ge r_{\mathrm{min}}\\
# \f}
# and the potential \f$ V(r) \f$ is (in energy units)
# \f{eqnarray*}
#            = & V_{\mathrm{user}}(r) & r \le r_{\mathrm{max}} and  r \ge r_{\mathrm{min}}\\
# \f}
# ,where \f$ \vec{r} \f$ is the vector pointing from one particle to the other in the %bond.  Care should be taken to 
# define the range of the bond so that it is not possible for the distance between two bonded particles to be outside the
# specified range.  On the CPU, this will throw an error.  On the GPU, this will throw an error if error checking is enabled.
#
# \f$  F_{\mathrm{user}}(r) \f$ and \f$ V_{\mathrm{user}}(r) \f$ are evaluated on *width* grid points between 
# \f$ r_{\mathrm{min}} \f$ and \f$ r_{\mathrm{max}} \f$. Values are interpolated linearly between grid points.
# For correctness, you must specify the force defined by: \f$ F = -\frac{\partial V}{\partial r}\f$  
#
# The following coefficients must be set per unique %pair of particle types.
# - \f$ F_{\mathrm{user}}(r) \f$ and \f$ V_{\mathrm{user}}(r) \f$ - evaluated by `func` (see example)
# - coefficients passed to `func` - `coeff` (see example)
# - \f$ r_{\mathrm{min}} \f$ - `rmin` (in distance units)
# - \f$ r_{\mathrm{max}} \f$ - `rmax` (in distance units)
#
# The table *width* is set once when bond.table is specified (see table.__init__())
# There are two ways to specify the other parameters. 
# 
# \par Example: Set table from a given function
# When you have a functional form for V and F, you can enter that
# directly into python. bond.table will evaluate the given function over \a width points between \a rmin and \a rmax
# and use the resulting values in the table.
# ~~~~~~~~~~~~~
#def harmonic(r, rmin, rmax, kappa, r0):
#    V = 0.5 * kappa * (r-r0)**2;
#    F = -kappa*(r-r0);
#    return (V, F)
#
# btable = bond.table(width=1000)
# btable.bond_coeff.set('bond1', func=harmonic, rmin=0.2, rmax=5.0, coeff=dict(kappa=330, r0=0.84))
# btable.bond_coeff.set('bond2', func=harmonic, rmin=0.2, rmax=5.0, coeff=dict(kappa=30, r0=1.0))
# ~~~~~~~~~~~~~
#
# \par Example: Set a table from a file
# When you have no function for for *V* or *F*, or you otherwise have the data listed in a file, bond.table can use the given
# values direcly. You must first specify the number of rows in your tables when initializing bond.table. Then use
# table.set_from_file() to read the file.
# ~~~~~~~~~~~~~
# btable = bond.table(width=1000)
# btable.set_from_file('polymer', 'btable.file')
# ~~~~~~~~~~~~~
#
# \par Example: Mix functions and files
# ~~~~~~~~~~~~~
# btable.bond_coeff.set('bond1', func=harmonic, rmin=0.2, rmax=5.0, coeff=dict(kappa=330, r0=0.84))
# btable.set_from_file('bond2', 'btable.file')
# ~~~~~~~~~~~~~
#
#
# \note For potentials that diverge near r=0, make sure to set \c rmin to a reasonable value. If a potential does 
# not diverge near r=0, then a setting of \c rmin=0 is valid.
#
# \note Coefficients for all bond types in the simulation must be
# set before it can be started with run().
class table(force._force):
    ## Specify the Tabulated %bond %force
    #
    # \param width Number of points to use to interpolate V and F (see documentation above)
    # \param name Name of the force instance
    #
    # \b Example:
    # \code
    # def har(r, rmin, rmax, kappa, r0):
    #   V = 0.5 * kappa * (r-r0)**2;
    #   F = -kappa*(r-r0);
    #   return (V, F)
    #
    # btable = bond.table(width=1000)
    # btable.bond_coeff.set('polymer', func=har, rmin=0.1, rmax=10.0, coeff=dict(kappa=330, r0=0.84))
    # \endcode
    #
    # \note For potentials that diverge near r=0, make sure to set \c rmin to a reasonable value. If a potential does
    # not diverge near r=0, then a setting of \c rmin=0 is valid.
    #
    # \note Be sure that \c rmin and \c rmax cover the range of bond values.  If gpu eror checking is on, a error will
    # be thrown if a bond distance is outside than this range.
    def __init__(self, width, name=None):
        util.print_status_line();

        # initialize the base class
        force._force.__init__(self, name);


        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = hoomd.BondTablePotential(globals.system_definition, int(width), self.name);
        else:
            self.cpp_force = hoomd.BondTablePotentialGPU(globals.system_definition, int(width), self.name);
            self.cpp_force.setBlockSize(tune._get_optimal_block_size('bond.table')); 

        globals.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficent matrix
        self.bond_coeff = coeff();

        # stash the width for later use
        self.width = width;

    def update_bond_table(self, btype, func, rmin, rmax, coeff):
        # allocate arrays to store V and F
        Vtable = hoomd.std_vector_scalar();
        Ftable = hoomd.std_vector_scalar();

        # calculate dr
        dr = (rmax - rmin) / float(self.width-1);

        # evaluate each point of the function
        for i in range(0, self.width):
            r = rmin + dr * i;
            (V,F) = func(r, rmin, rmax, **coeff);

            # fill out the tables
            Vtable.append(V);
            Ftable.append(F);

        # pass the tables on to the underlying cpp compute
        self.cpp_force.setTable(btype, Vtable, Ftable, rmin, rmax);


    def update_coeffs(self):
        # check that the bond coefficents are valid
        if not self.bond_coeff.verify(["func", "rmin", "rmax", "coeff"]):
            globals.msg.error("Not all bond coefficients are set for bond.table\n");
            raise RuntimeError("Error updating bond coefficients");

        # set all the params
        ntypes = globals.system_definition.getBondData().getNBondTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(globals.system_definition.getBondData().getNameByType(i));


        # loop through all of the unique type bonds and evaluate the table
        for i in range(0,ntypes):
            func = self.bond_coeff.get(type_list[i], "func");
            rmin = self.bond_coeff.get(type_list[i], "rmin");
            rmax = self.bond_coeff.get(type_list[i], "rmax");
            coeff = self.bond_coeff.get(type_list[i], "coeff");

            self.update_bond_table(i, func, rmin, rmax, coeff);

    ## Set a bond pair interaction from a file
    # \param bondname Name of bond 
    # \param filename Name of the file to read
    #
    # The provided file specifies V and F at equally spaced r values.
    # Example:
    # \code
    # #r  V    F
    # 1.0 2.0 -3.0
    # 1.1 3.0 -4.0
    # 1.2 2.0 -3.0
    # 1.3 1.0 -2.0
    # 1.4 0.0 -1.0
    # 1.5 -1.0 0.0
    #\endcode
    #
    # The first r value sets \a rmin, the last sets \a rmax. Any line with \# as the first non-whitespace character is
    # is treated as a comment. The \a r values must monotonically increase and be equally spaced. The table is read
    # directly into the grid points used to evaluate \f$  F_{\mathrm{user}}(r) \f$ and \f$ V_{\mathrm{user}}(r) \f$.
    #
    def set_from_file(self, bondname, filename):
          util.print_status_line();

          # open the file
          f = open(filename);

          r_table = [];
          V_table = [];
          F_table = [];

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
                  globals.msg.error("bond.table: file must have exactly 3 columns\n");
                  raise RuntimeError("Error reading table file");

              # append to the tables
              r_table.append(values[0]);
              V_table.append(values[1]);
              F_table.append(values[2]);

          # validate input
          if self.width != len(r_table):
              globals.msg.error("bond.table: file must have exactly " + str(self.width) + " rows\n");
              raise RuntimeError("Error reading table file");

          # extract rmin and rmax
          rmin_table = r_table[0];
          rmax_table = r_table[-1];

          # check for even spacing
          dr = (rmax_table - rmin_table) / float(self.width-1);
          for i in range(0,self.width):
              r = rmin_table + dr * i;
              if math.fabs(r - r_table[i]) > 1e-3:
                  globals.msg.error("bond.table: r must be monotonically increasing and evenly spaced\n");
                  raise RuntimeError("Error reading table file");

          util._disable_status_lines = True;
          self.bond_coeff.set(bondname, func=_table_eval, rmin=rmin_table, rmax=rmax_table, coeff=dict(V=V_table, F=F_table, width=self.width))
          util._disable_status_lines = True;
