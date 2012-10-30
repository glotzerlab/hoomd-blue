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
from hoomd_script import init;

import math;
import sys;


## Defines %angle coefficients
# \brief Defines angle potential coefficients
# The coefficients for all %angle force are specified using this class. Coefficients are
# specified per angle type.
#
# There are two ways to set the coefficients for a particular %angle %force.
# The first way is to save the %angle %force in a variable and call set() directly.
# See below for an example of this.
#
# The second method is to build the coeff class first and then assign it to the
# %angle %force. There are some advantages to this method in that you could specify a
# complicated set of %angle %force coefficients in a separate python file and import
# it into your job script.
#
# Example:
# \code
# my_coeffs = angle.coeff();
# my_angle_force.angle_coeff.set('polymer', k=330.0, r=0.84)
# my_angle_force.angle_coeff.set('backbone', k=330.0, r=0.84)
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

    ## Sets parameters for one angle type
    # \param type Type of angle
    # \param coeffs Named coefficients (see below for examples)
    #
    # Calling set() results in one or more parameters being set for a angle type. Types are identified
    # by name, and parameters are also added by name. Which parameters you need to specify depends on the %angle
    # %force you are setting these coefficients for, see the corresponding documentation.
    #
    # All possible angle types as defined in the simulation box must be specified before executing run().
    # You will receive an error if you fail to do so. It is not an error, however, to specify coefficients for
    # angle types that do not exist in the simulation. This can be useful in defining a %force field for many
    # different types of angles even when some simulations only include a subset.
    #
    # To set the same coefficients between many particle types, provide a list of type names instead of a single
    # one. All types in the list will be set to the same parameters. A convenient wildcard that lists all types
    # of particles in the simulation can be gotten from a saved \c system from the init command.
    #
    # \b Examples:
    # \code
    # my_angle_force.angle_coeff.set('polymer', k=330.0, r0=0.84)
    # my_angle_force.angle_coeff.set('backbone', k=1000.0, r0=1.0)
    # my_angle_force.angle_coeff.set(['angleA','angleB'], k=100, r0=0.0)
    # \endcode
    #
    # \note Single parameters can be updated. If both k and r0 have already been set for a particle type,
    # then executing coeff.set('polymer', r0=1.0) will %update the value of polymer angles and leave the other
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
            globals.msg.error("Cannot verify angle coefficients before initialization\n");
            raise RuntimeError('Error verifying force coefficients');

        # get a list of types from the particle data
        ntypes = globals.system_definition.getAngleData().getNAngleTypes();
        type_list = [];
        for i in xrange(0,ntypes):
            type_list.append(globals.system_definition.getAngleData().getNameByType(i));

        valid = True;
        # loop over all possible types and verify that all required variables are set
        for i in xrange(0,ntypes):
            type = type_list[i];

            if type not in self.values.keys():
                globals.msg.error("Angle type " +str(type) + " not found in angle coeff\n");
                valid = False;
                continue;

            # verify that all required values are set by counting the matches
            count = 0;
            for coeff_name in self.values[type].keys():
                if not coeff_name in required_coeffs:
                    globals.msg.notice(2, "Notice: Possible typo? Force coeff " + str(coeff_name) + " is specified for type " + str(type) + \
                          ", but is not used by the angle force\n");
                else:
                    count += 1;

            if count != len(required_coeffs):
                globals.msg.error("Angle type " + str(type) + " is missing required coefficients\n");
                valid = False;

        return valid;

    ## \internal
    # \brief Gets the value of a single %angle %force coefficient
    # \detail
    # \param type Name of angle type
    # \param coeff_name Coefficient to get
    def get(self, type, coeff_name):
        if type not in self.values.keys():
            globals.msg.error("Bug detected in force.coeff. Please report\n");
            raise RuntimeError("Error setting angle coeff");

        return self.values[type][coeff_name];


## \package hoomd_script.angle
# \brief Commands that specify %angle forces
#
# Angles add forces between specified triplets of particles and are typically used to 
# model chemical angles between two bonds. Angles between particles are set when an input XML file is read
# (init.read_xml) or when an another initializer creates them (like init.create_random_polymers)
#
# By themselves, angles that have been specified in an input file do nothing. Only when you 
# specify an angle force (i.e. angle.harmonic), are forces actually calculated between the 
# listed particles.

## Harmonic %angle force
#
# The command angle.harmonic specifies a %harmonic potential energy between every triplet of particles
# with an angle specified between them.
#
# \f[ V(\theta) = \frac{1}{2} k \left( \theta - \theta_0 \right)^2 \f]
# where \f$ \theta \f$ is the angle between the triplet of particles.
#
# Coefficients:
# - \f$ \theta_0 \f$ - rest %angle (in radians)
# - \f$ k \f$ - %force constant (in units of energy/radians^2)
#
# Coefficients \f$ k \f$ and \f$ \theta_0 \f$ must be set for each type of %angle in the simulation using
# set_coeff().
#
# \note Specifying the angle.harmonic command when no angles are defined in the simulation results in an error.
class harmonic(force._force):
    ## Specify the %harmonic %angle %force
    #
    # \b Example:
    # \code
    # harmonic = angle.harmonic()
    # \endcode
    def __init__(self):
        util.print_status_line();
        # check that some angles are defined
        if globals.system_definition.getAngleData().getNumAngles() == 0:
            globals.msg.error("No angles are defined.\n");
            raise RuntimeError("Error creating angle forces");
        
        # initialize the base class
        force._force.__init__(self);
        
        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = hoomd.HarmonicAngleForceCompute(globals.system_definition);
        else:
            self.cpp_force = hoomd.HarmonicAngleForceComputeGPU(globals.system_definition);
            self.cpp_force.setBlockSize(tune._get_optimal_block_size('angle.harmonic'));

        globals.system.addCompute(self.cpp_force, self.force_name);
        
        # variable for tracking which angle type coefficients have been set
        self.angle_types_set = [];
    
    ## Sets the %harmonic %angle coefficients for a particular %angle type
    #
    # \param angle_type Angle type to set coefficients for
    # \param k Coefficient \f$ k \f$ (in units of energy/radians^2)
    # \param t0 Coefficient \f$ \theta_0 \f$ (in radians)
    #
    # Using set_coeff() requires that the specified %angle %force has been saved in a variable. i.e.
    # \code
    # harmonic = angle.harmonic()
    # \endcode
    #
    # \b Examples:
    # \code
    # harmonic.set_coeff('polymer', k=3.0, t0=0.7851)
    # harmonic.set_coeff('backbone', k=100.0, t0=1.0)
    # \endcode
    #
    # The coefficients for every %angle type in the simulation must be set 
    # before the run() can be started.
    def set_coeff(self, angle_type, k, t0):
        util.print_status_line();
        
        # set the parameters for the appropriate type
        self.cpp_force.setParams(globals.system_definition.getAngleData().getTypeByName(angle_type), k, t0);
        
        # track which particle types we have set
        if not angle_type in self.angle_types_set:
            self.angle_types_set.append(angle_type);
        
    def update_coeffs(self):
        # get a list of all angle types in the simulation
        ntypes = globals.system_definition.getAngleData().getNAngleTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(globals.system_definition.getAngleData().getNameByType(i));
            
        # check to see if all particle types have been set
        for cur_type in type_list:
            if not cur_type in self.angle_types_set:
                globals.msg.error(str(cur_type) + " coefficients missing in angle.harmonic\n");
                raise RuntimeError("Error updating coefficients");

## CGCMM %angle force
#
# The command angle.cgcmm defines a regular %harmonic potential energy between every defined triplet
# of particles in the simulation, but in addition in adds the repulsive part of a CGCMM pair potential
# between the first and the third particle.
#
# Reference \cite Levine2011 describes the CGCMM implementation details in HOOMD-blue. Cite it
# if you utilize the CGCMM potential in your work.
#
# The total potential is thus,
# \f[ V(\theta) = \frac{1}{2} k \left( \theta - \theta_0 \right)^2 \f]
# where \f$ \theta \f$ is the current angle between the three particles
# and either
# \f[ V_{\mathrm{LJ}}(r_{13}) -V_{\mathrm{LJ}}(r_c) \mathrm{~with~~~} V_{\mathrm{LJ}}(r) = 4 \varepsilon \left[ 
#     \left( \frac{\sigma}{r} \right)^{12} - \left( \frac{\sigma}{r} \right)^{6} \right] 
#     \mathrm{~~~~for~} r <= r_c \mathrm{~~~} r_c = \sigma \cdot 2^{\frac{1}{6}} \f],
# or
# \f[ V_{\mathrm{LJ}}(r_{13}) -V_{\mathrm{LJ}}(r_c) \mathrm{~with~~~} 
#     V_{\mathrm{LJ}}(r) = \frac{27}{4} \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{9} - 
#     \left( \frac{\sigma}{r} \right)^{6} \right] 
#     \mathrm{~~~~for~} r <= r_c \mathrm{~~~} r_c = \sigma \cdot \left(\frac{3}{2}\right)^{\frac{1}{3}}\f],
# or
# \f[ V_{\mathrm{LJ}}(r_{13}) -V_{\mathrm{LJ}}(r_c) \mathrm{~with~~~}
#     V_{\mathrm{LJ}}(r) = \frac{3\sqrt{3}}{2} \varepsilon \left[ \left( \frac{\sigma}{r} \right)^{12} - 
#     \left( \frac{\sigma}{r} \right)^{4} \right] 
#     \mathrm{~~~~for~} r <= r_c \mathrm{~~~} r_c = \sigma \cdot 3^{\frac{1}{8}} \f],
#  \f$ r_{13} \f$ being the distance between the two outer particles of the angle.
#
# Coeffients:
# - \f$ \theta_0 \f$ - rest %angle (in radians)
# - \f$ k \f$ - %force constant (in units of energy/radians^2)
# - \f$ \varepsilon \f$ - strength of potential (in energy units)
# - \f$ \sigma \f$ - distance of interaction (in distance units)
#
# Coefficients \f$ k, \theta_0, \varepsilon,\f$ and \f$ \sigma \f$ and Lennard-Jones exponents pair must be set for 
# each type of %angle in the simulation using
# set_coeff().
#
# \note Specifying the angle.cgcmm command when no angles are defined in the simulation results in an error.
class cgcmm(force._force):
    ## Specify the %cgcmm %angle %force
    #
    # \b Example:
    # \code
    # cgcmmAngle = angle.cgcmm()
    # \endcode
    def __init__(self):
        util.print_status_line();
        # check that some angles are defined
        if globals.system_definition.getAngleData().getNumAngles() == 0:
            globals.msg.error("No angles are defined.\n");
            raise RuntimeError("Error creating CGCMM angle forces");
        
        # initialize the base class
        force._force.__init__(self);
        
        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = hoomd.CGCMMAngleForceCompute(globals.system_definition);
        else:
            self.cpp_force = hoomd.CGCMMAngleForceComputeGPU(globals.system_definition);
            self.cpp_force.setBlockSize(tune._get_optimal_block_size('angle.cgcmm'));

        globals.system.addCompute(self.cpp_force, self.force_name);
        
        # variable for tracking which angle type coefficients have been set
        self.angle_types_set = [];
    
    ## Sets the CG-CMM %angle coefficients for a particular %angle type
    #
    # \param angle_type Angle type to set coefficients for
    # \param k Coefficient \f$ k \f$ (in units of energy/radians^2)
    # \param t0 Coefficient \f$ \theta_0 \f$ (in radians)
    # \param exponents is the type of CG-angle exponents we want to use for the repulsion.
    # \param epsilon is the 1-3 repulsion strength (in energy units)
    # \param sigma is the CG particle radius (in distance units)
    #
    # Using set_coeff() requires that the specified CGCMM angle %force has been saved in a variable. i.e.
    # \code
    # cgcmm = angle.cgcmm()
    # \endcode
    #
    # \b Examples (note use of 'exponents' variable):
    # \code
    # cgcmm.set_coeff('polymer', k=3.0, t0=0.7851, exponents=126, epsilon=1.0, sigma=0.53)
    # cgcmm.set_coeff('backbone', k=100.0, t0=1.0, exponents=96, epsilon=23.0, sigma=0.1)
        # cgcmm.set_coeff('residue', k=100.0, t0=1.0, exponents='lj12_4', epsilon=33.0, sigma=0.02)
        # cgcmm.set_coeff('cg96', k=100.0, t0=1.0, exponents='LJ9-6', epsilon=9.0, sigma=0.3)
    # \endcode
    #
    # The coefficients for every CG-CMM angle type in the simulation must be set 
    # before the run() can be started.
    def set_coeff(self, angle_type, k, t0, exponents, epsilon, sigma):
        util.print_status_line();
        cg_type=0
        
        # set the parameters for the appropriate type
        if (exponents == 124) or  (exponents == 'lj12_4') or  (exponents == 'LJ12-4') :
            cg_type=2;

            self.cpp_force.setParams(globals.system_definition.getAngleData().getTypeByName(angle_type), 
                                     k,
                                     t0,
                                     cg_type,
                                     epsilon,
                                     sigma);
    
        elif (exponents == 96) or  (exponents == 'lj9_6') or  (exponents == 'LJ9-6') :
            cg_type=1;

            self.cpp_force.setParams(globals.system_definition.getAngleData().getTypeByName(angle_type),
                                     k,
                                     t0,
                                     cg_type,
                                     epsilon,
                                     sigma);

        elif (exponents == 126) or  (exponents == 'lj12_6') or  (exponents == 'LJ12-6') :
            cg_type=3;
                    
            self.cpp_force.setParams(globals.system_definition.getAngleData().getTypeByName(angle_type),
                                     k,
                                     t0,
                                     cg_type,
                                     epsilon,
                                     sigma);
        else:
            raise RuntimeError("Unknown exponent type.  Must be 'none' or one of MN, ljM_N, LJM-N with M/N in 12/4, 9/6, or 12/6");

        # track which particle types we have set
        if not angle_type in self.angle_types_set:
            self.angle_types_set.append(angle_type);
        
    def update_coeffs(self):
        # get a list of all angle types in the simulation
        ntypes = globals.system_definition.getAngleData().getNAngleTypes();
        type_list = [];
        for i in range(0,ntypes):
            type_list.append(globals.system_definition.getAngleData().getNameByType(i));
            
        # check to see if all particle types have been set
        for cur_type in type_list:
            if not cur_type in self.angle_types_set:
                globals.msg.error(str(cur_type) + " coefficients missing in angle.cgcmm\n");
                raise RuntimeError("Error updating coefficients");



def _table_eval(theta, V, T, width):
      dth = (math.pi) / float(width-1);
      i = int(round((theta)/dth))
      return (V[i], T[i])


## Tabulated %angle %force
#
# The command angle.table specifies that a tabulated  %angle %force should be added to every bonded triple of particles 
# in the simulation.
#
# The %torque \f$ \vec{F}\f$ is (in force units)
# \f{eqnarray*}
#  \vec{F}(\theta)     = & F_{\mathrm{user}}(r)\hat{r} & r \le r_{\mathrm{max}} and  r \ge r_{\mathrm{min}}\\
# \f}
# and the potential \f$ V(\theta) \f$ is (in energy units)
# \f{eqnarray*}
#            = & V_{\mathrm{user}}(\theta) \\
# \f}
# ,where \f$ \theta \f$ is the angle between the triple to the other in the %angle.  
#
# \f$  F_{\mathrm{user}}(r) \f$ and \f$ V_{\mathrm{user}}(r) \f$ are evaluated on *width* grid points between 
# \f$ r_{\mathrm{min}} \f$ and \f$ r_{\mathrm{max}} \f$. Values are interpolated linearly between grid points.
# For correctness, you must specify the force defined by: \f$ F = -\frac{\partial V}{\partial r}\f$  
#
# The following coefficients must be set per unique %pair of particle types.
# - \f$ F_{\mathrm{user}}(\theta) \f$ and \f$ V_{\mathrm{user}}(\theta) \f$ - evaluated by `func` (see example)
# - coefficients passed to `func` - `coeff` (see example)
#
# The table *width* is set once when angle.table is specified (see table.__init__())
# There are two ways to specify the other parameters. 
# 
# \par Example: Set table from a given function
# When you have a functional form for V and F, you can enter that
# directly into python. angle.table will evaluate the given function over \a width points between \a rmin and \a rmax
# and use the resulting values in the table.
# ~~~~~~~~~~~~~
#def harmonic(r, rmin, rmax, kappa, r0):
#    V = 0.5 * kappa * (r-r0)**2;
#    F = -kappa*(r-r0);
#    return (V, F)
#
# btable = angle.table(width=1000)
# btable.angle_coeff.set('angle1', func=harmonic, coeff=dict(kappa=330, r0=0.84))
# btable.angle_coeff.set('angle2', func=harmonic,coeff=dict(kappa=30, r0=1.0))
# ~~~~~~~~~~~~~
#
# \par Example: Set a table from a file
# When you have no function for for *V* or *F*, or you otherwise have the data listed in a file, angle.table can use the given
# values direcly. You must first specify the number of rows in your tables when initializing angle.table. Then use
# table.set_from_file() to read the file.
# ~~~~~~~~~~~~~
# btable = angle.table(width=1000)
# btable.set_from_file('polymer', 'btable.file')
# ~~~~~~~~~~~~~
#
# \par Example: Mix functions and files
# ~~~~~~~~~~~~~
# btable.angle_coeff.set('angle1', func=harmonic, rmin=0.2, rmax=5.0, coeff=dict(kappa=330, r0=0.84))
# btable.set_from_file('angle2', 'btable.file')
# ~~~~~~~~~~~~~
#
#
# \note For potentials that diverge near r=0, make sure to set \c rmin to a reasonable value. If a potential does 
# not diverge near r=0, then a setting of \c rmin=0 is valid.
#
# \note %Angle coefficients for all type angles in the simulation must be
# set before it can be started with run().

class table(force._force):
    ## Specify the Tabulated %angle %force
    #
    # \param width Number of points to use to interpolate V and F (see documentation above)
    # \param name Name of the force instance
    #
    # \b Example:
    # \code
    # def har(theta, kappa, theta_0):
    #   V = 0.5 * kappa * (theta-theta_0)**2;
    #   T = -kappa*(theta-theta_0);
    #   return (V, T)
    #
    # atable = angle.table(width=1000)
    # atable.angle_coeff.set('polymer', func=har, coeff=dict(kappa=330, theta_0=0.84))
    # \endcode
    #
    #
    #
    # \note, be sure that \c rmin and \c rmax cover the range of angle values.  If gpu eror checking is on, a error will
    # be thrown if a angle distance is outside than this range.
    #
    # \note %Pair coefficients for all type angles in the simulation must be
    # set before it can be started with run()
    def __init__(self, width, name=None):
        util.print_status_line();

        # initialize the base class
        force._force.__init__(self, name);


        # create the c++ mirror class
        if not globals.exec_conf.isCUDAEnabled():
            self.cpp_force = hoomd.TableAngleForceCompute(globals.system_definition, int(width), self.name);
        else:
            self.cpp_force = hoomd.TableAngleForceComputeGPU(globals.system_definition, int(width), self.name);
            self.cpp_force.setBlockSize(tune._get_optimal_block_size('angle.table')); 

        globals.system.addCompute(self.cpp_force, self.force_name);

        # setup the coefficent matrix
        self.angle_coeff = coeff();

        # stash the width for later use
        self.width = width;

    def update_angle_table(self, atype, func, coeff):
        # allocate arrays to store V and F
        Vtable = hoomd.std_vector_float();
        Ttable = hoomd.std_vector_float();

        # calculate dth
        dth = math.pi / float(self.width-1);

        # evaluate each point of the function
        for i in xrange(0, self.width):
            theta = dth * i;
            (V,T) = func(theta, **coeff);

            # fill out the tables
            Vtable.append(V);
            Ttable.append(T);

        # pass the tables on to the underlying cpp compute
        self.cpp_force.setTable(atype, Vtable, Ttable);


    def update_coeffs(self):
        # check that the angle coefficents are valid
        if not self.angle_coeff.verify(["func", "coeff"]):
            globals.msg.error("Not all angle coefficients are set for angle.table\n");
            raise RuntimeError("Error updating angle coefficients");

        # set all the params
        ntypes = globals.system_definition.getAngleData().getNAngleTypes();
        type_list = [];
        for i in xrange(0,ntypes):
            type_list.append(globals.system_definition.getAngleData().getNameByType(i));


        # loop through all of the unique type angles and evaluate the table
        for i in xrange(0,ntypes):
            func = self.angle_coeff.get(type_list[i], "func");
            coeff = self.angle_coeff.get(type_list[i], "coeff");

            self.update_angle_table(i, func, coeff);

      ## Set a angle pair interaction from a file
      # \param anglename Name of angle 
      # \param filename Name of the file to read
      #
     # The provided file specifies V and F at equally spaced theta values.
      # Example:
      # \code
      # #t  V    T
      # 0.0 2.0 -3.0
      # 1.5707 3.0 -4.0
      # 3.1414 2.0 -3.0
      #\endcode
      #
      # Note: The theta values are not used by the code.  It is assumed that a table that has N rows will start at 0, end at \pi
      # and that the \delta \theta = \pi/(N-1). The table is read
      # directly into the grid points used to evaluate \f$  T_{\mathrm{user}}(\theta) \f$ and \f$ V_{\mathrm{user}}(\theta) \f$.
    #
    def set_from_file(self, anglename, filename):
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
                  globals.msg.error("angle.table: file must have exactly 3 columns\n");
                  raise RuntimeError("Error reading table file");

              # append to the tables
              theta_table.append(values[0]);
              V_table.append(values[1]);
              T_table.append(values[2]);

          # validate input
          if self.width != len(theta_table):
              globals.msg.error("angle.table: file must have exactly " + str(self.width) + " rows\n");
              raise RuntimeError("Error reading table file");


          # check for even spacing
          dth = math.pi / float(self.width-1);
          for i in xrange(0,self.width):
              theta =  dth * i;
              if math.fabs(theta - theta_table[i]) > 1e-3:
                  globals.msg.error("angle.table: theta must be monotonically increasing and evenly spaced\n");
                  raise RuntimeError("Error reading table file");

          util._disable_status_lines = True;
          self.angle_coeff.set(anglename, func=_table_eval, coeff=dict(V=V_table, T=T_table, width=self.width))
          util._disable_status_lines = True;

