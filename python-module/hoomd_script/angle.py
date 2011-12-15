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

import math;
import sys;

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
# \f[ V(r) = \frac{1}{2} k \left( \theta - \theta_0 \right)^2 \f]
# where \f$ \vec{r} \f$ is the vector pointing from one particle to the other in the %pair.
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
            print >> sys.stderr, "\n***Error! No angles are defined.\n";
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
        for i in xrange(0,ntypes):
            type_list.append(globals.system_definition.getAngleData().getNameByType(i));
            
        # check to see if all particle types have been set
        for cur_type in type_list:
            if not cur_type in self.angle_types_set:
                print >> sys.stderr, "\n***Error:", cur_type, " coefficients missing in angle.harmonic\n";
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
            print >> sys.stderr, "\n***Error! No angles are defined.\n";
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
        for i in xrange(0,ntypes):
            type_list.append(globals.system_definition.getAngleData().getNameByType(i));
            
        # check to see if all particle types have been set
        for cur_type in type_list:
            if not cur_type in self.angle_types_set:
                print >> sys.stderr, "\n***Error:", cur_type, " coefficients missing in angle.cgcmm\n";
                raise RuntimeError("Error updating coefficients");


