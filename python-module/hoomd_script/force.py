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

import globals;
import sys;
import hoomd;
import util;
import data;
import init;

## \package hoomd_script.force
# \brief Other types of forces
#
# This package contains various forces that don't belong in any of the other categories

## \internal
# \brief Base class for forces
#
# A force in hoomd_script reflects a ForceCompute in c++. It is responsible
# for all high-level management that happens behind the scenes for hoomd_script
# writers. 1) The instance of the c++ analyzer itself is tracked and added to the
# System 2) methods are provided for disabling the force from being added to the
# net force on each particle
class _force:
    ## \internal
    # \brief Constructs the force
    #
    # \param name name of the force instance 
    #
    # Initializes the cpp_analyzer to None.
    # If specified, assigns a name to the instance
    # Assigns a name to the force in force_name;
    def __init__(self, name=None):
        # check if initialization has occured
        if not init.is_initialized():
            print >> sys.stderr, "\n***Error! Cannot create force before initialization\n";
            raise RuntimeError('Error creating force');
        
        # Allow force to store a name.  Used for discombobulation in the logger
        if name is None:    
            self.name = "";
        else:
            self.name="_" + name;
        
        self.cpp_force = None;

        # increment the id counter
        id = _force.cur_id;
        _force.cur_id += 1;
        
        self.force_name = "force%d" % (id);
        self.enabled = True;
        self.log =True;
        globals.forces.append(self);
        
        # create force data iterator
        self.forces = data.force_data(self);

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
        

    ## Disables the force
    # \param log Set to True if you plan to continue logging the potential energy associated with this force.
    #
    # \b Examples:
    # \code
    # force.disable()
    # force.disable(log=True)
    # \endcode
    #
    # Executing the disable command will remove the force from the simulation.
    # Any run() command executed after disabling a force will not calculate or 
    # use the force during the simulation. A disabled force can be re-enabled
    # with enable()
    #
    # By setting \a log to True, the values of the force can be logged even though the forces are not applied 
    # in the simulation.  For forces that use cutoff radii, setting \a log=True will cause the correct r_cut values 
    # to be used throughout the simulation, and therefore possibly drive the neighbor list size larger than it
    # otherwise would be. If \a log is left False, the potential energy associated with this force will not be
    # available for logging.
    #
    # To use this command, you must have saved the force in a variable, as 
    # shown in this example:
    # \code
    # force = pair.some_force()
    # # ... later in the script
    # force.disable()
    # force.disable(log=True)
    # \endcode
    def disable(self, log=False):
        util.print_status_line();
        self.check_initialization();
            
        # check if we are already disabled
        if not self.enabled:
            print "***Warning! Ignoring command to disable a force that is already disabled";
            return;
        
        self.enabled = False;
        self.log = log;
        
        # remove the compute from the system if it is not going to be logged
        if not log:
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
    # force = pair.some_force()
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
        
        # add the compute back to the system if it was removed
        if not self.log:
            globals.system.addCompute(self.cpp_force, self.force_name);
        
        self.enabled = True;
        self.log = True;
        
    ## \internal
    # \brief updates force coefficients
    def update_coeffs(self):
        pass
        raise RuntimeError("_force.update_coeffs should not be called");
        # does nothing: this is for derived classes to implement
    
# set default counter
_force.cur_id = 0;


## Constant %force
#
# The command force.constant specifies that a %constant %force should be added to every
# particle in the simulation.
#
class constant(_force):
    ## Specify the %constant %force
    #
    # \param fx x-component of the %force (in force units)
    # \param fy y-component of the %force (in force units)
    # \param fz z-component of the %force (in force units)
    #
    # \b Examples:
    # \code
    # force.constant(fx=1.0, fy=0.5, fz=0.25)
    # const = force.constant(fx=0.4, fy=1.0, fz=0.5)
    # \endcode
    def __init__(self, fx, fy, fz):
        util.print_status_line();
        
        # initialize the base class
        _force.__init__(self);
        
        # create the c++ mirror class
        self.cpp_force = hoomd.ConstForceCompute(globals.system_definition, fx, fy, fz);
            
        globals.system.addCompute(self.cpp_force, self.force_name);
        
    ## Change the value of the force
    #
    # \param fx New x-component of the %force (in force units)
    # \param fy New y-component of the %force (in force units)
    # \param fz New z-component of the %force (in force units)
    #
    # Using set_force() requires that you saved the created %constant %force in a variable. i.e.
    # \code
    # const = force.constant(fx=0.4, fy=1.0, fz=0.5)
    # \endcode
    #
    # \b Example:
    # \code
    # const.set_force(fx=0.2, fy=0.1, fz=-0.5)
    # \endcode
    def set_force(self, fx, fy, fz):
        self.check_initialization();
            
        self.cpp_force.setForce(fx, fy, fz);
        
    # there are no coeffs to update in the constant force compute
    def update_coeffs(self):
        pass

class const_external_field_dipole(_force):
    ## Specicify the %constant %field and %dipole moment
    #
    # \param field_x x-component of the %field (units?)
    # \param field_y y-component of the %field (units?)
    # \param field_z z-component of the %field (units?)
    # \param p magnitude of the particles' dipole moment in z direction
    # \b Examples:
    # \code
    # force.external_field_dipole(field_x=0.0, field_y=1.0 ,field_z=0.5, p=1.0)
    # const_ext_f_dipole = force.external_field_dipole(field_x=0.0, field_y=1.0 ,field_z=0.5, p=1.0)
    # \endcode
    def __init__(self, field_x,field_y,field_z,p):
        util.print_status_line()
        
        # initialize the base class
        _force.__init__(self)
        
        # create the c++ mirror class
        self.cpp_force = hoomd.ConstExternalFieldDipoleForceCompute(globals.system_definition, field_x, field_y, field_z, p)
        
        globals.system.addCompute(self.cpp_force, self.force_name)
        #


    ## Change the %constant %field and %dipole moment
    #
    # \param field_x x-component of the %field (units?)
    # \param field_y y-component of the %field (units?)
    # \param field_z z-component of the %field (units?)
    # \param p magnitude of the particles' dipole moment in z direction
    # \b Examples:
    # \code
    # const_ext_f_dipole = force.external_field_dipole(field_x=0.0, field_y=1.0 ,field_z=0.5, p=1.0)
    # const_ext_f_dipole.setParams(field_x=0.1, field_y=0.1, field_z=0.0, p=1.0))
    # \endcode
    def set_params(field_x, field_y,field_z,p):
        self.check_initialization()
        
        self.cpp_force.setParams(field_x,field_y,field_z,p)
        
    # there are no coeffs to update in the constant ExternalFieldDipoleForceCompute
    def update_coeffs(self):
        pass


