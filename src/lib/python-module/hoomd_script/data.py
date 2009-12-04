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
# Maintainer: joaander

import hoomd

## \package hoomd_script.data
# \brief Access particles, bonds, and other state information inside scripts
#
# Code in the data package provide high-level access to all of the particle, bond and other %data that define the
# current state of the system. By writing python code that modifies this %data, any conceivable initialization of the
# system can be achieved without needing to invoke external tools or generate xml files. Data can be read and additional
# analysis performed during or after simulation runs as well. Basically, the user's imagination is the limit to what can
# be done with the %data.
#
# The only thing to be aware of is that accessing the %data in this way can slow a simulation significantly if performed
# too often. As a general guideline, consider writing a high performance C++ / GPU  plugin (\ref sec_build_plugin)
# if particle %data needs to accessed more often than once every few thousand time steps.
#
# <h2>Documentation by example</h2>
#
# For most of the cases below, it is assumed that the result of the initialization command was saved at the beginning
# of the script, like so:
# \code
# system = init.read_xml(filename="input.xml")
# \endcode
# 
# <h3>Particle properties</h3>
# For a list of all particle properties that can be read and/or set, see the particle_data_proxy. The examples
# here only demonstrate changing a few of them.
#
# With the result of an init command saved in the variable \c system (see above), \c system.particles is a window
# into all of the particles in the system. It behaves like standard python list in many ways.
# - Its length (the number of particles in the system) can be queried
# \code
# >>> len(system.particles)
# 64000
# \endcode
# - A short summary can be printed of the list 
# \code
# >>> print system.particles
# Particle Data for 64000 particles of 1 type(s)
# \endcode
# - Individual particles can be accessed at random.
# \code
# >>> i = 4
# >>> p = system.particles[i]
# \endcode
# - Various properties can be accessed of any particle
# \code
# >>> p.tag
# 4
# >>> p.position
# (27.296911239624023, -3.5986068248748779, 10.364067077636719)
# >>> p.velocity
# (-0.60267972946166992, 2.6205904483795166, -1.7868227958679199)
# >>> p.mass
# 1.0
# >>> p.diameter
# 1.0
# >>> p.type
# 'A'
# \endcode
# (note that p can be replaced with system.particles.[i] above and the results are the same)
# - Particle properties can be set in the same way:
# \code
# >>> p.position = (1,2,3)
# >>> p.position
# (1.0, 2.0, 3.0)
# \endcode
# - Finally, all particles can be easily looped over
# \code
# for p in system.particles:
#     p.velocity = (0,0,0)
# \endcode
#
# Performance is decent, but not great. The for loop above that sets all velocities to 0 takes 0.86 seconds to execute
# on a 2.93 GHz core2 iMac. The interface has been designed to be flexible and easy to use for the widest variety of
# initialization tasks, not efficiency.
#
# There is a second way to access the particle data. Any defined group can be used in exactly the same way as
# \c system.particles above, only the particles accessed will be those just belonging to the group. For a specific
# example, the following will set the velocity of all particles of type A to 0.
# \code
# groupA = group.type(name="a-particles", type='A')
# for p in groupA:
#     p.velocity = (0,0,0)
# \endcode
# <hr>
# <b>Proxy references</b>
# 
# For advanced code using the particle data access from python, it is important to understand that the hoomd_script
# particles are accessed as proxies. This means that after
# \code
# p = system.particles[i]
# \endcode
# is executed, \a p \b doesn't store the position, velocity, ... of particle \a i. Instead, it just stores \a i and
# provides an interface to get/set the properties on demand. This has some side effects. They aren't necessarily 
# bad side effects, just some to be aware of.
# - First, it means that \a p (or any other proxy reference) always references the current state of the particle.
# As an example, note how the position of particle p moves after the run() command.
# \code
# >>> p.position
# (-21.317455291748047, -23.883811950683594, -22.159387588500977)
# >>> run(1000)
# ** starting run **
# ** run complete **
# >>> p.position
# (-19.774742126464844, -23.564577102661133, -21.418502807617188)
# \endcode
# - Second, it means that copies of the proxy reference cannot be changed independently.
# \code
# p.position
# >>> a = p
# >>> a.position
# (-19.774742126464844, -23.564577102661133, -21.418502807617188)
# >>> p.position = (0,0,0)
# >>> a.position
# (0.0, 0.0, 0.0)
# \endcode
#
# If you need to store some particle properties at one time in the simulation and access them again later, you will need
# to make copies of the actual property values themselves and not of the proxy references.
#

## Access system data
#
# system_data provides access to the different data structures that define the current state of the simulation.
# This documentation is intentionally left sparse, see hoomd_script.data for a full explanation of how to use
# system_data, documented by example.
#
class system_data:
    ## \internal
    # \brief create a system_data
    #
    # \param sysdef SystemDefinition to connect
    def __init__(self, sysdef):
        self.sysdef = sysdef;
        self.particles = particle_data(sysdef.getParticleData());

    ## \var sysdef
    # \internal
    # \brief SystemDefinition to which this instance is connected

## Access particle data
#
# particle_data provides access to the per-particle data of all particles in the system.
# This documentation is intentionally left sparse, see hoomd_script.data for a full explanation of how to use
# particle_data, documented by example.
#
class particle_data:
    ## \internal
    # \brief particle_data iterator
    class particle_data_iterator:
        def __init__(self, data):
            self.data = data;
            self.index = 0;
        def __iter__(self):
            return self;
        def next(self):
            if self.index == len(self.data):
                raise StopIteration;
            
            result = self.data[self.index];
            self.index += 1;
            return result;
    
    ## \internal
    # \brief create a particle_data
    #
    # \param pdata ParticleData to connect
    def __init__(self, pdata):
        self.pdata = pdata;
    
    ## \var pdata
    # \internal
    # \brief ParticleData to which this instance is connected

    ## \internal
    # \brief Get a particle_proxy reference to the particle with tag \a tag
    # \param tag Particle tag to access
    def __getitem__(self, tag):
        if tag >= len(self) or tag < 0:
            raise IndexError;
        return particle_data_proxy(self.pdata, tag);
    
    ## \internal
    # \brief Set a particle's properties
    # \param tag Particle tag to set
    # \param p Value containing properties to set
    def __setitem__(self, tag, p):
        raise RuntimeError('__setitem__ not implemented');
    
    ## \internal
    # \brief Get the number of particles
    def __len__(self):
        return self.pdata.getN();
    
    ## \internal
    # \brief Get an informal string representing the object
    def __str__(self):
        result = "Particle Data for %d particles of %d type(s)" % (self.pdata.getN(), self.pdata.getNTypes());
        return result
    
    ## \internal
    # \brief Return an interator
    def __iter__(self):
        return particle_data.particle_data_iterator(self);

## Access a single particle via a proxy
#
# particle_data_proxy provides access to all of the properties of a single particle in the system.
# This documentation is intentionally left sparse, see hoomd_script.data for a full explanation of how to use
# particle_data_proxy, documented by example.
#
# The following attributes are read only:
# - \c tag          : An integer indexing the particle in the system. Tags run from 0 to N-1;
# - \c acceleration : A 3-tuple of floats   (x, y, z) Note that acceleration is a calculated quantity and cannot be set
# - \c typeid       : An integer defining the type id
#
# The following attributes can be both read and set
# - \c position     : A 3-tuple of floats   (x, y, z)
# - \c image        : A 3-tuple of integers (x, y, z)
# - \c velocity     : A 3-tuple of floats   (x, y, z)
# - \c charge       : A single float
# - \c mass         : A single float
# - \c diameter     : A single float
# - \c type         : A string naming the type
#
# In the current version of the API, only already defined type names can be used. A future improvement will allow 
# dynamic creation of new type names from within the python API.
#
class particle_data_proxy:
    ## \internal
    # \brief create a particle_data_proxy
    #
    # \param pdata ParticleData to which this proxy belongs
    # \param tag Tag of this particle in \a pdata
    def __init__(self, pdata, tag):
        self.pdata = pdata;
        self.tag = tag;
    
    ## \internal
    # \brief Get an informal string representing the object
    def __str__(self):
        result = "";
        result += "tag         : " + str(self.tag) + "\n"
        result += "position    : " + str(self.position) + "\n";
        result += "image       : " + str(self.image) + "\n";
        result += "velocity    : " + str(self.velocity) + "\n";
        result += "acceleration: " + str(self.acceleration) + "\n";
        result += "charge      : " + str(self.charge) + "\n";
        result += "mass        : " + str(self.mass) + "\n";
        result += "diameter    : " + str(self.diameter) + "\n";
        result += "type        : " + str(self.type) + "\n";
        result += "typeid      : " + str(self.typeid) + "\n";
        return result;
    
    ## \internal
    # \brief Translate attribute accesses into the low level API function calls
    def __getattr__(self, name):        
        if name == "position":
            pos = self.pdata.getPosition(self.tag);
            return (pos.x, pos.y, pos.z);
        if name == "velocity":
            vel = self.pdata.getVelocity(self.tag);
            return (vel.x, vel.y, vel.z);
        if name == "acceleration":
            accel = self.pdata.getAcceleration(self.tag);
            return (accel.x, accel.y, accel.z);
        if name == "image":
            image = self.pdata.getImage(self.tag);
            return (image.x, image.y, image.z);
        if name == "charge":
            return self.pdata.getCharge(self.tag);
        if name == "mass":
            return self.pdata.getMass(self.tag);
        if name == "diameter":
            return self.pdata.getDiameter(self.tag);
        if name == "typeid":
            return self.pdata.getType(self.tag);
        if name == "type":
            typeid = self.pdata.getType(self.tag);
            return self.pdata.getNameByType(typeid);
        
        # if we get here, we haven't found any names that match, post an error
        raise AttributeError;
    
    ## \internal
    # \brief Translate attribute accesses into the low level API function calls
    def __setattr__(self, name, value):
        if name == "position":
            v = hoomd.Scalar3();
            v.x = float(value[0]);
            v.y = float(value[1]);
            v.z = float(value[2]);
            self.pdata.setPosition(self.tag, v);
            return;
        if name == "velocity":
            v = hoomd.Scalar3();
            v.x = float(value[0]);
            v.y = float(value[1]);
            v.z = float(value[2]);
            self.pdata.setVelocity(self.tag, v);
            return;
        if name == "image":
            v = hoomd.uint3();
            v.x = int(value[0]);
            v.y = int(value[1]);
            v.z = int(value[2]);
            self.pdata.setImage(self.tag, v);
            return;
        if name == "charge":
            self.pdata.setCharge(self.tag, float(value));
            return;
        if name == "mass":
            self.pdata.setMass(self.tag, float(value));
            return;
        if name == "diameter":
            self.pdata.setDiameter(self.tag, value);
            return;
        if name == "type":
            typeid = self.pdata.getTypeByName(value);
            self.pdata.setType(self.tag, typeid);
            return;
        if name == "typeid":
            raise AttributeError;
        if name == "acceleration":
            raise AttributeError;
 
        # otherwise, consider this an internal attribute to be set in the normal way
        self.__dict__[name] = value;
        
## Access force data
#
# particle_data provides access to the per-particle data of all particles in the system.
# This documentation is intentionally left sparse, see hoomd_script.data for a full explanation of how to use
# force_data, documented by example.
#
class force_data:
    ## \internal
    # \brief force_data iterator
    class force_data_iterator:
        def __init__(self, data):
            self.data = data;
            self.index = 0;
        def __iter__(self):
            return self;
        def next(self):
            if self.index == len(self.data):
                raise StopIteration;
            
            result = self.data[self.index];
            self.index += 1;
            return result;
    
    ## \internal
    # \brief create a force_data
    #
    # \param pdata ParticleData to connect
    def __init__(self, force):
        self.force = force;
    
    ## \var force
    # \internal
    # \brief ForceCompute to which this instance is connected

    ## \internal
    # \brief Get a force_proxy reference to the particle with tag \a tag
    # \param tag Particle tag to access
    def __getitem__(self, tag):
        if tag >= len(self) or tag < 0:
            raise IndexError;
        return force_data_proxy(self.force, tag);
    
    ## \internal
    # \brief Set a particle's properties
    # \param tag Particle tag to set
    # \param p Value containing properties to set
    def __setitem__(self, tag, p):
        raise RuntimeError('__setitem__ not implemented');
    
    ## \internal
    # \brief Get the number of particles
    def __len__(self):
        return globals.system_definition.getParticleData().getN();
    
    ## \internal
    # \brief Get an informal string representing the object
    def __str__(self):
        result = "Force Data for %d particles" % (len(self));
        return result
    
    ## \internal
    # \brief Return an interator
    def __iter__(self):
        return force_data.force_data_iterator(self);

## Access the force on a single particle via a proxy
#
# force_data_proxy provides access to the current force, virial, and energy of a single particle due to a single 
# force compuations.
#
# This documentation is intentionally left sparse, see hoomd_script.data for a full explanation of how to use
# force_data_proxy, documented by example.
#
# The following attributes are read only:
# - \c force          : A 3-tuple of floats (x, y, z) listing the current force on the particle
# - \c virial         : A float containing the contribution of this particle to the total virial
# - \c pe             : A float containing the contribution of this particle to the total potential energy
#
class force_data_proxy:
    ## \internal
    # \brief create a force_data_proxy
    #
    # \param force ForceCompute to which this proxy belongs
    # \param tag Tag of this particle in \a force
    def __init__(self, force, tag):
        self.pdata = force;
        self.tag = tag;
    
    ## \internal
    # \brief Get an informal string representing the object
    def __str__(self):
        result = "";
        result += "tag         : " + str(self.tag) + "\n"
        result += "force       : " + str(self.force) + "\n";
        result += "virial      : " + str(self.virial) + "\n";
        result += "energy      : " + str(self.energy) + "\n";
        return result;
    
    ## \internal
    # \brief Translate attribute accesses into the low level API function calls
    def __getattr__(self, name):
        if name == "force":
            f = self.force.getForce(self.tag);
            return (f.x, f.y, f.z);
        if name == "virial":
            return self.force.getVirial(self.tag);
        if name == "energy":
            accel = self.pdata.getEnergy(self.tag);
            return (accel.x, accel.y, accel.z);
        
        # if we get here, we haven't found any names that match, post an error
        raise AttributeError;

