/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

/*! \file ComputeThermo.cc
    \brief Contains code for the ComputeThermo class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "ComputeThermo.h"

#include <boost/python.hpp>
using namespace boost::python;

#include <iostream>
using namespace std;

/*! \param sysdef System for which to compute thermodynamic properties
    \param group Subset of the system over which properties are calculated
    \param suffix Suffix to append to all logged quantity names
*/
ComputeThermo::ComputeThermo(boost::shared_ptr<SystemDefinition> sysdef,
                             boost::shared_ptr<ParticleGroup> group,
                             const std::string& suffix)
    : Compute(sysdef), m_group(group), m_ndof(1)
    {
    assert(m_pdata);
    GPUArray< Scalar > properties(4, exec_conf);
    m_properties.swap(properties);

    m_logname_list.push_back(string("temperature") + suffix);
    m_logname_list.push_back(string("pressure") + suffix);
    m_logname_list.push_back(string("kinetic_energy") + suffix);
    m_logname_list.push_back(string("potential_energy") + suffix);
    m_logname_list.push_back(string("ndof") + suffix);
    m_logname_list.push_back(string("num_particles") + suffix);
    }


/*! \param ndof Number of degrees of freedom to set
*/
void ComputeThermo::setNDOF(unsigned int ndof)
    {
    if (ndof == 0)
        {
        cout << "***Warning! compute.thermo specified for a group with 0 degrees of freedom." << endl
             << "            overriding ndof=1 to avoid divide by 0 errors" << endl;
        ndof = 1;
        }

    m_ndof = ndof;
    }

/*! Calls computeProperties if the properties need updating
    \param timestep Current time step of the simulation
*/
void ComputeThermo::compute(unsigned int timestep)
    {
    if (!shouldCompute(timestep))
        return;
        
    computeProperties();
    }

std::vector< std::string > ComputeThermo::getProvidedLogQuantities()
    {
    return m_logname_list;
    }

Scalar ComputeThermo::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    compute(timestep);
    if (quantity == m_logname_list[0])
        {
        return getTemperature();
        }
    else if (quantity == m_logname_list[1])
        {
        return getPressure();
        }
    else if (quantity == m_logname_list[2])
        {
        return getKineticEnergy();
        }
    else if (quantity == m_logname_list[3])
        {
        return getPotentialEnergy();
        }
    else if (quantity == m_logname_list[4])
        {
        return Scalar(m_ndof);
        }
    else if (quantity == m_logname_list[5])
        {
        return Scalar(m_group->getNumMembers());
        }
    else
        {
        cerr << endl << "***Error! " << quantity << " is not a valid log quantity for ComputeThermo" << endl << endl;
        throw runtime_error("Error getting log value");
        }
    }

/*! Computes all thermodynamic properties of the system in one fell swoop.
*/
void ComputeThermo::computeProperties()
    {
    unsigned int group_size = m_group->getNumMembers();
    // just drop out if the group is an empty group
    if (group_size == 0)
        return;

    if (m_prof) m_prof->push("Thermo");
    
    assert(m_pdata);
    assert(m_ndof != 0);
    
    // access the particle data
    const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly();

    // access the net force, pe, and virial
    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
    const GPUArray< Scalar >& net_virial = m_pdata->getNetVirial();
    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_net_virial(net_virial, access_location::host, access_mode::read);

    // total kinetic energy 
    double ke_total = 0.0;
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        ke_total += (double)arrays.mass[j]*(  (double)arrays.vx[j] * (double)arrays.vx[j] 
                                            + (double)arrays.vy[j] * (double)arrays.vy[j] 
                                            + (double)arrays.vz[j] * (double)arrays.vz[j]);
        }
    ke_total *= 0.5;
    
    // total potential energy 
    double pe_total = 0.0;
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        pe_total += (double)h_net_force.data[j].w;
        }

    // total the virial
    double W = 0.0;
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        W += (double)h_net_virial.data[j];
        }

    m_pdata->release();

    // compute the temperature
    Scalar temperature = Scalar(2.0) * Scalar(ke_total) / Scalar(m_ndof);

    // compute the pressure
    // volume/area & other 2D stuff needed
    BoxDim box = m_pdata->getBox();
    Scalar volume;
    unsigned int D = m_sysdef->getNDimensions();
    if (D == 2)
        {
        // "volume" is area in 2D
        volume = (box.xhi - box.xlo)*(box.yhi - box.ylo);
        // W needs to be corrected since the 1/3 factor is built in
        W *= Scalar(3.0/2.0);
        }
    else
        {
        volume = (box.xhi - box.xlo)*(box.yhi - box.ylo)*(box.zhi-box.zlo);
        }

    // pressure: P = (N * K_B * T + W)/V
    Scalar pressure =  (2.0 * ke_total / Scalar(D) + W) / volume;
    
    // fill out the GPUArray
    ArrayHandle<Scalar> h_properties(m_properties, access_location::host, access_mode::overwrite);
    h_properties.data[thermo_index::temperature] = temperature;
    h_properties.data[thermo_index::pressure] = pressure;
    h_properties.data[thermo_index::kinetic_energy] = Scalar(ke_total);
    h_properties.data[thermo_index::potential_energy] = Scalar(pe_total);
    
    if (m_prof) m_prof->pop();
    }

void export_ComputeThermo()
    {
    class_<ComputeThermo, boost::shared_ptr<ComputeThermo>, bases<Compute>, boost::noncopyable >
    ("ComputeThermo", init< boost::shared_ptr<SystemDefinition>,
                      boost::shared_ptr<ParticleGroup>,
                      const std::string& >())
    .def("setNDOF", &ComputeThermo::setNDOF)
    .def("getTemperature", &ComputeThermo::getTemperature)
    .def("getPressure", &ComputeThermo::getPressure)
    .def("getKineticEnergy", &ComputeThermo::getKineticEnergy)
    .def("getPotentialEnergy", &ComputeThermo::getPotentialEnergy)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

