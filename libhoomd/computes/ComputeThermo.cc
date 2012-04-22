/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif

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
    m_exec_conf->msg->notice(5) << "Constructing ComputeThermo" << endl;

    assert(m_pdata);
    GPUArray< Scalar > properties(10, exec_conf);
    m_properties.swap(properties);

    m_logname_list.push_back(string("temperature") + suffix);
    m_logname_list.push_back(string("pressure") + suffix);
    m_logname_list.push_back(string("kinetic_energy") + suffix);
    m_logname_list.push_back(string("potential_energy") + suffix);
    m_logname_list.push_back(string("ndof") + suffix);
    m_logname_list.push_back(string("num_particles") + suffix);
    }

ComputeThermo::~ComputeThermo()
    {
    m_exec_conf->msg->notice(5) << "Destroying ComputeThermo" << endl;
    }

/*! \param ndof Number of degrees of freedom to set
*/
void ComputeThermo::setNDOF(unsigned int ndof)
    {
    if (ndof == 0)
        {
        m_exec_conf->msg->warning() << "compute.thermo: given a group with 0 degrees of freedom." << endl
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
        m_exec_conf->msg->error() << "compute.thermo: " << quantity << " is not a valid log quantity" << endl;
        throw runtime_error("Error getting log value");
        }
    }

/*! Computes all thermodynamic properties of the system in one fell swoop.
*/
void ComputeThermo::computeProperties()
    {
    unsigned int group_size = m_group->getNumLocalMembers();
    // just drop out if the group is an empty group
    if (group_size == 0)
        return;

#ifdef ENABLE_MPI
    boost::shared_ptr<const boost::mpi::communicator> mpi_comm;
    if (m_comm)
        {
        mpi_comm = m_comm->getMPICommunicator();
        assert(mpi_comm);
        }
#endif

    if (m_prof) m_prof->push("Thermo");
    
    assert(m_pdata);
    assert(m_ndof != 0);
    
    // access the particle data
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);

    // access the net force, pe, and virial
    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
    const GPUArray< Scalar >& net_virial = m_pdata->getNetVirial();
    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);
    ArrayHandle<Scalar> h_net_virial(net_virial, access_location::host, access_mode::read);

    // total kinetic energy 
    double ke_total = 0.0;

    PDataFlags flags = m_pdata->getFlags();

    double pressure_kinetic_xx = 0.0;
    double pressure_kinetic_xy = 0.0;
    double pressure_kinetic_xz = 0.0;
    double pressure_kinetic_yy = 0.0;
    double pressure_kinetic_yz = 0.0;
    double pressure_kinetic_zz = 0.0;

    if (flags[pdata_flag::pressure_tensor])
        {
        // Calculate kinetic part of pressure tensor
        for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
            {
            unsigned int j = m_group->getMemberIndex(group_idx);
            double mass = h_vel.data[j].w;
            pressure_kinetic_xx += mass*(  (double)h_vel.data[j].x * (double)h_vel.data[j].x );
            pressure_kinetic_xy += mass*(  (double)h_vel.data[j].x * (double)h_vel.data[j].y );
            pressure_kinetic_xz += mass*(  (double)h_vel.data[j].x * (double)h_vel.data[j].z );
            pressure_kinetic_yy += mass*(  (double)h_vel.data[j].y * (double)h_vel.data[j].y );
            pressure_kinetic_yz += mass*(  (double)h_vel.data[j].y * (double)h_vel.data[j].z );
            pressure_kinetic_zz += mass*(  (double)h_vel.data[j].z * (double)h_vel.data[j].z );
            }
#ifdef ENABLE_MPI
        if (mpi_comm)
            {
            pressure_kinetic_xx = all_reduce(*mpi_comm, pressure_kinetic_xx, std::plus<double>());
            pressure_kinetic_xy = all_reduce(*mpi_comm, pressure_kinetic_xy, std::plus<double>());
            pressure_kinetic_xz = all_reduce(*mpi_comm, pressure_kinetic_xz, std::plus<double>());
            pressure_kinetic_yy = all_reduce(*mpi_comm, pressure_kinetic_yy, std::plus<double>());
            pressure_kinetic_yz = all_reduce(*mpi_comm, pressure_kinetic_yz, std::plus<double>());
            pressure_kinetic_zz = all_reduce(*mpi_comm, pressure_kinetic_zz, std::plus<double>());
            }
#endif
        // kinetic energy = 1/2 trace of kinetic part of pressure tensor
        ke_total = Scalar(0.5)*(pressure_kinetic_xx + pressure_kinetic_yy + pressure_kinetic_zz);
        }
    else
        {
        // total kinetic energy
        for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
            {
            unsigned int j = m_group->getMemberIndex(group_idx);
            ke_total += (double)h_vel.data[j].w*( (double)h_vel.data[j].x * (double)h_vel.data[j].x
                                                + (double)h_vel.data[j].y * (double)h_vel.data[j].y
                                                + (double)h_vel.data[j].z * (double)h_vel.data[j].z);

            }

        ke_total *= Scalar(0.5);
#ifdef ENABLE_MPI
        if (mpi_comm)
            ke_total = all_reduce(*mpi_comm, ke_total, std::plus<double>());
#endif
        }
    
    // total potential energy 
    double pe_total = 0.0;
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        pe_total += (double)h_net_force.data[j].w;
        }
#ifdef ENABLE_MPI
    if (mpi_comm)
        pe_total = all_reduce(*mpi_comm, pe_total, std::plus<double>());
#endif
 

    double W = 0.0;
    double virial_xx = 0.0;
    double virial_xy = 0.0;
    double virial_xz = 0.0;
    double virial_yy = 0.0;
    double virial_yz = 0.0;
    double virial_zz = 0.0;

    if (flags[pdata_flag::pressure_tensor])
        {
        // Calculate symmetrized virial tensor
        unsigned int virial_pitch = net_virial.getPitch();
        for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
            {
            unsigned int j = m_group->getMemberIndex(group_idx);
            virial_xx += (double)h_net_virial.data[j+0*virial_pitch];
            virial_xy += (double)h_net_virial.data[j+1*virial_pitch];
            virial_xz += (double)h_net_virial.data[j+2*virial_pitch];
            virial_yy += (double)h_net_virial.data[j+3*virial_pitch];
            virial_yz += (double)h_net_virial.data[j+4*virial_pitch];
            virial_zz += (double)h_net_virial.data[j+5*virial_pitch];
            }
#ifdef ENABLE_MPI
        if (mpi_comm)
            {
            virial_xx = all_reduce(*mpi_comm, virial_xx, std::plus<double>());
            virial_xy = all_reduce(*mpi_comm, virial_xy, std::plus<double>());
            virial_xz = all_reduce(*mpi_comm, virial_xz, std::plus<double>());
            virial_yy = all_reduce(*mpi_comm, virial_yy, std::plus<double>());
            virial_yz = all_reduce(*mpi_comm, virial_yz, std::plus<double>());
            virial_zz = all_reduce(*mpi_comm, virial_zz, std::plus<double>());
            }
#endif

        if (flags[pdata_flag::isotropic_virial])
            {
            // isotropic virial = 1/3 trace of virial tensor
            W = Scalar(1./3.) * (virial_xx + virial_yy + virial_zz);
            }
        }
     else if (flags[pdata_flag::isotropic_virial])
        {
        // only sum up isotropic part of virial tensor
        unsigned int virial_pitch = net_virial.getPitch();
        for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
            {
            unsigned int j = m_group->getMemberIndex(group_idx);
            W += Scalar(1./3.)* ((double)h_net_virial.data[j+0*virial_pitch] +
                                 (double)h_net_virial.data[j+3*virial_pitch] +
                                 (double)h_net_virial.data[j+5*virial_pitch] );
            }
#ifdef ENABLE_MPI
        if (mpi_comm)
            W = all_reduce(*mpi_comm, W, std::plus<double>());
#endif
        }

    // compute the temperature
    Scalar temperature = Scalar(2.0) * Scalar(ke_total) / Scalar(m_ndof);

    // compute the pressure
    // volume/area & other 2D stuff needed
    BoxDim global_box = m_pdata->getGlobalBox();

    Scalar3 L = global_box.getL();
    Scalar volume;
    unsigned int D = m_sysdef->getNDimensions();
    if (D == 2)
        {
        // "volume" is area in 2D
        volume = L.x * L.y;
        // W needs to be corrected since the 1/3 factor is built in
        W *= Scalar(3.0/2.0);
        }
    else
        {
        volume = L.x * L.y * L.z;
        }

    // pressure: P = (N * K_B * T + W)/V
    Scalar pressure =  (2.0 * ke_total / Scalar(D) + W) / volume;

    // pressure tensor = (kinetic part + virial) / V
    Scalar pressure_xx = (pressure_kinetic_xx + virial_xx) / volume;
    Scalar pressure_xy = (pressure_kinetic_xy + virial_xy) / volume;
    Scalar pressure_xz = (pressure_kinetic_xz + virial_xz) / volume;
    Scalar pressure_yy = (pressure_kinetic_yy + virial_yy) / volume;
    Scalar pressure_yz = (pressure_kinetic_yz + virial_yz) / volume;
    Scalar pressure_zz = (pressure_kinetic_zz + virial_zz) / volume;

    // fill out the GPUArray
    ArrayHandle<Scalar> h_properties(m_properties, access_location::host, access_mode::overwrite);
    h_properties.data[thermo_index::temperature] = temperature;
    h_properties.data[thermo_index::pressure] = pressure;
    h_properties.data[thermo_index::kinetic_energy] = Scalar(ke_total);
    h_properties.data[thermo_index::potential_energy] = Scalar(pe_total);
    h_properties.data[thermo_index::pressure_xx] = pressure_xx;
    h_properties.data[thermo_index::pressure_xy] = pressure_xy;
    h_properties.data[thermo_index::pressure_xz] = pressure_xz;
    h_properties.data[thermo_index::pressure_yy] = pressure_yy;
    h_properties.data[thermo_index::pressure_yz] = pressure_yz;
    h_properties.data[thermo_index::pressure_zz] = pressure_zz;


   if(PPPMData::compute_pppm_flag) {
        Scalar2 pppm_thermo = ComputeThermo::PPPM_thermo_compute_cpu();
//        ArrayHandle<float> h_properties(m_properties, access_location::host, access_mode::readwrite);
        h_properties.data[thermo_index::pressure] += pppm_thermo.x;
        h_properties.data[thermo_index::potential_energy] += pppm_thermo.y;
        PPPMData::pppm_energy = pppm_thermo.y;
        }

    if (m_prof) m_prof->pop();
    }

Scalar2 ComputeThermo::PPPM_thermo_compute_cpu() 
    {
    BoxDim box = m_pdata->getBox();
    Scalar3 L = box.getL();

    ArrayHandle<cufftComplex> d_rho_real_space(PPPMData::m_rho_real_space, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar> d_green_hat(PPPMData::m_green_hat, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar3> d_vg(PPPMData::m_vg, access_location::host, access_mode::readwrite);
    Scalar2 pppm_virial_energy = make_scalar2(0.0, 0.0);

    for (int i = 0; i < PPPMData::Nx*PPPMData::Ny*PPPMData::Nz; i++) 
        {
        Scalar energy = d_green_hat.data[i]*(d_rho_real_space.data[i].x*d_rho_real_space.data[i].x +
                                             d_rho_real_space.data[i].y*d_rho_real_space.data[i].y);
        Scalar pressure = energy*(d_vg.data[i].x + d_vg.data[i].y + d_vg.data[i].z);
        pppm_virial_energy.x += pressure;
        pppm_virial_energy.y += energy;
        }
    pppm_virial_energy.x *= PPPMData::energy_virial_factor/ (3.0f * L.x * L.y * L.z);
    pppm_virial_energy.y *= PPPMData::energy_virial_factor;
    pppm_virial_energy.y -= PPPMData::q2 * PPPMData::kappa / 1.772453850905516027298168f;
    pppm_virial_energy.y -= 0.5*M_PI*PPPMData::q*PPPMData::q / (PPPMData::kappa*PPPMData::kappa* L.x * L.y * L.z);
    return pppm_virial_energy;


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

