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

/*! \file Integrator.cc
    \brief Defines the Integrator base class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "Integrator.h"

#include <boost/bind.hpp>
using namespace boost;

#ifdef ENABLE_CUDA
#include "Integrator.cuh"
#endif

using namespace std;

/*! \param sysdef System to update
    \param deltaT Time step to use
*/
Integrator::Integrator(boost::shared_ptr<SystemDefinition> sysdef, Scalar deltaT) : Updater(sysdef), m_deltaT(deltaT)
    {
    if (m_deltaT <= 0.0)
        cout << "***Warning! A timestep of less than 0.0 was specified to an integrator" << endl;
    }

Integrator::~Integrator()
    {
    }

/*! \param fc ForceCompute to add
*/
void Integrator::addForceCompute(boost::shared_ptr<ForceCompute> fc)
    {
    assert(fc);
    m_forces.push_back(fc);
    fc->setDeltaT(m_deltaT);      
    }

/*! \param fc ForceConstraint to add
*/
void Integrator::addForceConstraint(boost::shared_ptr<ForceConstraint> fc)
    {
    assert(fc);
    m_constraint_forces.push_back(fc);
    fc->setDeltaT(m_deltaT);
    }

/*! Call removeForceComputes() to completely wipe out the list of force computes
    that the integrator uses to sum forces.
*/
void Integrator::removeForceComputes()
    {
    m_forces.clear();
    m_constraint_forces.clear();
    }

/*! \param deltaT New time step to set
*/
void Integrator::setDeltaT(Scalar deltaT)
    {
    if (m_deltaT <= 0.0)
        cout << "***Warning! A timestep of less than 0.0 was specified to an integrator" << endl;
        
    for (unsigned int i=0; i < m_forces.size(); i++)
        m_forces[i]->setDeltaT(deltaT);    

    for (unsigned int i=0; i < m_constraint_forces.size(); i++)
        m_constraint_forces[i]->setDeltaT(deltaT);
   
     m_deltaT = deltaT;
    }

/*! \return the timestep deltaT
*/
Scalar Integrator::getDeltaT()
    {
    return m_deltaT;
    }

/*! Loops over all constraint forces in the Integrator and sums up the number of DOF removed
*/
unsigned int Integrator::getNDOFRemoved()
    {
    // start counting at 0
    unsigned int n = 0;
    
    // loop through all constraint forces
    std::vector< boost::shared_ptr<ForceConstraint> >::iterator force_compute;
    for (force_compute = m_constraint_forces.begin(); force_compute != m_constraint_forces.end(); ++force_compute)
        n += (*force_compute)->getNDOFRemoved();

    return n;
    }

/*! The base class Integrator provides all of the common logged quantities. This is the most convenient and
    sensible place to put it because most of the common quantities are computed by the various integrators.
    That, and there must be an integrator in any sensible simulation.

    \b ALL common quantities that are logged are specified in this getProvidedLogQuantities(). They are computed
    explicitly when requested by getLogValue(). Derived integrators may compute quantities like temperature or
    pressure for their own purposes. They are free (and encouraged) to provide an overridden call that returns
    the already computed value in that case.

    Derived integrators may also want to add additional quantities. They can do this in
    getProvidedLogQuantities() by calling Integrator::getProvidedLogQuantities() and adding their own custom
    provided quantities before returning.

    Integrator provides:
        - num_particles
        - volume
        - temperature
        - pressure
        - kinetic_energy
        - potential_energy
        - momentum
        - conserved_quantity

    See Logger for more information on what this is about.
*/
std::vector< std::string > Integrator::getProvidedLogQuantities()
    {
    vector<string> result;
    result.push_back("volume");
    result.push_back("momentum");
    result.push_back("conserved_quantity");
    return result;
    }

/*! \param quantity Name of the log quantity to get
    \param timestep Current time step of the simulation

    The Integrator base class will provide a number of quantities (see getProvidedLogQuantities()). Derived
    classes that calculate any of these on their own can (and should) return their calculated values. To do so
    an overridden getLogValue() should have the following logic:
    \code
    if (quantitiy == "my_calculated_quantitiy1")
        return my_calculated_quantity1;
    else if (quantitiy == "my_calculated_quantitiy2")
        return my_calculated_quantity2;
    else return Integrator::getLogValue(quantity, timestep);
    \endcode
    In this way the "overriden" quantity is handled by the derived class and any other quantities are passed up
    to the base class to be handled there.

    See Logger for more information on what this is about.
*/
Scalar Integrator::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == "volume")
        {
        BoxDim box = m_pdata->getBox();
        return (box.xhi - box.xlo)*(box.yhi - box.ylo)*(box.zhi-box.zlo);
        }
    else if (quantity == "momentum")
        return computeTotalMomentum(timestep);
    else if (quantity == "conserved_quantity")
        {
        cout << "***Warning! The integrator you are using doesn't report conserved_quantitiy, logging a value of 0.0"
             << endl;
        return Scalar(0.0);
        }
    else
        {
        cerr << endl << "***Error! " << quantity << " is not a valid log quantity for Integrator" << endl;
        throw runtime_error("Error getting log value");
        }
    }

/*! \param timestep Current timestep
    \param profiler_name Name of the profiler element to continue timing under
    \post \c arrays.ax, \c arrays.ay, and \c arrays.az are set based on the forces computed by the ForceComputes
*/
void Integrator::computeAccelerations(unsigned int timestep, const std::string& profiler_name)
    {
    // compute the net forces
    computeNetForce(timestep, profiler_name);
    
    if (m_prof)
        {
        m_prof->push(profiler_name);
        m_prof->push("Sum accel");
        }
    
    // now, get our own access to the arrays and calculate the accelerations
    ParticleDataArrays arrays = m_pdata->acquireReadWrite();
    ArrayHandle<Scalar4> h_net_force(m_pdata->getNetForce(), access_location::host, access_mode::read);
    
    // now, add up the accelerations
    for (unsigned int j = 0; j < arrays.nparticles; j++)
        {
        Scalar minv = Scalar(1.0) / arrays.mass[j];
        arrays.ax[j] = h_net_force.data[j].x*minv;
        arrays.ay[j] = h_net_force.data[j].y*minv;
        arrays.az[j] = h_net_force.data[j].z*minv;
        }
    
    m_pdata->release();
    
    if (m_prof)
        {
        m_prof->pop();
        m_prof->pop();
        }
    }

/*! \param timestep Current time step of the simulation

    computePotentialEnergy()  accesses the virial data of all attached force computes and calculates the
    total on the CPU
*/
Scalar Integrator::computePotentialEnergy(unsigned int timestep)
    {
    // total up the potential energy from the various force computes
    double pe_total = 0.0;
    for (unsigned int i=0; i < m_forces.size(); i++)
        {
        m_forces[i]->compute(timestep);
        pe_total += m_forces[i]->calcEnergySum();
        }
    return pe_total;
    }

/*! \param timestep Current time step of the simulation

    computeTotalMomentum()  accesses the particle data on the CPU, loops through it and calculates the magnitude of the total
    system momentum
*/
Scalar Integrator::computeTotalMomentum(unsigned int timestep)
    {
    // grab access to the particle data
    const ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
    
    // sum up the kinetic energy
    double p_tot_x = 0.0;
    double p_tot_y = 0.0;
    double p_tot_z = 0.0;
    for (unsigned int i=0; i < m_pdata->getN(); i++)
        {
        p_tot_x += (double)arrays.mass[i]*(double)arrays.vx[i];
        p_tot_y += (double)arrays.mass[i]*(double)arrays.vy[i];
        p_tot_z += (double)arrays.mass[i]*(double)arrays.vz[i];
        }
        
    double p_tot = sqrt(p_tot_x * p_tot_x + p_tot_y * p_tot_y + p_tot_z * p_tot_z) / Scalar(m_pdata->getN());
    
    // done!
    m_pdata->release();
    return Scalar(p_tot);
    }

/*! \param timestep Current time step of the simulation
    \param profile_name Name to profile the force summation under
    \post All added force computes in \a m_forces are computed and totaled up in \a m_net_force and \a m_net_virial
    \note The summation step is performed <b>on the CPU</b> and will result in a lot of data traffic back and forth
          if the forces and/or integrater are on the GPU. Call computeNetForcesGPU() to sum the forces on the GPU
*/
void Integrator::computeNetForce(unsigned int timestep, const std::string& profile_name)
    {
    // compute all the forces first
    std::vector< boost::shared_ptr<ForceCompute> >::iterator force_compute;
    for (force_compute = m_forces.begin(); force_compute != m_forces.end(); ++force_compute)
        (*force_compute)->compute(timestep);
    
    if (m_prof)
        {
        m_prof->push(profile_name);
        m_prof->push("Net force");
        }
    
        {
        // access the net force and virial arrays
        const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
        const GPUArray< Scalar >& net_virial = m_pdata->getNetVirial();
        ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar> h_net_virial(net_virial, access_location::host, access_mode::overwrite);
        
        // start by zeroing the net force and virial arrays
        memset((void *)h_net_force.data, 0, sizeof(Scalar4)*net_force.getNumElements());
        memset((void *)h_net_virial.data, 0, sizeof(Scalar)*net_virial.getNumElements());
        
        // now, add up the net forces
        unsigned int nparticles = m_pdata->getN();
        assert(nparticles == net_force.getNumElements());
        assert(nparticles == net_virial.getNumElements());
        for (force_compute = m_forces.begin(); force_compute != m_forces.end(); ++force_compute)
            {
            ForceDataArrays force_arrays = (*force_compute)->acquire();
        
            for (unsigned int j = 0; j < nparticles; j++)
                {
                h_net_force.data[j].x += force_arrays.fx[j];
                h_net_force.data[j].y += force_arrays.fy[j];
                h_net_force.data[j].z += force_arrays.fz[j];
                h_net_force.data[j].w += force_arrays.pe[j];
                h_net_virial.data[j] += force_arrays.virial[j];
                }
            }
        }
    
    if (m_prof)
        {
        m_prof->pop();
        m_prof->pop();
        }

    // return early if there are no constraint forces
    if (m_constraint_forces.size() == 0)
        return;

    // compute all the constraint forces next
    std::vector< boost::shared_ptr<ForceConstraint> >::iterator force_constraint;
    for (force_constraint = m_constraint_forces.begin(); force_constraint != m_constraint_forces.end(); ++force_constraint)
        (*force_constraint)->compute(timestep);
    
    if (m_prof)
        {
        m_prof->push(profile_name);
        m_prof->push("Net force");
        }
    
        {
        // access the net force and virial arrays
        const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
        const GPUArray< Scalar >& net_virial = m_pdata->getNetVirial();
        ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::overwrite);
        ArrayHandle<Scalar> h_net_virial(net_virial, access_location::host, access_mode::overwrite);

        // now, add up the net forces
        unsigned int nparticles = m_pdata->getN();
        assert(nparticles == net_force.getNumElements());
        assert(nparticles == net_virial.getNumElements());
        for (force_constraint = m_constraint_forces.begin(); force_constraint != m_constraint_forces.end(); ++force_constraint)
            {
            ForceDataArrays force_arrays = (*force_constraint)->acquire();
        
            for (unsigned int j = 0; j < nparticles; j++)
                {
                h_net_force.data[j].x += force_arrays.fx[j];
                h_net_force.data[j].y += force_arrays.fy[j];
                h_net_force.data[j].z += force_arrays.fz[j];
                h_net_force.data[j].w += force_arrays.pe[j];
                h_net_virial.data[j] += force_arrays.virial[j];
                }
            }
        }
    
    if (m_prof)
        {
        m_prof->pop();
        m_prof->pop();
        }
    }

#ifdef ENABLE_CUDA
/*! \param timestep Current time step of the simulation
    \param profile_name Name to profile the force summation under
    \post All added force computes in \a m_forces are computed and totaled up in \a m_net_force and \a m_net_virial
    \note The summation step is performed <b>on the GPU</b>.
*/
void Integrator::computeNetForceGPU(unsigned int timestep, const std::string& profile_name)
    {
    if (exec_conf.gpu.size() != 1)
        {
        cerr << endl << "***Error! Only 1 GPU is supported" << endl << endl;
        throw runtime_error("Error computing accelerations");
        }
    
    // compute all the normal forces first
    std::vector< boost::shared_ptr<ForceCompute> >::iterator force_compute;
    for (force_compute = m_forces.begin(); force_compute != m_forces.end(); ++force_compute)
        (*force_compute)->compute(timestep);
    
    if (m_prof)
        {
        m_prof->push(profile_name);
        m_prof->push(exec_conf, "Net force");
        }
    
        {
        // access the net force and virial arrays
        const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
        const GPUArray< Scalar >& net_virial = m_pdata->getNetVirial();
        ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_net_virial(net_virial, access_location::device, access_mode::overwrite);

        unsigned int nparticles = m_pdata->getN();
        assert(nparticles == net_force.getNumElements());
        assert(nparticles == net_virial.getNumElements());
        
        // there is no need to zero out the initial net force and virial here, the first call to the addition kernel
        // will do that
        // ahh!, but we do need to zer out the net force and virial if there are 0 forces!
        if (m_forces.size() == 0)
            {
            // start by zeroing the net force and virial arrays
            exec_conf.gpu[0]->call(bind(cudaMemset, d_net_force.data, 0, sizeof(Scalar4)*nparticles));
            exec_conf.gpu[0]->call(bind(cudaMemset, d_net_virial.data, 0, sizeof(Scalar)*nparticles));
            }
        
        // now, add up the accelerations
        // sum all the forces into the net force
        // perform the sum in groups of 6 to avoid kernel launch and memory access overheads
        for (unsigned int cur_force = 0; cur_force < m_forces.size(); cur_force += 6)
            {
            // grab the device pointers for the current set
            gpu_force_list force_list;
            const vector<ForceDataArraysGPU>& force0 = m_forces[cur_force]->acquireGPU();
            force_list.f0 = force0[0].d_data.force;
            force_list.v0 = force0[0].d_data.virial;
            
            if (cur_force+1 < m_forces.size())
                {
                const vector<ForceDataArraysGPU>& force1 = m_forces[cur_force+1]->acquireGPU();
                force_list.f1 = force1[0].d_data.force;
                force_list.v1 = force1[0].d_data.virial;
                }
            if (cur_force+2 < m_forces.size())
                {
                const vector<ForceDataArraysGPU>& force2 = m_forces[cur_force+2]->acquireGPU();
                force_list.f2 = force2[0].d_data.force;
                force_list.v2 = force2[0].d_data.virial;
                }
            if (cur_force+3 < m_forces.size())
                {
                const vector<ForceDataArraysGPU>& force3 = m_forces[cur_force+3]->acquireGPU();
                force_list.f3 = force3[0].d_data.force;
                force_list.v3 = force3[0].d_data.virial;
                }
            if (cur_force+4 < m_forces.size())
                {
                const vector<ForceDataArraysGPU>& force4 = m_forces[cur_force+4]->acquireGPU();
                force_list.f4 = force4[0].d_data.force;
                force_list.v4 = force4[0].d_data.virial;
                }
            if (cur_force+5 < m_forces.size())
                {
                const vector<ForceDataArraysGPU>& force5 = m_forces[cur_force+5]->acquireGPU();
                force_list.f5 = force5[0].d_data.force;
                force_list.v5 = force5[0].d_data.virial;
                }
            
            // clear on the first iteration only
            bool clear = (cur_force == 0);
            
            exec_conf.gpu[0]->call(bind(gpu_integrator_sum_net_force, 
                                             d_net_force.data,
                                             d_net_virial.data,
                                             force_list,
                                             nparticles,
                                             clear));
            }
        }
    
    if (m_prof)
        {
        m_prof->pop(exec_conf);
        m_prof->pop(exec_conf);
        }
    
    // return early if there are no constraint forces
    if (m_constraint_forces.size() == 0)
        return;
    
    // compute all the constraint forces next
    std::vector< boost::shared_ptr<ForceConstraint> >::iterator force_constraint;
    for (force_constraint = m_constraint_forces.begin(); force_constraint != m_constraint_forces.end(); ++force_constraint)
        (*force_constraint)->compute(timestep);
    
    if (m_prof)
        {
        m_prof->push(profile_name);
        m_prof->push(exec_conf, "Net force");
        }
    
        {
        // access the net force and virial arrays
        const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
        const GPUArray< Scalar >& net_virial = m_pdata->getNetVirial();
        ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_net_virial(net_virial, access_location::device, access_mode::overwrite);

        unsigned int nparticles = m_pdata->getN();
        assert(nparticles == net_force.getNumElements());
        assert(nparticles == net_virial.getNumElements());
        
        // now, add up the accelerations
        // sum all the forces into the net force
        // perform the sum in groups of 6 to avoid kernel launch and memory access overheads
        for (unsigned int cur_force = 0; cur_force < m_constraint_forces.size(); cur_force += 6)
            {
            // grab the device pointers for the current set
            gpu_force_list force_list;
            const vector<ForceDataArraysGPU>& force0 = m_constraint_forces[cur_force]->acquireGPU();
            force_list.f0 = force0[0].d_data.force;
            force_list.v0 = force0[0].d_data.virial;
            
            if (cur_force+1 < m_constraint_forces.size())
                {
                const vector<ForceDataArraysGPU>& force1 = m_constraint_forces[cur_force+1]->acquireGPU();
                force_list.f1 = force1[0].d_data.force;
                force_list.v1 = force1[0].d_data.virial;
                }
            if (cur_force+2 < m_constraint_forces.size())
                {
                const vector<ForceDataArraysGPU>& force2 = m_constraint_forces[cur_force+2]->acquireGPU();
                force_list.f2 = force2[0].d_data.force;
                force_list.v2 = force2[0].d_data.virial;
                }
            if (cur_force+3 < m_constraint_forces.size())
                {
                const vector<ForceDataArraysGPU>& force3 = m_constraint_forces[cur_force+3]->acquireGPU();
                force_list.f3 = force3[0].d_data.force;
                force_list.v3 = force3[0].d_data.virial;
                }
            if (cur_force+4 < m_constraint_forces.size())
                {
                const vector<ForceDataArraysGPU>& force4 = m_constraint_forces[cur_force+4]->acquireGPU();
                force_list.f4 = force4[0].d_data.force;
                force_list.v4 = force4[0].d_data.virial;
                }
            if (cur_force+5 < m_constraint_forces.size())
                {
                const vector<ForceDataArraysGPU>& force5 = m_constraint_forces[cur_force+5]->acquireGPU();
                force_list.f5 = force5[0].d_data.force;
                force_list.v5 = force5[0].d_data.virial;
                }
            
            // clear only on the first iteration AND if there are zero forces
            bool clear = (cur_force == 0) && (m_forces.size() == 0);
            
            exec_conf.gpu[0]->call(bind(gpu_integrator_sum_net_force, 
                                             d_net_force.data,
                                             d_net_virial.data,
                                             force_list,
                                             nparticles,
                                             clear));
            }
        }
    
    if (m_prof)
        {
        m_prof->pop(exec_conf);
        m_prof->pop(exec_conf);
        }
    }
#endif

/*! The base class integrator actually does nothing in update()
    \param timestep Current time step of the simulation
*/
void Integrator::update(unsigned int timestep)
    {
    }

void export_Integrator()
    {
    class_<Integrator, boost::shared_ptr<Integrator>, bases<Updater>, boost::noncopyable>
    ("Integrator", init< boost::shared_ptr<SystemDefinition>, Scalar >())
    .def("addForceCompute", &Integrator::addForceCompute)
    .def("addForceConstraint", &Integrator::addForceConstraint)
    .def("removeForceComputes", &Integrator::removeForceComputes)
    .def("setDeltaT", &Integrator::setDeltaT)
    .def("getNDOF", &Integrator::getNDOF)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

