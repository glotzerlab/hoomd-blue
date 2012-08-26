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

// Maintainer: jglaser

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "TwoStepNPTMTKGPU.h"
#include "TwoStepNPTMTKGPU.cuh"

#ifdef ENABLE_MPI
#include "Communicator.h"
#endif

/*! \file TwoStepNPTMTKGPU.h
    \brief Contains code for the TwoStepNPTMTKGPU class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param thermo_group ComputeThermo to compute thermo properties of the integrated \a group
    \param tau NPT temperature period
    \param tauP NPT pressure period
    \param T Temperature set point
    \param P Pressure set point
    \param mode Mode of integration
*/
TwoStepNPTMTKGPU::TwoStepNPTMTKGPU(boost::shared_ptr<SystemDefinition> sysdef,
                       boost::shared_ptr<ParticleGroup> group,
                       boost::shared_ptr<ComputeThermo> thermo_group,
                       Scalar tau,
                       Scalar tauP,
                       boost::shared_ptr<Variant> T,
                       boost::shared_ptr<Variant> P,
                       integrationMode mode)
    : TwoStepNPTMTK(sysdef, group, thermo_group, tau, tauP, T, P, mode)
    {
    if (!exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a TwoStepNPTMTKGPU with CUDA disabled" << endl;
        throw std::runtime_error("Error initializing TwoStepNPTMTKGPU");
        }

    m_exec_conf->msg->notice(5) << "Constructing TwoStepNPTMTKGPU" << endl;

    m_reduction_block_size = 512;

    // this breaks memory scaling (calculate memory requirements from global group size), but shouldn't be a big problem
    m_num_blocks = m_group->getNumMembers() / m_reduction_block_size + 1;
    GPUArray< Scalar > scratch(m_num_blocks, exec_conf);
    m_scratch.swap(scratch);

    GPUArray< Scalar> temperature(1, exec_conf);
    m_temperature.swap(temperature);
    }

TwoStepNPTMTKGPU::~TwoStepNPTMTKGPU()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepNPTMTKGPU" << endl;
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the Nose-Hoover
     thermostat and Anderson barostat
*/
void TwoStepNPTMTKGPU::integrateStepOne(unsigned int timestep)
    {
#ifdef ENABLE_MPI
    unsigned int group_size = m_group->getNumLocalMembers();
#else
    unsigned int group_size = m_group->getNumMembers();
#endif
    if (group_size == 0)
        return;

    // compute the current thermodynamic properties
    m_thermo_group->compute(timestep);

    // compute temperature for the next half time step
    m_curr_group_T = m_thermo_group->getTemperature();

    // compute pressure for the next half time step
    assert(m_mode == cubic || m_mode == orthorhombic || m_mode == tetragonal);
    if (m_mode == cubic)
        {
        m_curr_P = m_thermo_group->getPressure();

        // if it is not valid, assume that the current pressure is the set pressure (this should only happen in very
        // rare circumstances, usually at the start of the simulation before things are initialize)
        if (isnan(m_curr_P))
            m_curr_P = m_P->getValue(timestep);
        }
    else if (m_mode == orthorhombic || m_mode == tetragonal)
        {
        PressureTensor P;
        P = m_thermo_group->getPressureTensor();

        if ( isnan(P.xx) || isnan(P.xy) || isnan(P.xz) || isnan(P.yy) || isnan(P.yz) || isnan(P.zz) )
            {
            Scalar extP = m_P->getValue(timestep);
            m_curr_P_diag = make_scalar3(extP,extP,extP);
            }
        else
            {
            // store diagonal elements of pressure tensor
            m_curr_P_diag = make_scalar3(P.xx,P.yy,P.zz);
            }
        }

    // profile this step
    if (m_prof)
        m_prof->push("NPT MTK step 1");

    IntegratorVariables v = getIntegratorVariables();
    Scalar& eta = v.variable[0];  // Thermostat variable
    Scalar& xi = v.variable[1];   // Thermostat velocity
    Scalar& nux = v.variable[2];  // Barostat variable for x-direction
    Scalar& nuy = v.variable[3];  // Barostat variable for y-direction
    Scalar& nuz = v.variable[4];  // Barostat variable for z-direction

    {
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);

    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

    // advance barostat (nux, nuy, nuz) half a time step
    Scalar W = m_thermo_group->getNDOF()*m_T->getValue(timestep)*m_tauP*m_tauP;
    Scalar mtk_term = Scalar(1.0/2.0)*m_deltaT*m_curr_group_T/W;
    if (m_mode == cubic)
        {
        nux += Scalar(1.0/2.0)*m_deltaT*m_V/W*(m_curr_P - m_P->getValue(timestep)) + mtk_term;
        nuy = nuz = nux;
        }
    else if (m_mode == tetragonal)
        {
        nux += Scalar(1.0/2.0)*m_deltaT*m_V/W*(m_curr_P_diag.x - m_P->getValue(timestep)) + mtk_term;
        nuy += Scalar(1.0/2.0)*m_deltaT*m_V/W*((m_curr_P_diag.y+m_curr_P_diag.z)/Scalar(2.0) - m_P->getValue(timestep)) + mtk_term;
        nuz = nuy;
        }
    else if (m_mode == orthorhombic)
        {
        nux += Scalar(1.0/2.0)*m_deltaT*m_V/W*(m_curr_P_diag.x - m_P->getValue(timestep)) + mtk_term;
        nuy += Scalar(1.0/2.0)*m_deltaT*m_V/W*(m_curr_P_diag.y - m_P->getValue(timestep)) + mtk_term;
        nuz += Scalar(1.0/2.0)*m_deltaT*m_V/W*(m_curr_P_diag.z - m_P->getValue(timestep)) + mtk_term;
        }

    // advance thermostat (xi, eta) half a time step
    Scalar xi_prime = xi + Scalar(1.0/4.0)*m_deltaT/m_tau/m_tau*(m_curr_group_T/m_T->getValue(timestep) - Scalar(1.0));
    xi = xi_prime+ Scalar(1.0/4.0)*m_deltaT/(m_tau*m_tau)*(m_curr_group_T/m_T->getValue(timestep)*
          exp(-xi_prime*m_deltaT) - Scalar(1.0));

    eta += Scalar(1.0/2.0)*xi_prime*m_deltaT;

    // perform the particle update on the GPU
    gpu_npt_mtk_step_one(d_pos.data,
                         d_vel.data,
                         d_accel.data,
                         d_index_array.data,
                         group_size,
                         m_thermo_group->getNDOF(),
                         xi_prime,
                         make_scalar3(nux, nuy, nuz),
                         m_deltaT);
    } // end of GPUArray scope

    // advance box lengths
    Scalar3 box_len_scale = make_scalar3(exp(nux*m_deltaT), exp(nuy*m_deltaT), exp(nuz*m_deltaT));
    m_L = m_L*box_len_scale;

#ifdef ENABLE_MPI
    if (m_comm)
        {
        // broadcast integrator variables from rank 0 to other processors
        MPI_Bcast(&eta, 1, MPI_FLOAT, 0, *m_exec_conf->getMPICommunicator());
        MPI_Bcast(&xi, 1, MPI_FLOAT, 0, *m_exec_conf->getMPICommunicator());
        MPI_Bcast(&nux, 1, MPI_FLOAT, 0, *m_exec_conf->getMPICommunicator());
        MPI_Bcast(&nuy, 1, MPI_FLOAT, 0, *m_exec_conf->getMPICommunicator());
        MPI_Bcast(&nuz, 1, MPI_FLOAT, 0, *m_exec_conf->getMPICommunicator());

        // broadcast box dimensions
        MPI_Bcast(&m_L,sizeof(Scalar3), MPI_BYTE, 0, *m_exec_conf->getMPICommunicator());
        }
#endif

    // calculate volume
    m_V = m_L.x*m_L.y*m_L.z;

    m_pdata->setGlobalBoxL(m_L);

    // Get new (local) box lengths
    BoxDim box = m_pdata->getBox();

        {
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
        ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::readwrite);

        // Wrap particles
        gpu_npt_mtk_wrap(m_pdata->getN(),
                         d_pos.data,
                         d_image.data,
                         box);
        }

    setIntegratorVariables(v);

    // done profiling
    if (m_prof)
        m_prof->pop();

    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1
*/
void TwoStepNPTMTKGPU::integrateStepTwo(unsigned int timestep)
    {
#ifdef ENABLE_MPI
    unsigned int group_size = m_group->getNumLocalMembers();
#else
    unsigned int group_size = m_group->getNumMembers();
#endif
    if (group_size == 0)
        return;

    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();

    // profile this step
    if (m_prof)
        m_prof->push("NPT MTK step 2");

    IntegratorVariables v = getIntegratorVariables();
    Scalar& eta = v.variable[0];  // Thermostat variable
    Scalar& xi = v.variable[1];   // Thermostat velocity
    Scalar& nux = v.variable[2];  // Barostat variable for x-direction
    Scalar& nuy = v.variable[3];  // Barostat variable for y-direction
    Scalar& nuz = v.variable[4];  // Barostat variable for z-direction

    {
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::overwrite);

    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);


    // perform second half step of NPT integration (update velocities and accelerations)
    gpu_npt_mtk_step_two(d_vel.data,
                     d_accel.data,
                     d_index_array.data,
                     group_size,
                     m_thermo_group->getNDOF(),
                     d_net_force.data,
                     make_scalar3(nux, nuy, nuz),
                     m_deltaT);

    // recalulate temperature
        {
        ArrayHandle<Scalar> d_temperature(m_temperature, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_scratch(m_scratch, access_location::device, access_mode::overwrite);

#ifdef ENABLE_MPI
        m_num_blocks = m_group->getNumLocalMembers() / m_reduction_block_size + 1;
#endif
        gpu_npt_mtk_temperature(d_temperature.data,
                                d_vel.data,
                                d_scratch.data,
                                m_num_blocks,
                                m_reduction_block_size,
                                d_index_array.data,
                                group_size,
                                m_thermo_group->getNDOF());
        }

    // read back intermediate temperature from GPU
    Scalar T_prime(0.0);
        {
        ArrayHandle<Scalar> h_temperature(m_temperature, access_location::host, access_mode::read);
        T_prime = *h_temperature.data;
        }

#ifdef ENABLE_MPI
    if (m_comm)
        MPI_Allreduce(MPI_IN_PLACE, &T_prime, 1, MPI_FLOAT, MPI_SUM, *m_exec_conf->getMPICommunicator() );
#endif

    // Advance thermostat half a time step
    Scalar xi_prime = xi + Scalar(1.0/4.0)*m_deltaT/m_tau/m_tau*(T_prime/m_T->getValue(timestep) - Scalar(1.0));
    xi = xi_prime+ Scalar(1.0/4.0)*m_deltaT/(m_tau*m_tau)*(T_prime/m_T->getValue(timestep) *
          exp(-xi_prime*m_deltaT) - Scalar(1.0));

    eta += Scalar(1.0/2.0)*xi_prime*m_deltaT;


    // rescale velocities
    gpu_npt_mtk_thermostat(d_vel.data,
                           d_index_array.data,
                           group_size,
                           xi_prime,
                           m_deltaT);

    } // end GPUArray scope

    if (m_prof)
        m_prof->pop();

    // compute the current thermodynamic properties
    m_thermo_group->compute(timestep+1);

    if (m_prof)
        m_prof->push("NPT MTK step 2");

    // compute temperature for the next half time step
    m_curr_group_T = m_thermo_group->getTemperature();

    // compute pressure for the next half time step
    assert(m_mode == cubic || m_mode == orthorhombic || m_mode == tetragonal);
    if (m_mode == cubic)
        {
        m_curr_P = m_thermo_group->getPressure();

        // if it is not valid, assume that the current pressure is the set pressure (this should only happen in very
        // rare circumstances, usually at the start of the simulation before things are initialize)
        if (isnan(m_curr_P))
            m_curr_P = m_P->getValue(timestep);
        }
    else if (m_mode == orthorhombic || m_mode == tetragonal)
        {
        PressureTensor P;
        P = m_thermo_group->getPressureTensor();

        if ( isnan(P.xx) || isnan(P.xy) || isnan(P.xz) || isnan(P.yy) || isnan(P.yz) || isnan(P.zz) )
            {
            Scalar extP = m_P->getValue(timestep);
            m_curr_P_diag = make_scalar3(extP,extP,extP);
            }
        else
            {
            // store diagonal elements of pressure tensor
            m_curr_P_diag = make_scalar3(P.xx,P.yy,P.zz);
            }
        }

    // advance barostat (nux, nuy, nuz) half a time step
    Scalar W = m_thermo_group->getNDOF()*m_T->getValue(timestep)*m_tauP*m_tauP;
    Scalar mtk_term = Scalar(1.0/2.0)*m_deltaT*m_curr_group_T/W;
    if (m_mode == cubic)
        {
        nux += Scalar(1.0/2.0)*m_deltaT*m_V/W*(m_curr_P - m_P->getValue(timestep)) + mtk_term;
        nuy = nuz = nux;
        }
    else if (m_mode == tetragonal)
        {
        nux += Scalar(1.0/2.0)*m_deltaT*m_V/W*(m_curr_P_diag.x - m_P->getValue(timestep)) + mtk_term;
        nuy += Scalar(1.0/2.0)*m_deltaT*m_V/W*((m_curr_P_diag.y+m_curr_P_diag.z)/Scalar(2.0) - m_P->getValue(timestep)) + mtk_term;
        nuz = nuy;
        }
    else if (m_mode == orthorhombic)
        {
        nux += Scalar(1.0/2.0)*m_deltaT*m_V/W*(m_curr_P_diag.x - m_P->getValue(timestep)) + mtk_term;
        nuy += Scalar(1.0/2.0)*m_deltaT*m_V/W*(m_curr_P_diag.y - m_P->getValue(timestep)) + mtk_term;
        nuz += Scalar(1.0/2.0)*m_deltaT*m_V/W*(m_curr_P_diag.z - m_P->getValue(timestep)) + mtk_term;
        }

#ifdef ENABLE_MPI
    if (m_comm)
        {
        // broadcast integrator variables from rank 0 to other processors
        MPI_Bcast(&eta, 1, MPI_FLOAT, 0, *m_exec_conf->getMPICommunicator());
        MPI_Bcast(&xi, 1, MPI_FLOAT, 0, *m_exec_conf->getMPICommunicator());
        MPI_Bcast(&nux, 1, MPI_FLOAT, 0, *m_exec_conf->getMPICommunicator());
        MPI_Bcast(&nuy, 1, MPI_FLOAT, 0, *m_exec_conf->getMPICommunicator());
        MPI_Bcast(&nuz, 1, MPI_FLOAT, 0, *m_exec_conf->getMPICommunicator());
        }
#endif

    setIntegratorVariables(v);

    // done profiling
    if (m_prof)
        m_prof->pop();
    }


void export_TwoStepNPTMTKGPU()
    {
    class_<TwoStepNPTMTKGPU, boost::shared_ptr<TwoStepNPTMTKGPU>, bases<TwoStepNPTMTK>, boost::noncopyable>
        ("TwoStepNPTMTKGPU", init< boost::shared_ptr<SystemDefinition>,
                       boost::shared_ptr<ParticleGroup>,
                       boost::shared_ptr<ComputeThermo>,
                       Scalar,
                       Scalar,
                       boost::shared_ptr<Variant>,
                       boost::shared_ptr<Variant>,
                       TwoStepNPTMTKGPU::integrationMode>())
        ;

    }

#ifdef WIN32
#pragma warning( pop )
#endif
