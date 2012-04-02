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
#include <boost/bind.hpp>
using namespace boost;

#include "TwoStepNPHGPU.h"
#include "TwoStepNPHGPU.cuh"
#include "TwoStepNVTGPU.cuh"

/*! \file TwoStepNPHGPU.h
    \brief Contains code for the TwoStepNPHGPU class
*/

/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param thermo ComputeThermo to use
    \param W piston mass
    \param P pressure set point
    \param mode integration mode (cubic, orthorhombic or tetragonal)
    \param suffix suffix for log quantity
*/
TwoStepNPHGPU::TwoStepNPHGPU(boost::shared_ptr<SystemDefinition> sysdef,
                       boost::shared_ptr<ParticleGroup> group,
                       boost::shared_ptr<ComputeThermo> thermo,
                       Scalar W,
                       boost::shared_ptr<Variant> P,
                       integrationMode mode,
                       const std::string& suffix)
    : TwoStepNPH(sysdef, group, thermo, W, P, mode, suffix)
    {
    // only one GPU is supported
    if (!exec_conf->isCUDAEnabled())
        {
        cerr << endl << "***Error! Creating a TwoStepNPHGPU with CUDA disabled" << endl << endl;
        throw std::runtime_error("Error initializing TwoStepNVEGPU");
        }
    }

/*! \param timestep Current time step
    \post Particle positions are moved forward to timestep+1 and velocities to timestep+1/2 per the NPH method
*/
void TwoStepNPHGPU::integrateStepOne(unsigned int timestep)
    {

    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;

    // profile this step
    if (m_prof)
        m_prof->push("NPH step 1");

    if (!m_state_initialized)
        {
        //! compute the current pressure tensor
        m_thermo->compute(timestep);

        // compute pressure tensor for next half time step
        PressureTensor P;
        P = m_thermo->getPressureTensor();

        // If for some reason the pressure is not valid, assume internal pressure = external pressure
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

        m_state_initialized = true;
        }

    // access all the needed data
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::read);
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

    // get integrator variables
    IntegratorVariables v = getIntegratorVariables();
    Scalar &etax = v.variable[0];
    Scalar &etay = v.variable[1];
    Scalar &etaz = v.variable[2];

    // obtain box lengths
    Scalar volume = Scalar(0.0);

    BoxDim box = m_pdata->getBox();
    Scalar3 L = box.getL();
    volume = L.x*L.y*L.z;

    Scalar extP = m_P->getValue(timestep);

    // advance eta(t)->eta(t+deltaT/2)
    if (m_mode == orthorhombic)
        {
        Scalar VdeltaThalf = Scalar(1./2.)*volume*m_deltaT;
        etax += VdeltaThalf/L.x * (m_curr_P_diag.x - extP);
        etay += VdeltaThalf/L.y * (m_curr_P_diag.y - extP);
        etaz += VdeltaThalf/L.z * (m_curr_P_diag.z - extP);
        }
    else if (m_mode == tetragonal)
       {
       Scalar VdeltaThalf = Scalar(1./2.)*volume*m_deltaT;
       etax += VdeltaThalf/L.x * (m_curr_P_diag.x - extP);
       etay += VdeltaThalf/L.y * (m_curr_P_diag.y + m_curr_P_diag.z - Scalar(2.0)*extP);
       }
    else if (m_mode == cubic)
        {
        etax += Scalar(1./2.)*m_deltaT*(Scalar(1./3.)*(m_curr_P_diag.x + m_curr_P_diag.y + m_curr_P_diag.z) - extP);
        }

    // update the box length L(t) -> L(t+deltaT/2)
    // also pre-calculate L(t+deltaT)
    Scalar Lx_old = L.x;
    Scalar Ly_old = L.y;
    Scalar Lz_old = L.z;

    Scalar Lx_final = Scalar(0.0);
    Scalar Ly_final = Scalar(0.0);
    Scalar Lz_final = Scalar(0.0);

    Scalar deltaThalfoverW = Scalar(1./2.)*m_deltaT/m_W;

    if (m_mode == orthorhombic)
        {
        L.x += deltaThalfoverW*etax;
        L.y += deltaThalfoverW*etay;
        L.z += deltaThalfoverW*etaz;
        Lx_final = L.x + deltaThalfoverW*etax;
        Ly_final = L.y + deltaThalfoverW*etay;
        Lz_final = L.z + deltaThalfoverW*etaz;
        }
    else if (m_mode == tetragonal)
        {
        L.x += deltaThalfoverW*etax;
        L.y += Scalar(1./2.)*deltaThalfoverW*etay;
        L.z = L.y;
        Lx_final = L.x + deltaThalfoverW*etax;
        Ly_final = L.y + Scalar(1./2.)*deltaThalfoverW*etay;
        Lz_final = Ly_final;
        }
    else if (m_mode == cubic)
        {
        volume += deltaThalfoverW*etax;
        L.x = pow(volume,Scalar(1./3.)); // Lx = Ly = Lz = V^(1/3)
        L.y = L.x;
        L.z = L.x;
        Scalar volume_final = volume + deltaThalfoverW*etax;
        Lx_final = pow(volume_final,Scalar(1./3.)); // Lx = Ly = Lz = V^(1/3)
        Ly_final = Lx_final;
        Lz_final = Lx_final;
        }

    {
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);

    // perform the particle update on the GPU
    gpu_nph_step_one(d_pos.data,
                     d_vel.data,
                     d_accel.data,
                     d_index_array.data,
                     group_size,
                     make_scalar3(Lx_old,Ly_old,Lz_old),
                     make_scalar3(L.x,L.y,L.z),
                     make_scalar3(Lx_final,Ly_final,Lz_final),
                     m_deltaT);

    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    }

    BoxDim box_final(box);
    box_final.setL(make_scalar3(Lx_final, Ly_final, Lz_final));

    {
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);
    ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::readwrite);

    // wrap particles around new boundaries
    gpu_nph_wrap_particles(m_pdata->getN(), d_pos.data, d_image.data, box_final);

    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    }

    // update simulation box
    m_pdata->setGlobalBoxL(box_final.getL());


    setIntegratorVariables(v);

    // done profiling
    if (m_prof)
        m_prof->pop(exec_conf);
    }

/*! \param timestep Current time step
    \post particle velocities are moved forward to timestep+1 on the GPU
*/
void TwoStepNPHGPU::integrateStepTwo(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;

    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();

    // profile this step
    if (m_prof)
        m_prof->push(exec_conf, "NPH step 2");

    {
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

    // perform the update on the GPU
    gpu_nph_step_two(d_vel.data,
                     d_accel.data,
                     d_index_array.data,
                     group_size,
                     d_net_force.data,
                     m_deltaT);

    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    }

    // compute pressure tensor with updated virial and velocities
    m_thermo->compute(timestep+1);

    PressureTensor P;
    P = m_thermo->getPressureTensor();

    // Update state variables
    IntegratorVariables v = getIntegratorVariables();
    Scalar &etax = v.variable[0];
    Scalar &etay = v.variable[1];
    Scalar &etaz = v.variable[2];

    // If for some reason the pressure is not valid, assume internal pressure = external pressure
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

    // advance eta(t+deltaT/2) -> eta(t+deltaT)
    Scalar extP = m_P->getValue(timestep);

    if (m_mode == orthorhombic || m_mode == tetragonal)
        {
        const BoxDim &box = m_pdata->getBox();
        Scalar3 L = box.getL();
        Scalar volume = L.x*L.y*L.z;
        Scalar VdeltaThalf = Scalar(1./2.)*volume*m_deltaT;

        // eta_alpha(t+deltaT) = eta_alpha(t+deltaT/2) + deltaT/2 * V/L_alpha * ( P_{alpha,alpha}(t) - P )
        etax += VdeltaThalf/L.x * (m_curr_P_diag.x - extP);

        if (m_mode == orthorhombic)
            {
            etay += VdeltaThalf/L.y * (m_curr_P_diag.y - extP);
            etaz += VdeltaThalf/L.z * (m_curr_P_diag.z - extP);
            }
        else
            {
            //tetragonal
            etay += VdeltaThalf/L.y * (m_curr_P_diag.y + m_curr_P_diag.z - Scalar(2.0)*extP);
            }
        }
    else if (m_mode == cubic)
        {
        etax += Scalar(1./2.)*m_deltaT * (Scalar(1./3.)*(m_curr_P_diag.x + m_curr_P_diag.y + m_curr_P_diag.z) - extP);
        }

    setIntegratorVariables(v);

    // done profiling
    if (m_prof)
        m_prof->pop(exec_conf);
    }

void export_TwoStepNPHGPU()
    {
    class_<TwoStepNPHGPU, boost::shared_ptr<TwoStepNPHGPU>, bases<TwoStepNPH>, boost::noncopyable>
        ("TwoStepNPHGPU", init< boost::shared_ptr<SystemDefinition>,
                          boost::shared_ptr<ParticleGroup>,
                          boost::shared_ptr<ComputeThermo>,
                          Scalar,
                          boost::shared_ptr<Variant>,
                          TwoStepNPHGPU::integrationMode,
                          const std::string& >())
        ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

