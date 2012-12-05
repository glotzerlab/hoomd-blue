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
#include "HOOMDMPI.h"
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

    // this breaks memory scaling (calculate memory requirements from global group size)
    // unless we reallocate memory with every change of the maximum particle number
    m_num_blocks = m_group->getNumMembersGlobal() / m_reduction_block_size + 1;
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
    unsigned int group_size = m_group->getNumMembers();

    if (group_size == 0)
        return;

    // compute the current thermodynamic properties
    m_thermo_group->compute(timestep);

    // compute temperature for the next half time step
    m_curr_group_T = m_thermo_group->getTemperature();

    // compute pressure for the next half time step
    assert(m_mode == cubic || m_mode == orthorhombic || m_mode == tetragonal);

    PressureTensor P = m_thermo_group->getPressureTensor();

    if ( isnan(P.xx) || isnan(P.xy) || isnan(P.xz) || isnan(P.yy) || isnan(P.yz) || isnan(P.zz) )
        {
        Scalar extP = m_P->getValue(timestep);
        P.xx = P.yy = P.zz = extP;
        P.xy = P.xz = P.yz = Scalar(0.0);
        }

    // profile this step
    if (m_prof)
        m_prof->push("NPT MTK step 1");

    IntegratorVariables v = getIntegratorVariables();
    Scalar& eta = v.variable[0];  // Thermostat variable
    Scalar& xi = v.variable[1];   // Thermostat velocity
    Scalar& nuxx = v.variable[2];  // Barostat tensor, xx component
    Scalar& nuxy = v.variable[3];  // Barostat tensor, xy component
    Scalar& nuxz = v.variable[4];  // Barostat tensor, xz component
    Scalar& nuyy = v.variable[5];  // Barostat tensor, yy component
    Scalar& nuyz = v.variable[6];  // Barostat tensor, yz component
    Scalar& nuzz = v.variable[7];  // Barostat tensor, zz component

    // advance barostat (nux, nuy, nuz) half a time step
    Scalar W = m_thermo_group->getNDOF()*m_T->getValue(timestep)*m_tauP*m_tauP;
    Scalar mtk_term = Scalar(1.0/2.0)*m_deltaT*m_curr_group_T/W;
    if (m_mode == cubic)
        {
        Scalar P_iso = Scalar(1.0/3.0)*(P.xx + P.yy + P.zz);
        nuxx += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P_iso - m_P->getValue(timestep)) + mtk_term;
        nuyy = nuzz = nuxx;
        }
    else if (m_mode == tetragonal)
        {
        nuxx += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P.xx - m_P->getValue(timestep)) + mtk_term;
        nuyy += Scalar(1.0/2.0)*m_deltaT*m_V/W*((P.yy + P.zz)/Scalar(2.0) - m_P->getValue(timestep)) + mtk_term;
        nuzz = nuyy;
        }
    else if (m_mode == orthorhombic)
        {
        nuxx += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P.xx - m_P->getValue(timestep)) + mtk_term;
        nuyy += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P.yy - m_P->getValue(timestep)) + mtk_term;
        nuzz += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P.zz - m_P->getValue(timestep)) + mtk_term;
        }
    else if (m_mode == triclinic)
        {
        nuxx += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P.xx - m_P->getValue(timestep)) + mtk_term;
        nuxy += Scalar(1.0/2.0)*m_deltaT*m_V/W*P.xy;
        nuxz += Scalar(1.0/2.0)*m_deltaT*m_V/W*P.xz;
        nuyy += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P.yy - m_P->getValue(timestep)) + mtk_term;
        nuyz += Scalar(1.0/2.0)*m_deltaT*m_V/W*P.yz;
        nuzz += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P.zz - m_P->getValue(timestep)) + mtk_term;
        }

    // advance thermostat (xi, eta) half a time step
    Scalar xi_prime = xi + Scalar(1.0/4.0)*m_deltaT/m_tau/m_tau*(m_curr_group_T/m_T->getValue(timestep) - Scalar(1.0));
    xi = xi_prime+ Scalar(1.0/4.0)*m_deltaT/(m_tau*m_tau)*(m_curr_group_T/m_T->getValue(timestep)*
          exp(-xi_prime*m_deltaT) - Scalar(1.0));

    eta += Scalar(1.0/2.0)*xi_prime*m_deltaT;

    // store eigenvectors of barostat matrix in row major order
    Scalar *evec[3];
    evec[0] = &m_evec_arr[0];
    evec[1] = &m_evec_arr[3];
    evec[2] = &m_evec_arr[6];

    Scalar eval[3];
    if (m_mode == triclinic)
        {
        // find eigenvalues and -vectors of barostat matrix

        // store matrix in row-major order
        Scalar mat_array[9];
        Scalar *mat[3];
        mat[0] = &mat_array[0];
        mat[1] = &mat_array[3];
        mat[2] = &mat_array[6];

        mat[0][0] = nuxx; mat[0][1] = nuxy; mat[0][2] = nuxz;
        mat[1][0] = nuxy; mat[1][1] = nuyy; mat[1][2] = nuyz;
        mat[2][0] = nuxz; mat[2][1] = nuyz; mat[2][2] = nuzz;

        // the columns of evec are the normalized eigenvectors
        m_sysdef->getRigidData()->diagonalize(mat, eval, evec);
        }
    else
        {
        eval[0] = nuxx; eval[1] = nuyy; eval[2] = nuzz;
        evec[0][0] = Scalar(1.0); evec[0][1] = Scalar(0.0); evec[0][2] = Scalar(0.0);
        evec[1][0] = Scalar(0.0); evec[1][1] = Scalar(1.0); evec[1][2] = Scalar(0.0);
        evec[2][0] = Scalar(0.0); evec[2][1] = Scalar(0.0); evec[2][2] = Scalar(1.0);
        }

    Scalar mtk_term_2 = (nuxx+nuyy+nuzz)/m_thermo_group->getNDOF();
    Scalar3 v_fac = make_scalar3(Scalar(1.0/4.0)*(eval[0]+mtk_term_2),
                                 Scalar(1.0/4.0)*(eval[1]+mtk_term_2),
                                 Scalar(1.0/4.0)*(eval[2]+mtk_term_2));

    m_exp_v_fac = make_scalar3(exp(-v_fac.x*m_deltaT),
                               exp(-v_fac.y*m_deltaT),
                               exp(-v_fac.z*m_deltaT));
    Scalar3 exp_v_fac_2 = make_scalar3(exp(-(Scalar(2.0)*v_fac.x+Scalar(1.0/2.0)*xi_prime)*m_deltaT),
                               exp(-(Scalar(2.0)*v_fac.y+Scalar(1.0/2.0)*xi_prime)*m_deltaT),
                               exp(-(Scalar(2.0)*v_fac.z+Scalar(1.0/2.0)*xi_prime)*m_deltaT));

    Scalar3 r_fac = make_scalar3(Scalar(1.0/2.0)*eval[0],
                                 Scalar(1.0/2.0)*eval[1],
                                 Scalar(1.0/2.0)*eval[2]);
    Scalar3 exp_r_fac = make_scalar3(exp(r_fac.x*m_deltaT),
                                     exp(r_fac.y*m_deltaT),
                                     exp(r_fac.z*m_deltaT));

    // Coefficients of sinh(x)/x = a_0 + a_2 * x^2 + a_4 * x^4 + a_6 * x^6 + a_8 * x^8 + a_10 * x^10
    const Scalar coeff[] = {Scalar(1.0), Scalar(1.0/6.0), Scalar(1.0/120.0), Scalar(1.0/5040.0), Scalar(1.0/362880.0), Scalar(1.0/39916800.0)};

    Scalar3 arg_v = v_fac*m_deltaT;
    Scalar3 arg_r = r_fac*m_deltaT;

    m_sinhx_fac_v = make_scalar3(0.0,0.0,0.0);
    Scalar3 sinhx_fac_r = make_scalar3(0.0,0.0,0.0);
    Scalar3 term_v = make_scalar3(1.0,1.0,1.0);
    Scalar3 term_r = make_scalar3(1.0,1.0,1.0);

    for (unsigned int i = 0; i < 6; i++)
        {
        m_sinhx_fac_v += coeff[i] * term_v;
        sinhx_fac_r += coeff[i] * term_r;
        term_v = term_v * arg_v * arg_v;
        term_r = term_r * arg_r * arg_r;
        }

        {
        ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::readwrite);
        ArrayHandle<Scalar3> d_accel(m_pdata->getAccelerations(), access_location::device, access_mode::read);
        ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::readwrite);

        ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);


        // perform the particle update on the GPU
        gpu_npt_mtk_step_one(d_pos.data,
                             d_vel.data,
                             d_accel.data,
                             d_index_array.data,
                             group_size,
                             exp_r_fac,
                             m_exp_v_fac,
                             exp_v_fac_2,
                             sinhx_fac_r,
                             m_sinhx_fac_v,
                             m_evec_arr,
                             m_deltaT,
                             m_mode == triclinic);

        } // end of GPUArray scope

    // advance box lengths
    BoxDim global_box = m_pdata->getGlobalBox();
    Scalar3 a = global_box.getLatticeVector(0);
    Scalar3 b = global_box.getLatticeVector(1);
    Scalar3 c = global_box.getLatticeVector(2);

    // (a,b,c) are the columns of the cell parameter matrix
    Scalar3 scale = exp_r_fac*exp_r_fac;
    if (m_mode == triclinic)
        {
        // rotate cell parameter matrix
        Scalar3 a_rot, b_rot, c_rot;

        a_rot.x = evec[0][0]*a.x + evec[1][0]*a.y + evec[2][0]*a.z;
        a_rot.y = evec[0][1]*a.x + evec[1][1]*a.y + evec[2][1]*a.z;
        a_rot.z = evec[0][2]*a.x + evec[1][2]*a.y + evec[2][2]*a.z;

        b_rot.x = evec[0][0]*b.x + evec[1][0]*b.y + evec[2][0]*b.z;
        b_rot.y = evec[0][1]*b.x + evec[1][1]*b.y + evec[2][1]*b.z;
        b_rot.z = evec[0][2]*b.x + evec[1][2]*b.y + evec[2][2]*b.z;

        c_rot.x = evec[0][0]*c.x + evec[1][0]*c.y + evec[2][0]*c.z;
        c_rot.y = evec[0][1]*c.x + evec[1][1]*c.y + evec[2][1]*c.z;
        c_rot.z = evec[0][2]*c.x + evec[1][2]*c.y + evec[2][2]*c.z;

        a_rot *= scale;
        b_rot *= scale;
        c_rot *= scale;

        // rotate cell parameter matrix back
        a.x = evec[0][0]*a_rot.x + evec[0][1]*a_rot.y + evec[0][2]*a_rot.z;
        a.y = evec[1][0]*a_rot.x + evec[1][1]*a_rot.y + evec[1][2]*a_rot.z;
        a.z = evec[2][0]*a_rot.x + evec[2][1]*a_rot.y + evec[2][2]*a_rot.z;

        b.x = evec[0][0]*b_rot.x + evec[0][1]*b_rot.y + evec[0][2]*b_rot.z;
        b.y = evec[1][0]*b_rot.x + evec[1][1]*b_rot.y + evec[1][2]*b_rot.z;
        b.z = evec[2][0]*b_rot.x + evec[2][1]*b_rot.y + evec[2][2]*b_rot.z;

        c.x = evec[0][0]*c_rot.x + evec[0][1]*c_rot.y + evec[0][2]*c_rot.z;
        c.y = evec[1][0]*c_rot.x + evec[1][1]*c_rot.y + evec[1][2]*c_rot.z;
        c.z = evec[2][0]*c_rot.x + evec[2][1]*c_rot.y + evec[2][2]*c_rot.z;
        }
    else
        {
        a *= scale;
        b *= scale;
        c *= scale;
        }

#ifdef ENABLE_MPI
    if (m_comm)
        {
        // broadcast integrator variables from rank 0 to other processors
        MPI_Bcast(&eta, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&xi, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&nuxx, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&nuxy, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&nuyz, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&nuyy, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&nuyz, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&nuzz, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());

        MPI_Bcast(&a,sizeof(Scalar3), MPI_BYTE, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&b,sizeof(Scalar3), MPI_BYTE, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&c,sizeof(Scalar3), MPI_BYTE, 0, m_exec_conf->getMPICommunicator());
        }
#endif

    // update box dimensions
    global_box.setLatticeVectors(a,b,c);

    // set global box
    m_pdata->setGlobalBox(global_box);

    m_V = global_box.getVolume();  // volume

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
    unsigned int group_size = m_group->getNumMembers();

    if (group_size == 0)
        return;

    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();

    // profile this step
    if (m_prof)
        m_prof->push("NPT MTK step 2");

    IntegratorVariables v = getIntegratorVariables();
    Scalar& eta = v.variable[0];  // Thermostat variable
    Scalar& xi = v.variable[1];   // Thermostat velocity
    Scalar& nuxx = v.variable[2];  // Barostat tensor, xx component
    Scalar& nuxy = v.variable[3];  // Barostat tensor, xy component
    Scalar& nuxz = v.variable[4];  // Barostat tensor, xz component
    Scalar& nuyy = v.variable[5];  // Barostat tensor, yy component
    Scalar& nuyz = v.variable[6];  // Barostat tensor, yz component
    Scalar& nuzz = v.variable[7];  // Barostat tensor, zz component

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
                     d_net_force.data,
                     m_exp_v_fac,
                     m_sinhx_fac_v,
                     m_deltaT,
                     m_mode == triclinic);

        {
        // recalulate temperature
        ArrayHandle<Scalar> d_temperature(m_temperature, access_location::device, access_mode::overwrite);
        ArrayHandle<Scalar> d_scratch(m_scratch, access_location::device, access_mode::overwrite);
        
        // update number of blocks to current group size
        m_num_blocks = m_group->getNumMembers() / m_reduction_block_size + 1;

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
        MPI_Allreduce(MPI_IN_PLACE, &T_prime, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator() );
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
    PressureTensor P = m_thermo_group->getPressureTensor();

    if ( isnan(P.xx) || isnan(P.xy) || isnan(P.xz) || isnan(P.yy) || isnan(P.yz) || isnan(P.zz) )
        {
        Scalar extP = m_P->getValue(timestep);
        P.xx = P.yy = P.zz = extP;
        P.xy = P.xz = P.yz = Scalar(0.0);
        }

    // advance barostat (nux, nuy, nuz) half a time step
    Scalar W = m_thermo_group->getNDOF()*m_T->getValue(timestep)*m_tauP*m_tauP;
    Scalar mtk_term = Scalar(1.0/2.0)*m_deltaT*m_curr_group_T/W;
    if (m_mode == cubic)
        {
        Scalar P_iso = Scalar(1.0/3.0)*(P.xx + P.yy + P.zz);
        nuxx += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P_iso - m_P->getValue(timestep)) + mtk_term;
        nuyy = nuzz = nuxx;
        }
    else if (m_mode == tetragonal)
        {
        nuxx += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P.xx - m_P->getValue(timestep)) + mtk_term;
        nuyy += Scalar(1.0/2.0)*m_deltaT*m_V/W*((P.yy + P.zz)/Scalar(2.0) - m_P->getValue(timestep)) + mtk_term;
        nuzz = nuyy;
        }
    else if (m_mode == orthorhombic)
        {
        nuxx += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P.xx - m_P->getValue(timestep)) + mtk_term;
        nuyy += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P.yy - m_P->getValue(timestep)) + mtk_term;
        nuzz += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P.zz - m_P->getValue(timestep)) + mtk_term;
        }
    else if (m_mode == triclinic)
        {
        nuxx += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P.xx - m_P->getValue(timestep)) + mtk_term;
        nuxy += Scalar(1.0/2.0)*m_deltaT*m_V/W*P.xy;
        nuxz += Scalar(1.0/2.0)*m_deltaT*m_V/W*P.xz;
        nuyy += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P.yy - m_P->getValue(timestep)) + mtk_term;
        nuyz += Scalar(1.0/2.0)*m_deltaT*m_V/W*P.yz;
        nuzz += Scalar(1.0/2.0)*m_deltaT*m_V/W*(P.zz - m_P->getValue(timestep)) + mtk_term;
        }

#ifdef ENABLE_MPI
    if (m_comm)
        {
        // broadcast integrator variables from rank 0 to other processors
        MPI_Bcast(&eta, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&xi, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&nuxx, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&nuxy, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&nuyz, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&nuyy, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&nuyz, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
        MPI_Bcast(&nuzz, 1, MPI_HOOMD_SCALAR, 0, m_exec_conf->getMPICommunicator());
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
