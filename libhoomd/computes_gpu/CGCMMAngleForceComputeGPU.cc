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
// Maintainer: akohlmey

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include "CGCMMAngleForceComputeGPU.h"

#include <boost/python.hpp>
using namespace boost::python;

#include <boost/bind.hpp>
using namespace boost;

using namespace std;

/*! \param sysdef System to compute angle forces on
*/
CGCMMAngleForceComputeGPU::CGCMMAngleForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef)
        : CGCMMAngleForceCompute(sysdef), m_block_size(64)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!exec_conf->isCUDAEnabled())
        {
        cerr << endl << "***Error! Creating a CGCMMAngleForceComputeGPU with no GPU in the execution configuration" << endl << endl;
        throw std::runtime_error("Error initializing CGCMMAngleForceComputeGPU");
        }
        
    prefact[0] = Scalar(0.0);
    prefact[1] = Scalar(6.75);
    prefact[2] = Scalar(2.59807621135332);
    prefact[3] = Scalar(4.0);
    
    cgPow1[0]  = Scalar(0.0);
    cgPow1[1]  = Scalar(9.0);
    cgPow1[2]  = Scalar(12.0);
    cgPow1[3]  = Scalar(12.0);
    
    cgPow2[0]  = Scalar(0.0);
    cgPow2[1]  = Scalar(6.0);
    cgPow2[2]  = Scalar(4.0);
    cgPow2[3]  = Scalar(6.0);
    
    // allocate and zero device memory
    GPUArray<float2> params (m_CGCMMAngle_data->getNAngleTypes(),exec_conf);
    m_params.swap(params);
    GPUArray<float2> CGCMMsr(m_CGCMMAngle_data->getNAngleTypes(),exec_conf);
    m_CGCMMsr.swap(CGCMMsr);
    GPUArray<float4> CGCMMepow(m_CGCMMAngle_data->getNAngleTypes(),exec_conf);
    m_CGCMMepow.swap(CGCMMepow);
    }

CGCMMAngleForceComputeGPU::~CGCMMAngleForceComputeGPU()
    {
    }

/*! \param type Type of the angle to set parameters for
    \param K Stiffness parameter for the force computation
    \param t_0 Equilibrium angle (in radians) for the force computation
    \param cg_type the type of course grained angle we are using
    \param eps the well depth
    \param sigma the particle radius

    Sets parameters for the potential of a particular angle type and updates the
    parameters on the GPU.
*/
void CGCMMAngleForceComputeGPU::setParams(unsigned int type, Scalar K, Scalar t_0, unsigned int cg_type, Scalar eps, Scalar sigma)
    {
    CGCMMAngleForceCompute::setParams(type, K, t_0, cg_type, eps, sigma);
    
    const float myPow1 = cgPow1[cg_type];
    const float myPow2 = cgPow2[cg_type];
    const float myPref = prefact[cg_type];
    
    Scalar my_rcut = sigma*exp(1.0f/(myPow1-myPow2)*log(myPow1/myPow2));
   
    ArrayHandle<float2> h_params(m_params, access_location::host, access_mode::readwrite);
    ArrayHandle<float2> h_CGCMMsr(m_CGCMMsr, access_location::host, access_mode::readwrite);
    ArrayHandle<float4> h_CGCMMepow(m_CGCMMepow, access_location::host, access_mode::readwrite);
    // update the local copy of the memory
    h_params.data[type] = make_float2(K, t_0);
    h_CGCMMsr.data[type] = make_float2(sigma, my_rcut);
    h_CGCMMepow.data[type] = make_float4(eps, myPow1, myPow2, myPref);
    }

/*! Internal method for computing the forces on the GPU.
    \post The force data on the GPU is written with the calculated forces

    \param timestep Current time step of the simulation

    Calls gpu_compute_CGCMM_angle_forces to do the dirty work.
*/
void CGCMMAngleForceComputeGPU::computeForces(unsigned int timestep)
    {
    // start the profile
    if (m_prof) m_prof->push(exec_conf, "CGCMM Angle");
    
    gpu_angletable_array& gpu_angletable = m_CGCMMAngle_data->acquireGPU();
    
    // the angle table is up to date: we are good to go. Call the kernel
    gpu_pdata_arrays& pdata = m_pdata->acquireReadOnlyGPU();
    gpu_boxsize box = m_pdata->getBoxGPU();
   
    //Not necessary - force and virial are zeroed in the kernel
    //m_force.memclear();
    //m_virial.memclear();
    ArrayHandle<Scalar4> d_force(m_force,access_location::device,access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial,access_location::device,access_mode::overwrite);
    ArrayHandle<float2> d_params(m_params, access_location::device, access_mode::read);
    ArrayHandle<float2> d_CGCMMsr(m_CGCMMsr, access_location::device, access_mode::read);
    ArrayHandle<float4> d_CGCMMepow(m_CGCMMepow, access_location::device, access_mode::read);
    

    // run the kernel
    gpu_compute_CGCMM_angle_forces(d_force.data,
                                   d_virial.data,
                                   pdata,
                                   box,
                                   gpu_angletable,
                                   d_params.data,
                                   d_CGCMMsr.data,
                                   d_CGCMMepow.data,
                                   m_CGCMMAngle_data->getNAngleTypes(),
                                   m_block_size);
    
    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
   
    m_pdata->release();
    
    if (m_prof) m_prof->pop(exec_conf);
    }

void export_CGCMMAngleForceComputeGPU()
    {
    class_<CGCMMAngleForceComputeGPU, boost::shared_ptr<CGCMMAngleForceComputeGPU>, bases<CGCMMAngleForceCompute>, boost::noncopyable >
    ("CGCMMAngleForceComputeGPU", init< boost::shared_ptr<SystemDefinition> >())
    .def("setBlockSize", &CGCMMAngleForceComputeGPU::setBlockSize)
    ;
    }

