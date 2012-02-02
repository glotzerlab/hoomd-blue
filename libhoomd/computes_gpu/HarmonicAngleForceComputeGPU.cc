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

// Maintainer: dnlebard

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include "HarmonicAngleForceComputeGPU.h"

#include <boost/python.hpp>
using namespace boost::python;

#include <boost/bind.hpp>
using namespace boost;

using namespace std;

/*! \param sysdef System to compute angle forces on
*/
HarmonicAngleForceComputeGPU::HarmonicAngleForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef)
        : HarmonicAngleForceCompute(sysdef), m_block_size(64)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!exec_conf->isCUDAEnabled())
        {
        cerr << endl 
             << "***Error! Creating a AngleForceComputeGPU with no GPU in the execution configuration" 
             << endl << endl;
        throw std::runtime_error("Error initializing AngleForceComputeGPU");
        }
        
    // allocate and zero device memory
    GPUArray<float2> params(m_angle_data->getNAngleTypes(), exec_conf);
    m_params.swap(params);
    }

HarmonicAngleForceComputeGPU::~HarmonicAngleForceComputeGPU()
    {
    }

/*! \param type Type of the angle to set parameters for
    \param K Stiffness parameter for the force computation
    \param t_0 Equilibrium angle (in radians) for the force computation

    Sets parameters for the potential of a particular angle type and updates the
    parameters on the GPU.
*/
void HarmonicAngleForceComputeGPU::setParams(unsigned int type, Scalar K, Scalar t_0)
    {
    HarmonicAngleForceCompute::setParams(type, K, t_0);
    
    ArrayHandle<float2> h_params(m_params, access_location::host, access_mode::readwrite);
    // update the local copy of the memory
    h_params.data[type] = make_float2(K, t_0);
    }

/*! Internal method for computing the forces on the GPU.
    \post The force data on the GPU is written with the calculated forces

    \param timestep Current time step of the simulation

    Calls gpu_compute_harmonic_angle_forces to do the dirty work.
*/
void HarmonicAngleForceComputeGPU::computeForces(unsigned int timestep)
    {
    // start the profile
    if (m_prof) m_prof->push(exec_conf, "Harmonic Angle");
    
    gpu_angletable_array& gpu_angletable = m_angle_data->acquireGPU();
    
    // the angle table is up to date: we are good to go. Call the kernel
    gpu_pdata_arrays& pdata = m_pdata->acquireReadOnlyGPU();
    gpu_boxsize box = m_pdata->getBoxGPU();
      
    ArrayHandle<Scalar4> d_force(m_force,access_location::device,access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial,access_location::device,access_mode::overwrite);
    ArrayHandle<float2> d_params(m_params, access_location::device, access_mode::read);

    // run the kernel on the GPU
    gpu_compute_harmonic_angle_forces(d_force.data,
                                      d_virial.data,
                                      m_virial.getPitch(),
                                      pdata,
                                      box,
                                      gpu_angletable,
                                      d_params.data,
                                      m_angle_data->getNAngleTypes(),
                                      m_block_size);

    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    
    m_pdata->release();
    
    if (m_prof) m_prof->pop(exec_conf);
    }

void export_HarmonicAngleForceComputeGPU()
    {
    class_<HarmonicAngleForceComputeGPU, boost::shared_ptr<HarmonicAngleForceComputeGPU>, bases<HarmonicAngleForceCompute>, boost::noncopyable >
    ("HarmonicAngleForceComputeGPU", init< boost::shared_ptr<SystemDefinition> >())
    .def("setBlockSize", &HarmonicAngleForceComputeGPU::setBlockSize)
    ;
    }

