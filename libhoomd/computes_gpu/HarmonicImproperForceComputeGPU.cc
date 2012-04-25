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

/*! \file HarmonicImproperForceComputeGPU.cc
    \brief Defines HarmonicImproperForceComputeGPU
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include "HarmonicImproperForceComputeGPU.h"

#include <boost/python.hpp>
using namespace boost::python;

#include <boost/bind.hpp>
using namespace boost;

using namespace std;

/*! \param sysdef System to compute improper forces on
*/
HarmonicImproperForceComputeGPU::HarmonicImproperForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef)
        : HarmonicImproperForceCompute(sysdef), m_block_size(64)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a ImproperForceComputeGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing ImproperForceComputeGPU");
        }
        
    // allocate and zero device memory
    GPUArray<float2> params(m_improper_data->getNDihedralTypes(), exec_conf);
    m_params.swap(params);
    }

HarmonicImproperForceComputeGPU::~HarmonicImproperForceComputeGPU()
    {
    }

/*! \param type Type of the improper to set parameters for
    \param K Stiffness parameter for the force computation.
        \param chi Equilibrium value of the dihedral angle.

    Sets parameters for the potential of a particular improper type and updates the
    parameters on the GPU.
*/
void HarmonicImproperForceComputeGPU::setParams(unsigned int type, Scalar K, Scalar chi)
    {
    HarmonicImproperForceCompute::setParams(type, K, chi);
    
    ArrayHandle<float2> h_params(m_params, access_location::host, access_mode::readwrite);
    // update the local copy of the memory
    h_params.data[type] = make_float2(float(K), float(chi));
    }

/*! Internal method for computing the forces on the GPU.
    \post The force data on the GPU is written with the calculated forces

    \param timestep Current time step of the simulation

    Calls gpu_compute_harmonic_improper_forces to do the dirty work.
*/
void HarmonicImproperForceComputeGPU::computeForces(unsigned int timestep)
    {
    // start the profile
    if (m_prof) m_prof->push(exec_conf, "Harmonic Improper");
    

    ArrayHandle<uint4> d_gpu_dihedral_list(m_improper_data->getGPUDihedralList(), access_location::device,access_mode::read);
    ArrayHandle<unsigned int> d_n_dihedrals(m_improper_data->getNDihedralsArray(), access_location::device, access_mode::read);
    ArrayHandle<uint1> d_dihedrals_ABCD(m_improper_data->getDihedralABCD(), access_location::device, access_mode::read);

    // the improper table is up to date: we are good to go. Call the kernel
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    BoxDim box = m_pdata->getBox();
      
    ArrayHandle<Scalar4> d_force(m_force,access_location::device,access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial,access_location::device,access_mode::overwrite);
    ArrayHandle<float2> d_params(m_params, access_location::device, access_mode::read);

    // run the kernel in parallel on all GPUs
    gpu_compute_harmonic_improper_forces(d_force.data,
                                         d_virial.data,
                                         m_virial.getPitch(),
                                         m_pdata->getN(),
                                         d_pos.data,
                                         box,
                                         d_gpu_dihedral_list.data,
                                         d_dihedrals_ABCD.data,
                                         m_improper_data->getGPUDihedralList().getPitch(),
                                         d_n_dihedrals.data,
                                         d_params.data,
                                         m_improper_data->getNDihedralTypes(),
                                         m_block_size);
    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
     
    if (m_prof) m_prof->pop(exec_conf);
    }

void export_HarmonicImproperForceComputeGPU()
    {
    class_<HarmonicImproperForceComputeGPU, boost::shared_ptr<HarmonicImproperForceComputeGPU>, bases<HarmonicImproperForceCompute>, boost::noncopyable >
    ("HarmonicImproperForceComputeGPU", init< boost::shared_ptr<SystemDefinition> >())
    .def("setBlockSize", &HarmonicImproperForceComputeGPU::setBlockSize)
    ;
    }

