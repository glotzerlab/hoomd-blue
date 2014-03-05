/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
the University of Michigan All rights reserved.

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

/*! \file ComputeThermoGPU.cc
    \brief Contains code for the ComputeThermoGPU class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "ComputeThermoGPU.h"
#include "ComputeThermoGPU.cuh"

#include <boost/python.hpp>
using namespace boost::python;
#include <boost/bind.hpp>
using namespace boost;

#ifdef ENABLE_MPI
#include "Communicator.h"
#include "HOOMDMPI.h"
#endif

#include <iostream>
using namespace std;

/*! \param sysdef System for which to compute thermodynamic properties
    \param group Subset of the system over which properties are calculated
    \param suffix Suffix to append to all logged quantity names
*/

ComputeThermoGPU::ComputeThermoGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                   boost::shared_ptr<ParticleGroup> group,
                                   const std::string& suffix)
    : ComputeThermo(sysdef, group, suffix)
    {
    if (!exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a ComputeThermoGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing ComputeThermoGPU");
        }

    m_block_size = 512;
    // this allocates more memory than necessary but is needed unless the scratch memory
    // is reallocated when the maximum number of particles changes
    m_num_blocks = m_group->getNumMembersGlobal() / m_block_size + 1;

    GPUArray< Scalar4 > scratch(m_num_blocks, exec_conf);
    m_scratch.swap(scratch);

    GPUArray< Scalar > scratch_pressure_tensor(m_num_blocks * 6, exec_conf);
    m_scratch_pressure_tensor.swap(scratch_pressure_tensor);

    // override base class allocation using mapped memory
    GPUArray< Scalar > properties(thermo_index::num_quantities, exec_conf,true);
    m_properties.swap(properties);

    cudaEventCreate(&m_event, cudaEventDisableTiming);
    }

//! Destructor
ComputeThermoGPU::~ComputeThermoGPU()
    {
    cudaEventDestroy(m_event);
    }

/*! Computes all thermodynamic properties of the system in one fell swoop, on the GPU.
 */
void ComputeThermoGPU::computeProperties()
    {
    // just drop out if the group is an empty group
    if (m_group->getNumMembersGlobal() == 0)
        return;

    unsigned int group_size = m_group->getNumMembers();

    if (m_prof) m_prof->push(m_exec_conf,"Thermo");

    assert(m_pdata);
    assert(m_ndof != 0);

    // access the particle data
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(), access_location::device, access_mode::read);
    BoxDim box = m_pdata->getGlobalBox();

    PDataFlags flags = m_pdata->getFlags();

    { // scope these array handles so they are released before the additional terms are added
    // access the net force, pe, and virial
    const GPUArray< Scalar4 >& net_force = m_pdata->getNetForce();
    const GPUArray< Scalar >& net_virial = m_pdata->getNetVirial();
    ArrayHandle<Scalar4> d_net_force(net_force, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_net_virial(net_virial, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_scratch(m_scratch, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_scratch_pressure_tensor(m_scratch_pressure_tensor, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_properties(m_properties, access_location::device, access_mode::overwrite);

    // access the group
    ArrayHandle< unsigned int > d_index_array(m_group->getIndexArray(), access_location::device, access_mode::read);

    // build up args list
    m_num_blocks = m_group->getNumMembers() / m_block_size + 1;
    compute_thermo_args args;
    args.d_net_force = d_net_force.data;
    args.d_net_virial = d_net_virial.data;
    args.virial_pitch = net_virial.getPitch();
    args.ndof = m_ndof;
    args.D = m_sysdef->getNDimensions();
    args.d_scratch = d_scratch.data;
    args.d_scratch_pressure_tensor = d_scratch_pressure_tensor.data;
    args.block_size = m_block_size;
    args.n_blocks = m_num_blocks;
    args.external_virial_xx = m_pdata->getExternalVirial(0);
    args.external_virial_xy = m_pdata->getExternalVirial(1);
    args.external_virial_xz = m_pdata->getExternalVirial(2);
    args.external_virial_yy = m_pdata->getExternalVirial(3);
    args.external_virial_yz = m_pdata->getExternalVirial(4);
    args.external_virial_zz = m_pdata->getExternalVirial(5);

    // perform the computation on the GPU
    gpu_compute_thermo( d_properties.data,
                        d_vel.data,
                        d_index_array.data,
                        group_size,
                        box,
                        args,
                        flags[pdata_flag::pressure_tensor]);

    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    }

    #ifdef ENABLE_MPI
    // in MPI, reduce extensive quantities only when they're needed
    m_properties_reduced = !m_pdata->getDomainDecomposition();

    if (!m_properties_reduced) cudaEventRecord(m_event);
    #endif // ENABLE_MPI

    if (m_prof) m_prof->pop(m_exec_conf);
    }

#ifdef ENABLE_MPI
void ComputeThermoGPU::reduceProperties()
    {
    if (m_properties_reduced) return;

    ArrayHandleAsync<Scalar> h_properties(m_properties, access_location::host, access_mode::readwrite);
    cudaEventSynchronize(m_event);

    // reduce properties
    MPI_Allreduce(MPI_IN_PLACE, h_properties.data, thermo_index::num_quantities, MPI_HOOMD_SCALAR,
            MPI_SUM, m_exec_conf->getMPICommunicator());

    m_properties_reduced = true;
    }
#endif


void export_ComputeThermoGPU()
    {
    class_<ComputeThermoGPU, boost::shared_ptr<ComputeThermoGPU>, bases<ComputeThermo>, boost::noncopyable >
        ("ComputeThermoGPU", init< boost::shared_ptr<SystemDefinition>,
         boost::shared_ptr<ParticleGroup>,
         const std::string& >())
        ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif
