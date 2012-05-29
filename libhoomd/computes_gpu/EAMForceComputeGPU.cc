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


// Maintainer: morozov

/**
powered by:
Moscow group.
*/

/*! \file EAMForceComputeGPU.cc
    \brief Defines the EAMForceComputeGPU class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "EAMForceComputeGPU.h"
#include <cuda_runtime.h>

#include <stdexcept>

#include <boost/python.hpp>
using namespace boost::python;

#include <boost/bind.hpp>

using namespace boost;
using namespace std;

/*! \param sysdef System to compute forces on
    \param filename Name of EAM potential file to load
    \param type_of_file Undocumented parameter
*/
EAMForceComputeGPU::EAMForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef, char *filename, int type_of_file)
    : EAMForceCompute(sysdef, filename, type_of_file)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a EAMForceComputeGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing EAMForceComputeGPU");
        }

    m_block_size = 64;

    // allocate the coeff data on the GPU
    loadFile(filename, type_of_file);
    eam_data.nr = nr;
    eam_data.nrho = nrho;
    eam_data.dr = dr;
    eam_data.rdr = 1.0/dr;
    eam_data.drho = drho;
    eam_data.rdrho = 1.0/drho;
    eam_data.r_cut = m_r_cut;
    eam_data.r_cutsq = m_r_cut * m_r_cut;
    eam_data.block_size = m_block_size;

    cudaMalloc(&d_atomDerivativeEmbeddingFunction, m_pdata->getN() * sizeof(Scalar));
    cudaMemset(d_atomDerivativeEmbeddingFunction, 0, m_pdata->getN() * sizeof(Scalar));
    
    //Allocate mem on GPU for tables for EAM in cudaArray
    cudaChannelFormatDesc eam_desc = cudaCreateChannelDesc< Scalar >();

    cudaMallocArray(&eam_tex_data.electronDensity, &eam_desc, m_ntypes * nr, 1);
    cudaMemcpyToArray(eam_tex_data.electronDensity, 0, 0, &electronDensity[0], m_ntypes * nr * sizeof(Scalar), cudaMemcpyHostToDevice);
    
    cudaMallocArray(&eam_tex_data.embeddingFunction, &eam_desc, m_ntypes * nrho, 1);
    cudaMemcpyToArray(eam_tex_data.embeddingFunction, 0, 0, &embeddingFunction[0], m_ntypes * nrho * sizeof(Scalar), cudaMemcpyHostToDevice);

    cudaMallocArray(&eam_tex_data.derivativeElectronDensity, &eam_desc, m_ntypes * nr, 1);
    cudaMemcpyToArray(eam_tex_data.derivativeElectronDensity, 0, 0, &derivativeElectronDensity[0], m_ntypes * nr * sizeof(Scalar), cudaMemcpyHostToDevice);
    
    cudaMallocArray(&eam_tex_data.derivativeEmbeddingFunction, &eam_desc, m_ntypes * nrho, 1);
    cudaMemcpyToArray(eam_tex_data.derivativeEmbeddingFunction, 0, 0, &derivativeEmbeddingFunction[0], m_ntypes * nrho * sizeof(Scalar), cudaMemcpyHostToDevice);

    eam_desc = cudaCreateChannelDesc< Scalar2 >();
    cudaMallocArray(&eam_tex_data.pairPotential, &eam_desc,  ((m_ntypes * m_ntypes / 2) + 1) * nr, 1);
    cudaMemcpyToArray(eam_tex_data.pairPotential, 0, 0, &pairPotential[0], ((m_ntypes * m_ntypes / 2) + 1) * nr *sizeof(Scalar2), cudaMemcpyHostToDevice);
    
    CHECK_CUDA_ERROR();
    }


EAMForceComputeGPU::~EAMForceComputeGPU()
    {
    // free the coefficients on the GPU
    cudaFree(d_atomDerivativeEmbeddingFunction);
    cudaFreeArray(eam_tex_data.pairPotential);
    cudaFreeArray(eam_tex_data.electronDensity);
    cudaFreeArray(eam_tex_data.embeddingFunction);
    cudaFreeArray(eam_tex_data.derivativeElectronDensity);
    cudaFreeArray(eam_tex_data.derivativeEmbeddingFunction);
    }

/*! \param block_size Size of the block to run on the device
    Performance of the code may be dependant on the block size run
    on the GPU. \a block_size should be set to be a multiple of 32.
*/
void EAMForceComputeGPU::setBlockSize(int block_size)
    {
    m_block_size = block_size;
    }


void EAMForceComputeGPU::computeForces(unsigned int timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);

    // start the profile
    if (m_prof) m_prof->push(exec_conf, "EAM pair");

    // The GPU implementation CANNOT handle a half neighborlist, error out now
    bool third_law = m_nlist->getStorageMode() == NeighborList::half;
    if (third_law)
        {
        m_exec_conf->msg->error() << "EAMForceComputeGPU cannot handle a half neighborlist" << endl;
        throw runtime_error("Error computing forces in EAMForceComputeGPU");
        }

    // access the neighbor list, which just selects the neighborlist into the device's memory, copying
    // it there if needed
    ArrayHandle<unsigned int> d_n_neigh(this->m_nlist->getNNeighArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_nlist(this->m_nlist->getNListArray(), access_location::device, access_mode::read);
    Index2D nli = this->m_nlist->getNListIndexer();

    // access the particle data
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    BoxDim box = m_pdata->getBox();
  
    ArrayHandle<Scalar4> d_force(m_force,access_location::device,access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial,access_location::device,access_mode::overwrite);

    EAMTexInterArrays eam_arrays;
    eam_arrays.atomDerivativeEmbeddingFunction = (Scalar *)d_atomDerivativeEmbeddingFunction;
    gpu_compute_eam_tex_inter_forces(d_force.data,
                                     d_virial.data,
                                     m_virial.getPitch(),
                                     m_pdata->getN(),
                                     d_pos.data,
                                     box,
                                     d_n_neigh.data,
                                     d_nlist.data,
                                     nli,
                                     eam_tex_data,
                                     eam_arrays,
                                     eam_data);
    
    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (m_prof) m_prof->pop(exec_conf);
    }

void export_EAMForceComputeGPU()
    {
    class_<EAMForceComputeGPU, boost::shared_ptr<EAMForceComputeGPU>, bases<EAMForceCompute>, boost::noncopyable >
        ("EAMForceComputeGPU", init< boost::shared_ptr<SystemDefinition>, char*, int >())
        .def("setBlockSize", &EAMForceComputeGPU::setBlockSize)
        ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

