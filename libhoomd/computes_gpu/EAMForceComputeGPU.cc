/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
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
     \param nlist Neighborlist to use for computing the forces
    \param r_cut Cuttoff radius beyond which the force is 0
    \param filename     Name of potential`s file.
*/
EAMForceComputeGPU::EAMForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef, char *filename, int type_of_file)
    : EAMForceCompute(sysdef, filename, type_of_file)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!exec_conf->isCUDAEnabled())
        {
        cerr << endl << "***Error! Creating a EAMForceComputeGPU with no GPU in the execution configuration" << endl << endl;
        throw std::runtime_error("Error initializing EAMForceComputeGPU");
        }

    if (m_ntypes > 44)
        {
        cerr << endl << "***Error! EAMForceComputeGPU cannot handle " << m_ntypes << " types" << endl << endl;
        throw runtime_error("Error initializing EAMForceComputeGPU");
        }

    m_block_size = 64;

/*
    if (m_slj) cout << "Notice: Using Diameter-Shifted EAM Pair Potential for EAMForceComputeGPU" << endl;
    else cout << "Diameter-Shifted EAM Pair Potential is NOT set for EAMForceComputeGPU" << endl;
*/
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

    cudaMalloc(&d_atomDerivativeEmbeddingFunction, m_pdata->getN() * sizeof(float));
    cudaMemset(d_atomDerivativeEmbeddingFunction, 0, m_pdata->getN() * sizeof(float));
    
    //Allocate mem on GPU for tables for EAM in cudaArray
    cudaChannelFormatDesc eam_desc = cudaCreateChannelDesc< float >();
    #define copy_table(gpuname, cpuname, count) \
    cudaMallocArray(&eam_tex_data.gpuname, &eam_desc,  count, 1);\
    cudaMemcpyToArray(eam_tex_data.gpuname, 0, 0, &cpuname[0], count * sizeof(float), cudaMemcpyHostToDevice);

    copy_table(electronDensity, electronDensity, m_ntypes * nr);
    copy_table(embeddingFunction, embeddingFunction, m_ntypes * nrho);
    copy_table(derivativeElectronDensity, derivativeElectronDensity, m_ntypes * nr);
    copy_table(derivativeEmbeddingFunction, derivativeEmbeddingFunction, m_ntypes * nrho);

    #undef copy_table
    eam_desc = cudaCreateChannelDesc< float2 >();
    cudaMallocArray(&eam_tex_data.pairPotential, &eam_desc,  ((m_ntypes * m_ntypes / 2) + 1) * nr, 1);
    cudaMemcpyToArray(eam_tex_data.pairPotential, 0, 0, &pairPotential[0], ((m_ntypes * m_ntypes / 2) + 1) * nr *sizeof(float2), cudaMemcpyHostToDevice);
    
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
        cerr << endl << "***Error! EAMForceComputeGPU cannot handle a half neighborlist" << endl << endl;
        throw runtime_error("Error computing forces in EAMForceComputeGPU");
        }

    // access the neighbor list, which just selects the neighborlist into the device's memory, copying
    // it there if needed
    ArrayHandle<unsigned int> d_n_neigh(this->m_nlist->getNNeighArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_nlist(this->m_nlist->getNListArray(), access_location::device, access_mode::read);
    Index2D nli = this->m_nlist->getNListIndexer();

    // access the particle data
    gpu_pdata_arrays& d_pdata = m_pdata->acquireReadOnlyGPU();
    gpu_boxsize box = m_pdata->getBoxGPU();

    EAMTexInterArrays eam_arrays;
    eam_arrays.atomDerivativeEmbeddingFunction = (float *)d_atomDerivativeEmbeddingFunction;
    gpu_compute_eam_tex_inter_forces(m_gpu_forces.d_data,
                                     d_pdata,
                                     box,
                                     d_n_neigh.data,
                                     d_nlist.data,
                                     nli,
                                     eam_tex_data,
                                     eam_arrays,
                                     eam_data);
    
    if (exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_pdata->release();

    // the force data is now only up to date on the gpu
    m_data_location = gpu;

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

