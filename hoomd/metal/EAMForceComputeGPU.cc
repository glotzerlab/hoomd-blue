// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.



// Maintainer: morozov

/**
powered by:
Moscow group.
*/

/*! \file EAMForceComputeGPU.cc
    \brief Defines the EAMForceComputeGPU class
*/


#include "EAMForceComputeGPU.h"
#include <cuda_runtime.h>

#include <stdexcept>

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

namespace py = pybind11;
using namespace std;

/*! \param sysdef System to compute forces on
    \param filename Name of EAM potential file to load
    \param type_of_file Undocumented parameter
*/
EAMForceComputeGPU::EAMForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef, char *filename, int type_of_file)
    : EAMForceCompute(sysdef, filename, type_of_file)
    {
    #ifndef SINGLE_PRECISION
    m_exec_conf->msg->warning() << "pair.eam does not work on the GPU in double precision builds" << endl;
    #endif

    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a EAMForceComputeGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing EAMForceComputeGPU");
        }

    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "pair_eam", this->m_exec_conf));

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


void EAMForceComputeGPU::computeForces(unsigned int timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);

    // start the profile
    if (m_prof) m_prof->push(m_exec_conf, "EAM pair");

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
    ArrayHandle<unsigned int> d_head_list(this->m_nlist->getHeadList(), access_location::device, access_mode::read);

    // access the particle data
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    BoxDim box = m_pdata->getBox();

    ArrayHandle<Scalar4> d_force(m_force,access_location::device,access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial,access_location::device,access_mode::overwrite);

    EAMTexInterArrays eam_arrays;
    eam_arrays.atomDerivativeEmbeddingFunction = (Scalar *)d_atomDerivativeEmbeddingFunction;
    m_tuner->begin();
    eam_data.block_size = m_tuner->getParam();
    gpu_compute_eam_tex_inter_forces(d_force.data,
                                     d_virial.data,
                                     m_virial.getPitch(),
                                     m_pdata->getN(),
                                     d_pos.data,
                                     box,
                                     d_n_neigh.data,
                                     d_nlist.data,
                                     d_head_list.data,
                                     this->m_nlist->getNListArray().getPitch(),
                                     eam_tex_data,
                                     eam_arrays,
                                     eam_data,
                                     m_exec_conf->getComputeCapability()/10,
                                     m_exec_conf->dev_prop.maxTexture1DLinear);

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();

    if (m_prof) m_prof->pop(m_exec_conf);
    }

void export_EAMForceComputeGPU(py::module& m)
    {
    py::class_<EAMForceComputeGPU, std::shared_ptr<EAMForceComputeGPU> >(m, "EAMForceComputeGPU", py::base<EAMForceCompute>())
    .def(py::init< std::shared_ptr<SystemDefinition>, char*, int >())
        ;
    }
