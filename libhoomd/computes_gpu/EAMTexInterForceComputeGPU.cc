/**
powered by:
Moscow group.
*/

/*! \file EAMTexInterForceComputeGPU.cc
	\brief Defines the EAMTexInterForceComputeGPU class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "EAMTexInterForceComputeGPU.h"
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
	\param filename	 Name of potential`s file.
*/
EAMTexInterForceComputeGPU::EAMTexInterForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef, char *filename, int type_of_file)
	: EAMForceCompute(sysdef, filename, type_of_file)
	{
	// can't run on the GPU if there aren't any GPUs in the execution configuration
	if (!exec_conf->isCUDAEnabled())
		{
		cerr << endl << "***Error! Creating a EAMTexInterForceComputeGPU with no GPU in the execution configuration" << endl << endl;
		throw std::runtime_error("Error initializing EAMTexInterForceComputeGPU");
		}

	if (m_ntypes > 44)
		{
		cerr << endl << "***Error! EAMTexInterForceComputeGPU cannot handle " << m_ntypes << " types" << endl << endl;
		throw runtime_error("Error initializing EAMTexInterForceComputeGPU");
		}

    m_block_size = 64;

/*
	if (m_slj) cout << "Notice: Using Diameter-Shifted EAM Pair Potential for EAMTexInterForceComputeGPU" << endl;
	else cout << "Diameter-Shifted EAM Pair Potential is NOT set for EAMTexInterForceComputeGPU" << endl;
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
	const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly();
	cudaMalloc(&d_atomDerivativeEmbeddingFunction, arrays.nparticles * sizeof(float));
	cudaMemset(d_atomDerivativeEmbeddingFunction, 0, arrays.nparticles * sizeof(float));
    
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


EAMTexInterForceComputeGPU::~EAMTexInterForceComputeGPU()
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
void EAMTexInterForceComputeGPU::setBlockSize(int block_size)
	{
	m_block_size = block_size;
	}


void EAMTexInterForceComputeGPU::computeForces(unsigned int timestep)
	{
	// start by updating the neighborlist
	m_nlist->compute(timestep);

	// start the profile
	if (m_prof) m_prof->push(exec_conf, "EAM pair");

	// The GPU implementation CANNOT handle a half neighborlist, error out now
	bool third_law = m_nlist->getStorageMode() == NeighborList::half;
	if (third_law)
		{
		cerr << endl << "***Error! EAMTexInterForceComputeGPU cannot handle a half neighborlist" << endl << endl;
		throw runtime_error("Error computing forces in EAMTexInterForceComputeGPU");
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

void export_EAMTexInterForceComputeGPU()
	{
	class_<EAMTexInterForceComputeGPU, boost::shared_ptr<EAMTexInterForceComputeGPU>, bases<EAMForceCompute>, boost::noncopyable >
		("EAMTexInterForceComputeGPU", init< boost::shared_ptr<SystemDefinition>, char*, int >())
		.def("setBlockSize", &EAMTexInterForceComputeGPU::setBlockSize)
		;
	}

#ifdef WIN32
#pragma warning( pop )
#endif

