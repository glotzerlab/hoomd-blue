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
#include "cuda_runtime.h"

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
EAMForceComputeGPU::EAMForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef, char *filename, int type_of_file)
	: EAMForceCompute(sysdef, filename, type_of_file)
	{
		printf("Init Potential EAMForceComputeGPU\n");
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

	// default block size is the highest performance in testing on different hardware
	// choose based on compute capability of the device
    m_block_size = 64;
/*
	if (m_slj) cout << "Notice: Using Diameter-Shifted EAM Pair Potential for EAMForceComputeGPU" << endl;
	else cout << "Diameter-Shifted EAM Pair Potential is NOT set for EAMForceComputeGPU" << endl;
*/
	// allocate the coeff data on the GPU
	int nbytes = sizeof(float2)*m_pdata->getNTypes()*m_pdata->getNTypes();

	//Load file with data for potential
	loadFile(filename, type_of_file);
	// Set potential parameters
	eam_data.ntypes = m_ntypes;
	eam_data.nr = nr;
	eam_data.nrho = nrho;
	eam_data.dr = dr;
	eam_data.rdr = 1.0/dr;
	eam_data.drho = drho;
	eam_data.rdrho = 1.0/drho;
	eam_data.r_cut = m_r_cut;
	eam_data.r_cutsq = m_r_cut * m_r_cut;
	eam_data.block_size = m_block_size;


    cudaMalloc(&d_coeffs, nbytes);
	cudaMemset(d_coeffs, 0, nbytes);
	cudaMalloc(&d_atomDerivativeEmbeddingFunction, m_pdata->getN() * sizeof(float));
	cudaMemset(d_atomDerivativeEmbeddingFunction, 0, m_pdata->getN() * sizeof(float));

    #define copy_table(gpuname, cpuname, count) \
    cudaMalloc(&(gpuname), sizeof(Scalar) * count);\
    cudaMemcpy((gpuname), &(cpuname), sizeof(Scalar) * count, cudaMemcpyHostToDevice);

    copy_table(d_pairPotential, pairPotential, (int)(0.5 * nr * (m_ntypes + 1) * m_ntypes));
    copy_table(d_electronDensity, electronDensity, nr * m_ntypes * m_ntypes);
    copy_table(d_embeddingFunction, embeddingFunction, m_ntypes * nrho);
    copy_table(d_derivativePairPotential, derivativePairPotential, (int)(0.5 * nr * (m_ntypes + 1) * m_ntypes));
    copy_table(d_derivativeElectronDensity, derivativeElectronDensity, nr * m_ntypes * m_ntypes);
    copy_table(d_derivativeEmbeddingFunction, derivativeEmbeddingFunction, m_ntypes * nrho);
    
    CHECK_CUDA_ERROR();
    
    #undef copy_table

    // allocate the coeff data on the CPU
	h_coeffs = new float2[m_pdata->getNTypes()*m_pdata->getNTypes()];
	}


EAMForceComputeGPU::~EAMForceComputeGPU()
	{
	// free the coefficients on the GPU
	cudaFree(d_coeffs);
	cudaFree(d_atomDerivativeEmbeddingFunction);
	cudaFree(d_pairPotential);
	cudaFree(d_electronDensity);
	cudaFree(d_embeddingFunction);
	cudaFree(d_derivativePairPotential);
	cudaFree(d_derivativeElectronDensity);
	cudaFree(d_derivativeEmbeddingFunction);
	delete[] h_coeffs;
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

    EAMArrays eam_arrays;
    eam_arrays.electronDensity = (float *)d_electronDensity;
    eam_arrays.pairPotential = (float *)d_pairPotential;
    eam_arrays.embeddingFunction = (float *)d_embeddingFunction;
    eam_arrays.derivativeElectronDensity = (float *)d_derivativeElectronDensity;
    eam_arrays.derivativePairPotential = (float *)d_derivativePairPotential;
    eam_arrays.derivativeEmbeddingFunction = (float *)d_derivativeEmbeddingFunction;
    eam_arrays.atomDerivativeEmbeddingFunction = (float *)d_atomDerivativeEmbeddingFunction;
    gpu_compute_eam_forces(m_gpu_forces.d_data,
                           d_pdata,
                           box,
                           d_n_neigh.data,
                           d_nlist.data,
                           nli,
                           (float2*)d_coeffs,
                           m_pdata->getNTypes(),
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
