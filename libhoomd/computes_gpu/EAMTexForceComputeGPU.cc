/**
powered by:
Moscow group.
*/

/*! \file EAMTexForceComputeGPU.cc
	\brief Defines the EAMTexForceComputeGPU class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include "EAMTexForceComputeGPU.h"
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
EAMTexForceComputeGPU::EAMTexForceComputeGPU(boost::shared_ptr<SystemDefinition> sysdef, boost::shared_ptr<NeighborList> nlist, Scalar r_cut, char *filename) 
	: EAMForceCompute(sysdef, nlist, r_cut, filename)
	{
	// can't run on the GPU if there aren't any GPUs in the execution configuration
	if (exec_conf.gpu.size() == 0)
		{
		cerr << endl << "***Error! Creating a EAMTexForceComputeGPU with no GPU in the execution configuration" << endl << endl;
		throw std::runtime_error("Error initializing EAMTexForceComputeGPU");
		}
	
	if (m_ntypes > 44)
		{
		cerr << endl << "***Error! EAMTexForceComputeGPU cannot handle " << m_ntypes << " types" << endl << endl;
		throw runtime_error("Error initializing EAMTexForceComputeGPU");
		}
		
	// default block size is the highest performance in testing on different hardware
	// choose based on compute capability of the device
	cudaDeviceProp deviceProp;
	int dev;
	exec_conf.gpu[0]->call(bind(cudaGetDevice, &dev));	
	exec_conf.gpu[0]->call(bind(cudaGetDeviceProperties, &deviceProp, dev));
	if (deviceProp.major == 1 && deviceProp.minor == 0)
		m_block_size = 320;
	else if (deviceProp.major == 1 && deviceProp.minor == 1)
		m_block_size = 256;
	else if (deviceProp.major == 1 && deviceProp.minor < 4)
		m_block_size = 352;
	else
		{
		cout << "***Warning! Unknown compute " << deviceProp.major << "." << deviceProp.minor << " when tuning block size for EAMTexForceComputeGPU" << endl;
		m_block_size = 64;
		}
		
	// ulf workaround setup
	#ifndef DISABLE_ULF_WORKAROUND
	// the ULF workaround is needed on GTX280 and older GPUS
	// it is not needed on C1060, S1070, GTX285, GTX295, and (hopefully) newer ones
	m_ulf_workaround = true;
	
	if (deviceProp.major == 1 && deviceProp.minor >= 3)
		m_ulf_workaround = false;
	if (string(deviceProp.name) == "GTX 280")
		m_ulf_workaround = true;
	if (string(deviceProp.name) == "GeForce GTX 280")
		m_ulf_workaround = true;
		

	if (m_ulf_workaround)
		cout << "Notice: ULF bug workaround enabled for EAMTexForceComputeGPU" << endl;
	#else
	m_ulf_workaround = false;
	#endif
/*
	if (m_slj) cout << "Notice: Using Diameter-Shifted EAM Pair Potential for EAMTexForceComputeGPU" << endl;
	else cout << "Diameter-Shifted EAM Pair Potential is NOT set for EAMTexForceComputeGPU" << endl;
*/	
	// allocate the coeff data on the GPU
	int nbytes = sizeof(float2)*m_pdata->getNTypes()*m_pdata->getNTypes();
	
	d_coeffs.resize(exec_conf.gpu.size());
	d_atomDerivativeEmbeddingFunction.resize(exec_conf.gpu.size());
	eam_linear_data.resize(exec_conf.gpu.size());
	loadFile(filename);
	eam_data.nr = nr; 
	eam_data.nrho = nrho; 
	eam_data.dr = dr; 
	eam_data.rdr = 1.0/dr; 
	eam_data.drho = drho; 
	eam_data.rdrho = 1.0/drho; 
	eam_data.r_cut = r_cut; 
	eam_data.r_cutsq = r_cut * r_cut;
	eam_data.block_size = m_block_size;
	const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly();	
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		exec_conf.gpu[cur_gpu]->setTag(__FILE__, __LINE__);
		exec_conf.gpu[cur_gpu]->call(bind(cudaMallocHack, (void **)((void *)&d_coeffs[cur_gpu]), nbytes));
		assert(d_coeffs[cur_gpu]);
		exec_conf.gpu[cur_gpu]->call(bind(cudaMemset, (void *)d_coeffs[cur_gpu], 0, nbytes));
		exec_conf.gpu[cur_gpu]->call(bind(cudaMallocHack, (void **)((void *)&d_atomDerivativeEmbeddingFunction[cur_gpu]), arrays.nparticles * sizeof(float)));
		assert(d_atomDerivativeEmbeddingFunction[cur_gpu]);
		exec_conf.gpu[cur_gpu]->call(bind(cudaMemset, (void *)d_atomDerivativeEmbeddingFunction[cur_gpu], 0, arrays.nparticles * sizeof(float)));
		//Allocate mem on GPU for tables for EAM 
		cudaChannelFormatDesc eam_desc = cudaCreateChannelDesc< float >();
		#define copy_table(gpuname, cpuname, count) \
		exec_conf.gpu[cur_gpu]->call(bind(cudaMallocHack, (void **)((void *)&eam_linear_data[cur_gpu].gpuname), sizeof(Scalar) * count));\
		exec_conf.gpu[cur_gpu]->call(bind(cudaMemcpy, eam_linear_data[cur_gpu].gpuname,&cpuname[0], sizeof(Scalar) * count, cudaMemcpyHostToDevice));
		copy_table(pairPotential, pairPotential, ((m_ntypes * m_ntypes / 2) + 1) * nr);
		
		copy_table(electronDensity, electronDensity, m_ntypes * nr);
		copy_table(embeddingFunction, embeddingFunction, m_ntypes * nrho);
		copy_table(derivativePairPotential, derivativePairPotential, ((m_ntypes * m_ntypes / 2) + 1) * nr);
		copy_table(derivativeElectronDensity, derivativeElectronDensity, m_ntypes * nr);
		copy_table(derivativeEmbeddingFunction, derivativeEmbeddingFunction, m_ntypes * nrho);
		
		#undef copy_table
		eam_linear_data[cur_gpu].size_pairPotential = sizeof(Scalar) * ((m_ntypes * m_ntypes / 2) + 1) * nr;
		eam_linear_data[cur_gpu].size_electronDensity = sizeof(Scalar) *  m_ntypes * nr;
		eam_linear_data[cur_gpu].size_embeddingFunction = sizeof(Scalar) *  m_ntypes * nrho;
		eam_linear_data[cur_gpu].size_derivativePairPotential = sizeof(Scalar) * ((m_ntypes * m_ntypes / 2) + 1) * nr;
		eam_linear_data[cur_gpu].size_derivativeElectronDensity = sizeof(Scalar) * m_ntypes * nr;
		eam_linear_data[cur_gpu].size_derivativeEmbeddingFunction = sizeof(Scalar) * m_ntypes * nrho;
		}
	// allocate the coeff data on the CPU
	h_coeffs = new float2[m_pdata->getNTypes()*m_pdata->getNTypes()];
	
	}
	

EAMTexForceComputeGPU::~EAMTexForceComputeGPU()
	{
	// free the coefficients on the GPU
	exec_conf.tagAll(__FILE__, __LINE__);
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		assert(d_coeffs[cur_gpu]);
		exec_conf.gpu[cur_gpu]->call(bind(cudaFree, (void *)d_coeffs[cur_gpu]));
		}
	delete[] h_coeffs;
	}
	
/*! \param block_size Size of the block to run on the device
	Performance of the code may be dependant on the block size run
	on the GPU. \a block_size should be set to be a multiple of 32.
*/
void EAMTexForceComputeGPU::setBlockSize(int block_size)
	{
	m_block_size = block_size;
	}


void EAMTexForceComputeGPU::computeForces(unsigned int timestep)
	{
	// start by updating the neighborlist
	m_nlist->compute(timestep);
	
	// start the profile
	if (m_prof) m_prof->push(exec_conf, "EAM pair");
	
	// The GPU implementation CANNOT handle a half neighborlist, error out now
	bool third_law = m_nlist->getStorageMode() == NeighborList::half;
	if (third_law)
		{
		cerr << endl << "***Error! EAMTexForceComputeGPU cannot handle a half neighborlist" << endl << endl;
		throw runtime_error("Error computing forces in EAMTexForceComputeGPU");
		}
	
	// access the neighbor list, which just selects the neighborlist into the device's memory, copying
	// it there if needed
	vector<gpu_nlist_array>& nlist = m_nlist->getListGPU();

	// access the particle data
	vector<gpu_pdata_arrays>& pdata = m_pdata->acquireReadOnlyGPU();
	gpu_boxsize box = m_pdata->getBoxGPU();
	
	// run the kernel on all GPUs in parallel
	exec_conf.tagAll(__FILE__, __LINE__);
	
	for (unsigned int cur_gpu = 0; cur_gpu < exec_conf.gpu.size(); cur_gpu++)
		{
		EAMTexArrays eam_arrays;
		eam_arrays.atomDerivativeEmbeddingFunction = (float *)d_atomDerivativeEmbeddingFunction[cur_gpu];
		exec_conf.gpu[cur_gpu]->callAsync(bind(gpu_compute_eam_linear_forces, 
			m_gpu_forces[cur_gpu].d_data, 
			pdata[cur_gpu], 
			box, 
			nlist[cur_gpu], 
			(float2 *)d_coeffs[cur_gpu], 
			m_pdata->getNTypes(), 
			eam_linear_data[cur_gpu],
			eam_arrays,
			eam_data));
		}
		
	exec_conf.syncAll();

	m_pdata->release();
	
	// the force data is now only up to date on the gpu
	m_data_location = gpu;

	Scalar avg_neigh = m_nlist->estimateNNeigh();
	int64_t n_calc = int64_t(avg_neigh * m_pdata->getN());
	int64_t mem_transfer = m_pdata->getN() * (4 + 16 + 20) + n_calc * (4 + 16);
	int64_t flops = n_calc * (3+12+5+2+2+6+3+2+7);

	if (m_prof) m_prof->pop(exec_conf, flops, mem_transfer);
	}

void export_EAMTexForceComputeGPU()
	{
	class_<EAMTexForceComputeGPU, boost::shared_ptr<EAMTexForceComputeGPU>, bases<EAMForceCompute>, boost::noncopyable >
		("EAMTexForceComputeGPU", init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<NeighborList>, Scalar, char* >())
		.def("setBlockSize", &EAMTexForceComputeGPU::setBlockSize)
		;
	}

#ifdef WIN32
#pragma warning( pop )
#endif

