// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

/*! \file PCNDAngleForceComputeGPU.cc
    \brief Defines PCNDAngleForceComputeGPU
*/

#include "PCNDAngleForceComputeGPU.h"
#include <cuda.h>
//#include <curand.h>
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"
#include <time.h>

//curandGenerator_t gen;
float *devData, *hostData, *devCarryover, *hostCarryover;
uint16_t seed;
uint16_t seed2;
uint64_t PCNDtimestep;

using namespace std;

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute angle forces on
*/
PCNDAngleForceComputeGPU::PCNDAngleForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef)
        : PCNDAngleForceCompute(sysdef)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a PCNDAngleForceComputeGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing PCNDAngleForceComputeGPU");
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
    PCNDtimestep=0;

    // allocate and zero device memory
    GPUArray<Scalar2> params (m_pcnd_angle_data->getNTypes(), m_exec_conf);
    m_params.swap(params);
    GPUArray<Scalar2> PCNDsr(m_pcnd_angle_data->getNTypes(), m_exec_conf);
    m_PCNDsr.swap(PCNDsr);
    GPUArray<Scalar4> PCNDepow(m_pcnd_angle_data->getNTypes(), m_exec_conf);
    m_PCNDepow.swap(PCNDepow);

    unsigned int warp_size = m_exec_conf->dev_prop.warpSize;
    m_tuner.reset(
        new Autotuner(warp_size, 1024, warp_size, 5, 100000, "pcnd_angle", this->m_exec_conf));
    }

PCNDAngleForceComputeGPU::~PCNDAngleForceComputeGPU() { }

/*! \param type Type of the angle to set parameters for
    \param K Stiffness parameter for the force computation
    \param t_0 Equilibrium angle (in radians) for the force computation
    \param cg_type the type of course grained angle we are using
    \param eps the well depth
    \param sigma the particle radius

    Sets parameters for the potential of a particular angle type and updates the
    parameters on the GPU.
*/
void PCNDAngleForceComputeGPU::setParams(unsigned int type, Scalar Xi, Scalar Tau, unsigned int PCND_type, uint16_t particle_sum, Scalar particle_index)
    {
    PCNDAngleForceCompute::setParams(type, Xi, Tau, PCND_type, particle_sum, particle_index);

    const Scalar myPow1 = cgPow1[PCND_type];
    const Scalar myPow2 = cgPow2[PCND_type];
    const Scalar myPref = prefact[PCND_type];

    Scalar my_rcut = particle_index * exp(Scalar(1.0) / (myPow1 - myPow2) * log(myPow1 / myPow2));

    ArrayHandle<Scalar2> h_params(m_params, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar2> h_PCNDsr(m_PCNDsr, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_PCNDepow(m_PCNDepow, access_location::host, access_mode::readwrite);
    // update the local copy of the memory
    h_params.data[type] = make_scalar2(Xi, Tau);
    h_PCNDsr.data[type] = make_scalar2(particle_index, my_rcut);
    h_PCNDepow.data[type] = make_scalar4(particle_sum, myPow1, myPow2, myPref);
    
    seed = 7 * particle_sum;
    
    /* Allocate n floats on host */
    hostData = (float *)calloc(seed, sizeof(float));

    /* Allocate n floats on device */
    cudaMalloc((void **)&devData, seed*sizeof(float));
    
    //curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
    //curandSetPseudoRandomGeneratorSeed(gen,time(NULL));
    //curandSetPseudoRandomGeneratorSeed(gen,0);
    
    //allocate space to carryover data
    /* Allocate n floats on host */
    seed2 = 7 * (particle_sum + 1);
    hostCarryover = (float *)calloc(seed2, sizeof(float));
    /* Allocate n floats on device */
    cudaMalloc((void **)&devCarryover, seed2*sizeof(float));
                
    }
	
/*! Internal method for computing the forces on the GPU.
    \post The force data on the GPU is written with the calculated forces

    \param timestep Current time step of the simulation

    Calls gpu_compute_PCND_angle_forces to do the dirty work.
*/
void PCNDAngleForceComputeGPU::computeForces(uint64_t timestep)
    {
	//curandGenerateUniform(gen, devData, seed);
	
        //ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
        
	//unsigned int ptag = h_tag.data;
        RandomGenerator rng(hoomd::Seed(RNGIdentifier::PCNDAngleForceCompute, timestep, seed),
		            hoomd::Counter(0));
	
	UniformDistribution<Scalar> uniform(Scalar(0), Scalar(1));
	
	cudaMemcpy(hostData, devData, seed * sizeof(float),cudaMemcpyDeviceToHost);
	//int i;
	//for(i = 0; i < seed; i++) {
        //printf("%1.4f ", hostData[i]);
    //}
	//printf("\n\nRANDNUMN = %f\n\n",hostData[0]);
	
    // start the profile
    if (m_prof)
	m_prof->push(m_exec_conf, "PCND Angle");

    // the angle table is up to date: we are good to go. Call the kernel
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);

    BoxDim box = m_pdata->getGlobalBox();

    //Not necessary - force and virial are zeroed in the kernel
    //m_force.memclear();
    //m_virial.memclear();
    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar2> d_params(m_params, access_location::device, access_mode::read);
    ArrayHandle<Scalar2> d_PCNDsr(m_PCNDsr, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_PCNDepow(m_PCNDepow, access_location::device, access_mode::read);

    ArrayHandle<AngleData::members_t> d_gpu_anglelist(m_pcnd_angle_data->getGPUTable(),
		                                      access_location::device,
						      access_mode::read);
    ArrayHandle<unsigned int> d_gpu_angle_pos_list(m_pcnd_angle_data->getGPUPosTable(),
		                                   access_location::device,
						   access_mode::read);
    ArrayHandle<unsigned int> d_gpu_n_angles(m_pcnd_angle_data->getNGroupsArray(),
		                             access_location::device,
					     access_mode::read);

    // run the kernel
    m_tuner->begin();
    kernel::gpu_compute_PCND_angle_forces(d_force.data,
                                   d_virial.data,
                                   m_virial.getPitch(),
                                   m_pdata->getN(),
                                   d_pos.data,
                                   box,
                                   d_gpu_anglelist.data,
                                   d_gpu_angle_pos_list.data,
                                   m_pcnd_angle_data->getGPUTableIndexer().getW(),
                                   d_gpu_n_angles.data,
                                   d_params.data,
                                   d_PCNDsr.data,
                                   d_PCNDepow.data,
                                   m_pcnd_angle_data->getNTypes(),
                                   m_tuner->getParam(),
                                   //m_exec_conf->getComputeCapability(),
                                   timestep,
                                   devData,
                                   PCNDtimestep,
                                   devCarryover);
    PCNDtimestep=PCNDtimestep+1;

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();

    if (m_prof)
        m_prof->pop(m_exec_conf);
    }

namespace detail
    {
void export_PCNDAngleForceComputeGPU(pybind11::module& m)
    {
    pybind11::class_<PCNDAngleForceComputeGPU,
	             PCNDAngleForceCompute,
	             std::shared_ptr<PCNDAngleForceComputeGPU>>(m,
				                                "PCNDAngleForceComputeGPU")
    .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
