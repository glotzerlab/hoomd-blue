// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

/*! \file PCNDAngleForceComputeGPU.cc
    \brief Defines PCNDAngleForceComputeGPU
*/

#include "PCNDAngleForceComputeGPU.h"

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
        m_exec_conf->msg->error()
	    << "Creating a PCNDAngleForceComputeGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing PCNDAngleForceComputeGPU");
        }
    
    PCNDtimestep=0;

    // allocate and zero device memory
    GPUArray<Scalar2> params(m_pcnd_angle_data->getNTypes(), m_exec_conf);
    m_params.swap(params);

    unsigned int warp_size = m_exec_conf->dev_prop.warpSize;
    m_tuner.reset(
        new Autotuner(warp_size, 1024, warp_size, 5, 100000, "pcnd_angle", this->m_exec_conf));
    }

PCNDAngleForceComputeGPU::~PCNDAngleForceComputeGPU() { }

/*! \param type Type of the angle to set parameters for
    \param Xi Root mean square magnitude of the PCND forces
    \param Tau Correlation time

    Sets parameters for the potential of a particular angle type and updates the
    parameters on the GPU.
*/
void PCNDAngleForceComputeGPU::setParams(unsigned int type, Scalar Xi, Scalar Tau)
    {
    PCNDAngleForceCompute::setParams(type, Xi, Tau);

    ArrayHandle<Scalar2> h_params(m_params, access_location::host, access_mode::readwrite);
    // update the local copy of the memory
    h_params.data[type] = make_scalar2(Xi, Tau);
    }
	
/*! Internal method for computing the forces on the GPU.
    \post The force data on the GPU is written with the calculated forces

    \param timestep Current time step of the simulation

    Calls gpu_compute_PCND_angle_forces to do the dirty work.
*/
void PCNDAngleForceComputeGPU::computeForces(uint64_t timestep)
    {
    
    // start the profile
    if (m_prof)
	m_prof->push(m_exec_conf, "PCND Angle");

    // the angle table is up to date: we are good to go. Call the kernel
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_tag(m_pdata->getTags(), access_location::device, access_mode::read);

    BoxDim box = m_pdata->getGlobalBox();

    //Not necessary - force and virial are zeroed in the kernel
    //m_force.memclear();
    //m_virial.memclear();
    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar2> d_params(m_params, access_location::device, access_mode::read);
    
    ArrayHandle<AngleData::members_t> d_gpu_anglelist(m_pcnd_angle_data->getGPUTable(),
		                                      access_location::device,
						      access_mode::read);
    ArrayHandle<unsigned int> d_gpu_angle_pos_list(m_pcnd_angle_data->getGPUPosTable(),
		                                   access_location::device,
						   access_mode::read);
    ArrayHandle<unsigned int> d_gpu_n_angles(m_pcnd_angle_data->getNGroupsArray(),
		                             access_location::device,
					     access_mode::read);

    // run the kernel on the GPU
    m_tuner->begin();
    kernel::gpu_compute_PCND_angle_forces(d_force.data,
                                   d_virial.data,
                                   m_virial.getPitch(),
                                   d_tag.data,
				   m_pdata->getN(),
                                   d_pos.data,
                                   box,
                                   d_gpu_anglelist.data,
                                   d_gpu_angle_pos_list.data,
                                   m_pcnd_angle_data->getGPUTableIndexer().getW(),
                                   d_gpu_n_angles.data,
                                   d_params.data,
                                   m_pcnd_angle_data->getNTypes(),
                                   m_tuner->getParam(),
                                   timestep,
                                   PCNDtimestep);
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
