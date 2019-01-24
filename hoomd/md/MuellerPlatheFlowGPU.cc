// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "MuellerPlatheFlowGPU.h"
#include "hoomd/HOOMDMPI.h"

namespace py=pybind11;
using namespace std;

//! \file MuellerPlatheFlowGPU.cc Implementation of GPU version of MuellerPlatheFlow.

#ifdef ENABLE_CUDA
#include "MuellerPlatheFlowGPU.cuh"

MuellerPlatheFlowGPU::MuellerPlatheFlowGPU(std::shared_ptr<SystemDefinition> sysdef,
                                           std::shared_ptr<ParticleGroup> group,
                                           std::shared_ptr<Variant> flow_target,
                                           const flow_enum::Direction slab_direction,
                                           const flow_enum::Direction flow_direction,
                                           const unsigned int N_slabs,
                                           const unsigned int min_slab,
                                           const unsigned int max_slab)
:MuellerPlatheFlow(sysdef,group,flow_target,slab_direction,flow_direction,
                   N_slabs,min_slab,max_slab)
    {
    m_exec_conf->msg->notice(5) << "Constructing MuellerPlatheFlowGPU " << endl;
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a MuellerPlatheGPU with no GPU in the execution configuration" << endl;
        throw std::runtime_error("Error initializing MuellerPlatheFlowGPU");
        }

    m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "muellerplatheflow", this->m_exec_conf));
    }

MuellerPlatheFlowGPU::~MuellerPlatheFlowGPU(void)
    {
    m_exec_conf->msg->notice(5) << "Destroying MuellerPlatheFlowGPU " << endl;
    }



void MuellerPlatheFlowGPU::search_min_max_velocity(void)
    {
    const unsigned int group_size = m_group->getNumMembers();
    if(group_size == 0)
        return;
    if( !this->has_max_slab() and !this->has_min_slab())
        return;
    if(m_prof) m_prof->push("MuellerPlatheFlowGPU::search");
    const ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),access_location::device, access_mode::read);
    const ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),access_location::device, access_mode::read);
    const ArrayHandle<unsigned int> d_tag(m_pdata->getTags(),access_location::device, access_mode::read);
    const ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(),access_location::device, access_mode::read);
    const GlobalArray< unsigned int >& group_members = m_group->getIndexArray();
    const ArrayHandle<unsigned int> d_group_members(group_members, access_location::device, access_mode::read);

    const BoxDim& gl_box = m_pdata->getGlobalBox();



    m_tuner->begin();
    gpu_search_min_max_velocity(group_size,d_vel.data,d_pos.data,d_tag.data,d_rtag.data,
                                d_group_members.data,gl_box,this->get_N_slabs(),
                                this->get_max_slab(),this->get_min_slab(),&m_last_max_vel,
                                &m_last_min_vel,this->has_max_slab(),this->has_min_slab(),
                                m_tuner->getParam(),m_flow_direction,m_slab_direction);
    m_tuner->end();
    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if(m_prof) m_prof->pop();
    }

void MuellerPlatheFlowGPU::update_min_max_velocity(void)
    {
    if(m_prof) m_prof->push("MuellerPlatheFlowGPU::update");
    const ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_vel(m_pdata->getVelocities(),access_location::device,access_mode::readwrite);
    const unsigned int Ntotal = m_pdata->getN()+m_pdata->getNGhosts();

    gpu_update_min_max_velocity(d_rtag.data,d_vel.data,Ntotal,m_last_max_vel,
                                m_last_min_vel,m_flow_direction);

    if(m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if(m_prof) m_prof->pop();
    }



void export_MuellerPlatheFlowGPU(py::module&m)
    {
    py::class_< MuellerPlatheFlowGPU, std::shared_ptr<MuellerPlatheFlowGPU> >(m,"MuellerPlatheFlowGPU",py::base<MuellerPlatheFlow>())
        .def(py::init< std::shared_ptr<SystemDefinition>,std::shared_ptr<ParticleGroup>,
             std::shared_ptr<Variant>, const flow_enum::Direction,const flow_enum::Direction,
             const unsigned int, const unsigned int, const unsigned int >())
        ;
    }
#endif //ENABLE_CUDA
