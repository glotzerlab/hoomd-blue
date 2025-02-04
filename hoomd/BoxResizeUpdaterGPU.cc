// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file BoxResizeUpdater.cc
    \brief Defines the BoxResizeUpdater class
*/

#include "BoxResizeUpdaterGPU.h"
#include "BoxResizeUpdaterGPU.cuh"

namespace hoomd
    {
BoxResizeUpdaterGPU::BoxResizeUpdaterGPU(std::shared_ptr<SystemDefinition> sysdef,
                                         std::shared_ptr<Trigger> trigger,
                                         std::shared_ptr<VectorVariantBox> box,
                                         std::shared_ptr<ParticleGroup> group)
    : BoxResizeUpdater(sysdef, trigger, box, group)
    {
    // only one GPU is supported
    if (!m_exec_conf->isCUDAEnabled())
        {
        throw std::runtime_error("Cannot initialize BoxResizeUpdaterGPU on a CPU device.");
        }

    m_tuner_scale.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                         m_exec_conf,
                                         "box_resize_scale"));
    m_tuner_wrap.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                        m_exec_conf,
                                        "box_resize_wrap"));
    }

BoxResizeUpdaterGPU::~BoxResizeUpdaterGPU()
    {
    m_exec_conf->msg->notice(5) << "Destroying BoxResizeUpdater" << std::endl;
    }

/// Scale particles to the new box and wrap any others back into the box
void BoxResizeUpdaterGPU::scaleAndWrapParticles(const BoxDim& cur_box, const BoxDim& new_box)
    {
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(),
                               access_location::device,
                               access_mode::readwrite);

    ArrayHandle<int3> d_image(m_pdata->getImages(),
                              access_location::device,
                              access_mode::readwrite);

    unsigned int group_size = m_group->getNumMembers();
    ArrayHandle<unsigned int> d_group_members(m_group->getIndexArray(),
                                              access_location::device,
                                              access_mode::read);
    m_tuner_scale->begin();
    kernel::gpu_box_resize_scale(d_pos.data,
                                 cur_box,
                                 new_box,
                                 d_group_members.data,
                                 group_size,
                                 m_tuner_scale->getParam()[0]);
    m_tuner_scale->end();

    m_tuner_wrap->begin();
    kernel::gpu_box_resize_wrap(m_pdata->getN(),
                                d_pos.data,
                                d_image.data,
                                new_box,
                                m_tuner_wrap->getParam()[0]);
    m_tuner_wrap->end();
    }

namespace detail
    {
void export_BoxResizeUpdaterGPU(pybind11::module& m)
    {
    pybind11::class_<BoxResizeUpdaterGPU, BoxResizeUpdater, std::shared_ptr<BoxResizeUpdaterGPU>>(
        m,
        "BoxResizeUpdaterGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Trigger>,
                            std::shared_ptr<VectorVariantBox>,
                            std::shared_ptr<ParticleGroup>>());
    }

    } // end namespace detail
    } // end namespace hoomd
