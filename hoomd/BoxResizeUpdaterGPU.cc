// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file BoxResizeUpdater.cc
    \brief Defines the BoxResizeUpdater class
*/

#include "BoxResizeUpdaterGPU.h"
#include "BoxResizeUpdaterGPU.cuh"

namespace hoomd
    {
/*! \param sysdef System definition containing the particle data to set the box size on
    \param Lx length of the x dimension over time
    \param Ly length of the y dimension over time
    \param Lz length of the z dimension over time

    The default setting is to scale particle positions along with the box.
*/

BoxResizeUpdaterGPU::BoxResizeUpdaterGPU(std::shared_ptr<SystemDefinition> sysdef,
                                         std::shared_ptr<Trigger> trigger,
                                         std::shared_ptr<BoxDim> box1,
                                         std::shared_ptr<BoxDim> box2,
                                         std::shared_ptr<Variant> variant,
                                         std::shared_ptr<ParticleGroup> group)
    : BoxResizeUpdater(sysdef, trigger, box1, box2, variant, group)
    {
    }

BoxResizeUpdaterGPU::~BoxResizeUpdaterGPU()
    {
    m_exec_conf->msg->notice(5) << "Destroying BoxResizeUpdater" << std::endl;
    }

/** Perform the needed calculations to scale the box size
    \param timestep Current time step of the simulation
*/
void BoxResizeUpdaterGPU::update(uint64_t timestep)
    {
    Updater::update(timestep);
    m_exec_conf->msg->notice(10) << "Box resize update" << std::endl;

    // first, compute the new box
    BoxDim new_box = getCurrentBox(timestep);

    // check if the current box size is the same
    BoxDim cur_box = m_pdata->getGlobalBox();

    // only change the box if there is a change in the box dimensions
    if (new_box != cur_box)
        {
        // set the new box
        m_pdata->setGlobalBox(new_box);

        // scale the particle positions (if we have been asked to)
        // move the particles to be inside the new box
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

        kernel::gpu_box_resize_updater(m_pdata->getN(),
                                       d_pos.data,
                                       cur_box,
                                       new_box,
                                       d_group_members.data,
                                       group_size,
                                       d_image.data);
        }
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
                            std::shared_ptr<BoxDim>,
                            std::shared_ptr<BoxDim>,
                            std::shared_ptr<Variant>,
                            std::shared_ptr<ParticleGroup>>());
    }

    } // end namespace detail
    } // end namespace hoomd
