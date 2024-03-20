// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file BoxResizeUpdater.cc
    \brief Defines the BoxResizeUpdater class
*/

#include "BoxResizeUpdater.h"

#include <iostream>
#include <math.h>
#include <stdexcept>

using namespace std;

namespace hoomd
    {
/** @param sysdef System definition containing the particle data to set the box size on
    @param trigger Steps on which to execute.
    @param box Box as a function of time.
    @param group Particles to scale.
*/
BoxResizeUpdater::BoxResizeUpdater(std::shared_ptr<SystemDefinition> sysdef,
                                   std::shared_ptr<Trigger> trigger,
                                   std::shared_ptr<VectorVariantBox> box,
                                   std::shared_ptr<ParticleGroup> group)
    : Updater(sysdef, trigger), m_box(box), m_group(group)
    {
    assert(m_pdata);
    m_exec_conf->msg->notice(5) << "Constructing BoxResizeUpdater" << endl;
    }

BoxResizeUpdater::~BoxResizeUpdater()
    {
    m_exec_conf->msg->notice(5) << "Destroying BoxResizeUpdater" << endl;
    }

BoxDim BoxResizeUpdater::getCurrentBox(uint64_t timestep)
    {
    return BoxDim((*m_box)(timestep));
    }

/** @param timestep Current time step of the simulation
 */
void BoxResizeUpdater::update(uint64_t timestep)
    {
    Updater::update(timestep);
    m_exec_conf->msg->notice(10) << "Box resize update" << endl;

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
        scaleAndWrapParticles(cur_box, new_box);

        // scale the origin
        Scalar3 old_origin = m_pdata->getOrigin();
        Scalar3 fractional_old_origin = cur_box.makeFraction(old_origin);
        Scalar3 new_origin = new_box.makeCoordinates(fractional_old_origin);
        m_pdata->translateOrigin(new_origin - old_origin);
        }
    }

void BoxResizeUpdater::scaleAndWrapParticles(const BoxDim& cur_box, const BoxDim& new_box)
    {
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                               access_location::host,
                               access_mode::readwrite);

    for (unsigned int group_idx = 0; group_idx < m_group->getNumMembers(); group_idx++)
        {
        unsigned int j = m_group->getMemberIndex(group_idx);
        // obtain scaled coordinates in the old global box
        Scalar3 fractional_pos
            = cur_box.makeFraction(make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z));

        // intentionally scale both rigid body and free particles, this
        // may waste a few cycles but it enables the debug inBox checks
        // to be left as is (otherwise, setRV cannot fixup rigid body
        // positions without failing the check)
        Scalar3 scaled_pos = new_box.makeCoordinates(fractional_pos);
        h_pos.data[j].x = scaled_pos.x;
        h_pos.data[j].y = scaled_pos.y;
        h_pos.data[j].z = scaled_pos.z;
        }

    // ensure that the particles are still in their
    // local boxes by wrapping them if they are not
    ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);

    const BoxDim& local_box = m_pdata->getBox();

    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        // need to update the image if we move particles from one side
        // of the box to the other
        local_box.wrap(h_pos.data[i], h_image.data[i]);
        }
    }

namespace detail
    {
void export_BoxResizeUpdater(pybind11::module& m)
    {
    pybind11::class_<BoxResizeUpdater, Updater, std::shared_ptr<BoxResizeUpdater>>(
        m,
        "BoxResizeUpdater")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<Trigger>,
                            std::shared_ptr<VectorVariantBox>,
                            std::shared_ptr<ParticleGroup>>())
        .def_property("box", &BoxResizeUpdater::getBox, &BoxResizeUpdater::setBox)
        .def_property_readonly("filter",
                               [](const std::shared_ptr<BoxResizeUpdater> method)
                               { return method->getGroup()->getFilter(); })
        .def("get_current_box", &BoxResizeUpdater::getCurrentBox);
    }

    } // end namespace detail

    } // end namespace hoomd
