// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander

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
/*! \param sysdef System definition containing the particle data to set the box size on
    \param Lx length of the x dimension over time
    \param Ly length of the y dimension over time
    \param Lz length of the z dimension over time

    The default setting is to scale particle positions along with the box.
*/

BoxResizeUpdater::BoxResizeUpdater(std::shared_ptr<SystemDefinition> sysdef,
                                   pybind11::object box1,
                                   pybind11::object box2,
                                   std::shared_ptr<Variant> variant,
                                   std::shared_ptr<ParticleGroup> group)
    : Updater(sysdef), m_py_box1(box1), m_py_box2(box2),
      m_box1(detail::getBoxDimFromPyObject(box1)), m_box2(detail::getBoxDimFromPyObject(box2)),
      m_variant(variant), m_group(group)
    {
    assert(m_pdata);
    assert(m_variant);
    m_exec_conf->msg->notice(5) << "Constructing BoxResizeUpdater" << endl;
    }

BoxResizeUpdater::~BoxResizeUpdater()
    {
    m_exec_conf->msg->notice(5) << "Destroying BoxResizeUpdater" << endl;
    }

/// Set a new initial box from a python object
void BoxResizeUpdater::setPyBox1(pybind11::object box1)
    {
    m_py_box1 = box1;
    m_box1 = detail::getBoxDimFromPyObject(box1);
    }

/// Set a new final box from a python object
void BoxResizeUpdater::setPyBox2(pybind11::object box2)
    {
    m_py_box2 = box2;
    m_box2 = detail::getBoxDimFromPyObject(box2);
    }

/// Get the current box based on the timestep
BoxDim BoxResizeUpdater::getCurrentBox(uint64_t timestep)
    {
    Scalar min = m_variant->min();
    Scalar max = m_variant->max();
    Scalar cur_value = (*m_variant)(timestep);
    Scalar scale = 0;
    if (cur_value == max)
        {
        scale = 1;
        }
    else if (cur_value > min)
        {
        scale = (cur_value - min) / (max - min);
        }

    Scalar3 new_L = m_box2.getL() * scale + m_box1.getL() * (1.0 - scale);
    Scalar xy = m_box2.getTiltFactorXY() * scale + (1.0 - scale) * m_box1.getTiltFactorXY();
    Scalar xz = m_box2.getTiltFactorXZ() * scale + (1.0 - scale) * m_box1.getTiltFactorXZ();
    Scalar yz = m_box2.getTiltFactorYZ() * scale + (1.0 - scale) * m_box1.getTiltFactorYZ();

    BoxDim new_box = BoxDim(new_L);
    new_box.setTiltFactors(xy, xz, yz);
    return new_box;
    }

/** Perform the needed calculations to scale the box size
    \param timestep Current time step of the simulation
*/
void BoxResizeUpdater::update(uint64_t timestep)
    {
    Updater::update(timestep);
    m_exec_conf->msg->notice(10) << "Box resize update" << endl;
    if (m_prof)
        m_prof->push("BoxResize");

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
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::readwrite);

        for (unsigned int group_idx = 0; group_idx < m_group->getNumMembers(); group_idx++)
            {
            unsigned int j = m_group->getMemberIndex(group_idx);
            // obtain scaled coordinates in the old global box
            Scalar3 fractional_pos = cur_box.makeFraction(
                make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z));

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
        ArrayHandle<int3> h_image(m_pdata->getImages(),
                                  access_location::host,
                                  access_mode::readwrite);

        const BoxDim& local_box = m_pdata->getBox();

        for (unsigned int i = 0; i < m_pdata->getN(); i++)
            {
            // need to update the image if we move particles from one side
            // of the box to the other
            local_box.wrap(h_pos.data[i], h_image.data[i]);
            }
        }
    if (m_prof)
        m_prof->pop();
    }

namespace detail
    {
BoxDim& getBoxDimFromPyObject(pybind11::object box)
    {
    return box.attr("_cpp_obj").cast<BoxDim&>();
    }

void export_BoxResizeUpdater(pybind11::module& m)
    {
    pybind11::class_<BoxResizeUpdater, Updater, std::shared_ptr<BoxResizeUpdater>>(
        m,
        "BoxResizeUpdater")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            pybind11::object,
                            pybind11::object,
                            std::shared_ptr<Variant>,
                            std::shared_ptr<ParticleGroup>>())
        .def_property("box1", &BoxResizeUpdater::getPyBox1, &BoxResizeUpdater::setPyBox1)
        .def_property("box2", &BoxResizeUpdater::getPyBox2, &BoxResizeUpdater::setPyBox2)
        .def_property("variant", &BoxResizeUpdater::getVariant, &BoxResizeUpdater::setVariant)
        .def_property_readonly("filter",
                               [](const std::shared_ptr<BoxResizeUpdater> method)
                               { return method->getGroup()->getFilter(); })
        .def("get_current_box", &BoxResizeUpdater::getCurrentBox);
    }

    } // end namespace detail

    } // end namespace hoomd
