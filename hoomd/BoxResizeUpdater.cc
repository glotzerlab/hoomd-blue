// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander

/*! \file BoxResizeUpdater.cc
    \brief Defines the BoxResizeUpdater class
*/

#include "BoxResizeUpdater.h"

#include <math.h>
#include <iostream>
#include <stdexcept>

using namespace std;
namespace py = pybind11;

/*! \param sysdef System definition containing the particle data to set the box size on
    \param Lx length of the x dimension over time
    \param Ly length of the y dimension over time
    \param Lz length of the z dimension over time

    The default setting is to scale particle positions along with the box.
*/

BoxResizeUpdater::BoxResizeUpdater(std::shared_ptr<SystemDefinition> sysdef,
                                   pybind11::object initial_box,
                                   pybind11::object final_box,
                                   std::shared_ptr<Variant> variant)
    : Updater(sysdef),
      m_py_initial_box(initial_box),
      m_py_final_box(final_box),
      m_initial_box(getBoxDimFromPyObject(initial_box)),
      m_final_box(getBoxDimFromPyObject(final_box)),
      m_variant(variant), m_scale_particles(true)
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
void BoxResizeUpdater::setPyInitialBox(pybind11::object initial_box)
    {
    m_py_initial_box = initial_box;
    m_initial_box = getBoxDimFromPyObject(initial_box);
    }

/// Set a new final box from a python object
void BoxResizeUpdater::setPyFinalBox(pybind11::object final_box)
    {
    m_py_final_box = final_box;
    m_final_box = getBoxDimFromPyObject(final_box);
    }

/// Get the current box based on the timestep
BoxDim BoxResizeUpdater::getCurrentBox(unsigned int timestep)
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

Scalar3 new_L = m_final_box.getL() * scale +
            m_initial_box.getL() * (1.0 - scale);
Scalar xy = m_final_box.getTiltFactorXY() * scale +
        (1.0 - scale) * m_initial_box.getTiltFactorXY();
Scalar xz = m_final_box.getTiltFactorXZ() * scale +
        (1.0 - scale) * m_initial_box.getTiltFactorXZ();
Scalar yz = m_final_box.getTiltFactorYZ() * scale +
        (1.0 - scale) * m_initial_box.getTiltFactorYZ();

BoxDim new_box = BoxDim(new_L);
new_box.setTiltFactors(xy, xz, yz);
return new_box;
}

/** Perform the needed calculations to scale the box size
    \param timestep Current time step of the simulation
*/
void BoxResizeUpdater::update(unsigned int timestep)
    {
    m_exec_conf->msg->notice(10) << "Box resize update" << endl;
    if (m_prof) m_prof->push("BoxResize");

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
        if (m_scale_particles)
            {
            // move the particles to be inside the new box
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                                       access_location::host,
                                       access_mode::readwrite);

            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                {
                // obtain scaled coordinates in the old global box
                Scalar3 fractional_pos = cur_box.makeFraction(
                    make_scalar3(h_pos.data[i].x,
                                 h_pos.data[i].y,
                                 h_pos.data[i].z));

                // intentionally scale both rigid body and free particles, this
                // may waste a few cycles but it enables the debug inBox checks
                // to be left as is (otherwise, setRV cannot fixup rigid body
                // positions without failing the check)
                Scalar3 scaled_pos = new_box.makeCoordinates(fractional_pos);
                h_pos.data[i].x = scaled_pos.x;
                h_pos.data[i].y = scaled_pos.y;
                h_pos.data[i].z = scaled_pos.z;
                }
            }
        // otherwise, we need to ensure that the particles are still in their
        // local boxes by wrapping them if they are not
        else
            {
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                                       access_location::host,
                                       access_mode::readwrite);

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
        }
    if (m_prof) m_prof->pop();
    }

BoxDim& getBoxDimFromPyObject(pybind11::object box)
    {
    return box.attr("_cpp_obj").cast<BoxDim&>();
    }

void export_BoxResizeUpdater(py::module& m)
    {
    py::class_<BoxResizeUpdater, Updater,
               std::shared_ptr<BoxResizeUpdater> >(m,"BoxResizeUpdater")
    .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                        pybind11::object, pybind11::object,
                        std::shared_ptr<Variant> >())
    .def_property("scale_particles",
                  &BoxResizeUpdater::getScaleParticles,
                  &BoxResizeUpdater::setScaleParticles)
    .def_property("initial_box",
                  &BoxResizeUpdater::getPyInitialBox,
                  &BoxResizeUpdater::setPyInitialBox)
    .def_property("final_box",
                  &BoxResizeUpdater::getPyFinalBox,
                  &BoxResizeUpdater::setPyFinalBox)
    .def_property("variant",
                  &BoxResizeUpdater::getVariant,
                  &BoxResizeUpdater::setVariant)
    .def("get_current_box", &BoxResizeUpdater::getCurrentBox)
    ;
    }
