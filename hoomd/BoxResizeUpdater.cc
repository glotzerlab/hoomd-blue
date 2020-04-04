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
                                   BoxDim box1,
                                   BoxDim box2,
                                   std::shared_ptr<Variant> variant)
    : Updater(sysdef), m_box1(box1), m_py_box1(pybind11::none()), m_box2(box2),
      m_py_box2(pybind11::none()), m_variant(variant), m_scale_particles(true)
    {
    m_exec_conf->msg->notice(5) << "Constructing BoxResizeUpdater" << endl;
    }

BoxResizeUpdater::BoxResizeUpdater(std::shared_ptr<SystemDefinition> sysdef,
                                   pybind11::object box1,
                                   pybind11::object box2,
                                   std::shared_ptr<Variant> variant)
    : Updater(sysdef), m_box1(getBoxDimFromPyObject(box1)), m_py_box1(box1),
      m_box2(getBoxDimFromPyObject(box2)), m_py_box2(box2),
      m_variant(variant), m_scale_particles(true)
    {
    m_exec_conf->msg->notice(5) << "Constructing BoxResizeUpdater" << endl;
    }

BoxResizeUpdater::~BoxResizeUpdater()
    {
    m_exec_conf->msg->notice(5) << "Destroying BoxResizeUpdater" << endl;
    }

void BoxResizeUpdater::setBox1(BoxDim box)
    {
    if (!m_py_box1.is_none())
        {
        m_py_box1 = m_py_box1.attr("__class__")
                            .attr("_from_cpp")(box);
        }
    m_box1 = box;
    }

void BoxResizeUpdater::setBox1Py(pybind11::object box)
    {
    m_box1 = getBoxDimFromPyObject(box);
    m_py_box1 = box;
    }

void BoxResizeUpdater::setBox2(BoxDim box)
    {
    if (!m_py_box2.is_none())
        {
        m_py_box2 = m_py_box2.attr("__class__")
                            .attr("_from_cpp")(box);
        }
    m_box2 = box;
    }

void BoxResizeUpdater::setBox2Py(pybind11::object box)
    {
    m_box2 = getBoxDimFromPyObject(box);
    m_py_box2 = box;
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

    Scalar3 L1 = m_box1.getL();
    Scalar3 L2 = m_box2.getL();
    Scalar Lx = L2.x * scale + (1.0 - scale) * L1.x;
    Scalar Ly = L2.y * scale + (1.0 - scale) * L1.y;
    Scalar Lz = L2.z * scale + (1.0 - scale) * L1.z;
    Scalar xy = m_box2.getTiltFactorXY() * scale +
                (1.0 - scale) * m_box1.getTiltFactorXY();
    Scalar xz = m_box2.getTiltFactorXZ() * scale +
                (1.0 - scale) * m_box1.getTiltFactorXZ();
    Scalar yz = m_box2.getTiltFactorYZ() * scale +
                (1.0 - scale) * m_box1.getTiltFactorYZ();

    BoxDim new_box = BoxDim(make_scalar3(Lx, Ly, Lz));
    new_box.setTiltFactors(xy, xz, yz);
    return new_box;
    }

/// Return whether two boxes are equivalent upto 1e-7
bool BoxResizeUpdater::boxesAreEquivalent(BoxDim& box1, BoxDim& box2)
    {
    Scalar3 L1 = box1.getL();
    Scalar3 L2 = box2.getL();

    Scalar xy1 = box1.getTiltFactorXY();
    Scalar xy2 = box2.getTiltFactorXY();
    Scalar xz1 = box1.getTiltFactorXZ();
    Scalar xz2 = box2.getTiltFactorXZ();
    Scalar yz1 = box1.getTiltFactorYZ();
    Scalar yz2 = box2.getTiltFactorYZ();

    return L1.x == L2.x && L1.y == L2.y && L1.z == L2.z &&
           fabs((xy1 - xy2) / xy1) < 1e-7 &&
           fabs((xz1 - xz2) / xz1) < 1e-7 &&
           fabs((yz1 - yz2) / yz1) < 1e-7;
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
    if (!boxesAreEquivalent(new_box, cur_box))
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

BoxDim getBoxDimFromPyObject(pybind11::object box)
    {
    auto type_name = box.get_type().attr("__name__").cast<std::string>();
    if (type_name != "Box")
        {
        std::string err = "Expected type of Box. Received type ";
        throw std::runtime_error(err + type_name);
        }
    return box.attr("_cpp_obj").cast<BoxDim>();
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
    .def_property("box1",
                  &BoxResizeUpdater::getBox1Py,
                  &BoxResizeUpdater::setBox1Py)
    .def_property("box2",
                  &BoxResizeUpdater::getBox2Py,
                  &BoxResizeUpdater::setBox2Py)
    .def_property("variant",
                  &BoxResizeUpdater::getVariant,
                  &BoxResizeUpdater::setVariant)
    .def("get_current_box", &BoxResizeUpdater::getCurrentBox)
    ;
    }
