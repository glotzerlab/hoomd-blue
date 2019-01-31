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
                                   std::shared_ptr<Variant> Lx,
                                   std::shared_ptr<Variant> Ly,
                                   std::shared_ptr<Variant> Lz,
                                   std::shared_ptr<Variant> xy,
                                   std::shared_ptr<Variant> xz,
                                   std::shared_ptr<Variant> yz)
    : Updater(sysdef), m_Lx(Lx), m_Ly(Ly), m_Lz(Lz), m_xy(xy), m_xz(xz), m_yz(yz), m_scale_particles(true)
    {
    assert(m_pdata);
    assert(m_Lx);
    assert(m_Ly);
    assert(m_Lz);

    m_exec_conf->msg->notice(5) << "Constructing BoxResizeUpdater" << endl;
    }

BoxResizeUpdater::~BoxResizeUpdater()
    {
    m_exec_conf->msg->notice(5) << "Destroying BoxResizeUpdater" << endl;
    }

/*! \param scale_particles Set to true to scale particles with the box. Set to false to leave particle positions alone
    when scaling the box.
*/
void BoxResizeUpdater::setParams(bool scale_particles)
    {
    m_scale_particles = scale_particles;
    }

/*! Perform the needed calculations to scale the box size
    \param timestep Current time step of the simulation
*/
void BoxResizeUpdater::update(unsigned int timestep)
    {
    m_exec_conf->msg->notice(10) << "Box resize update" << endl;
    if (m_prof) m_prof->push("BoxResize");

    // first, compute what the current box size and tilt factors should be
    Scalar Lx = m_Lx->getValue(timestep);
    Scalar Ly = m_Ly->getValue(timestep);
    Scalar Lz = m_Lz->getValue(timestep);
    Scalar xy = m_xy->getValue(timestep);
    Scalar xz = m_xz->getValue(timestep);
    Scalar yz = m_yz->getValue(timestep);

    // check if the current box size is the same
    BoxDim curBox = m_pdata->getGlobalBox();
    Scalar3 curL = curBox.getL();
    Scalar curxy = curBox.getTiltFactorXY();
    Scalar curxz = curBox.getTiltFactorXZ();
    Scalar curyz = curBox.getTiltFactorYZ();

    // copy and setL + setTiltFactors instead of creating a new box
    BoxDim newBox = curBox;
    newBox.setL(make_scalar3(Lx, Ly, Lz));
    newBox.setTiltFactors(xy,xz,yz);

    bool no_change = fabs((Lx - curL.x) / Lx) < 1e-7 &&
                     fabs((Ly - curL.y) / Ly) < 1e-7 &&
                     fabs((Lz - curL.z) / Lz) < 1e-7 &&
                     fabs((xy - curxy) / xy) < 1e-7 &&
                     fabs((xz - curxz) / xz) < 1e-7 &&
                     fabs((yz - curyz) / yz) < 1e-7;

    // only change the box if there is a change in the box dimensions
    if (!no_change)
        {
        // set the new box
        m_pdata->setGlobalBox(newBox);

        // scale the particle positions (if we have been asked to)
        if (m_scale_particles)
            {
            // move the particles to be inside the new box
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);

            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                {
                // obtain scaled coordinates in the old global box
                Scalar3 f = curBox.makeFraction(make_scalar3(h_pos.data[i].x, h_pos.data[i].y, h_pos.data[i].z));

                // intentionally scale both rigid body and free particles, this may waste a few cycles but it enables
                // the debug inBox checks to be left as is (otherwise, setRV cannot fixup rigid body positions without
                // failing the check)
                Scalar3 scaled_pos = newBox.makeCoordinates(f);
                h_pos.data[i].x = scaled_pos.x;
                h_pos.data[i].y = scaled_pos.y;
                h_pos.data[i].z = scaled_pos.z;
                }
            }
        else
            {
            // otherwise, we need to ensure that the particles are still in the (local) box
            // move the particles to be inside the new box
            ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);
            ArrayHandle<int3> h_image(m_pdata->getImages(), access_location::host, access_mode::readwrite);

            const BoxDim& local_box = m_pdata->getBox();

            for (unsigned int i = 0; i < m_pdata->getN(); i++)
                {
                // need to update the image if we move particles from one side of the box to the other
                local_box.wrap(h_pos.data[i], h_image.data[i]);
                }
            }

        }

    if (m_prof) m_prof->pop();
    }

void export_BoxResizeUpdater(py::module& m)
    {
    py::class_<BoxResizeUpdater, std::shared_ptr<BoxResizeUpdater> >(m,"BoxResizeUpdater",py::base<Updater>())
    .def(py::init< std::shared_ptr<SystemDefinition>,
     std::shared_ptr<Variant>,
     std::shared_ptr<Variant>,
     std::shared_ptr<Variant>,
     std::shared_ptr<Variant>,
     std::shared_ptr<Variant>,
     std::shared_ptr<Variant> >())
    .def("setParams", &BoxResizeUpdater::setParams);
    }
