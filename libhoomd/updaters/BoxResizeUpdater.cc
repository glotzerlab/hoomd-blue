/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
the University of Michigan All rights reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
// Maintainer: joaander

/*! \file BoxResizeUpdater.cc
    \brief Defines the BoxResizeUpdater class
*/

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "BoxResizeUpdater.h"
#include "RigidData.h"

#include <math.h>
#include <iostream>
#include <stdexcept>

using namespace std;

/*! \param sysdef System definition containing the particle data to set the box size on
    \param Lx length of the x dimension over time
    \param Ly length of the y dimension over time
    \param Lz length of the z dimension over time

    The default setting is to scale particle positions along with the box.
*/
BoxResizeUpdater::BoxResizeUpdater(boost::shared_ptr<SystemDefinition> sysdef,
                                   boost::shared_ptr<Variant> Lx,
                                   boost::shared_ptr<Variant> Ly,
                                   boost::shared_ptr<Variant> Lz,
                                   boost::shared_ptr<Variant> xy,
                                   boost::shared_ptr<Variant> xz,
                                   boost::shared_ptr<Variant> yz)
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

    boost::shared_ptr<RigidData> rigid_data = m_sysdef->getRigidData();

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

            // also rescale rigid body COMs
            unsigned int n_bodies = rigid_data->getNumBodies();
            if (n_bodies > 0)
                {
                ArrayHandle<Scalar4> com_handle(rigid_data->getCOM(), access_location::host, access_mode::readwrite);

                for (unsigned int body = 0; body < n_bodies; body++)
                    {
                    // obtain scaled coordinates in the old global box
                    Scalar3 f = curBox.makeFraction(make_scalar3(com_handle.data[body].x,
                                                                 com_handle.data[body].y,
                                                                 com_handle.data[body].z));
                    Scalar3 scaled_cm = newBox.makeCoordinates(f);
                    com_handle.data[body].x = scaled_cm.x;
                    com_handle.data[body].y = scaled_cm.y;
                    com_handle.data[body].z = scaled_cm.z;
                    }
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
                // intentionally scale both rigid body and free particles, this may waste a few cycles but it enables
                // the debug inBox checks to be left as is (otherwise, setRV cannot fixup rigid body positions without
                // failing the check)

                // need to update the image if we move particles from one side of the box to the other
                local_box.wrap(h_pos.data[i], h_image.data[i]);
                }

            // do the same for rigid body COMs
            unsigned int n_bodies = rigid_data->getNumBodies();
            if (n_bodies > 0)
                {
                ArrayHandle<Scalar4> h_body_com(rigid_data->getCOM(), access_location::host, access_mode::readwrite);
                ArrayHandle<int3> h_body_image(rigid_data->getBodyImage(), access_location::host, access_mode::readwrite);

                for (unsigned int body = 0; body < n_bodies; body++)
                    {
                    // need to update the image if we move particles from one side of the box to the other
                    local_box.wrap(h_body_com.data[body], h_body_image.data[body]);
                    }
                }
            }

        // update the body particle positions to reflect the new rigid body positions
        rigid_data->setRV(true);
        }

    if (m_prof) m_prof->pop();
    }

void export_BoxResizeUpdater()
    {
    class_<BoxResizeUpdater, boost::shared_ptr<BoxResizeUpdater>, bases<Updater>, boost::noncopyable>
    ("BoxResizeUpdater", init< boost::shared_ptr<SystemDefinition>,
     boost::shared_ptr<Variant>,
     boost::shared_ptr<Variant>,
     boost::shared_ptr<Variant>,
     boost::shared_ptr<Variant>,
     boost::shared_ptr<Variant>,
     boost::shared_ptr<Variant> >())
    .def("setParams", &BoxResizeUpdater::setParams);
    }

#ifdef WIN32
#pragma warning( pop )
#endif
