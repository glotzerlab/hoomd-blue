/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

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
                                   boost::shared_ptr<Variant> Lz)
    : Updater(sysdef), m_Lx(Lx), m_Ly(Ly), m_Lz(Lz), m_scale_particles(true)
    {
    assert(m_pdata);
    assert(m_Lx);
    assert(m_Ly);
    assert(m_Lz);
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
    if (m_prof) m_prof->push("BoxResize");

    boost::shared_ptr<RigidData> rigid_data = m_sysdef->getRigidData();

    // first, compute what the current box size should be
    Scalar Lx = m_Lx->getValue(timestep);
    Scalar Ly = m_Ly->getValue(timestep);
    Scalar Lz = m_Lz->getValue(timestep);
    
    // check if the current box size is the same
    BoxDim curBox = m_pdata->getBox();
    BoxDim newBox(Lx, Ly, Lz);

    bool no_change = fabs((Lx - (curBox.xhi - curBox.xlo)) / Lx) < 1e-5 &&
                     fabs((Ly - (curBox.yhi - curBox.ylo)) / Ly) < 1e-5 &&
                     fabs((Lz - (curBox.zhi - curBox.zlo)) / Lz) < 1e-5;
                     
    // only change the box if there is a change in the box size
    if (!no_change)
        {
        // scale the particle positions (if we have been asked to)
        if (m_scale_particles)
            {
            Scalar sx = Lx / (curBox.xhi - curBox.xlo);
            Scalar sy = Ly / (curBox.yhi - curBox.ylo);
            Scalar sz = Lz / (curBox.zhi - curBox.zlo);
            
            // move the particles to be inside the new box
            ParticleDataArrays arrays = m_pdata->acquireReadWrite();
            
            for (unsigned int i = 0; i < arrays.nparticles; i++)
                {
                // intentionally scale both rigid body and free particles, this may waste a few cycles but it enables
                // the debug inBox checks to be left as is (otherwise, setRV cannot fixup rigid body positions without
                // failing the check)
                arrays.x[i] = (arrays.x[i] - curBox.xlo) * sx + newBox.xlo;
                arrays.y[i] = (arrays.y[i] - curBox.ylo) * sy + newBox.ylo;
                arrays.z[i] = (arrays.z[i] - curBox.zlo) * sz + newBox.zlo;
                }
                
            m_pdata->release();
            
            // also rescale rigid body COMs
            unsigned int n_bodies = rigid_data->getNumBodies();
            if (n_bodies > 0)
                {
                ArrayHandle<Scalar4> com_handle(rigid_data->getCOM(), access_location::host, access_mode::readwrite);
                
                for (unsigned int body = 0; body < n_bodies; body++)
                    {
                    com_handle.data[body].x = (com_handle.data[body].x - curBox.xlo) * sx + newBox.xlo;
                    com_handle.data[body].y = (com_handle.data[body].y - curBox.ylo) * sy + newBox.ylo;
                    com_handle.data[body].z = (com_handle.data[body].z - curBox.zlo) * sz + newBox.zlo;
                    }
                }
                
            }
        else if (Lx < (curBox.xhi - curBox.xlo) || Ly < (curBox.yhi - curBox.ylo) || Lz < (curBox.zhi - curBox.zlo))
            {
            // otherwise, we need to ensure that the particles are still in the box if it is smaller
            // move the particles to be inside the new box
            ParticleDataArrays arrays = m_pdata->acquireReadWrite();
            
            for (unsigned int i = 0; i < arrays.nparticles; i++)
                {
                // intentionally scale both rigid body and free particles, this may waste a few cycles but it enables
                // the debug inBox checks to be left as is (otherwise, setRV cannot fixup rigid body positions without
                // failing the check)
                
                // need to update the image if we move particles from one side of the box to the other
                float x_shift = rintf(arrays.x[i] / Lx);
                arrays.x[i] -= Lx * x_shift;
                arrays.ix[i] += (int)x_shift;

                float y_shift = rintf(arrays.y[i] / Ly);
                arrays.y[i] -= Ly * y_shift;
                arrays.iy[i] += (int)y_shift;
                
                float z_shift = rintf(arrays.z[i] / Lz);
                arrays.z[i] -= Lz * z_shift;
                arrays.iz[i] += (int)z_shift;
                }

            m_pdata->release();
            
            // do the same for rigid body COMs
            unsigned int n_bodies = rigid_data->getNumBodies();
            if (n_bodies > 0)
                {
                ArrayHandle<Scalar4> h_body_com(rigid_data->getCOM(), access_location::host, access_mode::readwrite);
                ArrayHandle<int3> h_body_image(rigid_data->getBodyImage(), access_location::host, access_mode::readwrite);
                
                for (unsigned int body = 0; body < n_bodies; body++)
                    {
                    // need to update the image if we move particles from one side of the box to the other
                    float x_shift = rintf(h_body_com.data[body].x / Lx);
                    h_body_com.data[body].x -= Lx * x_shift;
                    h_body_image.data[body].x += (int)x_shift;

                    float y_shift = rintf(h_body_com.data[body].y / Ly);
                    h_body_com.data[body].y -= Ly * y_shift;
                    h_body_image.data[body].y += (int)y_shift;
                    
                    float z_shift = rintf(h_body_com.data[body].z / Lz);
                    h_body_com.data[body].z -= Lz * z_shift;
                    h_body_image.data[body].z += (int)z_shift;
                    }
                }
            }
        
        // set the new box
        m_pdata->setBox(BoxDim(Lx, Ly, Lz));

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
     boost::shared_ptr<Variant> >())
    .def("setParams", &BoxResizeUpdater::setParams);
    }

#ifdef WIN32
#pragma warning( pop )
#endif

