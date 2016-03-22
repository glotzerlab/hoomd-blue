/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2016 The Regents of
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

/*! \file ZeroMomentumUpdater.cc
    \brief Defines the ZeroMomentumUpdater class
*/


#include "ZeroMomentumUpdater.h"

#include <boost/python.hpp>
using namespace boost::python;

#include <iostream>
#include <math.h>
#include <stdexcept>

using namespace std;

/*! \param sysdef System to zero the momentum of
*/
ZeroMomentumUpdater::ZeroMomentumUpdater(boost::shared_ptr<SystemDefinition> sysdef)
        : Updater(sysdef)
    {
    m_exec_conf->msg->notice(5) << "Constructing ZeroMomentumUpdater" << endl;
    assert(m_pdata);
    }

ZeroMomentumUpdater::~ZeroMomentumUpdater()
    {
    m_exec_conf->msg->notice(5) << "Destroyinging ZeroMomentumUpdater" << endl;
    }

/*! Perform the needed calculations to zero the system's momentum
    \param timestep Current time step of the simulation
*/
void ZeroMomentumUpdater::update(unsigned int timestep)
    {
    if (m_prof) m_prof->push("ZeroMomentum");

    // calculate the average momentum
    assert(m_pdata);

    {
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::readwrite);
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::readwrite);

    // temp variables for holding the sums
    Scalar sum_px = 0.0;
    Scalar sum_py = 0.0;
    Scalar sum_pz = 0.0;
    unsigned int n = 0;

    // add up the momentum of every free particle
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        if (h_body.data[i] == NO_BODY)
            {
            Scalar mass = h_vel.data[i].w;
            sum_px += mass*h_vel.data[i].x;
            sum_py += mass*h_vel.data[i].y;
            sum_pz += mass*h_vel.data[i].z;
            n++;
            }
        }

    #ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        MPI_Allreduce(MPI_IN_PLACE, &n, 1, MPI_INT, MPI_SUM, m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE, &sum_px, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE, &sum_py, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
        MPI_Allreduce(MPI_IN_PLACE, &sum_pz, 1, MPI_HOOMD_SCALAR, MPI_SUM, m_exec_conf->getMPICommunicator());
        }
    #endif

    // calculate the average
    Scalar avg_px = sum_px / Scalar(n);
    Scalar avg_py = sum_py / Scalar(n);
    Scalar avg_pz = sum_pz / Scalar(n);

    // subtract this momentum from every free partcile
    for (unsigned int i = 0; i < m_pdata->getN(); i++)
        {
        if (h_body.data[i] == NO_BODY)
            {
            Scalar mass = h_vel.data[i].w;
            h_vel.data[i].x -= avg_px/mass;
            h_vel.data[i].y -= avg_py/mass;
            h_vel.data[i].z -= avg_pz/mass;
            }
        }
    } // end GPUArray scope

    if (m_prof) m_prof->pop();
    }

void export_ZeroMomentumUpdater()
    {
    class_<ZeroMomentumUpdater, boost::shared_ptr<ZeroMomentumUpdater>, bases<Updater>, boost::noncopyable>
    ("ZeroMomentumUpdater", init< boost::shared_ptr<SystemDefinition> >())
    ;
    }
