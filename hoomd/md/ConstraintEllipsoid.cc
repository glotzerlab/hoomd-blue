/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2015 The Regents of
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


#include "ConstraintEllipsoid.h"
#include "EvaluatorConstraintEllipsoid.h"

#include <boost/python.hpp>
using namespace boost::python;

using namespace std;

/*! \file ConstraintEllipsoid.cc
    \brief Contains code for the ConstraintEllipsoid class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
    \param group Group of particles on which to apply this constraint
    \param P position of the Ellipsoid
    \param rx radius of the Ellipsoid in the X direction
    \param ry radius of the Ellipsoid in the Y direction
    \param rz radius of the Ellipsoid in the Z direction
    NOTE: For the algorithm to work, we must have _rx >= _rz, ry >= _rz, and _rz > 0.
*/
ConstraintEllipsoid::ConstraintEllipsoid(boost::shared_ptr<SystemDefinition> sysdef,
                                   boost::shared_ptr<ParticleGroup> group,
                                   Scalar3 P,
                                   Scalar rx,
                                   Scalar ry,
                                   Scalar rz)
        : Updater(sysdef), m_group(group), m_P(P), m_rx(rx), m_ry(ry), m_rz(rz)
    {
    m_exec_conf->msg->notice(5) << "Constructing ConstraintEllipsoid" << endl;

    validate();
    }

ConstraintEllipsoid::~ConstraintEllipsoid()
    {
    m_exec_conf->msg->notice(5) << "Destroying ConstraintEllipsoid" << endl;
    }

/*! Computes the specified constraint forces
    \param timestep Current timestep
*/
void ConstraintEllipsoid::update(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;

    if (m_prof) m_prof->push("ConstraintEllipsoid");

    assert(m_pdata);

    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::readwrite);

    EvaluatorConstraintEllipsoid Ellipsoid(m_P, m_rx, m_ry, m_rz);
    // for each of the particles in the group
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        // get the current particle properties
        unsigned int j = m_group->getMemberIndex(group_idx);
        Scalar3 X = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);

        // evaluate the constraint position
        Scalar3 C = Ellipsoid.evalClosest(X);

        // apply the constraint
        h_pos.data[j].x = C.x;
        h_pos.data[j].y = C.y;
        h_pos.data[j].z = C.z;
        }

    if (m_prof)
        m_prof->pop();
    }

/*! Print warning messages if the Ellipsoid is outside the box.
    Generate an error if any particle in the group is not near the Ellipsoid.
*/
void ConstraintEllipsoid::validate()
    {
    BoxDim box = m_pdata->getBox();
    Scalar3 lo = box.getLo();
    Scalar3 hi = box.getHi();

    if (m_P.x + m_rx > hi.x || m_P.x - m_rx < lo.x
        || m_P.y + m_ry > hi.y || m_P.y - m_ry < lo.y
        || m_P.z + m_rz > hi.z || m_P.z - m_rz < lo.z)
        {
        m_exec_conf->msg->warning() << "constrain.ellipsoid: ellipsoid constraint is outside of the box. Constrained particle positions may be incorrect"
             << endl;
        }

    if (m_rx == 0 || m_ry == 0 || m_rz == 0)
        {
        m_exec_conf->msg->warning() << "constrain.ellipsoid: one of the ellipsoid dimensions is 0. Constraint may be incorrect."
             << endl;
        }

    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::read);

    EvaluatorConstraintEllipsoid Ellipsoid(m_P, m_rx, m_ry, m_rz);
    // for each of the particles in the group
    bool errors = false;
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        // get the current particle properties
        unsigned int j = m_group->getMemberIndex(group_idx);

        Scalar3 X = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
        Scalar3 C = Ellipsoid.evalClosest(X);
        Scalar3 V;
        V.x = C.x - X.x;
        V.y = C.y - X.y;
        V.z = C.z - X.z;
        Scalar dist = slow::sqrt(V.x*V.x + V.y*V.y + V.z*V.z);

        if (dist > Scalar(1.0))
            {
            m_exec_conf->msg->error() << "constrain.ellipsoid: Particle " << h_tag.data[j] << " is more than 1 unit of"
                                      << " distance away from the closest point on the ellipsoid constraint" << endl;
            errors = true;
            }

        if (h_body.data[j] != NO_BODY)
            {
            m_exec_conf->msg->error() << "constrain.ellipsoid: Particle " << h_tag.data[j] << " belongs to a rigid body"
                                      << " - cannot constrain" << endl;
            errors = true;
            }
        }

    if (errors)
        {
        throw std::runtime_error("Invalid constraint specified");
        }
    }


void export_ConstraintEllipsoid()
    {
    class_< ConstraintEllipsoid, boost::shared_ptr<ConstraintEllipsoid>, bases<Updater>, boost::noncopyable >
    ("ConstraintEllipsoid", init< boost::shared_ptr<SystemDefinition>,
                                                 boost::shared_ptr<ParticleGroup>,
                                                 Scalar3,
                                                 Scalar,
                                                 Scalar,
                                                 Scalar >())
    ;
    }
