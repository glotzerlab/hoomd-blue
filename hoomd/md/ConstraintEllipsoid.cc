// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander


#include "ConstraintEllipsoid.h"
#include "EvaluatorConstraintEllipsoid.h"

namespace py = pybind11;

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
ConstraintEllipsoid::ConstraintEllipsoid(std::shared_ptr<SystemDefinition> sysdef,
                                   std::shared_ptr<ParticleGroup> group,
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


void export_ConstraintEllipsoid(py::module& m)
    {
    py::class_< ConstraintEllipsoid, std::shared_ptr<ConstraintEllipsoid> >(m, "ConstraintEllipsoid", py::base<Updater>())
        .def(py::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<ParticleGroup>, Scalar3, Scalar, Scalar, Scalar >())
    ;
    }
