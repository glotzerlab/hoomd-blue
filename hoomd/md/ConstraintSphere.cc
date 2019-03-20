// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander


#include "ConstraintSphere.h"
#include "EvaluatorConstraint.h"
#include "EvaluatorConstraintSphere.h"

namespace py = pybind11;

using namespace std;

/*! \file ConstraintSphere.cc
    \brief Contains code for the ConstraintSphere class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
    \param group Group of particles on which to apply this constraint
    \param P position of the sphere
    \param r radius of the sphere
*/
ConstraintSphere::ConstraintSphere(std::shared_ptr<SystemDefinition> sysdef,
                                   std::shared_ptr<ParticleGroup> group,
                                   Scalar3 P,
                                   Scalar r)
        : ForceConstraint(sysdef), m_group(group), m_P(P), m_r(r)
    {
    m_exec_conf->msg->notice(5) << "Constructing ConstraintSphere" << endl;

    validate();
    }

ConstraintSphere::~ConstraintSphere()
    {
    m_exec_conf->msg->notice(5) << "Destroying ConstraintSphere" << endl;
    }

/*!
    \param P position of the sphere
    \param r radius of the sphere
*/
void ConstraintSphere::setSphere(Scalar3 P, Scalar r)
    {
    m_P = P;
    m_r = r;
    validate();
    }

/*! ConstraintSphere removes 1 degree of freedom per particle in the group
*/
unsigned int ConstraintSphere::getNDOFRemoved()
    {
    return m_group->getNumMembersGlobal();
    }

/*! Computes the specified constraint forces
    \param timestep Current timestep
*/
void ConstraintSphere::computeForces(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;

    if (m_prof) m_prof->push("ConstraintSphere");

    assert(m_pdata);
    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_vel(m_pdata->getVelocities(), access_location::host, access_mode::read);

    const GlobalArray< Scalar4 >& net_force = m_pdata->getNetForce();
    ArrayHandle<Scalar4> h_net_force(net_force, access_location::host, access_mode::read);


    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::overwrite);
    unsigned int virial_pitch = m_virial.getPitch();

    // Zero data for force calculation.
    memset((void*)h_force.data,0,sizeof(Scalar4)*m_force.getNumElements());
    memset((void*)h_virial.data,0,sizeof(Scalar)*m_virial.getNumElements());

   // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);

    // for each of the particles in the group
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        // get the current particle properties
        unsigned int j = m_group->getMemberIndex(group_idx);
        Scalar3 X = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);
        Scalar3 V = make_scalar3(h_vel.data[j].x, h_vel.data[j].y, h_vel.data[j].z);
        Scalar3 F = make_scalar3(h_net_force.data[j].x, h_net_force.data[j].y, h_net_force.data[j].z);
        Scalar m = h_vel.data[j].w;

        // evaluate the constraint position
        EvaluatorConstraint constraint(X, V, F, m, m_deltaT);
        EvaluatorConstraintSphere sphere(m_P, m_r);
        Scalar3 C = sphere.evalClosest(constraint.evalU());

        // evaluate the constraint force
        Scalar3 FC;
        Scalar virial[6];
        constraint.evalConstraintForce(FC, virial, C);

        // apply the constraint force
        h_force.data[j].x = FC.x;
        h_force.data[j].y = FC.y;
        h_force.data[j].z = FC.z;
        for (int k = 0; k < 6; k++)
            h_virial.data[k*virial_pitch+j]  = virial[k];
        }

    if (m_prof)
        m_prof->pop();
    }

/*! Print warning messages if the sphere is outside the box.
    Generate an error if any particle in the group is not near the sphere.
*/
void ConstraintSphere::validate()
    {
    BoxDim box = m_pdata->getGlobalBox();
    Scalar3 lo = box.getLo();
    Scalar3 hi = box.getHi();

    if (m_P.x + m_r > hi.x || m_P.x - m_r < lo.x ||
        m_P.y + m_r > hi.y || m_P.y - m_r < lo.y ||
        m_P.z + m_r > hi.z || m_P.z - m_r < lo.z)
        {
        m_exec_conf->msg->warning() << "constrain.sphere: Sphere constraint is outside of the box. Constrained particle positions may be incorrect"
             << endl;
        }

    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;

    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(), access_location::host, access_mode::read);

    // for each of the particles in the group
    bool errors = false;
    for (unsigned int group_idx = 0; group_idx < group_size; group_idx++)
        {
        // get the current particle properties
        unsigned int j = m_group->getMemberIndex(group_idx);
        Scalar3 X = make_scalar3(h_pos.data[j].x, h_pos.data[j].y, h_pos.data[j].z);

        // evaluate the constraint position
        EvaluatorConstraintSphere sphere(m_P, m_r);
        Scalar3 C = sphere.evalClosest(X);
        Scalar3 V;
        V.x = C.x - X.x;
        V.y = C.y - X.y;
        V.z = C.z - X.z;
        Scalar dist = sqrt(V.x*V.x + V.y*V.y + V.z*V.z);

        if (dist > Scalar(1.0))
            {
            m_exec_conf->msg->error() << "constrain.sphere: Particle " << h_tag.data[j] << " is more than 1 unit of"
                                      << " distance away from the closest point on the sphere constraint" << endl;
            errors = true;
            }

        if (h_body.data[j] < MIN_FLOPPY)
            {
            m_exec_conf->msg->error() << "constrain.sphere: Particle " << h_tag.data[j] << " belongs to a rigid body"
                                      << " - cannot constrain" << endl;
            errors = true;
            }
        }

    if (errors)
        {
        throw std::runtime_error("Invalid constraint specified");
        }
    }


void export_ConstraintSphere(py::module& m)
    {
    py::class_< ConstraintSphere, std::shared_ptr<ConstraintSphere> >(m, "ConstraintSphere", py::base<ForceConstraint>())
    .def(py::init< std::shared_ptr<SystemDefinition>,
                     std::shared_ptr<ParticleGroup>,
                     Scalar3,
                     Scalar >())
    .def("setSphere", &ConstraintSphere::setSphere)
    .def("getNDOFRemoved", &ConstraintSphere::getNDOFRemoved)
    ;
    }
