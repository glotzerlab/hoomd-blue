// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander


#include "OneDConstraint.h"
#include "EvaluatorConstraint.h"

namespace py = pybind11;

using namespace std;

/*! \file OneDConstraint.cc
    \brief Contains code for the OneDConstraint class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
    \param group Group of particles on which to apply this constraint
*/
OneDConstraint::OneDConstraint(std::shared_ptr<SystemDefinition> sysdef,
                                   std::shared_ptr<ParticleGroup> group,
                                   Scalar3 constraint_vec)
        : ForceConstraint(sysdef), m_group(group), m_vec(constraint_vec)
    {
    m_exec_conf->msg->notice(5) << "Constructing OneDConstraint" << endl;

    }

OneDConstraint::~OneDConstraint()
    {
    m_exec_conf->msg->notice(5) << "Destroying OneDConstraint" << endl;
    }

/*!
    \param constraint_vec direction that particles are constrained to
*/
void OneDConstraint::setVector(Scalar3 constraint_vec)
    {
    m_vec = constraint_vec;
    }

/*! Computes the specified constraint forces
    \param timestep Current timestep
*/
void OneDConstraint::computeForces(unsigned int timestep)
    {
    unsigned int group_size = m_group->getNumMembers();
    if (group_size == 0)
        return;

    if (m_prof) m_prof->push("OneDConstraint");

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
        Scalar3 U = constraint.evalU();
        Scalar3 D = make_scalar3((U.x - X.x), (U.y - X.y), (U.z - X.z));
        Scalar n = (D.x*m_vec.x + D.y*m_vec.y + D.z*m_vec.z)/(m_vec.x*m_vec.x + m_vec.y*m_vec.y + m_vec.z*m_vec.z);
        Scalar3 C = make_scalar3((n*m_vec.x + X.x), (n*m_vec.y + X.y), (n*m_vec.z + X.z));

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

/*! OneDConstraint removes 2 degrees of freedom per particle in the group
*/
unsigned int OneDConstraint::getNDOFRemoved()
    {
    return m_group->getNumMembersGlobal();
    }

void export_OneDConstraint(py::module& m)
    {
    py::class_< OneDConstraint, std::shared_ptr<OneDConstraint> >(m, "OneDConstraint", py::base<ForceConstraint>())
    .def(py::init< std::shared_ptr<SystemDefinition>,
                   std::shared_ptr<ParticleGroup>,
                   Scalar3 >())
    .def("getNDOFRemoved", &OneDConstraint::getNDOFRemoved)
    .def("setVector", &OneDConstraint::setVector)
    ;
    }
