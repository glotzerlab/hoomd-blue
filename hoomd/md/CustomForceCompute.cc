// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// Maintainer: joaander

#include "hoomd/PythonLocalDataAccess.h"
#include "CustomForceCompute.h"

namespace py = pybind11;

using namespace std;

/*! \file CustomForceCompute.cc
    \brief Contains code for the CustomForceCompute class
*/

/*! \param sysdef SystemDefinition containing the ParticleData to compute forces on
*/
CustomForceCompute::CustomForceCompute(std::shared_ptr<SystemDefinition> sysdef)
    : ForceCompute(sysdef), m_need_rearrange_forces(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing ConstForceCompute" << endl;
    }

CustomForceCompute::~CustomForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying ConstForceCompute" << endl;
    }


void CustomForceCompute::rearrangeForces()
    {
    if (m_forces.size())
        {
        assert(m_forces.size() == m_torques.size());

        ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::readwrite);
        ArrayHandle<Scalar4> h_torque(m_torque, access_location::host, access_mode::readwrite);

        assert(h_force.data);
        assert(h_torque.data);

        for (auto it = m_forces.begin(); it != m_forces.end(); ++it)
            {
            unsigned int tag = it->first;
            unsigned int idx = m_pdata->getRTag(tag);
            bool ptl_local = (idx < m_pdata->getN());
            vec3<Scalar> f = it->second;
            vec3<Scalar> t = m_torques[it->first];

            if (ptl_local)
                {
                h_force.data[idx].x = f.x;
                h_force.data[idx].y = f.y;
                h_force.data[idx].z = f.z;
                h_force.data[idx].w = 0;

                h_torque.data[idx].x = t.x;
                h_torque.data[idx].y = t.y;
                h_torque.data[idx].z = t.z;
                h_torque.data[idx].w = 0;
                }
            }
        }

    m_need_rearrange_forces = false;
    }

/*! This function calls rearrangeForces() whenever the particles have been sorted
    \param timestep Current timestep
*/
void CustomForceCompute::computeForces(uint64_t timestep)
    {
    if (m_particles_sorted || m_need_rearrange_forces)
        rearrangeForces();

    // execute python callback to update the forces, if present
    if (m_callback && !m_callback.is(py::none()))
        {
        m_callback(timestep);
        }
    }

void export_CustomForceCompute(py::module& m)
    {
    py::class_<CustomForceCompute, ForceCompute, std::shared_ptr<CustomForceCompute>>(
        m,
        "CustomForceCompute")
        .def(py::init<std::shared_ptr<SystemDefinition>>())
        .def("setCallback", &CustomForceCompute::setCallback);
    }
