// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "ExternalPotentialLinear.h"

namespace hoomd
    {
namespace hpmc
    {

void ExternalPotentialLinear::setAlpha(const std::string& particle_type, LongReal alpha)
    {
    unsigned int particle_type_id = m_sysdef->getParticleData()->getTypeByName(particle_type);
    m_alpha[particle_type_id] = alpha;
    }

LongReal ExternalPotentialLinear::getAlpha(const std::string& particle_type)
    {
    unsigned int particle_type_id = m_sysdef->getParticleData()->getTypeByName(particle_type);
    return m_alpha[particle_type_id];
    }

LongReal ExternalPotentialLinear::particleEnergyImplementation(unsigned int type_i,
                                                               const vec3<LongReal>& r_i,
                                                               const quat<LongReal>& q_i,
                                                               LongReal charge_i,
                                                               bool trial)
    {
    return m_alpha[type_i] * dot(m_plane_normal, r_i - m_plane_origin);
    }

namespace detail
    {
void exportExternalPotentialLinear(pybind11::module& m)
    {
    pybind11::class_<ExternalPotentialLinear,
                     ExternalPotential,
                     std::shared_ptr<ExternalPotentialLinear>>(m, "ExternalPotentialLinear")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def_property("plane_origin",
                      &ExternalPotentialLinear::getPlaneOrigin,
                      &ExternalPotentialLinear::setPlaneOrigin)
        .def_property("plane_normal",
                      &ExternalPotentialLinear::getPlaneNormal,
                      &ExternalPotentialLinear::setPlaneNormal)
        .def("setAlpha", &ExternalPotentialLinear::setAlpha)
        .def("getAlpha", &ExternalPotentialLinear::getAlpha);
    }
    } // end namespace detail
    } // end namespace hpmc
    } // end namespace hoomd
