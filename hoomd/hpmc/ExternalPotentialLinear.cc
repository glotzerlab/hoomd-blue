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

static void setPlaneOrigin(std::shared_ptr<ExternalPotentialLinear> field, pybind11::tuple origin)
    {
    if (pybind11::len(origin) != 3)
        {
        throw std::length_error("Plane origin must have length 3");
        }
    vec3<LongReal> new_origin(origin[0].cast<LongReal>(),
                              origin[1].cast<LongReal>(),
                              origin[2].cast<LongReal>());
    field->setPlaneOrigin(new_origin);
    }

static pybind11::tuple getPlaneOrigin(std::shared_ptr<ExternalPotentialLinear> field)
    {
    vec3<LongReal> origin(field->getPlaneOrigin());
    return pybind11::make_tuple(origin.x, origin.y, origin.z);
    }

static void setPlaneNormal(std::shared_ptr<ExternalPotentialLinear> field, pybind11::tuple normal)
    {
    if (pybind11::len(normal) != 3)
        {
        throw std::length_error("Plane normal must have length 3");
        }
    vec3<LongReal> new_normal(normal[0].cast<LongReal>(),
                              normal[1].cast<LongReal>(),
                              normal[2].cast<LongReal>());
    field->setPlaneNormal(new_normal);
    }

static pybind11::tuple getPlaneNormal(std::shared_ptr<ExternalPotentialLinear> field)
    {
    vec3<LongReal> normal(field->getPlaneNormal());
    return pybind11::make_tuple(normal.x, normal.y, normal.z);
    }

void exportExternalPotentialLinear(pybind11::module& m)
    {
    pybind11::class_<ExternalPotentialLinear,
                     ExternalPotential,
                     std::shared_ptr<ExternalPotentialLinear>>(m, "ExternalPotentialLinear")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def_property("plane_origin", &getPlaneOrigin, &setPlaneOrigin)
        .def_property("plane_normal", &getPlaneNormal, &setPlaneNormal)
        .def("setAlpha", &ExternalPotentialLinear::setAlpha)
        .def("getAlpha", &ExternalPotentialLinear::getAlpha);
    }
    } // end namespace detail
    } // end namespace hpmc
    } // end namespace hoomd
