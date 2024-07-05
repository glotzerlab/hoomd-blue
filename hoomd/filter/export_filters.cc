// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "export_filters.h"
#include "ParticleFilter.h"
#include "ParticleFilterAll.h"
#include "ParticleFilterCustom.h"
#include "ParticleFilterIntersection.h"
#include "ParticleFilterNull.h"
#include "ParticleFilterRigid.h"
#include "ParticleFilterSetDifference.h"
#include "ParticleFilterTags.h"
#include "ParticleFilterType.h"
#include "ParticleFilterUnion.h"

namespace hoomd
    {
namespace detail
    {
void export_ParticleFilters(pybind11::module& m)
    {
    pybind11::class_<ParticleFilter, std::shared_ptr<ParticleFilter>>(m, "ParticleFilter")
        .def(pybind11::init<>())
        .def("_get_selected_tags", &ParticleFilter::getSelectedTags);

    pybind11::class_<ParticleFilterSetDifference,
                     ParticleFilter,
                     std::shared_ptr<ParticleFilterSetDifference>>(m, "ParticleFilterSetDifference")
        .def(pybind11::init<std::shared_ptr<ParticleFilter>, std::shared_ptr<ParticleFilter>>());

    pybind11::class_<ParticleFilterUnion, ParticleFilter, std::shared_ptr<ParticleFilterUnion>>(
        m,
        "ParticleFilterUnion")
        .def(pybind11::init<std::shared_ptr<ParticleFilter>, std::shared_ptr<ParticleFilter>>());

    pybind11::class_<ParticleFilterType, ParticleFilter, std::shared_ptr<ParticleFilterType>>(
        m,
        "ParticleFilterType")
        .def(pybind11::init<std::unordered_set<std::string>>());

    pybind11::class_<ParticleFilterAll, ParticleFilter, std::shared_ptr<ParticleFilterAll>>(
        m,
        "ParticleFilterAll")
        .def(pybind11::init<>());

    pybind11::class_<ParticleFilterNull, ParticleFilter, std::shared_ptr<ParticleFilterNull>>(
        m,
        "ParticleFilterNull")
        .def(pybind11::init<>());

    pybind11::class_<ParticleFilterIntersection,
                     ParticleFilter,
                     std::shared_ptr<ParticleFilterIntersection>>(m, "ParticleFilterIntersection")
        .def(pybind11::init<std::shared_ptr<ParticleFilter>, std::shared_ptr<ParticleFilter>>());

    pybind11::class_<ParticleFilterTags, ParticleFilter, std::shared_ptr<ParticleFilterTags>>(
        m,
        "ParticleFilterTags")
        .def(pybind11::init<pybind11::array_t<unsigned int, pybind11::array::c_style>>());

    pybind11::class_<ParticleFilterCustom, ParticleFilter, std::shared_ptr<ParticleFilterCustom>>(
        m,
        "ParticleFilterCustom")
        .def(pybind11::init<pybind11::object, pybind11::object>());

    pybind11::class_<ParticleFilterRigid, ParticleFilter, std::shared_ptr<ParticleFilterRigid>>(
        m,
        "ParticleFilterRigid")
        .def(pybind11::init<pybind11::tuple>());
    };

    } // end namespace detail

    } // end namespace hoomd
