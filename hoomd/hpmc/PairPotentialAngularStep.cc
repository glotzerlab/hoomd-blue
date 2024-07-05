// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "PairPotentialAngularStep.h"

namespace hoomd
    {
namespace hpmc
    {

PairPotentialAngularStep::PairPotentialAngularStep(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<PairPotential> isotropic_potential)
    : PairPotential(sysdef), m_isotropic_potential(isotropic_potential)
    {
    unsigned int ntypes = m_sysdef->getParticleData()->getNTypes();
    m_directors.resize(ntypes);
    m_cos_deltas.resize(ntypes);

    if (!m_isotropic_potential)
        {
        throw std::runtime_error("no isotropic potential given.");
        }
    }

void PairPotentialAngularStep::setMask(std::string particle_type, pybind11::object v)
    {
    unsigned int particle_type_id = m_sysdef->getParticleData()->getTypeByName(particle_type);

    if (v.is_none())
        {
        m_directors[particle_type_id].clear();
        m_cos_deltas[particle_type_id].clear();
        return;
        }
    pybind11::list directors = v["directors"];
    pybind11::list deltas = v["deltas"];

    auto N = pybind11::len(directors);

    if (pybind11::len(deltas) != N)
        {
        throw std::runtime_error("the length of the delta list should match the length"
                                 "of the director list.");
        }

    m_directors[particle_type_id].resize(N);
    m_cos_deltas[particle_type_id].resize(N);

    for (unsigned int i = 0; i < N; i++)
        {
        pybind11::tuple director_python = directors[i];
        if (pybind11::len(director_python) != 3)
            {
            throw std::length_error("director must be a list of 3-tuples.");
            }

        m_directors[particle_type_id][i] = vec3<LongReal>(director_python[0].cast<LongReal>(),
                                                          director_python[1].cast<LongReal>(),
                                                          director_python[2].cast<LongReal>());

        // normalize the directional vector
        m_directors[particle_type_id][i]
            /= fast::sqrt(dot(m_directors[particle_type_id][i], m_directors[particle_type_id][i]));

        pybind11::handle delta_python = deltas[i];
        m_cos_deltas[particle_type_id][i] = cos(delta_python.cast<LongReal>());
        }
    }

pybind11::object PairPotentialAngularStep::getMask(std::string particle_type)
    {
    unsigned int particle_type_id = m_sysdef->getParticleData()->getTypeByName(particle_type);
    size_t N = m_directors[particle_type_id].size();

    if (N == 0)
        {
        return pybind11::none();
        }

    pybind11::list directors;
    pybind11::list deltas;

    for (unsigned int i = 0; i < N; i++)
        {
        directors.append(pybind11::make_tuple(m_directors[particle_type_id][i].x,
                                              m_directors[particle_type_id][i].y,
                                              m_directors[particle_type_id][i].z));
        deltas.append(acos(m_cos_deltas[particle_type_id][i]));
        }

    pybind11::dict v;
    v["directors"] = directors;
    v["deltas"] = deltas;
    return v;
    }

// protected
LongReal PairPotentialAngularStep::energy(const LongReal r_squared,
                                          const vec3<LongReal>& r_ij,
                                          const unsigned int type_i,
                                          const quat<LongReal>& q_i,
                                          const LongReal charge_i,
                                          const unsigned int type_j,
                                          const quat<LongReal>& q_j,
                                          const LongReal charge_j) const
    {
    if (maskingFunction(r_squared, r_ij, type_i, q_i, type_j, q_j))
        {
        return m_isotropic_potential
            ->energy(r_squared, r_ij, type_i, q_i, charge_i, type_j, q_j, charge_j);
        }
    return 0;
    }

namespace detail
    {
void exportPairPotentialAngularStep(pybind11::module& m)
    {
    pybind11::class_<PairPotentialAngularStep,
                     PairPotential,
                     std::shared_ptr<PairPotentialAngularStep>>(m, "PairPotentialAngularStep")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<PairPotential>>())
        .def("setMask", &PairPotentialAngularStep::setMask)
        .def("getMask", &PairPotentialAngularStep::getMask);
    }
    } // end namespace detail

    } // end namespace hpmc
    } // end namespace hoomd
