// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include <numeric>

#include "PairPotentialUnion.h"

#include "hoomd/hpmc/OBBTree.h"

namespace hoomd
    {
namespace hpmc
    {
PairPotentialUnion::PairPotentialUnion(std::shared_ptr<SystemDefinition> sysdef,
                                       std::shared_ptr<PairPotential> constituent_potential)
    : PairPotential(sysdef), m_constituent_potential(constituent_potential), m_leaf_capacity(4)
    {
    unsigned int ntypes = m_sysdef->getParticleData()->getNTypes();
    m_extent_type.resize(ntypes, 0.0);
    m_type.resize(ntypes);
    m_position.resize(ntypes);
    m_orientation.resize(ntypes);
    m_charge.resize(ntypes);
    m_tree.resize(ntypes);
    }

void PairPotentialUnion::setBody(std::string body_type, pybind11::object v)
    {
    unsigned int body_type_id = m_sysdef->getParticleData()->getTypeByName(body_type);

    if (v.is_none())
        {
        m_extent_type[body_type_id] = 0;
        m_position[body_type_id].clear();
        m_orientation[body_type_id].clear();
        m_charge[body_type_id].clear();
        m_type[body_type_id].clear();
        return;
        }

    pybind11::list positions = v["positions"];
    pybind11::object types_obj = v["types"];
    pybind11::object orientations;
    pybind11::object charges;
    if (v.contains("orientations"))
        {
        orientations = v["orientations"];
        }
    else
        {
        orientations = pybind11::none();
        }
    if (v.contains("charges"))
        {
        charges = v["charges"];
        }
    else
        {
        charges = pybind11::none();
        }

    auto N = pybind11::len(positions);
    // Ensure proper list lengths
    for (const auto& object : {orientations, types_obj, charges})
        {
        if (!object.is_none() && pybind11::len(object) != N)
            {
            throw std::runtime_error("All attributes of a union's body must be the same length.");
            }
        }

    pybind11::list types(types_obj);

    m_position[body_type_id].resize(N);
    m_orientation[body_type_id].resize(N);
    m_charge[body_type_id].resize(N);
    m_type[body_type_id].resize(N);

    for (unsigned int i = 0; i < N; i++)
        {
        pybind11::tuple r_python = positions[i];
        if (pybind11::len(r_python) != 3)
            {
            throw std::length_error("positions must be a list of 3-tuples.");
            }
        m_position[body_type_id][i] = vec3<LongReal>(r_python[0].cast<LongReal>(),
                                                     r_python[1].cast<LongReal>(),
                                                     r_python[2].cast<LongReal>());

        if (orientations.is_none())
            {
            m_orientation[body_type_id][i] = quat<LongReal>(LongReal(1.0), vec3<LongReal>(0, 0, 0));
            }
        else
            {
            pybind11::tuple q_python = pybind11::list(orientations)[i];
            if (pybind11::len(q_python) != 4)
                {
                throw std::length_error("orientations must be a list of 4-tuples.");
                }

            m_orientation[body_type_id][i]
                = quat<LongReal>(q_python[0].cast<LongReal>(),
                                 vec3<LongReal>(q_python[1].cast<LongReal>(),
                                                q_python[2].cast<LongReal>(),
                                                q_python[3].cast<LongReal>()));
            }

        m_type[body_type_id][i]
            = m_sysdef->getParticleData()->getTypeByName(types[i].cast<std::string>());

        if (charges.is_none())
            {
            m_charge[body_type_id][i] = 0;
            }
        else
            {
            m_charge[body_type_id][i] = pybind11::list(charges)[i].cast<LongReal>();
            }
        }

    updateExtent(body_type_id);
    buildOBBTree(body_type_id);
    notifyRCutChanged();
    }

pybind11::object PairPotentialUnion::getBody(std::string body_type)
    {
    unsigned int body_type_id = m_sysdef->getParticleData()->getTypeByName(body_type);
    size_t N = m_position[body_type_id].size();
    if (N == 0)
        {
        return pybind11::none();
        }

    pybind11::list positions;
    pybind11::list orientations;
    pybind11::list types;
    pybind11::list charges;

    for (unsigned int i = 0; i < N; i++)
        {
        positions.append(pybind11::make_tuple(m_position[body_type_id][i].x,
                                              m_position[body_type_id][i].y,
                                              m_position[body_type_id][i].z));
        orientations.append(pybind11::make_tuple(m_orientation[body_type_id][i].s,
                                                 m_orientation[body_type_id][i].v.x,
                                                 m_orientation[body_type_id][i].v.y,
                                                 m_orientation[body_type_id][i].v.z));
        types.append(m_sysdef->getParticleData()->getNameByType(m_type[body_type_id][i]));
        charges.append(m_charge[body_type_id][i]);
        }

    pybind11::dict v;
    v["types"] = types;
    v["positions"] = positions;
    v["orientations"] = orientations;
    v["charges"] = charges;
    return v;
    }

LongReal PairPotentialUnion::computeRCutNonAdditive(unsigned int type_i, unsigned int type_j) const
    {
    // The non-additive part of the union r_cut is the maximum of the constituent non-additive
    // r_cuts over type pairs that interact between two union particles of type_i and type_j

    LongReal r_cut = 0;

    for (auto& constituent_type_i : m_type[type_i])
        {
        for (auto& constituent_type_j : m_type[type_j])
            {
            r_cut = std::max(r_cut,
                             m_constituent_potential->computeRCutNonAdditive(constituent_type_i,
                                                                             constituent_type_j));
            }
        }

    return r_cut;
    }

LongReal PairPotentialUnion::computeRCutAdditive(unsigned int type) const
    {
    // The additive cutoff is twice the radius of the constituent particle furthest from the
    // origin.
    assert(type <= m_extent_type.size());

    if (m_constituent_potential->computeRCutAdditive(type) > 0)
        {
        throw std::domain_error("Unsupported constituent potential.");
        }

    return m_extent_type[type];
    }

void PairPotentialUnion::updateExtent(unsigned int type_id)
    {
    // The extent is 2x the maximum distance of constituent particles to the origin
    m_extent_type[type_id] = 0;

    for (const auto& pos : m_position[type_id])
        {
        m_extent_type[type_id]
            = std::max(m_extent_type[type_id], LongReal(2) * slow::sqrt(dot(pos, pos)));
        }
    }

void PairPotentialUnion::buildOBBTree(unsigned int type_id)
    {
    if (m_leaf_capacity > 0)
        {
        auto N = static_cast<unsigned int>(m_position[type_id].size());
        hpmc::detail::OBB* obbs = new hpmc::detail::OBB[N];
        for (unsigned int i = 0; i < N; i++)
            {
            auto pos = m_position[type_id][i];

            // use a point-sized OBB
            obbs[i] = hpmc::detail::OBB(pos, 0.0);

            // we do not support exclusions
            obbs[i].mask = 1;
            }

        // build tree and store proxy structure
        hpmc::detail::OBBTree tree;
        tree.buildTree(obbs, N, m_leaf_capacity, false);
        delete[] obbs;
        m_tree[type_id] = hpmc::detail::GPUTree(tree, false);
        }
    }

LongReal PairPotentialUnion::compute_leaf_leaf_energy(vec3<LongReal> dr,
                                                      unsigned int type_a,
                                                      unsigned int type_b,
                                                      const quat<LongReal>& orientation_a,
                                                      const quat<LongReal>& orientation_b,
                                                      unsigned int cur_node_a,
                                                      unsigned int cur_node_b) const
    {
    LongReal energy = 0.0;
    vec3<LongReal> r_ab = rotate(conj(orientation_b), dr);

    // loop through leaf particles of cur_node_a
    unsigned int na = m_tree[type_a].getNumParticles(cur_node_a);
    unsigned int nb = m_tree[type_b].getNumParticles(cur_node_b);

    for (unsigned int i = 0; i < na; i++)
        {
        unsigned int ileaf = m_tree[type_a].getParticleByNode(cur_node_a, i);

        unsigned int type_i = m_type[type_a][ileaf];
        quat<LongReal> orientation_i
            = conj(orientation_b) * orientation_a * m_orientation[type_a][ileaf];
        vec3<LongReal> pos_i(rotate(conj(orientation_b) * orientation_a, m_position[type_a][ileaf])
                             - r_ab);

        // loop through leaf particles of cur_node_b
        for (unsigned int j = 0; j < nb; j++)
            {
            unsigned int jleaf = m_tree[type_b].getParticleByNode(cur_node_b, j);

            unsigned int type_j = m_type[type_b][jleaf];
            quat<LongReal> orientation_j = m_orientation[type_b][jleaf];
            vec3<LongReal> r_ij = m_position[type_b][jleaf] - pos_i;

            LongReal rsq = dot(r_ij, r_ij);
            if (rsq < m_constituent_potential->getRCutSquaredTotal(type_i, type_j))
                {
                energy += m_constituent_potential->energy(rsq,
                                                          r_ij,
                                                          type_i,
                                                          orientation_i,
                                                          m_charge[type_a][ileaf],
                                                          type_j,
                                                          orientation_j,
                                                          m_charge[type_b][jleaf]);
                }
            }
        }
    return energy;
    }

LongReal PairPotentialUnion::energy(const LongReal r_squared,
                                    const vec3<LongReal>& r_ij,
                                    const unsigned int type_i,
                                    const quat<LongReal>& q_i,
                                    const LongReal charge_i,
                                    const unsigned int type_j,
                                    const quat<LongReal>& q_j,
                                    const LongReal charge_j) const
    {
    if (m_leaf_capacity == 0)
        {
        return energyAll(r_squared, r_ij, type_i, q_i, charge_i, type_j, q_j, charge_j);
        }
    else
        {
        return energyOBB(r_squared, r_ij, type_i, q_i, charge_i, type_j, q_j, charge_j);
        }
    }

namespace detail
    {
void exportPairPotentialUnion(pybind11::module& m)
    {
    pybind11::class_<PairPotentialUnion, PairPotential, std::shared_ptr<PairPotentialUnion>>(
        m,
        "PairPotentialUnion")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<PairPotential>>())
        .def("getBody", &PairPotentialUnion::getBody)
        .def("setBody", &PairPotentialUnion::setBody)
        .def_property("leaf_capacity",
                      &PairPotentialUnion::getLeafCapacity,
                      &PairPotentialUnion::setLeafCapacity);
    }

    } // end namespace detail
    } // end namespace hpmc
    } // end namespace hoomd
