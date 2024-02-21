// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

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
        m_position[body_type_id][i] = vec3<ShortReal>(r_python[0].cast<ShortReal>(),
                                                      r_python[1].cast<ShortReal>(),
                                                      r_python[2].cast<ShortReal>());

        if (orientations.is_none())
            {
            m_orientation[body_type_id][i]
                = quat<ShortReal>(ShortReal(1.0), vec3<ShortReal>(0, 0, 0));
            }
        else
            {
            pybind11::tuple q_python = pybind11::list(orientations)[i];
            if (pybind11::len(q_python) != 4)
                {
                throw std::length_error("orientations must be a list of 4-tuples.");
                }

            m_orientation[body_type_id][i]
                = quat<ShortReal>(q_python[0].cast<ShortReal>(),
                                  vec3<ShortReal>(q_python[1].cast<ShortReal>(),
                                                  q_python[2].cast<ShortReal>(),
                                                  q_python[3].cast<ShortReal>()));
            }

        m_type[body_type_id][i]
            = m_sysdef->getParticleData()->getTypeByName(types[i].cast<std::string>());

        if (charges.is_none())
            {
            m_charge[body_type_id][i] = 0;
            }
        else
            {
            m_charge[body_type_id][i] = pybind11::list(charges)[i].cast<ShortReal>();
            }
        }

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
    return std::move(v);
    }

void PairPotentialUnion::buildOBBTree(unsigned int type)
    {
    auto N = static_cast<unsigned int>(m_position[type].size());
    hpmc::detail::OBB* obbs = new hpmc::detail::OBB[N];
    // extract member parameters, positions, and orientations and compute the rcut along the
    // way
    Scalar extent_i = 0.0; // 2x the dist of farthest-away constituent particle of type i,
                           // used for r_cut calc
    for (unsigned int i = 0; i < N; i++)
        {
        auto pos = m_position[type][i];

        // use a point-sized OBB
        obbs[i] = hpmc::detail::OBB(pos, 0.0);

        // calcuate distance from center of union to this constituent particle
        Scalar r = sqrt(dot(pos, pos));

        // extent_i is twice the distance to the farthest particle, so is ~ the circumsphere
        // diameter
        extent_i = std::max(extent_i, Scalar(2 * r));

        // we do not support exclusions
        obbs[i].mask = 1;
        }

    // set the diameter
    m_extent_type[type] = extent_i;

    // build tree and store proxy structure
    hpmc::detail::OBBTree tree;
    tree.buildTree(obbs, N, m_leaf_capacity, false);
    delete[] obbs;
    m_tree[type] = hpmc::detail::GPUTree(tree, false);
    }

LongReal PairPotentialUnion::compute_leaf_leaf_energy(vec3<ShortReal> dr,
                                                      unsigned int type_a,
                                                      unsigned int type_b,
                                                      const quat<ShortReal>& orientation_a,
                                                      const quat<ShortReal>& orientation_b,
                                                      unsigned int cur_node_a,
                                                      unsigned int cur_node_b) const
    {
    LongReal energy = 0.0;
    vec3<ShortReal> r_ab = rotate(conj(quat<ShortReal>(orientation_b)), vec3<ShortReal>(dr));

    // loop through leaf particles of cur_node_a
    unsigned int na = m_tree[type_a].getNumParticles(cur_node_a);
    unsigned int nb = m_tree[type_b].getNumParticles(cur_node_b);

    for (unsigned int i = 0; i < na; i++)
        {
        unsigned int ileaf = m_tree[type_a].getParticleByNode(cur_node_a, i);

        unsigned int type_i = m_type[type_a][ileaf];
        quat<ShortReal> orientation_i = conj(quat<ShortReal>(orientation_b))
                                        * quat<ShortReal>(orientation_a)
                                        * m_orientation[type_a][ileaf];
        vec3<ShortReal> pos_i(
            rotate(conj(quat<ShortReal>(orientation_b)) * quat<ShortReal>(orientation_a),
                   m_position[type_a][ileaf])
            - r_ab);

        // loop through leaf particles of cur_node_b
        for (unsigned int j = 0; j < nb; j++)
            {
            unsigned int jleaf = m_tree[type_b].getParticleByNode(cur_node_b, j);

            unsigned int type_j = m_type[type_b][jleaf];
            quat<ShortReal> orientation_j = m_orientation[type_b][jleaf];
            vec3<ShortReal> r_ij = m_position[type_b][jleaf] - pos_i;

            ShortReal rsq = dot(r_ij, r_ij);
            if (rsq < getRCutSquaredTotal(type_i, type_j))
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
    // TODO: Implement N^2 code path. Determine at what point OBB's make sense.

    const hpmc::detail::GPUTree& tree_a = m_tree[type_i];
    const hpmc::detail::GPUTree& tree_b = m_tree[type_j];
    ShortReal r_cut_constituent = ShortReal(m_constituent_potential->getMaxRCutNonAdditive());

    LongReal energy = 0.0;

    if (tree_a.getNumLeaves() <= tree_b.getNumLeaves())
        {
        for (unsigned int cur_leaf_a = 0; cur_leaf_a < tree_a.getNumLeaves(); cur_leaf_a++)
            {
            unsigned int cur_node_a = tree_a.getLeafNode(cur_leaf_a);
            hpmc::detail::OBB obb_a = tree_a.getOBB(cur_node_a);

            // add range of interaction
            obb_a.lengths.x += r_cut_constituent;
            obb_a.lengths.y += r_cut_constituent;
            obb_a.lengths.z += r_cut_constituent;

            // rotate and translate a's obb into b's body frame
            obb_a.affineTransform(conj(q_j) * q_i, rotate(conj(q_j), -r_ij));

            unsigned cur_node_b = 0;
            while (cur_node_b < tree_b.getNumNodes())
                {
                unsigned int query_node = cur_node_b;
                if (tree_b.queryNode(obb_a, cur_node_b))
                    energy += compute_leaf_leaf_energy(r_ij,
                                                       type_i,
                                                       type_j,
                                                       q_i,
                                                       q_j,
                                                       cur_node_a,
                                                       query_node);
                }
            }
        }
    else
        {
        for (unsigned int cur_leaf_b = 0; cur_leaf_b < tree_b.getNumLeaves(); cur_leaf_b++)
            {
            unsigned int cur_node_b = tree_b.getLeafNode(cur_leaf_b);
            hpmc::detail::OBB obb_b = tree_b.getOBB(cur_node_b);

            // add range of interaction
            obb_b.lengths.x += r_cut_constituent;
            obb_b.lengths.y += r_cut_constituent;
            obb_b.lengths.z += r_cut_constituent;

            // rotate and translate b's obb into a's body frame
            obb_b.affineTransform(conj(q_i) * q_j, rotate(conj(q_i), r_ij));

            unsigned cur_node_a = 0;
            while (cur_node_a < tree_a.getNumNodes())
                {
                unsigned int query_node = cur_node_a;
                if (tree_a.queryNode(obb_b, cur_node_a))
                    energy += compute_leaf_leaf_energy(-r_ij,
                                                       type_j,
                                                       type_i,
                                                       q_j,
                                                       q_i,
                                                       cur_node_b,
                                                       query_node);
                }
            }
        }

    return energy;
    }

namespace detail
    {
void export_PairPotentialUnion(pybind11::module& m)
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
