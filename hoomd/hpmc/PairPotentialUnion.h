// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "GPUTree.h"
#include "PairPotential.h"

namespace hoomd
    {
namespace hpmc
    {

/*** Evaluate pair potentials between sites on extended particles.

    The extended sites act as a union of particles. The constituent potential is applied between
    all pairs of sites between two particles.
*/
class PairPotentialUnion : public hpmc::PairPotential
    {
    public:
    PairPotentialUnion(std::shared_ptr<SystemDefinition> sysdef,
                       std::shared_ptr<PairPotential> constituent_potential);
    virtual ~PairPotentialUnion() { }

    /// Set OBB tree leaf capacity
    void setLeafCapacity(unsigned int leaf_capacity)
        {
        m_leaf_capacity = leaf_capacity;
        for (unsigned int type = 0; type < m_sysdef->getParticleData()->getNTypes(); type++)
            {
            buildOBBTree(type);
            }
        }

    /// Get OBB tree leaf capacity
    unsigned int getLeafCapacity()
        {
        return m_leaf_capacity;
        }

    /// Set union body parameters
    void setBody(std::string body_type, pybind11::object v);

    /// Get union body parameters
    pybind11::object getBody(std::string body_type);

    /*** Evaluate the energy of the pair interaction

        @param r_squared Pre-computed dot(r_ij, r_ij).
        @param r_ij Vector pointing from particle i to j.
        @param type_i Integer type index of particle i.
        @param charge_i Charge of particle i.
        @param q_i Orientation quaternion of particle i.
        @param type_j Integer type index of particle j.
        @param q_j Orientation quaternion of particle j.
        @param charge_j Charge of particle j.
        @returns Energy of the pair interaction.
    */
    virtual LongReal energy(const LongReal r_squared,
                            const vec3<LongReal>& r_ij,
                            const unsigned int type_i,
                            const quat<LongReal>& q_i,
                            const LongReal charge_i,
                            const unsigned int type_j,
                            const quat<LongReal>& q_j,
                            const LongReal charge_j) const;

    /// Compute the non-additive cuttoff radius
    virtual LongReal computeRCutNonAdditive(unsigned int type_i, unsigned int type_j) const;

    /// Returns the additive part of the cutoff distance for a given type.
    virtual LongReal computeRCutAdditive(unsigned int type) const;

    protected:
    /// The pair potential to apply between constituents.
    std::shared_ptr<PairPotential> m_constituent_potential;

    /// The tree acceleration structure per particle type.
    std::vector<hpmc::detail::GPUTree> m_tree;

    /// The per-type geometric extent.
    std::vector<Scalar> m_extent_type;

    /// The positions of the constituent particles.
    std::vector<std::vector<vec3<LongReal>>> m_position;

    // The orientations of the constituent particles.
    std::vector<std::vector<quat<LongReal>>> m_orientation;

    // The charges of the constituent particles.
    std::vector<std::vector<LongReal>> m_charge;

    // The type identifiers of the constituent particles.
    std::vector<std::vector<unsigned int>> m_type;

    // The number of particles in a leaf of the internal tree data structure.
    unsigned int m_leaf_capacity;

    /// Update extent.
    void updateExtent(unsigned int type_id);

    /// Builds OBB tree based on geometric properties of the constituent particles.
    void buildOBBTree(unsigned int type_id);

    /// Compute the energy of two overlapping leaf nodes.
    LongReal compute_leaf_leaf_energy(vec3<LongReal> dr,
                                      unsigned int type_a,
                                      unsigned int type_b,
                                      const quat<LongReal>& orientation_a,
                                      const quat<LongReal>& orientation_b,
                                      unsigned int cur_node_a,
                                      unsigned int cur_node_b) const;

    /*** Evaluate the energy of the pair interaction (all N * M pairs).
     */
    __attribute__((always_inline)) LongReal energyAll(const LongReal r_squared,
                                                      const vec3<LongReal>& r_ij,
                                                      const unsigned int type_i,
                                                      const quat<LongReal>& q_i,
                                                      const LongReal charge_i,
                                                      const unsigned int type_j,
                                                      const quat<LongReal>& q_j,
                                                      const LongReal charge_j) const
        {
        LongReal energy = 0.0;
        const size_t N_i = m_position[type_i].size();
        const size_t N_j = m_position[type_j].size();

        const quat<LongReal> conj_q_j(conj(q_j));
        const quat<LongReal> conj_q_j_q_i(conj_q_j * q_i);
        const rotmat3<LongReal> rotate_i_to_j(conj_q_j_q_i);

        vec3<LongReal> r_ij_rotated = rotate(conj_q_j, r_ij);

        for (unsigned int i = 0; i < N_i; i++)
            {
            // Rotate and translate the constituents of i to j's body frame.
            unsigned int constituent_type_i = m_type[type_i][i];
            quat<LongReal> constituent_orientation_i = conj_q_j_q_i * m_orientation[type_i][i];
            vec3<LongReal> constituent_position_i(rotate_i_to_j * m_position[type_i][i]
                                                  - r_ij_rotated);

            // loop through leaf particles of cur_node_b
            for (unsigned int j = 0; j < N_j; j++)
                {
                unsigned int constituent_type_j = m_type[type_j][j];
                quat<LongReal> constituent_orientation_j = m_orientation[type_j][j];
                vec3<LongReal> constituent_r_ij = m_position[type_j][j] - constituent_position_i;

                LongReal rsq = dot(constituent_r_ij, constituent_r_ij);
                if (rsq < m_constituent_potential->getRCutSquaredTotal(constituent_type_i,
                                                                       constituent_type_j))
                    {
                    energy += m_constituent_potential->energy(rsq,
                                                              constituent_r_ij,
                                                              constituent_type_i,
                                                              constituent_orientation_i,
                                                              m_charge[type_i][i],
                                                              constituent_type_j,
                                                              constituent_orientation_j,
                                                              m_charge[type_j][j]);
                    }
                }
            }
        return energy;
        }

    /*** Evaluate the energy of the pair interaction (using the OBB tree).
     */
    __attribute__((always_inline)) LongReal energyOBB(const LongReal r_squared,
                                                      const vec3<LongReal>& r_ij,
                                                      const unsigned int type_i,
                                                      const quat<LongReal>& q_i,
                                                      const LongReal charge_i,
                                                      const unsigned int type_j,
                                                      const quat<LongReal>& q_j,
                                                      const LongReal charge_j) const

        {
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
    };

    } // end namespace hpmc
    } // end namespace hoomd
