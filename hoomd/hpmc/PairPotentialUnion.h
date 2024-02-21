// Copyright (c) 2009-2023 The Regents of the University of Michigan.
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
    virtual LongReal computeRCutNonAdditive(unsigned int type_i, unsigned int type_j) const
        {
        // Pass on the non-additive r_cut from the constituent potential.
        return m_constituent_potential->computeRCutNonAdditive(type_i, type_j);
        }

    /// Returns the additive part of the cutoff distance for a given type.
    virtual LongReal computeRCutAdditive(unsigned int type) const
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
    LongReal energyAll(const LongReal r_squared,
                       const vec3<LongReal>& r_ij,
                       const unsigned int type_i,
                       const quat<LongReal>& q_i,
                       const LongReal charge_i,
                       const unsigned int type_j,
                       const quat<LongReal>& q_j,
                       const LongReal charge_j) const;

    /*** Evaluate the energy of the pair interaction (using the OBB tree).
     */
    LongReal energyOBB(const LongReal r_squared,
                       const vec3<LongReal>& r_ij,
                       const unsigned int type_i,
                       const quat<LongReal>& q_i,
                       const LongReal charge_i,
                       const unsigned int type_j,
                       const quat<LongReal>& q_j,
                       const LongReal charge_j) const;
    };

    } // end namespace hpmc
    } // end namespace hoomd
