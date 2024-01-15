// Copyright (c) 2009-2023 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "hoomd/HOOMDMath.h"
#include "hoomd/SystemDefinition.h"

namespace hoomd
    {
namespace hpmc
    {
/*** Functor that computes pair interactions between particles

    PairPotential allows cutoff energetic interactions to be included in an HPMC simulation. This
    abstract base class defines the API for the pair energy object, consisting of cutoff radius
    and the pair energy evaluation fuction.

    Provide a PairPotential instance to IntegratorHPMC. The pairwise potential energy will be
    evaluated when needed during the HPMC trial moves. IntegratorHPMC uses a spatial data
    structure to call energy() on only the pairs within a given distance r_cut. Beyond the
    cutoff distance, the interaction energy is 0.

    There are two components to r_cut.
        1) A non-additive value r_cut_non_additive_{i,j} that depends
           on the type ids of the particles i and j.
        2) An additive value 0.5 * (r_cut_additive_i + r_cut_additive_j) that checks whether the
           spheres about each particle in the pair overlap.

    HPMC always always uses both r_cuts for every potential:

        r_cut = r_cut_non_additive_{i,j} + 0.5 * (r_cut_additive_i + r_cut_additive_j)

    Of the two, the additive r_cut provides better performance when particle types have different
    sizes. The non-additive r_cut allows more freedom to implement models. Subclasses can use
    one of the two or both at the same time. r_cut_additive and r_cut_non_additive default to all
    0's - subclasses must call setRCutNonAdditive and/or setRCutAdditive to update the arrays.
*/
class PairPotential
    {
    public:
    PairPotential(std::shared_ptr<SystemDefinition> sysdef)
        : m_sysdef(sysdef), m_type_param_index(sysdef->getParticleData()->getNTypes()),
          m_r_cut_non_additive(m_type_param_index.getNumElements(), 0),
          m_r_cut_additive(sysdef->getParticleData()->getNTypes(), 0),
          m_r_cut_squared_total(m_type_param_index.getNumElements(), 0)
        {
        }
    virtual ~PairPotential() { }

    /// Returns the non-additive cutoff radius.
    inline LongReal getRCutNonAdditive(unsigned int type_i, unsigned int type_j) const
        {
        return m_r_cut_non_additive[m_type_param_index(type_i, type_j)];
        }

    /// Returns the additive part of the cutoff distance for a given type.
    inline LongReal getRCutAdditive(unsigned int type) const
        {
        return m_r_cut_additive[type];
        }

    /// Returns the total cutoff radius squared.
    inline LongReal getRCutSquaredTotal(unsigned int type_i, unsigned int type_j) const
        {
        return m_r_cut_squared_total[m_type_param_index(type_i, type_j)];
        }

    /*** Evaluate the energy of the pair interaction

        energy is given a pre-computed r_squared as many potentials use this parameter and the
        caller has already computed it.

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
                            const LongReal charge_j) const
        {
        return 0;
        }

    protected:
    /// The system definition.
    std::shared_ptr<SystemDefinition> m_sysdef;

    /// Indexer to access arrays by pairs of type parameters
    Index2D m_type_param_index;

    /// Set the non-additive cutoff radius for both i,j and j,i pairs.
    void setRCutNonAdditive(unsigned int type_i, unsigned int type_j, LongReal r)
        {
        m_r_cut_non_additive[m_type_param_index(type_i, type_j)] = r;
        m_r_cut_non_additive[m_type_param_index(type_j, type_i)] = r;

        updateRCutTotal(type_i, type_j);
        }

    /// Returns the additive part of the cutoff distance for a given type.
    void setRCutAdditive(unsigned int type, LongReal r)
        {
        m_r_cut_additive[type] = r;

        for (unsigned int type_j = 0; type_j < m_sysdef->getParticleData()->getNTypes(); type_j++)
            {
            updateRCutTotal(type, type_j);
            }
        }

    private:
    /// The non additive r_cut matrix (indexed by m_type_param_index).
    std::vector<LongReal> m_r_cut_non_additive;

    /// The additive r_cut values (indexed by type).
    std::vector<LongReal> m_r_cut_additive;

    /// Pre computed total r_cut squared (indexed by m_type_param_index).
    std::vector<LongReal> m_r_cut_squared_total;

    /// Update the pre-computed r_cut_total value.
    void updateRCutTotal(unsigned int type_i, unsigned int type_j)
        {
        unsigned int type_pair = m_type_param_index(type_i, type_j);
        LongReal r_cut = m_r_cut_non_additive[type_pair] + LongReal(0.5) * (m_r_cut_additive[type_i]
                         + m_r_cut_additive[type_j]);
        m_r_cut_squared_total[type_pair] = r_cut * r_cut;

        type_pair = m_type_param_index(type_j, type_i);
        m_r_cut_squared_total[type_pair] = r_cut * r_cut;
        }
    };

    } // end namespace hpmc

    } // end namespace hoomd
