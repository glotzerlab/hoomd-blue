// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "hoomd/HOOMDMath.h"
#include "hoomd/SystemDefinition.h"

namespace hoomd
    {
namespace hpmc
    {
/*** Functor that computes pair interactions between particles

    PairPotential allows energetic interactions to be included in an HPMC simulation. This
    abstract base class defines the API for the pair energy object, consisting of a cutoff radius
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
    one of the two or both at the same time.

    A PairPotential may be composed of another PairPotential (e.g. angular patches modulating a
   square well). Each PairPotential maintains a weak pointer to its parent. Only the r_cut of the
   top most parent is considered by IntegratorHPMC when computing interactions. Each PairPotential
   subclass should override the default computeRCutNonAdditive and computeRCutAdditive methods as
   needed to compute the various r_cut values as a function of parameters and child potentials. The
   base implementations return 0.

    The top level potential maintains cached values of the total and maximum r_cut values.
    Subclasses must call notifyRCutChanged whenever they would change the value of their computed
    r_cut values. The cached values ensure that we can use non-virtual inlined calls in the
    inner loops where the total r_cut values are checked.
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

    /// Returns the maximum non-additive r_cut beteween any two types.
    inline LongReal getMaxRCutNonAdditive() const
        {
        return m_max_r_cut_non_additive;
        }

    /// Set the parent potential
    void setParent(std::shared_ptr<PairPotential> parent)
        {
        m_parent = parent;

        if (auto parent = m_parent.lock())
            {
            parent->notifyRCutChanged();
            }
        }

    /*** Evaluate the energy of the pair interaction

        energy is given a pre-computed r_squared as many potentials use this parameter and the
        caller has already computed it.

        To avoid repeated evaluations of r_squared < r_cut_squared, the *caller* must perform the
        check before calling energy. Implementations of energy are free to compute non-zero values
        beyond r_cut when convenient.

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

    /// Compute the non-additive cuttoff radius
    virtual LongReal computeRCutNonAdditive(unsigned int type_i, unsigned int type_j) const
        {
        return 0;
        }

    /// Returns the additive part of the cutoff distance for a given type.
    virtual LongReal computeRCutAdditive(unsigned int type) const
        {
        return 0;
        }

    protected:
    /// The system definition.
    std::shared_ptr<SystemDefinition> m_sysdef;

    /// Indexer to access arrays by pairs of type parameters
    Index2D m_type_param_index;

    /// Notify all parents that r_cut has changed.
    void notifyRCutChanged()
        {
        if (auto parent = m_parent.lock())
            {
            parent->notifyRCutChanged();
            }

        updateRCutCache();
        }

    private:
    /// The non additive r_cut matrix (indexed by m_type_param_index).
    std::vector<LongReal> m_r_cut_non_additive;

    /// The additive r_cut values (indexed by type).
    std::vector<LongReal> m_r_cut_additive;

    /// Pre-computed total r_cut squared (indexed by m_type_param_index).
    std::vector<LongReal> m_r_cut_squared_total;

    /// Pre-computed maximum additive r_cut.
    LongReal m_max_r_cut_non_additive = 0;

    /// Parent potential
    std::weak_ptr<PairPotential> m_parent;

    /// Update r_cut cache
    void updateRCutCache()
        {
        const unsigned int n_types = m_sysdef->getParticleData()->getNTypes();

        for (unsigned int type = 0; type < n_types; type++)
            {
            m_r_cut_additive[type] = computeRCutAdditive(type);
            }

        m_max_r_cut_non_additive = 0;

        for (unsigned int type_i = 0; type_i < n_types; type_i++)
            {
            for (unsigned int type_j = type_i; type_j < n_types; type_j++)
                {
                LongReal r_cut_ij = computeRCutNonAdditive(type_i, type_j);
                m_max_r_cut_non_additive = std::max(m_max_r_cut_non_additive, r_cut_ij);

                unsigned int param_index_1 = m_type_param_index(type_i, type_j);
                unsigned int param_index_2 = m_type_param_index(type_j, type_i);
                m_r_cut_non_additive[param_index_1] = r_cut_ij;
                m_r_cut_non_additive[param_index_2] = r_cut_ij;

                LongReal r_cut
                    = m_r_cut_non_additive[param_index_1]
                      + LongReal(0.5) * (m_r_cut_additive[type_i] + m_r_cut_additive[type_j]);
                m_r_cut_squared_total[param_index_1] = r_cut * r_cut;
                m_r_cut_squared_total[param_index_2] = r_cut * r_cut;
                }
            }
        }
    };

    } // end namespace hpmc

    } // end namespace hoomd
