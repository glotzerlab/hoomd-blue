// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "PairPotential.h"

namespace hoomd
    {
namespace hpmc
    {

/*** Compute step function energies between two particles.

For use with HPMC simulations.
*/
class PairPotentialStep : public hpmc::PairPotential
    {
    public:
    PairPotentialStep(std::shared_ptr<SystemDefinition> sysdef);
    virtual ~PairPotentialStep() { }

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

    /// Set type pair dependent parameters to the potential.
    void setParamsPython(pybind11::tuple typ, pybind11::object params);

    /// Get type pair dependent parameters.
    pybind11::dict getParamsPython(pybind11::tuple typ);

    protected:
    /// Type pair parameters of LJ potential
    struct ParamType
        {
        ParamType() { }

        /// Construct a parameter set from a dictionary.
        ParamType(pybind11::object v);

        /// Convert a parameter set to a dictionary.
        pybind11::dict asDict();

        std::vector<LongReal> m_epsilon;
        std::vector<LongReal> m_r_squared;
        };

    /// Parameters per type pair.
    std::vector<ParamType> m_params;
    };

    } // end namespace hpmc
    } // end namespace hoomd
