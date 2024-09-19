// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "PairPotential.h"

namespace hoomd
    {
namespace hpmc
    {

/*** Compute Lennard-Jones energy between two particles.

For use with HPMC simulations.
*/
class PairPotentialOPP : public hpmc::PairPotential
    {
    public:
    PairPotentialOPP(std::shared_ptr<SystemDefinition> sysdef);
    virtual ~PairPotentialOPP() { }

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
        unsigned int param_index = m_type_param_index(type_i, type_j);
        return slow::sqrt(m_params[param_index].r_cut_squared);
        }

    /// Set type pair dependent parameters to the potential.
    void setParamsPython(pybind11::tuple typ, pybind11::dict params);

    /// Get type pair dependent parameters.
    pybind11::dict getParamsPython(pybind11::tuple typ);

    void setMode(const std::string& mode_str)
        {
        if (mode_str == "none")
            {
            m_mode = no_shift;
            }
        else if (mode_str == "shift")
            {
            m_mode = shift;
            }
        else if (mode_str == "xplor")
            {
            m_mode = xplor;
            }
        else
            {
            throw std::domain_error("Invalid mode " + mode_str);
            }
        }

    std::string getMode()
        {
        std::string result = "none";

        if (m_mode == shift)
            {
            result = "shift";
            }
        if (m_mode == xplor)
            {
            result = "xplor";
            }

        return result;
        }

    protected:
    /// Shifting modes that can be applied to the energy
    enum EnergyShiftMode
        {
        no_shift = 0,
        shift,
        xplor
        };

    /// Type pair parameters of LJ potential
    struct ParamType
        {
        ParamType()
            {
            C1=0;
            C2=0;
            eta1=0;
            eta2 = 0;
            k = 0;
            phi = 0;
            }

        ParamType(pybind11::dict v)
            {
            auto r_cut(v["r_cut"].cast<LongReal>());
            auto r_on(v["r_on"].cast<LongReal>());
            
            C1 = v["C1"].cast<LongReal>();
            C2 = v["C2"].cast<LongReal>();
            eta1 = v["eta1"].cast<LongReal>();
            eta2 = v["eta2"].cast<LongReal>();
            k = v["k"].cast<LongReal>();
            phi = v["phi"].cast<LongReal>();
            r_cut_squared = r_cut * r_cut;
            r_on_squared = r_on * r_on;
            }

        pybind11::dict asDict()
            {
            pybind11::dict result;

            result["C1"] = C1;
            result["C2"] = C2;
            result["eta1"] = eta1;
            result["eta2"] = eta2;
            result["k"] = k;
            result["phi"] = phi;
            result["r_cut"] = slow::sqrt(r_cut_squared);
            result["r_on"] = slow::sqrt(r_on_squared);

            return result;
            }

        LongReal C1;
        LongReal C2;
        LongReal eta1;
        LongReal eta2;
        LongReal k;
        LongReal phi;
        LongReal r_cut_squared;
        LongReal r_on_squared;
        };

    std::vector<ParamType> m_params;

    EnergyShiftMode m_mode = no_shift;
    };

    } // end namespace hpmc
    } // end namespace hoomd
