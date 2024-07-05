// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "IntegrationMethodTwoStep.h"
#include "hoomd/Variant.h"

#pragma once

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace md
    {
/** Base class for Langevin equation based integration method

    HOOMD implements Langevin dynamics and Brownian dynamics. Both are based on the same equation of
    motion, but the latter assumes an overdamped regime while the former assumes underdamped. This
    base class store and manages the data structures and settings that are common to the two of
    them, including temperature, seed, and gamma.
*/
class PYBIND11_EXPORT TwoStepLangevinBase : public IntegrationMethodTwoStep
    {
    public:
    /// Constructs the integration method and associates it with the system
    TwoStepLangevinBase(std::shared_ptr<SystemDefinition> sysdef,
                        std::shared_ptr<ParticleGroup> group,
                        std::shared_ptr<Variant> T);
    virtual ~TwoStepLangevinBase();

    /** Set a new temperature
        @param T new temperature to set
    */
    void setT(std::shared_ptr<Variant> T)
        {
        m_T = T;
        }

    /// Get the current temperature variant
    std::shared_ptr<Variant> getT()
        {
        return m_T;
        }

    /** Sets gamma for a given particle type
        @param typ Particle type to set gamma for
        @param gamma The gamma value to set
    */
    void setGamma(const std::string& type_name, Scalar gamma);

    /// Gets gamma for a given particle type
    Scalar getGamma(const std::string& type_name);

    /** Sets gamma_r for a given particle type
        @param typ Particle type to set gamma_r
        @param gamma The gamma_r value to set (a 3-tuple)
    */
    void setGammaR(const std::string& type_name, pybind11::tuple v);

    /// Gets gamma_r for a given particle type
    pybind11::tuple getGammaR(const std::string& type_name);

    //! Return true if the method is momentum conserving
    virtual bool isMomentumConserving() const
        {
        return false;
        }

    protected:
    /// The Temperature of the Stochastic Bath
    std::shared_ptr<Variant> m_T;

    /// List of per type gammas
    GlobalVector<Scalar> m_gamma;

    /// List of per type gamma_r (for 2D-only rotational noise) to use
    GlobalVector<Scalar3> m_gamma_r;
    };

    } // end namespace md
    } // end namespace hoomd
