// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "IntegrationMethodTwoStep.h"
#include "hoomd/Variant.h"

#pragma once

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

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
                            std::shared_ptr<Variant> T,
                            unsigned int seed);
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
        void setGamma(unsigned int typ, Scalar gamma);

        /// Gets gamma for a given particle type
        Scalar getGamma(unsigned int typ);

        /** Sets gamma_r for a given particle type
            @param typ Particle type to set gamma_r
            @param gamma The gamma_r value to set (a 3-tuple)
        */
        void setGammaR(unsigned int typ, pybind11::tuple v);

        /// Gets gamma_r for a given particle type
        pybind11::tuple getGammaR(unsigned int typ);

        /// Sets lambda
        void setLambda(pybind11::object lambda);

        /// Gets lambda
        pybind11::object getLambda();

        /// Get the seed
        unsigned int getSeed()
            {
            return m_seed;
            }

    protected:
        /// The Temperature of the Stochastic Bath
        std::shared_ptr<Variant> m_T;

        /// The seed for the RNG of the Stochastic Bath
        unsigned int m_seed;

        /// flag to enable gamma to be a scaled version of the diameter
        bool m_use_lambda;

        /// Scale factor to apply to diameter to get gamma
        Scalar m_lambda;

        /// List of per type gammas to use when m_use_lambda=false
        GlobalVector<Scalar> m_gamma;

        /// List of per type gamma_r (for 2D-only rotational noise) to use
        GlobalVector<Scalar3> m_gamma_r;

        /// Method to be called when number of types changes
        virtual void slotNumTypesChange();
    };

//! Exports the TwoStepLangevinBase class to python
void export_TwoStepLangevinBase(pybind11::module& m);
