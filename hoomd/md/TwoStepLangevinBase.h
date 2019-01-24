// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "IntegrationMethodTwoStep.h"
#include "hoomd/Variant.h"

#ifndef __TWO_STEP_LANGEVIN_BASE__
#define __TWO_STEP_LANGEVIN_BASE__

/*! \file TwoStepLangevinBase.h
    \brief Declares the TwoStepLangevinBase class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

//! Base class for Langevin equation based integration method
/*! HOOMD implements Langevin dynamics and Brownian dynamics. Both are based on the same equation of motion, but the
    latter assumes an overdamped regime while the former assumes underdamped. This base class store and manages
    the data structures and settings that are common to the two of them, including temperature, seed, and gamma.

    \ingroup updaters
*/
class PYBIND11_EXPORT TwoStepLangevinBase : public IntegrationMethodTwoStep
    {
    public:
        //! Constructs the integration method and associates it with the system
        TwoStepLangevinBase(std::shared_ptr<SystemDefinition> sysdef,
                            std::shared_ptr<ParticleGroup> group,
                            std::shared_ptr<Variant> T,
                            unsigned int seed,
                            bool use_lambda,
                            Scalar lambda);
        virtual ~TwoStepLangevinBase();

        //! Set a new temperature
        /*! \param T new temperature to set */
        void setT(std::shared_ptr<Variant> T)
            {
            m_T = T;
            }

        //! Sets gamma for a given particle type
        void setGamma(unsigned int typ, Scalar gamma);

        void setGamma_r(unsigned int typ, Scalar3 gamma_r);

    protected:
        std::shared_ptr<Variant> m_T;   //!< The Temperature of the Stochastic Bath
        unsigned int m_seed;              //!< The seed for the RNG of the Stochastic Bath
        bool m_use_lambda;                //!< flag to enable gamma to be a scaled version of the diameter
        Scalar m_lambda;                  //!< Scale factor to apply to diameter to get gamma

        GlobalVector<Scalar> m_gamma;         //!< List of per type gammas to use
        GlobalVector<Scalar3> m_gamma_r;      //!< List of per type gamma_r (for 2D-only rotational noise) to use

        //! Method to be called when number of types changes
        virtual void slotNumTypesChange();
    };

//! Exports the TwoStepLangevinBase class to python
void export_TwoStepLangevinBase(pybind11::module& m);

#endif // #ifndef __TWO_STEP_LANGEVIN_BASE__
