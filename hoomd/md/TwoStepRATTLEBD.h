// Copyright (c) 2009-2020 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenhoefer

#include "TwoStepLangevinBase.h"
#include "hoomd/Manifold.h"

#ifndef __TWO_STEP_RATTLE_BD_H__
#define __TWO_STEP_RATTLE_BD_H__

/*! \file TwoStepRATTLEBD.h
    \brief Declares the TwoStepRATTLEBD class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

//! Integrates part of the system forward in two steps with Brownian dynamics
/*! Implements RATTLE applied on Brownian dynamics.

    Brownian dynamics modifies the Langevin equation by setting the acceleration term to 0 and assuming terminal
    velocity.

    \ingroup updaters
*/
class PYBIND11_EXPORT TwoStepRATTLEBD : public TwoStepLangevinBase
    {
    public:
        //! Constructs the integration method and associates it with the system
        TwoStepRATTLEBD(std::shared_ptr<SystemDefinition> sysdef,
                    std::shared_ptr<ParticleGroup> group,
                    std::shared_ptr<Manifold> manifold,
                    std::shared_ptr<Variant> T,
                    unsigned int seed,
                    Scalar eta = 0.000001
                    );

        virtual ~TwoStepRATTLEBD();

        //! Performs the second step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

        //! Includes the RATTLE forces to the virial/net force
        virtual void IncludeRATTLEForce(unsigned int timestep);

        //! Get the number of degrees of freedom granted to a given group
        virtual Scalar getTranslationalDOF(std::shared_ptr<ParticleGroup> group);

        /// Sets eta
        void setEta(pybind11::object eta){ m_eta = pybind11::cast<Scalar>(eta); };

        /// Gets alpha
        pybind11::object getEta(){ return pybind11::cast(m_eta); };

    protected:
        std::shared_ptr<Manifold> m_manifold;  //!< The manifold used for the RATTLE constraint
        bool m_noiseless_t;
        bool m_noiseless_r;
        Scalar m_eta;                      //!< The eta value of the RATTLE algorithm, setting the tolerance to the manifold

        GPUArray<Scalar3> m_f_brownian; //! Brownian random force
    };

//! Exports the TwoStepLangevin class to python
void export_TwoStepRATTLEBD(pybind11::module& m);

#endif // #ifndef __TWO_STEP_RATTLE_BD_H__
