// Copyright (c) 2009-2016 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "TwoStepLangevinBase.h"

#ifndef __TWO_STEP_BD_H__
#define __TWO_STEP_BD_H__

/*! \file TwoStepLangevin.h
    \brief Declares the TwoStepLangevin class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! Integrates part of the system forward in two steps with Brownian dynamics
/*! Implements Brownian dynamics.

    Brownian dynamics modifies the Langevin equation by setting the acceleration term to 0 and assuming terminal
    velocity.

    \ingroup updaters
*/
class TwoStepBD : public TwoStepLangevinBase
    {
    public:
        //! Constructs the integration method and associates it with the system
        TwoStepBD(boost::shared_ptr<SystemDefinition> sysdef,
                    boost::shared_ptr<ParticleGroup> group,
                    boost::shared_ptr<Variant> T,
                    unsigned int seed,
                    bool use_lambda,
                    Scalar lambda,
                    bool noiseless_t,
                    bool noiseless_r
                    );
        
        virtual ~TwoStepBD();

        //! Performs the second step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);
        
    protected:
        bool m_noiseless_t;
        bool m_noiseless_r;
    };

//! Exports the TwoStepLangevin class to python
void export_TwoStepBD();

#endif // #ifndef __TWO_STEP_BD_H__
