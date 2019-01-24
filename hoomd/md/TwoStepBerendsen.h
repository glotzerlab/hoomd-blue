// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "IntegrationMethodTwoStep.h"
#include "hoomd/Variant.h"
#include "hoomd/ComputeThermo.h"

// inclusion guard
#ifndef __BERENDSEN_H__
#define __BERENDSEN_H__

/*! \file TwoStepBerendsen.h
    \brief Declaration of Berendsen thermostat
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

/*! Implements the Berendsen thermostat \cite Berendsen1984
*/
class PYBIND11_EXPORT TwoStepBerendsen : public IntegrationMethodTwoStep
    {
    public:
        //! Constructor
        TwoStepBerendsen(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<ParticleGroup> group,
                         std::shared_ptr<ComputeThermo> thermo,
                         Scalar tau,
                         std::shared_ptr<Variant> T);
        virtual ~TwoStepBerendsen();

        //! Update the temperature
        //! \param T New temperature to set
        virtual void setT(std::shared_ptr<Variant> T)
            {
            m_T = T;
            }

        //! Update the tau value
        //! \param tau New time constant to set
        virtual void setTau(Scalar tau)
            {
            m_tau = tau;
            }

        //! Performs the first step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

    protected:
        const std::shared_ptr<ComputeThermo> m_thermo; //!< compute for thermodynamic quantities
        Scalar m_tau;                    //!< time constant for Berendsen thermostat
        std::shared_ptr<Variant> m_T;    //!< set temperature
        bool m_warned_aniso;                             //!< true if we've already warned that we don't support aniso
    };

//! Export the Berendsen class to python
void export_Berendsen(pybind11::module& m);

#endif // _BERENDSEN_H_
