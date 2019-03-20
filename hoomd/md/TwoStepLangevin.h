// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

#include "TwoStepLangevinBase.h"

#ifndef __TWO_STEP_LANGEVIN_H__
#define __TWO_STEP_LANGEVIN_H__

/*! \file TwoStepLangevin.h
    \brief Declares the TwoStepLangevin class
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

//! Integrates part of the system forward in two steps with Langevin dynamics
/*! Implements Langevin dynamics.

    Langevin dynamics modifies standard NVE integration with two additional forces, a random force and a drag force.
    This implementation is very similar to TwoStepNVE with the additional forces. Note that this is not a really proper
    Langevin integrator, but it works well in practice.

    \ingroup updaters
*/
class PYBIND11_EXPORT TwoStepLangevin : public TwoStepLangevinBase
    {
    public:
        //! Constructs the integration method and associates it with the system
        TwoStepLangevin(std::shared_ptr<SystemDefinition> sysdef,
                     std::shared_ptr<ParticleGroup> group,
                     std::shared_ptr<Variant> T,
                     unsigned int seed,
                     bool use_lambda,
                     Scalar lambda,
                     bool noiseless_t,
                     bool noiseless_r,
                     const std::string& suffix = std::string(""));
        virtual ~TwoStepLangevin();

        //! Turn on or off Tally
        /*! \param tally if true, tallies energy exchange from the thermal reservoir */
        void setTally(bool tally)
            {
            m_tally= tally;
            }

        //! Returns a list of log quantities this integrator calculates
        virtual std::vector< std::string > getProvidedLogQuantities();

        //! Returns logged values
        Scalar getLogValue(const std::string& quantity, unsigned int timestep, bool &my_quantity_flag);

        //! Performs the second step of the integration
        virtual void integrateStepOne(unsigned int timestep);

        //! Performs the second step of the integration
        virtual void integrateStepTwo(unsigned int timestep);

    protected:
        Scalar m_reservoir_energy;         //!< The energy of the reservoir the system is coupled to.
        Scalar m_extra_energy_overdeltaT;  //!< An energy packet that isn't added until the next time step
        bool m_tally;                      //!< If true, changes to the energy of the reservoir are calculated
        std::string m_log_name;            //!< Name of the reservoir quantity that we log
        bool m_noiseless_t;                //!< If set true, there will be no translational noise (random force)
        bool m_noiseless_r;                //!< If set true, there will be no rotational noise (random torque)
    };

//! Exports the TwoStepLangevin class to python
void export_TwoStepLangevin(pybind11::module& m);

#endif // #ifndef __TWO_STEP_LANGEVIN_H__
