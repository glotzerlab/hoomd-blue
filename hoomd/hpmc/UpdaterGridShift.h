// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// inclusion guard
#ifndef _UPDATER_HPMC_GRID_SHIFT_
#define _UPDATER_HPMC_GRID_SHIFT_

/*! \file UpdaterGridShift.h
    \brief Declaration of UpdaterGridShift
*/

#include <hoomd/Updater.h>
#include "hoomd/RandomNumbers.h"

#include "IntegratorHPMC.h"
#include "RandomTrigger.h"

#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#endif

namespace hpmc
{

//! Shift the cell grid around randomly in simulations on the GPU or with MPI
class UpdaterGridShift : public Updater
    {
    public:
        //! Constructor
        /*! \param sysdef System definitino
         * \param mc MC integrator
         * \param seed RNG seed
         */
        UpdaterGridShift(std::shared_ptr<SystemDefinition> sysdef,
                      std::shared_ptr<IntegratorHPMC> mc,
                      const unsigned int seed,
                      std::shared_ptr<RandomTrigger> trigger);

        //! Destructor
        virtual ~UpdaterGridShift();

        //! Take one timestep forward
        /*! \param timestep timestep at which update is being evaluated
        */
        virtual void update(unsigned int timestep);

    protected:
        std::shared_ptr<IntegratorHPMC> m_mc;     //!< HPMC integrator object
        unsigned int m_seed;                      //!< Seed for pseudo-random number generator
        std::shared_ptr<RandomTrigger> m_trigger; //!< Random selection of MC moves
    };

//! Export UpdaterGridShift to Python
void export_UpdaterGridShift(pybind11::module& m);

} // end namespace hpmc

#endif // _UPDATER_HPMC_GRID_SHIFT_
