// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

// inclusion guard
#ifndef _UPDATER_HPMC_GRID_SHIFT_GPU_
#define _UPDATER_HPMC_GRID_SHIFT_GPU_

/*! \file UpdaterGridShiftGPU.h
    \brief Declaration of UpdaterGridShiftGPU
*/

#include "hoomd/RandomNumbers.h"

#include "UpdaterGridShift.h"
#include "IntegratorHPMC.h"

#ifndef __HIPCC__
#include <pybind11/pybind11.h>
#endif

namespace hpmc
{

//! Shift the cell grid around randomly in simulations on the GPU or with MPI
class UpdaterGridShiftGPU : public UpdaterGridShift
    {
    public:
        //! Constructor
        /*! \param sysdef System definitino
         * \param mc MC integrator
         * \param seed RNG seed
         */
        UpdaterGridShiftGPU(std::shared_ptr<SystemDefinition> sysdef,
                      std::shared_ptr<IntegratorHPMC> mc,
                      const unsigned int seed,
                      std::shared_ptr<RandomTrigger> trigger);

        //! Destructor
        virtual ~UpdaterGridShiftGPU();

        //! Take one timestep forward
        /*! \param timestep timestep at which update is being evaluated
        */
        virtual void update(unsigned int timestep);
    };

//! Export UpdaterGridShiftGPU to Python
void export_UpdaterGridShiftGPU(pybind11::module& m);

} // end namespace hpmc

#endif // _UPDATER_HPMC_GRID_SHIFT_GPU_
