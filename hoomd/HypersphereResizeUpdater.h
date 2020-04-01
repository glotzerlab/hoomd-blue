// Copyright (c) 2009-2020 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: pschoenh

/*! \file SphereResizeUpdater.h
    \brief Declares an updater that resizes the simulation sphere of the system
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "Updater.h"
#include "Variant.h"

#include <memory>
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#pragma once

//! Updates the simulation hypersphere over time, for hyperspherical simulations
/*! This simple updater gets the hypersphere radius from a specified variant and sets the hypersphere size
    over time. Particles always remain on the hypersphere.

    \ingroup updaters
*/
class PYBIND11_EXPORT HypersphereResizeUpdater : public Updater
    {
    public:
        //! Constructor
        HypersphereResizeUpdater(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<Variant> R);

        //! Destructor
        virtual ~HypersphereResizeUpdater();

        //! Take one timestep forward
        virtual void update(unsigned int timestep);

    private:
        std::shared_ptr<Variant> m_R;    //!< Hypersphere radius vs time
    };

//! Export the HypersphereResizeUpdater to python
void export_HypersphereResizeUpdater(pybind11::module& m);
