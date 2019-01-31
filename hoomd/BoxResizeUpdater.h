// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: joaander

/*! \file BoxResizeUpdater.h
    \brief Declares an updater that resizes the simulation box of the system
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include "Updater.h"
#include "Variant.h"

#include <memory>
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __BOXRESIZEUPDATER_H__
#define __BOXRESIZEUPDATER_H__

//! Updates the simulation box over time
/*! This simple updater gets the box lengths from specified variants and sets those box sizes
    over time. As an option, particles can be rescaled with the box lengths or left where they are.

    \ingroup updaters
*/
class PYBIND11_EXPORT BoxResizeUpdater : public Updater
    {
    public:
        //! Constructor
        BoxResizeUpdater(std::shared_ptr<SystemDefinition> sysdef,
                         std::shared_ptr<Variant> Lx,
                         std::shared_ptr<Variant> Ly,
                         std::shared_ptr<Variant> Lz,
                         std::shared_ptr<Variant> xy,
                         std::shared_ptr<Variant> xz,
                         std::shared_ptr<Variant> yz);

        //! Destructor
        virtual ~BoxResizeUpdater();

        //! Sets parameter flags
        void setParams(bool scale_particles);

        //! Take one timestep forward
        virtual void update(unsigned int timestep);

    private:
        std::shared_ptr<Variant> m_Lx;    //!< Box Lx vs time
        std::shared_ptr<Variant> m_Ly;    //!< Box Ly vs time
        std::shared_ptr<Variant> m_Lz;    //!< Box Lz vs time
        std::shared_ptr<Variant> m_xy;    //!< Box xy tilt factor vs time
        std::shared_ptr<Variant> m_xz;    //!< Box xz tilt factor vs time
        std::shared_ptr<Variant> m_yz;    //!< Box yz tilt factor vs time
        bool m_scale_particles;                //!< Set to true if particle positions are to be scaled as well
    };

//! Export the BoxResizeUpdater to python
void export_BoxResizeUpdater(pybind11::module& m);

#endif
