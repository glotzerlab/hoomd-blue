// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "Trigger.h"
#include "Updater.h"

#ifndef __TUNER_H__
#define __TUNER_H__

/*! \file Updater.h
    \brief Declares a base class for all updaters
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

/*! \ingroup hoomd_lib
    @{
*/

/*! \defgroup tuners Tuners
    \brief All classes that implement the Tuner concept.
    \details See \ref page_dev_info for more information
*/

/*! @}
 */

namespace hoomd
    {
//! Performs updates of performance critical parameters and data storage.
/*! Operations that do work to improve simulation performance like particle
 * sorters and GPU autotuners are system Tuners.
 * See \ref page_dev_info for more information
    \ingroup tuners
*/
class PYBIND11_EXPORT Tuner : public Updater
    {
    public:
    //! Constructs the compute and associates it with the ParticleData
    Tuner(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<Trigger> trigger);
    virtual ~Tuner() { };
    };

namespace detail
    {
//! Export the Updater class to python
void export_Tuner(pybind11::module& m);

    } // end namespace detail

    } // end namespace hoomd

#endif
