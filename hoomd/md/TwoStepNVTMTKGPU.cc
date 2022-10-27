// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TwoStepNVTMTKGPU.h"
#include "TwoStepNPTBaseGPU.cuh"
#include "TwoStepNVEGPU.cuh"

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#include "hoomd/HOOMDMPI.h"
#endif

using namespace std;

/*! \file TwoStepNVTMTKGPU.h
    \brief Contains code for the TwoStepNVTMTKGPU class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param thermo compute for thermodynamic quantities
    \param tau NVT period
    \param T Temperature set point
*/
TwoStepNVTMTKGPU::TwoStepNVTMTKGPU(std::shared_ptr<SystemDefinition> sysdef,
                                   std::shared_ptr<ParticleGroup> group,
                                   std::shared_ptr<ComputeThermo> thermo,
                                   Scalar tau,
                                   std::shared_ptr<Variant> T)
    : TwoStepNVTBase(sysdef, group, thermo, T),
      TwoStepNVTMTK(sysdef, group, thermo, tau, T),
      TwoStepNVTBaseGPU(sysdef, group, thermo, T)
    {

    }

namespace detail
    {
void export_TwoStepNVTMTKGPU(pybind11::module& m)
    {
    pybind11::class_<TwoStepNVTMTKGPU, TwoStepNVTMTK, TwoStepNVTBaseGPU, std::shared_ptr<TwoStepNVTMTKGPU>>(
        m,
        "TwoStepNVTMTKGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<ComputeThermo>,
                            Scalar,
                            std::shared_ptr<Variant>>());
    }
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
