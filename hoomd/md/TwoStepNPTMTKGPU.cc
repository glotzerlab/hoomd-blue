// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TwoStepNPTMTKGPU.h"
#include "TwoStepNPTBaseGPU.cuh"

#include "TwoStepNPTBase.h"
#include "TwoStepNVEGPU.cuh"

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#include "hoomd/HOOMDMPI.h"
#endif

using namespace std;

/*! \file TwoStepNPTMTKGPU.h
    \brief Contains code for the TwoStepNPTMTKGPU class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef SystemDefinition this method will act on. Must not be NULL.
    \param group The group of particles this integration method is to work on
    \param thermo_group ComputeThermo to compute thermo properties of the integrated \a group
    \param thermo_group ComputeThermo to compute thermo properties of the integrated \a group at
   full time step \param tau NPT temperature period \param tauS NPT pressure period \param T
   Temperature set point \param S Pressure or Stress set point. Pressure if one value, Stress if a
   list of 6. Stress should be ordered as [xx, yy, zz, yz, xz, xy] \param couple Coupling mode
    \param flags Barostatted simulation box degrees of freedom
*/
TwoStepNPTMTKGPU::TwoStepNPTMTKGPU(std::shared_ptr<SystemDefinition> sysdef,
                                   std::shared_ptr<ParticleGroup> group,
                                   std::shared_ptr<ComputeThermo> thermo_group,
                                   std::shared_ptr<ComputeThermo> thermo_group_t,
                                   Scalar tau,
                                   Scalar tauS,
                                   std::shared_ptr<Variant> T,
                                   const std::vector<std::shared_ptr<Variant>>& S,
                                   const std::string& couple,
                                   const std::vector<bool>& flags,
                                   const bool nph)

    : TwoStepNPTBase(sysdef, group, thermo_group, thermo_group_t, T, S, couple, flags, nph),
      TwoStepNPTMTK(sysdef,group,thermo_group,thermo_group_t,tau,tauS,T,S,couple,flags,nph),
      TwoStepNPTBaseGPU(sysdef, group, thermo_group, thermo_group_t, T, S, couple, flags, nph)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        throw std::runtime_error("Cannot create TwoStepNPTMTKGPU on a CPU device.");
        }

    m_exec_conf->msg->notice(5) << "Constructing TwoStepNPTMTKGPU" << endl;

    }

TwoStepNPTMTKGPU::~TwoStepNPTMTKGPU()
    {
    m_exec_conf->msg->notice(5) << "Destroying TwoStepNPTMTKGPU" << endl;
    }


namespace detail
    {
void export_TwoStepNPTMTKGPU(pybind11::module& m)
    {
    pybind11::class_<TwoStepNPTMTKGPU, TwoStepNPTMTK, std::shared_ptr<TwoStepNPTMTKGPU>>(
        m,
        "TwoStepNPTMTKGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<ParticleGroup>,
                            std::shared_ptr<ComputeThermo>,
                            std::shared_ptr<ComputeThermo>,
                            Scalar,
                            Scalar,
                            std::shared_ptr<Variant>,
                            const std::vector<std::shared_ptr<Variant>>&,
                            const std::string&,
                            const std::vector<bool>&,
                            const bool>());
    }
    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
