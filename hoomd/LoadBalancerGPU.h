// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file LoadBalancerGPU.h
    \brief Declares an updater that changes the MPI domain decomposition to balance the load using
   the GPU
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifdef ENABLE_HIP

#pragma once

#include "Autotuner.h"
#include "GPUFlags.h"
#include "HOOMDMath.h"
#include "LoadBalancer.h"
#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>
#include <pybind11/pybind11.h>

namespace hoomd
    {
//! GPU implementation of dynamic load balancing
class PYBIND11_EXPORT LoadBalancerGPU : public LoadBalancer
    {
    public:
    //! Constructor
    LoadBalancerGPU(std::shared_ptr<SystemDefinition> sysdef, std::shared_ptr<Trigger> trigger);

    //! Destructor
    virtual ~LoadBalancerGPU();

    //! Resize the per particle data when there is a max number of particle change
    void slotMaxNumChanged()
        {
        GPUArray<unsigned int> off_ranks(m_pdata->getMaxN(), m_exec_conf);
        m_off_ranks.swap(off_ranks);
        }

    protected:
#ifdef ENABLE_MPI
    //! Count the number of particles that have gone off either edge of the rank along a dimension
    //! on the GPU
    virtual void countParticlesOffRank(std::map<unsigned int, unsigned int>& cnts);
#endif

    private:
    /// Autotuner for block size counting particles
    std::shared_ptr<Autotuner<1>> m_tuner;

    /// Array to hold the ranks of particles that have moved
    GPUArray<unsigned int> m_off_ranks;
    };

namespace detail
    {
//! Export the LoadBalancerGPU to python
void export_LoadBalancerGPU(pybind11::module& m);

    } // end namespace detail

    } // end namespace hoomd

#endif // ENABLE_HIP
