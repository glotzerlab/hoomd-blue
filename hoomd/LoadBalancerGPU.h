// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: mphoward

/*! \file LoadBalancerGPU.h
    \brief Declares an updater that changes the MPI domain decomposition to balance the load using the GPU
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifdef ENABLE_MPI
#ifdef ENABLE_CUDA

#ifndef __LOADBALANCERGPU_H__
#define __LOADBALANCERGPU_H__

#include "HOOMDMath.h"
#include "GPUFlags.h"
#include "LoadBalancer.h"
#include "Autotuner.h"
#include <hoomd/extern/nano-signal-slot/nano_signal_slot.hpp>
#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

//! GPU implementation of dynamic load balancing
class PYBIND11_EXPORT LoadBalancerGPU : public LoadBalancer
    {
    public:
        //! Constructor
        LoadBalancerGPU(std::shared_ptr<SystemDefinition> sysdef,
                        std::shared_ptr<DomainDecomposition> decomposition);

        //! Destructor
        virtual ~LoadBalancerGPU();

        //! Set autotuner parameters
        /*!
         * \param enable Enable/disable autotuning
         * \param period period (approximate) in time steps when returning occurs
         */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            LoadBalancer::setAutotunerParams(enable, period);
            m_tuner->setPeriod(period);
            m_tuner->setEnabled(enable);
            }

        //! Resize the per particle data when there is a max number of particle change
        void slotMaxNumChanged()
            {
            GPUArray<unsigned int> off_ranks(m_pdata->getMaxN(), m_exec_conf);
            m_off_ranks.swap(off_ranks);
            }
    protected:
        //! Count the number of particles that have gone off either edge of the rank along a dimension on the GPU
        virtual void countParticlesOffRank(std::map<unsigned int, unsigned int>& cnts);

    private:
        std::unique_ptr<Autotuner> m_tuner;   //!< Autotuner for block size counting particles
        GPUArray<unsigned int> m_off_ranks;     //!< Array to hold the ranks of particles that have moved
    };

//! Export the LoadBalancerGPU to python
void export_LoadBalancerGPU(pybind11::module& m);

#endif // __LOADBALANCERGPU_H__

#endif // ENABLE_CUDA
#endif // ENABLE_MPI
