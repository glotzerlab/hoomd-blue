// Copyright (c) 2009-2025 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/ReverseNonequilibriumShearFlowGPU.h
 * \brief Declaration of reverse nonequilibrium shear flow updater on the GPU.
 */

#ifndef MPCD_REVERSE_NONEQUILIBRIUM_SHEAR_FLOW_GPU_H_
#define MPCD_REVERSE_NONEQUILIBRIUM_SHEAR_FLOW_GPU_H_

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include "ReverseNonequilibriumShearFlow.h"
#include "hoomd/Autotuner.h"
#include "hoomd/GPUFlags.h"
#include <pybind11/pybind11.h>

namespace hoomd
    {
namespace mpcd
    {
//! Reverse nonequilibrium shear flow updater on GPU
class PYBIND11_EXPORT ReverseNonequilibriumShearFlowGPU : public ReverseNonequilibriumShearFlow
    {
    public:
    //! Constructor
    ReverseNonequilibriumShearFlowGPU(std::shared_ptr<SystemDefinition> sysdef,
                                      std::shared_ptr<Trigger> trigger,
                                      unsigned int num_swap,
                                      Scalar slab_width,
                                      Scalar target_momentum);

    protected:
    //! Find candidate particles for swapping
    void findSwapParticles() override;

    //! Sort and copy only the top candidates for swapping
    void sortOutSwapParticles() override;

    //! Swap particle momenta
    void swapParticleMomentum() override;

    private:
    GPUArray<unsigned int> m_num_lo_hi;
    GPUFlags<Scalar> m_momentum_sum;
    std::shared_ptr<hoomd::Autotuner<1>> m_tuner_filter; //!< Autotuner for filtering particles
    std::shared_ptr<hoomd::Autotuner<1>> m_tuner_swap;   //!< Autotuner for swapping particles
    };
    } // end namespace mpcd
    } // end namespace hoomd
#endif // MPCD_REVERSE_NONEQUILIBRIUM_SHEAR_FLOW_GPU_H_
