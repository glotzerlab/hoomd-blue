// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "BondTablePotential.h"
#include "BondTablePotentialGPU.cuh"
#include "hoomd/Autotuner.h"

/*! \file BondTablePotentialGPU.h
    \brief Declares the BondTablePotentialGPU class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __BONDTABLEPOTENTIALGPU_H__
#define __BONDTABLEPOTENTIALGPU_H__

namespace hoomd
    {
namespace md
    {
//! Compute table based bond potentials on the GPU
/*! Calculates exactly the same thing as BondTablePotential, but on the GPU

    The GPU kernel for calculating this can be found in BondTablePotentialGPU.cu/
    \ingroup computes
*/
class PYBIND11_EXPORT BondTablePotentialGPU : public BondTablePotential
    {
    public:
    //! Constructs the compute
    BondTablePotentialGPU(std::shared_ptr<SystemDefinition> sysdef, unsigned int table_width);

    //! Destructor
    virtual ~BondTablePotentialGPU();

    //! Set autotuner parameters
    /*! \param enable Enable/disable autotuning
        \param period period (approximate) in time steps when returning occurs
    */
    virtual void setAutotunerParams(bool enable, unsigned int period)
        {
        BondTablePotential::setAutotunerParams(enable, period);
        m_tuner->setPeriod(period);
        m_tuner->setEnabled(enable);
        }

    private:
    std::unique_ptr<Autotuner> m_tuner; //!< Autotuner for block size
    GPUArray<unsigned int> m_flags;     //!< Flags set during the kernel execution

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };
namespace detail
    {
//! Exports the BondTablePotentialGPU class to python
void export_BondTablePotentialGPU(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
