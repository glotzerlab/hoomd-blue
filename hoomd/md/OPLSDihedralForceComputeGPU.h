// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "OPLSDihedralForceCompute.h"
#include "OPLSDihedralForceGPU.cuh"
#include "hoomd/Autotuner.h"

/*! \file OPLSDihedralForceComputeGPU.h
    \brief Declares the OPLSDihedralForceComputeGPU class
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __OPLSDIHEDRALFORCECOMPUTEGPU_H__
#define __OPLSDIHEDRALFORCECOMPUTEGPU_H__

namespace hoomd
    {
namespace md
    {
//! Computes OPLS-style dihedral potentials on the GPU
/*! Calculates the OPLS type dihedral force on the GPU

    The GPU kernel for calculating this can be found in OPLSDihedralForceComputeGPU.cu
    \ingroup computes
*/
class PYBIND11_EXPORT OPLSDihedralForceComputeGPU : public OPLSDihedralForceCompute
    {
    public:
    //! Constructs the compute
    OPLSDihedralForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef);

    //! Destructor
    virtual ~OPLSDihedralForceComputeGPU() { }

    //! Set autotuner parameters
    /*! \param enable Enable/disable autotuning
        \param period period (approximate) in time steps when returning occurs
    */
    virtual void setAutotunerParams(bool enable, unsigned int period)
        {
        OPLSDihedralForceCompute::setAutotunerParams(enable, period);
        m_tuner->setPeriod(period);
        m_tuner->setEnabled(enable);
        }

    private:
    std::unique_ptr<Autotuner> m_tuner; //!< Autotuner for block size

    virtual void computeForces(uint64_t timestep);
    };

namespace detail
    {
//! Exports the OPLSDihedralForceComputeGPU class to python
void export_OPLSDihedralForceComputeGPU(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
