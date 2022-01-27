// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "MeshVolumeConservation.h"
#include "MeshVolumeConservationGPU.cuh"
#include "hoomd/Autotuner.h"

#include <memory>

/*! \file MeshVolumeConservationGPU.h
    \brief Declares a class for computing volume constraint forces on the GPU
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#ifndef __MESHVOLUMECONSERVATION_GPU_H__
#define __MESHVOLUMECONSERVATION_GPU_H__

namespace hoomd
    {
namespace md
    {

//! Computes helfrich energy forces on the mesh on the GPU
/*! Helfrich energy forces are computed on every particle in a mesh.

    \ingroup computes
*/
class PYBIND11_EXPORT MeshVolumeConservationGPU : public MeshVolumeConservation
    {
    public:
    //! Constructs the compute
    MeshVolumeConservationGPU(std::shared_ptr<SystemDefinition> sysdef,
                              std::shared_ptr<MeshDefinition> meshdef);

    //! Set autotuner parameters
    /*! \param enable Enable/disable autotuning
        \param period period (approximate) in time steps when returning occurs
    */
    virtual void setAutotunerParams(bool enable, unsigned int period)
        {
        MeshVolumeConservation::setAutotunerParams(enable, period);
        m_tuner_force->setPeriod(period);
        m_tuner_force->setEnabled(enable);
        m_tuner_volume->setPeriod(period);
        m_tuner_volume->setEnabled(enable);
        }

    //! Set the parameters
    virtual void setParams(unsigned int type, Scalar K, Scalar V0);

    protected:
    std::unique_ptr<Autotuner> m_tuner_force;  //!< Autotuner for block size of force loop
    std::unique_ptr<Autotuner> m_tuner_volume; //!< Autotuner for block size of volume loop
    GPUArray<unsigned int> m_flags;            //!< Flags set during the kernel execution
    GPUArray<Scalar> m_params;                 //!< Parameters stored on the GPU

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

namespace detail
    {
//! Exports the MeshVolumeConservationGPU class to python
void export_MeshVolumeConservationGPU(pybind11::module& m);

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
