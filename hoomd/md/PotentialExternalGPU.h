// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "PotentialExternal.h"
#include "PotentialExternalGPU.cuh"
#include "hoomd/Autotuner.h"
#include <memory>

/*! \file PotentialExternalGPU.h
    \brief Declares a class for computing an external potential field on the GPU
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

#include <pybind11/pybind11.h>

#ifndef __POTENTIAL_EXTERNAL_GPU_H__
#define __POTENTIAL_EXTERNAL_GPU_H__

namespace hoomd
    {
namespace md
    {
//! Applys a constraint force to keep a group of particles on a sphere
/*! \ingroup computes
 */
template<class evaluator> class PotentialExternalGPU : public PotentialExternal<evaluator>
    {
    public:
    //! Constructs the compute
    PotentialExternalGPU(std::shared_ptr<SystemDefinition> sysdef);

    protected:
    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);

    std::shared_ptr<Autotuner<1>> m_tuner; //!< Autotuner for block size
    };

/*! Constructor
    \param sysdef system definition
 */
template<class evaluator>
PotentialExternalGPU<evaluator>::PotentialExternalGPU(std::shared_ptr<SystemDefinition> sysdef)
    : PotentialExternal<evaluator>(sysdef)
    {
    m_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                                   this->m_exec_conf,
                                   "external_" + evaluator::getName()));
    this->m_autotuners.push_back(m_tuner);
    }

/*! Computes the specified constraint forces
    \param timestep Current timestep
*/
template<class evaluator> void PotentialExternalGPU<evaluator>::computeForces(uint64_t timestep)
    {
    // access the particle data
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(),
                               access_location::device,
                               access_mode::read);
    ArrayHandle<Scalar4> d_orientation(this->m_pdata->getOrientationArray(),
                                       access_location::device,
                                       access_mode::read);
    ArrayHandle<Scalar> d_charge(this->m_pdata->getCharges(),
                                 access_location::device,
                                 access_mode::read);

    const BoxDim box = this->m_pdata->getGlobalBox();

    ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_torque(this->m_torque, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(this->m_virial, access_location::device, access_mode::overwrite);
    ArrayHandle<typename evaluator::param_type> d_params(this->m_params,
                                                         access_location::device,
                                                         access_mode::read);

    m_tuner->begin();
    kernel::gpu_compute_potential_external_forces<evaluator>(
        kernel::external_potential_args_t(d_force.data,
                                          d_torque.data,
                                          d_virial.data,
                                          this->m_virial.getPitch(),
                                          this->m_pdata->getN(),
                                          d_pos.data,
                                          d_orientation.data,
                                          d_charge.data,
                                          box,
                                          m_tuner->getParam()[0],
                                          this->m_exec_conf->dev_prop),
        d_params.data,
        this->m_field.get());

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_tuner->end();
    }

namespace detail
    {
//! Export this external potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Evaluator type to export.
*/
template<class T> void export_PotentialExternalGPU(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<PotentialExternalGPU<T>,
                     PotentialExternal<T>,
                     std::shared_ptr<PotentialExternalGPU<T>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
