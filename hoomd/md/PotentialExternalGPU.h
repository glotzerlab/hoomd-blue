// Copyright (c) 2009-2022 The Regents of the University of Michigan.
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

    //! Set autotuner parameters
    /*! \param enable Enable/disable autotuning
        \param period period (approximate) in time steps when returning occurs
    */
    virtual void setAutotunerParams(bool enable, unsigned int period)
        {
        PotentialExternal<evaluator>::setAutotunerParams(enable, period);
        m_tuner->setPeriod(period);
        m_tuner->setEnabled(enable);
        }

    protected:
    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);

    std::unique_ptr<Autotuner> m_tuner; //!< Autotuner for block size
    };

/*! Constructor
    \param sysdef system definition
 */
template<class evaluator>
PotentialExternalGPU<evaluator>::PotentialExternalGPU(std::shared_ptr<SystemDefinition> sysdef)
    : PotentialExternal<evaluator>(sysdef)
    {
    unsigned int warp_size = this->m_exec_conf->dev_prop.warpSize;
    this->m_tuner.reset(new Autotuner(warp_size,
                                      1024,
                                      warp_size,
                                      5,
                                      100000,
                                      "external_" + evaluator::getName(),
                                      this->m_exec_conf));
    }

/*! Computes the specified constraint forces
    \param timestep Current timestep
*/
template<class evaluator> void PotentialExternalGPU<evaluator>::computeForces(uint64_t timestep)
    {
    // start the profile
    if (this->m_prof)
        this->m_prof->push(this->m_exec_conf, "PotentialExternalGPU");

    // access the particle data
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(),
                               access_location::device,
                               access_mode::read);
    ArrayHandle<Scalar> d_diameter(this->m_pdata->getDiameters(),
                                   access_location::device,
                                   access_mode::read);
    ArrayHandle<Scalar> d_charge(this->m_pdata->getCharges(),
                                 access_location::device,
                                 access_mode::read);

    const BoxDim& box = this->m_pdata->getGlobalBox();

    ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(this->m_virial, access_location::device, access_mode::overwrite);
    ArrayHandle<typename evaluator::param_type> d_params(this->m_params,
                                                         access_location::device,
                                                         access_mode::read);

    // access flags
    PDataFlags flags = this->m_pdata->getFlags();

    this->m_tuner->begin();
    kernel::gpu_cpef<evaluator>(kernel::external_potential_args_t(d_force.data,
                                                                  d_virial.data,
                                                                  this->m_virial.getPitch(),
                                                                  this->m_pdata->getN(),
                                                                  d_pos.data,
                                                                  d_diameter.data,
                                                                  d_charge.data,
                                                                  box,
                                                                  this->m_tuner->getParam()),
                                d_params.data,
                                this->m_field.get());

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    this->m_tuner->end();

    if (this->m_prof)
        this->m_prof->pop();
    }

namespace detail
    {
//! Export this external potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated PotentialExternalGPU class template.
*/
template<class T, class base>
void export_PotentialExternalGPU(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<T, base, std::shared_ptr<T>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif
