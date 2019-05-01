// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: jglaser

#include <memory>
#include "PotentialExternal.h"
#include "PotentialExternalGPU.cuh"
#include "hoomd/Autotuner.h"

/*! \file PotentialExternalGPU.h
    \brief Declares a class for computing an external potential field on the GPU
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#include <hoomd/extern/pybind/include/pybind11/pybind11.h>

#ifndef __POTENTIAL_EXTERNAL_GPU_H__
#define __POTENTIAL_EXTERNAL_GPU_H__

//! Applys a constraint force to keep a group of particles on a sphere
/*! \ingroup computes
*/
template<class evaluator>
class PotentialExternalGPU : public PotentialExternal<evaluator>
    {
    public:
        //! Constructs the compute
        PotentialExternalGPU(std::shared_ptr<SystemDefinition> sysdef,
                             const std::string& log_suffix="");

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
        virtual void computeForces(unsigned int timestep);

        std::unique_ptr<Autotuner> m_tuner; //!< Autotuner for block size
    };

/*! Constructor
    \param sysdef system definition
 */
template<class evaluator>
PotentialExternalGPU<evaluator>::PotentialExternalGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                                const std::string& log_suffix)
    : PotentialExternal<evaluator>(sysdef, log_suffix)
    {
    this->m_tuner.reset(new Autotuner(32, 1024, 32, 5, 100000, "external_" + evaluator::getName(), this->m_exec_conf));
    }

/*! Computes the specified constraint forces
    \param timestep Current timestep
*/
template<class evaluator>
void PotentialExternalGPU<evaluator>::computeForces(unsigned int timestep)
    {
    // start the profile
    if (this->m_prof) this->m_prof->push(this->m_exec_conf, "PotentialExternalGPU");

    // access the particle data
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_diameter(this->m_pdata->getDiameters(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_charge(this->m_pdata->getCharges(), access_location::device, access_mode::read);

    const BoxDim& box = this->m_pdata->getGlobalBox();

    ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(this->m_virial, access_location::device, access_mode::overwrite);
    ArrayHandle<typename evaluator::param_type> d_params(this->m_params, access_location::device, access_mode::read);
    ArrayHandle<typename evaluator::field_type> d_field(this->m_field, access_location::device, access_mode::read);

    // access flags
    PDataFlags flags = this->m_pdata->getFlags();

    this->m_tuner->begin();
    gpu_cpef< evaluator >(external_potential_args_t(d_force.data,
                         d_virial.data,
                         this->m_virial.getPitch(),
                         this->m_pdata->getN(),
                         d_pos.data,
                         d_diameter.data,
                         d_charge.data,
                         box,
                         this->m_tuner->getParam()),
                         d_params.data,
                         d_field.data);

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    this->m_tuner->end();

    if (this->m_prof) this->m_prof->pop();

    if (flags[pdata_flag::external_field_virial])
        {
        bool virial_terms_defined=evaluator::requestFieldVirialTerm();
        if (!virial_terms_defined)
            {
            this->m_exec_conf->msg->error() << "The required virial terms are not defined for the current setup." << std::endl;
            throw std::runtime_error("NPT is not supported for requested features");
            }
        }

    }

//! Export this external potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated PotentialExternalGPU class template.
*/
template < class T, class base >
void export_PotentialExternalGPU(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<T, std::shared_ptr<T> >(m, name.c_str(), pybind11::base<base>())
                .def(pybind11::init< std::shared_ptr<SystemDefinition>, const std::string&  >())
                .def("setParams", &T::setParams)
                .def("setField", &T::setField)
                ;
    }

#endif
