/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: jglaser

#include <boost/shared_ptr.hpp>
#include <boost/python.hpp>
#include "PotentialExternal.h"
#include "PotentialExternalGPU.cuh"

/*! \file PotentialExternalGPU.h
    \brief Declares a class for computing an external potential field on the GPU
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

#ifndef __POTENTIAL_EXTERNAL_GPU_H__
#define __POTENTIAL_EXTERNAL_GPU_H__

//! Applys a constraint force to keep a group of particles on a sphere
/*! \ingroup computes
*/
template<class evaluator, cudaError_t gpu_cpef(const external_potential_args_t& external_potential_args,
                                               const typename evaluator::param_type *d_params)>
class PotentialExternalGPU : public PotentialExternal<evaluator>
    {
    public:
        //! Constructs the compute
        PotentialExternalGPU(boost::shared_ptr<SystemDefinition> sysdef);

        //! Set the block size to execute on the GPU
        /*! \param block_size Size of the block to run on the device
            Performance of the code may be dependant on the block size run
            on the GPU. \a block_size should be set to be a multiple of 32.
        */
        void setBlockSize(int block_size)
            {
            m_block_size = block_size;
            }

    protected:

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);

        //! block size
        unsigned int m_block_size;
    };

/*! Constructor
    \param sysdef system definition
 */
template<class evaluator, cudaError_t gpu_cpef(const external_potential_args_t& external_potential_args,
                                               const typename evaluator::param_type *d_params)>
PotentialExternalGPU<evaluator, gpu_cpef>::PotentialExternalGPU(boost::shared_ptr<SystemDefinition> sysdef)
    : PotentialExternal<evaluator>(sysdef), m_block_size(128)
    {
    }

/*! Computes the specified constraint forces
    \param timestep Current timestep
*/
template<class evaluator, cudaError_t gpu_cpef(const external_potential_args_t& external_potential_args,
                                               const typename evaluator::param_type *d_params)>
void PotentialExternalGPU<evaluator, gpu_cpef>::computeForces(unsigned int timestep)
    {
    // start the profile
    if (this->m_prof) this->m_prof->push(this->exec_conf, "PotentialExternalGPU");

    // access the particle data
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(), access_location::device, access_mode::read);
    const BoxDim& box = this->m_pdata->getGlobalBox();

    ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(this->m_virial, access_location::device, access_mode::overwrite);
    ArrayHandle<typename evaluator::param_type> d_params(this->m_params, access_location::device, access_mode::read);

    gpu_cpef(external_potential_args_t(d_force.data,
                         d_virial.data,
                         this->m_virial.getPitch(),
                         this->m_pdata->getN(),
                         d_pos.data,
                         box,
                         m_block_size), d_params.data);

    if (this->m_prof) this->m_prof->pop();

    }

//! Export this external potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated PotentialExternalGPU class template.
*/
template < class T, class base >
void export_PotentialExternalGPU(const std::string& name)
    {
    boost::python::class_<T, boost::shared_ptr<T>, boost::python::bases<base>, boost::noncopyable >
                  (name.c_str(), boost::python::init< boost::shared_ptr<SystemDefinition> >())
                  .def("setParams", &T::setParams)
                  .def("setBlockSize", &T::setBlockSize)
                  ;
    }

#endif

