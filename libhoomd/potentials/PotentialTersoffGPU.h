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

#ifndef __POTENTIAL_TERSOFF_GPU_H__
#define __POTENTIAL_TERSOFF_GPU_H__

#ifdef ENABLE_CUDA

#include <boost/bind.hpp>

#include "PotentialTersoff.h"
#include "PotentialTersoffGPU.cuh"

/*! \file PotentialTersoffGPU.h
    \brief Defines the template class computing certain three-body forces on the GPU
    \note This header cannot be compiled by nvcc
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! Template class for computing three-body potentials and forces on the GPU
/*! Derived from PotentialTersoff, this class provides exactly the same interface for computing
    the three-body potentials and forces.  In the same way as PotentialTersoff, this class serves
    as a shell dealing with all the details of looping while the evaluator actually computes the
    potential and forces.

    \tparam evaluator Evaluator class used to evaluate V(r) and F(r)/r
    \tparam gpu_cgpf Driver function that calls gpu_compute_tersoff_forces<evaluator>()

    \sa export_PotentialTersoffGPU()
*/
template< class evaluator, cudaError_t gpu_cgpf(const tersoff_args_t& pair_args,
                                                const typename evaluator::param_type *d_params) >
class PotentialTersoffGPU : public PotentialTersoff<evaluator>
    {
    public:
        //! Construct the potential
        PotentialTersoffGPU(boost::shared_ptr<SystemDefinition> sysdef,
                            boost::shared_ptr<NeighborList> nlist,
                            const std::string& log_suffix="");
        //! Destructor
        virtual ~PotentialTersoffGPU();

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
        unsigned int m_block_size;  //!< Block size to execute on the GPU

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

template< class evaluator, cudaError_t gpu_cgpf(const tersoff_args_t& pair_args,
                                                const typename evaluator::param_type *d_params) >
PotentialTersoffGPU< evaluator, gpu_cgpf >::PotentialTersoffGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                                                boost::shared_ptr<NeighborList> nlist,
                                                                const std::string& log_suffix)
    : PotentialTersoff<evaluator>(sysdef, nlist, log_suffix), m_block_size(64)
    {
    this->exec_conf->msg->notice(5) << "Constructing PotentialTersoffGPU" << endl;

    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!this->exec_conf->isCUDAEnabled())
        {
        this->exec_conf->msg->error() << "***Error! Creating a PotentialTersoffGPU with no GPU in the execution configuration"
                  << std::endl;
        throw std::runtime_error("Error initializing PotentialTersoffGPU");
        }
    }

template< class evaluator, cudaError_t gpu_cgpf(const tersoff_args_t& pair_args,
                                                const typename evaluator::param_type *d_params) >
PotentialTersoffGPU< evaluator, gpu_cgpf >::~PotentialTersoffGPU()
        {
        this->exec_conf->msg->notice(5) << "Destroying PotentialTersoffGPU" << endl;
        }

template< class evaluator, cudaError_t gpu_cgpf(const tersoff_args_t& pair_args,
                                                const typename evaluator::param_type *d_params) >
void PotentialTersoffGPU< evaluator, gpu_cgpf >::computeForces(unsigned int timestep)
    {
    // start by updating the neighborlist
    this->m_nlist->compute(timestep);

    // start the profile
    if (this->m_prof) this->m_prof->push(this->exec_conf, this->m_prof_name);

    // The GPU implementation CANNOT handle a half neighborlist, error out now
    bool third_law = this->m_nlist->getStorageMode() == NeighborList::half;
    if (third_law)
        {
        this->exec_conf->msg->error() << "***Error! PotentialTersoffGPU cannot handle a half neighborlist"
                  << std::endl;
        throw std::runtime_error("Error computing forces in PotentialTersoffGPU");
        }

    // access the neighbor list
    ArrayHandle<unsigned int> d_n_neigh(this->m_nlist->getNNeighArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_nlist(this->m_nlist->getNListArray(), access_location::device, access_mode::read);
    Index2D nli = this->m_nlist->getNListIndexer();

    // access the particle data
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(), access_location::device, access_mode::read);

    BoxDim box = this->m_pdata->getBox();
    
    // access parameters
    ArrayHandle<Scalar> d_ronsq(this->m_ronsq, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_rcutsq(this->m_rcutsq, access_location::device, access_mode::read);
    ArrayHandle<typename evaluator::param_type> d_params(this->m_params, access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(this->m_virial, access_location::device, access_mode::overwrite);

    gpu_cgpf(tersoff_args_t(d_force.data,
                            this->m_pdata->getN(),
                            d_pos.data,
                            box,
                            d_n_neigh.data,
                            d_nlist.data,
                            nli,
                            d_rcutsq.data,
                            d_ronsq.data,
                            this->m_pdata->getNTypes(),
                            m_block_size),
                            d_params.data);

    if (this->exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    if (this->m_prof) this->m_prof->pop(this->exec_conf);
    }

//! Export this three-body potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated PotentialTersoffGPU class template.
    \tparam Base Base class of \a T. \b Must be PotentialTersoff<evaluator> with the same evaluator as used in \a T.
*/
template < class T, class Base > void export_PotentialTersoffGPU(const std::string& name)
    {
     boost::python::class_<T, boost::shared_ptr<T>, boost::python::bases<Base>, boost::noncopyable >
              (name.c_str(), boost::python::init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<NeighborList>, const std::string& >())
              .def("setBlockSize", &T::setBlockSize)
              ;
    }

#endif // ENABLE_CUDA
#endif
