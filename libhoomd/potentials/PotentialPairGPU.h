/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: joaander

#ifndef __POTENTIAL_PAIR_GPU_H__
#define __POTENTIAL_PAIR_GPU_H__

#ifdef ENABLE_CUDA

#include <boost/bind.hpp>

#include "PotentialPair.h"
#include "PotentialPairGPU.cuh"

/*! \file PotentialPairGPU.h
    \brief Defines the template class for standard pair potentials on the GPU
    \note This header cannot be compiled by nvcc
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! Template class for computing pair potentials on the GPU
/*! Derived from PotentialPair, this class provides exactly the same interface for computing pair potentials and forces.
    In the same way as PotentialPair, this class serves as a shell dealing with all the details common to every pair
    potential calculation while the \a evaluator calculates V(r) in a generic way.
    
    Due to technical limitations, the instantiation of PotentialPairGPU cannot create a CUDA kernel automatically
    with the \a evaluator. Instead, a .cu file must be written that provides a driver function to call 
    gpu_compute_pair_forces() instantiated with the same evaluator. (See PotentialPairLJGPU.cu and 
    PotentialPairLJGPU.cuh for an example). That function is then passed into this class as another template parameter
    \a gpu_cgpf
    
    \tparam evaluator EvaluatorPair class used to evaluate V(r) and F(r)/r
    \tparam gpu_cgpf Driver function that calls gpu_compute_pair_forces<evaluator>()
    
    \sa export_PotentialPairGPU()
*/
template< class evaluator, cudaError_t gpu_cgpf(const pair_args_t& pair_args,
                                                const typename evaluator::param_type *d_params),
                           cudaError_t gpu_igpf()>
class PotentialPairGPU : public PotentialPair<evaluator>
    {
    public:
        //! Construct the pair potential
        PotentialPairGPU(boost::shared_ptr<SystemDefinition> sysdef,
                         boost::shared_ptr<NeighborList> nlist,
                         const std::string& log_suffix="");
        //! Destructor
        virtual ~PotentialPairGPU() {}
        
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
        virtual void computeForces(unsigned int timestep, bool ghost);

    };

template< class evaluator, cudaError_t gpu_cgpf(const pair_args_t& pair_args,
                                                const typename evaluator::param_type *d_params),
                           cudaError_t gpu_igpf()>
PotentialPairGPU< evaluator, gpu_cgpf, gpu_igpf >::PotentialPairGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                                          boost::shared_ptr<NeighborList> nlist, const std::string& log_suffix)
    : PotentialPair<evaluator>(sysdef, nlist, log_suffix), m_block_size(64)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!this->exec_conf->isCUDAEnabled())
        {
        this->m_exec_conf->msg->error() << "Creating a PotentialPairGPU with no GPU in the execution configuration" 
                  << std::endl;
        throw std::runtime_error("Error initializing PotentialPairGPU");
        }

    // set cache configuration
    gpu_igpf();
    }

template< class evaluator, cudaError_t gpu_cgpf(const pair_args_t& pair_args,
                                                const typename evaluator::param_type *d_params),
                           cudaError_t gpu_igpf()>
void PotentialPairGPU< evaluator, gpu_cgpf, gpu_igpf >::computeForces(unsigned int timestep, bool ghost)
    {
    // start by updating the neighborlist
    this->m_nlist->compute(timestep);
    
    // start the profile
    if (this->m_prof) this->m_prof->push(this->exec_conf, this->m_prof_name);
    
    // The GPU implementation CANNOT handle a half neighborlist, error out now
    bool third_law = this->m_nlist->getStorageMode() == NeighborList::half;
    if (third_law)
        {
        this->m_exec_conf->msg->error() << "PotentialPairGPU cannot handle a half neighborlist" 
                  << std::endl;
        throw std::runtime_error("Error computing forces in PotentialPairGPU");
        }
        
    // access the neighbor list
    ArrayHandle<unsigned int> d_n_neigh(ghost ? this->m_nlist->getNGhostNeighArray() :
                                                this->m_nlist->getNNeighArray(),
                                        access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_nlist(ghost ? this->m_nlist->getGhostNListArray() :
                                              this->m_nlist->getNListArray(),
                                      access_location::device, access_mode::read);
    Index2D nli = this->m_nlist->getNListIndexer();
    
    // access the particle data
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_diameter(this->m_pdata->getDiameters(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_charge(this->m_pdata->getCharges(), access_location::device, access_mode::read);

    BoxDim box = this->m_pdata->getBox();
    
    // access parameters
    ArrayHandle<Scalar> d_ronsq(this->m_ronsq, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_rcutsq(this->m_rcutsq, access_location::device, access_mode::read);
    ArrayHandle<typename evaluator::param_type> d_params(this->m_params, access_location::device, access_mode::read);
    
    ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(this->m_virial, access_location::device, access_mode::overwrite);
    
    // access flags
    PDataFlags flags = this->m_pdata->getFlags();

    gpu_cgpf(pair_args_t(d_force.data,
                         d_virial.data,
                         this->m_virial.getPitch(),
                         this->m_pdata->getN(),
                         d_pos.data,
                         d_diameter.data,
                         d_charge.data,
                         box,
                         d_n_neigh.data,
                         d_nlist.data,
                         nli,
                         d_rcutsq.data,
                         d_ronsq.data,
                         this->m_pdata->getNTypes(),
                         m_block_size,
                         this->m_shift_mode,
                         flags[pdata_flag::pressure_tensor] || flags[pdata_flag::isotropic_virial],
                         ghost),
             d_params.data);
    
    if (this->exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
   
    if (this->m_prof) this->m_prof->pop(this->exec_conf);
    }

//! Export this pair potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated PotentialPairGPU class template.
    \tparam Base Base class of \a T. \b Must be PotentialPair<evaluator> with the same evaluator as used in \a T.
*/
template < class T, class Base > void export_PotentialPairGPU(const std::string& name)
    {
     boost::python::class_<T, boost::shared_ptr<T>, boost::python::bases<Base>, boost::noncopyable >
              (name.c_str(), boost::python::init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<NeighborList>, const std::string& >())
              .def("setBlockSize", &T::setBlockSize)
              ;
    }

#endif // ENABLE_CUDA
#endif // __POTENTIAL_PAIR_GPU_H__

