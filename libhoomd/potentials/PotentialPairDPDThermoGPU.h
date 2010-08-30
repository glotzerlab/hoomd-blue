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

// $Id$
// $URL$
// Maintainer: phillicl

#ifndef __POTENTIAL_PAIR_DPDTHERMO_GPU_H__
#define __POTENTIAL_PAIR_DPDTHERMO_GPU_H__

#ifdef ENABLE_CUDA

#include <boost/bind.hpp>
#include "Variant.h"
#include "PotentialPairDPDThermo.h"
#include "PotentialPairDPDThermoGPU.cuh"
#include "AllPairPotentials.h"

/*! \file PotentialPairDPDThermoGPU.h
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
    
    Due to technical limitations, the instantiation of PotentialPairDPDThermoGPU cannot create a CUDA kernel automatically
    with the \a evaluator. Instead, a .cu file must be written that provides a driver function to call 
    gpu_compute_dpd_forces() instantiated with the same evaluator. (See PotentialPairLJGPU.cu and 
    PotentialPairLJGPU.cuh for an example). That function is then passed into this class as another template parameter
    \a gpu_cpdf
    
    \tparam evaluator EvaluatorPair class used to evaluate V(r) and F(r)/r
    \tparam gpu_cpdf Driver function that calls gpu_compute_dpd_forces<evaluator>()
    
    \sa export_PotentialPairDPDThermoGPU()
*/
template< class evaluator, cudaError_t gpu_cpdf(const gpu_force_data_arrays& force_data,
                                                const gpu_pdata_arrays &pdata,
                                                const gpu_boxsize &box,
                                                const gpu_nlist_array &nlist,
                                                typename evaluator::param_type *d_params,
                                                float *d_rcutsq,
                                                int ntypes,
                                                const dpd_pair_args& args) >
class PotentialPairDPDThermoGPU : public PotentialPairDPDThermo<evaluator>
    {
    public:
        //! Construct the pair potential
        PotentialPairDPDThermoGPU(boost::shared_ptr<SystemDefinition> sysdef,
                         boost::shared_ptr<NeighborList> nlist,
                         const std::string& log_suffix="");
        //! Destructor
        virtual ~PotentialPairDPDThermoGPU() { };
        
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

template< class evaluator, cudaError_t gpu_cpdf(const gpu_force_data_arrays& force_data,
                                                const gpu_pdata_arrays &pdata,
                                                const gpu_boxsize &box,
                                                const gpu_nlist_array &nlist,
                                                typename evaluator::param_type *d_params,
                                                float *d_rcutsq,
                                                int ntypes,                                               
                                                const dpd_pair_args& args) >
PotentialPairDPDThermoGPU< evaluator, gpu_cpdf >::PotentialPairDPDThermoGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                                          boost::shared_ptr<NeighborList> nlist, const std::string& log_suffix)
    : PotentialPairDPDThermo<evaluator>(sysdef, nlist, log_suffix), m_block_size(64)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!this->exec_conf->isCUDAEnabled())
        {
        std::cerr << std::endl << "***Error! Creating a PotentialPairDPDThermoGPU with no GPU in the execution configuration" 
                  << std::endl << std::endl;
        throw std::runtime_error("Error initializing PotentialPairDPDThermoGPU");
        }
        
    if (this->m_pdata->getNTypes() > 44)
        {
        std::cerr << std::endl << "***Error! PotentialPairDPDThermoGPU cannot handle " << this->m_pdata->getNTypes() << " types" 
                  << std::endl << std::endl;
        throw std::runtime_error("Error initializing PotentialPairDPDThermoGPU");
        }        
    }

template< class evaluator, cudaError_t gpu_cpdf(const gpu_force_data_arrays& force_data,
                                                const gpu_pdata_arrays &pdata,
                                                const gpu_boxsize &box,
                                                const gpu_nlist_array &nlist,
                                                typename evaluator::param_type *d_params,
                                                float *d_rcutsq,
                                                int ntypes,
                                                const dpd_pair_args& args) >
void PotentialPairDPDThermoGPU< evaluator, gpu_cpdf >::computeForces(unsigned int timestep)
    {
    // start by updating the neighborlist
    this->m_nlist->compute(timestep);
    
    // start the profile
    if (this->m_prof) this->m_prof->push(this->exec_conf, this->m_prof_name);
    
    // The GPU implementation CANNOT handle a half neighborlist, error out now
    bool third_law = this->m_nlist->getStorageMode() == NeighborList::half;
    if (third_law)
        {
        std::cerr << std::endl << "***Error! PotentialPairDPDThermoGPU cannot handle a half neighborlist" 
                  << std::endl << std::endl;
        throw std::runtime_error("Error computing forces in PotentialPairDPDThermoGPU");
        }
        
    // access the neighbor list, which just selects the neighborlist into the device's memory, copying
    // it there if needed
    gpu_nlist_array& nlist = this->m_nlist->getListGPU();
    
    // access the particle data
    gpu_pdata_arrays& pdata = this->m_pdata->acquireReadOnlyGPU();
    gpu_boxsize box = this->m_pdata->getBoxGPU();
    
    // access parameters
    ArrayHandle<Scalar> d_ronsq(this->m_ronsq, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_rcutsq(this->m_rcutsq, access_location::device, access_mode::read);
    ArrayHandle<typename evaluator::param_type> d_params(this->m_params, access_location::device, access_mode::read);
    
    // run the kernel on all GPUs in parallel
    dpd_pair_args opt;
    opt.block_size = m_block_size;
    opt.seed = this->m_seed;
    opt.timestep = timestep;
    opt.deltaT = this->m_deltaT;
    opt.T = this->m_T->getValue(timestep);    
    
    gpu_cpdf(this->m_gpu_forces.d_data,
             pdata,
             box,
             nlist,
             d_params.data,
             d_rcutsq.data,
             this->m_pdata->getNTypes(),
             opt);
    
    if (this->exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    
    this->m_pdata->release();
    
    // the force data is now only up to date on the gpu
    this->m_data_location = ForceCompute::gpu;
    
    if (this->m_prof) this->m_prof->pop(this->exec_conf);
    }

//! Export this pair potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated PotentialPairDPDThermoGPU class template.
    \tparam Base Base class of \a T. \b Must be PotentialPairDPDThermo<evaluator> with the same evaluator as used in \a T.
*/
template < class T, class Base > void export_PotentialPairDPDThermoGPU(const std::string& name)
    {
     boost::python::class_<T, boost::shared_ptr<T>, boost::python::bases<Base>, boost::noncopyable >
              (name.c_str(), boost::python::init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<NeighborList>, const std::string& >())
              .def("setBlockSize", &T::setBlockSize)
              ;
    }

#endif // ENABLE_CUDA
#endif // __POTENTIAL_PAIR_DPDTHERMO_GPU_H__

