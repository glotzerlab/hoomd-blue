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

#ifndef __POTENTIAL_BOND_GPU_H__
#define __POTENTIAL_BOND_GPU_H__

#ifdef ENABLE_CUDA

#include <boost/bind.hpp>

#include "PotentialBond.h"
#include "PotentialBondGPU.cuh"

/*! \file PotentialBondGPU.h
    \brief Defines the template class for standard bond potentials on the GPU
    \note This header cannot be compiled by nvcc
*/

#ifdef NVCC
#error This header cannot be compiled by nvcc
#endif

//! Template class for computing bond potentials on the GPU

/*!
    \tparam evaluator EvaluatorBond class used to evaluate V(r) and F(r)/r
    \tparam gpu_cgbf Driver function that calls gpu_compute_bond_forces<evaluator>()

    \sa export_PotentialBondGPU()
*/
template< class evaluator, cudaError_t gpu_cgbf(const bond_args_t& bond_args,
                                                const typename evaluator::param_type *d_params,
                                                unsigned int *d_flags),
                           cudaError_t gpu_igbf() >
class PotentialBondGPU : public PotentialBond<evaluator>
    {
    public:
        //! Construct the bond potential
        PotentialBondGPU(boost::shared_ptr<SystemDefinition> sysdef,
                         const std::string& log_suffix="");
        //! Destructor
        virtual ~PotentialBondGPU() {}

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
        unsigned int m_block_size;      //!< Block size to execute on the GPU

        GPUArray<unsigned int> m_flags; //!< Flags set during the kernel execution

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);

#ifdef ENABLE_MPI
        //! Compute forces due to ghost particles
        virtual void computeGhostForces(unsigned int timestep);
#endif
    };

template< class evaluator, cudaError_t gpu_cgbf(const bond_args_t& bond_args,
                                                const typename evaluator::param_type *d_params,
                                                unsigned int *d_flags),
                           cudaError_t gpu_igbf() >
PotentialBondGPU< evaluator, gpu_cgbf, gpu_igbf >::PotentialBondGPU(boost::shared_ptr<SystemDefinition> sysdef,
                                                          const std::string& log_suffix)
    : PotentialBond<evaluator>(sysdef, log_suffix), m_block_size(64)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!this->exec_conf->isCUDAEnabled())
        {
        this->m_exec_conf->msg->error() << "Creating a PotentialBondGPU with no GPU in the execution configuration" << std::endl;
        throw std::runtime_error("Error initializing PotentialBondGPU");
        }

     // allocate and zero device memory
    GPUArray<typename evaluator::param_type> params(this->m_bond_data->getNBondTypes(), this->exec_conf);
    this->m_params.swap(params);

     // allocate flags storage on the GPU
    GPUArray<unsigned int> flags(1, this->exec_conf);
    m_flags.swap(flags);

    // set cache config
    gpu_igbf();
    }

template< class evaluator, cudaError_t gpu_cgbf(const bond_args_t& bond_args,
                                                const typename evaluator::param_type *d_params,
                                                unsigned int *d_flags),
                           cudaError_t gpu_igbf()>
void PotentialBondGPU< evaluator, gpu_cgbf, gpu_igbf >::computeForces(unsigned int timestep)
    {
    // start the profile
    if (this->m_prof) this->m_prof->push(this->exec_conf, this->m_prof_name);

    // access the particle data
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_diameter(this->m_pdata->getDiameters(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_charge(this->m_pdata->getCharges(), access_location::device, access_mode::read);
    BoxDim box = this->m_pdata->getBox();

    // access parameters
    ArrayHandle<typename evaluator::param_type> d_params(this->m_params, access_location::device, access_mode::read);

    // access net force & virial
    ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(this->m_virial, access_location::device, access_mode::overwrite);

        {
        ArrayHandle<uint2> d_gpu_bondlist(this->m_bond_data->getGPUBondList(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int > d_gpu_n_bonds(this->m_bond_data->getNBondsArray(), access_location::device, access_mode::read);

        // access the flags array for overwriting
        ArrayHandle<unsigned int> d_flags(m_flags, access_location::device, access_mode::overwrite);

        gpu_cgbf(bond_args_t(d_force.data,
                             d_virial.data,
                             this->m_virial.getPitch(),
                             this->m_pdata->getN(),
                             d_pos.data,
                             d_charge.data,
                             d_diameter.data,
                             box,
                             d_gpu_bondlist.data,
                             this->m_bond_data->getGPUBondList().getPitch(),
                             d_gpu_n_bonds.data,
                             this->m_bond_data->getNBondTypes(),
                             m_block_size,
                             false),
                 d_params.data,
                 d_flags.data);
        }

    if (this->exec_conf->isCUDAErrorCheckingEnabled())
        {
        CHECK_CUDA_ERROR();

        // check the flags for any errors
        ArrayHandle<unsigned int> h_flags(m_flags, access_location::host, access_mode::read);

        if (h_flags.data[0]==1)
            {
            this->m_exec_conf->msg->error() << "bond." << evaluator::getName() << ": bond out of bounds" << endl << endl;
            throw std::runtime_error("Error in bond calculation");
            }

        }

    if (this->m_prof) this->m_prof->pop(this->exec_conf);
    }

#ifdef ENABLE_MPI
template< class evaluator, cudaError_t gpu_cgbf(const bond_args_t& bond_args,
                                                const typename evaluator::param_type *d_params,
                                                unsigned int *d_flags),
                           cudaError_t gpu_igbf() >
void PotentialBondGPU< evaluator, gpu_cgbf, gpu_igbf >::computeGhostForces(unsigned int timestep)
    {
    // start the profile
    if (this->m_prof) this->m_prof->push(this->exec_conf, this->m_prof_name);

    // access the particle data
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_diameter(this->m_pdata->getDiameters(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_charge(this->m_pdata->getCharges(), access_location::device, access_mode::read);
    BoxDim box = this->m_pdata->getBox();

    // access parameters
    ArrayHandle<typename evaluator::param_type> d_params(this->m_params, access_location::device, access_mode::read);

    // access net force & virial
    ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(this->m_virial, access_location::device, access_mode::overwrite);

        {
        ArrayHandle<uint2> d_gpu_ghost_bondlist(this->m_bond_data->getGPUGhostBondList(), access_location::device, access_mode::read);
        ArrayHandle<unsigned int > d_gpu_n_ghost_bonds(this->m_bond_data->getNGhostBondsArray(), access_location::device, access_mode::read);

        // access the flags array for overwriting
        ArrayHandle<unsigned int> d_flags(m_flags, access_location::device, access_mode::overwrite);

        gpu_cgbf(bond_args_t(d_force.data,
                             d_virial.data,
                             this->m_virial.getPitch(),
                             this->m_pdata->getN(),
                             d_pos.data,
                             d_charge.data,
                             d_diameter.data,
                             box,
                             d_gpu_ghost_bondlist.data,
                             this->m_bond_data->getGPUGhostBondList().getPitch(),
                             d_gpu_n_ghost_bonds.data,
                             this->m_bond_data->getNBondTypes(),
                             m_block_size,
                             true),
                 d_params.data,
                 d_flags.data);
        }

    if (this->exec_conf->isCUDAErrorCheckingEnabled())
        {
        CHECK_CUDA_ERROR();

        // check the flags for any errors
        ArrayHandle<unsigned int> h_flags(m_flags, access_location::host, access_mode::read);

        if (h_flags.data[0]==1)
            {
            this->m_exec_conf->msg->error() << "bond." << evaluator::getName() << ": bond out of bounds" << endl << endl;
            throw std::runtime_error("Error in bond calculation");
            }
        if (h_flags.data[0]==2)
            {
            this->m_exec_conf->msg->error() << "Found incomplete bond. Try increasing the bond stiffness or reduce number of domains."  << endl << endl;
            throw std::runtime_error("Error in bond calculation");
            }

        }

    if (this->m_prof) this->m_prof->pop(this->exec_conf);
    }
#endif

//! Export this bond potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated PotentialPairGPU class template.
    \tparam Base Base class of \a T. \b Must be PotentialPair<evaluator> with the same evaluator as used in \a T.
*/
template < class T, class Base > void export_PotentialBondGPU(const std::string& name)
    {
     boost::python::class_<T, boost::shared_ptr<T>, boost::python::bases<Base>, boost::noncopyable >
              (name.c_str(), boost::python::init< boost::shared_ptr<SystemDefinition>, const std::string& >())
              .def("setBlockSize", &T::setBlockSize)
              ;
    }

#endif // ENABLE_CUDA
#endif // __POTENTIAL_PAIR_GPU_H__

