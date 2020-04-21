// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.
//
// Maintainer: SCiarella

#ifndef __POTENTIAL_REVCROSS_GPU_H__
#define __POTENTIAL_REVCROSS_GPU_H__

#ifdef ENABLE_HIP

#include <memory>

#include "PotentialRevCross.h"
#include "PotentialRevCrossGPU.cuh"

/*! \file PotentialRevCrossGPU.h
    \brief Defines the template class computing certain three-body forces on the GPU
    \note This header cannot be compiled by nvcc
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

//! Template class for computing three-body potentials and forces on the GPU
/*! Derived from PotentialRevCross, this class provides exactly the same interface for computing
    the three-body potentials and forces.  In the same way as PotentialRevCross, this class serves
    as a shell dealing with all the details of looping while the evaluator actually computes the
    potential and forces.

    \tparam evaluator Evaluator class used to evaluate V(r) and F(r)/r
    \tparam gpu_cgpf Driver function that calls gpu_compute_revcross_forces<evaluator>()

    \sa export_PotentialRevCrossGPU()
*/
template< class evaluator, hipError_t gpu_cgpf(const revcross_args_t& pair_args,
                                                const typename evaluator::param_type *d_params) >
class PotentialRevCrossGPU : public PotentialRevCross<evaluator>
    {
    public:
        //! Construct the potential
        PotentialRevCrossGPU(std::shared_ptr<SystemDefinition> sysdef,
                            std::shared_ptr<NeighborList> nlist,
                            const std::string& log_suffix="");
        //! Destructor
        virtual ~PotentialRevCrossGPU();

        //! Set autotuner parameters
        /*! \param enable Enable/disable autotuning
            \param period period (approximate) in time steps when returning occurs
        */
        virtual void setAutotunerParams(bool enable, unsigned int period)
            {
            PotentialRevCross<evaluator>::setAutotunerParams(enable, period);
            this->m_tuner->setPeriod(period);
            this->m_tuner->setEnabled(enable);
            }

    protected:
        std::unique_ptr<Autotuner> m_tuner; //!< Autotuner for block size

        //! Actually compute the forces
        virtual void computeForces(unsigned int timestep);
    };

template< class evaluator, hipError_t gpu_cgpf(const revcross_args_t& pair_args,
                                                const typename evaluator::param_type *d_params) >
PotentialRevCrossGPU< evaluator, gpu_cgpf >::PotentialRevCrossGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                                std::shared_ptr<NeighborList> nlist,
                                                                const std::string& log_suffix)
    : PotentialRevCross<evaluator>(sysdef, nlist, log_suffix)
    {
    this->m_exec_conf->msg->notice(5) << "Constructing PotentialRevCrossGPU" << std::endl;

    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!this->m_exec_conf->isCUDAEnabled())
        {
        this->m_exec_conf->msg->error() << "***Error! Creating a PotentialRevCrossGPU with no GPU in the execution configuration"
                  << std::endl;
        throw std::runtime_error("Error initializing PotentialRevCrossGPU");
        }

    // initialize autotuner
    // the full block size and threads_per_particle matrix is searched,
    // encoded as block_size*10000 + threads_per_particle
    unsigned int max_tpp = 1;
    max_tpp = this->m_exec_conf->dev_prop.warpSize;

    std::vector<unsigned int> valid_params;
    for (unsigned int block_size = 32; block_size <= 1024; block_size += 32)
        {
        unsigned int s=1;

        while (s <= max_tpp)
            {
            valid_params.push_back(block_size*10000 + s);
            s = s * 2;
            }
        }

    m_tuner.reset(new Autotuner(valid_params, 5, 100000, "pair_revcross", this->m_exec_conf));
    }

template< class evaluator, hipError_t gpu_cgpf(const revcross_args_t& pair_args,
                                                const typename evaluator::param_type *d_params) >
PotentialRevCrossGPU< evaluator, gpu_cgpf >::~PotentialRevCrossGPU()
        {
        this->m_exec_conf->msg->notice(5) << "Destroying PotentialRevCrossGPU" << std::endl;
        }

template< class evaluator, hipError_t gpu_cgpf(const revcross_args_t& pair_args,
                                                const typename evaluator::param_type *d_params) >
void PotentialRevCrossGPU< evaluator, gpu_cgpf >::computeForces(unsigned int timestep)
    {
    // start by updating the neighborlist
    this->m_nlist->compute(timestep);

    // start the profile
    if (this->m_prof) this->m_prof->push(this->m_exec_conf, this->m_prof_name);

    // The GPU implementation CANNOT handle a half neighborlist, error out now
    bool third_law = this->m_nlist->getStorageMode() == NeighborList::half;
    if (third_law)
        {
        this->m_exec_conf->msg->error() << "***Error! PotentialRevCrossGPU cannot handle a half neighborlist"
                  << std::endl;
        throw std::runtime_error("Error computing forces in PotentialRevCrossGPU");
        }

    // access the neighbor list
    ArrayHandle<unsigned int> d_n_neigh(this->m_nlist->getNNeighArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_nlist(this->m_nlist->getNListArray(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_head_list(this->m_nlist->getHeadList(), access_location::device, access_mode::read);

    // access the particle data
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(), access_location::device, access_mode::read);

    BoxDim box = this->m_pdata->getBox();

    // access parameters
    ArrayHandle<Scalar> d_ronsq(this->m_ronsq, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_rcutsq(this->m_rcutsq, access_location::device, access_mode::read);
    ArrayHandle<typename evaluator::param_type> d_params(this->m_params, access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(this->m_virial, access_location::device, access_mode::overwrite);
    
    // access flags
    PDataFlags flags = this->m_pdata->getFlags();
    bool compute_virial = flags[pdata_flag::pressure_tensor] || flags[pdata_flag::isotropic_virial];

    this->m_tuner->begin();
    unsigned int param =  this->m_tuner->getParam();
    unsigned int block_size = param / 10000;
    unsigned int threads_per_particle = param % 10000;

    gpu_cgpf(revcross_args_t(d_force.data,
                            this->m_pdata->getN(),
                            this->m_pdata->getNGhosts(),
                            d_virial.data,
                            this->m_virial_pitch,
                            compute_virial,
                            d_pos.data,
                            box,
                            d_n_neigh.data,
                            d_nlist.data,
                            d_head_list.data,
                            d_rcutsq.data,
                            d_ronsq.data,
                            this->m_nlist->getNListArray().getPitch(),
                            this->m_pdata->getNTypes(),
                            block_size,
                            threads_per_particle,
                            this->m_exec_conf->dev_prop),
                            d_params.data);

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    this->m_tuner->end();

    if (this->m_prof) this->m_prof->pop(this->m_exec_conf);
    }

//! Export this three-body potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Class type to export. \b Must be an instantiated PotentialRevCrossGPU class template.
    \tparam Base Base class of \a T. \b Must be PotentialRevCross<evaluator> with the same evaluator as used in \a T.
*/
template < class T, class Base > void export_PotentialRevCrossGPU(pybind11::module& m, const std::string& name)
    {
     pybind11::class_<T, std::shared_ptr<T> >(m, name.c_str(), pybind11::base<Base>())
        .def(pybind11::init< std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>, const std::string& >())
    ;
    }

#endif // ENABLE_HIP
#endif
