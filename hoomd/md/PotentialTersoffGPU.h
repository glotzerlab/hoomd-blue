// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __POTENTIAL_TERSOFF_GPU_H__
#define __POTENTIAL_TERSOFF_GPU_H__

#ifdef ENABLE_HIP

#include <memory>

#include "PotentialTersoff.h"
#include "PotentialTersoffGPU.cuh"
#include "hoomd/Autotuner.h"

/*! \file PotentialTersoffGPU.h
    \brief Defines the template class computing certain three-body forces on the GPU
    \note This header cannot be compiled by nvcc
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

namespace hoomd
    {
namespace md
    {
//! Template class for computing three-body potentials and forces on the GPU
/*! Derived from PotentialTersoff, this class provides exactly the same interface for computing
    the three-body potentials and forces.  In the same way as PotentialTersoff, this class serves
    as a shell dealing with all the details of looping while the evaluator actually computes the
    potential and forces.

    \tparam evaluator Evaluator class used to evaluate V(r) and F(r)/r
    \tparam gpu_cgpf Driver function that calls gpu_compute_tersoff_forces<evaluator>()

    \sa export_PotentialTersoffGPU()
*/
template<class evaluator> class PotentialTersoffGPU : public PotentialTersoff<evaluator>
    {
    public:
    //! Construct the potential
    PotentialTersoffGPU(std::shared_ptr<SystemDefinition> sysdef,
                        std::shared_ptr<NeighborList> nlist);
    //! Destructor
    virtual ~PotentialTersoffGPU();

    protected:
    std::shared_ptr<Autotuner<2>> m_tuner; //!< Autotuner for block size

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

template<class evaluator>
PotentialTersoffGPU<evaluator>::PotentialTersoffGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                    std::shared_ptr<NeighborList> nlist)
    : PotentialTersoff<evaluator>(sysdef, nlist)
    {
    this->m_exec_conf->msg->notice(5) << "Constructing PotentialTersoffGPU" << std::endl;

    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!this->m_exec_conf->isCUDAEnabled())
        {
        this->m_exec_conf->msg->error()
            << "***Error! Creating a PotentialTersoffGPU with no GPU in the execution configuration"
            << std::endl;
        throw std::runtime_error("Error initializing PotentialTersoffGPU");
        }

    // Initialize autotuner.
    m_tuner.reset(new Autotuner<2>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf),
                                    AutotunerBase::getTppListPow2(this->m_exec_conf)},
                                   this->m_exec_conf,
                                   "pair_tersoff"));
    this->m_autotuners.push_back(m_tuner);
    }

template<class evaluator> PotentialTersoffGPU<evaluator>::~PotentialTersoffGPU()
    {
    this->m_exec_conf->msg->notice(5) << "Destroying PotentialTersoffGPU" << std::endl;
    }

template<class evaluator> void PotentialTersoffGPU<evaluator>::computeForces(uint64_t timestep)
    {
    // start by updating the neighborlist
    this->m_nlist->compute(timestep);

    // The GPU implementation CANNOT handle a half neighborlist, error out now
    bool third_law = this->m_nlist->getStorageMode() == NeighborList::half;
    if (third_law)
        {
        this->m_exec_conf->msg->error()
            << "***Error! PotentialTersoffGPU cannot handle a half neighborlist" << std::endl;
        throw std::runtime_error("Error computing forces in PotentialTersoffGPU");
        }

    // access the neighbor list
    ArrayHandle<unsigned int> d_n_neigh(this->m_nlist->getNNeighArray(),
                                        access_location::device,
                                        access_mode::read);
    ArrayHandle<unsigned int> d_nlist(this->m_nlist->getNListArray(),
                                      access_location::device,
                                      access_mode::read);
    ArrayHandle<size_t> d_head_list(this->m_nlist->getHeadList(),
                                    access_location::device,
                                    access_mode::read);

    // access the particle data
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(),
                               access_location::device,
                               access_mode::read);

    BoxDim box = this->m_pdata->getBox();

    // access parameters
    ArrayHandle<Scalar> d_rcutsq(this->m_rcutsq, access_location::device, access_mode::read);
    ArrayHandle<typename evaluator::param_type> d_params(this->m_params,
                                                         access_location::device,
                                                         access_mode::read);

    ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(this->m_virial, access_location::device, access_mode::overwrite);

    PDataFlags flags = this->m_pdata->getFlags();
    bool compute_virial = flags[pdata_flag::pressure_tensor];

    m_tuner->begin();
    auto param = m_tuner->getParam();
    unsigned int block_size = param[0];
    unsigned int threads_per_particle = param[1];

    kernel::gpu_compute_triplet_forces<evaluator>(
        kernel::tersoff_args_t(d_force.data,
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
                               this->m_nlist->getNListArray().getPitch(),
                               this->m_pdata->getNTypes(),
                               block_size,
                               threads_per_particle,
                               this->m_exec_conf->dev_prop),
        d_params.data);

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_tuner->end();
    }

namespace detail
    {
//! Export this three-body potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Evaluator type to export.
*/
template<class T> void export_PotentialTersoffGPU(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<PotentialTersoffGPU<T>,
                     PotentialTersoff<T>,
                     std::shared_ptr<PotentialTersoffGPU<T>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif // ENABLE_HIP
#endif
