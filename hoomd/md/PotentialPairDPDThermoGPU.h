// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __POTENTIAL_PAIR_DPDTHERMO_GPU_H__
#define __POTENTIAL_PAIR_DPDTHERMO_GPU_H__

#ifdef ENABLE_HIP

#include "PotentialPairDPDThermo.h"
#include "PotentialPairDPDThermoGPU.cuh"
#include "hoomd/Autotuner.h"

/*! \file PotentialPairDPDThermoGPU.h
    \brief Defines the template class for standard pair potentials on the GPU
    \note This header cannot be compiled by nvcc
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

namespace hoomd
    {
namespace md
    {
//! Template class for computing pair potentials on the GPU
/*! Derived from PotentialPair, this class provides exactly the same interface for computing pair
   potentials and forces. In the same way as PotentialPair, this class serves as a shell dealing
   with all the details common to every pair potential calculation while the \a evaluator calculates
   V(r) in a generic way.

    Due to technical limitations, the instantiation of PotentialPairDPDThermoGPU cannot create a
   CUDA kernel automatically with the \a evaluator. Instead, a .cu file must be written that
   provides a driver function to call gpu_compute_dpd_forces() instantiated with the same evaluator.
   (See PotentialPairDPDThermoGPU.cuh for an example). That function is then passed into this class
   as another template parameter \a gpu_cpdf

    \tparam evaluator EvaluatorPair class used to evaluate V(r) and F(r)/r
    \tparam gpu_cpdf Driver function that calls gpu_compute_dpd_forces<evaluator>()

    \sa export_PotentialPairDPDThermoGPU()
*/
template<class evaluator> class PotentialPairDPDThermoGPU : public PotentialPairDPDThermo<evaluator>
    {
    public:
    //! Construct the pair potential
    PotentialPairDPDThermoGPU(std::shared_ptr<SystemDefinition> sysdef,
                              std::shared_ptr<NeighborList> nlist);
    //! Destructor
    virtual ~PotentialPairDPDThermoGPU() {};

    protected:
    std::shared_ptr<Autotuner<2>> m_tuner; //!< Autotuner for block size and threads per particle

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

template<class evaluator>
PotentialPairDPDThermoGPU<evaluator>::PotentialPairDPDThermoGPU(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<NeighborList> nlist)
    : PotentialPairDPDThermo<evaluator>(sysdef, nlist)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!this->m_exec_conf->isCUDAEnabled())
        {
        this->m_exec_conf->msg->error()
            << "Creating a PotentialPairDPDThermoGPU with no GPU in the execution configuration"
            << std::endl;
        throw std::runtime_error("Error initializing PotentialPairDPDThermoGPU");
        }

    // Initialize autotuner.
    m_tuner.reset(new Autotuner<2>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf),
                                    AutotunerBase::getTppListPow2(this->m_exec_conf)},
                                   this->m_exec_conf,
                                   "pair_" + evaluator::getName()));
    this->m_autotuners.push_back(m_tuner);

#ifdef ENABLE_MPI
    // synchronize autotuner results across ranks
    m_tuner->setSync(bool(this->m_pdata->getDomainDecomposition()));
#endif
    }

template<class evaluator>
void PotentialPairDPDThermoGPU<evaluator>::computeForces(uint64_t timestep)
    {
    this->m_nlist->compute(timestep);

    // The GPU implementation CANNOT handle a half neighborlist, error out now
    bool third_law = this->m_nlist->getStorageMode() == NeighborList::half;
    if (third_law)
        {
        this->m_exec_conf->msg->error()
            << "PotentialPairDPDThermoGPU cannot handle a half neighborlist" << std::endl
            << std::endl;
        throw std::runtime_error("Error computing forces in PotentialPairDPDThermoGPU");
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
    ArrayHandle<Scalar4> d_vel(this->m_pdata->getVelocities(),
                               access_location::device,
                               access_mode::read);
    ArrayHandle<unsigned int> d_tag(this->m_pdata->getTags(),
                                    access_location::device,
                                    access_mode::read);

    BoxDim box = this->m_pdata->getBox();

    // access parameters
    ArrayHandle<Scalar> d_rcutsq(this->m_rcutsq, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(this->m_virial, access_location::device, access_mode::overwrite);

    // access flags
    PDataFlags flags = this->m_pdata->getFlags();

    m_tuner->begin();

    auto param = m_tuner->getParam();
    unsigned int block_size = param[0];
    unsigned int threads_per_particle = param[1];

    kernel::gpu_compute_dpd_forces<evaluator>(
        kernel::dpd_pair_args_t(d_force.data,
                                d_virial.data,
                                this->m_virial.getPitch(),
                                this->m_pdata->getN(),
                                this->m_pdata->getMaxN(),
                                d_pos.data,
                                d_vel.data,
                                d_tag.data,
                                box,
                                d_n_neigh.data,
                                d_nlist.data,
                                d_head_list.data,
                                d_rcutsq.data,
                                this->m_nlist->getNListArray().getPitch(),
                                this->m_pdata->getNTypes(),
                                block_size,
                                this->m_sysdef->getSeed(),
                                timestep,
                                this->m_deltaT,
                                (*this->m_T)(timestep),
                                this->m_shift_mode,
                                flags[pdata_flag::pressure_tensor],
                                threads_per_particle,
                                this->m_exec_conf->dev_prop),
        this->m_params.data());

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_tuner->end();
    }

namespace detail
    {
//! Export this pair potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Evaluator type to export.

*/
template<class T>
void export_PotentialPairDPDThermoGPU(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<PotentialPairDPDThermoGPU<T>,
                     PotentialPairDPDThermo<T>,
                     std::shared_ptr<PotentialPairDPDThermoGPU<T>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<NeighborList>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif // ENABLE_HIP
#endif // __POTENTIAL_PAIR_DPDTHERMO_GPU_H__
