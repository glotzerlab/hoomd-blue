// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __POTENTIAL_BOND_GPU_H__
#define __POTENTIAL_BOND_GPU_H__

#ifdef ENABLE_HIP

#include "PotentialBond.h"
#include "PotentialBondGPU.cuh"
#include "hoomd/Autotuner.h"

/*! \file PotentialBondGPU.h
    \brief Defines the template class for standard bond potentials on the GPU
    \note This header cannot be compiled by nvcc
*/

#ifdef __HIPCC__
#error This header cannot be compiled by nvcc
#endif

namespace hoomd
    {
namespace md
    {
//! Template class for computing bond potentials on the GPU
/*!
    \tparam evaluator EvaluatorBond class used to evaluate V(r) and F(r)/r

    \sa export_PotentialBondGPU()
*/
template<class evaluator, class Bonds>
class PotentialBondGPU : public PotentialBond<evaluator, Bonds>
    {
    public:
    //! Construct the bond potential
    PotentialBondGPU(std::shared_ptr<SystemDefinition> sysdef);
    //! Construct the mesh_bond potential
    PotentialBondGPU(std::shared_ptr<SystemDefinition> sysdef,
                     std::shared_ptr<MeshDefinition> meshdef);
    //! Destructor
    virtual ~PotentialBondGPU() { }

    protected:
    std::shared_ptr<Autotuner<1>> m_tuner; //!< Autotuner for block size
    GPUArray<unsigned int> m_flags;        //!< Flags set during the kernel execution

    //! Actually compute the forces
    virtual void computeForces(uint64_t timestep);
    };

template<class evaluator, class Bonds>
PotentialBondGPU<evaluator, Bonds>::PotentialBondGPU(std::shared_ptr<SystemDefinition> sysdef)
    : PotentialBond<evaluator, Bonds>(sysdef)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!this->m_exec_conf->isCUDAEnabled())
        {
        this->m_exec_conf->msg->error()
            << "Creating a PotentialBondGPU with no GPU in the execution configuration"
            << std::endl;
        throw std::runtime_error("Error initializing PotentialBondGPU");
        }

    // allocate and zero device memory
    GPUArray<typename evaluator::param_type> params(this->m_bond_data->getNTypes(),
                                                    this->m_exec_conf);
    this->m_params.swap(params);

    // allocate flags storage on the GPU
    GPUArray<unsigned int> flags(1, this->m_exec_conf);
    m_flags.swap(flags);

    // reset flags
    ArrayHandle<unsigned int> h_flags(m_flags, access_location::host, access_mode::overwrite);
    h_flags.data[0] = 0;

    m_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                                   this->m_exec_conf,
                                   "bond_" + evaluator::getName()));
    this->m_autotuners.push_back(m_tuner);
    }

template<class evaluator, class Bonds>
PotentialBondGPU<evaluator, Bonds>::PotentialBondGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                     std::shared_ptr<MeshDefinition> meshdef)
    : PotentialBond<evaluator, Bonds>(sysdef, meshdef)
    {
    // can't run on the GPU if there aren't any GPUs in the execution configuration
    if (!this->m_exec_conf->isCUDAEnabled())
        {
        this->m_exec_conf->msg->error()
            << "Creating a PotentialMeshBondGPU with no GPU in the execution configuration"
            << std::endl;
        throw std::runtime_error("Error initializing PotentialMeshBondGPU");
        }

    // allocate and zero device memory
    GPUArray<typename evaluator::param_type> params(this->m_bond_data->getNTypes(),
                                                    this->m_exec_conf);
    this->m_params.swap(params);

    // allocate flags storage on the GPU
    GPUArray<unsigned int> flags(1, this->m_exec_conf);
    m_flags.swap(flags);

    // reset flags
    ArrayHandle<unsigned int> h_flags(m_flags, access_location::host, access_mode::overwrite);
    h_flags.data[0] = 0;

    m_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(this->m_exec_conf)},
                                   this->m_exec_conf,
                                   "bond_" + evaluator::getName()));
    this->m_autotuners.push_back(m_tuner);
    }

template<class evaluator, class Bonds>
void PotentialBondGPU<evaluator, Bonds>::computeForces(uint64_t timestep)
    {
    // access the particle data
    ArrayHandle<Scalar4> d_pos(this->m_pdata->getPositions(),
                               access_location::device,
                               access_mode::read);
    ArrayHandle<Scalar> d_charge(this->m_pdata->getCharges(),
                                 access_location::device,
                                 access_mode::read);

    // we are using the minimum image of the global box here
    // to ensure that ghosts are always correctly wrapped (even if a bond exceeds half the domain
    // length)
    BoxDim box = this->m_pdata->getGlobalBox();

    // access parameters
    ArrayHandle<typename evaluator::param_type> d_params(this->m_params,
                                                         access_location::device,
                                                         access_mode::read);

    // access net force & virial
    ArrayHandle<Scalar4> d_force(this->m_force, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_virial(this->m_virial, access_location::device, access_mode::readwrite);

        {
        const GPUArray<typename Bonds::members_t>& gpu_bond_list = this->m_bond_data->getGPUTable();
        const Index2D& gpu_table_indexer = this->m_bond_data->getGPUTableIndexer();

        ArrayHandle<typename Bonds::members_t> d_gpu_bondlist(gpu_bond_list,
                                                              access_location::device,
                                                              access_mode::read);

        ArrayHandle<unsigned int> d_gpu_bond_pos_list(this->m_bond_data->getGPUPosTable(),
                                                      access_location::device,
                                                      access_mode::read);
        ArrayHandle<unsigned int> d_gpu_n_bonds(this->m_bond_data->getNGroupsArray(),
                                                access_location::device,
                                                access_mode::read);

        // access the flags array for overwriting
        ArrayHandle<unsigned int> d_flags(m_flags, access_location::device, access_mode::readwrite);

        this->m_tuner->begin();
        kernel::gpu_compute_bond_forces<evaluator, Bonds::size>(
            kernel::bond_args_t<Bonds::size>(d_force.data,
                                             d_virial.data,
                                             this->m_virial.getPitch(),
                                             this->m_pdata->getN(),
                                             this->m_pdata->getMaxN(),
                                             d_pos.data,
                                             d_charge.data,
                                             box,
                                             d_gpu_bondlist.data,
                                             gpu_table_indexer,
                                             d_gpu_bond_pos_list.data,
                                             d_gpu_n_bonds.data,
                                             this->m_bond_data->getNTypes(),
                                             this->m_tuner->getParam()[0],
                                             this->m_exec_conf->dev_prop),
            d_params.data,
            d_flags.data);
        }

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        {
        CHECK_CUDA_ERROR();

        // check the flags for any errors
        ArrayHandle<unsigned int> h_flags(m_flags, access_location::host, access_mode::read);

        if (h_flags.data[0] & 1)
            {
            this->m_exec_conf->msg->error()
                << "bond." << evaluator::getName() << ": bond out of bounds (" << h_flags.data[0]
                << ")" << std::endl
                << std::endl;
            throw std::runtime_error("Error in bond calculation");
            }
        }
    this->m_tuner->end();
    }

namespace detail
    {
//! Export this bond potential to python
/*! \param name Name of the class in the exported python module
    \tparam T Evaluator type to export.
*/
template<class T> void export_PotentialBondGPU(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<PotentialBondGPU<T, BondData>,
                     PotentialBond<T, BondData>,
                     std::shared_ptr<PotentialBondGPU<T, BondData>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
    }

template<class T> void export_PotentialMeshBondGPU(pybind11::module& m, const std::string& name)
    {
    pybind11::class_<PotentialBondGPU<T, MeshBondData>,
                     PotentialBond<T, MeshBondData>,
                     std::shared_ptr<PotentialBondGPU<T, MeshBondData>>>(m, name.c_str())
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<MeshDefinition>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif // ENABLE_HIP
#endif // __POTENTIAL_PAIR_GPU_H__
