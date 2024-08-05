// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "BendingRigidityMeshForceComputeGPU.h"

using namespace std;

/*! \file BendingRigidityMeshForceComputeGPU.cc
    \brief Contains code for the BendingRigidityMeshForceComputeGPU class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute forces on
    \param meshdef Mesh triangulation
    \post Memory is allocated, and forces are zeroed.
*/
BendingRigidityMeshForceComputeGPU::BendingRigidityMeshForceComputeGPU(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<MeshDefinition> meshdef)
    : BendingRigidityMeshForceCompute(sysdef, meshdef)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a BendingRigidityMeshForceComputeGPU with no GPU in "
                                     "the execution configuration"
                                  << endl;
        throw std::runtime_error("Error initializing BendingRigidityMeshForceComputeGPU");
        }

    // allocate and zero device memory
    GPUArray<Scalar> params(this->m_mesh_data->getMeshTriangleData()->getNTypes(),
                            this->m_exec_conf);
    m_params.swap(params);

    m_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                   m_exec_conf,
                                   "bending_rigidity"));
    m_autotuners.push_back(m_tuner);
    }

BendingRigidityMeshForceComputeGPU::~BendingRigidityMeshForceComputeGPU() { }

void BendingRigidityMeshForceComputeGPU::setParams(unsigned int type, Scalar K)
    {
    BendingRigidityMeshForceCompute::setParams(type, K);

    ArrayHandle<Scalar> h_params(m_params, access_location::host, access_mode::readwrite);
    // update the local copy of the memory
    h_params.data[type] = K;
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void BendingRigidityMeshForceComputeGPU::computeForces(uint64_t timestep)
    {
    // access the particle data arrays
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(),
                                     access_location::device,
                                     access_mode::read);

    BoxDim box = this->m_pdata->getGlobalBox();

    const GPUArray<typename MeshBond::members_t>& gpu_meshbond_list
        = this->m_mesh_data->getMeshBondData()->getGPUTable();
    const Index2D& gpu_table_indexer = this->m_mesh_data->getMeshBondData()->getGPUTableIndexer();

    ArrayHandle<typename MeshBond::members_t> d_gpu_meshbondlist(gpu_meshbond_list,
                                                                 access_location::device,
                                                                 access_mode::read);

    ArrayHandle<unsigned int> d_gpu_meshbond_pos_list(
        this->m_mesh_data->getMeshBondData()->getGPUPosTable(),
        access_location::device,
        access_mode::read);

    ArrayHandle<unsigned int> d_gpu_n_meshbond(
        this->m_mesh_data->getMeshBondData()->getNGroupsArray(),
        access_location::device,
        access_mode::read);

    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_params(m_params, access_location::device, access_mode::read);

    // run the kernel on the GPU
    m_tuner->begin();
    kernel::gpu_compute_bending_rigidity_force(d_force.data,
                                               d_virial.data,
                                               m_virial.getPitch(),
                                               m_pdata->getN(),
                                               d_pos.data,
                                               d_rtag.data,
                                               box,
                                               d_gpu_meshbondlist.data,
                                               gpu_table_indexer,
                                               d_gpu_meshbond_pos_list.data,
                                               d_gpu_n_meshbond.data,
                                               d_params.data,
                                               m_mesh_data->getMeshBondData()->getNTypes(),
                                               m_tuner->getParam()[0]);

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();
    }

namespace detail
    {
void export_BendingRigidityMeshForceComputeGPU(pybind11::module& m)
    {
    pybind11::class_<BendingRigidityMeshForceComputeGPU,
                     BendingRigidityMeshForceCompute,
                     std::shared_ptr<BendingRigidityMeshForceComputeGPU>>(
        m,
        "BendingRigidityMeshForceComputeGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<MeshDefinition>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
