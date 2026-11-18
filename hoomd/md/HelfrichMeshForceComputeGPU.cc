// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "HelfrichMeshForceComputeGPU.h"

using namespace std;

/*! \file HelfrichMeshForceComputeGPU.cc
    \brief Contains code for the HelfrichMeshForceComputeGPU class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute forces on
    \param meshdef Mesh triangulation
    \post Memory is allocated, and forces are zeroed.
*/
HelfrichMeshForceComputeGPU::HelfrichMeshForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                                         std::shared_ptr<MeshDefinition> meshdef)
    : HelfrichMeshForceCompute(sysdef, meshdef)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error()
            << "Creating a HelfrichMeshForceComputeGPU with no GPU in the execution configuration"
            << endl;
        throw std::runtime_error("Error initializing HelfrichMeshForceComputeGPU");
        }

    m_tuner_force.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                         m_exec_conf,
                                         "helfrich_force"));

    m_tuner_sigma.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                         m_exec_conf,
                                         "helfrich_sigma"));

    m_autotuners.insert(m_autotuners.end(), {m_tuner_force, m_tuner_sigma});
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void HelfrichMeshForceComputeGPU::computeForces(uint64_t timestep)
    {
    resizeSigmaArrays();
    precomputeParameter();
    communicateSigma();

    // access the particle data arrays
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(),
                                     access_location::device,
                                     access_mode::read);

    ArrayHandle<Scalar> d_sigma(m_sigma, access_location::device, access_mode::read);
    ArrayHandle<Scalar3> d_sigma_dash(m_sigma_dash, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_normal(m_normal, access_location::device, access_mode::read);

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
    ArrayHandle<helfrich_param_t> d_params(m_params, access_location::device, access_mode::read);

    m_tuner_force->begin();
    kernel::gpu_compute_helfrich_force(d_force.data,
                                       d_virial.data,
                                       m_virial.getPitch(),
                                       m_pdata->getN(),
                                       d_pos.data,
                                       d_rtag.data,
                                       box,
                                       d_sigma.data,
                                       d_sigma_dash.data,
                                       d_normal.data,
                                       d_gpu_meshbondlist.data,
                                       gpu_table_indexer,
                                       d_gpu_meshbond_pos_list.data,
                                       d_gpu_n_meshbond.data,
                                       d_params.data,
                                       m_mesh_data->getMeshBondData()->getNTypes(),
                                       m_has_spontaneous_curvature,
                                       m_tuner_force->getParam()[0]);

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_force->end();
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void HelfrichMeshForceComputeGPU::precomputeParameter()
    {
    const GPUArray<typename MeshBond::members_t>& gpu_meshbond_list
        = this->m_mesh_data->getMeshBondData()->getGPUTable();

    // access the particle data arrays
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_rtag(m_pdata->getRTags(),
                                     access_location::device,
                                     access_mode::read);

    ArrayHandle<Scalar> d_sigma(m_sigma, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar3> d_sigma_dash(m_sigma_dash,
                                      access_location::device,
                                      access_mode::readwrite);
    ArrayHandle<Scalar4> d_normal(m_normal, access_location::device, access_mode::readwrite);

    BoxDim box = this->m_pdata->getGlobalBox();

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

    m_tuner_sigma->begin();
    kernel::gpu_compute_helfrich_sigma(d_sigma.data,
                                       d_sigma_dash.data,
                                       d_normal.data,
                                       m_pdata->getN(),
                                       d_pos.data,
                                       d_rtag.data,
                                       box,
                                       d_gpu_meshbondlist.data,
                                       gpu_table_indexer,
                                       d_gpu_meshbond_pos_list.data,
                                       d_gpu_n_meshbond.data,
                                       m_has_spontaneous_curvature || m_force_normals,
                                       m_tuner_sigma->getParam()[0]);

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        {
        CHECK_CUDA_ERROR();
        }

    m_tuner_sigma->end();
    }

namespace detail
    {
void export_HelfrichMeshForceComputeGPU(pybind11::module& m)
    {
    pybind11::class_<HelfrichMeshForceComputeGPU,
                     HelfrichMeshForceCompute,
                     std::shared_ptr<HelfrichMeshForceComputeGPU>>(m, "HelfrichMeshForceComputeGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<MeshDefinition>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
