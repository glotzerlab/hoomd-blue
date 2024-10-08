// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "TriangleAreaConservationMeshForceComputeGPU.h"

using namespace std;

/*! \file TriangleAreaConservationMeshForceComputeGPU.cc
    \brief Contains code for the TriangleAreaConservationhMeshForceComputeGPU class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute forces on
    \param meshdef Mesh triangulation 
    \post Memory is allocated, and forces are zeroed.
*/
TriangleAreaConservationMeshForceComputeGPU::TriangleAreaConservationMeshForceComputeGPU(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<MeshDefinition> meshdef)
    : TriangleAreaConservationMeshForceCompute(sysdef, meshdef)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error()
            << "Creating a TriangleAreaConservationMeshForceComputeGPU with no GPU "
               "in the execution configuration"
            << endl;
        throw std::runtime_error("Error initializing TriangleAreaConservationMeshForceComputeGPU");
        }

    // allocate and zero device memory
    GPUArray<Scalar2> params(this->m_mesh_data->getMeshTriangleData()->getNTypes(),
                             this->m_exec_conf);
    m_params.swap(params);

    GPUArray<Scalar> sum(this->m_mesh_data->getMeshTriangleData()->getNTypes(), m_exec_conf);
    m_sum.swap(sum);

    m_block_size = 256;
    unsigned int group_size = m_pdata->getN();
    m_num_blocks = group_size / m_block_size;
    m_num_blocks += 1;
    m_num_blocks *= this->m_mesh_data->getMeshTriangleData()->getNTypes();
    GPUArray<Scalar> partial_sum(m_num_blocks, m_exec_conf);
    m_partial_sum.swap(partial_sum);

    m_tuner.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                   m_exec_conf,
                                   "taconstraint_force"));
    m_autotuners.push_back(m_tuner);
    }

void TriangleAreaConservationMeshForceComputeGPU::computeForces(uint64_t timestep)
    {
    // access the particle data arrays
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);

    const GPUArray<typename Angle::members_t>& gpu_meshtriangle_list
        = this->m_mesh_data->getMeshTriangleData()->getGPUTable();
    const Index2D& gpu_table_indexer
        = this->m_mesh_data->getMeshTriangleData()->getGPUTableIndexer();

    ArrayHandle<typename Angle::members_t> d_gpu_meshtrianglelist(gpu_meshtriangle_list,
                                                                  access_location::device,
                                                                  access_mode::read);

    ArrayHandle<unsigned int> d_gpu_meshtriangle_pos_list(
        m_mesh_data->getMeshTriangleData()->getGPUPosTable(),
        access_location::device,
        access_mode::read);

    ArrayHandle<unsigned int> d_gpu_n_meshtriangle(
        this->m_mesh_data->getMeshTriangleData()->getNGroupsArray(),
        access_location::device,
        access_mode::read);

    BoxDim box = this->m_pdata->getGlobalBox();

    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar2> d_params(m_params, access_location::device, access_mode::read);

    m_tuner->begin();
    kernel::gpu_compute_TriangleAreaConservation_force(
        d_force.data,
        d_virial.data,
        m_virial.getPitch(),
        m_pdata->getN(),
        d_pos.data,
        box,
        d_gpu_meshtrianglelist.data,
        d_gpu_meshtriangle_pos_list.data,
        gpu_table_indexer,
        d_gpu_n_meshtriangle.data,
        d_params.data,
        m_mesh_data->getMeshTriangleData()->getNTypes(),
        m_tuner->getParam()[0]);

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner->end();
    }

void TriangleAreaConservationMeshForceComputeGPU::computeArea()
    {
    // access the particle data arrays
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);

    const GPUArray<typename Angle::members_t>& gpu_meshtriangle_list
        = this->m_mesh_data->getMeshTriangleData()->getGPUTable();
    const Index2D& gpu_table_indexer
        = this->m_mesh_data->getMeshTriangleData()->getGPUTableIndexer();

    ArrayHandle<typename Angle::members_t> d_gpu_meshtrianglelist(gpu_meshtriangle_list,
                                                                  access_location::device,
                                                                  access_mode::read);

    ArrayHandle<unsigned int> d_gpu_n_meshtriangle(
        this->m_mesh_data->getMeshTriangleData()->getNGroupsArray(),
        access_location::device,
        access_mode::read);
    ArrayHandle<unsigned int> d_gpu_meshtriangle_pos_list(
        m_mesh_data->getMeshTriangleData()->getGPUPosTable(),
        access_location::device,
        access_mode::read);

    BoxDim box = this->m_pdata->getGlobalBox();

    m_num_blocks = m_pdata->getN() / m_block_size + 1;

    ArrayHandle<Scalar> d_partial_sumA(m_partial_sum,
                                       access_location::device,
                                       access_mode::overwrite);
    ArrayHandle<Scalar> d_sumA(m_sum, access_location::device, access_mode::overwrite);

    unsigned int NTypes = m_mesh_data->getMeshTriangleData()->getNTypes();

    kernel::gpu_compute_area_constraint_area(d_sumA.data,
                                             d_partial_sumA.data,
                                             m_pdata->getN(),
                                             NTypes,
                                             d_pos.data,
                                             box,
                                             d_gpu_meshtrianglelist.data,
                                             d_gpu_meshtriangle_pos_list.data,
                                             gpu_table_indexer,
					     false,
                                             d_gpu_n_meshtriangle.data,
                                             m_block_size,
                                             m_num_blocks);

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        {
        CHECK_CUDA_ERROR();
        }

    ArrayHandle<Scalar> h_sumA(m_sum, access_location::host, access_mode::read);
#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        MPI_Allreduce(MPI_IN_PLACE,
                      &h_sumA.data[0],
                      NTypes,
                      MPI_HOOMD_SCALAR,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        }
#endif
    for (unsigned int i = 0; i < m_mesh_data->getMeshTriangleData()->getNTypes(); i++)
        m_area[i] = h_sumA.data[i];
    }

namespace detail
    {
void export_TriangleAreaConservationMeshForceComputeGPU(pybind11::module& m)
    {
    pybind11::class_<TriangleAreaConservationMeshForceComputeGPU,
                     TriangleAreaConservationMeshForceCompute,
                     std::shared_ptr<TriangleAreaConservationMeshForceComputeGPU>>(
        m,
        "TriangleAreaConservationMeshForceComputeGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<MeshDefinition>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
