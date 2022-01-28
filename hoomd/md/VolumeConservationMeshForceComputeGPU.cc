// Copyright (c) 2009-2022 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "VolumeConservationMeshForceComputeGPU.h"

using namespace std;

/*! \file VolumeConservationMeshForceComputeGPU.cc
    \brief Contains code for the VolumeConservationMeshForceComputeGPU class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute forces on
    \post Memory is allocated, and forces are zeroed.
*/
VolumeConservationMeshForceComputeGPU::VolumeConservationMeshForceComputeGPU(
    std::shared_ptr<SystemDefinition> sysdef,
    std::shared_ptr<MeshDefinition> meshdef)
    : VolumeConservationMeshForceCompute(sysdef, meshdef), m_block_size(256)
    {
    if (!m_exec_conf->isCUDAEnabled())
        {
        m_exec_conf->msg->error() << "Creating a VolumeConservationMeshForceComputeGPU with no GPU "
                                     "in the execution configuration"
                                  << endl;
        throw std::runtime_error("Error initializing VolumeConservationMeshForceComputeGPU");
        }

    // allocate and zero device memory
    GPUArray<Scalar2> params(this->m_angle_data->getNTypes(), m_exec_conf);
    m_params.swap(params);

    // allocate flags storage on the GPU
    GPUArray<unsigned int> flags(1, this->m_exec_conf);
    m_flags.swap(flags);

    GPUArray<Scalar> sum(1, m_exec_conf);
    m_sum.swap(sum);

    unsigned int group_size = m_pdata->getN();

    m_num_blocks = group_size / m_block_size + 1;
    GPUArray<Scalar> partial_sum(m_num_blocks, m_exec_conf);
    m_partial_sum.swap(partial_sum);

    // reset flags
    ArrayHandle<unsigned int> h_flags(m_flags, access_location::host, access_mode::overwrite);
    h_flags.data[0] = 0;

    unsigned int warp_size = this->m_exec_conf->dev_prop.warpSize;
    m_tuner.reset(new Autotuner(warp_size,
                                1024,
                                warp_size,
                                5,
                                100000,
                                "vconstraint_forces",
                                this->m_exec_conf));
    }

void VolumeConservationMeshForceComputeGPU::setParams(unsigned int type, Scalar K, Scalar V0)
    {
    VolumeConservationMeshForceCompute::setParams(type, K, V0);

    ArrayHandle<Scalar> h_params(m_params, access_location::host, access_mode::readwrite);
    // update the local copy of the memory
    h_params.data[type] = make_scalar2(K, V0);
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void VolumeConservationMeshForceComputeGPU::computeForces(uint64_t timestep)
    {
    // start the profile
    if (this->m_prof)
        this->m_prof->push(this->m_exec_conf, "VolumeConstraint");

    computeVolume();

    // access the particle data arrays
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::read);

    BoxDim box = this->m_pdata->getGlobalBox();

    const GPUArray<typename MeshTriangle::members_t>& gpu_meshtriangle_list
        = this->m_mesh_data->getMeshTriangleData()->getGPUTable();
    const Index2D& gpu_table_indexer
        = this->m_mesh_data->getMeshTriangleData()->getGPUTableIndexer();

    ArrayHandle<typename MeshTriangle::members_t> d_gpu_meshtrianglelist(gpu_meshtriangle_list,
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

    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_virial(m_virial, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar> d_params(m_params, access_location::device, access_mode::read);

    // access the flags array for overwriting
    ArrayHandle<unsigned int> d_flags(m_flags, access_location::device, access_mode::readwrite);

    m_tuner->begin();
    kernel::gpu_compute_volume_constraint_force(d_force.data,
                                                d_virial.data,
                                                m_virial.getPitch(),
                                                m_pdata->getN(),
                                                d_pos.data,
                                                d_image.data,
                                                box,
                                                m_volume,
                                                d_gpu_meshtrianglelist.data,
                                                d_gpu_meshtriangle_pos_list.data,
                                                gpu_table_indexer,
                                                d_gpu_n_meshtriangle.data,
                                                d_params.data,
                                                m_mesh_data->getMeshTriangleData()->getNTypes(),
                                                m_tuner->getParam(),
                                                d_flags.data);

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        {
        CHECK_CUDA_ERROR();

        // check the flags for any errors
        ArrayHandle<unsigned int> h_flags(m_flags, access_location::host, access_mode::read);

        if (h_flags.data[0] & 1)
            {
            this->m_exec_conf->msg->error() << "volume constraint: triangle out of bounds ("
                                            << h_flags.data[0] << ")" << std::endl
                                            << std::endl;
            throw std::runtime_error("Error in meshtriangle calculation");
            }
        }
    m_tuner->end();

    if (this->m_prof)
        this->m_prof->pop(this->m_exec_conf);
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void VolumeConservationMeshForceComputeGPU::computeVolume()
    {
    // start the profile
    if (this->m_prof)
        this->m_prof->push(this->m_exec_conf, "VolumeCalculation");

    // access the particle data arrays
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<int3> d_image(m_pdata->getImages(), access_location::device, access_mode::read);

    BoxDim box = this->m_pdata->getGlobalBox();

    m_num_blocks = m_pdata->getN() / m_block_size + 1;

    const GPUArray<typename MeshTriangle::members_t>& gpu_meshtriangle_list
        = this->m_mesh_data->getMeshTriangleData()->getGPUTable();
    const Index2D& gpu_table_indexer
        = this->m_mesh_data->getMeshTriangleData()->getGPUTableIndexer();

    ArrayHandle<typename MeshTriangle::members_t> d_gpu_meshtrianglelist(gpu_meshtriangle_list,
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

    ArrayHandle<Scalar> d_partial_sumVol(m_partial_sum,
                                         access_location::device,
                                         access_mode::overwrite);
    ArrayHandle<Scalar> d_sumVol(m_sum, access_location::device, access_mode::overwrite);

    kernel::gpu_compute_volume_constraint_volume(d_sumVol,
                                                 d_partial_sumVol m_pdata->getN(),
                                                 d_pos.data,
                                                 d_image.data,
                                                 box,
                                                 d_gpu_meshtrianglelist.data,
                                                 d_gpu_meshtriangle_pos_list,
                                                 gpu_table_indexer,
                                                 d_gpu_n_meshtriangle.data,
                                                 m_block_size,
                                                 m_num_blocks);

    if (this->m_exec_conf->isCUDAErrorCheckingEnabled())
        {
        CHECK_CUDA_ERROR();
        }

    ArrayHandle<Scalar> h_sumVol(m_sum, access_location::host, access_mode::read);
#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        MPI_Allreduce(MPI_IN_PLACE,
                      &h_sumVol.data[0],
                      1,
                      MPI_HOOMD_SCALAR,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        }
#endif
    m_volume = h_sumVol.data[0];

    if (this->m_prof)
        this->m_prof->pop(this->m_exec_conf);
    }

namespace detail
    {
void export_VolumeConservationMeshForceComputeGPU(pybind11::module& m)
    {
    pybind11::class_<VolumeConservationMeshForceComputeGPU,
                     VolumeConservationMeshForceCompute,
                     std::shared_ptr<VolumeConservationMeshForceComputeGPU>>(
        m,
        "VolumeConservationMeshForceComputeGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<MeshDefinition>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
