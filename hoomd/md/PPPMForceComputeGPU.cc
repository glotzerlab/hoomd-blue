// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "PPPMForceComputeGPU.h"

#ifdef ENABLE_HIP
#include "PPPMForceComputeGPU.cuh"

namespace hoomd
    {
namespace md
    {
/*! \param sysdef The system definition
    \param nlist Neighbor list
    \param group Particle group to apply forces to
 */
PPPMForceComputeGPU::PPPMForceComputeGPU(std::shared_ptr<SystemDefinition> sysdef,
                                         std::shared_ptr<NeighborList> nlist,
                                         std::shared_ptr<ParticleGroup> group)
    : PPPMForceCompute(sysdef, nlist, group), m_local_fft(true), m_sum(m_exec_conf),
      m_block_size(256)
    {
    m_tuner_assign.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                          m_exec_conf,
                                          "pppm_assign"));
    m_tuner_reduce_mesh.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                               m_exec_conf,
                                               "pppm_reduce_mesh"));
    m_tuner_update.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                          m_exec_conf,
                                          "pppm_update_mesh"));
    m_tuner_force.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                         m_exec_conf,
                                         "pppm_force"));
    m_tuner_influence.reset(new Autotuner<1>({AutotunerBase::makeBlockSizeRange(m_exec_conf)},
                                             m_exec_conf,
                                             "pppm_influence"));

    m_autotuners.insert(
        m_autotuners.end(),
        {m_tuner_assign, m_tuner_reduce_mesh, m_tuner_update, m_tuner_force, m_tuner_influence});

    m_cufft_initialized = false;
    m_cuda_dfft_initialized = false;
    }

PPPMForceComputeGPU::~PPPMForceComputeGPU()
    {
    if (m_local_fft && m_cufft_initialized)
        {
#ifdef __HIP_PLATFORM_HCC__
        CHECK_HIPFFT_ERROR(hipfftDestroy(m_hipfft_plan));
#else
        CHECK_HIPFFT_ERROR(cufftDestroy(m_hipfft_plan));
#endif
        }
#ifdef ENABLE_MPI
    else if (m_cuda_dfft_initialized)
        {
        dfft_destroy_plan(m_dfft_plan_forward);
        dfft_destroy_plan(m_dfft_plan_inverse);
        }
#endif
    }

void PPPMForceComputeGPU::initializeFFT()
    {
    // free plans if they have already been initialized
    if (m_local_fft && m_cufft_initialized)
        {
#ifdef __HIP_PLATFORM_HCC__
        CHECK_HIPFFT_ERROR(hipfftDestroy(m_hipfft_plan));
#else
        CHECK_HIPFFT_ERROR(cufftDestroy(m_hipfft_plan));
#endif
        }
#ifdef ENABLE_MPI
    else if (m_cuda_dfft_initialized)
        {
        dfft_destroy_plan(m_dfft_plan_forward);
        dfft_destroy_plan(m_dfft_plan_inverse);
        }
#endif

#ifdef ENABLE_MPI
    m_local_fft = !m_pdata->getDomainDecomposition();

    if (!m_local_fft)
        {
        // ghost cell communicator for charge interpolation
        m_gpu_grid_comm_forward
            = std::shared_ptr<CommunicatorGridGPUComplex>(new CommunicatorGridGPUComplex(
                m_sysdef,
                make_uint3(m_mesh_points.x, m_mesh_points.y, m_mesh_points.z),
                make_uint3(m_grid_dim.x, m_grid_dim.y, m_grid_dim.z),
                m_n_ghost_cells,
                true));
        // ghost cell communicator for force mesh
        m_gpu_grid_comm_reverse
            = std::shared_ptr<CommunicatorGridGPUComplex>(new CommunicatorGridGPUComplex(
                m_sysdef,
                make_uint3(m_mesh_points.x, m_mesh_points.y, m_mesh_points.z),
                make_uint3(m_grid_dim.x, m_grid_dim.y, m_grid_dim.z),
                m_n_ghost_cells,
                false));

        // set up distributed FFT
        int gdim[3];
        int pdim[3];
        Index3D decomp_idx = m_pdata->getDomainDecomposition()->getDomainIndexer();
        pdim[0] = decomp_idx.getD();
        pdim[1] = decomp_idx.getH();
        pdim[2] = decomp_idx.getW();
        gdim[0] = m_mesh_points.z * pdim[0];
        gdim[1] = m_mesh_points.y * pdim[1];
        gdim[2] = m_mesh_points.x * pdim[2];
        int embed[3];
        embed[0] = m_mesh_points.z + 2 * m_n_ghost_cells.z;
        embed[1] = m_mesh_points.y + 2 * m_n_ghost_cells.y;
        embed[2] = m_mesh_points.x + 2 * m_n_ghost_cells.x;
        m_ghost_offset
            = (m_n_ghost_cells.z * embed[1] + m_n_ghost_cells.y) * embed[2] + m_n_ghost_cells.x;
        uint3 pcoord = m_pdata->getDomainDecomposition()->getGridPos();
        int pidx[3];
        pidx[0] = pcoord.z;
        pidx[1] = pcoord.y;
        pidx[2] = pcoord.x;
        int row_m = 0; /* both local grid and proc grid are row major, no transposition necessary */
        ArrayHandle<unsigned int> h_cart_ranks(m_pdata->getDomainDecomposition()->getCartRanks(),
                                               access_location::host,
                                               access_mode::read);
#ifndef USE_HOST_DFFT
        dfft_cuda_create_plan(&m_dfft_plan_forward,
                              3,
                              gdim,
                              embed,
                              NULL,
                              pdim,
                              pidx,
                              row_m,
                              0,
                              1,
                              m_exec_conf->getMPICommunicator(),
                              (int*)h_cart_ranks.data);
        dfft_cuda_create_plan(&m_dfft_plan_inverse,
                              3,
                              gdim,
                              NULL,
                              embed,
                              pdim,
                              pidx,
                              row_m,
                              0,
                              1,
                              m_exec_conf->getMPICommunicator(),
                              (int*)h_cart_ranks.data);
#else
        dfft_create_plan(&m_dfft_plan_forward,
                         3,
                         gdim,
                         embed,
                         NULL,
                         pdim,
                         pidx,
                         row_m,
                         0,
                         1,
                         m_exec_conf->getMPICommunicator(),
                         (int*)h_cart_ranks.data);
        dfft_create_plan(&m_dfft_plan_inverse,
                         3,
                         gdim,
                         NULL,
                         embed,
                         pdim,
                         pidx,
                         row_m,
                         0,
                         1,
                         m_exec_conf->getMPICommunicator(),
                         (int*)h_cart_ranks.data);
#endif

        m_cuda_dfft_initialized = true;
        }
#endif // ENABLE_MPI

    if (m_local_fft)
        {
// create plan on every device
#ifdef __HIP_PLATFORM_HCC__
        CHECK_HIPFFT_ERROR(hipfftPlan3d(&m_hipfft_plan,
                                        m_mesh_points.z,
                                        m_mesh_points.y,
                                        m_mesh_points.x,
                                        HIPFFT_C2C));
#else
        CHECK_HIPFFT_ERROR(cufftPlan3d(&m_hipfft_plan,
                                       m_mesh_points.z,
                                       m_mesh_points.y,
                                       m_mesh_points.x,
                                       CUFFT_C2C));
#endif
        m_cufft_initialized = true;
        }

    // allocate mesh and transformed mesh

    unsigned int ngpu = m_exec_conf->getNumActiveGPUs();

    if (ngpu > 1)
        {
        unsigned int mesh_elements = (m_n_cells + m_ghost_offset);
        GlobalArray<hipfftComplex> mesh_scratch(mesh_elements * ngpu, m_exec_conf);
        m_mesh_scratch.swap(mesh_scratch);

#ifdef __HIP_PLATFORM_NVCC__
        auto gpu_map = m_exec_conf->getGPUIds();
        for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            cudaMemAdvise(m_mesh_scratch.get() + idev * mesh_elements,
                          mesh_elements * sizeof(hipfftComplex),
                          cudaMemAdviseSetPreferredLocation,
                          gpu_map[idev]);
            cudaMemPrefetchAsync(m_mesh_scratch.get() + idev * mesh_elements,
                                 mesh_elements * sizeof(hipfftComplex),
                                 gpu_map[idev]);
            CHECK_CUDA_ERROR();
            }

        // accessed by GPU 0
        cudaMemAdvise(m_mesh_scratch.get(),
                      mesh_elements * ngpu,
                      cudaMemAdviseSetAccessedBy,
                      gpu_map[0]);
        CHECK_CUDA_ERROR();
#endif
        }

    // pad with offset
    GlobalArray<hipfftComplex> mesh(m_n_cells + m_ghost_offset, m_exec_conf);
    m_mesh.swap(mesh);

    // pad with offset
    unsigned int inv_mesh_elements = m_n_cells + m_ghost_offset;
    GlobalArray<hipfftComplex> inv_fourier_mesh_x(inv_mesh_elements, m_exec_conf);
    m_inv_fourier_mesh_x.swap(inv_fourier_mesh_x);

    GlobalArray<hipfftComplex> inv_fourier_mesh_y(inv_mesh_elements, m_exec_conf);
    m_inv_fourier_mesh_y.swap(inv_fourier_mesh_y);

    GlobalArray<hipfftComplex> inv_fourier_mesh_z(inv_mesh_elements, m_exec_conf);
    m_inv_fourier_mesh_z.swap(inv_fourier_mesh_z);

#ifdef __HIP_PLATFORM_NVCC__
    if (m_exec_conf->allConcurrentManagedAccess())
        {
        auto gpu_map = m_exec_conf->getGPUIds();
        for (unsigned int idev = 0; idev < m_exec_conf->getNumActiveGPUs(); ++idev)
            {
            cudaMemAdvise(m_inv_fourier_mesh_x.get(),
                          inv_mesh_elements * sizeof(hipfftComplex),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_inv_fourier_mesh_y.get(),
                          inv_mesh_elements * sizeof(hipfftComplex),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            cudaMemAdvise(m_inv_fourier_mesh_z.get(),
                          inv_mesh_elements * sizeof(hipfftComplex),
                          cudaMemAdviseSetAccessedBy,
                          gpu_map[idev]);
            CHECK_CUDA_ERROR();
            }
        }
#endif

    unsigned int n_blocks
        = (m_mesh_points.x * m_mesh_points.y * m_mesh_points.z) / m_block_size + 1;
    GlobalArray<Scalar> sum_partial(n_blocks, m_exec_conf);
    m_sum_partial.swap(sum_partial);

    GlobalArray<Scalar> sum_virial_partial(6 * n_blocks, m_exec_conf);
    m_sum_virial_partial.swap(sum_virial_partial);

    GlobalArray<Scalar> sum_virial(6, m_exec_conf);
    m_sum_virial.swap(sum_virial);
    }

//! Assignment of particles to mesh using three-point scheme (triangular shaped cloud)
/*! This is a second order accurate scheme with continuous value and continuous derivative
 */
void PPPMForceComputeGPU::assignParticles()
    {
    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(),
                                   access_location::device,
                                   access_mode::read);
    ArrayHandle<hipfftComplex> d_mesh(m_mesh, access_location::device, access_mode::overwrite);
    ArrayHandle<hipfftComplex> d_mesh_scratch(m_mesh_scratch,
                                              access_location::device,
                                              access_mode::overwrite);
    ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::read);

    // access the group
    ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);
    unsigned int group_size = m_group->getNumMembers();

    // access interpolation polynomial coefficients
    ArrayHandle<Scalar> d_rho_coeff(m_rho_coeff, access_location::device, access_mode::read);

    this->m_exec_conf->beginMultiGPU();

    m_tuner_assign->begin();
    unsigned int block_size = m_tuner_assign->getParam()[0];

    kernel::gpu_assign_particles(m_mesh_points,
                                 m_n_ghost_cells,
                                 m_grid_dim,
                                 group_size,
                                 d_index_array.data,
                                 d_postype.data,
                                 d_charge.data,
                                 d_mesh.data,
                                 d_mesh_scratch.data,
                                 (unsigned int)m_mesh.getNumElements(),
                                 m_order,
                                 m_pdata->getBox(),
                                 block_size,
                                 d_rho_coeff.data,
                                 m_exec_conf->dev_prop,
                                 m_group->getGPUPartition());

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_assign->end();

    this->m_exec_conf->endMultiGPU();

    if (m_exec_conf->getNumActiveGPUs() > 1)
        {
        m_tuner_reduce_mesh->begin();
        kernel::gpu_reduce_meshes((unsigned int)m_mesh.getNumElements(),
                                  d_mesh_scratch.data,
                                  d_mesh.data,
                                  m_exec_conf->getNumActiveGPUs(),
                                  m_tuner_reduce_mesh->getParam()[0]);
        m_tuner_reduce_mesh->end();

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }
    }

void PPPMForceComputeGPU::updateMeshes()
    {
    if (m_local_fft)
        {
        // locally transform the particle mesh
        ArrayHandle<hipfftComplex> d_mesh(m_mesh, access_location::device, access_mode::read);

#ifdef __HIP_PLATFORM_HCC__
        CHECK_HIPFFT_ERROR(hipfftExecC2C(m_hipfft_plan, d_mesh.data, d_mesh.data, HIPFFT_FORWARD));
#else
        CHECK_HIPFFT_ERROR(cufftExecC2C(m_hipfft_plan, d_mesh.data, d_mesh.data, CUFFT_FORWARD));
#endif
        }
#ifdef ENABLE_MPI
    else
        {
        // update inner cells of particle mesh
        m_exec_conf->msg->notice(8) << "charge.pppm: Ghost cell update" << std::endl;
        m_gpu_grid_comm_forward->communicate(m_mesh);

        // perform a distributed FFT
        m_exec_conf->msg->notice(8) << "charge.pppm: Distributed FFT mesh" << std::endl;
#ifndef USE_HOST_DFFT
        ArrayHandle<hipfftComplex> d_mesh(m_mesh, access_location::device, access_mode::read);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            dfft_cuda_check_errors(&m_dfft_plan_forward, 1);
        else
            dfft_cuda_check_errors(&m_dfft_plan_forward, 0);

        dfft_cuda_execute(d_mesh.data + m_ghost_offset,
                          d_mesh.data + m_ghost_offset,
                          0,
                          &m_dfft_plan_forward);
#else
        ArrayHandle<hipfftComplex> h_mesh(m_mesh, access_location::host, access_mode::read);

        dfft_execute((cpx_t*)(h_mesh.data + m_ghost_offset),
                     (cpx_t*)(h_fourier_mesh.data + m_ghost_offset),
                     0,
                     m_dfft_plan_forward);
#endif
        }
#endif

        {
        ArrayHandle<hipfftComplex> d_mesh(m_mesh, access_location::device, access_mode::readwrite);
        ArrayHandle<hipfftComplex> d_inv_fourier_mesh_x(m_inv_fourier_mesh_x,
                                                        access_location::device,
                                                        access_mode::overwrite);
        ArrayHandle<hipfftComplex> d_inv_fourier_mesh_y(m_inv_fourier_mesh_y,
                                                        access_location::device,
                                                        access_mode::overwrite);
        ArrayHandle<hipfftComplex> d_inv_fourier_mesh_z(m_inv_fourier_mesh_z,
                                                        access_location::device,
                                                        access_mode::overwrite);

        ArrayHandle<Scalar> d_inf_f(m_inf_f, access_location::device, access_mode::read);
        ArrayHandle<Scalar3> d_k(m_k, access_location::device, access_mode::read);

        unsigned int block_size = m_tuner_update->getParam()[0];
        m_tuner_update->begin();
        kernel::gpu_update_meshes(m_n_inner_cells,
                                  d_mesh.data + m_ghost_offset,
                                  d_inv_fourier_mesh_x.data + m_ghost_offset,
                                  d_inv_fourier_mesh_y.data + m_ghost_offset,
                                  d_inv_fourier_mesh_z.data + m_ghost_offset,
                                  d_inf_f.data,
                                  d_k.data,
                                  m_global_dim.x * m_global_dim.y * m_global_dim.z,
                                  block_size);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        m_tuner_update->end();
        }

    if (m_local_fft)
        {
        // do local inverse transform of all three components of the force mesh
        ArrayHandle<hipfftComplex> d_inv_fourier_mesh_x(m_inv_fourier_mesh_x,
                                                        access_location::device,
                                                        access_mode::overwrite);
        ArrayHandle<hipfftComplex> d_inv_fourier_mesh_y(m_inv_fourier_mesh_y,
                                                        access_location::device,
                                                        access_mode::overwrite);
        ArrayHandle<hipfftComplex> d_inv_fourier_mesh_z(m_inv_fourier_mesh_z,
                                                        access_location::device,
                                                        access_mode::overwrite);

        // do inverse FFT in-place

        m_exec_conf->beginMultiGPU();

#ifdef __HIP_PLATFORM_HCC__
        CHECK_HIPFFT_ERROR(hipfftExecC2C(m_hipfft_plan,
                                         d_inv_fourier_mesh_x.data,
                                         d_inv_fourier_mesh_x.data,
                                         HIPFFT_BACKWARD));
        CHECK_HIPFFT_ERROR(hipfftExecC2C(m_hipfft_plan,
                                         d_inv_fourier_mesh_y.data,
                                         d_inv_fourier_mesh_y.data,
                                         HIPFFT_BACKWARD));
        CHECK_HIPFFT_ERROR(hipfftExecC2C(m_hipfft_plan,
                                         d_inv_fourier_mesh_z.data,
                                         d_inv_fourier_mesh_z.data,
                                         HIPFFT_BACKWARD));
#else
        CHECK_HIPFFT_ERROR(cufftExecC2C(m_hipfft_plan,
                                        d_inv_fourier_mesh_x.data,
                                        d_inv_fourier_mesh_x.data,
                                        CUFFT_INVERSE));
        CHECK_HIPFFT_ERROR(cufftExecC2C(m_hipfft_plan,
                                        d_inv_fourier_mesh_y.data,
                                        d_inv_fourier_mesh_y.data,
                                        CUFFT_INVERSE));
        CHECK_HIPFFT_ERROR(cufftExecC2C(m_hipfft_plan,
                                        d_inv_fourier_mesh_z.data,
                                        d_inv_fourier_mesh_z.data,
                                        CUFFT_INVERSE));
#endif
        m_exec_conf->endMultiGPU();
        }
#ifdef ENABLE_MPI
    else
        {
        // Distributed inverse transform of force mesh
        m_exec_conf->msg->notice(8) << "charge.pppm: Distributed iFFT" << std::endl;
#ifndef USE_HOST_DFFT
        ArrayHandle<hipfftComplex> d_inv_fourier_mesh_x(m_inv_fourier_mesh_x,
                                                        access_location::device,
                                                        access_mode::overwrite);
        ArrayHandle<hipfftComplex> d_inv_fourier_mesh_y(m_inv_fourier_mesh_y,
                                                        access_location::device,
                                                        access_mode::overwrite);
        ArrayHandle<hipfftComplex> d_inv_fourier_mesh_z(m_inv_fourier_mesh_z,
                                                        access_location::device,
                                                        access_mode::overwrite);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            dfft_cuda_check_errors(&m_dfft_plan_inverse, 1);
        else
            dfft_cuda_check_errors(&m_dfft_plan_inverse, 0);

        dfft_cuda_execute(d_inv_fourier_mesh_x.data + m_ghost_offset,
                          d_inv_fourier_mesh_x.data + m_ghost_offset,
                          1,
                          &m_dfft_plan_inverse);
        dfft_cuda_execute(d_inv_fourier_mesh_y.data + m_ghost_offset,
                          d_inv_fourier_mesh_y.data + m_ghost_offset,
                          1,
                          &m_dfft_plan_inverse);
        dfft_cuda_execute(d_inv_fourier_mesh_z.data + m_ghost_offset,
                          d_inv_fourier_mesh_z.data + m_ghost_offset,
                          1,
                          &m_dfft_plan_inverse);
#else
        ArrayHandle<hipfftComplex> h_inv_fourier_mesh_x(m_inv_fourier_mesh_x,
                                                        access_location::host,
                                                        access_mode::overwrite);
        ArrayHandle<hipfftComplex> h_inv_fourier_mesh_y(m_inv_fourier_mesh_y,
                                                        access_location::host,
                                                        access_mode::overwrite);
        ArrayHandle<hipfftComplex> h_inv_fourier_mesh_z(m_inv_fourier_mesh_z,
                                                        access_location::host,
                                                        access_mode::overwrite);
        dfft_execute((cpx_t*)h_inv_fourier_mesh_x.data + m_ghost_offset,
                     (cpx_t*)h_inv_fourier_mesh_x.data + m_ghost_offset,
                     1,
                     m_dfft_plan_inverse);
        dfft_execute((cpx_t*)h_inv_fourier_mesh_y.data + m_ghost_offset,
                     (cpx_t*)h_inv_fourier_mesh_y.data + m_ghost_offset,
                     1,
                     m_dfft_plan_inverse);
        dfft_execute((cpx_t*)h_inv_fourier_mesh_z.data + m_ghost_offset,
                     (cpx_t*)h_inv_fourier_mesh_z.data + m_ghost_offset,
                     1,
                     m_dfft_plan_inverse);
#endif
        }
#endif

#ifdef ENABLE_MPI
    if (!m_local_fft)
        {
        // update outer cells of inverse Fourier meshes using ghost cells from neighboring
        // processors
        m_exec_conf->msg->notice(8) << "charge.pppm: Ghost cell update" << std::endl;
        m_gpu_grid_comm_reverse->communicate(m_inv_fourier_mesh_x);
        m_gpu_grid_comm_reverse->communicate(m_inv_fourier_mesh_y);
        m_gpu_grid_comm_reverse->communicate(m_inv_fourier_mesh_z);
        }
#endif
    }

void PPPMForceComputeGPU::interpolateForces()
    {
    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(),
                                   access_location::device,
                                   access_mode::read);
    ArrayHandle<hipfftComplex> d_inv_fourier_mesh_x(m_inv_fourier_mesh_x,
                                                    access_location::device,
                                                    access_mode::read);
    ArrayHandle<hipfftComplex> d_inv_fourier_mesh_y(m_inv_fourier_mesh_y,
                                                    access_location::device,
                                                    access_mode::read);
    ArrayHandle<hipfftComplex> d_inv_fourier_mesh_z(m_inv_fourier_mesh_z,
                                                    access_location::device,
                                                    access_mode::read);
    ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::read);

    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::overwrite);

    // access the group
    ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);

    // access polynomial interpolation coefficients
    ArrayHandle<Scalar> d_rho_coeff(m_rho_coeff, access_location::device, access_mode::read);

    m_exec_conf->beginMultiGPU();

    unsigned int block_size = m_tuner_force->getParam()[0];
    m_tuner_force->begin();
    kernel::gpu_compute_forces(m_pdata->getN(),
                               d_postype.data,
                               d_force.data,
                               d_inv_fourier_mesh_x.data,
                               d_inv_fourier_mesh_y.data,
                               d_inv_fourier_mesh_z.data,
                               m_grid_dim,
                               m_n_ghost_cells,
                               d_charge.data,
                               m_pdata->getBox(),
                               m_order,
                               d_index_array.data,
                               m_group->getGPUPartition(),
                               m_pdata->getGPUPartition(),
                               d_rho_coeff.data,
                               block_size,
                               m_local_fft,
                               m_n_cells + m_ghost_offset);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    m_tuner_force->end();

    m_exec_conf->endMultiGPU();
    }

void PPPMForceComputeGPU::computeVirial()
    {
    ArrayHandle<hipfftComplex> d_mesh(m_mesh, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_inf_f(m_inf_f, access_location::device, access_mode::read);
    ArrayHandle<Scalar3> d_k(m_k, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_virial_mesh(m_virial_mesh,
                                      access_location::device,
                                      access_mode::overwrite);

    bool exclude_dc = true;
#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        uint3 my_pos = m_pdata->getDomainDecomposition()->getGridPos();
        exclude_dc = !my_pos.x && !my_pos.y && !my_pos.z;
        }
#endif

    kernel::gpu_compute_mesh_virial(m_n_inner_cells,
                                    d_mesh.data + m_ghost_offset,
                                    d_inf_f.data,
                                    d_virial_mesh.data,
                                    d_k.data,
                                    exclude_dc,
                                    m_kappa);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

        {
        ArrayHandle<Scalar> d_sum_virial(m_sum_virial,
                                         access_location::device,
                                         access_mode::overwrite);
        ArrayHandle<Scalar> d_sum_virial_partial(m_sum_virial_partial,
                                                 access_location::device,
                                                 access_mode::overwrite);

        kernel::gpu_compute_virial(m_n_inner_cells,
                                   d_sum_virial_partial.data,
                                   d_sum_virial.data,
                                   d_virial_mesh.data,
                                   m_block_size);

        if (m_exec_conf->isCUDAErrorCheckingEnabled())
            CHECK_CUDA_ERROR();
        }

    ArrayHandle<Scalar> h_sum_virial(m_sum_virial, access_location::host, access_mode::read);

    Scalar V = m_pdata->getGlobalBox().getVolume();
    Scalar scale = Scalar(1.0) / ((Scalar)(m_global_dim.x * m_global_dim.y * m_global_dim.z));

    for (unsigned int i = 0; i < 6; ++i)
        m_external_virial[i] = Scalar(0.5) * V * scale * scale * h_sum_virial.data[i];
    }

Scalar PPPMForceComputeGPU::computePE()
    {
    ArrayHandle<hipfftComplex> d_mesh(m_mesh, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_inf_f(m_inf_f, access_location::device, access_mode::read);

    ArrayHandle<Scalar> d_sum_partial(m_sum_partial,
                                      access_location::device,
                                      access_mode::overwrite);

    bool exclude_dc = true;
#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        uint3 my_pos = m_pdata->getDomainDecomposition()->getGridPos();
        exclude_dc = !my_pos.x && !my_pos.y && !my_pos.z;
        }
#endif

    kernel::gpu_compute_pe(m_n_inner_cells,
                           d_sum_partial.data,
                           m_sum.getDeviceFlags(),
                           d_mesh.data + m_ghost_offset,
                           d_inf_f.data,
                           m_block_size,
                           m_mesh_points,
                           exclude_dc);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    Scalar sum = m_sum.readFlags();

    Scalar V = m_pdata->getGlobalBox().getVolume();
    Scalar scale = Scalar(1.0) / ((Scalar)(m_global_dim.x * m_global_dim.y * m_global_dim.z));
    sum *= Scalar(0.5) * V * scale * scale;

    if (m_exec_conf->getRank() == 0)
        {
        // subtract self-energy on rank 0 (see Frenkel and Smit, and Salin and Caillol)
        sum -= m_q2
               * (m_kappa / sqrt(Scalar(M_PI))
                      * exp(-m_alpha * m_alpha / (Scalar(4.0) * m_kappa * m_kappa))
                  - Scalar(0.5) * m_alpha * erfc(m_alpha / (Scalar(2.0) * m_kappa)));

        // k = 0 term already accounted for by exclude_dc
        // sum -= Scalar(0.5*M_PI)*m_q*m_q / (m_kappa*m_kappa* V);
        }

    // apply rigid body correction
    sum += m_body_energy;

    // store this rank's contribution as external potential energy
    m_external_energy = sum;

#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        // reduce sum
        MPI_Allreduce(MPI_IN_PLACE,
                      &sum,
                      1,
                      MPI_HOOMD_SCALAR,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        }
#endif

    return sum;
    }

//! Compute the optimal influence function
void PPPMForceComputeGPU::computeInfluenceFunction()
    {
    ArrayHandle<Scalar> d_inf_f(m_inf_f, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar3> d_k(m_k, access_location::device, access_mode::overwrite);

    uint3 global_dim = m_mesh_points;
    uint3 pidx = make_uint3(0, 0, 0);
    uint3 pdim = make_uint3(0, 0, 0);
#ifdef ENABLE_MPI
    if (m_pdata->getDomainDecomposition())
        {
        const Index3D& didx = m_pdata->getDomainDecomposition()->getDomainIndexer();
        global_dim.x *= didx.getW();
        global_dim.y *= didx.getH();
        global_dim.z *= didx.getD();
        pidx = m_pdata->getDomainDecomposition()->getGridPos();
        pdim = make_uint3(didx.getW(), didx.getH(), didx.getD());
        }
#endif

    ArrayHandle<Scalar> d_gf_b(m_gf_b, access_location::device, access_mode::read);

    unsigned int block_size = m_tuner_influence->getParam()[0];
    m_tuner_influence->begin();
    kernel::gpu_compute_influence_function(m_mesh_points,
                                           global_dim,
                                           d_inf_f.data,
                                           d_k.data,
                                           m_pdata->getGlobalBox(),
                                           m_local_fft,
                                           pidx,
                                           pdim,
                                           EPS_HOC,
                                           m_kappa,
                                           m_alpha,
                                           d_gf_b.data,
                                           m_order,
                                           block_size);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();

    m_tuner_influence->end();
    }

void PPPMForceComputeGPU::fixExclusions()
    {
    ArrayHandle<unsigned int> d_exlist(m_nlist->getExListArray(),
                                       access_location::device,
                                       access_mode::read);
    ArrayHandle<unsigned int> d_n_ex(m_nlist->getNExArray(),
                                     access_location::device,
                                     access_mode::read);
    ArrayHandle<Scalar4> d_force(m_force, access_location::device, access_mode::readwrite);
    ArrayHandle<Scalar> d_virial(m_virial, access_location::device, access_mode::readwrite);
    ArrayHandle<unsigned int> d_index_array(m_group->getIndexArray(),
                                            access_location::device,
                                            access_mode::read);
    unsigned int group_size = m_group->getNumMembers();

    ArrayHandle<Scalar4> d_postype(m_pdata->getPositions(),
                                   access_location::device,
                                   access_mode::read);
    ArrayHandle<Scalar> d_charge(m_pdata->getCharges(), access_location::device, access_mode::read);

    // reset virial
    hipMemset(d_virial.data, 0, sizeof(Scalar) * m_virial.getNumElements());

    Index2D nex = m_nlist->getExListIndexer();

    kernel::gpu_fix_exclusions(d_force.data,
                               d_virial.data,
                               m_virial.getPitch(),
                               m_pdata->getN() + m_pdata->getNGhosts(),
                               d_postype.data,
                               d_charge.data,
                               m_pdata->getBox(),
                               d_n_ex.data,
                               d_exlist.data,
                               nex,
                               m_kappa,
                               m_alpha,
                               d_index_array.data,
                               group_size,
                               m_block_size);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    }

namespace detail
    {
void export_PPPMForceComputeGPU(pybind11::module& m)
    {
    pybind11::class_<PPPMForceComputeGPU, PPPMForceCompute, std::shared_ptr<PPPMForceComputeGPU>>(
        m,
        "PPPMForceComputeGPU")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>,
                            std::shared_ptr<NeighborList>,
                            std::shared_ptr<ParticleGroup>>());
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd

#endif // ENABLE_HIP
