// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file NeighborListGPUStencil.cc
    \brief Defines NeighborListGPUStencil
*/

#include "NeighborListGPUStencil.h"
#include "NeighborListGPUStencil.cuh"

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif

namespace hoomd
    {
namespace md
    {
/*!
 * \param sysdef System definition
 * \param r_cut Default cutoff radius
 * \param r_buff Neighbor list buffer width
 * \param cl Cell list
 * \param cls Cell list stencil
 *
 * A default cell list and stencil will be constructed if \a cl or \a cls are not instantiated.
 */
NeighborListGPUStencil::NeighborListGPUStencil(std::shared_ptr<SystemDefinition> sysdef,
                                               Scalar r_buff)
    : NeighborListGPU(sysdef, r_buff), m_cl(std::make_shared<CellListGPU>(sysdef)),
      m_cls(std::make_shared<CellListStencil>(sysdef, m_cl)), m_needs_resort(true)
    {
    m_exec_conf->msg->notice(5) << "Constructing NeighborListGPUStencil" << std::endl;

    m_cl->setRadius(1);
    // types are always required now
    m_cl->setComputeTypeBody(true);
    m_cl->setFlagIndex();
    m_cl->setComputeAdjList(false);

    CHECK_CUDA_ERROR();

    // Initialize autotuner.
    m_tuner.reset(new Autotuner<2>({AutotunerBase::makeBlockSizeRange(m_exec_conf),
                                    AutotunerBase::getTppListPow2(m_exec_conf)},
                                   m_exec_conf,
                                   "nlist_stencil"));
    m_autotuners.push_back(m_tuner);
    m_last_tuned_timestep = 0;

#ifdef ENABLE_MPI
    // synchronize over MPI
    m_tuner->setSync(bool(m_pdata->getDomainDecomposition()));
#endif

    // cell sizes need update by default
    m_update_cell_size = true;

    m_pdata->getMaxParticleNumberChangeSignal()
        .connect<NeighborListGPUStencil, &NeighborListGPUStencil::slotMaxNumChanged>(this);
    m_pdata->getParticleSortSignal()
        .connect<NeighborListGPUStencil, &NeighborListGPUStencil::slotParticleSort>(this);

    // needs realloc on size change...
    GPUArray<unsigned int> pid_map(m_pdata->getMaxN(), m_exec_conf);
    m_pid_map.swap(pid_map);
    }

NeighborListGPUStencil::~NeighborListGPUStencil()
    {
    m_exec_conf->msg->notice(5) << "Destroying NeighborListGPUStencil" << std::endl;
    m_pdata->getMaxParticleNumberChangeSignal()
        .disconnect<NeighborListGPUStencil, &NeighborListGPUStencil::slotMaxNumChanged>(this);
    m_pdata->getParticleSortSignal()
        .disconnect<NeighborListGPUStencil, &NeighborListGPUStencil::slotParticleSort>(this);
    }

void NeighborListGPUStencil::updateRStencil()
    {
    ArrayHandle<Scalar> h_rcut_max(m_rcut_max, access_location::host, access_mode::read);
    std::vector<Scalar> rstencil(m_pdata->getNTypes(), -1.0);
    for (unsigned int cur_type = 0; cur_type < m_pdata->getNTypes(); ++cur_type)
        {
        Scalar rcut = h_rcut_max.data[cur_type];
        if (rcut > Scalar(0.0))
            {
            Scalar rlist = rcut + m_r_buff;
            rstencil[cur_type] = rlist;
            }
        }
    m_cls->setRStencil(rstencil);
    }

/*!
 * Rearranges the particle indexes by type to reduce execution divergence during the neighbor list
 * build. Radix sort is (supposed to be) stable so that the spatial sorting from SFC is also
 * preserved within a type.
 */
void NeighborListGPUStencil::sortTypes()
    {
    // always just fill in the particle indexes from 1 to N
    ArrayHandle<unsigned int> d_pids(m_pid_map, access_location::device, access_mode::overwrite);
    ScopedAllocation<unsigned int> d_pids_alt(m_exec_conf->getCachedAllocator(), m_pdata->getN());
    ScopedAllocation<unsigned int> d_types(m_exec_conf->getCachedAllocator(), m_pdata->getN());
    ScopedAllocation<unsigned int> d_types_alt(m_exec_conf->getCachedAllocator(), m_pdata->getN());

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    kernel::gpu_compute_nlist_stencil_fill_types(d_pids.data,
                                                 d_types(),
                                                 d_pos.data,
                                                 m_pdata->getN());

    // only sort with more than one type
    if (m_pdata->getNTypes() > 1)
        {
        // perform the sort
        void* d_tmp_storage = NULL;
        size_t tmp_storage_bytes = 0;
        bool swap = false;
        kernel::gpu_compute_nlist_stencil_sort_types(d_pids.data,
                                                     d_pids_alt(),
                                                     d_types(),
                                                     d_types_alt(),
                                                     d_tmp_storage,
                                                     tmp_storage_bytes,
                                                     swap,
                                                     m_pdata->getN());

        size_t alloc_size = (tmp_storage_bytes > 0) ? tmp_storage_bytes : 4;
        // unsigned char = 1 B
        ScopedAllocation<unsigned char> d_alloc(m_exec_conf->getCachedAllocator(), alloc_size);
        d_tmp_storage = (void*)d_alloc();

        kernel::gpu_compute_nlist_stencil_sort_types(d_pids.data,
                                                     d_pids_alt(),
                                                     d_types(),
                                                     d_types_alt(),
                                                     d_tmp_storage,
                                                     tmp_storage_bytes,
                                                     swap,
                                                     m_pdata->getN());

        if (swap)
            {
            hipMemcpy(d_pids.data,
                      d_pids_alt(),
                      sizeof(unsigned int) * m_pdata->getN(),
                      hipMemcpyDeviceToDevice);
            }
        }
    }

void NeighborListGPUStencil::buildNlist(uint64_t timestep)
    {
    if (m_storage_mode != full)
        {
        throw std::runtime_error("GPU neighbor lists require a full storage mode.");
        }

    if (m_update_cell_size)
        {
        if (!m_override_cell_width)
            {
            Scalar rmin = getMinRCut() + m_r_buff;
            m_cl->setNominalWidth(rmin);
            }
        m_update_cell_size = false;
        }

    m_cl->compute(timestep);

    // update the stencil radii if there was a change
    if (m_needs_restencil)
        {
        updateRStencil();
        m_needs_restencil = false;
        }
    m_cls->compute(timestep);

    // sort the particles by type
    if (m_needs_resort)
        {
        sortTypes();
        m_needs_resort = false;
        }

    // acquire the particle data
    ArrayHandle<unsigned int> d_pid_map(m_pid_map, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_body(m_pdata->getBodies(),
                                     access_location::device,
                                     access_mode::read);

    const BoxDim& box = m_pdata->getBox();
    Scalar3 nearest_plane_distance = box.getNearestPlaneDistance();

    // access the cell list data arrays
    ArrayHandle<unsigned int> d_cell_size(m_cl->getCellSizeArray(),
                                          access_location::device,
                                          access_mode::read);
    ArrayHandle<Scalar4> d_cell_xyzf(m_cl->getXYZFArray(),
                                     access_location::device,
                                     access_mode::read);
    ArrayHandle<uint2> d_cell_type_body(m_cl->getTypeBodyArray(),
                                        access_location::device,
                                        access_mode::read);
    ArrayHandle<Scalar4> d_stencil(m_cls->getStencils(),
                                   access_location::device,
                                   access_mode::read);
    ArrayHandle<unsigned int> d_n_stencil(m_cls->getStencilSizes(),
                                          access_location::device,
                                          access_mode::read);
    const Index2D& stencil_idx = m_cls->getStencilIndexer();

    ArrayHandle<size_t> d_head_list(m_head_list, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_Nmax(m_Nmax, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_conditions(m_conditions,
                                           access_location::device,
                                           access_mode::readwrite);
    ArrayHandle<unsigned int> d_nlist(m_nlist, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_n_neigh(m_n_neigh, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_last_pos(m_last_pos, access_location::device, access_mode::overwrite);

    // the maximum cutoff that any particle can participate in
    Scalar rmax = getMaxRCut() + m_r_buff;

    ArrayHandle<Scalar> d_r_cut(m_r_cut, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_r_listsq(m_r_listsq, access_location::device, access_mode::read);

    if ((box.getPeriodic().x && nearest_plane_distance.x <= rmax * 2.0)
        || (box.getPeriodic().y && nearest_plane_distance.y <= rmax * 2.0)
        || (this->m_sysdef->getNDimensions() == 3 && box.getPeriodic().z
            && nearest_plane_distance.z <= rmax * 2.0))
        {
        std::ostringstream oss;
        oss << "nlist: Simulation box is too small! Particles would be interacting with themselves."
            << "rmax=" << rmax << std::endl;

        if (box.getPeriodic().x)
            oss << "nearest_plane_distance.x=" << nearest_plane_distance.x << std::endl;
        if (box.getPeriodic().y)
            oss << "nearest_plane_distance.y=" << nearest_plane_distance.y << std::endl;
        if (this->m_sysdef->getNDimensions() == 3 && box.getPeriodic().z)
            oss << "nearest_plane_distance.z=" << nearest_plane_distance.z << std::endl;
        throw std::runtime_error(oss.str());
        }

    // we should not call the tuner with MPI sync enabled
    // if the kernel is launched more than once in the same timestep,
    // since those kernel launches may occur only on some, not all MPI ranks
    bool tune = m_last_tuned_timestep != timestep;

    if (tune)
        m_tuner->begin();
    auto param = m_tuner->getParam();
    unsigned int block_size = param[0];
    unsigned int threads_per_particle = param[1];

    // launch neighbor list kernel
    kernel::gpu_compute_nlist_stencil(d_nlist.data,
                                      d_n_neigh.data,
                                      d_last_pos.data,
                                      d_conditions.data,
                                      d_Nmax.data,
                                      d_head_list.data,
                                      d_pid_map.data,
                                      d_pos.data,
                                      d_body.data,
                                      m_pdata->getN(),
                                      d_cell_size.data,
                                      d_cell_xyzf.data,
                                      d_cell_type_body.data,
                                      m_cl->getCellIndexer(),
                                      m_cl->getCellListIndexer(),
                                      d_stencil.data,
                                      d_n_stencil.data,
                                      stencil_idx,
                                      box,
                                      d_r_cut.data,
                                      m_r_buff,
                                      m_pdata->getNTypes(),
                                      m_cl->getGhostWidth(),
                                      m_filter_body,
                                      threads_per_particle,
                                      block_size,
                                      m_exec_conf->dev_prop);

    if (m_exec_conf->isCUDAErrorCheckingEnabled())
        CHECK_CUDA_ERROR();
    if (tune)
        this->m_tuner->end();

    m_last_tuned_timestep = timestep;
    }

namespace detail
    {
void export_NeighborListGPUStencil(pybind11::module& m)
    {
    pybind11::class_<NeighborListGPUStencil,
                     NeighborListGPU,
                     std::shared_ptr<NeighborListGPUStencil>>(m, "NeighborListGPUStencil")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, Scalar>())
        .def("setCellWidth", &NeighborListGPUStencil::setCellWidth);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
