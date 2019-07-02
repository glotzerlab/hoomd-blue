// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: mphoward

/*! \file NeighborListGPUStencil.cc
    \brief Defines NeighborListGPUStencil
*/

#include "NeighborListGPUStencil.h"
#include "NeighborListGPUStencil.cuh"

namespace py = pybind11;

#ifdef ENABLE_MPI
#include "hoomd/Communicator.h"
#endif

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
                                               Scalar r_cut,
                                               Scalar r_buff,
                                               std::shared_ptr<CellList> cl,
                                               std::shared_ptr<CellListStencil> cls)
    : NeighborListGPU(sysdef, r_cut, r_buff), m_cl(cl), m_cls(cls), m_override_cell_width(false),
      m_needs_restencil(true), m_needs_resort(true)
    {
    m_exec_conf->msg->notice(5) << "Constructing NeighborListGPUStencil" << std::endl;

    // create a default cell list if one was not specified
    if (!m_cl)
        m_cl = std::shared_ptr<CellList>(new CellList(sysdef));

    // construct the default cell list stencil generator for the current cell list if one was not specified already
    if (!m_cls)
        m_cls = std::shared_ptr<CellListStencil>(new CellListStencil(m_sysdef, m_cl));

    m_cl->setRadius(1);
    // types are always required now
    m_cl->setComputeTDB(true);
    m_cl->setFlagIndex();
    m_cl->setComputeAdjList(false);

    CHECK_CUDA_ERROR();

    // initialize autotuner
    // the full block size and threads_per_particle matrix is searched,
    // encoded as block_size*10000 + threads_per_particle
    std::vector<unsigned int> valid_params;

    const unsigned int max_tpp = m_exec_conf->dev_prop.warpSize;
    for (unsigned int block_size = 32; block_size <= 1024; block_size += 32)
        {
        unsigned int s=1;

        while (s <= max_tpp)
            {
            valid_params.push_back(block_size*10000 + s);
            s = s * 2;
            }
        }

    m_tuner.reset(new Autotuner(valid_params, 5, 100000, "nlist_stencil", this->m_exec_conf));
    m_last_tuned_timestep = 0;

    #ifdef ENABLE_MPI
    // synchronize over MPI
    m_tuner->setSync(bool(m_pdata->getDomainDecomposition()));
    #endif

    // call this class's special setRCut
    setRCut(r_cut, r_buff);

    getRCutChangeSignal().connect<NeighborListGPUStencil, &NeighborListGPUStencil::slotRCutChange>(this);
    m_pdata->getMaxParticleNumberChangeSignal().connect<NeighborListGPUStencil, &NeighborListGPUStencil::slotMaxNumChanged>(this);
    m_pdata->getParticleSortSignal().connect<NeighborListGPUStencil, &NeighborListGPUStencil::slotParticleSort>(this);

    // needs realloc on size change...
    GPUArray<unsigned int> pid_map(m_pdata->getMaxN(), m_exec_conf);
    m_pid_map.swap(pid_map);
    }

NeighborListGPUStencil::~NeighborListGPUStencil()
    {
    m_exec_conf->msg->notice(5) << "Destroying NeighborListGPUStencil" << std::endl;
    getRCutChangeSignal().disconnect<NeighborListGPUStencil, &NeighborListGPUStencil::slotRCutChange>(this);
    m_pdata->getMaxParticleNumberChangeSignal().disconnect<NeighborListGPUStencil, &NeighborListGPUStencil::slotMaxNumChanged>(this);
    m_pdata->getParticleSortSignal().disconnect<NeighborListGPUStencil, &NeighborListGPUStencil::slotParticleSort>(this);
    }

void NeighborListGPUStencil::setRCut(Scalar r_cut, Scalar r_buff)
    {
    NeighborListGPU::setRCut(r_cut, r_buff);

    if (!m_override_cell_width)
        {
        Scalar rmin = getMinRCut() + m_r_buff;
        if (m_diameter_shift)
            rmin += m_d_max - Scalar(1.0);

        m_cl->setNominalWidth(rmin);
        }
    }

void NeighborListGPUStencil::setRCutPair(unsigned int typ1, unsigned int typ2, Scalar r_cut)
    {
    NeighborListGPU::setRCutPair(typ1,typ2,r_cut);

    if (!m_override_cell_width)
        {
        Scalar rmin = getMinRCut() + m_r_buff;
        if (m_diameter_shift)
            rmin += m_d_max - Scalar(1.0);

        m_cl->setNominalWidth(rmin);
        }
    }

void NeighborListGPUStencil::setMaximumDiameter(Scalar d_max)
    {
    NeighborListGPU::setMaximumDiameter(d_max);

    if (!m_override_cell_width)
        {
        Scalar rmin = getMinRCut() + m_r_buff;
        if (m_diameter_shift)
            rmin += m_d_max - Scalar(1.0);

        m_cl->setNominalWidth(rmin);
        }
    }

void NeighborListGPUStencil::updateRStencil()
    {
    ArrayHandle<Scalar> h_rcut_max(m_rcut_max, access_location::host, access_mode::read);
    std::vector<Scalar> rstencil(m_pdata->getNTypes(), -1.0);
    for (unsigned int cur_type=0; cur_type < m_pdata->getNTypes(); ++cur_type)
        {
        Scalar rcut = h_rcut_max.data[cur_type];
        if (rcut > Scalar(0.0))
            {
            Scalar rlist = rcut + m_r_buff;
            if (m_diameter_shift)
                rlist += m_d_max - Scalar(1.0);
            rstencil[cur_type] = rlist;
            }
        }
    m_cls->setRStencil(rstencil);
    }

/*!
 * Rearranges the particle indexes by type to reduce execution divergence during the neighbor list build.
 * Radix sort is (supposed to be) stable so that the spatial sorting from SFC is also preserved within a type.
 */
void NeighborListGPUStencil::sortTypes()
    {
    if (m_prof) m_prof->push(m_exec_conf, "sort");

    // always just fill in the particle indexes from 1 to N
    ArrayHandle<unsigned int> d_pids(m_pid_map, access_location::device, access_mode::overwrite);
    ScopedAllocation<unsigned int> d_pids_alt(m_exec_conf->getCachedAllocator(), m_pdata->getN());
    ScopedAllocation<unsigned int> d_types(m_exec_conf->getCachedAllocator(), m_pdata->getN());
    ScopedAllocation<unsigned int> d_types_alt(m_exec_conf->getCachedAllocator(), m_pdata->getN());

    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    gpu_compute_nlist_stencil_fill_types(d_pids.data, d_types(), d_pos.data, m_pdata->getN());

    // only sort with more than one type
    if (m_pdata->getNTypes() > 1)
        {
        // perform the sort
        void *d_tmp_storage = NULL;
        size_t tmp_storage_bytes = 0;
        bool swap = false;
        gpu_compute_nlist_stencil_sort_types(d_pids.data, d_pids_alt(), d_types(), d_types_alt(), d_tmp_storage, tmp_storage_bytes, swap, m_pdata->getN());

        size_t alloc_size = (tmp_storage_bytes > 0) ? tmp_storage_bytes : 4;
        // unsigned char = 1 B
        ScopedAllocation<unsigned char> d_alloc(m_exec_conf->getCachedAllocator(), alloc_size);
        d_tmp_storage = (void *)d_alloc();

        gpu_compute_nlist_stencil_sort_types(d_pids.data, d_pids_alt(), d_types(), d_types_alt(), d_tmp_storage, tmp_storage_bytes, swap, m_pdata->getN());

        if (swap)
            {
            cudaMemcpy(d_pids.data, d_pids_alt(), sizeof(unsigned int)*m_pdata->getN(), cudaMemcpyDeviceToDevice);
            }
        }

    if (m_prof) m_prof->pop(m_exec_conf);
    }

void NeighborListGPUStencil::buildNlist(unsigned int timestep)
    {
    if (m_storage_mode != full)
        {
        m_exec_conf->msg->error() << "Only full mode nlists can be generated on the GPU" << std::endl;
        throw std::runtime_error("Error computing neighbor list");
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

    if (m_prof)
        m_prof->push(m_exec_conf, "compute");

    // acquire the particle data
    ArrayHandle<unsigned int> d_pid_map(m_pid_map, access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_pos(m_pdata->getPositions(), access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_diameter(m_pdata->getDiameters(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_body(m_pdata->getBodies(), access_location::device, access_mode::read);

    const BoxDim& box = m_pdata->getBox();
    Scalar3 nearest_plane_distance = box.getNearestPlaneDistance();

    // access the cell list data arrays
    ArrayHandle<unsigned int> d_cell_size(m_cl->getCellSizeArray(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_cell_xyzf(m_cl->getXYZFArray(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_cell_tdb(m_cl->getTDBArray(), access_location::device, access_mode::read);
    ArrayHandle<Scalar4> d_stencil(m_cls->getStencils(), access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_n_stencil(m_cls->getStencilSizes(), access_location::device, access_mode::read);
    const Index2D& stencil_idx = m_cls->getStencilIndexer();

    ArrayHandle<unsigned int> d_head_list(m_head_list, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_Nmax(m_Nmax, access_location::device, access_mode::read);
    ArrayHandle<unsigned int> d_conditions(m_conditions, access_location::device, access_mode::readwrite);
    ArrayHandle<unsigned int> d_nlist(m_nlist, access_location::device, access_mode::overwrite);
    ArrayHandle<unsigned int> d_n_neigh(m_n_neigh, access_location::device, access_mode::overwrite);
    ArrayHandle<Scalar4> d_last_pos(m_last_pos, access_location::device, access_mode::overwrite);

    // the maximum cutoff that any particle can participate in
    Scalar rmax = getMaxRCut() + m_r_buff;
    if (m_diameter_shift)
        rmax += m_d_max - Scalar(1.0);

    if (m_filter_body)
        {
        // add the maximum diameter of all composite particles
        Scalar max_d_comp = m_pdata->getMaxCompositeParticleDiameter();
        rmax += 0.5*max_d_comp;
        }

    ArrayHandle<Scalar> d_r_cut(m_r_cut, access_location::device, access_mode::read);
    ArrayHandle<Scalar> d_r_listsq(m_r_listsq, access_location::device, access_mode::read);

    if ((box.getPeriodic().x && nearest_plane_distance.x <= rmax * 2.0) ||
        (box.getPeriodic().y && nearest_plane_distance.y <= rmax * 2.0) ||
        (this->m_sysdef->getNDimensions() == 3 && box.getPeriodic().z && nearest_plane_distance.z <= rmax * 2.0))
        {
        m_exec_conf->msg->error() << "nlist: Simulation box is too small! Particles would be interacting with themselves." << std::endl;
        throw std::runtime_error("Error updating neighborlist bins");
        }

    // we should not call the tuner with MPI sync enabled
    // if the kernel is launched more than once in the same timestep,
    // since those kernel launches may occur only on some, not all MPI ranks
    bool tune = m_last_tuned_timestep != timestep;

    if (tune) m_tuner->begin();
    unsigned int param = m_tuner->getParam();
    unsigned int block_size = param / 10000;
    unsigned int threads_per_particle = param % 10000;

    // launch neighbor list kernel
    gpu_compute_nlist_stencil(d_nlist.data,
                              d_n_neigh.data,
                              d_last_pos.data,
                              d_conditions.data,
                              d_Nmax.data,
                              d_head_list.data,
                              d_pid_map.data,
                              d_pos.data,
                              d_body.data,
                              d_diameter.data,
                              m_pdata->getN(),
                              d_cell_size.data,
                              d_cell_xyzf.data,
                              d_cell_tdb.data,
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
                              m_diameter_shift,
                              threads_per_particle,
                              block_size);

    if(m_exec_conf->isCUDAErrorCheckingEnabled()) CHECK_CUDA_ERROR();
    if (tune) this->m_tuner->end();

    m_last_tuned_timestep = timestep;

    if (m_prof)
        m_prof->pop(m_exec_conf);
    }

void export_NeighborListGPUStencil(py::module& m)
    {
    py::class_<NeighborListGPUStencil, std::shared_ptr<NeighborListGPUStencil> >(m, "NeighborListGPUStencil", py::base<NeighborListGPU>())
        .def(py::init< std::shared_ptr<SystemDefinition>, Scalar, Scalar, std::shared_ptr<CellList>, std::shared_ptr<CellListStencil> >())
        .def("setCellWidth", &NeighborListGPUStencil::setCellWidth);
    }
