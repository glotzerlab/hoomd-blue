// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "CellList.h"

#ifdef ENABLE_MPI
#include "Communicator.h"
#include "hoomd/Communicator.h"
#endif // ENABLE_MPI
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

/*!
 * \file mpcd/CellList.cc
 * \brief Definition of mpcd::CellList
 */

namespace hoomd
    {
mpcd::CellList::CellList(std::shared_ptr<SystemDefinition> sysdef, Scalar cell_size, bool shift)
    : Compute(sysdef), m_mpcd_pdata(m_sysdef->getMPCDParticleData()), m_cell_np_max(4),
      m_cell_np(m_exec_conf), m_cell_list(m_exec_conf), m_embed_cell_ids(m_exec_conf),
      m_conditions(m_exec_conf), m_needs_compute_dim(true), m_particles_sorted(false),
      m_virtual_change(false)
    {
    assert(m_mpcd_pdata);
    m_exec_conf->msg->notice(5) << "Constructing MPCD CellList" << std::endl;

    setCellSize(cell_size);
    m_origin_idx = make_int3(0, 0, 0);
    m_cell_dim = make_uint3(0, 0, 0);

    m_enable_grid_shift = shift;
    m_grid_shift = make_scalar3(0.0, 0.0, 0.0);

    resetConditions();

#ifdef ENABLE_MPI
    m_decomposition = m_pdata->getDomainDecomposition();
    m_num_extra = 0;
    m_cover_box = m_pdata->getBox();
#endif // ENABLE_MPI

    m_mpcd_pdata->getSortSignal().connect<mpcd::CellList, &mpcd::CellList::sort>(this);
    m_mpcd_pdata->getNumVirtualSignal().connect<mpcd::CellList, &mpcd::CellList::slotNumVirtual>(
        this);
    m_pdata->getParticleSortSignal().connect<mpcd::CellList, &mpcd::CellList::slotSorted>(this);
    m_pdata->getBoxChangeSignal().connect<mpcd::CellList, &mpcd::CellList::slotBoxChanged>(this);
    }

mpcd::CellList::CellList(std::shared_ptr<SystemDefinition> sysdef,
                         const uint3& global_cell_dim,
                         bool shift)
    : Compute(sysdef), m_mpcd_pdata(m_sysdef->getMPCDParticleData()), m_cell_np_max(4),
      m_cell_np(m_exec_conf), m_cell_list(m_exec_conf), m_embed_cell_ids(m_exec_conf),
      m_conditions(m_exec_conf), m_needs_compute_dim(true), m_particles_sorted(false),
      m_virtual_change(false)
    {
    assert(m_mpcd_pdata);
    m_exec_conf->msg->notice(5) << "Constructing MPCD CellList" << std::endl;

    setGlobalDim(global_cell_dim);
    m_origin_idx = make_int3(0, 0, 0);
    m_cell_dim = make_uint3(0, 0, 0);

    m_enable_grid_shift = shift;
    m_grid_shift = make_scalar3(0.0, 0.0, 0.0);

    resetConditions();

#ifdef ENABLE_MPI
    m_decomposition = m_pdata->getDomainDecomposition();
    m_num_extra = 0;
    m_cover_box = m_pdata->getBox();
#endif // ENABLE_MPI

    m_mpcd_pdata->getSortSignal().connect<mpcd::CellList, &mpcd::CellList::sort>(this);
    m_mpcd_pdata->getNumVirtualSignal().connect<mpcd::CellList, &mpcd::CellList::slotNumVirtual>(
        this);
    m_pdata->getParticleSortSignal().connect<mpcd::CellList, &mpcd::CellList::slotSorted>(this);
    m_pdata->getBoxChangeSignal().connect<mpcd::CellList, &mpcd::CellList::slotBoxChanged>(this);
    }

mpcd::CellList::~CellList()
    {
    m_exec_conf->msg->notice(5) << "Destroying MPCD CellList" << std::endl;
    m_mpcd_pdata->getSortSignal().disconnect<mpcd::CellList, &mpcd::CellList::sort>(this);
    m_mpcd_pdata->getNumVirtualSignal().disconnect<mpcd::CellList, &mpcd::CellList::slotNumVirtual>(
        this);
    m_pdata->getParticleSortSignal().disconnect<mpcd::CellList, &mpcd::CellList::slotSorted>(this);
    m_pdata->getBoxChangeSignal().disconnect<mpcd::CellList, &mpcd::CellList::slotBoxChanged>(this);
    }

void mpcd::CellList::compute(uint64_t timestep)
    {
    Compute::compute(timestep);

    if (m_virtual_change)
        {
        m_virtual_change = false;
        m_force_compute = true;
        }

    if (m_particles_sorted)
        {
        m_particles_sorted = false;
        m_force_compute = true;
        }

    if (m_needs_compute_dim)
        {
        computeDimensions();
        m_force_compute = true;
        }

    if (peekCompute(timestep))
        {
        // ensure grid is shifted
        drawGridShift(timestep);

#ifdef ENABLE_MPI
        // exchange embedded particles if necessary
        if (m_sysdef->isDomainDecomposed() && needsEmbedMigrate(timestep))
            {
            auto comm = m_sysdef->getCommunicator().lock();
            if (!comm)
                {
                throw std::runtime_error("Embedded particle communicator needed but not set");
                }
            comm->forceMigrate();
            comm->communicate(timestep);
            }
#endif // ENABLE_MPI

        // resize to be able to hold the number of embedded particles
        if (m_embed_group)
            {
            m_embed_cell_ids.resize(m_embed_group->getNumMembers());
            }

        bool overflowed = false;
        do
            {
            buildCellList();

            overflowed = checkConditions();

            if (overflowed)
                {
                reallocate();
                resetConditions();
                }
            } while (overflowed);

        // we are finished building, explicitly mark everything (rather than using shouldCompute)
        m_first_compute = false;
        m_force_compute = false;
        m_last_computed = timestep;

        // signal to the ParticleData that the cell list cache is now valid
        m_mpcd_pdata->validateCellCache();
        }
    }

void mpcd::CellList::reallocate()
    {
    m_exec_conf->msg->notice(6) << "Allocating MPCD cell list, " << m_cell_np_max
                                << " particles in " << m_cell_indexer.getNumElements() << " cells."
                                << std::endl;
    m_cell_list_indexer = Index2D(m_cell_np_max, m_cell_indexer.getNumElements());
    m_cell_list.resize(m_cell_list_indexer.getNumElements());
    }

void mpcd::CellList::computeDimensions()
    {
    if (!m_needs_compute_dim)
        return;

#ifdef ENABLE_MPI
    uchar3 communicating = make_uchar3(0, 0, 0);
    if (m_decomposition)
        {
        const Index3D& di = m_decomposition->getDomainIndexer();
        communicating.x = (di.getW() > 1);
        communicating.y = (di.getH() > 1);
        communicating.z = (di.getD() > 1);
        }

    // Only do complicated sizing if some direction is being communicated
    if (communicating.x || communicating.y || communicating.z)
        {
        // Global simulation box for absolute position referencing
        const BoxDim& global_box = m_pdata->getGlobalBox();
        const BoxDim& local_box = m_pdata->getBox();
        const Scalar3 max_fractional_shift = Scalar(0.5) * m_global_cell_dim_inv;

        // setup lo bin
        const Scalar3 lo = local_box.getLo();
        const Scalar3 lo_shifted_down_fractional = global_box.makeFraction(lo - m_max_grid_shift);
        int3 my_lo_bin
            = make_int3((int)std::floor(lo_shifted_down_fractional.x * m_global_cell_dim.x),
                        (int)std::floor(lo_shifted_down_fractional.y * m_global_cell_dim.y),
                        (int)std::floor(lo_shifted_down_fractional.z * m_global_cell_dim.z));
        const Scalar3 lo_shifted_up_fractional = global_box.makeFraction(lo + m_max_grid_shift);
        int3 lo_neigh_bin
            = make_int3((int)std::ceil(lo_shifted_up_fractional.x * m_global_cell_dim.x),
                        (int)std::ceil(lo_shifted_up_fractional.y * m_global_cell_dim.y),
                        (int)std::ceil(lo_shifted_up_fractional.z * m_global_cell_dim.z));

        // setup hi bin
        const Scalar3 hi = local_box.getHi();
        const Scalar3 hi_shifted_up_fractional = global_box.makeFraction(hi + m_max_grid_shift);
        int3 my_hi_bin
            = make_int3((int)std::ceil(hi_shifted_up_fractional.x * m_global_cell_dim.x),
                        (int)std::ceil(hi_shifted_up_fractional.y * m_global_cell_dim.y),
                        (int)std::ceil(hi_shifted_up_fractional.z * m_global_cell_dim.z));
        const Scalar3 hi_shifted_down_fractional = global_box.makeFraction(hi - m_max_grid_shift);
        int3 hi_neigh_bin
            = make_int3((int)std::floor(hi_shifted_down_fractional.x * m_global_cell_dim.x),
                        (int)std::floor(hi_shifted_down_fractional.y * m_global_cell_dim.y),
                        (int)std::floor(hi_shifted_down_fractional.z * m_global_cell_dim.z));

        // initially size the grid assuming one rank in each direction, and then resize based on
        // communication
        m_cell_dim = m_global_cell_dim;
        m_origin_idx = make_int3(0, 0, 0);
        std::fill(m_num_comm.begin(), m_num_comm.end(), 0);

        // Compute size of the box with diffusion layer
        Scalar3 fractional_cover_lo = global_box.makeFraction(lo);
        Scalar3 fractional_cover_hi = global_box.makeFraction(hi);
        uchar3 cover_periodic = local_box.getPeriodic();

        if (communicating.x)
            {
            // number of cells and cell origin, padding with extra cells in diffusion layer
            m_cell_dim.x = my_hi_bin.x - my_lo_bin.x + 2 * m_num_extra;
            m_origin_idx.x = my_lo_bin.x - m_num_extra;

            // number of communication cells along each direction
            m_num_comm[static_cast<unsigned int>(mpcd::detail::face::east)]
                = my_hi_bin.x - hi_neigh_bin.x + m_num_extra;
            m_num_comm[static_cast<unsigned int>(mpcd::detail::face::west)]
                = lo_neigh_bin.x - my_lo_bin.x + m_num_extra;

            // "safe" size of the diffusion layer
            fractional_cover_lo.x
                = m_global_cell_dim_inv.x * m_origin_idx.x + max_fractional_shift.x;
            fractional_cover_hi.x = m_global_cell_dim_inv.x * (m_origin_idx.x + m_cell_dim.x)
                                    - max_fractional_shift.x;
            cover_periodic.x = 0;
            }

        if (communicating.y)
            {
            m_cell_dim.y = my_hi_bin.y - my_lo_bin.y + 2 * m_num_extra;
            m_origin_idx.y = my_lo_bin.y - m_num_extra;

            m_num_comm[static_cast<unsigned int>(mpcd::detail::face::north)]
                = my_hi_bin.y - hi_neigh_bin.y + m_num_extra;
            m_num_comm[static_cast<unsigned int>(mpcd::detail::face::south)]
                = lo_neigh_bin.y - my_lo_bin.y + m_num_extra;

            fractional_cover_lo.y
                = m_global_cell_dim_inv.y * m_origin_idx.y + max_fractional_shift.y;
            fractional_cover_hi.y = m_global_cell_dim_inv.y * (m_origin_idx.y + m_cell_dim.y)
                                    - max_fractional_shift.y;
            cover_periodic.y = 0;
            }

        if (m_sysdef->getNDimensions() == 3 && communicating.z)
            {
            m_cell_dim.z = my_hi_bin.z - my_lo_bin.z + 2 * m_num_extra;
            m_origin_idx.z = my_lo_bin.z - m_num_extra;

            m_num_comm[static_cast<unsigned int>(mpcd::detail::face::up)]
                = my_hi_bin.z - hi_neigh_bin.z + m_num_extra;
            m_num_comm[static_cast<unsigned int>(mpcd::detail::face::down)]
                = lo_neigh_bin.z - my_lo_bin.z + m_num_extra;

            fractional_cover_lo.z
                = m_global_cell_dim_inv.z * m_origin_idx.z + max_fractional_shift.z;
            fractional_cover_hi.z = m_global_cell_dim_inv.z * (m_origin_idx.z + m_cell_dim.z)
                                    - max_fractional_shift.z;
            cover_periodic.z = 0;
            }

        // set the box covered by this cell list
        const Scalar3 cover_lo = global_box.makeCoordinates(fractional_cover_lo);
        const Scalar3 cover_hi = global_box.makeCoordinates(fractional_cover_hi);
        m_cover_box = BoxDim(cover_lo, cover_hi, cover_periodic);
        }
    else
#endif // ENABLE_MPI
        {
        m_cell_dim = m_global_cell_dim;
        m_origin_idx = make_int3(0, 0, 0);
        }

    // resize the cell indexers and per-cell counter
    m_global_cell_indexer = Index3D(m_global_cell_dim.x, m_global_cell_dim.y, m_global_cell_dim.z);
    m_cell_indexer = Index3D(m_cell_dim.x, m_cell_dim.y, m_cell_dim.z);
    m_cell_np.resize(m_cell_indexer.getNumElements());

    // reallocate per-cell memory
    reallocate();

    // dimensions are now current
    m_needs_compute_dim = false;
    notifySizeChange();
    }

#ifdef ENABLE_MPI
/*!
 * \param dir Direction of communication
 * \returns True if communication is occurring along \a dir
 *
 * The size of the domain indexer is checked along the direction of communication
 * to see if there are multiple ranks that must communicate.
 */
bool mpcd::CellList::isCommunicating(mpcd::detail::face dir)
    {
    if (!m_decomposition)
        return false;

    const Index3D& di = m_decomposition->getDomainIndexer();
    bool result = true;
    if ((dir == mpcd::detail::face::east || dir == mpcd::detail::face::west) && di.getW() == 1)
        result = false;
    else if ((dir == mpcd::detail::face::north || dir == mpcd::detail::face::south)
             && di.getH() == 1)
        result = false;
    else if ((dir == mpcd::detail::face::up || dir == mpcd::detail::face::down) && di.getD() == 1)
        result = false;

    return result;
    }
#endif // ENABLE_MPI

/*!
 * \param timestep Current simulation timestep
 */
void mpcd::CellList::buildCellList()
    {
    const BoxDim& box = m_pdata->getBox();
    const uchar3 periodic = box.getPeriodic();

    ArrayHandle<unsigned int> h_cell_list(m_cell_list,
                                          access_location::host,
                                          access_mode::overwrite);
    ArrayHandle<unsigned int> h_cell_np(m_cell_np, access_location::host, access_mode::overwrite);
    // zero the cell counter
    memset(h_cell_np.data, 0, sizeof(unsigned int) * m_cell_indexer.getNumElements());

    uint3 conditions = make_uint3(0, 0, 0);

    ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(),
                               access_location::host,
                               access_mode::read);
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);
    unsigned int N_mpcd = m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual();
    unsigned int N_tot = N_mpcd;

    // we can't modify the velocity of embedded particles, so we only read their position
    std::unique_ptr<ArrayHandle<unsigned int>> h_embed_cell_ids;
    std::unique_ptr<ArrayHandle<Scalar4>> h_pos_embed;
    std::unique_ptr<ArrayHandle<unsigned int>> h_embed_member_idx;
    if (m_embed_group)
        {
        h_embed_cell_ids.reset(new ArrayHandle<unsigned int>(m_embed_cell_ids,
                                                             access_location::host,
                                                             access_mode::overwrite));
        h_pos_embed.reset(new ArrayHandle<Scalar4>(m_pdata->getPositions(),
                                                   access_location::host,
                                                   access_mode::read));
        h_embed_member_idx.reset(new ArrayHandle<unsigned int>(m_embed_group->getIndexArray(),
                                                               access_location::host,
                                                               access_mode::read));
        N_tot += m_embed_group->getNumMembers();
        }

    // total effective number of cells in the global box, optionally padded by
    // extra cells in MPI simulations
    uint3 n_global_cells = m_global_cell_dim;
#ifdef ENABLE_MPI
    if (isCommunicating(mpcd::detail::face::east))
        n_global_cells.x += 2 * m_num_extra;
    if (isCommunicating(mpcd::detail::face::north))
        n_global_cells.y += 2 * m_num_extra;
    if (isCommunicating(mpcd::detail::face::up))
        n_global_cells.z += 2 * m_num_extra;
#endif // ENABLE_MPI

    const BoxDim& global_box = m_pdata->getGlobalBox();

    for (unsigned int cur_p = 0; cur_p < N_tot; ++cur_p)
        {
        Scalar4 postype_i;
        if (cur_p < N_mpcd)
            {
            postype_i = h_pos.data[cur_p];
            }
        else
            {
            postype_i = h_pos_embed->data[h_embed_member_idx->data[cur_p - N_mpcd]];
            }
        Scalar3 pos_i = make_scalar3(postype_i.x, postype_i.y, postype_i.z);

        if (std::isnan(pos_i.x) || std::isnan(pos_i.y) || std::isnan(pos_i.z))
            {
            conditions.y = cur_p + 1;
            continue;
            }

        // bin particle
        const Scalar3 fractional_pos_i = global_box.makeFraction(pos_i - m_grid_shift);
        int3 global_bin = make_int3((int)std::floor(fractional_pos_i.x * m_global_cell_dim.x),
                                    (int)std::floor(fractional_pos_i.y * m_global_cell_dim.y),
                                    (int)std::floor(fractional_pos_i.z * m_global_cell_dim.z));

        // wrap cell back through the boundaries (grid shifting may send +/- 1 outside of range)
        // this is done using periodic from the "local" box, since this will be periodic
        // only when there is one rank along the dimension
        if (periodic.x)
            {
            if (global_bin.x == (int)n_global_cells.x)
                global_bin.x = 0;
            else if (global_bin.x == -1)
                global_bin.x = n_global_cells.x - 1;
            }
        if (periodic.y)
            {
            if (global_bin.y == (int)n_global_cells.y)
                global_bin.y = 0;
            else if (global_bin.y == -1)
                global_bin.y = n_global_cells.y - 1;
            }
        if (periodic.z)
            {
            if (global_bin.z == (int)n_global_cells.z)
                global_bin.z = 0;
            else if (global_bin.z == -1)
                global_bin.z = n_global_cells.z - 1;
            }

        // compute the local cell
        int3 bin = make_int3(global_bin.x - m_origin_idx.x,
                             global_bin.y - m_origin_idx.y,
                             global_bin.z - m_origin_idx.z);

        // validate and make sure no particles blew out of the box
        if ((bin.x < 0 || bin.x >= (int)m_cell_dim.x) || (bin.y < 0 || bin.y >= (int)m_cell_dim.y)
            || (bin.z < 0 || bin.z >= (int)m_cell_dim.z))
            {
            conditions.z = cur_p + 1;
            continue;
            }

        unsigned int bin_idx = m_cell_indexer(bin.x, bin.y, bin.z);
        unsigned int offset = h_cell_np.data[bin_idx];
        if (offset < m_cell_np_max)
            {
            h_cell_list.data[m_cell_list_indexer(offset, bin_idx)] = cur_p;
            }
        else
            {
            // overflow
            conditions.x = std::max(conditions.x, offset + 1);
            }

        // stash the current particle bin into the velocity array
        if (cur_p < N_mpcd)
            {
            h_vel.data[cur_p].w = __int_as_scalar(bin_idx);
            }
        else
            {
            h_embed_cell_ids->data[cur_p - N_mpcd] = bin_idx;
            }

        // increment the counter always
        ++h_cell_np.data[bin_idx];
        }

    // write out the conditions
    m_conditions.resetFlags(conditions);
    }

/*!
 * \param timestep Timestep that the sorting occurred
 * \param order Mapping of sorted particle indexes onto old particle indexes
 * \param rorder Mapping of old particle indexes onto sorted particle indexes
 */
void mpcd::CellList::sort(uint64_t timestep,
                          const GPUArray<unsigned int>& order,
                          const GPUArray<unsigned int>& rorder)
    {
    // no need to do any sorting if we can still be called at the current timestep
    if (peekCompute(timestep))
        return;

    // if mapping is not valid, signal that we need to force a recompute next time
    // that the cell list is needed. We don't call forceCompute() directly because this always
    // runs compute(), and we just want to defer to the next compute() call.
    if (rorder.isNull())
        {
        m_force_compute = true;
        return;
        }

    // iterate through particles in cell list, and update their indexes using reverse mapping
    ArrayHandle<unsigned int> h_rorder(rorder, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_cell_np(m_cell_np, access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_cell_list(m_cell_list,
                                          access_location::host,
                                          access_mode::readwrite);
    const unsigned int N_mpcd = m_mpcd_pdata->getN();

    for (unsigned int idx = 0; idx < getNCells(); ++idx)
        {
        const unsigned int np = h_cell_np.data[idx];
        for (unsigned int offset = 0; offset < np; ++offset)
            {
            const unsigned int cl_idx = m_cell_list_indexer(offset, idx);
            const unsigned int pid = h_cell_list.data[cl_idx];
            // only update indexes of MPCD particles, not virtual or embedded particles
            if (pid < N_mpcd)
                {
                h_cell_list.data[cl_idx] = h_rorder.data[pid];
                }
            }
        }
    }

#ifdef ENABLE_MPI
bool mpcd::CellList::needsEmbedMigrate(uint64_t timestep)
    {
    // no migrate needed if no embedded particles
    if (!m_embed_group)
        return false;

    // ensure that the cell list has been sized first
    computeDimensions();

    // coverage box dimensions
    const uchar3 periodic = m_cover_box.getPeriodic();
    const unsigned int ndim = m_sysdef->getNDimensions();

    // particle data
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_group(m_embed_group->getIndexArray(),
                                      access_location::host,
                                      access_mode::read);
    const unsigned int N = m_embed_group->getNumMembers();

    // check if any particle lies outside of the box on this rank
    char migrate = 0;
    for (unsigned int i = 0; i < N && !migrate; ++i)
        {
        const unsigned int idx = h_group.data[i];
        const Scalar4 postype = h_pos.data[idx];
        const Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);
        const Scalar3 fractional_pos = m_cover_box.makeFraction(pos);
        if ((!periodic.x && (fractional_pos.x >= Scalar(1.0) || fractional_pos.x < Scalar(0.0)))
            || (!periodic.y && (fractional_pos.y >= Scalar(1.0) || fractional_pos.y < Scalar(0.0)))
            || (!periodic.z && ndim == 3
                && (fractional_pos.z >= Scalar(1.0) || fractional_pos.z < Scalar(0.0))))
            {
            migrate = 1;
            }
        }

    // reduce across all ranks
    MPI_Allreduce(MPI_IN_PLACE, &migrate, 1, MPI_CHAR, MPI_MAX, m_exec_conf->getMPICommunicator());

    return static_cast<bool>(migrate);
    }
#endif // ENABLE_MPI

bool mpcd::CellList::checkConditions()
    {
    bool result = false;

    uint3 conditions = m_conditions.readFlags();

    if (conditions.x > m_cell_np_max)
        {
        m_cell_np_max = conditions.x;
        result = true;
        }
    if (conditions.y)
        {
        unsigned int n = conditions.y - 1;
        if (n < m_mpcd_pdata->getN())
            m_exec_conf->msg->errorAllRanks()
                << "MPCD particle " << n << " has position NaN" << std::endl;
        else if (n < m_mpcd_pdata->getNVirtual())
            m_exec_conf->msg->errorAllRanks()
                << "MPCD virtual particle " << n << " has position NaN" << std::endl;
        else
            {
            ArrayHandle<unsigned int> h_embed_member_idx(m_embed_group->getIndexArray(),
                                                         access_location::host,
                                                         access_mode::read);
            m_exec_conf->msg->errorAllRanks()
                << "Embedded particle "
                << h_embed_member_idx.data[n - (m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual())]
                << " has position NaN" << std::endl;
            }
        throw std::runtime_error("Error computing cell list");
        }
    if (conditions.z)
        {
        unsigned int n = conditions.z - 1;
        Scalar4 pos_empty_i;
        if (n < m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual())
            {
            ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(),
                                       access_location::host,
                                       access_mode::read);
            pos_empty_i = h_pos.data[n];
            if (n < m_mpcd_pdata->getN())
                m_exec_conf->msg->errorAllRanks()
                    << "MPCD particle is no longer in the simulation box" << std::endl;
            else
                m_exec_conf->msg->errorAllRanks()
                    << "MPCD virtual particle is no longer in the simulation box" << std::endl;
            }
        else
            {
            ArrayHandle<Scalar4> h_pos_embed(m_pdata->getPositions(),
                                             access_location::host,
                                             access_mode::read);
            ArrayHandle<unsigned int> h_embed_member_idx(m_embed_group->getIndexArray(),
                                                         access_location::host,
                                                         access_mode::read);
            pos_empty_i
                = h_pos_embed
                      .data[h_embed_member_idx
                                .data[n - (m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual())]];
            m_exec_conf->msg->errorAllRanks()
                << "Embedded particle is no longer in the simulation box" << std::endl;
            }

        Scalar3 pos = make_scalar3(pos_empty_i.x, pos_empty_i.y, pos_empty_i.z);
        m_exec_conf->msg->errorAllRanks()
            << "Cartesian coordinates: " << std::endl
            << "x: " << pos.x << " y: " << pos.y << " z: " << pos.z << std::endl
            << "Grid shift: " << std::endl
            << "x: " << m_grid_shift.x << " y: " << m_grid_shift.y << " z: " << m_grid_shift.z
            << std::endl;

        const BoxDim cover_box = getCoverageBox();
        Scalar3 lo = cover_box.getLo();
        Scalar3 hi = cover_box.getHi();
        uchar3 periodic = cover_box.getPeriodic();
        m_exec_conf->msg->errorAllRanks()
            << "Covered box lo: (" << lo.x << ", " << lo.y << ", " << lo.z << ")" << std::endl
            << "            hi: (" << hi.x << ", " << hi.y << ", " << hi.z << ")" << std::endl
            << "          tilt: (" << cover_box.getTiltFactorXY() << ", "
            << cover_box.getTiltFactorXZ() << ", " << cover_box.getTiltFactorYZ() << std::endl
            << "      periodic: (" << ((periodic.x) ? "1" : "0") << " "
            << ((periodic.y) ? "1" : "0") << " " << ((periodic.z) ? "1" : "0") << ")" << std::endl;
        throw std::runtime_error("Error computing cell list");
        }

    return result;
    }

void mpcd::CellList::resetConditions()
    {
    m_conditions.resetFlags(make_uint3(0, 0, 0));
    }

/*!
 * \param timestep Timestep to set shifting for
 *
 * \post The MPCD cell list has its grid shift set for \a timestep.
 *
 * If grid shifting is enabled, three uniform random numbers are drawn using
 * the Mersenne twister generator. (In two dimensions, only two numbers are drawn.)
 *
 * If grid shifting is disabled, a zero vector is instead set.
 */
void mpcd::CellList::drawGridShift(uint64_t timestep)
    {
    if (m_enable_grid_shift)
        {
        computeDimensions();

        uint16_t seed = m_sysdef->getSeed();

        // PRNG using seed and timestep as seeds
        hoomd::RandomGenerator rng(hoomd::Seed(hoomd::RNGIdentifier::MPCDCellList, timestep, seed),
                                   hoomd::Counter());

        // draw shift variables from uniform distribution
        const BoxDim& global_box = m_pdata->getGlobalBox();
        hoomd::UniformDistribution<Scalar> uniform(Scalar(-0.5), Scalar(0.5));
        Scalar3 shift = uniform(rng) * m_global_cell_dim_inv.x * global_box.getLatticeVector(0)
                        + uniform(rng) * m_global_cell_dim_inv.y * global_box.getLatticeVector(1);
        if (m_sysdef->getNDimensions() == 3)
            {
            shift += uniform(rng) * m_global_cell_dim_inv.z * global_box.getLatticeVector(2);
            }
        setGridShift(shift);
        }
    }

void mpcd::CellList::setGlobalDim(const uint3& global_cell_dim)
    {
    if (global_cell_dim.x == 0 || global_cell_dim.y == 0)
        {
        throw std::runtime_error("Global cell dimensions must be at least 1");
        }

    m_global_cell_dim = global_cell_dim;
    if (m_sysdef->getNDimensions() == 2)
        {
        m_global_cell_dim.z = 1;
        }

    m_global_cell_dim_inv = make_scalar3(Scalar(1.0) / global_cell_dim.x,
                                         Scalar(1.0) / global_cell_dim.y,
                                         Scalar(1.0) / global_cell_dim.z);

    const BoxDim& global_box = m_pdata->getGlobalBox();
    const Scalar3 max_fractional_shift = Scalar(0.5) * m_global_cell_dim_inv;
    m_max_grid_shift = max_fractional_shift.x * global_box.getLatticeVector(0)
                       + max_fractional_shift.y * global_box.getLatticeVector(1)
                       + max_fractional_shift.z * global_box.getLatticeVector(2);

    m_needs_compute_dim = true;
    }

/*!
 * \param global Global cell index to shift into the local box
 * \returns Local cell index
 *
 * \warning The returned cell index may lie outside the local grid. It is the
 *          caller's responsibility to check that the index is valid.
 */
const int3 mpcd::CellList::getLocalCell(const int3& global)
    {
    computeDimensions();

    int3 local = make_int3(global.x - m_origin_idx.x,
                           global.y - m_origin_idx.y,
                           global.z - m_origin_idx.z);

    return local;
    }

/*!
 * \param local Local cell index to shift into the global box
 * \returns Global cell index
 *
 * Local cell coordinates are wrapped around the global box so that a valid global
 * index is computed.
 */
const int3 mpcd::CellList::getGlobalCell(const int3& local)
    {
    computeDimensions();

    int3 global
        = make_int3(local.x + m_origin_idx.x, local.y + m_origin_idx.y, local.z + m_origin_idx.z);
    return wrapGlobalCell(global);
    }

Scalar3 mpcd::CellList::getCellSize()
    {
    computeDimensions();

    const BoxDim& global_box = m_pdata->getGlobalBox();
    const Scalar3 L = global_box.getL();
    return make_scalar3(L.x * m_global_cell_dim_inv.x,
                        L.y * m_global_cell_dim_inv.y,
                        L.z * m_global_cell_dim_inv.z);
    }

/*!
 * \param cell_size Grid spacing
 * \note Calling forces a resize of the cell list on the next update
 */
void mpcd::CellList::setCellSize(Scalar cell_size)
    {
    const BoxDim& global_box = m_pdata->getGlobalBox();
    const Scalar3 L = global_box.getL();
    uint3 global_cell_dim = make_uint3((unsigned int)round(L.x / cell_size),
                                       (unsigned int)round(L.y / cell_size),
                                       (unsigned int)round(L.z / cell_size));
    if (m_sysdef->getNDimensions() == 2)
        {
        global_cell_dim.z = 1;
        }

    // check that box is a multiple of cell size
    const double eps = 1e-5;
    if (fabs((double)L.x - global_cell_dim.x * (double)cell_size) > eps * cell_size
        || fabs((double)L.y - global_cell_dim.y * (double)cell_size) > eps * cell_size
        || (m_sysdef->getNDimensions() == 3
            && fabs((double)L.z - global_cell_dim.z * (double)cell_size) > eps * cell_size))
        {
        throw std::runtime_error("MPCD cell size must evenly divide box");
        }

    setGlobalDim(global_cell_dim);
    }

/*!
 * \param cell Cell coordinates to wrap back into the global box
 *
 * \warning Only up to one global box size is wrapped. This method is intended
 *          to be used for wrapping cells off by only one or two from the global boundary.
 */
const int3 mpcd::CellList::wrapGlobalCell(const int3& cell)
    {
    computeDimensions();

    int3 wrap = cell;

    if (wrap.x >= (int)m_global_cell_dim.x)
        wrap.x -= m_global_cell_dim.x;
    else if (wrap.x < 0)
        wrap.x += m_global_cell_dim.x;

    if (wrap.y >= (int)m_global_cell_dim.y)
        wrap.y -= m_global_cell_dim.y;
    else if (wrap.y < 0)
        wrap.y += m_global_cell_dim.y;

    if (wrap.z >= (int)m_global_cell_dim.z)
        wrap.z -= m_global_cell_dim.z;
    else if (wrap.z < 0)
        wrap.z += m_global_cell_dim.z;

    return wrap;
    }

namespace mpcd
    {
namespace detail
    {
void export_CellList(pybind11::module& m)
    {
    pybind11::class_<mpcd::CellList, Compute, std::shared_ptr<mpcd::CellList>>(m, "CellList")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, Scalar, bool>())
        .def("setCellSize", &mpcd::CellList::setCellSize)
        .def_property(
            "num_cells",
            [](const mpcd::CellList& cl)
            {
                const auto num_cells = cl.getGlobalDim();
                return pybind11::make_tuple(num_cells.x, num_cells.y, num_cells.z);
            },
            [](mpcd::CellList& cl, const pybind11::tuple& num_cells)
            {
                cl.setGlobalDim(make_uint3(pybind11::cast<unsigned int>(num_cells[0]),
                                           pybind11::cast<unsigned int>(num_cells[1]),
                                           pybind11::cast<unsigned int>(num_cells[2])));
            })
        .def_property("shift",
                      &mpcd::CellList::isGridShifting,
                      &mpcd::CellList::enableGridShifting);
    }
    } // namespace detail
    } // namespace mpcd
    } // end namespace hoomd
