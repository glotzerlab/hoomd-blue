// Copyright (c) 2009-2026 The Regents of the University of Michigan.
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
    : Compute(sysdef), m_mpcd_pdata(m_sysdef->getMPCDParticleData()), m_cell_np(m_exec_conf),
      m_embed_cell_ids(m_exec_conf), m_conditions(m_exec_conf), m_cell_vel(m_exec_conf),
      m_cell_energy(m_exec_conf), m_cell_temp(m_exec_conf), m_needs_net_reduce(true),
      m_needs_compute_dim(true), m_particles_sorted(false), m_virtual_change(false)
    {
    assert(m_mpcd_pdata);
    m_exec_conf->msg->notice(5) << "Constructing MPCD CellList" << std::endl;

    GPUArray<double> net_properties(mpcd::detail::thermo_index::num_quantities, m_exec_conf);
    m_net_properties.swap(net_properties);

    setCellSize(cell_size);
    m_origin_idx = make_uint3(0, 0, 0);
    m_cell_dim = make_uint3(0, 0, 0);

    m_enable_grid_shift = shift;
    m_grid_shift = make_scalar3(0.0, 0.0, 0.0);

    resetConditions();

#ifdef ENABLE_MPI
    m_decomposition = m_pdata->getDomainDecomposition();
    m_mpi_comm = m_exec_conf->getMPICommunicator();
    initializeCommunicationSetup();
#endif // ENABLE_MPI

    m_mpcd_pdata->getNumVirtualSignal().connect<mpcd::CellList, &mpcd::CellList::slotNumVirtual>(
        this);
    m_pdata->getParticleSortSignal().connect<mpcd::CellList, &mpcd::CellList::slotSorted>(this);
    m_pdata->getBoxChangeSignal().connect<mpcd::CellList, &mpcd::CellList::slotBoxChanged>(this);
    }

mpcd::CellList::CellList(std::shared_ptr<SystemDefinition> sysdef,
                         const uint3& global_cell_dim,
                         bool shift)
    : Compute(sysdef), m_mpcd_pdata(m_sysdef->getMPCDParticleData()), m_cell_np(m_exec_conf),
      m_embed_cell_ids(m_exec_conf), m_conditions(m_exec_conf), m_cell_vel(m_exec_conf),
      m_cell_energy(m_exec_conf), m_cell_temp(m_exec_conf), m_needs_compute_dim(true),
      m_particles_sorted(false), m_virtual_change(false)
    {
    assert(m_mpcd_pdata);
    m_exec_conf->msg->notice(5) << "Constructing MPCD CellList" << std::endl;

    GPUArray<double> net_properties(mpcd::detail::thermo_index::num_quantities, m_exec_conf);
    m_net_properties.swap(net_properties);

    setGlobalDim(global_cell_dim);
    m_origin_idx = make_uint3(0, 0, 0);
    m_cell_dim = make_uint3(0, 0, 0);

    m_enable_grid_shift = shift;
    m_grid_shift = make_scalar3(0.0, 0.0, 0.0);

    resetConditions();

#ifdef ENABLE_MPI
    m_decomposition = m_pdata->getDomainDecomposition();
    m_mpi_comm = m_exec_conf->getMPICommunicator();
    initializeCommunicationSetup();
#endif // ENABLE_MPI

    m_mpcd_pdata->getNumVirtualSignal().connect<mpcd::CellList, &mpcd::CellList::slotNumVirtual>(
        this);
    m_pdata->getParticleSortSignal().connect<mpcd::CellList, &mpcd::CellList::slotSorted>(this);
    m_pdata->getBoxChangeSignal().connect<mpcd::CellList, &mpcd::CellList::slotBoxChanged>(this);
    }

mpcd::CellList::~CellList()
    {
    m_exec_conf->msg->notice(5) << "Destroying MPCD CellList" << std::endl;
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

    // ensure optional computation flags are up to date
    updateFlags();

    if (peekCompute(timestep))
        {
        // ensure grid is shifted
        drawGridShift(timestep);

        // resize to be able to hold the number of embedded particles
        if (m_embed_group)
            {
            m_embed_cell_ids.resize(m_embed_group->getNumMembers());
            }

        // bin particles and compute cell properties
        buildCellList();
        checkConditions();
#ifdef ENABLE_MPI
        fillGhostBuffers();
        sendGhosts();
        addGhostsToCells();
        checkGhostConditions();
#endif // ENABLE_MPI
        finishComputeProperties();

        // we are finished building, explicitly mark everything (rather than using shouldCompute)
        m_first_compute = false;
        m_force_compute = false;
        m_needs_net_reduce = true;
        m_last_computed = timestep;

        // signal to the ParticleData that the cell list cache is now valid
        m_mpcd_pdata->validateCellCache();
        }
    }

void mpcd::CellList::reallocate()
    {
    m_cell_np.resize(m_cell_indexer.getNumElements());
    m_cell_vel.resize(m_cell_indexer.getNumElements());
    m_cell_energy.resize(m_cell_indexer.getNumElements());
    m_cell_temp.resize(m_cell_indexer.getNumElements());
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
        // fractional bounds of domain
        const auto grid_pos = m_decomposition->getGridPos();
        Scalar3 fractional_lo = make_scalar3(m_decomposition->getCumulativeFraction(0, grid_pos.x),
                                             m_decomposition->getCumulativeFraction(1, grid_pos.y),
                                             m_decomposition->getCumulativeFraction(2, grid_pos.z));
        const Scalar3 fractional_hi
            = make_scalar3(m_decomposition->getCumulativeFraction(0, grid_pos.x + 1),
                           m_decomposition->getCumulativeFraction(1, grid_pos.y + 1),
                           m_decomposition->getCumulativeFraction(2, grid_pos.z + 1));

        // setup lo bin
        uint3 my_lo_bin
            = make_uint3(static_cast<uint>(std::round(fractional_lo.x * m_global_cell_dim.x)),
                         static_cast<uint>(std::round(fractional_lo.y * m_global_cell_dim.y)),
                         static_cast<uint>(std::round(fractional_lo.z * m_global_cell_dim.z)));

        // setup hi bin
        uint3 my_hi_bin
            = make_uint3(static_cast<uint>(std::round(fractional_hi.x * m_global_cell_dim.x)),
                         static_cast<uint>(std::round(fractional_hi.y * m_global_cell_dim.y)),
                         static_cast<uint>(std::round(fractional_hi.z * m_global_cell_dim.z)));

        // initially size the grid assuming one rank in each direction, and then resize based on
        // communication
        m_cell_dim = m_global_cell_dim;
        m_origin_idx = make_uint3(0, 0, 0);
        if (communicating.x)
            {
            // number of cells and cell origin
            m_cell_dim.x = my_hi_bin.x - my_lo_bin.x;
            m_origin_idx.x = my_lo_bin.x;
            }

        if (communicating.y)
            {
            m_cell_dim.y = my_hi_bin.y - my_lo_bin.y;
            m_origin_idx.y = my_lo_bin.y;
            }

        if (m_sysdef->getNDimensions() == 3 && communicating.z)
            {
            m_cell_dim.z = my_hi_bin.z - my_lo_bin.z;
            m_origin_idx.z = my_lo_bin.z;
            }
        }
    else
#endif // ENABLE_MPI
        {
        m_cell_dim = m_global_cell_dim;
        m_origin_idx = make_uint3(0, 0, 0);
        }

    // resize the cell indexers and per-cell counter
    m_global_cell_indexer = Index3D(m_global_cell_dim.x, m_global_cell_dim.y, m_global_cell_dim.z);
    m_cell_indexer = Index3D(m_cell_dim.x, m_cell_dim.y, m_cell_dim.z);

#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        const unsigned int num_cells = m_global_cell_indexer.getNumElements();
        if (num_cells >= (~(1 << 31)))
            {
            throw std::runtime_error("Number of global cells exceeds size allotted.");
            }
        }
#endif // ENABLE_MPI

    // reallocate per-cell memory
    reallocate();

    // dimensions are now current
    m_needs_compute_dim = false;
    notifySizeChange();
    }

void mpcd::CellList::startAutotuning()
    {
    Compute::startAutotuning();
#ifdef ENABLE_MPI
#endif // ENABLE_MPI
    }

bool mpcd::CellList::isAutotuningComplete()
    {
    bool result = Compute::isAutotuningComplete();
#ifdef ENABLE_MPI
#endif // ENABLE_MPI
    return result;
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
    const BoxDim& global_box = m_pdata->getGlobalBox();
    const uchar3 periodic = global_box.getPeriodic();

    // zero the cell counter
    m_cell_np.zeroFill();
    m_cell_vel.zeroFill();
    if (m_flags[mpcd::detail::thermo_options::energy])
        {
        m_cell_energy.zeroFill();
        }

    ArrayHandle<unsigned int> h_cell_np(m_cell_np, access_location::host, access_mode::readwrite);
    ArrayHandle<double4> h_cell_vel(m_cell_vel, access_location::host, access_mode::readwrite);
    ArrayHandle<double> h_cell_energy(m_cell_energy, access_location::host, access_mode::readwrite);

    uint3 conditions = make_uint3(0, 0, 0);

    ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(),
                               access_location::host,
                               access_mode::read);
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);
    unsigned int N_mpcd = m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual();
    unsigned int N_tot = N_mpcd;

#ifdef ENABLE_MPI
    std::unique_ptr<ArrayHandle<uint2>> h_mpcd_comm_key;
    uint3 rank_size = make_uint3(0, 0, 0);
    const bool is_decomposition = bool(m_decomposition);
    if (is_decomposition)
        {
        // allocate space for communication flag
        m_mpcd_comm_key.resize(N_mpcd);
        h_mpcd_comm_key.reset(
            new ArrayHandle<uint2>(m_mpcd_comm_key, access_location::host, access_mode::overwrite));

        // allocate the the number of ranks in each dimension
        Index3D di = m_decomposition->getDomainIndexer();
        rank_size = make_uint3(di.getW(), di.getH(), di.getD());
        }

#endif // ENABLE_MPI

    // we can't modify the velocity of embedded particles, so we only read their position
    std::unique_ptr<ArrayHandle<unsigned int>> h_embed_cell_ids;
    std::unique_ptr<ArrayHandle<Scalar4>> h_pos_embed;
    std::unique_ptr<ArrayHandle<Scalar4>> h_vel_embed;
    std::unique_ptr<ArrayHandle<unsigned int>> h_embed_member_idx;
    if (m_embed_group)
        {
        h_embed_cell_ids.reset(new ArrayHandle<unsigned int>(m_embed_cell_ids,
                                                             access_location::host,
                                                             access_mode::overwrite));
        h_pos_embed.reset(new ArrayHandle<Scalar4>(m_pdata->getPositions(),
                                                   access_location::host,
                                                   access_mode::read));
        h_vel_embed.reset(new ArrayHandle<Scalar4>(m_pdata->getVelocities(),
                                                   access_location::host,
                                                   access_mode::read));
        h_embed_member_idx.reset(new ArrayHandle<unsigned int>(m_embed_group->getIndexArray(),
                                                               access_location::host,
                                                               access_mode::read));
        N_tot += m_embed_group->getNumMembers();
        }

    for (unsigned int cur_p = 0; cur_p < N_tot; ++cur_p)
        {
        Scalar4 postype_i;
        Scalar4 vel_mass_i;
        double mass_i;
        if (cur_p < N_mpcd)
            {
            postype_i = h_pos.data[cur_p];
            vel_mass_i = h_vel.data[cur_p];
            mass_i = m_mpcd_pdata->getMass();
            }
        else
            {
            postype_i = h_pos_embed->data[h_embed_member_idx->data[cur_p - N_mpcd]];
            vel_mass_i = h_vel_embed->data[h_embed_member_idx->data[cur_p - N_mpcd]];
            mass_i = vel_mass_i.w;
            }
        const Scalar3 pos_i = make_scalar3(postype_i.x, postype_i.y, postype_i.z);
        const double3 vel_i = make_double3(vel_mass_i.x, vel_mass_i.y, vel_mass_i.z);

        if (std::isnan(pos_i.x) || std::isnan(pos_i.y) || std::isnan(pos_i.z))
            {
            conditions.y = cur_p + 1;
            continue;
            }

        // bin particle
        const Scalar3 fractional_pos_i = global_box.makeFraction(pos_i) - m_grid_shift;
        int3 global_bin = make_int3((int)std::floor(fractional_pos_i.x * m_global_cell_dim.x),
                                    (int)std::floor(fractional_pos_i.y * m_global_cell_dim.y),
                                    (int)std::floor(fractional_pos_i.z * m_global_cell_dim.z));

        // wrap cell back through the boundaries (grid shifting may send +/- 1 outside of range)
        // this is done using periodic from the global box
        if (periodic.x)
            {
            if (global_bin.x >= (int)m_global_cell_dim.x)
                global_bin.x -= m_global_cell_dim.x;
            else if (global_bin.x < 0)
                global_bin.x += m_global_cell_dim.x;
            }
        if (periodic.y)
            {
            if (global_bin.y >= (int)m_global_cell_dim.y)
                global_bin.y -= m_global_cell_dim.y;
            else if (global_bin.y < 0)
                global_bin.y += m_global_cell_dim.y;
            }
        if (periodic.z)
            {
            if (global_bin.z >= (int)m_global_cell_dim.z)
                global_bin.z -= m_global_cell_dim.z;
            else if (global_bin.z < 0)
                global_bin.z += m_global_cell_dim.z;
            }

        // validate and make sure no particles blew out of the box
        if ((global_bin.x < 0 || global_bin.x >= (int)m_global_cell_dim.x)
            || (global_bin.y < 0 || global_bin.y >= (int)m_global_cell_dim.y)
            || (global_bin.z < 0 || global_bin.z >= (int)m_global_cell_dim.z))
            {
            conditions.z = cur_p + 1;
            continue;
            }

        // compute the local cell
        int3 bin = make_int3(global_bin.x - m_origin_idx.x,
                             global_bin.y - m_origin_idx.y,
                             global_bin.z - m_origin_idx.z);
        unsigned int bin_idx;
        bool is_local = true;

#ifdef ENABLE_MPI
        if (is_decomposition)
            {
            is_local = (0 <= bin.x && bin.x < (int)m_cell_dim.x)
                       && (0 <= bin.y && bin.y < (int)m_cell_dim.y)
                       && (0 <= bin.z && bin.z < (int)m_cell_dim.z);
            if (is_local)
                {
                bin_idx = m_cell_indexer(bin.x, bin.y, bin.z);
                if (cur_p < N_mpcd)
                    {
                    h_mpcd_comm_key->data[cur_p] = make_uint2(0xffffffff, cur_p);
                    }
                }
            else
                {
                // determine from the bin which rank the particle's cell belongs to
                int ix = 0;
                if (bin.x >= (int)m_cell_dim.x)
                    ix = (rank_size.x > 2) ? 1 : -1;
                else if (bin.x < 0)
                    ix = -1;

                int iy = 0;
                if (bin.y >= (int)m_cell_dim.y)
                    iy = (rank_size.y > 2) ? 1 : -1;
                else if (bin.y < 0)
                    iy = -1;

                int iz = 0;
                if (bin.z >= (int)m_cell_dim.z)
                    iz = (rank_size.z > 2) ? 1 : -1;
                else if (bin.z < 0)
                    iz = -1;

                // get shifted direction index
                int dir = ((iz + 1) * 3 + (iy + 1)) * 3 + (ix + 1);
                dir = dir + ((ix == 1) ? -2 : 1) + ((iy == 1) ? -6 : 3) + ((iz == 1) ? -12 : 9);

                // mark particle to be sent to neighboring rank
                if (cur_p < N_mpcd)
                    {
                    h_mpcd_comm_key->data[cur_p] = make_uint2(dir, cur_p);
                    }

                // set the bin idx to be the global index with highest bit set to 1
                bin_idx = m_global_cell_indexer(global_bin.x, global_bin.y, global_bin.z);
                bin_idx |= 1 << 31;
                }
            }
        else
#endif // ENABLE_MPI
            {
            bin_idx = m_cell_indexer(bin.x, bin.y, bin.z);
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

        if (!is_local)
            {
            continue;
            }
        // compute the contribution of the particle to cell velocity
        double4& cell_vel = h_cell_vel.data[bin_idx];
        cell_vel.x += mass_i * vel_i.x;
        cell_vel.y += mass_i * vel_i.y;
        cell_vel.z += mass_i * vel_i.z;
        cell_vel.w += mass_i;
        // compute optional cell properties
        if (m_flags[mpcd::detail::thermo_options::energy])
            {
            h_cell_energy.data[bin_idx]
                += 0.5 * mass_i * (vel_i.x * vel_i.x + vel_i.y * vel_i.y + vel_i.z * vel_i.z);
            }
        // increment the counter always
        ++h_cell_np.data[bin_idx];
        }
    // write out the conditions
    m_conditions.resetFlags(conditions);
    }

void mpcd::CellList::finishComputeProperties()
    {
    const bool need_energy = m_flags[mpcd::detail::thermo_options::energy];
    if (need_energy)
        {
        m_cell_temp.zeroFill();
        }

    ArrayHandle<unsigned int> h_cell_np(m_cell_np, access_location::host, access_mode::read);
    ArrayHandle<double4> h_cell_vel(m_cell_vel, access_location::host, access_mode::readwrite);
    ArrayHandle<double> h_cell_energy(m_cell_energy, access_location::host, access_mode::read);
    ArrayHandle<double> h_cell_temp(m_cell_temp, access_location::host, access_mode::overwrite);
    // iterate over all cells and compute average velocity, energy, temperature
    for (unsigned int idx = 0; idx < getNCells(); ++idx)
        {
        // average cell properties if the cell has mass
        const double4 cell_vel = h_cell_vel.data[idx];
        double3 vel_cm = make_double3(cell_vel.x, cell_vel.y, cell_vel.z);
        const double mass = cell_vel.w;

        // get center of mass velocity from momentum
        if (mass > 0.)
            {
            // average velocity is only defined when there is some mass in the cell
            vel_cm.x /= mass;
            vel_cm.y /= mass;
            vel_cm.z /= mass;
            }
        h_cell_vel.data[idx] = make_double4(vel_cm.x, vel_cm.y, vel_cm.z, mass);

        if (need_energy)
            {
            const double cell_energy = h_cell_energy.data[idx];
            double temp(0.0);
            const unsigned int np = h_cell_np.data[idx];
            // temperature is only defined for 2 or more particles
            if (np > 1)
                {
                const double ke_cm
                    = 0.5 * mass
                      * (vel_cm.x * vel_cm.x + vel_cm.y * vel_cm.y + vel_cm.z * vel_cm.z);
                temp = 2. * (cell_energy - ke_cm) / (m_sysdef->getNDimensions() * (np - 1));
                }
            h_cell_temp.data[idx] = temp;
            }
        }
    }

void mpcd::CellList::computeNetProperties()
    {
    unsigned int n_temp_cells = 0;
        {
        const bool need_energy = m_flags[mpcd::detail::thermo_options::energy];
        ArrayHandle<unsigned int> h_cell_np(m_cell_np, access_location::host, access_mode::read);
        ArrayHandle<double4> h_cell_vel(m_cell_vel, access_location::host, access_mode::read);
        ArrayHandle<double> h_cell_energy(m_cell_energy, access_location::host, access_mode::read);
        ArrayHandle<double> h_cell_temp(m_cell_temp, access_location::host, access_mode::read);
        double3 net_momentum = make_double3(0, 0, 0);
        double net_energy(0.0), net_temp(0.0);

        // iterate over all cells and compute average velocity, energy, temperature
        for (unsigned int idx = 0; idx < getNCells(); ++idx)
            {
            // average cell properties if the cell has mass
            const double4 cell_vel = h_cell_vel.data[idx];
            const double3 vel_cm = make_double3(cell_vel.x, cell_vel.y, cell_vel.z);
            const double mass = cell_vel.w;

            // add accumulated momentum to total net momentum
            net_momentum.x += vel_cm.x * mass;
            net_momentum.y += vel_cm.y * mass;
            net_momentum.z += vel_cm.z * mass;

            if (need_energy)
                {
                const double cell_energy = h_cell_energy.data[idx];
                const double cell_temp = h_cell_temp.data[idx];
                // temperature is only defined for 2 or more particles
                if (h_cell_np.data[idx] > 1)
                    {
                    ++n_temp_cells;
                    }
                // accumulate
                net_energy += cell_energy;
                net_temp += cell_temp;
                }
            }
        ArrayHandle<double> h_net_properties(m_net_properties,
                                             access_location::host,
                                             access_mode::overwrite);

        h_net_properties.data[mpcd::detail::thermo_index::momentum_x] = net_momentum.x;
        h_net_properties.data[mpcd::detail::thermo_index::momentum_y] = net_momentum.y;
        h_net_properties.data[mpcd::detail::thermo_index::momentum_z] = net_momentum.z;
        h_net_properties.data[mpcd::detail::thermo_index::energy] = net_energy;
        h_net_properties.data[mpcd::detail::thermo_index::temperature] = net_temp;
        }

#ifdef ENABLE_MPI
    if (m_decomposition)
        {
        ArrayHandle<double> h_net_properties(m_net_properties,
                                             access_location::host,
                                             access_mode::readwrite);
        MPI_Allreduce(MPI_IN_PLACE,
                      h_net_properties.data,
                      mpcd::detail::thermo_index::num_quantities,
                      MPI_DOUBLE,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());

        MPI_Allreduce(MPI_IN_PLACE,
                      &n_temp_cells,
                      1,
                      MPI_UNSIGNED,
                      MPI_SUM,
                      m_exec_conf->getMPICommunicator());
        }
#endif // ENABLE_MPI
    if (n_temp_cells > 0)
        {
        ArrayHandle<double> h_net_properties(m_net_properties,
                                             access_location::host,
                                             access_mode::readwrite);
        h_net_properties.data[mpcd::detail::thermo_index::temperature] /= (double)n_temp_cells;
        }
    // ensure that properties will not be normalized twice
    m_needs_net_reduce = false;
    }

#ifdef ENABLE_MPI
void mpcd::CellList::initializeCommunicationSetup()
    {
    m_num_mpcd_ghosts_recv = 0;
    m_num_mpcd_ghosts_send = 0;
    m_num_unique_neigh = 0;
    if (m_decomposition)
        {
        // create buffer and ghost arrays
        GPUVector<uint2> mpcd_comm_key(m_exec_conf);
        m_mpcd_comm_key.swap(mpcd_comm_key);
        GPUVector<Scalar4> mpcd_vel_sendbuf(m_exec_conf);
        m_mpcd_vel_sendbuf.swap(mpcd_vel_sendbuf);
        GPUVector<Scalar4> mpcd_ghost_vel(m_exec_conf);
        m_mpcd_ghost_vel.swap(mpcd_ghost_vel);

        Index3D di = m_decomposition->getDomainIndexer();
        uint3 mypos = m_decomposition->getGridPos();
        ArrayHandle<unsigned int> h_cart_ranks(m_decomposition->getCartRanks(),
                                               access_location::host,
                                               access_mode::read);

        // reserve per neighbor memory
        m_num_mpcd_send_ptls.resize(27);
        m_num_mpcd_recv_ptls.resize(27);
        GPUVector<unsigned int> mpcd_send_offsets(27, m_exec_conf);
        m_mpcd_send_offsets.swap(mpcd_send_offsets);
        GPUVector<unsigned int> mpcd_recv_offsets(27, m_exec_conf);
        m_mpcd_recv_offsets.swap(mpcd_recv_offsets);
        m_neigh_rank.resize(26);
        std::fill(m_neigh_rank.begin(), m_neigh_rank.end(), 0xffffffff);

        int l = mypos.x;
        int m = mypos.y;
        int n = mypos.z;
        unsigned int num_neigh = 0;
        // loop over neighbors
        for (int ix = -1; ix <= 1; ix++)
            {
            int i = ix + l;
            if (i == (int)di.getW())
                i = 0;
            else if (i < 0)
                i += di.getW();

            // skip if not communicating along x-direction (either only 1 rank or wrong direction
            // with 2 ranks)
            if ((ix != 0 && di.getW() == 1) || (ix == 1 && di.getW() == 2))
                continue;

            for (int iy = -1; iy <= 1; iy++)
                {
                int j = iy + m;

                if (j == (int)di.getH())
                    j = 0;
                else if (j < 0)
                    j += di.getH();

                // skip if not communicating along y-direction (either only 1 rank or wrong
                // direction with 2 ranks)
                if ((iy != 0 && di.getH() == 1) || (iy == 1 && di.getH() == 2))
                    continue;

                for (int iz = -1; iz <= 1; iz++)
                    {
                    int k = iz + n;

                    if (k == (int)di.getD())
                        k = 0;
                    else if (k < 0)
                        k += di.getD();

                    // skip if not communicating along z-direction (either only 1 rank or wrong
                    // direction with 2 ranks)
                    if ((iz != 0 && di.getD() == 1) || (iz == 1 && di.getD() == 2))
                        continue;

                    // exclude ourselves
                    if (!ix && !iy && !iz)
                        continue;

                    unsigned int dir = ((iz + 1) * 3 + (iy + 1)) * 3 + (ix + 1);
                    // shift so not moving is last index
                    dir = dir + ((ix == 1) ? -2 : 1) + ((iy == 1) ? -6 : 3) + ((iz == 1) ? -12 : 9);

                    unsigned int neighbor = h_cart_ranks.data[di(i, j, k)];
                    m_adj_dir.insert(m_adj_dir.end(), dir);
                    m_neigh_rank[dir] = neighbor;
                    num_neigh++;
                    }
                }
            }
        m_num_unique_neigh = num_neigh;
        // sort neighbors according to how directions will be sorted
        sort(m_adj_dir.begin(), m_adj_dir.end(), std::less<uint>());
        }
    }

void mpcd::CellList::fillGhostBuffers()
    {
    if (!m_decomposition)
        {
        return;
        }

    // reset values
    m_num_mpcd_ghosts_recv = 0;
    m_num_mpcd_ghosts_send = 0;
        {
        ArrayHandle<unsigned int> h_mpcd_send_offsets(m_mpcd_send_offsets,
                                                      access_location::host,
                                                      access_mode::overwrite);
        std::fill(h_mpcd_send_offsets.data,
                  h_mpcd_send_offsets.data + m_mpcd_send_offsets.getNumElements(),
                  0xffffffff);
        }

    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);
    ArrayHandle<uint2> h_mpcd_comm_key(m_mpcd_comm_key,
                                       access_location::host,
                                       access_mode::readwrite);
    unsigned int N_mpcd = m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual();

    // sort the ghost particles by direction, least to greatest
    std::sort(h_mpcd_comm_key.data,
              h_mpcd_comm_key.data + N_mpcd,
              [](uint2& a, uint2& b) { return a.x < b.x; });
    auto temp = std::lower_bound(h_mpcd_comm_key.data,
                                 h_mpcd_comm_key.data + N_mpcd,
                                 0xffffffff,
                                 [](uint2 a, uint b) { return a.x < b; });
    m_num_mpcd_ghosts_send = static_cast<unsigned int>(std::distance(h_mpcd_comm_key.data, temp));

    // resize buffer array
    m_mpcd_vel_sendbuf.resize(m_num_mpcd_ghosts_send);

        // add velocity to send buffers and count particle to send per rank
        {
        ArrayHandle<Scalar4> h_mpcd_vel_sendbuf(m_mpcd_vel_sendbuf,
                                                access_location::host,
                                                access_mode::readwrite);
        ArrayHandle<unsigned int> h_mpcd_send_offsets(m_mpcd_send_offsets,
                                                      access_location::host,
                                                      access_mode::readwrite);
        // get number of neighbors and how many particles to send them
        unsigned int cur_dir = 0xffffffff;
        unsigned int comm_dir = 0;
        for (unsigned int i = 0; i < m_num_mpcd_ghosts_send; i++)
            {
            comm_dir = h_mpcd_comm_key.data[i].x;
            const unsigned int particle_index = h_mpcd_comm_key.data[i].y;

            // add particle data to send buffers
            const Scalar4 vel_mass = h_vel.data[particle_index];
            h_mpcd_vel_sendbuf.data[i] = vel_mass;

            // count the number of neighbors and how many particles for each
            if (comm_dir != cur_dir)
                {
                h_mpcd_send_offsets.data[comm_dir] = i;
                cur_dir = comm_dir;
                }
            }
        h_mpcd_send_offsets.data[26] = m_num_mpcd_ghosts_send;
        }
    }

void mpcd::CellList::sendGhosts()
    {
    if (!m_decomposition)
        {
        return;
        }

        // determine number of particles being sent per neighbor
        {
        ArrayHandle<unsigned int> h_mpcd_send_offsets(m_mpcd_send_offsets,
                                                      access_location::host,
                                                      access_mode::read);
        unsigned int last_start_offset = 0xffffffff;
        unsigned int last_start_dir = 0xffffffff;
        for (unsigned int i = 0; i < m_mpcd_send_offsets.getNumElements(); i++)
            {
            if (h_mpcd_send_offsets.data[i] == 0xffffffff)
                {
                m_num_mpcd_send_ptls[i] = 0;
                }
            else
                {
                if (last_start_dir != 0xffffffff)
                    {
                    m_num_mpcd_send_ptls[last_start_dir]
                        = h_mpcd_send_offsets.data[i] - last_start_offset;
                    }
                last_start_offset = h_mpcd_send_offsets.data[i];
                last_start_dir = i;
                }
            }
        }

    // communicate how many particles are being sent
    unsigned int num_recv_tot = 0;
        {
        // fill receive arrays
        ArrayHandle<unsigned int> h_mpcd_recv_offsets(m_mpcd_recv_offsets,
                                                      access_location::host,
                                                      access_mode::readwrite);
        for (unsigned int i = 0; i < m_mpcd_recv_offsets.getNumElements(); i++)
            {
            m_num_mpcd_recv_ptls[i] = 0;
            h_mpcd_recv_offsets.data[i] = 0;
            }

        // do communication
        unsigned int nreq = 0;
        m_reqs.resize(2 * m_num_unique_neigh);
        for (unsigned int ineigh = 0; ineigh < m_num_unique_neigh; ++ineigh)
            {
            // rank of neighbor processor
            unsigned int dir = m_adj_dir[ineigh];
            unsigned int neighbor = m_neigh_rank[dir];

            MPI_Isend(&m_num_mpcd_send_ptls[dir],
                      1,
                      MPI_UNSIGNED,
                      neighbor,
                      0,
                      m_mpi_comm,
                      &m_reqs[nreq++]);
            MPI_Irecv(&m_num_mpcd_recv_ptls[dir],
                      1,
                      MPI_UNSIGNED,
                      neighbor,
                      0,
                      m_mpi_comm,
                      &m_reqs[nreq++]);
            } // end neighbor loop
        MPI_Waitall(nreq, m_reqs.data(), MPI_STATUSES_IGNORE);
        // sum up receive counts
        for (unsigned int ineigh = 0; ineigh < m_num_unique_neigh; ++ineigh)
            {
            unsigned int dir = m_adj_dir[ineigh];
            h_mpcd_recv_offsets.data[dir] = num_recv_tot;
            num_recv_tot += m_num_mpcd_recv_ptls[dir];
            }
        m_num_mpcd_ghosts_recv = num_recv_tot;
        }
    // communicate ghost particles
    // resize ghost arrays to fit the particles being received
    m_mpcd_ghost_vel.resize(m_num_mpcd_ghosts_recv);
        {
        ArrayHandle<Scalar4> h_mpcd_ghost_vel(m_mpcd_ghost_vel,
                                              access_location::host,
                                              access_mode::overwrite);
        ArrayHandle<Scalar4> h_mpcd_vel_sendbuf(m_mpcd_vel_sendbuf,
                                                access_location::host,
                                                access_mode::read);
        ArrayHandle<unsigned int> h_mpcd_recv_offsets(m_mpcd_recv_offsets,
                                                      access_location::host,
                                                      access_mode::read);
        ArrayHandle<unsigned int> h_mpcd_send_offsets(m_mpcd_send_offsets,
                                                      access_location::host,
                                                      access_mode::read);
        const MPI_Datatype mpi_scalar4 = m_exec_conf->getMPIConfig()->getScalar4Datatype();
        // loop over neighbors
        unsigned int nreq = 0;
        for (unsigned int ineigh = 0; ineigh < m_num_unique_neigh; ++ineigh)
            {
            // rank of neighbor processor
            unsigned int dir = m_adj_dir[ineigh];
            unsigned int neighbor = m_neigh_rank[dir];

            // exchange particle data
            if (m_num_mpcd_send_ptls[dir])
                {
                MPI_Isend(h_mpcd_vel_sendbuf.data + h_mpcd_send_offsets.data[dir],
                          (int)m_num_mpcd_send_ptls[dir],
                          mpi_scalar4,
                          neighbor,
                          2,
                          m_mpi_comm,
                          &m_reqs[nreq++]);
                }

            if (m_num_mpcd_recv_ptls[dir])
                {
                MPI_Irecv(h_mpcd_ghost_vel.data + h_mpcd_recv_offsets.data[dir],
                          (int)m_num_mpcd_recv_ptls[dir],
                          mpi_scalar4,
                          neighbor,
                          2,
                          m_mpi_comm,
                          &m_reqs[nreq++]);
                }
            }
        MPI_Waitall(nreq, m_reqs.data(), MPI_STATUSES_IGNORE);
        }
    }

void mpcd::CellList::addGhostsToCells()
    {
    if (!m_decomposition || m_num_mpcd_ghosts_recv == 0)
        {
        return;
        }

    ArrayHandle<unsigned int> h_cell_np(m_cell_np, access_location::host, access_mode::readwrite);
    ArrayHandle<double4> h_cell_vel(m_cell_vel, access_location::host, access_mode::readwrite);
    ArrayHandle<double> h_cell_energy(m_cell_energy, access_location::host, access_mode::readwrite);

    uint3 conditions = make_uint3(0, 0, 0);

    ArrayHandle<Scalar4> h_mpcd_ghost_vel(m_mpcd_ghost_vel,
                                          access_location::host,
                                          access_mode::readwrite);

    for (unsigned int cur_p = 0; cur_p < m_num_mpcd_ghosts_recv; ++cur_p)
        {
        double mass_i = m_mpcd_pdata->getMass();
        const Scalar4 vel_mass_i = h_mpcd_ghost_vel.data[cur_p];
        const double3 vel_i = make_double3(vel_mass_i.x, vel_mass_i.y, vel_mass_i.z);

        // unset the highest bit of the global bin index
        unsigned int global_bin_index = __scalar_as_int(vel_mass_i.w);
        global_bin_index &= ~(1 << 31);

        // turn global bin index back into global bin
        int3 global_bin = make_int3(0, 0, 0);
        global_bin.x = global_bin_index % m_global_cell_dim.x;
        global_bin.y = (global_bin_index / m_global_cell_dim.x) % m_global_cell_dim.y;
        global_bin.z = int(global_bin_index / (m_global_cell_dim.x * m_global_cell_dim.y));

        // compute the local cell
        int3 bin = make_int3(global_bin.x - m_origin_idx.x,
                             global_bin.y - m_origin_idx.y,
                             global_bin.z - m_origin_idx.z);
        unsigned int bin_idx;
        if ((0 <= bin.x && bin.x < (int)m_cell_dim.x) && (0 <= bin.y && bin.y < (int)m_cell_dim.y)
            && (0 <= bin.z && bin.z < (int)m_cell_dim.z))
            {
            bin_idx = m_cell_indexer(bin.x, bin.y, bin.z);
            }
        else
            {
            conditions.x = cur_p + 1;
            continue;
            }

        // stash the current particle bin into the velocity array
        h_mpcd_ghost_vel.data[cur_p].w = __int_as_scalar(bin_idx);

        // compute the contribution of the particle to cell velocity
        double4& cell_vel = h_cell_vel.data[bin_idx];
        cell_vel.x += mass_i * vel_i.x;
        cell_vel.y += mass_i * vel_i.y;
        cell_vel.z += mass_i * vel_i.z;
        cell_vel.w += mass_i;
        // compute optional cell properties
        if (m_flags[mpcd::detail::thermo_options::energy])
            {
            h_cell_energy.data[bin_idx]
                += 0.5 * mass_i * (vel_i.x * vel_i.x + vel_i.y * vel_i.y + vel_i.z * vel_i.z);
            }
        // increment the counter always
        ++h_cell_np.data[bin_idx];
        }
    // write out the conditions
    m_conditions.resetFlags(conditions);
    }

void mpcd::CellList::reverseSendGhosts()
    {
    if (!m_decomposition || (m_num_mpcd_ghosts_send == 0 && m_num_mpcd_ghosts_recv == 0))
        {
        return;
        }
    ArrayHandle<Scalar4> h_mpcd_ghost_vel(m_mpcd_ghost_vel,
                                          access_location::host,
                                          access_mode::read);
    ArrayHandle<Scalar4> h_mpcd_vel_sendbuf(m_mpcd_vel_sendbuf,
                                            access_location::host,
                                            access_mode::overwrite);
    ArrayHandle<unsigned int> h_mpcd_recv_offsets(m_mpcd_recv_offsets,
                                                  access_location::host,
                                                  access_mode::read);
    ArrayHandle<unsigned int> h_mpcd_send_offsets(m_mpcd_send_offsets,
                                                  access_location::host,
                                                  access_mode::read);
    const MPI_Datatype mpi_scalar4 = m_exec_conf->getMPIConfig()->getScalar4Datatype();
    unsigned int nreq = 0;
    m_reqs.resize(2 * m_num_unique_neigh);
    for (unsigned int ineigh = 0; ineigh < m_num_unique_neigh; ++ineigh)
        {
        // rank of neighbor processor
        unsigned int dir = m_adj_dir[ineigh];
        unsigned int neighbor = m_neigh_rank[dir];

        // exchange particle data
        if (m_num_mpcd_recv_ptls[dir])
            {
            MPI_Isend(h_mpcd_ghost_vel.data + h_mpcd_recv_offsets.data[dir],
                      (int)m_num_mpcd_recv_ptls[dir],
                      mpi_scalar4,
                      neighbor,
                      2,
                      m_mpi_comm,
                      &m_reqs[nreq++]);
            }

        if (m_num_mpcd_send_ptls[dir])
            {
            MPI_Irecv(h_mpcd_vel_sendbuf.data + h_mpcd_send_offsets.data[dir],
                      (int)m_num_mpcd_send_ptls[dir],
                      mpi_scalar4,
                      neighbor,
                      2,
                      m_mpi_comm,
                      &m_reqs[nreq++]);
            }
        }
    MPI_Waitall(nreq, m_reqs.data(), MPI_STATUSES_IGNORE);
    }

void mpcd::CellList::updateLocalFromGhosts()
    {
    if (!m_decomposition || m_num_mpcd_ghosts_send == 0)
        {
        return;
        }
    ArrayHandle<Scalar4> h_mpcd_vel_sendbuf(m_mpcd_vel_sendbuf,
                                            access_location::host,
                                            access_mode::read);
    ArrayHandle<uint2> h_mpcd_comm_key(m_mpcd_comm_key, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_vel(m_mpcd_pdata->getVelocities(),
                               access_location::host,
                               access_mode::readwrite);
    for (unsigned int cur_p = 0; cur_p < m_num_mpcd_ghosts_send; ++cur_p)
        {
        const Scalar4 vel_i = h_mpcd_vel_sendbuf.data[cur_p];
        const double3 new_vel = make_double3(vel_i.x, vel_i.y, vel_i.z);
        const unsigned int idx = h_mpcd_comm_key.data[cur_p].y;
        const Scalar4 old_vel = h_vel.data[idx];
        h_vel.data[idx] = make_scalar4(new_vel.x, new_vel.y, new_vel.z, old_vel.w);
        }
    }

bool mpcd::CellList::checkGhostConditions()
    {
    bool result = false;

    uint3 conditions = m_conditions.readFlags();
    if (conditions.x)
        {
        unsigned int n = conditions.x - 1;
        if (n < getNMPCDGhosts())
            {
            m_exec_conf->msg->errorAllRanks()
                << "MPCD ghost particle " << n << " has no valid cell" << std::endl;
            }
        throw std::runtime_error("Error computing cell list");
        }

    return result;
    }
#endif // ENABLE_MPI

void mpcd::CellList::updateFlags()
    {
    mpcd::detail::ThermoFlags flags;

    if (!m_flag_signal.empty())
        {
        m_flag_signal.emit_accumulate([&](mpcd::detail::ThermoFlags f) { flags |= f; });
        }

    m_flags = flags;
    }

bool mpcd::CellList::checkConditions()
    {
    bool result = false;

    uint3 conditions = m_conditions.readFlags();
    if (conditions.x)
        {
        unsigned int n = conditions.x - 1;
        if (n < m_mpcd_pdata->getN())
            m_exec_conf->msg->errorAllRanks()
                << "MPCD particle " << n << " has no valid cell" << std::endl;
        else if (n < m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual())
            m_exec_conf->msg->errorAllRanks()
                << "MPCD virtual particle " << n << " has no valid cell" << std::endl;
        else
            {
            ArrayHandle<unsigned int> h_embed_member_idx(m_embed_group->getIndexArray(),
                                                         access_location::host,
                                                         access_mode::read);
            m_exec_conf->msg->errorAllRanks()
                << "Embedded particle "
                << h_embed_member_idx.data[n - (m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual())]
                << " has no valid cell" << std::endl;
            }
        throw std::runtime_error("Error computing cell list");
        }
    if (conditions.y)
        {
        unsigned int n = conditions.y - 1;
        if (n < m_mpcd_pdata->getN())
            m_exec_conf->msg->errorAllRanks()
                << "MPCD particle " << n << " has position NaN" << std::endl;
        else if (n < m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual())
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
        std::string msg;
        if (n < m_mpcd_pdata->getN() + m_mpcd_pdata->getNVirtual())
            {
            ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(),
                                       access_location::host,
                                       access_mode::read);
            pos_empty_i = h_pos.data[n];
            if (n < m_mpcd_pdata->getN())
                msg = "MPCD particle is no longer in the simulation box";
            else
                msg = "MPCD virtual particle is no longer in the simulation box";
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
            msg = "Embedded particle is no longer in the simulation box";
            }

        Scalar3 pos = make_scalar3(pos_empty_i.x, pos_empty_i.y, pos_empty_i.z);
        m_exec_conf->msg->errorAllRanks()
            << msg << std::endl
            << "Cartesian coordinates: " << std::endl
            << "x: " << pos.x << " y: " << pos.y << " z: " << pos.z << std::endl
            << "Grid shift: " << std::endl
            << "x: " << m_grid_shift.x << " y: " << m_grid_shift.y << " z: " << m_grid_shift.z
            << std::endl;
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
        hoomd::UniformDistribution<Scalar> uniform(Scalar(-0.5), Scalar(0.5));
        Scalar3 shift = make_scalar3(
            uniform(rng) * m_global_cell_dim_inv.x,
            uniform(rng) * m_global_cell_dim_inv.y,
            (m_sysdef->getNDimensions() == 3) ? uniform(rng) * m_global_cell_dim_inv.z : 0);
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

    m_max_grid_shift = Scalar(0.5) * m_global_cell_dim_inv;

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
 * \param global Global cell index to check if it exists in local cell index
 * \returns True if the global cell lies within the local grid, false otherwise
 *
 */
const bool mpcd::CellList::hasGlobalCell(const int3& global)
    {
    computeDimensions();

    int3 local = make_int3(global.x - m_origin_idx.x,
                           global.y - m_origin_idx.y,
                           global.z - m_origin_idx.z);
    return ((0 <= local.x && local.x < (int)m_cell_dim.x)
            && (0 <= local.y && local.y < (int)m_cell_dim.y)
            && (0 <= local.z && local.z < (int)m_cell_dim.z));
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
