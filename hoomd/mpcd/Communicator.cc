// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*!
 * \file mpcd/Communicator.cc
 * \brief Implements the mpcd::Communicator class
 */

#ifdef ENABLE_MPI

#include "Communicator.h"
#include "hoomd/SystemDefinition.h"

#include <algorithm>
#include <pybind11/stl.h>

using namespace std;

namespace hoomd
    {
// 27 neighbors is the maximum in 3 dimensions
const unsigned int mpcd::Communicator::neigh_max = 27;

/*!
 * \param sysdef System definition the communicator is associated with
 * \param decomposition Domain decomposition of the global box
 */
mpcd::Communicator::Communicator(std::shared_ptr<SystemDefinition> sysdef)
    : m_sysdef(sysdef), m_pdata(m_sysdef->getParticleData()), m_exec_conf(m_pdata->getExecConf()),
      m_mpcd_pdata(m_sysdef->getMPCDParticleData()), m_mpi_comm(m_exec_conf->getMPICommunicator()),
      m_decomposition(m_pdata->getDomainDecomposition()), m_is_communicating(false),
      m_check_decomposition(true), m_nneigh(0), m_n_unique_neigh(0), m_sendbuf(m_exec_conf),
      m_recvbuf(m_exec_conf), m_force_migrate(false)
    {
    // initialize array of neighbor processor ids
    assert(m_mpi_comm);
    assert(m_decomposition);

    m_exec_conf->msg->notice(5) << "Constructing MPCD Communicator" << endl;

    // allocate memory
    GPUArray<unsigned int> neighbors(neigh_max, m_exec_conf);
    m_neighbors.swap(neighbors);

    GPUArray<unsigned int> unique_neighbors(neigh_max, m_exec_conf);
    m_unique_neighbors.swap(unique_neighbors);

    // neighbor masks
    GPUArray<unsigned int> adj_mask(neigh_max, m_exec_conf);
    m_adj_mask.swap(adj_mask);

    // create new data type for the pdata_element
    const int nitems = 4;
    int blocklengths[nitems] = {4, 4, 1, 1};
    MPI_Datatype types[nitems] = {MPI_HOOMD_SCALAR, MPI_HOOMD_SCALAR, MPI_UNSIGNED, MPI_UNSIGNED};
    MPI_Aint offsets[nitems];
    offsets[0] = offsetof(mpcd::detail::pdata_element, pos);
    offsets[1] = offsetof(mpcd::detail::pdata_element, vel);
    offsets[2] = offsetof(mpcd::detail::pdata_element, tag);
    offsets[3] = offsetof(mpcd::detail::pdata_element, comm_flag);
    // this needs to be made via the resize method to get its upper bound correctly
    MPI_Datatype tmp;
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &tmp);
    MPI_Type_commit(&tmp);
    MPI_Type_create_resized(tmp, 0, sizeof(mpcd::detail::pdata_element), &m_pdata_element);
    MPI_Type_commit(&m_pdata_element);
    MPI_Type_free(&tmp);

    initializeNeighborArrays();
    }

mpcd::Communicator::~Communicator()
    {
    m_exec_conf->msg->notice(5) << "Destroying MPCD Communicator" << std::endl;
    detachCallbacks();
    MPI_Type_free(&m_pdata_element);
    }

void mpcd::Communicator::initializeNeighborArrays()
    {
    Index3D di = m_decomposition->getDomainIndexer();

    uint3 mypos = m_decomposition->getGridPos();
    int l = mypos.x;
    int m = mypos.y;
    int n = mypos.z;

    ArrayHandle<unsigned int> h_neighbors(m_neighbors,
                                          access_location::host,
                                          access_mode::overwrite);
    ArrayHandle<unsigned int> h_adj_mask(m_adj_mask, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_cart_ranks(m_decomposition->getCartRanks(),
                                           access_location::host,
                                           access_mode::read);

    m_nneigh = 0;

    // loop over neighbors
    for (int ix = -1; ix <= 1; ix++)
        {
        int i = ix + l;
        if (i == (int)di.getW())
            i = 0;
        else if (i < 0)
            i += di.getW();

        // only if communicating along x-direction
        if (ix && di.getW() == 1)
            continue;

        for (int iy = -1; iy <= 1; iy++)
            {
            int j = iy + m;

            if (j == (int)di.getH())
                j = 0;
            else if (j < 0)
                j += di.getH();

            // only if communicating along y-direction
            if (iy && di.getH() == 1)
                continue;

            for (int iz = -1; iz <= 1; iz++)
                {
                int k = iz + n;

                if (k == (int)di.getD())
                    k = 0;
                else if (k < 0)
                    k += di.getD();

                // only if communicating along z-direction
                if (iz && di.getD() == 1)
                    continue;

                // exclude ourselves
                if (!ix && !iy && !iz)
                    continue;

                unsigned int dir = ((iz + 1) * 3 + (iy + 1)) * 3 + (ix + 1);
                unsigned int mask = 1 << dir;

                unsigned int neighbor = h_cart_ranks.data[di(i, j, k)];
                h_neighbors.data[m_nneigh] = neighbor;
                h_adj_mask.data[m_nneigh] = mask;
                m_nneigh++;
                }
            }
        }

    ArrayHandle<unsigned int> h_unique_neighbors(m_unique_neighbors,
                                                 access_location::host,
                                                 access_mode::overwrite);

    // filter neighbors, combining adjacency masks
    std::map<unsigned int, unsigned int> neigh_map;
    for (unsigned int i = 0; i < m_nneigh; ++i)
        {
        unsigned int m = 0;

        for (unsigned int j = 0; j < m_nneigh; ++j)
            if (h_neighbors.data[j] == h_neighbors.data[i])
                m |= h_adj_mask.data[j];

        neigh_map.insert(std::make_pair(h_neighbors.data[i], m));
        }

    m_n_unique_neigh = (unsigned int)neigh_map.size();

    m_unique_neigh_map.clear();
    n = 0;
    for (auto it = neigh_map.begin(); it != neigh_map.end(); ++it)
        {
        h_unique_neighbors.data[n] = it->first;
        m_unique_neigh_map.insert(std::make_pair(it->first, n));
        h_adj_mask.data[n] = it->second;
        n++;
        }
    }

/*!
 * \param timestep Current timestep for communication
 */
void mpcd::Communicator::communicate(uint64_t timestep)
    {
    if (!m_cl)
        {
        throw std::runtime_error("Cell list has not been set");
        }

    if (m_is_communicating)
        {
        m_exec_conf->msg->warning()
            << "MPCD communication currently underway, ignoring request" << std::endl;
        return;
        }

    // Guard to prevent recursive triggering of migration
    m_is_communicating = true;

    // force the cell list to adopt the correct dimensions before proceeding,
    // which will trigger any size change signals
    m_cl->computeDimensions();

    if (m_check_decomposition)
        {
        checkDecomposition();
        m_check_decomposition = false;
        }

    // check for and attempt particle migration
    bool migrate = m_force_migrate;
    if (!migrate)
        {
        m_migrate_requests.emit_accumulate([&](bool r) { migrate = migrate || r; }, timestep);
        }
    if (migrate)
        {
        migrateParticles(timestep);
        m_force_migrate = false;
        }

    m_is_communicating = false;
    }

namespace mpcd
    {
namespace detail
    {
//! Partition operation for migrating particles
/*!
 * This functor is used in combination with std::partition to sort migrated
 * particles in the send / receive buffers. The communication flags in an
 * mpcd::detail::pdata_element are checked with bitwise AND against a mask.
 * The result is cast to bool and negated. This behavior results in std::partition
 * moving all particles for which the flags are bitwise AND true with the mask
 * to the end of the buffer. This effectively compacts the buffers in place for
 * sending.
 */
class MigratePartitionOp
    {
    public:
    MigratePartitionOp(const unsigned int mask_) : mask(mask_) { }

    inline bool operator()(const mpcd::detail::pdata_element& e) const
        {
        return !static_cast<bool>(e.comm_flag & mask);
        }

    private:
    const unsigned int mask; //!< Mask for the current communication stage
    };
    } // namespace detail
    } // namespace mpcd

void mpcd::Communicator::migrateParticles(uint64_t timestep)
    {
    if (m_mpcd_pdata->getNVirtual() > 0)
        {
        m_exec_conf->msg->warning()
            << "MPCD communication with virtual particles set is not supported, removing them."
            << std::endl;
        m_mpcd_pdata->removeVirtualParticles();
        }

    // determine local particles that are to be sent to neighboring processors
    const BoxDim box = m_cl->getCoverageBox();
    setCommFlags(box);

    // fill send buffer once
    m_mpcd_pdata->removeParticles(m_sendbuf, 0xffffffff, timestep);

    // fill the buffers and send in each direction
    unsigned int n_recv = 0;
    for (unsigned int dim = 0; dim < m_sysdef->getNDimensions(); ++dim)
        {
        if (!isCommunicating(static_cast<mpcd::detail::face>(2 * dim)))
            continue;

        const unsigned int right_mask = 1 << (2 * dim);
        const unsigned int left_mask = 1 << (2 * dim + 1);
        const unsigned int stage_mask = right_mask | left_mask;

        // neighbor ranks
        const unsigned int right_neigh = m_decomposition->getNeighborRank(2 * dim);
        const unsigned int left_neigh = m_decomposition->getNeighborRank(2 * dim + 1);

        // partition the send buffer by destination, leaving unsent particles at the front
        unsigned int n_keep, n_send_left, n_send_right;
            {
            ArrayHandle<mpcd::detail::pdata_element> h_sendbuf(m_sendbuf,
                                                               access_location::host,
                                                               access_mode::readwrite);

            // first, partition off particles that may be sent in either direction
            mpcd::detail::MigratePartitionOp part_op(stage_mask);
            auto bound = std::partition(h_sendbuf.data, h_sendbuf.data + m_sendbuf.size(), part_op);
            n_keep = (unsigned int)(&(*bound) - h_sendbuf.data);

            // then, partition the sent particles into the left and right ranks so that particles
            // getting sent right come first
            if (left_neigh != right_neigh)
                {
                // partition the remaining particles left and right
                mpcd::detail::MigratePartitionOp sort_op(left_mask);
                bound = std::partition(h_sendbuf.data + n_keep,
                                       h_sendbuf.data + m_sendbuf.size(),
                                       sort_op);
                n_send_right = (unsigned int)(&(*bound) - (h_sendbuf.data + n_keep));
                n_send_left = (unsigned int)(m_sendbuf.size() - n_keep - n_send_right);
                }
            else
                {
                n_send_right = (unsigned int)(m_sendbuf.size() - n_keep);
                n_send_left = 0;
                }
            }

        // communicate size of the message that will contain the particle data
        unsigned int n_recv_left, n_recv_right;
        if (left_neigh != right_neigh)
            {
            m_reqs.resize(4);
            MPI_Isend(&n_send_right, 1, MPI_UNSIGNED, right_neigh, 0, m_mpi_comm, &m_reqs[0]);
            MPI_Irecv(&n_recv_right, 1, MPI_UNSIGNED, right_neigh, 0, m_mpi_comm, &m_reqs[1]);
            MPI_Isend(&n_send_left, 1, MPI_UNSIGNED, left_neigh, 0, m_mpi_comm, &m_reqs[2]);
            MPI_Irecv(&n_recv_left, 1, MPI_UNSIGNED, left_neigh, 0, m_mpi_comm, &m_reqs[3]);
            MPI_Waitall(4, m_reqs.data(), MPI_STATUSES_IGNORE);
            }
        else
            {
            // send right, receive left (same thing, really) only if neighbors match
            n_recv_right = 0;
            m_reqs.resize(2);
            MPI_Isend(&n_send_right, 1, MPI_UNSIGNED, right_neigh, 0, m_mpi_comm, &m_reqs[0]);
            MPI_Irecv(&n_recv_left, 1, MPI_UNSIGNED, left_neigh, 0, m_mpi_comm, &m_reqs[1]);
            MPI_Waitall(2, m_reqs.data(), MPI_STATUSES_IGNORE);
            }

        // exchange particle data
        m_recvbuf.resize(n_recv + n_recv_left + n_recv_right);
            {
            ArrayHandle<mpcd::detail::pdata_element> h_sendbuf(m_sendbuf,
                                                               access_location::host,
                                                               access_mode::read);
            ArrayHandle<mpcd::detail::pdata_element> h_recvbuf(m_recvbuf,
                                                               access_location::host,
                                                               access_mode::overwrite);
            m_reqs.resize(4);
            int nreq = 0;
            if (n_send_right != 0)
                {
                MPI_Isend(h_sendbuf.data + n_keep,
                          n_send_right,
                          m_pdata_element,
                          right_neigh,
                          1,
                          m_mpi_comm,
                          &m_reqs[nreq++]);
                }
            if (n_send_left != 0)
                {
                MPI_Isend(h_sendbuf.data + n_keep + n_send_right,
                          n_send_left,
                          m_pdata_element,
                          left_neigh,
                          1,
                          m_mpi_comm,
                          &m_reqs[nreq++]);
                }
            if (n_recv_right != 0)
                {
                MPI_Irecv(h_recvbuf.data + n_recv,
                          n_recv_right,
                          m_pdata_element,
                          right_neigh,
                          1,
                          m_mpi_comm,
                          &m_reqs[nreq++]);
                }
            if (n_recv_left != 0)
                {
                MPI_Irecv(h_recvbuf.data + n_recv + n_recv_right,
                          n_recv_left,
                          m_pdata_element,
                          left_neigh,
                          1,
                          m_mpi_comm,
                          &m_reqs[nreq++]);
                }
            MPI_Waitall(nreq, m_reqs.data(), MPI_STATUSES_IGNORE);
            }

            // now we pass through and unpack the particles, either by holding onto them in the
            // receive buffer or by passing them back into the send buffer for the next stage
            {
            // partition the receive buffer so that particles that need to be sent are at the end
            ArrayHandle<mpcd::detail::pdata_element> h_recvbuf(m_recvbuf,
                                                               access_location::host,
                                                               access_mode::readwrite);
            mpcd::detail::MigratePartitionOp part_op(~stage_mask);
            auto bound = std::partition(h_recvbuf.data + n_recv,
                                        h_recvbuf.data + m_recvbuf.size(),
                                        part_op);
            n_recv = (unsigned int)(&(*bound) - h_recvbuf.data);

            // move particles to resend over to the send buffer and unset the bits from this stage
            const unsigned int n_resend = (unsigned int)(m_recvbuf.size() - n_recv);
            m_sendbuf.resize(n_keep + n_resend);
            ArrayHandle<mpcd::detail::pdata_element> h_sendbuf(m_sendbuf,
                                                               access_location::host,
                                                               access_mode::readwrite);
            std::copy(h_recvbuf.data + n_recv,
                      h_recvbuf.data + m_recvbuf.size(),
                      h_sendbuf.data + n_keep);
            for (unsigned int idx = n_keep; idx < m_sendbuf.size(); ++idx)
                {
                h_sendbuf.data[idx].comm_flag &= ~stage_mask;
                }
            }
        m_recvbuf.resize(n_recv); // free up memory from the end of the receive buffer
        } // end dir loop

        // fill particle data with wrapped, received particles
        {
        ArrayHandle<mpcd::detail::pdata_element> h_recvbuf(m_recvbuf,
                                                           access_location::host,
                                                           access_mode::readwrite);
        const BoxDim wrap_box = getWrapBox(box);
        for (unsigned int idx = 0; idx < n_recv; ++idx)
            {
            mpcd::detail::pdata_element& p = h_recvbuf.data[idx];
            Scalar4& postype = p.pos;
            int3 image = make_int3(0, 0, 0);

            wrap_box.wrap(postype, image);
            }
        }

    // this mask will totally unset any bits that could still be set (there should be none)
    m_mpcd_pdata->addParticles(m_recvbuf, 0xffffffff, timestep);
    }

/*!
 * \param box Bounding box
 *
 * Particles lying outside of \a box have their communication flags set along
 * that face.
 */
void mpcd::Communicator::setCommFlags(const BoxDim& box)
    {
    // mark all particles which have left the box for sending
    unsigned int N = m_mpcd_pdata->getN();
    ArrayHandle<Scalar4> h_pos(m_mpcd_pdata->getPositions(),
                               access_location::host,
                               access_mode::read);
    ArrayHandle<unsigned int> h_comm_flag(m_mpcd_pdata->getCommFlags(),
                                          access_location::host,
                                          access_mode::overwrite);

    // since box is orthorhombic, just use branching to compute comm flags
    const Scalar3 lo = box.getLo();
    const Scalar3 hi = box.getHi();
    for (unsigned int idx = 0; idx < N; ++idx)
        {
        const Scalar4& postype = h_pos.data[idx];
        const Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

        unsigned int flags = 0;
        if (pos.x >= hi.x)
            flags |= static_cast<unsigned int>(mpcd::detail::send_mask::east);
        else if (pos.x < lo.x)
            flags |= static_cast<unsigned int>(mpcd::detail::send_mask::west);

        if (pos.y >= hi.y)
            flags |= static_cast<unsigned int>(mpcd::detail::send_mask::north);
        else if (pos.y < lo.y)
            flags |= static_cast<unsigned int>(mpcd::detail::send_mask::south);

        if (pos.z >= hi.z)
            flags |= static_cast<unsigned int>(mpcd::detail::send_mask::up);
        else if (pos.z < lo.z)
            flags |= static_cast<unsigned int>(mpcd::detail::send_mask::down);

        h_comm_flag.data[idx] = flags;
        }
    }

/*!
 * Checks that the simulation box is not overdecomposed so that communication can
 * be achieved using the assumed single step. This is a collective call that
 * raises an error on all ranks if any rank reports an error.
 */
void mpcd::Communicator::checkDecomposition()
    {
    // determine the bounds of this box
    const BoxDim box = m_cl->getCoverageBox();
    const Scalar3 lo = box.getLo();
    const Scalar3 hi = box.getHi();

    // bounds of global box for wrapping
    const BoxDim& global_box = m_pdata->getGlobalBox();
    const Scalar3 global_L = global_box.getL();

    // 4 requests will be needed per call
    m_reqs.resize(4);

    int error = 0;
    for (unsigned int dim = 0; dim < m_sysdef->getNDimensions(); ++dim)
        {
        if (!isCommunicating(static_cast<mpcd::detail::face>(2 * dim)))
            continue;

        Scalar my_lo, my_hi;
        if (dim == 0) // x
            {
            my_lo = lo.x;
            my_hi = hi.x;
            }
        else if (dim == 1) // y
            {
            my_lo = lo.y;
            my_hi = hi.y;
            }
        else // z
            {
            my_lo = lo.z;
            my_hi = hi.z;
            }

        // get the neighbors this rank sends to
        const unsigned int right_neigh = m_decomposition->getNeighborRank(2 * dim);
        const unsigned int left_neigh = m_decomposition->getNeighborRank(2 * dim + 1);

        // send boundaries left and right
        Scalar right_lo, left_hi;
        MPI_Isend(&my_hi, 1, MPI_HOOMD_SCALAR, right_neigh, 0, m_mpi_comm, &m_reqs[0]);
        MPI_Isend(&my_lo, 1, MPI_HOOMD_SCALAR, left_neigh, 1, m_mpi_comm, &m_reqs[1]);
        MPI_Irecv(&right_lo, 1, MPI_HOOMD_SCALAR, right_neigh, 1, m_mpi_comm, &m_reqs[2]);
        MPI_Irecv(&left_hi, 1, MPI_HOOMD_SCALAR, left_neigh, 0, m_mpi_comm, &m_reqs[3]);
        MPI_Waitall(4, m_reqs.data(), MPI_STATUSES_IGNORE);

        // if at right edge of simulation box, then wrap the lo back in
        if (m_decomposition->isAtBoundary(2 * dim)) // right edge
            {
            if (dim == 0) // x
                {
                right_lo += global_L.x;
                }
            else if (dim == 1) // y
                {
                right_lo += global_L.y;
                }
            else // z
                {
                right_lo += global_L.z;
                }
            }
        // otherwise if at left of simulation, wrap the hi back in
        else if (m_decomposition->isAtBoundary(2 * dim + 1)) // left edge
            {
            if (dim == 0) // x
                {
                left_hi -= global_L.x;
                }
            else if (dim == 1) // y
                {
                left_hi -= global_L.y;
                }
            else // z
                {
                left_hi -= global_L.z;
                }
            }

        // check for an error between neighbors and save it
        // we can't quit early because this could cause a stall
        if (right_lo < my_lo || left_hi >= my_hi)
            {
            error = 1;
            }
        }

    // check for any error on all ranks with all-to-all communication
    MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_INT, MPI_SUM, m_mpi_comm);
    if (error)
        {
        m_is_communicating = false;
        m_exec_conf->msg->error() << "Simulation box is overdecomposed for MPCD communicator"
                                  << std::endl;
        throw std::runtime_error("Overdecomposed simulation box");
        }
    }

/*!
 * \param box Box used to determine communication for this rank
 * \returns A box suitable for wrapping received particles through
 *          the global boundary.
 *
 * If a domain lies on a boundary, shift the global box so that it covers the
 * region lying outside the box.
 * \b Assumptions
 *  1. The boxes are orthorhombic.
 *  2. The communication \a box can only exceed the global box in one dimension.
 *     (This should be guaranteed by the minimum domain size of the cell list.)
 */
BoxDim mpcd::Communicator::getWrapBox(const BoxDim& box)
    {
    // bounds of the current box
    const Scalar3 hi = box.getHi();
    const Scalar3 lo = box.getLo();

    // bounds of the global box
    const BoxDim& global_box = m_pdata->getGlobalBox();
    const Scalar3 global_hi = global_box.getHi();
    const Scalar3 global_lo = global_box.getLo();

    // shift box
    uint3 grid_size = m_decomposition->getGridSize();
    Scalar3 shift = make_scalar3(0., 0., 0.);
    if (grid_size.x > 1)
        {
        // exclusive or, it doesn't make sense for both these conditions to be true
        assert(!(hi.x > global_hi.x && lo.x < global_lo.x));

        if (hi.x > global_hi.x)
            shift.x = hi.x - global_hi.x;
        else if (lo.x < global_lo.x)
            shift.x = lo.x - global_lo.x;
        }
    if (grid_size.y > 1)
        {
        assert(!(hi.y > global_hi.y && lo.y < global_lo.y));

        if (hi.y > global_hi.y)
            shift.y = hi.y - global_hi.y;
        else if (lo.y < global_lo.y)
            shift.y = lo.y - global_lo.y;
        }
    if (grid_size.z > 1)
        {
        assert(!(hi.z > global_hi.z && lo.z < global_lo.z));

        if (hi.z > global_hi.z)
            shift.z = hi.z - global_hi.z;
        else if (lo.z < global_lo.z)
            shift.z = lo.z - global_lo.z;
        }

    // only wrap in the direction being communicated
    uchar3 periodic = make_uchar3(0, 0, 0);
    periodic.x = (isCommunicating(mpcd::detail::face::east)) ? 1 : 0;
    periodic.y = (isCommunicating(mpcd::detail::face::north)) ? 1 : 0;
    periodic.z = (isCommunicating(mpcd::detail::face::up)) ? 1 : 0;

    return BoxDim(global_lo + shift, global_hi + shift, periodic);
    }

void mpcd::Communicator::attachCallbacks()
    {
    assert(m_cl);
    m_cl->getSizeChangeSignal().connect<mpcd::Communicator, &mpcd::Communicator::slotBoxChanged>(
        this);
    }

void mpcd::Communicator::detachCallbacks()
    {
    if (m_cl)
        {
        m_cl->getSizeChangeSignal()
            .disconnect<mpcd::Communicator, &mpcd::Communicator::slotBoxChanged>(this);
        }
    }

namespace mpcd
    {
namespace detail
    {
/*!
 * \param m Python module to export to
 */
void export_Communicator(pybind11::module& m)
    {
    pybind11::class_<mpcd::Communicator, std::shared_ptr<mpcd::Communicator>>(m, "Communicator")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
    }
    } // namespace detail
    } // namespace mpcd
    } // end namespace hoomd
#endif // ENABLE_MPI
