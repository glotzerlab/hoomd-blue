// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file LoadBalancer.cc
    \brief Defines the LoadBalancer class
*/

#include "LoadBalancer.h"
#include "Communicator.h"

#include "hoomd/extern/BVLSSolver.h"
#include <Eigen/Dense>

#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

using namespace std;

namespace hoomd
    {
/*!
 * \param sysdef System definition
 * \param decomposition Domain decomposition
 */
LoadBalancer::LoadBalancer(std::shared_ptr<SystemDefinition> sysdef,
                           std::shared_ptr<Trigger> trigger)
    : Tuner(sysdef, trigger),
#ifdef ENABLE_MPI
      m_mpi_comm(m_exec_conf->getMPICommunicator()),
#endif
      m_max_imbalance(Scalar(1.0)), m_recompute_max_imbalance(true), m_needs_migrate(false),
      m_needs_recount(false), m_tolerance(Scalar(1.05)), m_maxiter(1), m_max_scale(Scalar(0.05)),
      m_N_own(m_pdata->getN()), m_max_max_imbalance(1.0), m_total_max_imbalance(0.0), m_n_calls(0),
      m_n_iterations(0), m_n_rebalances(0)
    {
    m_exec_conf->msg->notice(5) << "Constructing LoadBalancer" << endl;

#ifdef ENABLE_MPI
    m_decomposition = sysdef->getParticleData()->getDomainDecomposition();

    // default initialize the load balancing based on domain grid
    if (m_sysdef->isDomainDecomposed())
        {
        const Index3D& di = m_decomposition->getDomainIndexer();
        m_enable_x = (di.getW() > 1);
        m_enable_y = (di.getH() > 1);
        m_enable_z = (di.getD() > 1);

        auto comm_weak = m_sysdef->getCommunicator();
        assert(comm_weak.lock());
        m_comm = comm_weak.lock();
        }
    else
#endif // ENABLE_MPI
        {
        m_enable_x = m_enable_y = m_enable_z = false;
        }
    }

LoadBalancer::~LoadBalancer()
    {
    m_exec_conf->msg->notice(5) << "Destroying LoadBalancer" << endl;
    }

/*!
 * \param timestep Current time step of the simulation
 *
 * Computes the load imbalance along each slice and adjusts the domain boundaries. This process is
 * repeated iteratively in each dimension taking into account the adjusted boundaries each time.
 */
void LoadBalancer::update(uint64_t timestep)
    {
    Updater::update(timestep);

#ifdef ENABLE_MPI
    // do nothing if this run is not on MPI with more than 1 rank
    if (!m_sysdef->isDomainDecomposed())
        return;

    // no adjustment has been made yet, so set m_N_own to the number of particles on the rank
    resetNOwn(m_pdata->getN());

    // figure out which rank is the reduction root for broadcasting
    const Index3D& di = m_decomposition->getDomainIndexer();
    unsigned int reduce_root(0);
        {
        ArrayHandle<unsigned int> h_cart_ranks(m_decomposition->getCartRanks(),
                                               access_location::host,
                                               access_mode::read);
        reduce_root = h_cart_ranks.data[di(0, 0, 0)];
        }

    // get the minimum domain size
    const BoxDim& box = m_pdata->getGlobalBox();
    Scalar3 L = box.getL();
    const Scalar3 min_domain_frac
        = Scalar(2.0) * m_comm->getGhostLayerMaxWidth() / box.getNearestPlaneDistance();

    // compute the current imbalance always for the average in printed stats
    m_total_max_imbalance += getMaxImbalance();
    ++m_n_calls;

    // attempt load balancing
    for (unsigned int cur_iter = 0; cur_iter < m_maxiter && getMaxImbalance() > m_tolerance;
         ++cur_iter)
        {
        // increment the number of attempted balances
        ++m_n_iterations;

        for (unsigned int dim = 0;
             dim < m_sysdef->getNDimensions() && getMaxImbalance() > m_tolerance;
             ++dim)
            {
            Scalar L_i(0.0);
            Scalar min_frac_i(0.0);
            if (dim == 0)
                {
                if (!m_enable_x || di.getW() == 1)
                    continue; // skip this dimension if balancing is turned off
                L_i = L.x;
                min_frac_i = min_domain_frac.x;
                }
            else if (dim == 1)
                {
                if (!m_enable_y || di.getH() == 1)
                    continue;
                L_i = L.y;
                min_frac_i = min_domain_frac.y;
                }
            else
                {
                if (!m_enable_z || di.getD() == 1)
                    continue;
                L_i = L.z;
                min_frac_i = min_domain_frac.z;
                }

            vector<unsigned int> N_i;
            bool adjusted = false;

            // reduce the number of particles in the slice along dim
            bool active = reduce(N_i, dim, reduce_root);

            // attempt an adjustment
            vector<Scalar> cum_frac = m_decomposition->getCumulativeFractions(dim);
            if (active)
                {
                adjusted = adjust(cum_frac, N_i, L_i, min_frac_i);
                }

            // broadcast if an adjustment has been made on the root
            bcast(adjusted, reduce_root, m_mpi_comm);

            // update the cumulative fractions and signal
            if (adjusted)
                {
                m_decomposition->setCumulativeFractions(dim, cum_frac, reduce_root);
                m_pdata->setGlobalBox(box); // force a domain resizing to trigger
                signalResize();
                }
            }

        // force a particle migration if one is needed
        if (m_needs_migrate)
            {
            m_comm->forceMigrate();
            m_comm->communicate(timestep);
            resetNOwn(m_pdata->getN());
            m_needs_migrate = false;

            // increment the number of rebalances actually performed
            ++m_n_rebalances;
            }
        }
#endif // ENABLE_MPI
    }

#ifdef ENABLE_MPI

/*!
 * Computes the imbalance factor I = N / <N> for each rank, and computes the maximum among all
 * ranks.
 */
Scalar LoadBalancer::getMaxImbalance()
    {
    if (m_recompute_max_imbalance)
        {
        Scalar cur_imb = Scalar(getNOwn())
                         / (Scalar(m_pdata->getNGlobal()) / Scalar(m_exec_conf->getNRanks()));
        Scalar max_imb(0.0);
        MPI_Allreduce(&cur_imb, &max_imb, 1, MPI_HOOMD_SCALAR, MPI_MAX, m_mpi_comm);

        m_max_imbalance = max_imb;
        m_recompute_max_imbalance = false;

        // save as a statistic if new max imbalance
        if (m_max_imbalance > m_max_max_imbalance)
            m_max_max_imbalance = m_max_imbalance;
        }
    return m_max_imbalance;
    }

/*!
 * \param N_i Vector holding the total number of particles in each slice (will be allocated on call)
 * \param dim The dimension of the slices (x=0, y=1, z=2)
 * \param reduce_root The rank to perform the reduction on
 * \returns true if the current rank holds the active \a N_i
 *
 * \post \a N_i holds the number of particles in each slice along \a dim
 *
 * \note reduce() relies on collective MPI calls, and so all ranks must call it. However, for
 * efficiency the data will be active only on Cartesian rank \a reduce_root, as indicated by the
 * return value. As a result, only \a reduce_root actually needs to allocate memory for \a N_i.
 *
 * The reduction is performed by performing an all-to-one gather, followed by summation on \a
 * reduce_root. This operation may be suboptimal for very large numbers of processors, and could be
 * replaced by cascading send operations down dimensions. Generally, load balancing should not be
 * performed too frequently, and so we do not pursue this optimization right now.
 */
bool LoadBalancer::reduce(std::vector<unsigned int>& N_i,
                          unsigned int dim,
                          unsigned int reduce_root)
    {
    // do nothing if there is only one rank
    if (N_i.size() == 1)
        return false;

    const Index3D& di = m_decomposition->getDomainIndexer();
    std::vector<unsigned int> N_per_rank(di.getNumElements());

    // get the number of particles the current rank owns (the quantity to be reduced)
    unsigned int N_own = getNOwn();

    MPI_Gather(&N_own, 1, MPI_UNSIGNED, &N_per_rank[0], 1, MPI_UNSIGNED, reduce_root, m_mpi_comm);

    // only the root rank performs the reduction
    if (m_exec_conf->getRank() != reduce_root)
        return false;

    // rearrange the data from ranks to cartesian order in case it is jumbled around
    ArrayHandle<unsigned int> h_cart_ranks_inv(m_decomposition->getInverseCartRanks(),
                                               access_location::host,
                                               access_mode::read);
    std::vector<unsigned int> N_per_cart_rank(di.getNumElements());
    for (unsigned int cur_rank = 0; cur_rank < di.getNumElements(); ++cur_rank)
        {
        N_per_cart_rank[h_cart_ranks_inv.data[cur_rank]] = N_per_rank[cur_rank];
        }

    // perform the summation along dim in as cache friendly of a way as we can manage
    if (dim == 0) // to x
        {
        N_i.clear();
        N_i.resize(di.getW());
        for (unsigned int i = 0; i < di.getW(); ++i)
            {
            N_i[i] = 0;
            for (unsigned int k = 0; k < di.getD(); ++k)
                {
                for (unsigned int j = 0; j < di.getH(); ++j)
                    {
                    N_i[i] += N_per_cart_rank[di(i, j, k)];
                    }
                }
            }
        }
    else if (dim == 1) // to y
        {
        N_i.clear();
        N_i.resize(di.getH());
        for (unsigned int j = 0; j < di.getH(); ++j)
            {
            N_i[j] = 0;
            for (unsigned int k = 0; k < di.getD(); ++k)
                {
                for (unsigned int i = 0; i < di.getW(); ++i)
                    {
                    N_i[j] += N_per_cart_rank[di(i, j, k)];
                    }
                }
            }
        }
    else if (dim == 2) // to z
        {
        N_i.clear();
        N_i.resize(di.getD());
        for (unsigned int k = 0; k < di.getD(); ++k)
            {
            N_i[k] = 0;
            for (unsigned int j = 0; j < di.getH(); ++j)
                {
                for (unsigned int i = 0; i < di.getW(); ++i)
                    {
                    N_i[k] += N_per_cart_rank[di(i, j, k)];
                    }
                }
            }
        }
    else
        {
        throw runtime_error("Unknown dimension for particle reduction.");
        }

    return true;
    }

/*!
 * \param cum_frac_i The cumulative fraction array to write output into
 * \param N_i The reduced number of particles along the dimension
 * \param L_i The global box length along the dimension
 * \param min_frac_i The minimum fractional width of a domain
 *
 * \returns true if an adjustment occurred
 *
 * An adjustment is attempted as follows:
 *  1. Compute the imbalance factor (and scale factor) for each slice. Enforce the maximum 5% target
 * for adjustment in the scale factor. Compute the target new width for each rank.
 *  2. Construct a set of linear equations with box constraints that will enforce the necessary
 * constraints. This is done through a matrix A that converts slices between domains into widths
 * while conserving total length. A is then augmented to include an inequality constraint on the
 * minimum domain size through a slack variable w. Additional box constraints are enforced on the
 * new positions of the domain slices.
 *  3. Minimize the cost function using bounded variable least-squares (BVLSSolver).
 *  4. Sanity check the adjustment. Domains must be big enough and cannot have inverted. If the
 * minimization was successful, apply the adjustment to \a cum_frac_i.
 */
bool LoadBalancer::adjust(vector<Scalar>& cum_frac_i,
                          const vector<unsigned int>& N_i,
                          Scalar L_i,
                          Scalar min_frac_i)
    {
    if (N_i.size() == 1)
        return false;

    // target particles per rank is uniform distribution
    const Scalar target = Scalar(m_pdata->getNGlobal()) / Scalar(N_i.size());

    // make the minimum domain slightly bigger so that the optimization won't fail at equality
    const Scalar min_domain_size = Scalar(1.00001) * min_frac_i * L_i;
    // if system is overconstrained (exactly decomposed) don't do any adjusting
    if (min_domain_size * Scalar(N_i.size()) >= L_i)
        {
        return false;
        }

    // imbalance factors for each rank
    vector<Scalar> new_widths(N_i.size());
    for (unsigned int i = 0; i < N_i.size(); ++i)
        {
        const Scalar imb_factor = Scalar(N_i[i]) / target;
        Scalar scale_factor
            = (N_i[i] > 0)
                  ? Scalar(1.0) / imb_factor
                  : (Scalar(1.0)
                     + m_max_scale); // as in gromacs, use half the imbalance factor to scale

        // limit rescaling to 5% either direction
        // we should use absolute distance here, it is necessary to control balancing in corrugated
        // systems
        if (scale_factor > (Scalar(1.0) + m_max_scale))
            {
            scale_factor = (Scalar(1.0) + m_max_scale);
            }
        else if (scale_factor < (Scalar(1.0) - m_max_scale))
            {
            scale_factor = (Scalar(1.0) - m_max_scale);
            }

        // compute the new domain width (can't be smaller than the threshold, if it is, this is not
        // a free variable)
        new_widths[i] = scale_factor * (cum_frac_i[i + 1] - cum_frac_i[i]) * L_i;
        }

    // setup the augmented A matrix, with scale factor eps for the actual least squares part (to
    // enforce the inequality constraints correctly)
    const Scalar eps(0.001);
    unsigned int m = (unsigned int)N_i.size();
    unsigned int n = m - 1;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(2 * m, n + m);
    A(0, 0) = 1.0;
    A(m, 0) = eps;
    A(m - 1, n - 1) = -1.0;
    A(2 * m - 1, n - 1) = -eps;
    for (unsigned int i = 1; i < m - 1; ++i)
        {
        // upper left block is A
        A(i, i - 1) = -1.0;
        A(i, i) = 1.0;

        // lower left block is e A
        A(i + m, i - 1) = -eps;
        A(i + m, i) = eps;
        }
    A.block(0, n, m, m) = -1.0 * Eigen::MatrixXd::Identity(m, m);

    // initialize the augmented b array
    Eigen::VectorXd b(2 * m);
    b.fill(min_domain_size);
    for (unsigned int i = 0; i < m; ++i)
        {
        b(m + i) = eps * new_widths[i];
        }
    b(m - 1) -= L_i;           // the last equation applies the total length constraint
    b(2 * m - 1) -= eps * L_i; // the last equation applies the total length constraint

    // position constraints limit movement due to over-decomposition
    Eigen::VectorXd l(n + m), u(n + m);
    l.fill(0.0);
    u.fill(std::numeric_limits<Scalar>::max()); // the default is lower limit 0 and no upper limit
                                                // (for slack vars)
    for (unsigned int j = 0; j < n; ++j)
        {
        l(j) = Scalar(0.5) * (cum_frac_i[j] + cum_frac_i[j + 1]) * L_i;
        u(j) = Scalar(0.5) * (cum_frac_i[j + 1] + cum_frac_i[j + 2]) * L_i;
        }

    BVLSSolver solver(A, b, l, u);
    solver.setMaxIterations(3 * (n + m));
    solver.solve();
    if (solver.converged())
        {
        Eigen::VectorXd x = solver.getSolution();
        vector<Scalar> sorted_f(n);
        // do validation / sanity checking
        for (unsigned int cur_div = 0; cur_div < n; ++cur_div)
            {
            if (x(cur_div) < min_domain_size)
                {
                m_exec_conf->msg->warning()
                    << "LoadBalancer: no convergence, domains too small" << endl;
                return false;
                }
            sorted_f[cur_div] = x(cur_div) / L_i;
            if (cur_div > 0 && sorted_f[cur_div] < sorted_f[cur_div - 1])
                {
                m_exec_conf->msg->warning() << "LoadBalancer: domains attempting to flip" << endl;
                return false;
                }
            }
        // only push back the solution after we know it is valid
        for (unsigned int cur_div = 0; cur_div < sorted_f.size(); ++cur_div)
            {
            cum_frac_i[cur_div + 1] = sorted_f[cur_div];
            }
        return true;
        }
    else
        {
        m_exec_conf->msg->warning() << "LoadBalancer: converged load balance not found" << endl;
        return false;
        }

    return false;
    }

/*!
 * \param cnts Map holding result of number of particles on each rank that neighbors the local rank
 */
void LoadBalancer::countParticlesOffRank(std::map<unsigned int, unsigned int>& cnts)
    {
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_cart_ranks(m_decomposition->getCartRanks(),
                                           access_location::host,
                                           access_mode::read);

    const BoxDim& box = m_pdata->getBox();
    const Index3D& di = m_decomposition->getDomainIndexer();
    const uint3 rank_pos = m_decomposition->getGridPos();

    for (unsigned int cur_p = 0; cur_p < m_pdata->getN(); ++cur_p)
        {
        const Scalar4 cur_postype = h_pos.data[cur_p];
        const Scalar3 cur_pos = make_scalar3(cur_postype.x, cur_postype.y, cur_postype.z);
        const Scalar3 f = box.makeFraction(cur_pos);

        int3 grid_pos = make_int3(rank_pos.x, rank_pos.y, rank_pos.z);

        bool moved(false);
        if (f.x >= Scalar(1.0))
            {
            ++grid_pos.x;
            moved = true;
            }
        if (f.x < Scalar(0.0))
            {
            --grid_pos.x;
            moved = true;
            }

        if (f.y >= Scalar(1.0))
            {
            ++grid_pos.y;
            moved = true;
            }
        if (f.y < Scalar(0.0))
            {
            --grid_pos.y;
            moved = true;
            }

        if (f.z >= Scalar(1.0))
            {
            ++grid_pos.z;
            moved = true;
            }
        if (f.z < Scalar(0.0))
            {
            --grid_pos.z;
            moved = true;
            }

        if (moved)
            {
            if (grid_pos.x == (int)di.getW())
                grid_pos.x = 0;
            else if (grid_pos.x < 0)
                grid_pos.x += di.getW();

            if (grid_pos.y == (int)di.getH())
                grid_pos.y = 0;
            else if (grid_pos.y < 0)
                grid_pos.y += di.getH();

            if (grid_pos.z == (int)di.getD())
                grid_pos.z = 0;
            else if (grid_pos.z < 0)
                grid_pos.z += di.getD();

            unsigned int cur_rank = h_cart_ranks.data[di(grid_pos.x, grid_pos.y, grid_pos.z)];
            cnts[cur_rank]++;
            }
        }
    }

/*!
 * Each rank calls countParticlesOffRank() to count the number of particles to send to other ranks.
 * Neighboring ranks then perform send/receive calls, and count the new number of particles they own
 * as the number they owned locally plus the number received minus the number sent.
 *
 * \note All ranks must participate in this call since it involves send/receive operations between
 * neighboring domains.
 */
void LoadBalancer::computeOwnedParticles()
    {
    // don't do anything if nobody has signaled a change
    if (!m_needs_recount)
        return;

    // count the particles that are off the rank
    ArrayHandle<unsigned int> h_unique_neigh(m_comm->getUniqueNeighbors(),
                                             access_location::host,
                                             access_mode::read);

    // fill the map initially to zeros (not necessary since should be auto-initialized to zero, but
    // just playing it safe)
    std::map<unsigned int, unsigned int> cnts;
    for (unsigned int i = 0; i < m_comm->getNUniqueNeighbors(); ++i)
        {
        cnts[h_unique_neigh.data[i]] = 0;
        }
    countParticlesOffRank(cnts);

    MPI_Request req[2 * m_comm->getNUniqueNeighbors()];
    MPI_Status stat[2 * m_comm->getNUniqueNeighbors()];
    unsigned int nreq = 0;

    unsigned int n_send_ptls[m_comm->getNUniqueNeighbors()];
    unsigned int n_recv_ptls[m_comm->getNUniqueNeighbors()];
    for (unsigned int cur_neigh = 0; cur_neigh < m_comm->getNUniqueNeighbors(); ++cur_neigh)
        {
        unsigned int neigh_rank = h_unique_neigh.data[cur_neigh];
        n_send_ptls[cur_neigh] = cnts[neigh_rank];

        MPI_Isend(&n_send_ptls[cur_neigh],
                  1,
                  MPI_UNSIGNED,
                  neigh_rank,
                  0,
                  m_mpi_comm,
                  &req[nreq++]);
        MPI_Irecv(&n_recv_ptls[cur_neigh],
                  1,
                  MPI_UNSIGNED,
                  neigh_rank,
                  0,
                  m_mpi_comm,
                  &req[nreq++]);
        }
    MPI_Waitall(nreq, req, stat);

    // reduce the particles sent to me
    int N_own = m_pdata->getN();
    for (unsigned int cur_neigh = 0; cur_neigh < m_comm->getNUniqueNeighbors(); ++cur_neigh)
        {
        N_own += n_recv_ptls[cur_neigh];
        N_own -= n_send_ptls[cur_neigh];
        }

    // set the count
    resetNOwn(N_own);
    }

#endif // ENABLE_MPI

/*!
 * Zero the counters.
 */
void LoadBalancer::resetStats()
    {
    m_n_calls = m_n_iterations = m_n_rebalances = 0;
    m_total_max_imbalance = 0.0;
    m_max_max_imbalance = Scalar(1.0);
    }

namespace detail
    {
void export_LoadBalancer(pybind11::module& m)
    {
    pybind11::class_<LoadBalancer, Tuner, std::shared_ptr<LoadBalancer>>(m, "LoadBalancer")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>, std::shared_ptr<Trigger>>())
        .def_property("tolerance", &LoadBalancer::getTolerance, &LoadBalancer::setTolerance)
        .def_property("max_iterations",
                      &LoadBalancer::getMaxIterations,
                      &LoadBalancer::setMaxIterations)
        .def_property("x", &LoadBalancer::getEnableX, &LoadBalancer::setEnableX)
        .def_property("y", &LoadBalancer::getEnableY, &LoadBalancer::setEnableY)
        .def_property("z", &LoadBalancer::getEnableZ, &LoadBalancer::setEnableZ);
    }

    } // end namespace detail

    } // end namespace hoomd
