// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file DomainDecomposition.cc
    \brief Implements the DomainDecomposition class
*/

#ifdef ENABLE_MPI
#include "DomainDecomposition.h"

#include "ParticleData.h"
#include "SystemDefinition.h"

#include "HOOMDMPI.h"
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <numeric>

using namespace std;

namespace hoomd
    {
//! Constructor
/*! The constructor performs a spatial domain decomposition of the simulation box of processor with
 * rank \b exec_conf->getMPIroot(). The domain dimensions are distributed on the other processors.
 */
DomainDecomposition::DomainDecomposition(std::shared_ptr<ExecutionConfiguration> exec_conf,
                                         Scalar3 L,
                                         unsigned int nx,
                                         unsigned int ny,
                                         unsigned int nz,
                                         bool twolevel)
    : m_exec_conf(exec_conf), m_mpi_comm(m_exec_conf->getMPICommunicator())
    {
    m_exec_conf->msg->notice(5) << "Constructing DomainDecomposition" << endl;

    // setup the decomposition grid
    initializeDomainGrid(L, nx, ny, nz, twolevel);

    // default initialization is to uniform slices
    std::vector<Scalar> cur_fxs(m_nx - 1, Scalar(1.0) / Scalar(m_nx));
    std::vector<Scalar> cur_fys(m_ny - 1, Scalar(1.0) / Scalar(m_ny));
    std::vector<Scalar> cur_fzs(m_nz - 1, Scalar(1.0) / Scalar(m_nz));
    initializeCumulativeFractions(cur_fxs, cur_fys, cur_fzs);
    }

/*!
 * \param exec_conf The execution configuration
 * \param L Box lengths of global box to sub-divide
 * \param fxs Array of fractions to decompose box in x for first nx-1 processors
 * \param fys Array of fractions to decompose box in y for first ny-1 processors
 * \param fzs Array of fractions to decompose box in z for first nz-1 processors
 *
 * If a fraction is not specified, a default value is chosen with uniform spacing. Note that the
 * chosen value for the number of processors is not guaranteed to be optimal.
 */
DomainDecomposition::DomainDecomposition(std::shared_ptr<ExecutionConfiguration> exec_conf,
                                         Scalar3 L,
                                         const std::vector<Scalar>& fxs,
                                         const std::vector<Scalar>& fys,
                                         const std::vector<Scalar>& fzs)
    : m_exec_conf(exec_conf), m_mpi_comm(m_exec_conf->getMPICommunicator())
    {
    m_exec_conf->msg->notice(5) << "Constructing DomainDecomposition" << endl;

    size_t nx = (fxs.size() > 0) ? (fxs.size() + 1) : 0;
    size_t ny = (fys.size() > 0) ? (fys.size() + 1) : 0;
    size_t nz = (fzs.size() > 0) ? (fzs.size() + 1) : 0;
    initializeDomainGrid(L, (unsigned int)nx, (unsigned int)ny, (unsigned int)nz, false);

    std::vector<Scalar> try_fxs = fxs;
    std::vector<Scalar> try_fys = fys;
    std::vector<Scalar> try_fzs = fzs;
    if ((nx > 0 && try_fxs.size() != m_nx - 1) || (ny > 0 && try_fys.size() != m_ny - 1)
        || (nz > 0 && try_fzs.size() != m_nz - 1))
        {
        std::ostringstream o;
        o << "Domain decomposition grid does not match specification:" << "(" << m_nx << "," << m_ny
          << "," << m_nz << ") is not (" << try_fxs.size() + 1 << "," << try_fys.size() + 1 << ","
          << try_fzs.size() + 1 << ")";
        throw std::invalid_argument(o.str());
        }

    // if domain fractions weren't set, fill the fractions uniformly
    if (m_nx > 1 && try_fxs.empty())
        {
        try_fxs = std::vector<Scalar>(m_nx - 1, Scalar(1.0) / Scalar(m_nx));
        }
    if (m_ny > 1 && try_fys.empty())
        {
        try_fys = std::vector<Scalar>(m_ny - 1, Scalar(1.0) / Scalar(m_ny));
        }
    if (m_nz > 1 && try_fzs.empty())
        {
        try_fzs = std::vector<Scalar>(m_nz - 1, Scalar(1.0) / Scalar(m_nz));
        }

    initializeCumulativeFractions(try_fxs, try_fys, try_fzs);
    }

/*!
 * \param L Box lengths of global box to sub-divide
 * \param nx Requested number of domains along the x direction (0 == choose default)
 * \param ny Requested number of domains along the y direction (0 == choose default)
 * \param nz Requested number of domains along the z direction (0 == choose default)
 * \param twolevel If true, attempt two level decomposition (default == false)
 */
void DomainDecomposition::initializeDomainGrid(Scalar3 L,
                                               unsigned int nx,
                                               unsigned int ny,
                                               unsigned int nz,
                                               bool twolevel)
    {
    unsigned int rank = m_exec_conf->getRank();
    unsigned int nranks = m_exec_conf->getNRanks();

    // initialize node names
    findCommonNodes();

    m_max_n_node = 0;
    m_twolevel = twolevel;

    if (twolevel)
        {
        // find out if we can do a node-level decomposition
        initializeTwoLevel();
        }

    unsigned int nx_node = 0, ny_node = 0, nz_node = 0;
    unsigned int nx_intra = 0, ny_intra = 0, nz_intra = 0;

    if (nx || ny || nz)
        m_twolevel = false;

    if (rank == 0)
        {
        if (m_twolevel)
            {
            // every node has the same number of ranks, so nranks == num_nodes * num_ranks_per_node
            unsigned int n_nodes = (unsigned int)(m_nodes.size());

            // subdivide the global grid
            findDecomposition(nranks, L, nx, ny, nz);

            // subdivide the local grid
            subdivide(nranks / n_nodes, L, nx, ny, nz, nx_intra, ny_intra, nz_intra);

            nx_node = nx / nx_intra;
            ny_node = ny / ny_intra;
            nz_node = nz / nz_intra;
            }
        else
            {
            bool found_decomposition = findDecomposition(nranks, L, nx, ny, nz);
            if (!found_decomposition)
                {
                throw std::invalid_argument(
                    "Unable to find a decomposition with the requested dimensions.");
                }
            }
        m_nx = nx;
        m_ny = ny;
        m_nz = nz;
        }

    // broadcast grid dimensions
    bcast(m_nx, 0, m_mpi_comm);
    bcast(m_ny, 0, m_mpi_comm);
    bcast(m_nz, 0, m_mpi_comm);

    // Initialize domain indexer
    m_index = Index3D(m_nx, m_ny, m_nz);

    // map cartesian grid onto ranks
    GlobalArray<unsigned int> cart_ranks(nranks, m_exec_conf);
    m_cart_ranks.swap(cart_ranks);

    GlobalArray<unsigned int> cart_ranks_inv(nranks, m_exec_conf);
    m_cart_ranks_inv.swap(cart_ranks_inv);

    ArrayHandle<unsigned int> h_cart_ranks(m_cart_ranks,
                                           access_location::host,
                                           access_mode::overwrite);
    ArrayHandle<unsigned int> h_cart_ranks_inv(m_cart_ranks_inv,
                                               access_location::host,
                                               access_mode::overwrite);

    if (m_twolevel)
        {
        bcast(nx_intra, 0, m_mpi_comm);
        bcast(ny_intra, 0, m_mpi_comm);
        bcast(nz_intra, 0, m_mpi_comm);

        bcast(nx_node, 0, m_mpi_comm);
        bcast(ny_node, 0, m_mpi_comm);
        bcast(nz_node, 0, m_mpi_comm);

        m_node_grid = Index3D(nx_node, ny_node, nz_node);
        m_intra_node_grid = Index3D(nx_intra, ny_intra, nz_intra);

        std::vector<unsigned int> node_ranks(m_max_n_node);
        std::set<std::string>::iterator node_it = m_nodes.begin();

        std::set<unsigned int> node_rank_set;

        // iterate over node grid
        for (unsigned int ix_node = 0; ix_node < m_node_grid.getW(); ix_node++)
            for (unsigned int iy_node = 0; iy_node < m_node_grid.getH(); iy_node++)
                for (unsigned int iz_node = 0; iz_node < m_node_grid.getD(); iz_node++)
                    {
                    // get ranks for this node
                    typedef std::multimap<std::string, unsigned int> map_t;
                    std::string node = *(node_it++);
                    std::pair<map_t::iterator, map_t::iterator> p = m_node_map.equal_range(node);

                    // Insert ranks per node into an ordered set (multimap doesn't guarantee order
                    // and thus order can be different on different ranks, especially after
                    // deserialization)
                    node_rank_set.clear();
                    for (map_t::iterator it = p.first; it != p.second; ++it)
                        {
                        node_rank_set.insert(it->second);
                        }

                    std::set<unsigned int>::iterator set_it;

                    std::ostringstream oss;
                    oss << "Node " << node << ": ranks";
                    unsigned int i = 0;
                    for (set_it = node_rank_set.begin(); set_it != node_rank_set.end(); ++set_it)
                        {
                        unsigned int r = *set_it;
                        oss << " " << r;
                        node_ranks[i++] = r;
                        }
                    m_exec_conf->msg->notice(5) << oss.str() << std::endl;

                    // iterate over local ranks
                    for (unsigned int ix_intra = 0; ix_intra < m_intra_node_grid.getW(); ix_intra++)
                        for (unsigned int iy_intra = 0; iy_intra < m_intra_node_grid.getH();
                             iy_intra++)
                            for (unsigned int iz_intra = 0; iz_intra < m_intra_node_grid.getD();
                                 iz_intra++)
                                {
                                unsigned int ilocal
                                    = m_intra_node_grid(ix_intra, iy_intra, iz_intra);

                                unsigned int ix = ix_node * nx_intra + ix_intra;
                                unsigned int iy = iy_node * ny_intra + iy_intra;
                                unsigned int iz = iz_node * nz_intra + iz_intra;

                                unsigned int iglob = m_index(ix, iy, iz);

                                // add rank to table
                                h_cart_ranks.data[iglob] = node_ranks[ilocal];
                                h_cart_ranks_inv.data[node_ranks[ilocal]] = iglob;
                                }
                    }
        } // twolevel
    else
        {
        // simply map the global grid in sequential order to ranks
        for (unsigned int iglob = 0; iglob < nranks; ++iglob)
            {
            h_cart_ranks.data[iglob] = iglob;
            h_cart_ranks_inv.data[iglob] = iglob;
            }
        }

    // Print out information about the domain decomposition
    m_exec_conf->msg->notice(2) << "Using domain decomposition: n_x = " << m_nx << " n_y = " << m_ny
                                << " n_z = " << m_nz << "." << std::endl;

    if (m_twolevel)
        m_exec_conf->msg->notice(2) << nx_intra << " x " << ny_intra << " x " << nz_intra
                                    << " local grid on " << m_nodes.size() << " nodes" << std::endl;

    // compute position of this box in the domain grid by reverse look-up
    m_grid_pos = m_index.getTriple(h_cart_ranks_inv.data[rank]);
    }

/*!
 * \param fxs Array of fractions to decompose box in x for first nx-1 processors
 * \param fys Array of fractions to decompose box in y for first ny-1 processors
 * \param fzs Array of fractions to decompose box in z for first nz-1 processors
 *
 * The constructor that calls this method should ensure that the array sizes match the domain
 * decomposition grid size. A partial sum is performed on the fractions to fill up the cumulative
 * fraction arrays, which begin with 0 and end with 1 always.
 */
void DomainDecomposition::initializeCumulativeFractions(const std::vector<Scalar>& fxs,
                                                        const std::vector<Scalar>& fys,
                                                        const std::vector<Scalar>& fzs)
    {
    // final sanity check (constructor should handle this correctly)
    assert(fxs.size() + 1 == m_nx);
    assert(fys.size() + 1 == m_ny);
    assert(fzs.size() + 1 == m_nz);

    // adjust the fraction arrays
    m_cumulative_frac_x.resize(m_nx + 1);
    m_cumulative_frac_y.resize(m_ny + 1);
    m_cumulative_frac_z.resize(m_nz + 1);

    // fill the beginning and end points
    m_cumulative_frac_x[0] = Scalar(0.0);
    m_cumulative_frac_x[m_nx] = Scalar(1.0);
    m_cumulative_frac_y[0] = Scalar(0.0);
    m_cumulative_frac_y[m_ny] = Scalar(1.0);
    m_cumulative_frac_z[0] = Scalar(0.0);
    m_cumulative_frac_z[m_nz] = Scalar(1.0);

    // fill in the cumulative fractions by partial summation
    std::partial_sum(fxs.begin(), fxs.end(), m_cumulative_frac_x.begin() + 1);
    std::partial_sum(fys.begin(), fys.end(), m_cumulative_frac_y.begin() + 1);
    std::partial_sum(fzs.begin(), fzs.end(), m_cumulative_frac_z.begin() + 1);

    // broadcast the floating point result from rank 0 to all ranks
    MPI_Bcast(&m_cumulative_frac_x[0], m_nx + 1, MPI_HOOMD_SCALAR, 0, m_mpi_comm);
    MPI_Bcast(&m_cumulative_frac_y[0], m_ny + 1, MPI_HOOMD_SCALAR, 0, m_mpi_comm);
    MPI_Bcast(&m_cumulative_frac_z[0], m_nz + 1, MPI_HOOMD_SCALAR, 0, m_mpi_comm);
    }

//! Find a domain decomposition with given parameters
bool DomainDecomposition::findDecomposition(unsigned int nranks,
                                            const Scalar3 L,
                                            unsigned int& nx,
                                            unsigned int& ny,
                                            unsigned int& nz)
    {
    assert(L.x > 0);
    assert(L.y > 0);

    // Calculate the number of sub-domains in every direction
    // by minimizing the surface area between domains at constant number of domains
    bool is2D = L.z == 0.0;
    double min_surface_area; // surface area in 3D, perimeter length in 2D
    if (is2D)
        {
        min_surface_area = L.x * (double)(nranks - 1);
        }
    else
        {
        min_surface_area = L.x * L.z * (double)(nranks - 1);
        }

    unsigned int nx_in = nx;
    unsigned int ny_in = ny;
    unsigned int nz_in = nz;

    bool found_decomposition = (nx_in == 0 && ny_in == 0 && nz_in == 0);

    // initial guess
    nx = 1;
    if (is2D)
        {
        ny = nranks;
        nz = 1;
        }
    else
        {
        ny = 1;
        nz = nranks;
        }

    for (unsigned int nx_try = 1; nx_try <= nranks; nx_try++)
        {
        if (nx_in != 0 && nx_try != nx_in)
            continue;
        for (unsigned int ny_try = 1; nx_try * ny_try <= nranks; ny_try++)
            {
            if (ny_in != 0 && ny_try != ny_in)
                continue;
            for (unsigned int nz_try = 1; nx_try * ny_try * nz_try <= nranks; nz_try++)
                {
                if (nz_in != 0 && nz_try != nz_in)
                    continue;
                if (nx_try * ny_try * nz_try != nranks)
                    continue;
                if (is2D && nz_try > 1)
                    continue;
                double surface_area;
                if (is2D)
                    {
                    surface_area = L.x * (ny_try - 1) + L.y * (nx_try - 1);
                    }
                else
                    {
                    surface_area = L.x * L.y * (double)(nz_try - 1)
                                   + L.x * L.z * (double)(ny_try - 1)
                                   + L.y * L.z * (double)(nx_try - 1);
                    }
                if (surface_area < min_surface_area || !found_decomposition)
                    {
                    nx = nx_try;
                    ny = ny_try;
                    nz = nz_try;
                    min_surface_area = surface_area;
                    found_decomposition = true;
                    }
                }
            }
        }

    return found_decomposition;
    }

//! Find a two-level decomposition of the global grid
void DomainDecomposition::subdivide(unsigned int n_node_ranks,
                                    Scalar3 L,
                                    unsigned int nx,
                                    unsigned int ny,
                                    unsigned int nz,
                                    unsigned int& nx_intra,
                                    unsigned int& ny_intra,
                                    unsigned int& nz_intra)
    {
    assert(L.x > 0);
    assert(L.y > 0);
    assert(L.z > 0);

    // initial guess
    nx_intra = 1;
    ny_intra = 1;
    nz_intra = n_node_ranks;

    for (unsigned int nx_intra_try = 1; nx_intra_try <= n_node_ranks; nx_intra_try++)
        for (unsigned int ny_intra_try = 1; nx_intra_try * ny_intra_try <= n_node_ranks;
             ny_intra_try++)
            for (unsigned int nz_intra_try = 1;
                 nx_intra_try * ny_intra_try * nz_intra_try <= n_node_ranks;
                 nz_intra_try++)
                {
                if (nx_intra_try * ny_intra_try * nz_intra_try != n_node_ranks)
                    continue;
                if (nx % nx_intra_try || ny % ny_intra_try || nz % nz_intra_try)
                    continue;

                nx_intra = nx_intra_try;
                ny_intra = ny_intra_try;
                nz_intra = nz_intra_try;
                }
    }

/*! \param dir Spatial direction to find neighbor in
               0: east, 1: west, 2: north, 3: south, 4: up, 5: down
 */
unsigned int DomainDecomposition::getNeighborRank(unsigned int dir) const
    {
    assert(0 <= dir && dir < 6);

    int adj[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};

    // determine neighbor position
    int ineigh = (int)m_grid_pos.x + adj[dir][0];
    int jneigh = (int)m_grid_pos.y + adj[dir][1];
    int kneigh = (int)m_grid_pos.z + adj[dir][2];

    // wrap across boundaries
    if (ineigh < 0)
        ineigh += m_nx;
    else if (ineigh == (int)m_nx)
        ineigh -= m_nx;

    if (jneigh < 0)
        jneigh += m_ny;
    else if (jneigh == (int)m_ny)
        jneigh -= m_ny;

    if (kneigh < 0)
        kneigh += m_nz;
    else if (kneigh == (int)m_nz)
        kneigh -= m_nz;

    unsigned int idx = m_index(ineigh, jneigh, kneigh);
    ArrayHandle<unsigned int> h_cart_ranks(m_cart_ranks, access_location::host, access_mode::read);
    return h_cart_ranks.data[idx];
    }

//! Determines whether the local box shares a boundary with the global box
bool DomainDecomposition::isAtBoundary(unsigned int dir) const
    {
    return ((dir == 0 && m_grid_pos.x == m_nx - 1) || (dir == 1 && m_grid_pos.x == 0)
            || (dir == 2 && m_grid_pos.y == m_ny - 1) || (dir == 3 && m_grid_pos.y == 0)
            || (dir == 4 && m_grid_pos.z == m_nz - 1) || (dir == 5 && m_grid_pos.z == 0));
    }

/*!
 * \param dir Direction (0=x, 1=y, 2=z) to set fractions
 * \param cum_frac Vector of cumulative fractions, beginning with 0 and ending with 1
 * \param root Rank to broadcast the set fractions from
 *
 * \note Setting the cumulative fractions is a collective call requiring all ranks to participate in
 * order to keep the decomposition properly synchronized between ranks.
 */
void DomainDecomposition::setCumulativeFractions(unsigned int dir,
                                                 const std::vector<Scalar>& cum_frac,
                                                 unsigned int root)
    {
    if (dir > 2)
        {
        throw std::invalid_argument("Requested direction does not exist.");
        }

    bool changed = false;
    if (m_exec_conf->getRank() == root)
        {
        if (dir == 0 && cum_frac.size() == m_cumulative_frac_x.size())
            {
            m_cumulative_frac_x = cum_frac;
            changed = true;
            }
        else if (dir == 1 && cum_frac.size() == m_cumulative_frac_y.size())
            {
            m_cumulative_frac_y = cum_frac;
            changed = true;
            }
        else if (dir == 2 && cum_frac.size() == m_cumulative_frac_z.size())
            {
            m_cumulative_frac_z = cum_frac;
            changed = true;
            }
        }

    // sync the update from the root to all ranks
    bcast(changed, root, m_mpi_comm);
    if (changed)
        {
        if (dir == 0)
            {
            MPI_Bcast(&m_cumulative_frac_x[0], m_nx + 1, MPI_HOOMD_SCALAR, root, m_mpi_comm);

            if (m_cumulative_frac_x.front() != Scalar(0.0)
                || m_cumulative_frac_x.back() != Scalar(1.0))
                {
                throw std::invalid_argument("Specified fractions are invalid.");
                }
            }
        else if (dir == 1)
            {
            MPI_Bcast(&m_cumulative_frac_y[0], m_ny + 1, MPI_HOOMD_SCALAR, root, m_mpi_comm);

            if (m_cumulative_frac_y.front() != Scalar(0.0)
                || m_cumulative_frac_y.back() != Scalar(1.0))
                {
                throw std::invalid_argument("Specified fractions are invalid.");
                }
            }
        else if (dir == 2)
            {
            MPI_Bcast(&m_cumulative_frac_z[0], m_nz + 1, MPI_HOOMD_SCALAR, root, m_mpi_comm);

            if (m_cumulative_frac_z.front() != Scalar(0.0)
                || m_cumulative_frac_z.back() != Scalar(1.0))
                {
                throw std::invalid_argument("Specified fractions are invalid.");
                }
            }
        }
    else // if no change, it's because things don't match up
        {
        throw std::invalid_argument(
            "Domain decomposition cannot change topology after construction.");
        }
    }

/*!
 * \param global_box The global simulation box
 * \returns The local simulation box for the current rank
 */
const BoxDim DomainDecomposition::calculateLocalBox(const BoxDim& global_box)
    {
    // initialize local box with all properties of global box
    BoxDim box = global_box;
    Scalar3 L = global_box.getL();

    // position of this domain in the grid
    Scalar3 lo_cumulative_frac = make_scalar3(m_cumulative_frac_x[m_grid_pos.x],
                                              m_cumulative_frac_y[m_grid_pos.y],
                                              m_cumulative_frac_z[m_grid_pos.z]);
    Scalar3 lo = global_box.getLo() + lo_cumulative_frac * L;

    Scalar3 hi_cumulative_frac = make_scalar3(m_cumulative_frac_x[m_grid_pos.x + 1],
                                              m_cumulative_frac_y[m_grid_pos.y + 1],
                                              m_cumulative_frac_z[m_grid_pos.z + 1]);
    Scalar3 hi = global_box.getLo() + hi_cumulative_frac * L;

    // set periodic flags
    // we are periodic in a direction along which there is only one box
    uchar3 periodic = make_uchar3(m_nx == 1 ? 1 : 0, m_ny == 1 ? 1 : 0, m_nz == 1 ? 1 : 0);

    box.setLoHi(lo, hi);
    box.setPeriodic(periodic);
    return box;
    }

/*!
 * \param global_box Global simulation box
 * \param pos Particle position
 * \returns the rank of the processor that should receive the particle
 */
unsigned int DomainDecomposition::placeParticle(const BoxDim& global_box,
                                                Scalar3 pos,
                                                const unsigned int* cart_ranks)
    {
    // get fractional coordinates in the global box
    Scalar3 f = global_box.makeFraction(pos);

    Scalar tol(1e-5);
    // check user input
    if (f.x < -tol || f.x >= 1.0 + tol || f.y < -tol || f.y >= 1.0 + tol || f.z < -tol
        || f.z >= 1.0 + tol)
        {
        std::ostringstream o;
        o << "Particle coordinates outside global box." << std::endl;
        o << "Cartesian coordinates: " << std::endl;
        o << "x: " << pos.x << " y: " << pos.y << " z: " << pos.z << std::endl;
        o << "Fractional coordinates: " << std::endl;
        o << "f.x: " << f.x << " f.y: " << f.y << " f.z: " << f.z << std::endl;
        Scalar3 lo = global_box.getLo();
        Scalar3 hi = global_box.getHi();
        o << "Global box lo: (" << lo.x << ", " << lo.y << ", " << lo.z << ")" << std::endl;
        o << "           hi: (" << hi.x << ", " << hi.y << ", " << hi.z << ")" << std::endl;

        throw std::runtime_error(o.str());
        }

    // compute the box the particle should be placed into
    // use the lower_bound (the first element that does not compare last < the search term)
    // then, the domain to place into is it-1 (since we want to place into the one that it actually
    // belongs to)

    // it is OK for particles to be slightly outside the box, they will get migrated to their
    // correct boxes, as long as we don't wrap them around. Therefore, shift back into nearest box
    // if that is the case
    std::vector<Scalar>::iterator it;
    it = std::lower_bound(m_cumulative_frac_x.begin(), m_cumulative_frac_x.end(), f.x);
    int ix = int(it - 1 - m_cumulative_frac_x.begin());
    if (ix < 0)
        ix++;
    else if (ix >= (int)m_nx)
        ix--;

    it = std::lower_bound(m_cumulative_frac_y.begin(), m_cumulative_frac_y.end(), f.y);
    int iy = int(it - 1 - m_cumulative_frac_y.begin());
    if (iy < 0)
        iy++;
    else if (iy >= (int)m_ny)
        iy--;

    it = std::lower_bound(m_cumulative_frac_z.begin(), m_cumulative_frac_z.end(), f.z);
    int iz = int(it - 1 - m_cumulative_frac_z.begin());
    if (iz < 0)
        iz++;
    else if (iz >= (int)m_nz)
        iz--;

    unsigned int rank = cart_ranks[m_index(ix, iy, iz)];

    return rank;
    }

void DomainDecomposition::findCommonNodes()
    {
    // get MPI node name
    char procname[MPI_MAX_PROCESSOR_NAME];
    int len;
    MPI_Get_processor_name(procname, &len);
    std::string s(procname, len);

    // collect node names from all ranks on rank zero
    std::vector<std::string> nodes;
    gather_v(s, nodes, 0, m_exec_conf->getMPICommunicator());

    // construct map of node names
    if (m_exec_conf->getRank() == 0)
        {
        unsigned int nranks = m_exec_conf->getNRanks();
        for (unsigned int r = 0; r < nranks; r++)
            {
            // insert into set
            m_nodes.insert(nodes[r]);

            // insert into map
            m_node_map.insert(std::make_pair(nodes[r], r));
            }
        }

    // broadcast to other ranks
    bcast(m_nodes, 0, m_exec_conf->getMPICommunicator());
    bcast(m_node_map, 0, m_exec_conf->getMPICommunicator());
    }

void DomainDecomposition::initializeTwoLevel()
    {
    typedef std::multimap<std::string, unsigned int> map_t;
    m_twolevel = true;
    m_max_n_node = 0;
    for (std::set<std::string>::iterator it = m_nodes.begin(); it != m_nodes.end(); ++it)
        {
        std::pair<map_t::iterator, map_t::iterator> p = m_node_map.equal_range(*it);
        unsigned int n_node = (unsigned int)(std::distance(p.first, p.second));

        // if we have a non-uniform number of ranks per node use one-level decomposition
        if (m_max_n_node != 0 && n_node != m_max_n_node)
            m_twolevel = false;

        if (n_node > m_max_n_node)
            m_max_n_node = n_node;
        }
    }

namespace detail
    {
//! Export DomainDecomposition class to python
void export_DomainDecomposition(pybind11::module& m)
    {
    pybind11::class_<DomainDecomposition, std::shared_ptr<DomainDecomposition>>(
        m,
        "DomainDecomposition")
        .def(pybind11::init<std::shared_ptr<ExecutionConfiguration>,
                            Scalar3,
                            unsigned int,
                            unsigned int,
                            unsigned int,
                            bool>())
        .def(pybind11::init<std::shared_ptr<ExecutionConfiguration>,
                            Scalar3,
                            const std::vector<Scalar>&,
                            const std::vector<Scalar>&,
                            const std::vector<Scalar>&>())
        .def("getCumulativeFractions", &DomainDecomposition::getCumulativeFractions);
    }
    } // end namespace detail

    } // end namespace hoomd
#endif // ENABLE_MPI
