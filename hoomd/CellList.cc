// Copyright (c) 2009-2024 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

/*! \file CellList.cc
    \brief Defines CellList
*/

#include "CellList.h"
#include "Communicator.h"

#include <algorithm>

using namespace std;

namespace hoomd
    {
/*! \param sysdef system to compute the cell list of
 */
CellList::CellList(std::shared_ptr<SystemDefinition> sysdef)
    : Compute(sysdef), m_nominal_width(Scalar(1.0)), m_radius(1), m_compute_xyzf(true),
      m_compute_type_body(false), m_compute_orientation(false), m_compute_idx(false),
      m_flag_charge(false), m_flag_type(false), m_sort_cell_list(false), m_compute_adj_list(true)
    {
    m_exec_conf->msg->notice(5) << "Constructing CellList" << endl;

    // allocation is deferred until the first compute() call - initialize values to dummy variables
    m_dim = make_uint3(0, 0, 0);
    m_Nmax = 0;
    m_params_changed = true;
    m_particles_sorted = false;
    m_box_changed = false;
    m_multiple = 1;

    GlobalArray<uint3> conditions(1, m_exec_conf);
    std::swap(m_conditions, conditions);

        {
        // reset conditions
        ArrayHandle<uint3> h_conditions(m_conditions,
                                        access_location::host,
                                        access_mode::overwrite);
        *h_conditions.data = make_uint3(0, 0, 0);
        }

    m_actual_width = make_scalar3(0.0, 0.0, 0.0);
    m_ghost_width = make_scalar3(0.0, 0.0, 0.0);

    m_pdata->getParticleSortSignal().connect<CellList, &CellList::slotParticlesSorted>(this);
    m_pdata->getBoxChangeSignal().connect<CellList, &CellList::slotBoxChanged>(this);

#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        auto comm_weak = m_sysdef->getCommunicator();
        assert(comm_weak.lock());
        m_comm = comm_weak.lock();
        }
#endif
    }

CellList::~CellList()
    {
    m_exec_conf->msg->notice(5) << "Destroying CellList" << endl;
    m_pdata->getParticleSortSignal().disconnect<CellList, &CellList::slotParticlesSorted>(this);
    m_pdata->getBoxChangeSignal().disconnect<CellList, &CellList::slotBoxChanged>(this);
    }

//! Round down to the nearest multiple
/*! \param v Value to round
    \param m Multiple
    \returns \a v if it is a multiple of \a m, otherwise, \a v rounded down to the nearest multiple
   of \a m.
*/
static unsigned int roundDown(unsigned int v, unsigned int m)
    {
    // use integer floor division
    unsigned int d = v / m;
    return d * m;
    }

/*! \returns Cell dimensions that match with the current width, and box dimension
 */
uint3 CellList::computeDimensions()
    {
    uint3 dim;

    // calculate the bin dimensions
    const BoxDim& box = m_pdata->getBox();

    Scalar3 L = box.getNearestPlaneDistance();

    // size the ghost layer width
    m_ghost_width = make_scalar3(0.0, 0.0, 0.0);
#ifdef ENABLE_MPI
    if (m_sysdef->isDomainDecomposed())
        {
        Scalar ghost_width = m_comm->getGhostLayerMaxWidth();
        if (ghost_width > Scalar(0.0))
            {
            if (!box.getPeriodic().x)
                m_ghost_width.x = ghost_width;

            if (!box.getPeriodic().y)
                m_ghost_width.y = ghost_width;

            if (m_sysdef->getNDimensions() == 3 && !box.getPeriodic().z)
                m_ghost_width.z = ghost_width;
            }
        }
#endif

    dim.x
        = roundDown((unsigned int)((L.x + 2.0 * m_ghost_width.x) / (m_nominal_width)), m_multiple);
    dim.y
        = roundDown((unsigned int)((L.y + 2.0 * m_ghost_width.y) / (m_nominal_width)), m_multiple);
    dim.z = (m_sysdef->getNDimensions() == 3)
                ? roundDown((unsigned int)((L.z + 2.0 * m_ghost_width.z) / (m_nominal_width)),
                            m_multiple)
                : 1;

    // In extremely small boxes, the calculated dimensions could go to zero, but need at least one
    // cell in each dimension for particles to be in a cell and to pass the checkCondition tests.
    if (dim.x == 0)
        dim.x = 1;
    if (dim.y == 0)
        dim.y = 1;
    if (dim.z == 0)
        dim.z = 1;

    return dim;
    }

void CellList::compute(uint64_t timestep)
    {
    Compute::compute(timestep);
    bool force = false;

    m_exec_conf->msg->notice(10) << "Cell list compute" << endl;

    if (m_nominal_width == 0)
        {
        throw std::runtime_error("Cell: cell width must be non-zero");
        }

    if (m_params_changed)
        {
        m_exec_conf->msg->notice(10) << "Cell list params changed" << endl;
        // need to fully reinitialize on any parameter change
        initializeAll();
        m_params_changed = false;
        force = true;
        }

    if (m_box_changed)
        {
        uint3 new_dim = computeDimensions();
        m_exec_conf->msg->notice(10)
            << "Cell list box changed " << m_dim.x << " x " << m_dim.y << " x " << m_dim.z << " -> "
            << new_dim.x << " x " << new_dim.y << " x " << new_dim.z << " -> " << endl;
        if (new_dim.x == m_dim.x && new_dim.y == m_dim.y && new_dim.z == m_dim.z)
            {
            // number of bins has not changed, only need to update width
            initializeWidth();
            }
        else
            {
            // number of bins has changed, need to fully reinitialize memory
            initializeAll();
            }

        m_box_changed = false;
        force = true;
        }

    if (m_particles_sorted)
        {
        // sorted particles simply need a forced update to get the proper indices in the data
        // structure
        m_particles_sorted = false;
        force = true;
        }

    // only update if we need to
    if (shouldCompute(timestep) || force)
        {
        bool overflowed = false;
        do
            {
            computeCellList();

            overflowed = checkConditions();
            // if we overflowed, need to reallocate memory and reset the conditions
            if (overflowed)
                {
                initializeAll();
                resetConditions();
                }
            } while (overflowed);
        }
    }

void CellList::initializeAll()
    {
    initializeWidth();
    initializeMemory();
    }

void CellList::initializeWidth()
    {
    m_exec_conf->msg->notice(10) << "Cell list initialize width" << endl;

    // get the local box
    const BoxDim& box = m_pdata->getBox();

    // initialize dimensions and width
    m_dim = computeDimensions();

    // stash the current actual cell width
    const Scalar3 L = box.getNearestPlaneDistance();
    m_actual_width = make_scalar3((L.x + Scalar(2.0) * m_ghost_width.x) / Scalar(m_dim.x),
                                  (L.y + Scalar(2.0) * m_ghost_width.y) / Scalar(m_dim.y),
                                  (L.z + Scalar(2.0) * m_ghost_width.z) / Scalar(m_dim.z));

    // signal that the width has changed
    m_width_change.emit();
    }

void CellList::initializeMemory()
    {
    m_exec_conf->msg->notice(10) << "Cell list initialize memory" << endl;

    // if it is still set at 0, estimate Nmax
    if (m_Nmax == 0)
        {
        unsigned int estim_Nmax
            = (unsigned int)(ceil(double((m_pdata->getN() + m_pdata->getNGhosts()) * 1.0
                                         / double(m_dim.x * m_dim.y * m_dim.z))));
        m_Nmax = estim_Nmax;
        if (m_Nmax == 0)
            m_Nmax = 1;
        }

    m_exec_conf->msg->notice(6) << "cell list: allocating " << m_dim.x << " x " << m_dim.y << " x "
                                << m_dim.z << " x " << m_Nmax << endl;

    // initialize indexers
    m_cell_indexer = Index3D(m_dim.x, m_dim.y, m_dim.z);
    m_cell_list_indexer = Index2D(m_Nmax, m_cell_indexer.getNumElements());

    // allocate memory
    GlobalArray<unsigned int> cell_size(m_cell_indexer.getNumElements(), m_exec_conf);
    m_cell_size.swap(cell_size);
    TAG_ALLOCATION(m_cell_size);

    if (m_compute_adj_list)
        {
        // if we have less than radius*2+1 cells in a direction, restrict to unique neighbors
        uint3 n_unique_neighbors = m_dim;
        n_unique_neighbors.x = n_unique_neighbors.x > m_radius * 2 + 1
                                   ? m_radius * 2 + 1
                                   : (unsigned int)n_unique_neighbors.x;
        n_unique_neighbors.y = n_unique_neighbors.y > m_radius * 2 + 1
                                   ? m_radius * 2 + 1
                                   : (unsigned int)n_unique_neighbors.y;
        n_unique_neighbors.z = n_unique_neighbors.z > m_radius * 2 + 1
                                   ? m_radius * 2 + 1
                                   : (unsigned int)n_unique_neighbors.z;

        unsigned int n_adj;
        if (m_sysdef->getNDimensions() == 2)
            n_adj = n_unique_neighbors.x * n_unique_neighbors.y;
        else
            n_adj = n_unique_neighbors.x * n_unique_neighbors.y * n_unique_neighbors.z;

        m_cell_adj_indexer = Index2D(n_adj, m_cell_indexer.getNumElements());

        GlobalArray<unsigned int> cell_adj(m_cell_adj_indexer.getNumElements(), m_exec_conf);
        m_cell_adj.swap(cell_adj);
        TAG_ALLOCATION(m_cell_adj);
        }
    else
        {
        m_cell_adj_indexer = Index2D();

        // array is not needed, discard it
        GlobalArray<unsigned int> cell_adj;
        m_cell_adj.swap(cell_adj);
        }

    if (m_compute_xyzf)
        {
        GlobalArray<Scalar4> xyzf(m_cell_list_indexer.getNumElements(), m_exec_conf);
        m_xyzf.swap(xyzf);
        TAG_ALLOCATION(m_xyzf);
        }
    else
        {
        GlobalArray<Scalar4> xyzf;
        m_xyzf.swap(xyzf);
        }

    if (m_compute_type_body)
        {
        GlobalArray<uint2> type_body(m_cell_list_indexer.getNumElements(), m_exec_conf);
        m_type_body.swap(type_body);
        TAG_ALLOCATION(m_type_body);
        }
    else
        {
        // array is no longer needed, discard it
        GlobalArray<uint2> type_body;
        m_type_body.swap(type_body);
        }

    if (m_compute_orientation)
        {
        GlobalArray<Scalar4> orientation(m_cell_list_indexer.getNumElements(), m_exec_conf);
        m_orientation.swap(orientation);
        TAG_ALLOCATION(m_orientation);
        }
    else
        {
        // array is no longer needed, discard it
        GlobalArray<Scalar4> orientation;
        m_orientation.swap(orientation);
        }

    if (m_compute_idx || m_sort_cell_list)
        {
        GlobalArray<unsigned int> idx(m_cell_list_indexer.getNumElements(), m_exec_conf);
        m_idx.swap(idx);
        TAG_ALLOCATION(m_idx);
        }
    else
        {
        // array is no longer needed, discard it
        GlobalArray<unsigned int> idx;
        m_idx.swap(idx);
        }

    // only initialize the adjacency list if requested
    if (m_compute_adj_list)
        initializeCellAdj();
    }

void CellList::initializeCellAdj()
    {
    ArrayHandle<unsigned int> h_cell_adj(m_cell_adj, access_location::host, access_mode::overwrite);

    // per cell temporary storage of neighbors
    std::vector<unsigned int> adj;

    // loop over all cells
    for (int k = 0; k < int(m_dim.z); k++)
        for (int j = 0; j < int(m_dim.y); j++)
            for (int i = 0; i < int(m_dim.x); i++)
                {
                unsigned int cur_cell = m_cell_indexer(i, j, k);

                adj.clear();

                // loop over neighboring cells
                // need signed integer values for performing index calculations with negative values
                int r = int(m_radius);
                int rk = r;
                if (m_sysdef->getNDimensions() == 2)
                    rk = 0;

                int mx = int(m_dim.x);
                int my = int(m_dim.y);
                int mz = int(m_dim.z);

                for (int nk = k - rk; nk <= k + rk; nk++)
                    for (int nj = j - r; nj <= j + r; nj++)
                        for (int ni = i - r; ni <= i + r; ni++)
                            {
                            int wrapi = ni % mx;
                            if (wrapi < 0)
                                wrapi += mx;

                            int wrapj = nj % my;
                            if (wrapj < 0)
                                wrapj += my;

                            int wrapk = nk % mz;
                            if (wrapk < 0)
                                wrapk += mz;

                            unsigned int neigh_cell = m_cell_indexer(wrapi, wrapj, wrapk);
                            adj.push_back(neigh_cell);
                            }

                // sort the adj list for each cell
                sort(adj.begin(), adj.end());

                // remove duplicate entries
                adj.erase(unique(adj.begin(), adj.end()), adj.end());

                // copy to adj array
                copy(adj.begin(), adj.end(), &h_cell_adj.data[m_cell_adj_indexer(0, cur_cell)]);
                }
    }

void CellList::computeCellList()
    {
    // acquire the particle data
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),
                                       access_location::host,
                                       access_mode::read);
    ArrayHandle<Scalar> h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_body(m_pdata->getBodies(),
                                     access_location::host,
                                     access_mode::read);
    const BoxDim& box = m_pdata->getBox();

    // access the cell list data arrays
    ArrayHandle<unsigned int> h_cell_size(m_cell_size,
                                          access_location::host,
                                          access_mode::overwrite);
    ArrayHandle<Scalar4> h_xyzf(m_xyzf, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_cell_orientation(m_orientation,
                                            access_location::host,
                                            access_mode::overwrite);
    ArrayHandle<unsigned int> h_cell_idx(m_idx, access_location::host, access_mode::overwrite);
    ArrayHandle<uint2> h_type_body(m_type_body, access_location::host, access_mode::overwrite);
    uint3 conditions = make_uint3(0, 0, 0);

    // shorthand copies of the indexers
    Index3D ci = m_cell_indexer;
    Index2D cli = m_cell_list_indexer;

    // clear the bin sizes to 0
    memset(h_cell_size.data, 0, sizeof(unsigned int) * m_cell_indexer.getNumElements());

    Scalar3 ghost_width = getGhostWidth();

    // get periodic flags
    uchar3 periodic = box.getPeriodic();

    // for each particle
    unsigned n_tot_particles = m_pdata->getN() + m_pdata->getNGhosts();

    for (unsigned int n = 0; n < n_tot_particles; n++)
        {
        Scalar3 p = make_scalar3(h_pos.data[n].x, h_pos.data[n].y, h_pos.data[n].z);
        if (std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z))
            {
            conditions.y = n + 1;
            continue;
            }

        // find the bin each particle belongs in
        Scalar3 f = box.makeFraction(p, ghost_width);
        int ib = (int)(f.x * m_dim.x);
        int jb = (int)(f.y * m_dim.y);
        int kb = (int)(f.z * m_dim.z);

        // check if the particle is inside the unit cell + ghost layer in all dimensions
        if ((f.x < Scalar(-0.00001) || f.x >= Scalar(1.00001))
            || (f.y < Scalar(-0.00001) || f.y >= Scalar(1.00001))
            || (f.z < Scalar(-0.00001) || f.z >= Scalar(1.00001)))
            {
            // if a ghost particle is out of bounds, silently ignore it
            if (n < m_pdata->getN())
                conditions.z = n + 1;
            continue;
            }

        // need to handle the case where the particle is exactly at the box hi
        if (ib == (int)m_dim.x && periodic.x)
            ib = 0;
        if (jb == (int)m_dim.y && periodic.y)
            jb = 0;
        if (kb == (int)m_dim.z && periodic.z)
            kb = 0;

        // sanity check
        assert((ib < (int)(m_dim.x) && jb < (int)(m_dim.y) && kb < (int)(m_dim.z))
               || n >= m_pdata->getN());

        // record its bin
        unsigned int bin = ci(ib, jb, kb);

        // all particles should be in a valid cell
        if (ib < 0 || ib >= (int)m_dim.x || jb < 0 || jb >= (int)m_dim.y || kb < 0
            || kb >= (int)m_dim.z)
            {
            // but ghost particles that are out of range should not produce an error
            if (n < m_pdata->getN())
                conditions.z = n + 1;
            continue;
            }

        // setup the flag value to store
        Scalar flag;
        if (m_flag_charge)
            flag = h_charge.data[n];
        else if (m_flag_type)
            flag = h_pos.data[n].w;
        else
            flag = __int_as_scalar(n);

        // store the bin entries
        unsigned int offset = h_cell_size.data[bin];

        if (offset < m_Nmax)
            {
            if (m_compute_xyzf)
                {
                h_xyzf.data[cli(offset, bin)]
                    = make_scalar4(h_pos.data[n].x, h_pos.data[n].y, h_pos.data[n].z, flag);
                }

            if (m_compute_type_body)
                {
                h_type_body.data[cli(offset, bin)]
                    = make_uint2(__scalar_as_int(h_pos.data[n].w), h_body.data[n]);
                }

            if (m_compute_orientation)
                {
                h_cell_orientation.data[cli(offset, bin)] = h_orientation.data[n];
                }

            if (m_compute_idx)
                {
                h_cell_idx.data[cli(offset, bin)] = n;
                }
            }
        else
            {
            conditions.x = max((unsigned int)conditions.x, offset + 1);
            }

        // increment the cell occupancy counter
        h_cell_size.data[bin]++;
        }

        {
        // write out conditions
        ArrayHandle<uint3> h_conditions(m_conditions,
                                        access_location::host,
                                        access_mode::overwrite);
        *h_conditions.data = conditions;
        }
    }

bool CellList::checkConditions()
    {
    bool result = false;

    uint3 conditions;
    conditions = readConditions();

    // up m_Nmax to the overflow value, reallocate memory and set the overflow condition
    if (conditions.x > m_Nmax)
        {
        m_Nmax = conditions.x;
        result = true;
        }

    // detect nan position errors
    if (conditions.y)
        {
        unsigned int n = conditions.y - 1;
        ArrayHandle<unsigned int> h_tag(m_pdata->getTags(),
                                        access_location::host,
                                        access_mode::read);

        ostringstream s;
        s << "Particle with unique tag " << h_tag.data[n] << " has NaN for its position." << endl;
        throw runtime_error(s.str());
        }

    // detect particles leaving box errors
    if (conditions.z)
        {
        unsigned int n = conditions.z - 1;
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(),
                                   access_location::host,
                                   access_mode::read);
        ArrayHandle<unsigned int> h_tag(m_pdata->getTags(),
                                        access_location::host,
                                        access_mode::read);

        Scalar3 f = m_pdata->getBox().makeFraction(
            make_scalar3(h_pos.data[n].x, h_pos.data[n].y, h_pos.data[n].z));
        Scalar3 lo = m_pdata->getBox().getLo();
        Scalar3 hi = m_pdata->getBox().getHi();

        ostringstream s;
        s << "Particle with unique tag " << h_tag.data[n] << " is no longer in the simulation box."
          << std::endl
          << std::endl
          << "Cartesian coordinates: " << std::endl
          << "x: " << h_pos.data[n].x << " y: " << h_pos.data[n].y << " z: " << h_pos.data[n].z
          << std::endl
          << "Fractional coordinates: " << std::endl
          << "f.x: " << f.x << " f.y: " << f.y << " f.z: " << f.z << std::endl
          << "Local box lo: (" << lo.x << ", " << lo.y << ", " << lo.z << ")" << std::endl
          << "          hi: (" << hi.x << ", " << hi.y << ", " << hi.z << ")" << std::endl;
        throw runtime_error(s.str());
        }

    return result;
    }

void CellList::resetConditions()
    {
    // reset conditions
    ArrayHandle<uint3> h_conditions(m_conditions, access_location::host, access_mode::overwrite);
    *h_conditions.data = make_uint3(0, 0, 0);
    }

uint3 CellList::readConditions()
    {
    ArrayHandle<uint3> h_conditions(m_conditions, access_location::host, access_mode::read);
    return *h_conditions.data;
    }

namespace detail
    {
void export_CellList(pybind11::module& m)
    {
    pybind11::class_<CellList, Compute, std::shared_ptr<CellList>>(m, "CellList")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>());
    }

    } // end namespace detail

    } // end namespace hoomd
