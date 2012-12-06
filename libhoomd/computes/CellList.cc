/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008-2011 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

You may redistribute, use, and create derivate works of HOOMD-blue, in source
and binary forms, provided you abide by the following conditions:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer both in the code and
prominently in any materials provided with the distribution.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* All publications and presentations based on HOOMD-blue, including any reports
or published results obtained, in whole or in part, with HOOMD-blue, will
acknowledge its use according to the terms posted at the time of submission on:
http://codeblue.umich.edu/hoomd-blue/citations.html

* Any electronic documents citing HOOMD-Blue will link to the HOOMD-Blue website:
http://codeblue.umich.edu/hoomd-blue/

* Apart from the above required attributions, neither the name of the copyright
holder nor the names of HOOMD-blue's contributors may be used to endorse or
promote products derived from this software without specific prior written
permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS'' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR ANY
WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// Maintainer: joaander

/*! \file CellList.cc
    \brief Defines CellList
*/

#include <boost/python.hpp>
#include <boost/bind.hpp>
#include <algorithm>

#include "CellList.h"

using namespace boost;
using namespace boost::python;
using namespace std;

/*! \param sysdef system to compute the cell list of
*/
CellList::CellList(boost::shared_ptr<SystemDefinition> sysdef)
    : Compute(sysdef),  m_nominal_width(Scalar(1.0f)), m_radius(1), m_max_cells(UINT_MAX), m_compute_tdb(false),
      m_compute_orientation(false), m_compute_idx(false), m_flag_charge(false), m_flag_type(false)
    {
    m_exec_conf->msg->notice(5) << "Constructing CellList" << endl;

    // allocation is deferred until the first compute() call - initialize values to dummy variables
    m_width = make_scalar3(0.0, 0.0, 0.0);
    m_dim = make_uint3(0,0,0);
    m_Nmax = 0;
    m_params_changed = true;
    m_particles_sorted = false;
    m_box_changed = false;
    m_multiple = 1;

    GPUFlags<uint3> conditions(exec_conf);
    m_conditions.swap(conditions);
    resetConditions();

    m_num_ghost_cells = make_uint3(0,0,0);
    
    m_sort_connection = m_pdata->connectParticleSort(bind(&CellList::slotParticlesSorted, this));
    m_boxchange_connection = m_pdata->connectBoxChange(bind(&CellList::slotBoxChanged, this));
    }

CellList::~CellList()
    {
    m_exec_conf->msg->notice(5) << "Destroying CellList" << endl;
    m_sort_connection.disconnect();
    m_boxchange_connection.disconnect();
    }

//! Round down to the nearest multiple
/*! \param v Value to ound
    \param m Multiple
    \returns \a v if it is a multiple of \a m, otherwise, \a v rounded down to the nearest multiple of \a m.
*/
static unsigned int roundDown(unsigned int v, unsigned int m)
    {
    // use integer floor division
    unsigned int d = v/m;
    return d*m;
    }

/*! \returns Cell dimensions that match with the current width, box dimension, and max_cells setting
*/
uint3 CellList::computeDimensions()
    {
    uint3 dim;
    
    // calculate the bin dimensions
    const BoxDim& box = m_pdata->getBox();

    Scalar3 L = box.getL();
    dim.x = roundDown((unsigned int)((L.x) / (m_nominal_width)), m_multiple);
    dim.y = roundDown((unsigned int)((L.y) / (m_nominal_width)), m_multiple);

    // Add a ghost layer on every side where boundary conditions are non-periodic
    if (! box.getPeriodic().x)
        dim.x += 2;
    if (! box.getPeriodic().y)
        dim.y += 2;

    if (m_sysdef->getNDimensions() == 2)
        {
        dim.z = 1;
    
        // decrease the number of bins if it exceeds the max
        if (dim.x * dim.y * dim.z > m_max_cells)
            {
            float scale_factor = powf(float(m_max_cells) / float(dim.x*dim.y*dim.z), 1.0f/2.0f);
            dim.x = int(float(dim.x)*scale_factor);
            dim.y = int(float(dim.y)*scale_factor);
            }
        }
    else
        {
        dim.z = roundDown((unsigned int)((L.z) / (m_nominal_width)), m_multiple);

        // add ghost layer if necessary
        if (! box.getPeriodic().z)
            dim.z += 2;

        // decrease the number of bins if it exceeds the max
        if (dim.x * dim.y * dim.z > m_max_cells)
            {
            float scale_factor = powf(float(m_max_cells) / float(dim.x*dim.y*dim.z), 1.0f/3.0f);
            dim.x = int(float(dim.x)*scale_factor);
            dim.y = int(float(dim.y)*scale_factor);
            dim.z = int(float(dim.z)*scale_factor);
            }
        }
    
    return dim;
    }

void CellList::compute(unsigned int timestep)
    {
    bool force = false;
    
    if (m_prof)
        m_prof->push("Cell");
    
    if (m_params_changed)
        {
        // need to fully reinitialize on any parameter change
        initializeAll();
        m_params_changed = false;
        force = true;
        }
    
    if (m_box_changed)
        {
        uint3 new_dim = computeDimensions();
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
        // sorted particles simply need a forced update to get the proper indices in the data structure
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
    
    if (m_prof)
        m_prof->pop();
    }

/*! \param num_iters Number of iterations to average for the benchmark
    \returns Milliseconds of execution time per calculation

    Calls computeCellList repeatedly to benchmark the compute
*/
double CellList::benchmark(unsigned int num_iters)
    {
    ClockSource t;
    
    // ensure that any changed parameters have been propagaged and memory allocated
    compute(0);
    
    // warm up run
    computeCellList();
    
#ifdef ENABLE_CUDA
    if (exec_conf->isCUDAEnabled())
        {
        cudaThreadSynchronize();
        CHECK_CUDA_ERROR();
        }
#endif
    
    // benchmark
    uint64_t start_time = t.getTime();
    for (unsigned int i = 0; i < num_iters; i++)
        computeCellList();
        
#ifdef ENABLE_CUDA
    if (exec_conf->isCUDAEnabled())
        cudaThreadSynchronize();
#endif
    uint64_t total_time_ns = t.getTime() - start_time;
    
    // convert the run time to milliseconds
    return double(total_time_ns) / 1e6 / double(num_iters);
    }

void CellList::initializeAll()
    {
    initializeWidth();
    initializeMemory();
    }

void CellList::initializeWidth()
    {
    if (m_prof)
        m_prof->push("init");
    
    // initialize dimensions and width
    m_dim = computeDimensions();

    const BoxDim& box = m_pdata->getBox();

    // the number of ghost cells along every non-periodic direction is two (one on each side)
    if (m_sysdef->getNDimensions() == 2)
        m_num_ghost_cells = make_uint3(box.getPeriodic().x ? 0 : 2,
                                       box.getPeriodic().y ? 0 : 2,
                                       0);
    else
        m_num_ghost_cells = make_uint3(box.getPeriodic().x ? 0 : 2,
                                       box.getPeriodic().y ? 0 : 2,
                                       box.getPeriodic().z ? 0 : 2);

 
    Scalar3 L = box.getL();
    m_width.x = (L.x + m_nominal_width*m_num_ghost_cells.x) / Scalar(m_dim.x);
    m_width.y = (L.y + m_nominal_width*m_num_ghost_cells.y) / Scalar(m_dim.y);
    m_width.z = (L.z + m_nominal_width*m_num_ghost_cells.z) / Scalar(m_dim.z);

    if (m_prof)
        m_prof->pop();

    }

void CellList::initializeMemory()
    {
    if (m_prof)
        m_prof->push("init");

    // if it is still set at 0, estimate Nmax
    if (m_Nmax == 0)
        {
        unsigned int estim_Nmax = (unsigned int)(ceilf(float((m_pdata->getN()+m_pdata->getNGhosts())*1.0f / float(m_dim.x*m_dim.y*m_dim.z))));
        m_Nmax = estim_Nmax + 8 - (estim_Nmax & 7);
        }
    else
        {
        // otherwise, round up to the nearest multiple of 8 if we are not already on one
        if ((m_Nmax & 7) != 0)
            m_Nmax = m_Nmax + 8 - (m_Nmax & 7);
        }

    m_exec_conf->msg->notice(6) << "cell list: allocating " << m_dim.x << " x " << m_dim.y << " x " << m_dim.z
                                << " x " << m_Nmax << endl;

    // initialize indexers
    m_cell_indexer = Index3D(m_dim.x, m_dim.y, m_dim.z);
    m_cell_list_indexer = Index2D(m_Nmax, m_cell_indexer.getNumElements());
    
    unsigned int n_adj;
    if (m_sysdef->getNDimensions() == 2)
        n_adj = (m_radius*2+1)*(m_radius*2+1);
    else
        n_adj = (m_radius*2+1)*(m_radius*2+1)*(m_radius*2+1);
    
    m_cell_adj_indexer = Index2D(n_adj, m_cell_indexer.getNumElements());
    
    // allocate memory
    GPUArray<unsigned int> cell_size(m_cell_indexer.getNumElements(), exec_conf);
    m_cell_size.swap(cell_size);

    GPUArray<unsigned int> cell_adj(m_cell_adj_indexer.getNumElements(), exec_conf);
    m_cell_adj.swap(cell_adj);
    
    GPUArray<Scalar4> xyzf(m_cell_list_indexer.getNumElements(), exec_conf);
    m_xyzf.swap(xyzf);
    
    if (m_compute_tdb)
        {
        GPUArray<Scalar4> tdb(m_cell_list_indexer.getNumElements(), exec_conf);
        m_tdb.swap(tdb);
        }
    else
        {
        // array is no longer needed, discard it
        GPUArray<Scalar4> tdb;
        m_tdb.swap(tdb);
        }

    if (m_compute_orientation)
        {
        GPUArray<Scalar4> orientation(m_cell_list_indexer.getNumElements(), exec_conf);
        m_orientation.swap(orientation);
        }
    else
        {
        // array is no longer needed, discard it
        GPUArray<Scalar4> orientation;
        m_orientation.swap(orientation);
        }

    if (m_compute_idx)
        {
        GPUArray<unsigned int> idx(m_cell_list_indexer.getNumElements(), exec_conf);
        m_idx.swap(idx);
        }
    else
        {
        // array is no longer needed, discard it
        GPUArray<unsigned int> idx;
        m_idx.swap(idx);
        }

    if (m_prof)
        m_prof->pop();
    
    initializeCellAdj();
    }

void CellList::initializeCellAdj()
    {
    if (m_prof)
        m_prof->push("init");
    
    ArrayHandle<unsigned int> h_cell_adj(m_cell_adj, access_location::host, access_mode::overwrite);
    
    // loop over all cells
    for (int k = 0; k < int(m_dim.z); k++)
        for (int j = 0; j < int(m_dim.y); j++)
            for (int i = 0; i < int(m_dim.x); i++)
                {
                unsigned int cur_cell = m_cell_indexer(i,j,k);
                unsigned int offset = 0;
                
                // loop over neighboring cells
                // need signed integer values for performing index calculations with negative values
                int r = int(m_radius);
                int rk = r;
                if (m_sysdef->getNDimensions() == 2)
                    rk = 0;
                
                int mx = int(m_dim.x);
                int my = int(m_dim.y);
                int mz = int(m_dim.z);
                
                
                for (int nk = k-rk; nk <= k+rk; nk++)
                    for (int nj = j-r; nj <= j+r; nj++)
                        for (int ni = i-r; ni <= i+r; ni++)
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
                            h_cell_adj.data[m_cell_adj_indexer(offset, cur_cell)] = neigh_cell;
                            offset++;
                            }
                
                // sort the adj list for each cell
                sort(&h_cell_adj.data[m_cell_adj_indexer(0, cur_cell)],
                     &h_cell_adj.data[m_cell_adj_indexer(offset, cur_cell)]);
                }
    
    if (m_prof)
        m_prof->pop();
    }

void CellList::computeCellList()
    {
    if (m_prof)
        m_prof->push("compute");
    
    // acquire the particle data
    ArrayHandle< Scalar4 > h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle< Scalar4 > h_orientation(m_pdata->getOrientationArray(), access_location::host, access_mode::read);
    ArrayHandle< Scalar > h_charge(m_pdata->getCharges(), access_location::host, access_mode::read);
    ArrayHandle< unsigned int > h_body(m_pdata->getBodies(), access_location::host, access_mode::read);
    ArrayHandle< Scalar > h_diameter(m_pdata->getDiameters(), access_location::host, access_mode::read);
    const BoxDim& box = m_pdata->getBox();
  
    // access the cell list data arrays
    ArrayHandle<unsigned int> h_cell_size(m_cell_size, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_xyzf(m_xyzf, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_cell_orientation(m_orientation, access_location::host, access_mode::overwrite);
    ArrayHandle<unsigned int> h_cell_idx(m_idx, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar4> h_tdb(m_tdb, access_location::host, access_mode::overwrite);
    uint3 conditions = make_uint3(0,0,0);

    // shorthand copies of the indexers
    Index3D ci = m_cell_indexer;
    Index2D cli = m_cell_list_indexer;
    
    // clear the bin sizes to 0
    memset(h_cell_size.data, 0, sizeof(unsigned int) * m_cell_indexer.getNumElements());
    
    // for each particle
    unsigned n_tot_particles = m_pdata->getN() + m_pdata->getNGhosts();

    Scalar3 ghost_width = m_nominal_width/Scalar(2.0)*make_scalar3((Scalar)m_num_ghost_cells.x, (Scalar)m_num_ghost_cells.y, (Scalar)m_num_ghost_cells.z);

    for (unsigned int n = 0; n < n_tot_particles; n++)
        {
        Scalar3 p = make_scalar3(h_pos.data[n].x, h_pos.data[n].y, h_pos.data[n].z);
        if (isnan(p.x) || isnan(p.y) || isnan(p.z))
            {
            conditions.y = n+1;
            continue;
            }
            
        // find the bin each particle belongs in
        Scalar3 f = box.makeFraction(p,ghost_width);
        int ib = (int)(f.x * m_dim.x);
        int jb = (int)(f.y * m_dim.y);
        int kb = (int)(f.z * m_dim.z);
        
        // need to handle the case where the particle is exactly at the box hi
        if (ib == (int)m_dim.x)
            ib = 0;
        if (jb == (int)m_dim.y)
            jb = 0;
        if (kb == (int)m_dim.z)
            kb = 0;

        // sanity check
        assert(ib < (int)(m_dim.x) && jb < (int)(m_dim.y) && kb < (int)(m_dim.z));
        
        // record its bin
        unsigned int bin = ci(ib, jb, kb);
        // check if the particle is inside the dimensions
        if (bin >= ci.getNumElements())
            {
            // if a ghost particle is out of bounds, silently ignore it
            if (n < m_pdata->getN())
                conditions.z = n+1;
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
            h_xyzf.data[cli(offset, bin)] = make_scalar4(h_pos.data[n].x, h_pos.data[n].y, h_pos.data[n].z, flag);
            if (m_compute_tdb)
                {
                h_tdb.data[cli(offset, bin)] = make_scalar4(h_pos.data[n].w,
                                                            h_diameter.data[n],
                                                            __int_as_scalar(h_body.data[n]),
                                                            Scalar(0.0));
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
            conditions.x = max(conditions.x, offset+1);
            }
        
        // increment the cell occupancy counter
        h_cell_size.data[bin]++;
        }

    // write out conditions
    m_conditions.resetFlags(conditions);

    if (m_prof)
        m_prof->pop();
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
        ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);
        m_exec_conf->msg->error() << "Particle " << h_tag.data[n] << " has NaN for its position." << endl;
        throw runtime_error("Error computing cell list");
        }

    // detect particles leaving box errors
    if (conditions.z)
        {
        unsigned int n = conditions.z - 1;
        ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
        ArrayHandle<unsigned int> h_tag(m_pdata->getTags(), access_location::host, access_mode::read);

        m_exec_conf->msg->error() << (n >= m_pdata->getN() ? "Ghost" : "")
                                  <<"Particle " << h_tag.data[n] << " is no longer in the simulation box."
                                  << endl << endl;

        m_exec_conf->msg->notice(2) << "x: " << h_pos.data[n].x << " y: " << h_pos.data[n].y << " z: " << h_pos.data[n].z << std::endl;
        Scalar3 lo = m_pdata->getBox().getLo();
        Scalar3 hi = m_pdata->getBox().getHi();
        m_exec_conf->msg->notice(2) << "Local box lo: (" << lo.x << ", " << lo.y << ", " << lo.z << ")" << std::endl;
        m_exec_conf->msg->notice(2) << "          hi: (" << hi.x << ", " << hi.y << ", " << hi.z << ")" << std::endl;
        throw runtime_error("Error computing cell list");
        }

    return result;
    }

void CellList::resetConditions()
    {
    m_conditions.resetFlags(make_uint3(0,0,0));
    }

uint3 CellList::readConditions()
    {
    return m_conditions.readFlags();
    }

/*! Generic statistics that apply to any cell list, Derived classes should
    print any pertinent information they see fit to.
 */
void CellList::printStats()
    {
    // return earsly if the notice level is less than 1
    if (m_exec_conf->msg->getNoticeLevel() < 1)
        return;

    m_exec_conf->msg->notice(1) << "-- Cell list stats:" << endl;
    m_exec_conf->msg->notice(1) << "Dimension: " << m_dim.x << ", " << m_dim.y << ", " << m_dim.z << "" << endl;
    m_exec_conf->msg->notice(1) << "Width    : " << m_width.x << ", " << m_width.y << ", " << m_width.z << "" << endl;

    // access the number of cell members to generate stats
    ArrayHandle<unsigned int> h_cell_size(m_cell_size, access_location::host, access_mode::read);

    // handle the rare case where printStats is called before the cell list is initialized
    if (h_cell_size.data != NULL)
        {
        // build some simple statistics of the number of neighbors
        unsigned int n_min = h_cell_size.data[0];
        unsigned int n_max = h_cell_size.data[0];

        for (unsigned int i = 0; i < m_cell_indexer.getNumElements(); i++)
            {
            unsigned int n = (unsigned int)h_cell_size.data[i];
            if (n < n_min)
                n_min = n;
            if (n > n_max)
                n_max = n;
            }

        // divide to get the average
        Scalar n_avg = Scalar(m_pdata->getN() + m_pdata->getNGhosts()) / Scalar(m_cell_indexer.getNumElements());
        m_exec_conf->msg->notice(1) << "n_min    : " << n_min << " / n_max: " << n_max << " / n_avg: " << n_avg << endl;
        }
    }


void export_CellList()
    {
    class_<CellList, boost::shared_ptr<CellList>, bases<Compute>, boost::noncopyable >
        ("CellList", init< boost::shared_ptr<SystemDefinition> >())
        .def("setNominalWidth", &CellList::setNominalWidth)
        .def("setRadius", &CellList::setRadius)
        .def("setMaxCells", &CellList::setMaxCells)
        .def("setComputeTDB", &CellList::setComputeTDB)
        .def("setFlagCharge", &CellList::setFlagCharge)
        .def("setFlagIndex", &CellList::setFlagIndex)
        .def("getWidth", &CellList::getWidth, return_internal_reference<>())
        .def("getDim", &CellList::getDim, return_internal_reference<>())
        .def("getNmax", &CellList::getNmax)
        .def("benchmark", &CellList::benchmark)
        ;
    }

