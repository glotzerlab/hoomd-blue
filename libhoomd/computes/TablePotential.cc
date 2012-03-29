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

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 4267 )
#endif

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
using namespace boost::python;

#include "TablePotential.h"

#ifdef ENABLE_OPENMP
#include <omp.h>
#endif

#include <stdexcept>

/*! \file TablePotential.cc
    \brief Defines the TablePotential class
*/

using namespace std;

/*! \param sysdef System to compute forces on
    \param nlist Neighborlist to use for computing the forces
    \param table_width Width the tables will be in memory
    \param log_suffix Name given to this instance of the table potential
*/
TablePotential::TablePotential(boost::shared_ptr<SystemDefinition> sysdef,
                               boost::shared_ptr<NeighborList> nlist,
                               unsigned int table_width,
                               const std::string& log_suffix)
        : ForceCompute(sysdef), m_nlist(nlist), m_table_width(table_width)
    {
    m_exec_conf->msg->notice(5) << "Constructing TablePotential" << endl;

    // sanity checks
    assert(m_pdata);
    assert(m_nlist);
    
    if (table_width == 0)
        {
        m_exec_conf->msg->error() << "pair.table: Table width of 0 is invalid" << endl;
        throw runtime_error("Error initializing TablePotential");
        }
        
    // initialize the number of types value
    m_ntypes = m_pdata->getNTypes();
    assert(m_ntypes > 0);
    
    // allocate storage for the tables and parameters
    Index2DUpperTriangular table_index(m_ntypes);
    GPUArray<float2> tables(m_table_width, table_index.getNumElements(), exec_conf);
    m_tables.swap(tables);
    GPUArray<Scalar4> params(table_index.getNumElements(), exec_conf);
    m_params.swap(params);
    
    assert(!m_tables.isNull());
    assert(!m_params.isNull());
    
    // initialize memory for per thread reduction
    allocateThreadPartial();
    
    m_log_name = std::string("pair_table_energy") + log_suffix;
    }
    
TablePotential::~TablePotential()
    {
    m_exec_conf->msg->notice(5) << "Destroying TablePotential" << endl;
    }

/*! \param typ1 First particle type index in the pair to set
    \param typ2 Second particle type index in the pair to set
    \param V Table for the potential V
    \param F Table for the potential F (must be - dV / dr)
    \param rmin Minimum r in the potential
    \param rmax Maximum r in the potential
    \post Values from \a V and \a F are copied into the interal storage for type pair (typ1, typ2)
    \note There is no need to call this again for typ2,typ1
    \note See TablePotential for a detailed definiton of rmin and rmax
*/
void TablePotential::setTable(unsigned int typ1,
                              unsigned int typ2,
                              const std::vector<float> &V,
                              const std::vector<float> &F,
                              Scalar rmin,
                              Scalar rmax)
    {
    // helpers to compute indices
    unsigned int cur_table_index = Index2DUpperTriangular(m_ntypes)(typ1, typ2);
    Index2D table_value(m_table_width);
    
    // access the arrays
    ArrayHandle<float2> h_tables(m_tables, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_params(m_params, access_location::host, access_mode::readwrite);
    
    // range check on the parameters
    if (rmin < 0 || rmax < 0 || rmax <= rmin)
        {
        m_exec_conf->msg->error() << "pair.table rmin, rmax (" << rmin << "," << rmax
             << ") is invalid" << endl;
        throw runtime_error("Error initializing TablePotential");
        }
        
    if (V.size() != m_table_width || F.size() != m_table_width)
        {
        m_exec_conf->msg->error() << "pair.table: table provided to setTable is not of the correct size" << endl;
        throw runtime_error("Error initializing TablePotential");
        }
        
    // fill out the parameters
    h_params.data[cur_table_index].x = rmin;
    h_params.data[cur_table_index].y = rmax;
    h_params.data[cur_table_index].z = (rmax - rmin) / Scalar(m_table_width - 1);
    
    // fill out the table
    for (unsigned int i = 0; i < m_table_width; i++)
        {
        h_tables.data[table_value(i, cur_table_index)].x = V[i];
        h_tables.data[table_value(i, cur_table_index)].y = F[i];
        }
    }

/*! TablePotential provides
    - \c pair_table_energy
*/
std::vector< std::string > TablePotential::getProvidedLogQuantities()
    {
    vector<string> list;
    list.push_back(m_log_name);
    return list;
    }

Scalar TablePotential::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == m_log_name)
        {
        compute(timestep);
        return calcEnergySum();
        }
    else
        {
        m_exec_conf->msg->error() << "pair.table: " << quantity << " is not a valid log quantity for TablePotential" << endl;
        throw runtime_error("Error getting log value");
        }
    }

/*! \post The table based forces are computed for the given timestep. The neighborlist's
compute method is called to ensure that it is up to date.

\param timestep specifies the current time step of the simulation
*/
void TablePotential::computeForces(unsigned int timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);
    
    // start the profile for this compute
    if (m_prof) m_prof->push("Table pair");
    
    // depending on the neighborlist settings, we can take advantage of newton's third law
    // to reduce computations at the cost of memory access complexity: set that flag now
    bool third_law = m_nlist->getStorageMode() == NeighborList::half;
    
    // access the neighbor list
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(), access_location::host, access_mode::read);
    Index2D nli = m_nlist->getNListIndexer();
    
    // access the particle data
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::overwrite);
    
    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);
    
    // Zero data for force calculation.
    memset((void*)h_force.data,0,sizeof(Scalar4)*m_force.getNumElements());
    memset((void*)h_virial.data,0,sizeof(Scalar)*m_virial.getNumElements());

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();
    // sanity check
    assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);
    
    // precalculate box lenghts for use in the periodic imaging
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;
    
    // access the table data
    ArrayHandle<float2> h_tables(m_tables, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_params(m_params, access_location::host, access_mode::read);
    
    // index calculation helpers
    Index2DUpperTriangular table_index(m_ntypes);
    Index2D table_value(m_table_width);
    
#pragma omp parallel
    {
    #ifdef ENABLE_OPENMP
    int tid = omp_get_thread_num();
    #else
    int tid = 0;
    #endif

    // need to start from a zero force, energy and virial
    memset(&m_fdata_partial[m_index_thread_partial(0,tid)] , 0, sizeof(Scalar4)*m_pdata->getN());
    memset(&m_virial_partial[6*m_index_thread_partial(0,tid)] , 0, 6*sizeof(Scalar)*m_pdata->getN());
    
    // for each particle
#pragma omp for schedule(guided)
    for (int i = 0; i < (int) m_pdata->getN(); i++)
        {
        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        Scalar xi = h_pos.data[i].x;
        Scalar yi = h_pos.data[i].y;
        Scalar zi = h_pos.data[i].z;
        unsigned int typei = __scalar_as_int(h_pos.data[i].w);
        // sanity check
        assert(typei < m_pdata->getNTypes());
        
        // initialize current particle force, potential energy, and virial to 0
        Scalar fxi = 0.0;
        Scalar fyi = 0.0;
        Scalar fzi = 0.0;
        Scalar pei = 0.0;
        Scalar virialxxi = 0.0;
        Scalar virialxyi = 0.0;
        Scalar virialxzi = 0.0;
        Scalar virialyyi = 0.0;
        Scalar virialyzi = 0.0;
        Scalar virialzzi = 0.0;

        // loop over all of the neighbors of this particle
        const unsigned int size = (unsigned int)h_n_neigh.data[i];
        for (unsigned int j = 0; j < size; j++)
            {
            // access the index of this neighbor
            unsigned int k = h_nlist.data[nli(i, j)];
            // sanity check
            assert(k < m_pdata->getN());
            
            // calculate dr
            Scalar dx = xi - h_pos.data[k].x;
            Scalar dy = yi - h_pos.data[k].y;
            Scalar dz = zi - h_pos.data[k].z;
            
            // access the type of the neighbor particle
            unsigned int typej = __scalar_as_int(h_pos.data[k].w);
            // sanity check
            assert(typej < m_pdata->getNTypes());
            
            // apply periodic boundary conditions
            if (dx >= box.xhi)
                dx -= Lx;
            else if (dx < box.xlo)
                dx += Lx;
                
            if (dy >= box.yhi)
                dy -= Ly;
            else if (dy < box.ylo)
                dy += Ly;
                
            if (dz >= box.zhi)
                dz -= Lz;
            else if (dz < box.zlo)
                dz += Lz;
                
            // access needed parameters
            unsigned int cur_table_index = table_index(typei, typej);
            Scalar4 params = h_params.data[cur_table_index];
            Scalar rmin = params.x;
            Scalar rmax = params.y;
            Scalar delta_r = params.z;
            
            // start computing the force
            Scalar rsq = dx*dx + dy*dy + dz*dz;
            Scalar r = sqrt(rsq);
            
            // only compute the force if the particles are within the region defined by V
            if (r < rmax && r >= rmin)
                {
                // precomputed term
                Scalar value_f = (r - rmin) / delta_r;
                
                // compute index into the table and read in values
                unsigned int value_i = (unsigned int)floor(value_f);
                float2 VF0 = h_tables.data[table_value(value_i, cur_table_index)];
                float2 VF1 = h_tables.data[table_value(value_i+1, cur_table_index)];
                // unpack the data
                Scalar V0 = VF0.x;
                Scalar V1 = VF1.x;
                Scalar F0 = VF0.y;
                Scalar F1 = VF1.y;
                
                // compute the linear interpolation coefficient
                Scalar f = value_f - float(value_i);
                
                // interpolate to get V and F;
                Scalar V = V0 + f * (V1 - V0);
                Scalar F = F0 + f * (F1 - F0);
                
                // convert to standard variables used by the other pair computes in HOOMD-blue
                Scalar forcemag_divr = Scalar(0.0);
                if (r > Scalar(0.0))
                    forcemag_divr = F / r;
                Scalar pair_eng = Scalar(0.5) * V;
                
                // compute the virial
                Scalar forcemag_div2r = Scalar(0.5) * forcemag_divr;
                virialxxi += forcemag_div2r*dx*dx;
                virialxyi += forcemag_div2r*dx*dy;
                virialxzi += forcemag_div2r*dx*dz;
                virialyyi += forcemag_div2r*dy*dy;
                virialyzi += forcemag_div2r*dy*dz;
                virialzzi += forcemag_div2r*dz*dz;

                // add the force, potential energy and virial to the particle i
                fxi += dx*forcemag_divr;
                fyi += dy*forcemag_divr;
                fzi += dz*forcemag_divr;
                pei += pair_eng;
                
                // add the force to particle j if we are using the third law
                if (third_law)
                    {
                    unsigned int mem_idx = m_index_thread_partial(k,tid);
                    m_fdata_partial[mem_idx].x -= dx*forcemag_divr;
                    m_fdata_partial[mem_idx].y -= dy*forcemag_divr;
                    m_fdata_partial[mem_idx].z -= dz*forcemag_divr;
                    m_fdata_partial[mem_idx].w += pair_eng;
                    m_virial_partial[0+6*mem_idx] += forcemag_div2r * dx * dx;
                    m_virial_partial[1+6*mem_idx] += forcemag_div2r * dx * dy;
                    m_virial_partial[2+6*mem_idx] += forcemag_div2r * dx * dz;
                    m_virial_partial[3+6*mem_idx] += forcemag_div2r * dy * dy;
                    m_virial_partial[4+6*mem_idx] += forcemag_div2r * dy * dz;
                    m_virial_partial[5+6*mem_idx] += forcemag_div2r * dz * dz;
                    }
                }
            }
            
        // finally, increment the force, potential energy and virial for particle i
        unsigned int mem_idx = m_index_thread_partial(i,tid);
        m_fdata_partial[mem_idx].x += fxi;
        m_fdata_partial[mem_idx].y += fyi;
        m_fdata_partial[mem_idx].z += fzi;
        m_fdata_partial[mem_idx].w += pei;
        m_virial_partial[0+6*mem_idx] += virialxxi;
        m_virial_partial[1+6*mem_idx] += virialxyi;
        m_virial_partial[2+6*mem_idx] += virialxzi;
        m_virial_partial[3+6*mem_idx] += virialyyi;
        m_virial_partial[4+6*mem_idx] += virialyzi;
        m_virial_partial[5+6*mem_idx] += virialzzi;
        }
    
#pragma omp barrier
    
    // now that the partial sums are complete, sum up the results in parallel
#pragma omp for
    for (int i = 0; i < (int)m_pdata->getN(); i++)
        {
        // assign result from thread 0
        h_force.data[i].x = m_fdata_partial[i].x;
        h_force.data[i].y = m_fdata_partial[i].y;
        h_force.data[i].z = m_fdata_partial[i].z;
        h_force.data[i].w = m_fdata_partial[i].w;
        for (int j = 0; j < 6; j++)
            h_virial.data[j*m_virial_pitch+i] = m_virial_partial[j+6*i];

        #ifdef ENABLE_OPENMP
        // add results from other threads
        int nthreads = omp_get_num_threads();
        for (int thread = 1; thread < nthreads; thread++)
            {
            unsigned int mem_idx = m_index_thread_partial(i,thread);
            h_force.data[i].x += m_fdata_partial[mem_idx].x;
            h_force.data[i].y += m_fdata_partial[mem_idx].y;
            h_force.data[i].z += m_fdata_partial[mem_idx].z;
            h_force.data[i].w += m_fdata_partial[mem_idx].w;
            for (int j = 0; j < 6; j++)
                h_virial.data[j*m_virial_pitch+i] += m_virial_partial[j+6*mem_idx];
            }
        #endif
        }
    } // end omp parallel

        
    if (m_prof) m_prof->pop();
    }

//! Exports the TablePotential class to python
void export_TablePotential()
    {
    class_<TablePotential, boost::shared_ptr<TablePotential>, bases<ForceCompute>, boost::noncopyable >
    ("TablePotential", init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<NeighborList>, unsigned int, const std::string& >())
    .def("setTable", &TablePotential::setTable)
    ;
    
    class_<std::vector<float> >("std_vector_float")
    .def(vector_indexing_suite<std::vector<float> >())
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

