/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2008, 2009 Ames Laboratory
Iowa State University and The Regents of the University of Michigan All rights
reserved.

HOOMD-blue may contain modifications ("Contributions") provided, and to which
copyright is held, by various Contributors who have granted The Regents of the
University of Michigan the right to modify and/or distribute such Contributions.

Redistribution and use of HOOMD-blue, in source and binary forms, with or
without modification, are permitted, provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice, this
list of conditions, and the following disclaimer in the documentation and/or
other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of HOOMD-blue's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND CONTRIBUTORS ``AS IS''
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND/OR
ANY WARRANTIES THAT THIS SOFTWARE IS FREE OF INFRINGEMENT ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: joaander

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
using namespace boost::python;

#include "TablePotential.h"
#include <stdexcept>

/*! \file TablePotential.cc
	\brief Defines the TablePotential class
*/

using namespace std;

/*! \param sysdef System to compute forces on
    \param nlist Neighborlist to use for computing the forces
    \param table_width Width the tables will be in memory
*/
TablePotential::TablePotential(boost::shared_ptr<SystemDefinition> sysdef,
                               boost::shared_ptr<NeighborList> nlist,
                               unsigned int table_width)
    : ForceCompute(sysdef), m_nlist(nlist), m_table_width(table_width)  
    {
    // sanity checks
    assert(m_pdata);
    assert(m_nlist);
    
    if (table_width == 0)
        {
        cerr << endl << "***Error! Table width of 0 given to TablePotential makes no sense" << endl << endl;
        throw runtime_error("Error initializing TablePotential");
        }
        
    // initialize the number of types value
    m_ntypes = m_pdata->getNTypes();
    assert(m_ntypes > 0);
    
    // allocate storage for the tables and parameters
    Index2DUpperTriangular table_index(m_ntypes);
    GPUArray<Scalar2> tables(m_table_width, table_index.getNumElements(), m_pdata->getExecConf());
    m_tables.swap(tables);
    GPUArray<Scalar4> params(table_index.getNumElements(), m_pdata->getExecConf());
    m_params.swap(params);
    
    assert(!m_tables.isNull());
    assert(!m_params.isNull());
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
                         const std::vector<Scalar> &V,
                         const std::vector<Scalar> &F,
                         Scalar rmin,
                         Scalar rmax)
    {
    // helpers to compute indices
    unsigned int cur_table_index = Index2DUpperTriangular(m_ntypes)(typ1, typ2);
    Index2D table_value(m_table_width);
    
    // access the arrays
    ArrayHandle<Scalar2> h_tables(m_tables, access_location::host, access_mode::readwrite);
    ArrayHandle<Scalar4> h_params(m_params, access_location::host, access_mode::readwrite);
    
    // range check on the parameters
    if (rmin < 0 || rmax < 0 || rmax <= rmin)
        {
        cerr << "***Error! rmin, rmax (" << rmin << "," << rmax 
             << ") given to TablePotential make no sense." << endl << endl;
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
    list.push_back("pair_table_energy");
    return list;
    }
	
Scalar TablePotential::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == string("pair_table_energy"))
        {
        compute(timestep);
        return calcEnergySum();
        }
    else
        {
        cerr << endl << "***Error! " << quantity << " is not a valid log quantity for TablePotential" << endl << endl;
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
    const vector< vector< unsigned int > >& full_list = m_nlist->getList();

    // access the particle data
    const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly(); 
    // sanity check
    assert(arrays.x != NULL && arrays.y != NULL && arrays.z != NULL);

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();
    // sanity check
    assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);	

    // precalculate box lenghts for use in the periodic imaging
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;

    // need to start from a zero force, energy and virial
    memset(m_fx, 0, sizeof(Scalar)*arrays.nparticles);
    memset(m_fy, 0, sizeof(Scalar)*arrays.nparticles);
    memset(m_fz, 0, sizeof(Scalar)*arrays.nparticles);
    memset(m_pe, 0, sizeof(Scalar)*arrays.nparticles);
    memset(m_virial, 0, sizeof(Scalar)*arrays.nparticles);

    // access the table data
    ArrayHandle<Scalar2> h_tables(m_tables, access_location::host, access_mode::read);
    ArrayHandle<Scalar4> h_params(m_params, access_location::host, access_mode::read);

    // index calculation helpers
    Index2DUpperTriangular table_index(m_ntypes);
    Index2D table_value(m_table_width);

    // for each particle
    for (unsigned int i = 0; i < arrays.nparticles; i++)
        {
        // access the particle's position and type (MEM TRANSFER: 4 scalars)
        Scalar xi = arrays.x[i];
        Scalar yi = arrays.y[i];
        Scalar zi = arrays.z[i];
        unsigned int typei = arrays.type[i];
        // sanity check
        assert(typei < m_pdata->getNTypes());
        
        // initialize current particle force, potential energy, and virial to 0
        Scalar fxi = 0.0;
        Scalar fyi = 0.0;
        Scalar fzi = 0.0;
        Scalar pei = 0.0;
        Scalar viriali = 0.0;
        
        // loop over all of the neighbors of this particle
        const vector< unsigned int >& list = full_list[i];
        const unsigned int size = (unsigned int)list.size();
        for (unsigned int j = 0; j < size; j++)
            {            
            // access the index of this neighbor
            unsigned int k = list[j];
            // sanity check
            assert(k < m_pdata->getN());
                
            // calculate dr
            Scalar dx = xi - arrays.x[k];
            Scalar dy = yi - arrays.y[k];
            Scalar dz = zi - arrays.z[k];
            
            // access the type of the neighbor particle
            unsigned int typej = arrays.type[k];
            // sanity check
            assert(typej < m_pdata->getNTypes());
            
            // apply periodic boundary conditions 
            if (dx >= box.xhi)
                dx -= Lx;
            else
            if (dx < box.xlo)
                dx += Lx;

            if (dy >= box.yhi)
                dy -= Ly;
            else
            if (dy < box.ylo)
                dy += Ly;

            if (dz >= box.zhi)
                dz -= Lz;
            else
            if (dz < box.zlo)
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
                unsigned int value_i = floor(value_f);
                Scalar2 VF0 = h_tables.data[table_value(value_i, cur_table_index)];
                Scalar2 VF1 = h_tables.data[table_value(value_i+1, cur_table_index)];
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
                
                // compute the virial (FLOPS: 2)
                // note the sign in the virial calculation, this is because dx,dy,dz are \vec{r}_{ji} thus
                // there is no - in the 1/6 to compensate	
                Scalar pair_virial = Scalar(1.0/6.0) * rsq * forcemag_divr;
                                
                // add the force, potential energy and virial to the particle i
                fxi += dx*forcemag_divr;
                fyi += dy*forcemag_divr;
                fzi += dz*forcemag_divr;
                pei += pair_eng;
                viriali += pair_virial;
                
                // add the force to particle j if we are using the third law
                if (third_law)
                    {
                    m_fx[k] -= dx*forcemag_divr;
                    m_fy[k] -= dy*forcemag_divr;
                    m_fz[k] -= dz*forcemag_divr;
                    m_pe[k] += pair_eng;
                    m_virial[k] += pair_virial;
                    }
                }
            }
            
        // finally, increment the force, potential energy and virial for particle i
        m_fx[i] += fxi;
        m_fy[i] += fyi;
        m_fz[i] += fzi;
        m_pe[i] += pei;
        m_virial[i] += viriali;
        }

    m_pdata->release();

    #ifdef ENABLE_CUDA
    // the force data is now only up to date on the cpu
    m_data_location = cpu;
    #endif

    if (m_prof) m_prof->pop();
    }

//! Exports the TablePotential class to python
void export_TablePotential()
    {
    class_<TablePotential, boost::shared_ptr<TablePotential>, bases<ForceCompute>, boost::noncopyable >
        ("TablePotential", init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<NeighborList>, unsigned int >())
        .def("setTable", &TablePotential::setTable)
        ;
        
    class_<std::vector<float> >("std_vector_float")
        .def(vector_indexing_suite<std::vector<float> >())
        ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif