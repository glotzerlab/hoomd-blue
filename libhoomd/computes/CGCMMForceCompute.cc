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
// Maintainer: akohlmey

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "CGCMMForceCompute.h"
#include <stdexcept>

/*! \file CGCMMForceCompute.cc
    \brief Defines the CGCMMForceCompute class
*/

using namespace std;

/*! \param sysdef System to compute forces on
    \param nlist Neighborlist to use for computing the forces
    \param r_cut Cuttoff radius beyond which the force is 0
    \post memory is allocated and all parameters ljX are set to 0.0
*/
CGCMMForceCompute::CGCMMForceCompute(boost::shared_ptr<SystemDefinition> sysdef,
                                     boost::shared_ptr<NeighborList> nlist,
                                     Scalar r_cut)
    : ForceCompute(sysdef), m_nlist(nlist), m_r_cut(r_cut)
    {
    assert(m_pdata);
    assert(m_nlist);
    
    if (r_cut < 0.0)
        {
        cerr << endl << "***Error! Negative r_cut in CGCMMForceCompute makes no sense" << endl << endl;
        throw runtime_error("Error initializing CGCMMForceCompute");
        }
        
    // initialize the number of types value
    m_ntypes = m_pdata->getNTypes();
    assert(m_ntypes > 0);
    
    // allocate storage for lj12, lj9, lj6, and lj4 parameters
    m_lj12 = new Scalar[m_ntypes*m_ntypes,exec_conf];
    m_lj9 = new Scalar[m_ntypes*m_ntypes,exec_conf];
    m_lj6 = new Scalar[m_ntypes*m_ntypes,exec_conf];
    m_lj4 = new Scalar[m_ntypes*m_ntypes,exec_conf];
    
    assert(m_lj12);
    assert(m_lj9);
    assert(m_lj6);
    assert(m_lj4);

    memset((void*)m_lj12, 0, sizeof(Scalar)*m_ntypes*m_ntypes);
    memset((void*)m_lj9,  0, sizeof(Scalar)*m_ntypes*m_ntypes);
    memset((void*)m_lj6,  0, sizeof(Scalar)*m_ntypes*m_ntypes);
    memset((void*)m_lj4,  0, sizeof(Scalar)*m_ntypes*m_ntypes);

    }


CGCMMForceCompute::~CGCMMForceCompute()
    {
    // deallocate our memory
    delete[] m_lj12;
    delete[] m_lj9;
    delete[] m_lj6;
    delete[] m_lj4;
    m_lj12 = NULL;
    m_lj9 = NULL;
    m_lj6 = NULL;
    m_lj4 = NULL;
    }


/*! \post The parameters \a lj12 through \a lj4 are set for the pairs \a typ1, \a typ2 and \a typ2, \a typ1.
    \note \a lj? are low level parameters used in the calculation. In order to specify
    these for a 12-4 and 9-6 lennard jones formula (with alpha), they should be set to the following.

        12-4
    - \a lj12 = 2.598076 * epsilon * pow(sigma,12.0)
    - \a lj9 = 0.0
    - \a lj6 = 0.0
    - \a lj4 = -alpha * 2.598076 * epsilon * pow(sigma,4.0)

        9-6
    - \a lj12 = 0.0
    - \a lj9 = 6.75 * epsilon * pow(sigma,9.0);
    - \a lj6 = -alpha * 6.75 * epsilon * pow(sigma,6.0)
    - \a lj4 = 0.0

       12-6
    - \a lj12 = 4.0 * epsilon * pow(sigma,12.0)
    - \a lj9 = 0.0
    - \a lj6 = -alpha * 4.0 * epsilon * pow(sigma,4.0)
    - \a lj4 = 0.0

    Setting the parameters for typ1,typ2 automatically sets the same parameters for typ2,typ1: there
    is no need to call this funciton for symmetric pairs. Any pairs that this function is not called
    for will have lj12 through lj4 set to 0.0.

    \param typ1 Specifies one type of the pair
    \param typ2 Specifies the second type of the pair
    \param lj12 1/r^12 term
    \param lj9  1/r^9 term
    \param lj6  1/r^6 term
    \param lj4  1/r^4 term
*/
void CGCMMForceCompute::setParams(unsigned int typ1, unsigned int typ2, Scalar lj12, Scalar lj9, Scalar lj6, Scalar lj4)
    {
    if (typ1 >= m_ntypes || typ2 >= m_ntypes)
        {
        cerr << endl << "***Error! Trying to set CGCMM params for a non existant type! " << typ1 << "," << typ2 << endl << endl;
        throw runtime_error("Error setting parameters in CGCMMForceCompute");
        }
        
    // set lj12 in both symmetric positions in the matrix
    m_lj12[typ1*m_ntypes + typ2] = lj12;
    m_lj12[typ2*m_ntypes + typ1] = lj12;
    
    // set lj9 in both symmetric positions in the matrix
    m_lj9[typ1*m_ntypes + typ2] = lj9;
    m_lj9[typ2*m_ntypes + typ1] = lj9;
    
    // set lj6 in both symmetric positions in the matrix
    m_lj6[typ1*m_ntypes + typ2] = lj6;
    m_lj6[typ2*m_ntypes + typ1] = lj6;
    
    // set lj4 in both symmetric positions in the matrix
    m_lj4[typ1*m_ntypes + typ2] = lj4;
    m_lj4[typ2*m_ntypes + typ1] = lj4;
    }

/*! CGCMMForceCompute provides
    - \c cgcmm_energy
*/
std::vector< std::string > CGCMMForceCompute::getProvidedLogQuantities()
    {
    vector<string> list;
    list.push_back("pair_cgcmm_energy");
    return list;
    }

Scalar CGCMMForceCompute::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == string("pair_cgcmm_energy"))
        {
        compute(timestep);
        return calcEnergySum();
        }
    else
        {
        cerr << endl << "***Error! " << quantity << " is not a valid log quantity for CGCMMForceCompute" << endl << endl;
        throw runtime_error("Error getting log value");
        }
    }

/*! \post The CGCMM forces are computed for the given timestep. The neighborlist's
    compute method is called to ensure that it is up to date.

    \param timestep specifies the current time step of the simulation
*/
void CGCMMForceCompute::computeForces(unsigned int timestep)
    {
    // start by updating the neighborlist
    m_nlist->compute(timestep);
    
    // start the profile for this compute
    if (m_prof) m_prof->push("CGCMM pair");
   
    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::overwrite);

    // Zero data for force calculation.
    memset((void*)h_force,0,sizeof(Scalar4)*m_force.getNumElements);
    memset((void*)h_virial,0,sizeof(Scalar)*m_virial.getNumElements);

   // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
       
    // depending on the neighborlist settings, we can take advantage of newton's third law
    // to reduce computations at the cost of memory access complexity: set that flag now
    bool third_law = m_nlist->getStorageMode() == NeighborList::half;
    
    // access the neighbor list
    ArrayHandle<unsigned int> h_n_neigh(m_nlist->getNNeighArray(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_nlist(m_nlist->getNListArray(), access_location::host, access_mode::read);
    Index2D nli = m_nlist->getNListIndexer();
    
    // access the particle data
    const ParticleDataArraysConst& arrays = m_pdata->acquireReadOnly();
    // sanity check
    assert(arrays.x != NULL && arrays.y != NULL && arrays.z != NULL);
    
    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();
    // sanity check
    assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);
    
    // create a temporary copy of r_cut sqaured
    Scalar r_cut_sq = m_r_cut * m_r_cut;
    
    // precalculate box lenghts for use in the periodic imaging
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;
    
    // tally up the number of forces calculated
    int64_t n_calc = 0;
    
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
        
        // access the lj12 and lj9 rows for the current particle type
        Scalar * __restrict__ lj12_row = &(m_lj12[typei*m_ntypes]);
        Scalar * __restrict__ lj9_row = &(m_lj9[typei*m_ntypes]);
        Scalar * __restrict__ lj6_row = &(m_lj6[typei*m_ntypes]);
        Scalar * __restrict__ lj4_row = &(m_lj4[typei*m_ntypes]);
        
        // initialize current particle force, potential energy, and virial to 0
        Scalar fxi = 0.0;
        Scalar fyi = 0.0;
        Scalar fzi = 0.0;
        Scalar pei = 0.0;
        Scalar viriali = 0.0;
        
        // loop over all of the neighbors of this particle
        const unsigned int size = (unsigned int)h_n_neigh.data[i];
        for (unsigned int j = 0; j < size; j++)
            {
            // increment our calculation counter
            n_calc++;
            
            // access the index of this neighbor (MEM TRANSFER: 1 scalar)
            unsigned int k = h_nlist.data[nli(i, j)];
            // sanity check
            assert(k < m_pdata->getN());
            
            // calculate dr (MEM TRANSFER: 3 scalars / FLOPS: 3)
            Scalar dx = xi - arrays.x[k];
            Scalar dy = yi - arrays.y[k];
            Scalar dz = zi - arrays.z[k];
            
            // access the type of the neighbor particle (MEM TRANSFER: 1 scalar
            unsigned int typej = arrays.type[k];
            // sanity check
            assert(typej < m_pdata->getNTypes());
            
            // apply periodic boundary conditions (FLOPS: 9 (worst case: first branch is missed, the 2nd is taken and the add is done)
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
                
            // start computing the force
            // calculate r squared (FLOPS: 5)
            Scalar rsq = dx*dx + dy*dy + dz*dz;
            
            // only compute the force if the particles are closer than the cuttoff (FLOPS: 1)
            if (rsq < r_cut_sq)
                {
                // compute the force magnitude/r in forcemag_divr (FLOPS: 14)
                Scalar r2inv = Scalar(1.0)/rsq;
                Scalar r3inv = r2inv / sqrt(rsq);
                Scalar r6inv = r3inv * r3inv;
                Scalar forcemag_divr = r6inv * (r2inv * (Scalar(12.0)*lj12_row[typej]*r6inv + Scalar(9.0)*r3inv*lj9_row[typej]
                                                         + Scalar(6.0)*lj6_row[typej]) + Scalar(4.0)*lj4_row[typej]);
                                                         
                // compute the pair energy and virial (FLOPS: 6)
                // note the sign in the virial calculation, this is because dx,dy,dz are \vec{r}_{ji} thus
                // there is no - in the 1/6 to compensate
                Scalar pair_virial = Scalar(1.0/6.0) * rsq * forcemag_divr;
                Scalar pair_eng = Scalar(0.5) * (r6inv * (lj12_row[typej] * r6inv + lj9_row[typej] * r3inv + lj6_row[typej]) + lj4_row[typej] * r2inv * r2inv);
                
                // add the force, potential energy and virial to the particle i
                // (FLOPS: 8)
                fxi += dx*forcemag_divr;
                fyi += dy*forcemag_divr;
                fzi += dz*forcemag_divr;
                pei += pair_eng;
                viriali += pair_virial;
                
                // add the force to particle j if we are using the third law (MEM TRANSFER: 10 scalars / FLOPS: 8)
                if (third_law)
                    {
                    h_force.data[k].x -= dx*forcemag_divr;
                    h_force.data[k].y -= dy*forcemag_divr;
                    h_force.data[k].z -= dz*forcemag_divr;
                    h_force.data[k].w += pair_eng;
                    h_virial.data[k] += pair_virial;
                    }
                }
                
            }
            
        // finally, increment the force, potential energy and virial for particle i
        // (MEM TRANSFER: 10 scalars / FLOPS: 5)
        h_force.data[i].x     += fxi;
        h_force.data[i].y  += fyi;
        h_force.data[i].z  += fzi;
        h_force.data[i].w  += pei;
        h_virial.data[i] += viriali;
        }
        
    m_pdata->release();
       
    int64_t flops = m_pdata->getN() * 5 + n_calc * (3+5+9+1+14+6+8);
    if (third_law) flops += n_calc * 8;
    int64_t mem_transfer = m_pdata->getN() * (5+4+10)*sizeof(Scalar) + n_calc * (1+3+1)*sizeof(Scalar);
    if (third_law) mem_transfer += n_calc*10*sizeof(Scalar);
    if (m_prof) m_prof->pop(flops, mem_transfer);
    }

void export_CGCMMForceCompute()
    {
    class_<CGCMMForceCompute, boost::shared_ptr<CGCMMForceCompute>, bases<ForceCompute>, boost::noncopyable >
    ("CGCMMForceCompute", init< boost::shared_ptr<SystemDefinition>, boost::shared_ptr<NeighborList>, Scalar >())
    .def("setParams", &CGCMMForceCompute::setParams)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

