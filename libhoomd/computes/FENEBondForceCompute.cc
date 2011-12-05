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

// Maintainer: phillicl

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4103 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "FENEBondForceCompute.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <math.h>

using namespace std;

/*! \file FENEBondForceCompute.cc
    \brief Defines the FENEBondForceCompute class
*/

/*! \param sysdef System to compute forces on
    \param log_suffix Name given to this instance of the fene bond
    \post Memory is allocated, default parameters are set and forces are zeroed.
*/
FENEBondForceCompute::FENEBondForceCompute(boost::shared_ptr<SystemDefinition> sysdef, const std::string& log_suffix) : ForceCompute(sysdef),
        m_K(NULL), m_r_0(NULL), m_lj1(NULL), m_lj2(NULL), m_epsilon(NULL)
    {
    // access the bond data for later use
    m_bond_data = m_sysdef->getBondData();
    m_log_name = std::string("bond_fene_energy") + log_suffix;
  
    // check for some silly errors a user could make
    if (m_bond_data->getNBondTypes() == 0)
        {
        cout << endl << "***Error! No bond types specified" << endl << endl;
        throw runtime_error("Error initializing FENEBondForceCompute");
        }
        
    // allocate the parameters
    m_K = new Scalar[m_bond_data->getNBondTypes()];
    m_r_0 = new Scalar[m_bond_data->getNBondTypes()];
    m_lj1 =new Scalar[m_bond_data->getNBondTypes()];
    m_lj2 =new Scalar[m_bond_data->getNBondTypes()];
    m_epsilon =new Scalar[m_bond_data->getNBondTypes()];
    
    
    for (unsigned int i = 0; i < m_bond_data->getNBondTypes(); i++) m_lj1[i]=Scalar(1.0);
    for (unsigned int i = 0; i < m_bond_data->getNBondTypes(); i++) m_lj2[i]=Scalar(1.0);
    }

FENEBondForceCompute::~FENEBondForceCompute()
    {
    delete[] m_K;
    delete[] m_r_0;
    delete[] m_lj1;
    delete[] m_lj2;
    delete[] m_epsilon;
    }

/*! \param type Type of the bond to set parameters for
    \param K Stiffness parameter for the force computation
    \param r_0 maximum bond length for the force computation
    \param sigma Value of sigma in the force calculation
    \param epsilon Value of epsilon in the force calculation

    Sets parameters for the potential of a particular bond type
*/
void FENEBondForceCompute::setParams(unsigned int type, Scalar K, Scalar r_0, Scalar sigma, Scalar epsilon)
    {
    // make sure the type is valid
    if (type >= m_bond_data->getNBondTypes())
        {
        cout << endl << "***Error! Invalid bond type specified" << endl << endl;
        throw runtime_error("Error setting parameters in FENEBondForceCompute");
        }
        
    m_K[type] = K;
    m_r_0[type] = r_0;
    m_lj1[type] = Scalar(4.0)*epsilon*pow(sigma,Scalar(12.0));
    m_lj2[type] = Scalar(4.0)*epsilon*pow(sigma,Scalar(6.0));
    m_epsilon[type] = epsilon;
        
    // check for some silly errors a user could make
    if (K < 0)
        cout << "***Warning! K < 0 specified for fene bond" << endl;
    if (r_0 < 0)
        cout << "***Warning! r_0 < 0 specified for fene bond" << endl;
    if (sigma < 0)
        cout << "***Warning! sigma < 0 specified for fene bond" << endl;
    if (epsilon < 0)
        cout << "***Warning! epsilon < 0 specified for fene bond" << endl;
    }

/*! BondForceCompute provides
    - \c bond_fene_energy
*/
std::vector< std::string > FENEBondForceCompute::getProvidedLogQuantities()
    {
    vector<string> list;
    list.push_back(m_log_name);
    return list;
    }

/*! \param quantity Name of the quantity to get the log value of
    \param timestep Current time step of the simulation
*/
Scalar FENEBondForceCompute::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == m_log_name)
        {
        compute(timestep);
        return calcEnergySum();
        }
    else
        {
        cerr << endl << "***Error! " << quantity 
             << " is not a valid log quantity for FENEBondForceCompute" 
             << endl << endl;
        throw runtime_error("Error getting log value");
        }
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void FENEBondForceCompute::computeForces(unsigned int timestep)
    {
    if (m_prof) m_prof->push("FENE");
    
    assert(m_pdata);
    // access the particle data arrays
    ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();

    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::overwrite);

    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
    assert(arrays.x);
    assert(arrays.y);
    assert(arrays.z);
    assert(arrays.diameter);
    
    // Zero data for force calculation.
    memset((void*)h_force.data,0,sizeof(Scalar4)*m_force.getNumElements());
    memset((void*)h_virial.data,0,sizeof(Scalar)*m_virial.getNumElements());
    
    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();
    // sanity check
    assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);
    
    // precalculate box lengths
    Scalar Lx = box.xhi - box.xlo;
    Scalar Ly = box.yhi - box.ylo;
    Scalar Lz = box.zhi - box.zlo;
    Scalar Lx2 = Lx / Scalar(2.0);
    Scalar Ly2 = Ly / Scalar(2.0);
    Scalar Lz2 = Lz / Scalar(2.0);
    
    // for each of the bonds
    const unsigned int size = (unsigned int)m_bond_data->getNumBonds();
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the bond
        const Bond& bond = m_bond_data->getBond(i);
        assert(bond.a < m_pdata->getN());
        assert(bond.b < m_pdata->getN());
       
        // transform a and b into indicies into the particle data arrays
        // (MEM TRANSFER: 4 integers)
        unsigned int idx_a = arrays.rtag[bond.a];
        unsigned int idx_b = arrays.rtag[bond.b];
        assert(idx_a < m_pdata->getN());
        assert(idx_b < m_pdata->getN());
        
        // calculate d\vec{r}
        // (MEM TRANSFER: 6 Scalars / FLOPS: 3)
        Scalar dx = arrays.x[idx_b] - arrays.x[idx_a];
        Scalar dy = arrays.y[idx_b] - arrays.y[idx_a];
        Scalar dz = arrays.z[idx_b] - arrays.z[idx_a];
        Scalar diameter_a = arrays.diameter[idx_a];
        Scalar diameter_b = arrays.diameter[idx_b];
        
        // if the vector crosses the box, pull it back
        // (FLOPS: 9 (worst case: first branch is missed, the 2nd is taken and the add is done))
        if (dx >= Lx2)
            dx -= Lx;
        else if (dx < -Lx2)
            dx += Lx;
            
        if (dy >= Ly2)
            dy -= Ly;
        else if (dy < -Ly2)
            dy += Ly;
            
        if (dz >= Lz2)
            dz -= Lz;
        else if (dz < -Lz2)
            dz += Lz;
            
        // sanity check
        assert(dx >= box.xlo && dx < box.xhi);
        assert(dy >= box.ylo && dx < box.yhi);
        assert(dz >= box.zlo && dx < box.zhi);
        
        // on paper, the formula turns out to be: F = -K/(1-(r/r_0)^2) * \vec{r} + (12*lj1/r^12 - 6*lj2/r^6) *\vec{r}
        // FLOPS: 5
        Scalar rsq = dx*dx+dy*dy+dz*dz;
        Scalar rmdoverr = 1.0f;
        
        // Correct the rsq for particles that are not unit in size.
        Scalar rtemp = sqrt(rsq) - diameter_a/2 - diameter_b/2 + 1.0;
        rmdoverr = rtemp/sqrt(rsq);
        rsq = rtemp*rtemp;
         
        // compute the force magnitude/r in forcemag_divr (FLOPS: 9)
        Scalar r2inv = Scalar(1.0)/rsq;
        Scalar r6inv = r2inv * r2inv * r2inv;
        
        Scalar WCAforcemag_divr = Scalar(0.0);
        Scalar pair_eng = Scalar(0.0);

        // add != 0.0f check to allow epsilon=0 FENE bonds to go to r=0
        if (rsq < 1.2599210498 && m_epsilon[bond.type] != 0)     //wcalimit squared (2^(1/6))^2
            {
            WCAforcemag_divr = r2inv * r6inv * (Scalar(12.0)*m_lj1[bond.type]*r6inv - Scalar(6.0)*m_lj2[bond.type]);
            pair_eng = Scalar(0.5) * (r6inv * (m_lj1[bond.type]*r6inv - m_lj2[bond.type]) + m_epsilon[bond.type]);
            }
        if (!isfinite(pair_eng))
            pair_eng = 0.0f;
            
        // Additional check for FENE spring
        assert(rsq < m_r_0[bond.type]*m_r_0[bond.type]);
        
        // calculate force and energy
        // MEM TRANSFER 2 Scalars: FLOPS: 13
        Scalar forcemag_divr = -m_K[bond.type] / (Scalar(1.0) - rsq /
                                (m_r_0[bond.type]*m_r_0[bond.type]))*rmdoverr + WCAforcemag_divr*rmdoverr;  //FLOPS 4
        Scalar bond_eng = -Scalar(0.5) * Scalar(0.5) * m_K[bond.type] * (m_r_0[bond.type] * m_r_0[bond.type]) * 
                           log(Scalar(1.0) - rsq/(m_r_0[bond.type] * m_r_0[bond.type]));
        
        // detect non-finite results and zero them. This will result in the correct 0 force for r ~= 0. The energy
        // will be incorrect for r > r_0, however. Assuming that r > r_0 because K == 0, this is fine.
        if (!isfinite(forcemag_divr))
            forcemag_divr = 0.0f;
        if (!isfinite(bond_eng))
            bond_eng = 0.0f;

        // calculate virial (FLOPS: 2)
        Scalar bond_virial = Scalar(1.0/6.0) * rsq * forcemag_divr;
        
        // add the force to the particles
        // (MEM TRANSFER: 20 Scalars / FLOPS 16)
        h_force.data[idx_b].x += forcemag_divr * dx;
        h_force.data[idx_b].y += forcemag_divr * dy;
        h_force.data[idx_b].z += forcemag_divr * dz;
        h_force.data[idx_b].w += bond_eng + pair_eng;
        h_virial.data[idx_b]  += bond_virial;

        h_force.data[idx_a].x -= forcemag_divr * dx;
        h_force.data[idx_a].y -= forcemag_divr * dy;
        h_force.data[idx_a].z -= forcemag_divr * dz;
        h_force.data[idx_a].w += bond_eng + pair_eng;
        h_virial.data[idx_a]  += bond_virial;
        
        }
        
    m_pdata->release();
    
    if (m_prof) m_prof->pop(m_bond_data->getNumBonds() * (3+9+5+13+2+16), 
                            m_pdata->getN() * 5 * sizeof(Scalar) + m_bond_data->getNumBonds() * 
                            ( (4) * sizeof(unsigned int) + (6+2+20) ) );
    }

void export_FENEBondForceCompute()
    {
    class_<FENEBondForceCompute, boost::shared_ptr<FENEBondForceCompute>, bases<ForceCompute>, boost::noncopyable >
    ("FENEBondForceCompute", init< boost::shared_ptr<SystemDefinition>, const std::string& >())
    .def("setParams", &FENEBondForceCompute::setParams)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

