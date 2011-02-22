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
// Maintainer: grva

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "DipoleDipoleForceCompute.h"
#include "QuaternionMath.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <math.h>

using namespace std;

/*! \file DipoleDipoleForceCompute.cc
    \brief Contains code for the DipoleDipoleForceCompute class
*/

/*! \param sysdef System to compute forces on
    \param log_suffix Name given to this instance of the harmonic bond
    \post Memory is allocated, and forces are zeroed.
*/
DipoleDipoleForceCompute::DipoleDipoleForceCompute(boost::shared_ptr<SystemDefinition> sysdef, const std::string& log_suffix) 
    : ForceCompute(sysdef), m_K(NULL), m_r_0(NULL)
    {
    // access the bond data for later use
    m_bond_data = m_sysdef->getBondData();
    m_log_name = std::string( "bond_harmonic_energy") + log_suffix;
    
    // check for some silly errors a user could make
    if (m_bond_data->getNBondTypes() == 0)
        {
        cout << endl << "***Error! No bond types specified" << endl << endl;
        throw runtime_error("Error initializing DipoleDipoleForceCompute");
        }
        
    // allocate the parameters
    m_K = new Scalar[m_bond_data->getNBondTypes()];
    m_r_0 = new Scalar[m_bond_data->getNBondTypes()];
    
    }

DipoleDipoleForceCompute::~DipoleDipoleForceCompute()
    {
    delete[] m_K;
    delete[] m_r_0;
    }

/*! \param type Type of the bond to set parameters for
    \param K Stiffness parameter for the force computation
    \param r_0 Equilibrium length for the force computation

    Sets parameters for the potential of a particular bond type
*/
void DipoleDipoleForceCompute::setParams(unsigned int type, Scalar K, Scalar r_0)
    {
    // make sure the type is valid
    if (type >= m_bond_data->getNBondTypes())
        {
        cout << endl << "***Error! Invalid bond type specified" << endl << endl;
        throw runtime_error("Error setting parameters in DipoleDipoleForceCompute");
        }
        
    m_K[type] = K;
    m_r_0[type] = r_0;
    
    // check for some silly errors a user could make
    if (K <= 0)
        cout << "***Warning! K <= 0 specified for harmonic bond" << endl;
    if (r_0 < 0)
        cout << "***Warning! r_0 <= 0 specified for harmonic bond" << endl;
    }

/*! BondForceCompute provides
    - \c harmonic_energy
*/
std::vector< std::string > DipoleDipoleForceCompute::getProvidedLogQuantities()
    {
    vector<string> list;
    list.push_back(m_log_name);
    return list;
    }

/*! \param quantity Name of the quantity to get the log value of
    \param timestep Current time step of the simulation
*/
Scalar DipoleDipoleForceCompute::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == m_log_name)
        {
        compute(timestep);
        return calcEnergySum();
        }
    else
        {
        cerr << endl << "***Error! " << quantity << " is not a valid log quantity for BondForceCompute" << endl << endl;
        throw runtime_error("Error getting log value");
        }
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void DipoleDipoleForceCompute::computeForces(unsigned int timestep)
    {
    if (m_prof) m_prof->push("Harmonic");
    
    assert(m_pdata);

     //Accquire necessary arrays        
    // access the particle data arrays
    ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
        
    // get write handles to force, torque and virial
    ArrayHandle<Scalar4> h_torque(m_torque,access_location::host,access_mode::overwrite);
    ArrayHandle<Scalar4> h_force(m_force,access_location::host,access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host,access_mode::overwrite);

    // get read handle to orientation
    ArrayHandle<Scalar4> h_orientation(m_pdata->getOrientationArray(),access_location::host,access_mode::read);

    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_torque.data);
    assert(h_force.data);
    assert(h_virial.data);
    assert(h_orientation.data);
    assert(arrays.x);
    assert(arrays.y);
    assert(arrays.z);
    
    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();
    // sanity check
    assert(box.xhi > box.xlo && box.yhi > box.ylo && box.zhi > box.zlo);
    
    // precalculate box lenghts
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
        // MEM TRANSFER: 4 ints
        unsigned int idx_a = arrays.rtag[bond.a];
        unsigned int idx_b = arrays.rtag[bond.b];
        assert(idx_a < m_pdata->getN());
        assert(idx_b < m_pdata->getN());
        
        // calculate d\vec{r}
        // MEM_TRANSFER: 6 Scalars / FLOPS 3
        Scalar dx = arrays.x[idx_b] - arrays.x[idx_a];
        Scalar dy = arrays.y[idx_b] - arrays.y[idx_a];
        Scalar dz = arrays.z[idx_b] - arrays.z[idx_a];
        
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
        
        // on paper, the formula turns out to be: F = K*\vec{r} * (r_0/r - 1)
        // FLOPS: 14 / MEM TRANSFER: 2 Scalars
        Scalar rsq = dx*dx+dy*dy+dz*dz;
        Scalar r = sqrt(rsq);

	// unit vector between the particles
	Scalar3 rhat = { dx/r , dy/r, dz/r };

	// compute current dipole moments
	Scalar4 q_a = h_orientation.data[idx_a];
	Scalar4 q_b = h_orientation.data[idx_b];

	// to be filled by conjugates
	Scalar4 q_a_H;
	Scalar4 q_b_H;

	// conjugation
	quatconj(q_a,q_a_H);
	quatconj(q_b,q_b_H);

	// base frame dipole orientation
	Scalar4 base = {0,0,1,0};

	// temporary
	Scalar4 temp;

	// moments
	Scalar4 p_a;
	Scalar4 p_b;

	// do the rotation for a
	quatvec(q_a,base,temp);
	quatquat(temp,q_a_H,p_a);

	// do the rotation for b
	quatvec(q_b,base,temp);
	quatquat(temp,q_b_H,p_b);

	// scalar products
	Scalar p_adotr = p_a.y*rhat.x+p_a.z*rhat.y+p_a.w*rhat.z;
	Scalar p_bdotr = p_b.y*rhat.x+p_b.z*rhat.y+p_b.w*rhat.z;
	Scalar pdotp = p_a.y*p_b.y+p_a.z*p_b.z+p_a.w*p_b.w;

	// dipole-dipole energy
        Scalar bond_eng = (pdotp-3*p_adotr*p_bdotr)/rsq/r;
        
	// common parts of force
	Scalar fact0 = -3.0/rsq/rsq*(pdotp-5*p_adotr*p_bdotr);
	Scalar fact1 = -3.0/rsq/rsq*p_adotr;
	Scalar fact2 = -3.0/rsq/rsq*p_bdotr;

        // calculate the virial
        Scalar bond_virial=Scalar(1.0/6.0)*(-3.0*pdotp+9*p_adotr*p_bdotr)/rsq/r;

        // add the force to the particles
        h_force.data[idx_b].x -= fact0*rhat.x + fact1 * p_b.y + fact2 * p_a.y;
        h_force.data[idx_b].y -= fact0*rhat.y + fact1 * p_b.z + fact2 * p_a.z;
        h_force.data[idx_b].z -= fact0*rhat.z + fact1 * p_b.w + fact2 * p_a.w;
        h_force.data[idx_b].w += bond_eng;
        h_virial.data[idx_b] += bond_virial;
        
        h_force.data[idx_a].x += fact0*rhat.x + fact1 * p_b.y + fact2 * p_a.y;
        h_force.data[idx_a].y += fact0*rhat.y + fact1 * p_b.z + fact2 * p_a.z;
        h_force.data[idx_a].z += fact0*rhat.z + fact1 * p_b.w + fact2 * p_a.w;
        h_force.data[idx_a].w += bond_eng;
        h_virial.data[idx_a] += bond_virial;

	// compute the torques:
	// first get field as seen by particles
        Scalar3 field_a = {(p_b.y-3*rhat.x*p_bdotr)/rsq/r,
                           (p_b.z-3*rhat.y*p_bdotr)/rsq/r,
                           (p_b.w-3*rhat.z*p_bdotr)/rsq/r};
        Scalar3 field_b = {(p_a.y-3*rhat.x*p_adotr)/rsq/r,
                           (p_a.z-3*rhat.y*p_adotr)/rsq/r,
                           (p_a.w-3*rhat.z*p_adotr)/rsq/r};

	// then get torques
	h_torque.data[idx_a].x = field_a.y*p_a.w-field_a.z*p_a.z;
	h_torque.data[idx_a].y = field_a.z*p_a.y-field_a.x*p_a.w;
	h_torque.data[idx_a].z = field_a.x*p_a.z-field_a.y*p_a.y;
	h_torque.data[idx_a].w = Scalar(0);

	h_torque.data[idx_b].x = field_b.y*p_b.w-field_b.z*p_b.z;
	h_torque.data[idx_b].y = field_b.z*p_b.y-field_b.x*p_b.w;
	h_torque.data[idx_b].z = field_b.x*p_b.z-field_b.y*p_b.y;
	h_torque.data[idx_b].w = Scalar(0);
        }
        
    m_pdata->release();
       
    }

void export_DipoleDipoleForceCompute()
    {
    class_<DipoleDipoleForceCompute,boost::shared_ptr<DipoleDipoleForceCompute>,
	    bases<ForceCompute>, boost::noncopyable >
    ("DipoleDipoleForceCompute", init< boost::shared_ptr<SystemDefinition>,
     const std::string& >())
    .def("setParams", &DipoleDipoleForceCompute::setParams)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

