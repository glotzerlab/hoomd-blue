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
#pragma warning( disable : 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "HarmonicImproperForceCompute.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <math.h>

using namespace std;

//! SMALL a relatively small number
#define SMALL 0.001f

/*! \file HarmonicImproperForceCompute.cc
    \brief Contains code for the HarmonicImproperForceCompute class
*/

/*! \param sysdef System to compute forces on
    \post Memory is allocated, and forces are zeroed.
*/
HarmonicImproperForceCompute::HarmonicImproperForceCompute(boost::shared_ptr<SystemDefinition> sysdef) 
    : ForceCompute(sysdef), m_K(NULL), m_chi(NULL)
    {
    // access the improper data for later use
    m_improper_data = m_sysdef->getImproperData();
    
    // check for some silly errors a user could make
    if (m_improper_data->getNDihedralTypes() == 0)
        {
        cout << endl << "***Error! No improper types specified" << endl << endl;
        throw runtime_error("Error initializing HarmonicImproperForceCompute");
        }
        
    // allocate the parameters
    m_K = GPUArray<Scalar>(m_improper_data->getNDihedralTypes(),exec_conf);
    m_chi = GPUArray<Scalar>(m_improper_data->getNDihedralTypes(),exec_conf);
    
    }

HarmonicImproperForceCompute::~HarmonicImproperForceCompute()
    {
    delete[] m_K;
    delete[] m_chi;
    m_K = NULL;
    m_chi = NULL;
    }

/*! \param type Type of the improper to set parameters for
    \param K Stiffness parameter for the force computation.
    \param chi Equilibrium value of the dihedral angle.

    Sets parameters for the potential of a particular improper type
*/
void HarmonicImproperForceCompute::setParams(unsigned int type, Scalar K, Scalar chi)
    {
    // make sure the type is valid
    if (type >= m_improper_data->getNDihedralTypes())
        {
        cout << endl << "***Error! Invalid improper type specified" << endl << endl;
        throw runtime_error("Error setting parameters in HarmonicImproperForceCompute");
        }
        
    m_K[type] = K;
    m_chi[type] = chi;
    
    // check for some silly errors a user could make
    if (K <= 0)
        cout << "***Warning! K <= 0 specified for harmonic improper" << endl;
    if (chi <= 0)
        cout << "***Warning! Chi <= 0 specified for harmonic improper" << endl;
    }

/*! ImproperForceCompute provides
    - \c improper_harmonic_energy
*/
std::vector< std::string > HarmonicImproperForceCompute::getProvidedLogQuantities()
    {
    vector<string> list;
    list.push_back("improper_harmonic_energy");
    return list;
    }

/*! \param quantity Name of the quantity to get the log value of
    \param timestep Current time step of the simulation
*/
Scalar HarmonicImproperForceCompute::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == string("improper_harmonic_energy"))
        {
        compute(timestep);
        return calcEnergySum();
        }
    else
        {
        cerr << endl << "***Error! " 
             << quantity << " is not a valid log quantity for ImproperForceCompute" 
             << endl << endl;
        throw runtime_error("Error getting log value");
        }
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void HarmonicImproperForceCompute::computeForces(unsigned int timestep)
    {
    if (m_prof) m_prof->push("Harmonic Improper");
    
    assert(m_pdata);
    // access the particle data arrays
    ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();

		// need to start from a zero force
    // MEM TRANSFER: 5*N Scalars
		m_force.memclear();
		m_virial.memclear();
		
		ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
		ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::overwrite);

   // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
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
    
    // for each of the impropers
    const unsigned int size = (unsigned int)m_improper_data->getNumDihedrals();
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the improper
        const Dihedral& improper = m_improper_data->getDihedral(i);
        assert(improper.a < m_pdata->getN());
        assert(improper.b < m_pdata->getN());
        assert(improper.c < m_pdata->getN());
        assert(improper.d < m_pdata->getN());
        
        // transform a, b, and c into indicies into the particle data arrays
        // MEM TRANSFER: 6 ints
        unsigned int idx_a = arrays.rtag[improper.a];
        unsigned int idx_b = arrays.rtag[improper.b];
        unsigned int idx_c = arrays.rtag[improper.c];
        unsigned int idx_d = arrays.rtag[improper.d];
        assert(idx_a < m_pdata->getN());
        assert(idx_b < m_pdata->getN());
        assert(idx_c < m_pdata->getN());
        assert(idx_d < m_pdata->getN());
        
        // calculate d\vec{r}
        // MEM_TRANSFER: 18 Scalars / FLOPS 9
        Scalar dxab = arrays.x[idx_a] - arrays.x[idx_b];
        Scalar dyab = arrays.y[idx_a] - arrays.y[idx_b];
        Scalar dzab = arrays.z[idx_a] - arrays.z[idx_b];
        
        Scalar dxcb = arrays.x[idx_c] - arrays.x[idx_b];
        Scalar dycb = arrays.y[idx_c] - arrays.y[idx_b];
        Scalar dzcb = arrays.z[idx_c] - arrays.z[idx_b];
        
        Scalar dxdc = arrays.x[idx_d] - arrays.x[idx_c];
        Scalar dydc = arrays.y[idx_d] - arrays.y[idx_c];
        Scalar dzdc = arrays.z[idx_d] - arrays.z[idx_c];
        
        // if the a->b vector crosses the box, pull it back
        // (total FLOPS: 27 (worst case: first branch is missed, the 2nd is taken and the add is done, for each))
        if (dxab >= Lx2)
            dxab -= Lx;
        else if (dxab < -Lx2)
            dxab += Lx;
            
        if (dyab >= Ly2)
            dyab -= Ly;
        else if (dyab < -Ly2)
            dyab += Ly;
            
        if (dzab >= Lz2)
            dzab -= Lz;
        else if (dzab < -Lz2)
            dzab += Lz;
            
        // if the b<-c vector crosses the box, pull it back
        if (dxcb >= Lx2)
            dxcb -= Lx;
        else if (dxcb < -Lx2)
            dxcb += Lx;
            
        if (dycb >= Ly2)
            dycb -= Ly;
        else if (dycb < -Ly2)
            dycb += Ly;
            
        if (dzcb >= Lz2)
            dzcb -= Lz;
        else if (dzcb < -Lz2)
            dzcb += Lz;
            
        // if the d<-c vector crosses the box, pull it back
        if (dxdc >= Lx2)
            dxdc -= Lx;
        else if (dxdc < -Lx2)
            dxdc += Lx;
            
        if (dydc >= Ly2)
            dydc -= Ly;
        else if (dydc < -Ly2)
            dydc += Ly;
            
        if (dzdc >= Lz2)
            dzdc -= Lz;
        else if (dzdc < -Lz2)
            dzdc += Lz;
            
            
        // sanity check
        assert((dxab >= box.xlo && dxab < box.xhi) && (dxcb >= box.xlo && dxcb < box.xhi) && (dxdc >= box.xlo && dxdc < box.xhi));
        assert((dyab >= box.ylo && dyab < box.yhi) && (dycb >= box.ylo && dycb < box.yhi) && (dydc >= box.ylo && dydc < box.yhi));
        assert((dzab >= box.zlo && dzab < box.zhi) && (dzcb >= box.zlo && dzcb < box.zhi) && (dzdc >= box.zlo && dzdc < box.zhi));
        
        Scalar dxcbm = -dxcb;
        Scalar dycbm = -dycb;
        Scalar dzcbm = -dzcb;
        
        // if the d->c vector crosses the box, pull it back
        if (dxcbm >= Lx2)
            dxcbm -= Lx;
        else if (dxcbm < -Lx2)
            dxcbm += Lx;
            
        if (dycbm >= Ly2)
            dycbm -= Ly;
        else if (dycbm < -Ly2)
            dycbm += Ly;
            
        if (dzcbm >= Lz2)
            dzcbm -= Lz;
        else if (dzcbm < -Lz2)
            dzcbm += Lz;
            
            
        Scalar ss1 = 1.0 / (dxab*dxab + dyab*dyab + dzab*dzab);
        Scalar ss2 = 1.0 / (dxcb*dxcb + dycb*dycb + dzcb*dzcb);
        Scalar ss3 = 1.0 / (dxdc*dxdc + dydc*dydc + dzdc*dzdc);
        
        Scalar r1 = sqrt(ss1);
        Scalar r2 = sqrt(ss2);
        Scalar r3 = sqrt(ss3);
        
        // Cosine and Sin of the angle between the planes
        Scalar c0 = (dxab*dxdc + dyab*dydc + dzab*dzdc)* r1 * r3;
        Scalar c1 = (dxab*dxcb + dyab*dycb + dzab*dzcb)* r1 * r2;
        Scalar c2 = -(dxdc*dxcb + dydc*dycb + dzdc*dzcb)* r3 * r2;
        
        Scalar s1 = 1.0 - c1*c1;
        if (s1 < SMALL) s1 = SMALL;
        s1 = 1.0 / s1;
        
        Scalar s2 = 1.0 - c2*c2;
        if (s2 < SMALL) s2 = SMALL;
        s2 = 1.0 / s2;
        
        Scalar s12 = sqrt(s1*s2);
        Scalar c = (c1*c2 + c0) * s12;
        
        if (c > 1.0) c = 1.0;
        if (c < -1.0) c = -1.0;
        
        Scalar s = sqrt(1.0 - c*c);
        if (s < SMALL) s = SMALL;
        
        Scalar domega = acos(c) - m_chi[improper.type];
        Scalar a = m_K[improper.type] * domega;
        
        // calculate the energy, 1/4th for each atom
        //Scalar improper_eng = Scalar(0.25)*a*domega;
        Scalar improper_eng = Scalar(0.125)*a*domega; // the .125 term is 1/2 * 1/4
        //a = -a * 2.0/s;
        a = -a / s; // the missing 2.0 factor is to ensure K/2 is factored in for the forces
        c = c * a;
        
        s12 = s12 * a;
        Scalar a11 = c*ss1*s1;
        Scalar a22 = -ss2 * (2.0*c0*s12 - c*(s1+s2));
        Scalar a33 = c*ss3*s2;
        
        Scalar a12 = -r1*r2*(c1*c*s1 + c2*s12);
        Scalar a13 = -r1*r3*s12;
        Scalar a23 = r2*r3*(c2*c*s2 + c1*s12);
        
        Scalar sx2  = a22*dxcb + a23*dxdc + a12*dxab;
        Scalar sy2  = a22*dycb + a23*dydc + a12*dyab;
        Scalar sz2  = a22*dzcb + a23*dzdc + a12*dzab;
        
        // calculate the forces for each particle
        Scalar ffax = a12*dxcb + a13*dxdc + a11*dxab;
        Scalar ffay = a12*dycb + a13*dydc + a11*dyab;
        Scalar ffaz = a12*dzcb + a13*dzdc + a11*dzab;
        
        Scalar ffbx = -sx2 - ffax;
        Scalar ffby = -sy2 - ffay;
        Scalar ffbz = -sz2 - ffaz;
        
        Scalar ffdx = a23*dxcb + a33*dxdc + a13*dxab;
        Scalar ffdy = a23*dycb + a33*dydc + a13*dyab;
        Scalar ffdz = a23*dzcb + a33*dzdc + a13*dzab;
        
        Scalar ffcx = sx2 - ffdx;
        Scalar ffcy = sy2 - ffdy;
        Scalar ffcz = sz2 - ffdz;
        
        // and calculate the virial
        Scalar vx = dxab*ffax + dxcb*ffcx + (dxdc+dxcb)*ffdx;
        Scalar vy = dyab*ffay + dycb*ffcy + (dydc+dycb)*ffdy;
        Scalar vz = dzab*ffaz + dzcb*ffcz + (dzdc+dzcb)*ffdz;
        
        // compute 1/4 of the virial, 1/4 for each atom in the improper
        Scalar improper_virial = Scalar(1.0/12.0)*(vx + vy + vz);
        
        
        // accumulate the forces
				h_force.data[idx_a].x + =ffax; 
				h_force.data[idx_a].y += ffay; 
				h_force.data[idx_a].z += ffaz; 
				h_force.data[idx_a].w += improper_eng; 
				h_virial.data[idx_a]  += improper_virial; 

				h_force.data[idx_b].x + =ffbx; 
				h_force.data[idx_b].y += ffby; 
				h_force.data[idx_b].z += ffbz; 
				h_force.data[idx_b].w += improper_eng; 
				h_virial.data[idx_b]  += improper_virial; 

				h_force.data[idx_c].x + =ffcx; 
				h_force.data[idx_c].y += ffcy; 
				h_force.data[idx_c].z += ffcz; 
				h_force.data[idx_c].w += improper_eng; 
				h_virial.data[idx_c]  += improper_virial; 

				h_force.data[idx_d].x + =ffdx; 
				h_force.data[idx_d].y += ffdy; 
				h_force.data[idx_d].z += ffdz; 
				h_force.data[idx_d].w += improper_eng; 
				h_virial.data[idx_d]  += improper_virial; 
				}
        
    m_pdata->release();
   
    if (m_prof) m_prof->pop();
    }

void export_HarmonicImproperForceCompute()
    {
    class_<HarmonicImproperForceCompute, boost::shared_ptr<HarmonicImproperForceCompute>, bases<ForceCompute>, boost::noncopyable >
    ("HarmonicImproperForceCompute", init< boost::shared_ptr<SystemDefinition> >())
    .def("setParams", &HarmonicImproperForceCompute::setParams)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

