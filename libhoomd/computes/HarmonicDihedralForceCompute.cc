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

#include "HarmonicDihedralForceCompute.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <math.h>

using namespace std;

//! SMALL a relatively small number
#define SMALL 0.001f

/*! \file HarmonicDihedralForceCompute.cc
    \brief Contains code for the HarmonicDihedralForceCompute class
*/

/*! \param sysdef System to compute forces on
    \post Memory is allocated, and forces are zeroed.
*/
HarmonicDihedralForceCompute::HarmonicDihedralForceCompute(boost::shared_ptr<SystemDefinition> sysdef) 
    : ForceCompute(sysdef), m_K(NULL), m_sign(NULL), m_multi(NULL)
    {
    // access the dihedral data for later use
    m_dihedral_data = m_sysdef->getDihedralData();
    
    // check for some silly errors a user could make
    if (m_dihedral_data->getNDihedralTypes() == 0)
        {
        cout << endl << "***Error! No dihedral types specified" << endl << endl;
        throw runtime_error("Error initializing HarmonicDihedralForceCompute");
        }
        
    // allocate the parameters
    m_K = GPUArray<Scalar>(m_dihedral_data->getNDihedralTypes(),exec_conf);
    m_sign = GPUArray<Scalar>(m_dihedral_data->getNDihedralTypes(),exec_conf);
    m_multi = GPUArray<Scalar>(m_dihedral_data->getNDihedralTypes(),exec_conf);
    
    }

HarmonicDihedralForceCompute::~HarmonicDihedralForceCompute()
    {
    delete[] m_K;
    delete[] m_sign;
    delete[] m_multi;
    m_K = NULL;
    m_sign = NULL;
    m_multi = NULL;
    }

/*! \param type Type of the dihedral to set parameters for
    \param K Stiffness parameter for the force computation
    \param sign the sign of the cosign term
    \param multiplicity of the dihedral itself

    Sets parameters for the potential of a particular dihedral type
*/
void HarmonicDihedralForceCompute::setParams(unsigned int type, Scalar K, int sign, unsigned int multiplicity)
    {
    // make sure the type is valid
    if (type >= m_dihedral_data->getNDihedralTypes())
        {
        cout << endl << "***Error! Invalid dihedral type specified" << endl << endl;
        throw runtime_error("Error setting parameters in HarmonicDihedralForceCompute");
        }
        
    m_K[type] = K;
    m_sign[type] = (Scalar)sign;
    m_multi[type] = (Scalar)multiplicity;
    
    // check for some silly errors a user could make
    if (K <= 0)
        cout << "***Warning! K <= 0 specified for harmonic dihedral" << endl;
    if (sign != 1 && sign != -1)
        cout << "***Warning! a non unitary sign was specified for harmonic dihedral" << endl;
    }

/*! DihedralForceCompute provides
    - \c dihedral_harmonic_energy
*/
std::vector< std::string > HarmonicDihedralForceCompute::getProvidedLogQuantities()
    {
    vector<string> list;
    list.push_back("dihedral_harmonic_energy");
    return list;
    }

/*! \param quantity Name of the quantity to get the log value of
    \param timestep Current time step of the simulation
*/
Scalar HarmonicDihedralForceCompute::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == string("dihedral_harmonic_energy"))
        {
        compute(timestep);
        return calcEnergySum();
        }
    else
        {
        cerr << endl << "***Error! " 
             << quantity << " is not a valid log quantity for DihedralForceCompute" 
             << endl << endl;
        throw runtime_error("Error getting log value");
        }
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void HarmonicDihedralForceCompute::computeForces(unsigned int timestep)
    {
    if (m_prof) m_prof->push("Harmonic Dihedral");
    
    assert(m_pdata);
    // access the particle data arrays
    ParticleDataArraysConst arrays = m_pdata->acquireReadOnly();
    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(m_fx);
    assert(m_fy);
    assert(m_fz);
    assert(m_pe);
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
    
    // need to start from a zero force
    // MEM TRANSFER: 5*N Scalars
    memset((void*)m_fx, 0, sizeof(Scalar) * m_pdata->getN());
    memset((void*)m_fy, 0, sizeof(Scalar) * m_pdata->getN());
    memset((void*)m_fz, 0, sizeof(Scalar) * m_pdata->getN());
    memset((void*)m_pe, 0, sizeof(Scalar) * m_pdata->getN());
    memset((void*)m_virial, 0, sizeof(Scalar) * m_pdata->getN());
    
    // for each of the dihedrals
    const unsigned int size = (unsigned int)m_dihedral_data->getNumDihedrals();
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the dihedral
        const Dihedral& dihedral = m_dihedral_data->getDihedral(i);
        assert(dihedral.a < m_pdata->getN());
        assert(dihedral.b < m_pdata->getN());
        assert(dihedral.c < m_pdata->getN());
        assert(dihedral.d < m_pdata->getN());
        
        // transform a, b, and c into indicies into the particle data arrays
        // MEM TRANSFER: 6 ints
        unsigned int idx_a = arrays.rtag[dihedral.a];
        unsigned int idx_b = arrays.rtag[dihedral.b];
        unsigned int idx_c = arrays.rtag[dihedral.c];
        unsigned int idx_d = arrays.rtag[dihedral.d];
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
            
        Scalar aax = dyab*dzcbm - dzab*dycbm;
        Scalar aay = dzab*dxcbm - dxab*dzcbm;
        Scalar aaz = dxab*dycbm - dyab*dxcbm;
        
        Scalar bbx = dydc*dzcbm - dzdc*dycbm;
        Scalar bby = dzdc*dxcbm - dxdc*dzcbm;
        Scalar bbz = dxdc*dycbm - dydc*dxcbm;
        
        Scalar raasq = aax*aax + aay*aay + aaz*aaz;
        Scalar rbbsq = bbx*bbx + bby*bby + bbz*bbz;
        Scalar rgsq = dxcbm*dxcbm + dycbm*dycbm + dzcbm*dzcbm;
        Scalar rg = sqrt(rgsq);
        
        Scalar rginv, raa2inv, rbb2inv;
        rginv = raa2inv = rbb2inv = 0.0f;
        if (rg > 0.0f) rginv = 1.0f/rg;
        if (raasq > 0.0f) raa2inv = 1.0f/raasq;
        if (rbbsq > 0.0f) rbb2inv = 1.0f/rbbsq;
        Scalar rabinv = sqrt(raa2inv*rbb2inv);
        
        Scalar c_abcd = (aax*bbx + aay*bby + aaz*bbz)*rabinv;
        Scalar s_abcd = rg*rabinv*(aax*dxdc + aay*dydc + aaz*dzdc);
        
        if (c_abcd > 1.0) c_abcd = 1.0;
        if (c_abcd < -1.0) c_abcd = -1.0;
        
        int multi = (int)m_multi[dihedral.type];
        Scalar p = 1.0f;
        Scalar dfab = 0.0f;
        Scalar ddfab;
        
        for (int j = 0; j < multi; j++)
            {
            ddfab = p*c_abcd - dfab*s_abcd;
            dfab = p*s_abcd + dfab*c_abcd;
            p = ddfab;
            }
            
/////////////////////////
// FROM LAMMPS: sin_shift is always 0... so dropping all sin_shift terms!!!!
/////////////////////////

        Scalar sign = m_sign[dihedral.type];
        p = p*sign;
        dfab = dfab*sign;
        dfab *= (Scalar)-multi;
        p += 1.0f;
        
        if (multi == 0)
            {
            p =  1.0f + sign;
            dfab = 0.0f;
            }
            
            
        Scalar fg = dxab*dxcbm + dyab*dycbm + dzab*dzcbm;
        Scalar hg = dxdc*dxcbm + dydc*dycbm + dzdc*dzcbm;
        
        Scalar fga = fg*raa2inv*rginv;
        Scalar hgb = hg*rbb2inv*rginv;
        Scalar gaa = -raa2inv*rg;
        Scalar gbb = rbb2inv*rg;
        
        Scalar dtfx = gaa*aax;
        Scalar dtfy = gaa*aay;
        Scalar dtfz = gaa*aaz;
        Scalar dtgx = fga*aax - hgb*bbx;
        Scalar dtgy = fga*aay - hgb*bby;
        Scalar dtgz = fga*aaz - hgb*bbz;
        Scalar dthx = gbb*bbx;
        Scalar dthy = gbb*bby;
        Scalar dthz = gbb*bbz;
        
//      Scalar df = -m_K[dihedral.type] * dfab;
        Scalar df = -m_K[dihedral.type] * dfab * Scalar(0.500); // the 0.5 term is for 1/2K in the forces
        
        Scalar sx2 = df*dtgx;
        Scalar sy2 = df*dtgy;
        Scalar sz2 = df*dtgz;
        
        Scalar ffax = df*dtfx;
        Scalar ffay= df*dtfy;
        Scalar ffaz = df*dtfz;
        
        Scalar ffbx = sx2 - ffax;
        Scalar ffby = sy2 - ffay;
        Scalar ffbz = sz2 - ffaz;
        
        Scalar ffdx = df*dthx;
        Scalar ffdy = df*dthy;
        Scalar ffdz = df*dthz;
        
        Scalar ffcx = -sx2 - ffdx;
        Scalar ffcy = -sy2 - ffdy;
        Scalar ffcz = -sz2 - ffdz;
        
        // Now, apply the force to each individual atom a,b,c,d
        // and accumlate the energy/virial
        // compute 1/4 of the energy, 1/4 for each atom in the dihedral
        //Scalar dihedral_eng = p*m_K[dihedral.type]*Scalar(1.0/4.0);
        Scalar dihedral_eng = p*m_K[dihedral.type]*Scalar(0.125);  // the .125 term is (1/2)K * 1/4
        
        Scalar vx = dxab*ffax + dxcb*ffcx + (dxdc+dxcb)*ffdx;
        Scalar vy = dyab*ffay + dycb*ffcy + (dydc+dycb)*ffdy;
        Scalar vz = dzab*ffaz + dzcb*ffcz + (dzdc+dzcb)*ffdz;
        
        // compute 1/4 of the virial, 1/4 for each atom in the dihedral
        Scalar dihedral_virial = Scalar(1.0/12.0)*(vx + vy + vz);
        
        m_fx[idx_a] += ffax;
        m_fy[idx_a] += ffay;
        m_fz[idx_a] += ffaz;
        m_pe[idx_a] += dihedral_eng;
        m_virial[idx_a] += dihedral_virial;
        
        m_fx[idx_b] += ffbx;
        m_fy[idx_b] += ffby;
        m_fz[idx_b] += ffbz;
        m_pe[idx_b] += dihedral_eng;
        m_virial[idx_b] += dihedral_virial;
        
        m_fx[idx_c] += ffcx;
        m_fy[idx_c] += ffcy;
        m_fz[idx_c] += ffcz;
        m_pe[idx_c] += dihedral_eng;
        m_virial[idx_c] += dihedral_virial;
        
        m_fx[idx_d] += ffdx;
        m_fy[idx_d] += ffdy;
        m_fz[idx_d] += ffdz;
        m_pe[idx_d] += dihedral_eng;
        m_virial[idx_d] += dihedral_virial;
        
        // FLOPS: ?? / MEM TRANSFER: ?? Scalars
        }
        
    m_pdata->release();
    
#ifdef ENABLE_CUDA
    // the data is now only up to date on the CPU
    m_data_location = cpu;
#endif
    
    if (m_prof) m_prof->pop();
    }

void export_HarmonicDihedralForceCompute()
    {
    class_<HarmonicDihedralForceCompute, boost::shared_ptr<HarmonicDihedralForceCompute>, bases<ForceCompute>, boost::noncopyable >
    ("HarmonicDihedralForceCompute", init< boost::shared_ptr<SystemDefinition> >())
    .def("setParams", &HarmonicDihedralForceCompute::setParams)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

