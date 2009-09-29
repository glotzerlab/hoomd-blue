/*
Highly Optimized Object-Oriented Molecular Dynamics (HOOMD) Open
Source Software License
Copyright (c) 2008 Ames Laboratory Iowa State University
All rights reserved.

Redistribution and use of HOOMD, in source and binary forms, with or
without modification, are permitted, provided that the following
conditions are met:

* Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names HOOMD's
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

Disclaimer

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDER AND
CONTRIBUTORS ``AS IS''  AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS  BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
*/

// $Id$
// $URL$
// Maintainer: dnlebard

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "HarmonicAngleForceCompute.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <math.h>

using namespace std;

//! SMALL a relatively small number
#define SMALL 0.001f

/*! \file HarmonicAngleForceCompute.cc
    \brief Contains code for the HarmonicAngleForceCompute class
*/

/*! \param sysdef System to compute forces on
    \post Memory is allocated, and forces are zeroed.
*/
HarmonicAngleForceCompute::HarmonicAngleForceCompute(boost::shared_ptr<SystemDefinition> sysdef) :  ForceCompute(sysdef),
        m_K(NULL), m_t_0(NULL)
    {
    // access the angle data for later use
    m_angle_data = m_sysdef->getAngleData();
    
    // check for some silly errors a user could make
    if (m_angle_data->getNAngleTypes() == 0)
        {
        cout << endl << "***Error! No angle types specified" << endl << endl;
        throw runtime_error("Error initializing HarmonicAngleForceCompute");
        }
        
    // allocate the parameters
    m_K = new Scalar[m_angle_data->getNAngleTypes()];
    m_t_0 = new Scalar[m_angle_data->getNAngleTypes()];
    
    // zero parameters
    memset(m_K, 0, sizeof(Scalar) * m_angle_data->getNAngleTypes());
    memset(m_t_0, 0, sizeof(Scalar) * m_angle_data->getNAngleTypes());
    }

HarmonicAngleForceCompute::~HarmonicAngleForceCompute()
    {
    delete[] m_K;
    delete[] m_t_0;
    m_K = NULL;
    m_t_0 = NULL;
    }

/*! \param type Type of the angle to set parameters for
    \param K Stiffness parameter for the force computation
    \param t_0 Equilibrium angle in radians for the force computation

    Sets parameters for the potential of a particular angle type
*/
void HarmonicAngleForceCompute::setParams(unsigned int type, Scalar K, Scalar t_0)
    {
    // make sure the type is valid
    if (type >= m_angle_data->getNAngleTypes())
        {
        cout << endl << "***Error! Invalid angle type specified" << endl << endl;
        throw runtime_error("Error setting parameters in HarmonicAngleForceCompute");
        }
        
    m_K[type] = K;
    m_t_0[type] = t_0;
    
    // check for some silly errors a user could make
    if (K <= 0)
        cout << "***Warning! K <= 0 specified for harmonic angle" << endl;
    if (t_0 <= 0)
        cout << "***Warning! t_0 <= 0 specified for harmonic angle" << endl;
    }

/*! AngleForceCompute provides
    - \c angle_harmonic_energy
*/
std::vector< std::string > HarmonicAngleForceCompute::getProvidedLogQuantities()
    {
    vector<string> list;
    list.push_back("angle_harmonic_energy");
    return list;
    }

/*! \param quantity Name of the quantity to get the log value of
    \param timestep Current time step of the simulation
*/
Scalar HarmonicAngleForceCompute::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == string("angle_harmonic_energy"))
        {
        compute(timestep);
        return calcEnergySum();
        }
    else
        {
        cerr << endl << "***Error! " << quantity << " is not a valid log quantity for AngleForceCompute" << endl << endl;
        throw runtime_error("Error getting log value");
        }
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void HarmonicAngleForceCompute::computeForces(unsigned int timestep)
    {
    if (m_prof) m_prof->push("Harmonic Angle");
    
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
    
    // for each of the angles
    const unsigned int size = (unsigned int)m_angle_data->getNumAngles();
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the angle
        const Angle& angle = m_angle_data->getAngle(i);
        assert(angle.a < m_pdata->getN());
        assert(angle.b < m_pdata->getN());
        assert(angle.c < m_pdata->getN());
        
        // transform a, b, and c into indicies into the particle data arrays
        // MEM TRANSFER: 6 ints
        unsigned int idx_a = arrays.rtag[angle.a];
        unsigned int idx_b = arrays.rtag[angle.b];
        unsigned int idx_c = arrays.rtag[angle.c];
        assert(idx_a < m_pdata->getN());
        assert(idx_b < m_pdata->getN());
        assert(idx_c < m_pdata->getN());
        
        // calculate d\vec{r}
        // MEM_TRANSFER: 18 Scalars / FLOPS 9
        Scalar dxab = arrays.x[idx_a] - arrays.x[idx_b];
        Scalar dyab = arrays.y[idx_a] - arrays.y[idx_b];
        Scalar dzab = arrays.z[idx_a] - arrays.z[idx_b];
        
        Scalar dxcb = arrays.x[idx_c] - arrays.x[idx_b];
        Scalar dycb = arrays.y[idx_c] - arrays.y[idx_b];
        Scalar dzcb = arrays.z[idx_c] - arrays.z[idx_b];
        
        Scalar dxac = arrays.x[idx_a] - arrays.x[idx_c]; // used for the 1-3 JL interaction
        Scalar dyac = arrays.y[idx_a] - arrays.y[idx_c];
        Scalar dzac = arrays.z[idx_a] - arrays.z[idx_c];
        
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
            
        // if the b->c vector crosses the box, pull it back
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
            
        // if the a->c vector crosses the box, pull it back
        if (dxac >= Lx2)
            dxac -= Lx;
        else if (dxac < -Lx2)
            dxac += Lx;
            
        if (dyac >= Ly2)
            dyac -= Ly;
        else if (dyac < -Ly2)
            dyac += Ly;
            
        if (dzac >= Lz2)
            dzac -= Lz;
        else if (dzac < -Lz2)
            dzac += Lz;
            
            
        // sanity check
        assert((dxab >= box.xlo && dxab < box.xhi) && (dxcb >= box.xlo && dxcb < box.xhi) && (dxac >= box.xlo && dxac < box.xhi));
        assert((dyab >= box.ylo && dyab < box.yhi) && (dycb >= box.ylo && dycb < box.yhi) && (dyac >= box.ylo && dyac < box.yhi));
        assert((dzab >= box.zlo && dzab < box.zhi) && (dzcb >= box.zlo && dzcb < box.zhi) && (dzac >= box.zlo && dzac < box.zhi));
        
        // on paper, the formula turns out to be: F = K*\vec{r} * (r_0/r - 1)
        // FLOPS: 14 / MEM TRANSFER: 2 Scalars
        
        
        // FLOPS: 42 / MEM TRANSFER: 6 Scalars
        Scalar rsqab = dxab*dxab+dyab*dyab+dzab*dzab;
        Scalar rab = sqrt(rsqab);
        Scalar rsqcb = dxcb*dxcb+dycb*dycb+dzcb*dzcb;
        Scalar rcb = sqrt(rsqcb);
        
        Scalar c_abbc = dxab*dxcb+dyab*dycb+dzab*dzcb;
        c_abbc /= rab*rcb;
        
        if (c_abbc > 1.0) c_abbc = 1.0;
        if (c_abbc < -1.0) c_abbc = -1.0;
        
        Scalar s_abbc = sqrt(1.0 - c_abbc*c_abbc);
        if (s_abbc < SMALL) s_abbc = SMALL;
        s_abbc = 1.0/s_abbc;
        
        // actually calculate the force
        Scalar dth = acos(c_abbc) - m_t_0[angle.type];
        Scalar tk = m_K[angle.type]*dth;
        
        Scalar a = -1.0 * tk * s_abbc;
        Scalar a11 = a*c_abbc/rsqab;
        Scalar a12 = -a / (rab*rcb);
        Scalar a22 = a*c_abbc / rsqcb;
        
        Scalar fab[3], fcb[3];
        
        fab[0] = a11*dxab + a12*dxcb;
        fab[1] = a11*dyab + a12*dycb;
        fab[2] = a11*dzab + a12*dzcb;
        
        fcb[0] = a22*dxcb + a12*dxab;
        fcb[1] = a22*dycb + a12*dyab;
        fcb[2] = a22*dzcb + a12*dzab;
        
        // compute 1/3 of the energy, 1/3 for each atom in the angle
        Scalar angle_eng = (tk*dth)*Scalar(1.0/6.0);
        
        // do we really need a virial here for harmonic angles?
        // ... if not, this may be wrong...
        Scalar vx = dxab*fab[0] + dxcb*fcb[0];
        Scalar vy = dyab*fab[1] + dycb*fcb[1];
        Scalar vz = dzab*fab[2] + dzcb*fcb[2];
        
        Scalar angle_virial = Scalar(1.0/6.0)*(vx + vy + vz);
        
        // Now, apply the force to each individual atom a,b,c, and accumlate the energy/virial
        m_fx[idx_a] += fab[0];
        m_fy[idx_a] += fab[1];
        m_fz[idx_a] += fab[2];
        m_pe[idx_a] += angle_eng;
        m_virial[idx_a] += angle_virial;
        
        m_fx[idx_b] -= fab[0] + fcb[0];
        m_fy[idx_b] -= fab[1] + fcb[1];
        m_fz[idx_b] -= fab[2] + fcb[2];
        m_pe[idx_b] += angle_eng;
        m_virial[idx_b] += angle_virial;
        
        m_fx[idx_c] += fcb[0];
        m_fy[idx_c] += fcb[1];
        m_fz[idx_c] += fcb[2];
        m_pe[idx_c] += angle_eng;
        m_virial[idx_c] += angle_virial;
        }
        
    m_pdata->release();
    
#ifdef ENABLE_CUDA
    // the data is now only up to date on the CPU
    m_data_location = cpu;
#endif
    
    if (m_prof) m_prof->pop();
    }

void export_HarmonicAngleForceCompute()
    {
    class_<HarmonicAngleForceCompute, boost::shared_ptr<HarmonicAngleForceCompute>, bases<ForceCompute>, boost::noncopyable >
    ("HarmonicAngleForceCompute", init< boost::shared_ptr<SystemDefinition> >())
    .def("setParams", &HarmonicAngleForceCompute::setParams)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif
