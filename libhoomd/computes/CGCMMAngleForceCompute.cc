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

// Maintainer: dnlebard

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "CGCMMAngleForceCompute.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <math.h>

using namespace std;

// \param SMALL a relatively small number
#define SMALL Scalar(0.001)

/*! \file CGCMMAngleForceCompute.cc
    \brief Contains code for the CGCMMAngleForceCompute class
*/

/*! \param sysdef System to compute forces on
    \post Memory is allocated, and forces are zeroed.
*/
CGCMMAngleForceCompute::CGCMMAngleForceCompute(boost::shared_ptr<SystemDefinition> sysdef) 
    : ForceCompute(sysdef), m_K(NULL), m_t_0(NULL), m_eps(NULL), m_sigma(NULL), m_rcut(NULL), m_cg_type(NULL)
    {
    m_exec_conf->msg->notice(5) << "Constructing CGCMMAngleForceCompute" << endl;

    // access the angle data for later use
    m_CGCMMAngle_data = m_sysdef->getAngleData();
    
    // check for some silly errors a user could make
    if (m_CGCMMAngle_data->getNAngleTypes() == 0)
        {
        m_exec_conf->msg->error() << "angle.cgcmm: No angle types specified" << endl;
        throw runtime_error("Error initializing CGCMMAngleForceCompute");
        }
        
    // allocate the parameters
    m_K = new Scalar[m_CGCMMAngle_data->getNAngleTypes()];
    m_t_0 = new Scalar[m_CGCMMAngle_data->getNAngleTypes()];
    m_eps =  new Scalar[m_CGCMMAngle_data->getNAngleTypes()];
    m_sigma = new Scalar[m_CGCMMAngle_data->getNAngleTypes()];
    m_rcut =  new Scalar[m_CGCMMAngle_data->getNAngleTypes()];
    m_cg_type = new unsigned int[m_CGCMMAngle_data->getNAngleTypes()];
    
    assert(m_K);
    assert(m_t_0);
    assert(m_eps);
    assert(m_sigma);
    assert(m_rcut);
    assert(m_cg_type);
    
    memset((void*)m_K,0,sizeof(Scalar)*m_CGCMMAngle_data->getNAngleTypes());
    memset((void*)m_t_0,0,sizeof(Scalar)*m_CGCMMAngle_data->getNAngleTypes());
    memset((void*)m_eps,0,sizeof(Scalar)*m_CGCMMAngle_data->getNAngleTypes());
    memset((void*)m_sigma,0,sizeof(Scalar)*m_CGCMMAngle_data->getNAngleTypes());
    memset((void*)m_rcut,0,sizeof(Scalar)*m_CGCMMAngle_data->getNAngleTypes());
    memset((void*)m_cg_type,0,sizeof(unsigned int)*m_CGCMMAngle_data->getNAngleTypes());

    prefact[0] = Scalar(0.0);
    prefact[1] = Scalar(6.75);
    prefact[2] = Scalar(2.59807621135332);
    prefact[3] = Scalar(4.0);
    
    cgPow1[0]  = Scalar(0.0);
    cgPow1[1]  = Scalar(9.0);
    cgPow1[2]  = Scalar(12.0);
    cgPow1[3]  = Scalar(12.0);
    
    cgPow2[0]  = Scalar(0.0);
    cgPow2[1]  = Scalar(6.0);
    cgPow2[2]  = Scalar(4.0);
    cgPow2[3]  = Scalar(6.0);
    }

CGCMMAngleForceCompute::~CGCMMAngleForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying CGCMMAngleForceCompute" << endl;

    delete[] m_K;
    delete[] m_t_0;
    delete[] m_cg_type;
    delete[] m_eps;
    delete[] m_sigma;
    delete[] m_rcut;
    m_K = NULL;
    m_t_0 = NULL;
    m_cg_type = NULL;
    m_eps = NULL;
    m_sigma = NULL;
    m_rcut = NULL;
    }

/*! \param type Type of the angle to set parameters for
    \param K Stiffness parameter for the force computation
    \param t_0 Equilibrium angle in radians for the force computation
    \param cg_type the type of coarse grained angle
    \param eps the epsilon parameter for the 1-3 repulsion term
    \param sigma the sigma parameter for the 1-3 repulsion term

    Sets parameters for the potential of a particular angle type
*/
void CGCMMAngleForceCompute::setParams(unsigned int type, Scalar K, Scalar t_0, unsigned int cg_type, Scalar eps, Scalar sigma)
    {
    // make sure the type is valid
    if (type >= m_CGCMMAngle_data->getNAngleTypes())
        {
        m_exec_conf->msg->error() << "angle.cgcmm: Invalid angle type specified" << endl;
        throw runtime_error("Error setting parameters in CGCMMAngleForceCompute");
        }
        
    const double myPow1 = cgPow1[cg_type];
    const double myPow2 = cgPow2[cg_type];
    
    Scalar my_rcut = sigma * ((Scalar) exp(1.0/(myPow1-myPow2)*log(myPow1/myPow2)));
    
    m_K[type] = K;
    m_t_0[type] = t_0;
    m_cg_type[type] = cg_type;
    m_eps[type] = eps;
    m_sigma[type] = sigma;
    m_rcut[type] = my_rcut;
    
    // check for some silly errors a user could make
    if (cg_type > 3)
        m_exec_conf->msg->warning() << "angle.cgcmm: Unrecognized exponents specified" << endl;
    if (K <= 0)
        m_exec_conf->msg->warning() << "angle.cgcmm: specified K <= 0" << endl;
    if (t_0 <= 0)
        m_exec_conf->msg->warning() << "angle.cgcmm: specified t_0 <= 0" << endl;
    if (eps <= 0)
        m_exec_conf->msg->warning() << "angle.cgcmm: specified eps <= 0" << endl;
    if (sigma <= 0)
        m_exec_conf->msg->warning() << "angle.cgcmm: specified sigma <= 0" << endl;
    }

/*! CGCMMAngleForceCompute provides
    - \c angle_cgcmm_energy
*/
std::vector< std::string > CGCMMAngleForceCompute::getProvidedLogQuantities()
    {
    vector<string> list;
    list.push_back("angle_cgcmm_energy");
    return list;
    }

/*! \param quantity Name of the quantity to get the log value of
    \param timestep Current time step of the simulation
*/
Scalar CGCMMAngleForceCompute::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == string("angle_cgcmm_energy"))
        {
        compute(timestep);
        return calcEnergySum();
        }
    else
        {
        m_exec_conf->msg->error() << "angle.cgcmm: " << quantity << " is not a valid log quantity for CGCMMAngleForceCompute" << endl;
        throw runtime_error("Error getting log value");
        }
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void CGCMMAngleForceCompute::computeForces(unsigned int timestep)
    {
    if (m_prof) m_prof->push("CGCMMAngle");
    
    assert(m_pdata);
    // access the particle data arrays
    ArrayHandle< unsigned int > h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle< Scalar4 > h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
   
    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::overwrite);
    unsigned int virial_pitch = m_virial.getPitch();

    // Zero data for force calculation.
    memset((void*)h_force.data,0,sizeof(Scalar4)*m_force.getNumElements());
    memset((void*)h_virial.data,0,sizeof(Scalar)*m_virial.getNumElements());

    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);
    assert(h_rtag.data);
    
    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();
    
    // allocate forces
    Scalar fab[3], fcb[3];
    Scalar fac;
    
    Scalar eac;
    Scalar vac[6];
    // for each of the angles
    const unsigned int size = (unsigned int)m_CGCMMAngle_data->getNumAngles();
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the angle
        const Angle& angle = m_CGCMMAngle_data->getAngle(i);
        assert(angle.a < m_pdata->getN());
        assert(angle.b < m_pdata->getN());
        assert(angle.c < m_pdata->getN());
        
        // transform a, b, and c into indicies into the particle data arrays
        // MEM TRANSFER: 6 ints
        unsigned int idx_a = h_rtag.data[angle.a];
        unsigned int idx_b = h_rtag.data[angle.b];
        unsigned int idx_c = h_rtag.data[angle.c];
        assert(idx_a < m_pdata->getN());
        assert(idx_b < m_pdata->getN());
        assert(idx_c < m_pdata->getN());
        
        // calculate d\vec{r}
        // MEM_TRANSFER: 18 Scalars / FLOPS 9
        Scalar3 dab;
        dab.x = h_pos.data[idx_a].x - h_pos.data[idx_b].x;
        dab.y = h_pos.data[idx_a].y - h_pos.data[idx_b].y;
        dab.z = h_pos.data[idx_a].z - h_pos.data[idx_b].z;
        
        Scalar3 dcb;
        dcb.x = h_pos.data[idx_c].x - h_pos.data[idx_b].x;
        dcb.y = h_pos.data[idx_c].y - h_pos.data[idx_b].y;
        dcb.z = h_pos.data[idx_c].z - h_pos.data[idx_b].z;
        
        Scalar3 dac;
        dac.x = h_pos.data[idx_a].x - h_pos.data[idx_c].x; // used for the 1-3 JL interaction
        dac.y = h_pos.data[idx_a].y - h_pos.data[idx_c].y;
        dac.z = h_pos.data[idx_a].z - h_pos.data[idx_c].z;
        
        // apply minimum image conventions to all 3 vectors
        dab = box.minImage(dab);
        dcb = box.minImage(dcb);
        dac = box.minImage(dac);

        // on paper, the formula turns out to be: F = K*\vec{r} * (r_0/r - 1)
        // FLOPS: 14 / MEM TRANSFER: 2 Scalars
        
        // FLOPS: 42 / MEM TRANSFER: 6 Scalars
        Scalar rsqab = dab.x*dab.x+dab.y*dab.y+dab.z*dab.z;
        Scalar rab = sqrt(rsqab);
        Scalar rsqcb = dcb.x*dcb.x+dcb.y*dcb.y+dcb.z*dcb.z;
        Scalar rcb = sqrt(rsqcb);
        Scalar rsqac = dac.x*dac.x+dac.y*dac.y+dac.z*dac.z;
        Scalar rac = sqrt(rsqac);
        
        Scalar c_abbc = dab.x*dcb.x+dab.y*dcb.y+dab.z*dcb.z;
        c_abbc /= rab*rcb;
        
        if (c_abbc > 1.0) c_abbc = 1.0;
        if (c_abbc < -1.0) c_abbc = -1.0;
        
        Scalar s_abbc = sqrt(1.0 - c_abbc*c_abbc);
        if (s_abbc < SMALL) s_abbc = SMALL;
        s_abbc = 1.0/s_abbc;
        
        //////////////////////////////////////////
        // THIS CODE DOES THE 1-3 LJ repulsions //
        //////////////////////////////////////////////////////////////////////////////
        fac = Scalar(0.0);
        eac = Scalar(0.0);
        for (int k = 0; k < 6; k++)
            vac[k] = Scalar(0.0);

        if (rac < m_rcut[angle.type])
            {
            const unsigned int cg_type = m_cg_type[angle.type];
            const Scalar cg_pow1 = cgPow1[cg_type];
            const Scalar cg_pow2 = cgPow2[cg_type];
            const Scalar cg_pref = prefact[cg_type];
            
            const Scalar cg_ratio = m_sigma[angle.type]/rac;
            const Scalar cg_eps   = m_eps[angle.type];
            
            fac = cg_pref*cg_eps / rsqac * (cg_pow1*pow(cg_ratio,cg_pow1) - cg_pow2*pow(cg_ratio,cg_pow2));
            eac = cg_eps + cg_pref*cg_eps * (pow(cg_ratio,cg_pow1) - pow(cg_ratio,cg_pow2));

            vac[0] = fac * dac.x*dac.x;
            vac[1] = fac * dac.x*dac.y;
            vac[2] = fac * dac.x*dac.z;
            vac[3] = fac * dac.y*dac.y;
            vac[4] = fac * dac.y*dac.z;
            vac[5] = fac * dac.z*dac.z;
            }
        //////////////////////////////////////////////////////////////////////////////
        
        // actually calculate the force
        Scalar dth = acos(c_abbc) - m_t_0[angle.type];
        Scalar tk = m_K[angle.type]*dth;
        
        Scalar a = -1.0 * tk * s_abbc;
        Scalar a11 = a*c_abbc/rsqab;
        Scalar a12 = -a / (rab*rcb);
        Scalar a22 = a*c_abbc / rsqcb;
        
        fab[0] = a11*dab.x + a12*dcb.x;
        fab[1] = a11*dab.y + a12*dcb.y;
        fab[2] = a11*dab.z + a12*dcb.z;
        
        fcb[0] = a22*dcb.x + a12*dab.x;
        fcb[1] = a22*dcb.y + a12*dab.y;
        fcb[2] = a22*dcb.z + a12*dab.z;
        
        // compute 1/3 of the energy, 1/3 for each atom in the angle
        Scalar angle_eng = (0.5*tk*dth + eac)*Scalar(1.0/3.0);

        // compute 1/3 of the virial, 1/3 for each atom in the angle
        // upper triangular version of virial tensor
        Scalar angle_virial[6];
        angle_virial[0] = Scalar(1./3.) * ( dab.x*fab[0] + dcb.x*fcb[0] );
        angle_virial[1] = Scalar(1./3.) * ( dab.y*fab[0] + dcb.y*fcb[0] );
        angle_virial[2] = Scalar(1./3.) * ( dab.z*fab[0] + dcb.z*fcb[0] );
        angle_virial[3] = Scalar(1./3.) * ( dab.y*fab[1] + dcb.y*fcb[1] );
        angle_virial[4] = Scalar(1./3.) * ( dab.z*fab[1] + dcb.z*fcb[1] );
        angle_virial[5] = Scalar(1./3.) * ( dab.z*fab[2] + dcb.z*fcb[2] );
        Scalar virial[6];
        for (unsigned int k=0; k < 6; k++)
            virial[k] = angle_virial[k] + Scalar(1./3.)*vac[k];

        // Now, apply the force to each individual atom a,b,c, and accumlate the energy/virial
            
        h_force.data[idx_a].x += fab[0] + fac*dac.x;
        h_force.data[idx_a].y += fab[1] + fac*dac.y;
        h_force.data[idx_a].z += fab[2] + fac*dac.z;
        h_force.data[idx_a].w += angle_eng;
        for (int k = 0; k < 6; k++)
            h_virial.data[k*virial_pitch+idx_a] += virial[k];
        
        h_force.data[idx_b].x -= fab[0] + fcb[0];
        h_force.data[idx_b].y -= fab[1] + fcb[1];
        h_force.data[idx_b].z -= fab[2] + fcb[2];
        h_force.data[idx_b].w += angle_eng;
        for (int k = 0; k < 6; k++)
            h_virial.data[k*virial_pitch+idx_b] += virial[k];
        
        h_force.data[idx_c].x += fcb[0] - fac*dac.x;
        h_force.data[idx_c].y += fcb[1] - fac*dac.y;
        h_force.data[idx_c].z += fcb[2] - fac*dac.z;
        h_force.data[idx_c].w += angle_eng;
        for (int k = 0; k < 6; k++)
            h_virial.data[k*virial_pitch+idx_c] += virial[k];
        }
        
    if (m_prof) m_prof->pop();
    }

void export_CGCMMAngleForceCompute()
    {
    class_<CGCMMAngleForceCompute, boost::shared_ptr<CGCMMAngleForceCompute>, bases<ForceCompute>, boost::noncopyable >
    ("CGCMMAngleForceCompute", init< boost::shared_ptr<SystemDefinition> >())
    .def("setParams", &CGCMMAngleForceCompute::setParams)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

