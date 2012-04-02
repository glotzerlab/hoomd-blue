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

#include "HarmonicDihedralForceCompute.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <math.h>

using namespace std;

// SMALL a relatively small number
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
    m_exec_conf->msg->notice(5) << "Constructing HarmonicDihedralForceCompute" << endl;

    // access the dihedral data for later use
    m_dihedral_data = m_sysdef->getDihedralData();
    
    // check for some silly errors a user could make
    if (m_dihedral_data->getNDihedralTypes() == 0)
        {
        m_exec_conf->msg->error() << "dihedral.harmonic: No dihedral types specified" << endl;
        throw runtime_error("Error initializing HarmonicDihedralForceCompute");
        }
        
    // allocate the parameters
    m_K = new Scalar[m_dihedral_data->getNDihedralTypes()];
    m_sign = new Scalar[m_dihedral_data->getNDihedralTypes()];
    m_multi = new Scalar[m_dihedral_data->getNDihedralTypes()];
    
    }

HarmonicDihedralForceCompute::~HarmonicDihedralForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying HarmonicDihedralForceCompute" << endl;

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
        m_exec_conf->msg->error() << "dihedral.harmonic: Invalid dihedral type specified" << endl;
        throw runtime_error("Error setting parameters in HarmonicDihedralForceCompute");
        }
        
    m_K[type] = K;
    m_sign[type] = (Scalar)sign;
    m_multi[type] = (Scalar)multiplicity;
    
    // check for some silly errors a user could make
    if (K <= 0)
        m_exec_conf->msg->warning() << "dihedral.harmonic: specified K <= 0" << endl;
    if (sign != 1 && sign != -1)
        m_exec_conf->msg->warning() << "dihedral.harmonic: a non unitary sign was specified" << endl;
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
        m_exec_conf->msg->error() << "dihedral.harmonic: " << quantity << " is not a valid log quantity" << endl;
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
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

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
        unsigned int idx_a = h_rtag.data[dihedral.a];
        unsigned int idx_b = h_rtag.data[dihedral.b];
        unsigned int idx_c = h_rtag.data[dihedral.c];
        unsigned int idx_d = h_rtag.data[dihedral.d];
        assert(idx_a < m_pdata->getN());
        assert(idx_b < m_pdata->getN());
        assert(idx_c < m_pdata->getN());
        assert(idx_d < m_pdata->getN());
        
        // calculate d\vec{r}
        Scalar3 dab;
        dab.x = h_pos.data[idx_a].x - h_pos.data[idx_b].x;
        dab.y = h_pos.data[idx_a].y - h_pos.data[idx_b].y;
        dab.z = h_pos.data[idx_a].z - h_pos.data[idx_b].z;
        
        Scalar3 dcb;
        dcb.x = h_pos.data[idx_c].x - h_pos.data[idx_b].x;
        dcb.y = h_pos.data[idx_c].y - h_pos.data[idx_b].y;
        dcb.z = h_pos.data[idx_c].z - h_pos.data[idx_b].z;
        
        Scalar3 ddc;
        ddc.x = h_pos.data[idx_d].x - h_pos.data[idx_c].x;
        ddc.y = h_pos.data[idx_d].y - h_pos.data[idx_c].y;
        ddc.z = h_pos.data[idx_d].z - h_pos.data[idx_c].z;
        
        // apply periodic boundary conditions
        dab = box.minImage(dab);
        dcb = box.minImage(dcb);
        ddc = box.minImage(ddc);
        
        Scalar3 dcbm;
        dcbm.x = -dcb.x;
        dcbm.y = -dcb.y;
        dcbm.z = -dcb.z;

        dcbm = box.minImage(dcbm);
        
        Scalar aax = dab.y*dcbm.z - dab.z*dcbm.y;
        Scalar aay = dab.z*dcbm.x - dab.x*dcbm.z;
        Scalar aaz = dab.x*dcbm.y - dab.y*dcbm.x;
        
        Scalar bbx = ddc.y*dcbm.z - ddc.z*dcbm.y;
        Scalar bby = ddc.z*dcbm.x - ddc.x*dcbm.z;
        Scalar bbz = ddc.x*dcbm.y - ddc.y*dcbm.x;
        
        Scalar raasq = aax*aax + aay*aay + aaz*aaz;
        Scalar rbbsq = bbx*bbx + bby*bby + bbz*bbz;
        Scalar rgsq = dcbm.x*dcbm.x + dcbm.y*dcbm.y + dcbm.z*dcbm.z;
        Scalar rg = sqrt(rgsq);
        
        Scalar rginv, raa2inv, rbb2inv;
        rginv = raa2inv = rbb2inv = 0.0f;
        if (rg > 0.0f) rginv = 1.0f/rg;
        if (raasq > 0.0f) raa2inv = 1.0f/raasq;
        if (rbbsq > 0.0f) rbb2inv = 1.0f/rbbsq;
        Scalar rabinv = sqrt(raa2inv*rbb2inv);
        
        Scalar c_abcd = (aax*bbx + aay*bby + aaz*bbz)*rabinv;
        Scalar s_abcd = rg*rabinv*(aax*ddc.x + aay*ddc.y + aaz*ddc.z);
        
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
            
            
        Scalar fg = dab.x*dcbm.x + dab.y*dcbm.y + dab.z*dcbm.z;
        Scalar hg = ddc.x*dcbm.x + ddc.y*dcbm.y + ddc.z*dcbm.z;
        
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

        // compute 1/4 of the virial, 1/4 for each atom in the dihedral
        // symmetrized version of virial tensor
        Scalar dihedral_virial[6];
        dihedral_virial[0] = (1./4.)*(dab.x*ffax + dcb.x*ffcx + (ddc.x+dcb.x)*ffdx);
        dihedral_virial[1] = (1./8.)*(dab.x*ffay + dcb.x*ffcy + (ddc.x+dcb.x)*ffdy
                                     +dab.y*ffax + dcb.y*ffcx + (ddc.y+dcb.y)*ffdx);
        dihedral_virial[2] = (1./8.)*(dab.x*ffaz + dcb.x*ffcz + (ddc.x+dcb.x)*ffdz
                                     +dab.z*ffax + dcb.z*ffcx + (ddc.z+dcb.z)*ffdx);
        dihedral_virial[3] = (1./4.)*(dab.y*ffay + dcb.y*ffcy + (ddc.y+dcb.y)*ffdy);
        dihedral_virial[4] = (1./8.)*(dab.y*ffaz + dcb.y*ffcz + (ddc.y+dcb.y)*ffdz
                                     +dab.z*ffay + dcb.z*ffcy + (ddc.z+dcb.z)*ffdy);
        dihedral_virial[5] = (1./4.)*(dab.z*ffaz + dcb.z*ffcz + (ddc.z+dcb.z)*ffdz);
       
        h_force.data[idx_a].x += ffax; 
        h_force.data[idx_a].y += ffay; 
        h_force.data[idx_a].z += ffaz; 
        h_force.data[idx_a].w += dihedral_eng; 
        for (int k = 0; k < 6; k++)
           h_virial.data[virial_pitch*k+idx_a]  += dihedral_virial[k];

        h_force.data[idx_b].x += ffbx; 
        h_force.data[idx_b].y += ffby; 
        h_force.data[idx_b].z += ffbz; 
        h_force.data[idx_b].w += dihedral_eng; 
        for (int k = 0; k < 6; k++)
           h_virial.data[virial_pitch*k+idx_b]  += dihedral_virial[k];

        h_force.data[idx_c].x += ffcx; 
        h_force.data[idx_c].y += ffcy; 
        h_force.data[idx_c].z += ffcz; 
        h_force.data[idx_c].w += dihedral_eng; 
        for (int k = 0; k < 6; k++)
           h_virial.data[virial_pitch*k+idx_c]  += dihedral_virial[k];

        h_force.data[idx_d].x += ffdx; 
        h_force.data[idx_d].y += ffdy; 
        h_force.data[idx_d].z += ffdz; 
        h_force.data[idx_d].w += dihedral_eng; 
        for (int k = 0; k < 6; k++)
           h_virial.data[virial_pitch*k+idx_d]  += dihedral_virial[k];
       }
        
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

