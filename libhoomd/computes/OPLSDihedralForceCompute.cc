/*
Highly Optimized Object-oriented Many-particle Dynamics -- Blue Edition
(HOOMD-blue) Open Source Software License Copyright 2009-2014 The Regents of
the University of Michigan All rights reserved.

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

// Maintainer: ksil

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4244 )
#endif

#include <boost/python.hpp>
using namespace boost::python;

#include "OPLSDihedralForceCompute.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cmath>

using namespace std;

// SMALL a relatively small number
#define SMALL     0.001
#define SMALLER   0.00001

/*! \file OPLSDihedralForceCompute.cc
    \brief Contains code for the OPLSDihedralForceCompute class
*/

/*! \param sysdef System to compute forces on
    \post Memory is allocated, and forces are zeroed.
*/
OPLSDihedralForceCompute::OPLSDihedralForceCompute(boost::shared_ptr<SystemDefinition> sysdef)
    : ForceCompute(sysdef)
{
    m_exec_conf->msg->notice(5) << "Constructing OPLSDihedralForceCompute" << endl;

    // access the dihedral data for later use
    m_dihedral_data = m_sysdef->getDihedralData();

    // check for some silly errors a user could make
    if (m_dihedral_data->getNTypes() == 0)
        {
        m_exec_conf->msg->error() << "dihedral.opls: No dihedral types specified" << endl;
        throw runtime_error("Error initializing OPLSDihedralForceCompute");
        }

    // allocate the parameters
    GPUArray<Scalar4> params(m_dihedral_data->getNTypes(), exec_conf);
    m_params.swap(params);
}

OPLSDihedralForceCompute::~OPLSDihedralForceCompute()
{
    m_exec_conf->msg->notice(5) << "Destroying OPLSDihedralForceCompute" << endl;
}

/*! \param type Type of the dihedral to set parameters for
    \param k1 Force paramater in OPLS-style dihedral
    \param k2 Force paramater in OPLS-style dihedral
    \param k3 Force paramater in OPLS-style dihedral
    \param k4 Force paramater in OPLS-style dihedral

    Sets the parameters for the potential of an OPLS Dihedral, storing them with the
    1/2 prefactor.
*/
void OPLSDihedralForceCompute::setParams(unsigned int type, Scalar k1, Scalar k2, Scalar k3, Scalar k4)
{
    // make sure the type is valid
    if (type >= m_dihedral_data->getNTypes())
        {
        m_exec_conf->msg->error() << "dihedral.opls: Invalid dihedral type specified" << endl;
        throw runtime_error("Error setting parameters in OPLSDihedralForceCompute");
        }
    
    // set parameters in m_params
    ArrayHandle<Scalar4> h_params(m_params, access_location::host, access_mode::readwrite);
    h_params.data[type] = make_scalar4(k1/2.0, k2/2.0, k3/2.0, k4/2.0);
}

/*! DihedralForceCompute provides
    - \c dihedral_opls_energy
*/
std::vector< std::string > OPLSDihedralForceCompute::getProvidedLogQuantities()
    {
    vector<string> list;
    list.push_back("dihedral_opls_energy");
    return list;
    }

/*! \param quantity Name of the quantity to get the log value of
    \param timestep Current time step of the simulation
*/
Scalar OPLSDihedralForceCompute::getLogValue(const std::string& quantity, unsigned int timestep)
    {
    if (quantity == string("dihedral_opls_energy"))
        {
        compute(timestep);
        return calcEnergySum();
        }
    else
        {
        m_exec_conf->msg->error() << "dihedral.opls: " << quantity << " is not a valid log quantity" << endl;
        throw runtime_error("Error getting log value");
        }
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void OPLSDihedralForceCompute::computeForces(unsigned int timestep)
    {
    if (m_prof) m_prof->push("OPLS Dihedral");

    assert(m_pdata);
    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    // access the force and virial tensor arrays
    ArrayHandle<Scalar4> h_force(m_force, access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial, access_location::host, access_mode::overwrite);
    
    // access parameter data
    ArrayHandle<Scalar4> h_params(m_params, access_location::host, access_mode::read);

    // Zero data for force calculation before computation
    memset((void*)h_force.data,0,sizeof(Scalar4)*m_force.getNumElements());
    memset((void*)h_virial.data,0,sizeof(Scalar)*m_virial.getNumElements());

    // there are enough other checks on the input data, but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);
    assert(h_rtag.data);

    unsigned int virial_pitch = m_virial.getPitch();

    // From LAMMPS OPLS dihedral implementation
    unsigned int i1,i2,i3,i4,n,dihedral_type;
    Scalar3 vb1,vb2,vb3,vb2m;
    Scalar4 f1,f2,f3,f4;
    Scalar sb1,sb2,sb3,rb1,rb3,c0,b1mag2,b1mag,b2mag2;
    Scalar b2mag,b3mag2,b3mag,ctmp,r12c1,c1mag,r12c2,e_dihedral;
    Scalar c2mag,sc1,sc2,s1,s12,c,p,pd,a,a11,a22;
    Scalar a33,a12,a13,a23,sx2,sy2,sz2;
    Scalar s2,cx,cy,cz,cmag,dx,phi,si,siinv,sin2;
    Scalar k1,k2,k3,k4;
    Scalar dihedral_virial[6];
    
    // get a local copy of the simulation box
    const BoxDim& box = m_pdata->getBox();

    // iterate through each dihedral
    const unsigned int numDihedrals = (unsigned int)m_dihedral_data->getN();
    for (n = 0; n < numDihedrals; n++)
        {
        // lookup the tag of each of the particles participating in the dihedral
        const ImproperData::members_t& dihedral = m_dihedral_data->getMembersByIndex(n);
        assert(dihedral.tag[0] < m_pdata->getNGlobal());
        assert(dihedral.tag[1] < m_pdata->getNGlobal());
        assert(dihedral.tag[2] < m_pdata->getNGlobal());
        assert(dihedral.tag[3] < m_pdata->getNGlobal());
    
        // i1 to i4 are the tags
        i1 = h_rtag.data[dihedral.tag[0]];
        i2 = h_rtag.data[dihedral.tag[1]];
        i3 = h_rtag.data[dihedral.tag[2]];
        i4 = h_rtag.data[dihedral.tag[3]];
        
        // throw an error if this angle is incomplete
        if (i1 == NOT_LOCAL|| i2 == NOT_LOCAL || i3 == NOT_LOCAL || i4 == NOT_LOCAL)
            {
            this->m_exec_conf->msg->error() << "dihedral.opls: dihedral " <<
                dihedral.tag[0] << " " << dihedral.tag[1] << " " << dihedral.tag[2] << " " << dihedral.tag[3]
                << " incomplete." << endl << endl;
            throw std::runtime_error("Error in dihedral calculation");
            }
        
        assert(idx_a < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_b < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_c < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_d < m_pdata->getN() + m_pdata->getNGhosts());

        // 1st bond

        vb1.x = h_pos.data[i1].x - h_pos.data[i2].x;
        vb1.y = h_pos.data[i1].y - h_pos.data[i2].y;
        vb1.z = h_pos.data[i1].z - h_pos.data[i2].z;

        // 2nd bond

        vb2.x = h_pos.data[i3].x - h_pos.data[i2].x;
        vb2.y = h_pos.data[i3].y - h_pos.data[i2].y;
        vb2.z = h_pos.data[i3].z - h_pos.data[i2].z;

        // 3rd bond

        vb3.x = h_pos.data[i4].x - h_pos.data[i3].x;
        vb3.y = h_pos.data[i4].y - h_pos.data[i3].y;
        vb3.z = h_pos.data[i4].z - h_pos.data[i3].z;
        
        // apply periodic boundary conditions
        vb1 = box.minImage(vb1);
        vb2 = box.minImage(vb2);
        vb3 = box.minImage(vb3);
        
        vb2m.x = -vb2.x;
        vb2m.y = -vb2.y;
        vb2m.z = -vb2.z;
        vb2m = box.minImage(vb2m);

        // c0 calculation

        sb1 = 1.0 / (vb1.x*vb1.x + vb1.y*vb1.y + vb1.z*vb1.z);
        sb2 = 1.0 / (vb2.x*vb2.x + vb2.y*vb2.y + vb2.z*vb2.z);
        sb3 = 1.0 / (vb3.x*vb3.x + vb3.y*vb3.y + vb3.z*vb3.z);

        rb1 = sqrt(sb1);
        rb3 = sqrt(sb3);

        c0 = (vb1.x*vb3.x + vb1.y*vb3.y + vb1.z*vb3.z) * rb1*rb3;

        // 1st and 2nd angle

        b1mag2 = vb1.x*vb1.x + vb1.y*vb1.y + vb1.z*vb1.z;
        b1mag = sqrt(b1mag2);
        b2mag2 = vb2.x*vb2.x + vb2.y*vb2.y + vb2.z*vb2.z;
        b2mag = sqrt(b2mag2);
        b3mag2 = vb3.x*vb3.x + vb3.y*vb3.y + vb3.z*vb3.z;
        b3mag = sqrt(b3mag2);

        ctmp = vb1.x*vb2.x + vb1.y*vb2.y + vb1.z*vb2.z;
        r12c1 = 1.0 / (b1mag*b2mag);
        c1mag = ctmp * r12c1;

        ctmp = vb2m.x*vb3.x + vb2m.y*vb3.y + vb2m.z*vb3.z;
        r12c2 = 1.0 / (b2mag*b3mag);
        c2mag = ctmp * r12c2;

        // cos and sin of 2 angles and final c

        sin2 = 1.0 - c1mag*c1mag;
        if (sin2 < 0.0) sin2 = 0.0;
        sc1 = sqrt(sin2);
        if (sc1 < SMALL) sc1 = SMALL;
        sc1 = 1.0/sc1;

        sin2 = 1.0 - c2mag*c2mag;
        if (sin2 < 0.0) sin2 = 0.0;
        sc2 = sqrt(sin2);
        if (sc2 < SMALL) sc2 = SMALL;
        sc2 = 1.0/sc2;

        s1 = sc1 * sc1;
        s2 = sc2 * sc2;
        s12 = sc1 * sc2;
        c = (c0 + c1mag*c2mag) * s12;

        cx = vb1.y*vb2.z - vb1.z*vb2.y;
        cy = vb1.z*vb2.x - vb1.x*vb2.z;
        cz = vb1.x*vb2.y - vb1.y*vb2.x;
        cmag = sqrt(cx*cx + cy*cy + cz*cz);
        dx = (cx*vb3.x + cy*vb3.y + cz*vb3.z)/cmag/b3mag;

        if (c > 1.0) c = 1.0;
        if (c < -1.0) c = -1.0;

        // force & energy
        // p = sum (i=1,4) k_i * (1 + (-1)**(i+1)*cos(i*phi) )
        // pd = dp/dc

        phi = acos(c);
        if (dx < 0.0) phi *= -1.0;
        si = sin(phi);
        if (fabs(si) < SMALLER) si = SMALLER;
        siinv = 1.0/si;
        
        // get values for k1/2 through k4/2
        // ----- The 1/2 factor is already stored in the parameters --------
        dihedral_type = m_dihedral_data->getTypeByIndex(n);
        k1 = h_params.data[dihedral_type].x;
        k2 = h_params.data[dihedral_type].y;
        k3 = h_params.data[dihedral_type].z;
        k4 = h_params.data[dihedral_type].w;

        // the potential energy of the dihedral
        p = k1*(1.0 + c) + k2*(1.0 - cos(2.0*phi)) + k3*(1.0 + cos(3.0*phi)) + k4*(1.0 - cos(4.0*phi));
        pd = k1 - 2.0*k2*sin(2.0*phi)*siinv + 3.0*k3*sin(3.0*phi)*siinv - 4.0*k4*sin(4.0*phi)*siinv;

        // Compute 1/4 of energy and assign to each of 4 atoms in the dihedral
        e_dihedral = 0.25*p;

        a = pd;
        c = c * a;
        s12 = s12 * a;
        a11 = c*sb1*s1;
        a22 = -sb2 * (2.0*c0*s12 - c*(s1+s2));
        a33 = c*sb3*s2;
        a12 = -r12c1 * (c1mag*c*s1 + c2mag*s12);
        a13 = -rb1*rb3*s12;
        a23 = r12c2 * (c2mag*c*s2 + c1mag*s12);

        sx2  = a12*vb1.x + a22*vb2.x + a23*vb3.x;
        sy2  = a12*vb1.y + a22*vb2.y + a23*vb3.y;
        sz2  = a12*vb1.z + a22*vb2.z + a23*vb3.z;

        f1.x = a11*vb1.x + a12*vb2.x + a13*vb3.x;
        f1.y = a11*vb1.y + a12*vb2.y + a13*vb3.y;
        f1.z = a11*vb1.z + a12*vb2.z + a13*vb3.z;
        f1.w = e_dihedral;

        f2.x = -sx2 - f1.x;
        f2.y = -sy2 - f1.y;
        f2.z = -sz2 - f1.z;
        f2.w = e_dihedral;

        f4.x = a13*vb1.x + a23*vb2.x + a33*vb3.x;
        f4.y = a13*vb1.y + a23*vb2.y + a33*vb3.y;
        f4.z = a13*vb1.z + a23*vb2.z + a33*vb3.z;
        f4.w = e_dihedral;

        f3.x = sx2 - f4.x;
        f3.y = sy2 - f4.y;
        f3.z = sz2 - f4.z;
        f3.w = e_dihedral;
        
        // Apply force to each of the 4 atoms
        h_force.data[i1].x += f1.x;
        h_force.data[i1].y += f1.y;
        h_force.data[i1].z += f1.z;
        h_force.data[i1].w += f1.w;
        h_force.data[i2].x += f2.x;
        h_force.data[i2].y += f2.y;
        h_force.data[i2].z += f2.z;
        h_force.data[i2].w += f2.w;
        h_force.data[i3].x += f3.x;
        h_force.data[i3].y += f3.y;
        h_force.data[i3].z += f3.z;
        h_force.data[i3].w += f3.w;
        h_force.data[i4].x += f4.x;
        h_force.data[i4].y += f4.y;
        h_force.data[i4].z += f4.z;
        h_force.data[i4].w += f4.w;
        
        // Compute 1/4 of the virial, 1/4 for each atom in the dihedral
        // upper triangular version of virial tensor
        dihedral_virial[0] = 0.25*(vb1.x*f1.x + vb2.x*f3.x + (vb3.x+vb2.x)*f4.x);
        dihedral_virial[1] = 0.25*(vb1.y*f1.x + vb2.y*f3.x + (vb3.y+vb2.y)*f4.x);
        dihedral_virial[2] = 0.25*(vb1.z*f1.x + vb2.z*f3.x + (vb3.z+vb2.z)*f4.x);
        dihedral_virial[3] = 0.25*(vb1.y*f1.y + vb2.y*f3.y + (vb3.y+vb2.y)*f4.y);
        dihedral_virial[4] = 0.25*(vb1.z*f1.y + vb2.z*f3.y + (vb3.z+vb2.z)*f4.y);
        dihedral_virial[5] = 0.25*(vb1.z*f1.z + vb2.z*f3.z + (vb3.z+vb2.z)*f4.z);
        
        for (int k = 0; k < 6; k++)
            {
            h_virial.data[virial_pitch*k+i1]  += dihedral_virial[k];
            h_virial.data[virial_pitch*k+i2]  += dihedral_virial[k];
            h_virial.data[virial_pitch*k+i3]  += dihedral_virial[k];
            h_virial.data[virial_pitch*k+i4]  += dihedral_virial[k];
            }
        }

    if (m_prof) m_prof->pop();
    }

void export_OPLSDihedralForceCompute()
    {
    class_<OPLSDihedralForceCompute, boost::shared_ptr<OPLSDihedralForceCompute>, bases<ForceCompute>, boost::noncopyable >
    ("OPLSDihedralForceCompute", init< boost::shared_ptr<SystemDefinition> >())
    .def("setParams", &OPLSDihedralForceCompute::setParams)
    ;
    }

#ifdef WIN32
#pragma warning( pop )
#endif

#undef SMALL
#undef SMALLER