// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: ksil



#include "OPLSDihedralForceCompute.h"

namespace py = pybind11;

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cmath>

using namespace std;

/*! \file OPLSDihedralForceCompute.cc
    \brief Contains code for the OPLSDihedralForceCompute class
*/

/*! \param sysdef System to compute forces on
    \post Memory is allocated, and forces are zeroed.
*/
OPLSDihedralForceCompute::OPLSDihedralForceCompute(std::shared_ptr<SystemDefinition> sysdef)
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
    GPUArray<Scalar4> params(m_dihedral_data->getNTypes(), m_exec_conf);
    m_params.swap(params);
}

OPLSDihedralForceCompute::~OPLSDihedralForceCompute()
{
    m_exec_conf->msg->notice(5) << "Destroying OPLSDihedralForceCompute" << endl;
}

/*! \param type Type of the dihedral to set parameters for
    \param k1 Force parameter in OPLS-style dihedral
    \param k2 Force parameter in OPLS-style dihedral
    \param k3 Force parameter in OPLS-style dihedral
    \param k4 Force parameter in OPLS-style dihedral

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
    Scalar ax,ay,az,bx,by,bz,rasq,rbsq,rgsq,rg,rginv,ra2inv,rb2inv,rabinv;
    Scalar df,df1,ddf1,fg,hg,fga,hgb,gaa,gbb;
    Scalar dtfx,dtfy,dtfz,dtgx,dtgy,dtgz,dthx,dthy,dthz;
    Scalar c,s,p,sx2,sy2,sz2,cos_term,e_dihedral;
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

        assert(i1 < m_pdata->getN() + m_pdata->getNGhosts());
        assert(i2 < m_pdata->getN() + m_pdata->getNGhosts());
        assert(i3 < m_pdata->getN() + m_pdata->getNGhosts());
        assert(i4 < m_pdata->getN() + m_pdata->getNGhosts());

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

        // c,s calculation

        ax = vb1.y*vb2m.z - vb1.z*vb2m.y;
        ay = vb1.z*vb2m.x - vb1.x*vb2m.z;
        az = vb1.x*vb2m.y - vb1.y*vb2m.x;
        bx = vb3.y*vb2m.z - vb3.z*vb2m.y;
        by = vb3.z*vb2m.x - vb3.x*vb2m.z;
        bz = vb3.x*vb2m.y - vb3.y*vb2m.x;

        rasq = ax*ax + ay*ay + az*az;
        rbsq = bx*bx + by*by + bz*bz;
        rgsq = vb2m.x*vb2m.x + vb2m.y*vb2m.y + vb2m.z*vb2m.z;
        rg = sqrt(rgsq);

        rginv = ra2inv = rb2inv = 0.0;
        if (rg > 0) rginv = 1.0/rg;
        if (rasq > 0) ra2inv = 1.0/rasq;
        if (rbsq > 0) rb2inv = 1.0/rbsq;
        rabinv = sqrt(ra2inv*rb2inv);

        c = (ax*bx + ay*by + az*bz)*rabinv;
        s = rg*rabinv*(ax*vb3.x + ay*vb3.y + az*vb3.z);

        if (c > 1.0) c = 1.0;
        if (c < -1.0) c = -1.0;

        // get values for k1/2 through k4/2
        // ----- The 1/2 factor is already stored in the parameters --------
        dihedral_type = m_dihedral_data->getTypeByIndex(n);
        k1 = h_params.data[dihedral_type].x;
        k2 = h_params.data[dihedral_type].y;
        k3 = h_params.data[dihedral_type].z;
        k4 = h_params.data[dihedral_type].w;

        // calculate the potential p = sum (i=1,4) k_i * (1 + (-1)**(i+1)*cos(i*phi) )
        // and df = dp/dc

        // cos(phi) term
        ddf1 = c;
        df1 = s;
        cos_term = ddf1;

        p = k1 * (1.0 + cos_term);
        df = k1*df1;

        // cos(2*phi) term
        ddf1 = cos_term*c - df1*s;
        df1 = cos_term*s + df1*c;
        cos_term = ddf1;

        p += k2 * (1.0 - cos_term);
        df += -2.0*k2*df1;

        // cos(3*phi) term
        ddf1 = cos_term*c - df1*s;
        df1 = cos_term*s + df1*c;
        cos_term = ddf1;

        p += k3 * (1.0 + cos_term);
        df += 3.0*k3*df1;

        // cos(4*phi) term
        ddf1 = cos_term*c - df1*s;
        df1 = cos_term*s + df1*c;
        cos_term = ddf1;

        p += k4 * (1.0 - cos_term);
        df += -4.0*k4*df1;

        // Compute 1/4 of energy to assign to each of 4 atoms in the dihedral
        e_dihedral = 0.25*p;

        fg = vb1.x*vb2m.x + vb1.y*vb2m.y + vb1.z*vb2m.z;
        hg = vb3.x*vb2m.x + vb3.y*vb2m.y + vb3.z*vb2m.z;
        fga = fg*ra2inv*rginv;
        hgb = hg*rb2inv*rginv;
        gaa = -ra2inv*rg;
        gbb = rb2inv*rg;

        dtfx = gaa*ax;
        dtfy = gaa*ay;
        dtfz = gaa*az;
        dtgx = fga*ax - hgb*bx;
        dtgy = fga*ay - hgb*by;
        dtgz = fga*az - hgb*bz;
        dthx = gbb*bx;
        dthy = gbb*by;
        dthz = gbb*bz;

        sx2 = df*dtgx;
        sy2 = df*dtgy;
        sz2 = df*dtgz;

        f1.x = df*dtfx;
        f1.y = df*dtfy;
        f1.z = df*dtfz;
        f1.w = e_dihedral;

        f2.x = sx2 - f1.x;
        f2.y = sy2 - f1.y;
        f2.z = sz2 - f1.z;
        f2.w = e_dihedral;

        f4.x = df*dthx;
        f4.y = df*dthy;
        f4.z = df*dthz;
        f4.w = e_dihedral;

        f3.x = -sx2 - f4.x;
        f3.y = -sy2 - f4.y;
        f3.z = -sz2 - f4.z;
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

void export_OPLSDihedralForceCompute(py::module& m)
    {
    py::class_<OPLSDihedralForceCompute, std::shared_ptr<OPLSDihedralForceCompute> >(m, "OPLSDihedralForceCompute", py::base<ForceCompute>())
    .def(py::init< std::shared_ptr<SystemDefinition> >())
    .def("setParams", &OPLSDihedralForceCompute::setParams)
    ;
    }
