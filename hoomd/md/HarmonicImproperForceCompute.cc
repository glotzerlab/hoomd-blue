// Copyright (c) 2009-2019 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.


// Maintainer: dnlebard



#include "HarmonicImproperForceCompute.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <math.h>

using namespace std;
namespace py = pybind11;

// SMALL a relatively small number
#define SMALL Scalar(0.001)

/*! \file HarmonicImproperForceCompute.cc
    \brief Contains code for the HarmonicImproperForceCompute class
*/

/*! \param sysdef System to compute forces on
    \post Memory is allocated, and forces are zeroed.
*/
HarmonicImproperForceCompute::HarmonicImproperForceCompute(std::shared_ptr<SystemDefinition> sysdef)
    : ForceCompute(sysdef), m_K(NULL), m_chi(NULL)
    {
    m_exec_conf->msg->notice(5) << "Constructing HarmonicImproperForceCompute" << endl;

    // access the improper data for later use
    m_improper_data = m_sysdef->getImproperData();

    // check for some silly errors a user could make
    if (m_improper_data->getNTypes() == 0)
        {
        m_exec_conf->msg->error() << "improper.harmonic: No improper types specified" << endl;
        throw runtime_error("Error initializing HarmonicImproperForceCompute");
        }

    // allocate the parameters
    m_K = new Scalar[m_improper_data->getNTypes()];
    m_chi = new Scalar[m_improper_data->getNTypes()];
    }

HarmonicImproperForceCompute::~HarmonicImproperForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying HarmonicImproperForceCompute" << endl;

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
    if (type >= m_improper_data->getNTypes())
        {
        m_exec_conf->msg->error() << "improper.harmonic: Invalid improper type specified" << endl;
        throw runtime_error("Error setting parameters in HarmonicImproperForceCompute");
        }

    m_K[type] = K;
    m_chi[type] = chi;

    // check for some silly errors a user could make
    if (K <= 0)
        m_exec_conf->msg->warning() << "improper.harmonic: specified K <= 0" << endl;
    if (chi <= 0)
        m_exec_conf->msg->warning() << "improper.harmonic: specified Chi <= 0" << endl;
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
        m_exec_conf->msg->error() << "improper.harmonic: " << quantity << " is not a valid log quantity" << endl;
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
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::overwrite);
    unsigned int virial_pitch = m_virial.getPitch();

    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);
    assert(h_rtag.data);

    // Zero data for force calculation.
    memset((void*)h_force.data,0,sizeof(Scalar4)*m_force.getNumElements());
    memset((void*)h_virial.data,0,sizeof(Scalar)*m_virial.getNumElements());

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getBox();

    // for each of the impropers
    const unsigned int size = (unsigned int)m_improper_data->getN();
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the improper
        const ImproperData::members_t& improper = m_improper_data->getMembersByIndex(i);
        assert(improper.tag[0] <= m_pdata->getMaximumTag());
        assert(improper.tag[1] <= m_pdata->getMaximumTag());
        assert(improper.tag[2] <= m_pdata->getMaximumTag());
        assert(improper.tag[3] <= m_pdata->getMaximumTag());

        // transform a, b, and c into indices into the particle data arrays
        // MEM TRANSFER: 6 ints
        unsigned int idx_a = h_rtag.data[improper.tag[0]];
        unsigned int idx_b = h_rtag.data[improper.tag[1]];
        unsigned int idx_c = h_rtag.data[improper.tag[2]];
        unsigned int idx_d = h_rtag.data[improper.tag[3]];

        // throw an error if this angle is incomplete
        if (idx_a == NOT_LOCAL|| idx_b == NOT_LOCAL || idx_c == NOT_LOCAL || idx_d == NOT_LOCAL)
            {
            this->m_exec_conf->msg->error() << "improper.harmonic: improper " <<
                improper.tag[0] << " " << improper.tag[1] << " " << improper.tag[2] << " " << improper.tag[3]
                << " incomplete." << endl << endl;
            throw std::runtime_error("Error in improper calculation");
            }

        assert(idx_a < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_b < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_c < m_pdata->getN() + m_pdata->getNGhosts());
        assert(idx_d < m_pdata->getN() + m_pdata->getNGhosts());

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

        Scalar ss1 = 1.0 / (dab.x*dab.x + dab.y*dab.y + dab.z*dab.z);
        Scalar ss2 = 1.0 / (dcb.x*dcb.x + dcb.y*dcb.y + dcb.z*dcb.z);
        Scalar ss3 = 1.0 / (ddc.x*ddc.x + ddc.y*ddc.y + ddc.z*ddc.z);

        Scalar r1 = sqrt(ss1);
        Scalar r2 = sqrt(ss2);
        Scalar r3 = sqrt(ss3);

        // Cosine and Sin of the angle between the planes
        Scalar c0 = (dab.x*ddc.x + dab.y*ddc.y + dab.z*ddc.z)* r1 * r3;
        Scalar c1 = (dab.x*dcb.x + dab.y*dcb.y + dab.z*dcb.z)* r1 * r2;
        Scalar c2 = -(ddc.x*dcb.x + ddc.y*dcb.y + ddc.z*dcb.z)* r3 * r2;

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

        unsigned int improper_type = m_improper_data->getTypeByIndex(i);
        Scalar domega = acos(c) - m_chi[improper_type];
        Scalar a = m_K[improper_type] * domega;

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

        Scalar sx2  = a22*dcb.x + a23*ddc.x + a12*dab.x;
        Scalar sy2  = a22*dcb.y + a23*ddc.y + a12*dab.y;
        Scalar sz2  = a22*dcb.z + a23*ddc.z + a12*dab.z;

        // calculate the forces for each particle
        Scalar ffax = a12*dcb.x + a13*ddc.x + a11*dab.x;
        Scalar ffay = a12*dcb.y + a13*ddc.y + a11*dab.y;
        Scalar ffaz = a12*dcb.z + a13*ddc.z + a11*dab.z;

        Scalar ffbx = -sx2 - ffax;
        Scalar ffby = -sy2 - ffay;
        Scalar ffbz = -sz2 - ffaz;

        Scalar ffdx = a23*dcb.x + a33*ddc.x + a13*dab.x;
        Scalar ffdy = a23*dcb.y + a33*ddc.y + a13*dab.y;
        Scalar ffdz = a23*dcb.z + a33*ddc.z + a13*dab.z;

        Scalar ffcx = sx2 - ffdx;
        Scalar ffcy = sy2 - ffdy;
        Scalar ffcz = sz2 - ffdz;

        // and calculate the virial (upper triangular version)
        // compute 1/4 of the virial, 1/4 for each atom in the improper
        Scalar improper_virial[6];
        improper_virial[0] = (1./4.)*(dab.x*ffax + dcb.x*ffcx + (ddc.x+dcb.x)*ffdx);
        improper_virial[1] = (1./4.)*(dab.y*ffax + dcb.y*ffcx + (ddc.y+dcb.y)*ffdx);
        improper_virial[2] = (1./4.)*(dab.z*ffax + dcb.z*ffcx + (ddc.z+dcb.z)*ffdx);
        improper_virial[3] = (1./4.)*(dab.y*ffay + dcb.y*ffcy + (ddc.y+dcb.y)*ffdy);
        improper_virial[4] = (1./4.)*(dab.z*ffay + dcb.z*ffcy + (ddc.z+dcb.z)*ffdy);
        improper_virial[5] = (1./4.)*(dab.z*ffaz + dcb.z*ffcz + (ddc.z+dcb.z)*ffdz);

        if (idx_a < m_pdata->getN())
            {
            // accumulate the forces
            h_force.data[idx_a].x += ffax;
            h_force.data[idx_a].y += ffay;
            h_force.data[idx_a].z += ffaz;
            h_force.data[idx_a].w += improper_eng;
            for (int k = 0; k < 6; k++)
                h_virial.data[k*virial_pitch+idx_a]  += improper_virial[k];
            }

        if (idx_b < m_pdata->getN())
            {
            h_force.data[idx_b].x += ffbx;
            h_force.data[idx_b].y += ffby;
            h_force.data[idx_b].z += ffbz;
            h_force.data[idx_b].w += improper_eng;
            for (int k = 0; k < 6; k++)
                h_virial.data[k*virial_pitch+idx_b]  += improper_virial[k];
            }

        if (idx_c < m_pdata->getN())
            {
            h_force.data[idx_c].x += ffcx;
            h_force.data[idx_c].y += ffcy;
            h_force.data[idx_c].z += ffcz;
            h_force.data[idx_c].w += improper_eng;
            for (int k = 0; k < 6; k++)
                h_virial.data[k*virial_pitch+idx_c]  += improper_virial[k];
            }

        if (idx_d < m_pdata->getN())
            {
            h_force.data[idx_d].x += ffdx;
            h_force.data[idx_d].y += ffdy;
            h_force.data[idx_d].z += ffdz;
            h_force.data[idx_d].w += improper_eng;
            for (int k = 0; k < 6; k++)
                h_virial.data[k*virial_pitch+idx_d]  += improper_virial[k];
            }
        }

    if (m_prof) m_prof->pop();
    }

void export_HarmonicImproperForceCompute(py::module& m)
    {
    py::class_<HarmonicImproperForceCompute, std::shared_ptr<HarmonicImproperForceCompute> >(m, "HarmonicImproperForceCompute", py::base<ForceCompute>())
    .def(py::init< std::shared_ptr<SystemDefinition> >())
    .def("setParams", &HarmonicImproperForceCompute::setParams)
    ;
    }
