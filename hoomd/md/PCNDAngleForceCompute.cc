// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "PCNDAngleForceCompute.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <math.h>

using namespace std;

// \param SMALL a relatively small number
#define SMALL Scalar(0.001)

/*! \file PCNDAngleForceCompute.cc
    \brief Contains code for the PCNDAngleForceCompute class
*/

namespace hoomd
    {
namespace md
    {
/*! \param sysdef System to compute forces on
    \post Memory is allocated, and forces are zeroed.
*/
PCNDAngleForceCompute::PCNDAngleForceCompute(std::shared_ptr<SystemDefinition> sysdef)
    : ForceCompute(sysdef), m_Xi(NULL), m_Tau(NULL), m_PCND_type(NULL), m_particle_sum(NULL), m_particle_index(NULL), m_rcut(NULL)
    {
    m_exec_conf->msg->notice(5) << "Constructing PCNDAngleForceCompute" << endl;

    // access the angle data for later use
    m_pcnd_angle_data = m_sysdef->getAngleData();

    // check for some silly errors a user could make
    if (m_pcnd_angle_data->getNTypes() == 0)
        {
        m_exec_conf->msg->error() << "angle.pcnd: No angle types specified" << endl;
        throw runtime_error("Error initializing PCNDAngleForceCompute");
        }

    // allocate the parameters
    m_Xi = new Scalar[m_pcnd_angle_data->getNTypes()];
    m_Tau = new Scalar[m_pcnd_angle_data->getNTypes()];
    m_PCND_type= new unsigned int[m_pcnd_angle_data->getNTypes()];
    m_particle_sum =  new uint16_t [m_pcnd_angle_data->getNTypes()];
    m_particle_index = new Scalar[m_pcnd_angle_data->getNTypes()];
    m_rcut =  new Scalar[m_pcnd_angle_data->getNTypes()];

    assert(m_Xi);
    assert(m_Tau);
    assert(m_PCND_type);
    assert(m_particle_sum);
    assert(m_particle_index);
    assert(m_rcut);

    memset((void*)m_Xi,0,sizeof(Scalar)*m_pcnd_angle_data->getNTypes());
    memset((void*)m_Tau,0,sizeof(Scalar)*m_pcnd_angle_data->getNTypes());
    memset((void*)m_PCND_type,0,sizeof(unsigned int)*m_pcnd_angle_data->getNTypes());
    memset((void*)m_particle_sum,0,sizeof(uint16_t)*m_pcnd_angle_data->getNTypes());
    memset((void*)m_particle_index,0,sizeof(Scalar)*m_pcnd_angle_data->getNTypes());
    memset((void*)m_rcut,0,sizeof(Scalar)*m_pcnd_angle_data->getNTypes());

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

PCNDAngleForceCompute::~PCNDAngleForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying PCNDAngleForceCompute" << endl;

    delete[] m_Xi;
    delete[] m_Tau;
    delete[] m_PCND_type;
    delete[] m_particle_sum;
    delete[] m_particle_index;
    delete[] m_rcut;
    m_Xi = NULL;
    m_Tau = NULL;
    m_PCND_type = NULL;
    m_particle_sum = NULL;
    m_particle_index = NULL;
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
void PCNDAngleForceCompute::setParams(unsigned int type, Scalar Xi, Scalar Tau, unsigned int PCND_type, uint16_t particle_sum, Scalar particle_index)
    {
    // make sure the type is valid
    if (type >= m_pcnd_angle_data->getNTypes())
        {
        m_exec_conf->msg->error() << "angle.pcnd: Invalid angle type specified" << endl;
        throw runtime_error("Error setting parameters in PCNDAngleForceCompute");
        }

    const double myPow1 = cgPow1[PCND_type];
    const double myPow2 = cgPow2[PCND_type];

    Scalar my_rcut = particle_index * ((Scalar) exp(1.0 / (myPow1 - myPow2) * log(myPow1 / myPow2)));

    m_Xi[type] = Xi;
    m_Tau[type] = Tau;
    m_PCND_type[type] = PCND_type;
    m_particle_sum[type] = particle_sum;
    m_particle_index[type] = particle_index;
    m_rcut[type] = my_rcut;

    // check for some silly errors a user could make
    /*
    if (cg_type > 3)
        m_exec_conf->msg->warning() << "angle.pcnd: Unrecognized exponents specified" << endl;
    if (K <= 0)
        m_exec_conf->msg->warning() << "angle.pcnd: specified K <= 0" << endl;
    if (t_0 <= 0)
        m_exec_conf->msg->warning() << "angle.pcnd: specified t_0 <= 0" << endl;
    if (eps <= 0)
        m_exec_conf->msg->warning() << "angle.pcnd: specified eps <= 0" << endl;
    if (sigma <= 0)
        m_exec_conf->msg->warning() << "angle.pcnd: specified sigma <= 0" << endl;
        */
    }

void PCNDAngleForceCompute::setParamsPython(std::string type, pybind11::dict params)
    {                                                                           
    auto typ = m_pcnd_angle_data->getTypeByName(type);                               
    auto _params = angle_pcnd_params(params);                               
    setParams(typ, _params.Xi, _params.Tau, _params.PCND_type, _params.particle_sum, _params.particle_index);
    }                                                                                 

pybind11::dict PCNDAngleForceCompute::getParams(std::string type)           
    {                                                                           
    auto typ = m_pcnd_angle_data->getTypeByName(type);                               
    if (typ >= m_pcnd_angle_data->getNTypes())                                       
        {                                                                       
        throw runtime_error("Invalid angle type.");                             
        }                                                                       
    pybind11::dict params;                                                      
    params["Xi"] = m_Xi[typ];                                                     
    params["Tau"] = m_Tau[typ];                                                  
    params["PCND_type"] = m_PCND_type[typ];
    params["particle_sum"] = m_particle_sum[typ];
    params["particle_index"] = m_particle_index[typ];
    return params;                                                              
    }

/*! PCNDAngleForceCompute provides
    - \c angle_pcnd_energy
*/
std::vector< std::string > PCNDAngleForceCompute::getProvidedLogQuantities()
    {
    vector<string> list;
    list.push_back("angle_pcnd_energy");
    return list;
    }

/*! \param quantity Name of the quantity to get the log value of
    \param timestep Current time step of the simulation
*/
Scalar PCNDAngleForceCompute::getLogValue(const std::string& quantity, uint64_t timestep)
    {
    if (quantity == string("angle_pcnd_energy"))
        {
        compute(timestep);
        return calcEnergySum();
        }
    else
        {
        m_exec_conf->msg->error() << "angle.pcnd: " << quantity << " is not a valid log quantity for PCNDAngleForceCompute" << endl;
        throw runtime_error("Error getting log value");
        }
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void PCNDAngleForceCompute::computeForces(uint64_t timestep)
    {
    if (m_prof) m_prof->push("PCNDAngle");

    assert(m_pdata);
    // access the particle data arrays
    ArrayHandle< unsigned int > h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);
    ArrayHandle< Scalar4 > h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::overwrite);
    size_t virial_pitch = m_virial.getPitch();

    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);
    assert(h_rtag.data);

    // Zero data for force calculation.
    memset((void*)h_force.data,0,sizeof(Scalar4)*m_force.getNumElements());
    memset((void*)h_virial.data,0,sizeof(Scalar)*m_virial.getNumElements());

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getGlobalBox();

    // allocate forces
    // Scalar fab[3], fcb[3];
    Scalar fac;

    Scalar eac;
    Scalar vac[6];
    // for each of the angles
    const unsigned int size = (unsigned int)m_pcnd_angle_data->getN();
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the angle
        const AngleData::members_t& angle = m_pcnd_angle_data->getMembersByIndex(i);
        assert(angle.tag[0] <= m_pdata->getMaximumTag());
        assert(angle.tag[1] <= m_pdata->getMaximumTag());
        assert(angle.tag[1] <= m_pdata->getMaximumTag());

        // transform a, b, and c into indices into the particle data arrays
        // MEM TRANSFER: 6 ints
        unsigned int idx_a = h_rtag.data[angle.tag[0]];
        unsigned int idx_b = h_rtag.data[angle.tag[1]];
        unsigned int idx_c = h_rtag.data[angle.tag[2]];

        // throw an error if this angle is incomplete
        if (idx_a == NOT_LOCAL|| idx_b == NOT_LOCAL || idx_c == NOT_LOCAL)
            {
            this->m_exec_conf->msg->error() << "angle.harmonic: angle " <<
                angle.tag[0] << " " << angle.tag[1] << " " << angle.tag[2] << " incomplete." << endl << endl;
            throw std::runtime_error("Error in angle calculation");
            }

        assert(idx_a < m_pdata->getN()+m_pdata->getNGhosts());
        assert(idx_b < m_pdata->getN()+m_pdata->getNGhosts());
        assert(idx_c < m_pdata->getN()+m_pdata->getNGhosts());

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
        Scalar rsqab = dab.x * dab.x + dab.y * dab.y + dab.z * dab.z;
        Scalar rab = sqrt(rsqab);
        Scalar rsqcb = dcb.x * dcb.x + dcb.y * dcb.y + dcb.z * dcb.z;
        Scalar rcb = sqrt(rsqcb);
        Scalar rsqac = dac.x * dac.x + dac.y * dac.y + dac.z * dac.z;
        Scalar rac = sqrt(rsqac);

        Scalar c_abbc = dab.x * dcb.x + dab.y * dcb.y + dab.z * dcb.z;
        c_abbc /= rab * rcb;

        if (c_abbc > 1.0)
            c_abbc = 1.0;
        if (c_abbc < -1.0)
	    c_abbc = -1.0;

        Scalar s_abbc = sqrt(1.0 - c_abbc * c_abbc);
        if (s_abbc < SMALL)
            s_abbc = SMALL;
        s_abbc = 1.0 / s_abbc;

        //////////////////////////////////////////
        // THIS CODE DOES THE 1-3 LJ repulsions //
        //////////////////////////////////////////////////////////////////////////////
        unsigned int angle_type = m_pcnd_angle_data->getTypeByIndex(i);
        fac = Scalar(0.0);
        eac = Scalar(0.0);
        for (int k = 0; k < 6; k++)
            vac[k] = Scalar(0.0);

        if (rac < m_rcut[angle_type])
            {
            const unsigned int PCND_type = m_PCND_type[angle_type];
            const Scalar cg_pow1 = cgPow1[PCND_type];
            const Scalar cg_pow2 = cgPow2[PCND_type];
            const Scalar cg_pref = prefact[PCND_type];

            const Scalar cg_ratio = m_particle_index[angle_type] / rac;
            const uint16_t cg_eps   = m_particle_sum[angle_type];

            fac = cg_pref * cg_eps / rsqac * (cg_pow1 * pow(cg_ratio, cg_pow1) - cg_pow2 * pow(cg_ratio, cg_pow2));
            eac = cg_eps + cg_pref * cg_eps * (pow(cg_ratio, cg_pow1) - pow(cg_ratio, cg_pow2));

            vac[0] = fac * dac.x * dac.x;
            vac[1] = fac * dac.x * dac.y;
            vac[2] = fac * dac.x * dac.z;
            vac[3] = fac * dac.y * dac.y;
            vac[4] = fac * dac.y * dac.z;
            vac[5] = fac * dac.z * dac.z;
            }
        //////////////////////////////////////////////////////////////////////////////

        // actually calculate the force
	Scalar dth = acos(c_abbc) - m_Tau[angle_type];
        Scalar tk = m_Xi[angle_type] * dth;

        Scalar a = -1.0 * tk * s_abbc;
        Scalar a11 = a * c_abbc / rsqab;
        Scalar a12 = -a / (rab * rcb);
        Scalar a22 = a * c_abbc / rsqcb;

        Scalar fab[3], fcb[3];

        fab[0] = a11 * dab.x + a12 * dcb.x;
        fab[1] = a11 * dab.y + a12 * dcb.y;
        fab[2] = a11 * dab.z + a12 * dcb.z;

        fcb[0] = a22 * dcb.x + a12 * dab.x;
        fcb[1] = a22 * dcb.y + a12 * dab.y;
        fcb[2] = a22 * dcb.z + a12 * dab.z;

        // compute 1/3 of the energy, 1/3 for each atom in the angle
        Scalar angle_eng = (tk * dth + eac) * Scalar(1.0/6.0);

        // compute 1/3 of the virial, 1/3 for each atom in the angle
        // upper triangular version of virial tensor
        Scalar angle_virial[6];
        angle_virial[0] = Scalar(1. /3.) * (dab.x * fab[0] + dcb.x * fcb[0]);
        angle_virial[1] = Scalar(1. /3.) * (dab.y * fab[0] + dcb.y * fcb[0]);
        angle_virial[2] = Scalar(1. /3.) * (dab.z * fab[0] + dcb.z * fcb[0]);
        angle_virial[3] = Scalar(1. /3.) * (dab.y * fab[1] + dcb.y * fcb[1]);
        angle_virial[4] = Scalar(1. /3.) * (dab.z * fab[1] + dcb.z * fcb[1]);
        angle_virial[5] = Scalar(1. /3.) * (dab.z * fab[2] + dcb.z * fcb[2]);
        Scalar virial[6];
        for (unsigned int k=0; k < 6; k++)
            virial[k] = angle_virial[k] + Scalar(1. /3.) * vac[k];

        // Now, apply the force to each individual atom a,b,c, and accumulate the energy/virial
        // only apply force to local particles
        if (idx_a < m_pdata->getN())
            {
            h_force.data[idx_a].x += fab[0] + fac * dac.x;
            h_force.data[idx_a].y += fab[1] + fac * dac.y;
            h_force.data[idx_a].z += fab[2] + fac * dac.z;
            h_force.data[idx_a].w += angle_eng;
            for (int k = 0; k < 6; k++)
                h_virial.data[k * virial_pitch + idx_a] += virial[k];
            }

        if (idx_b < m_pdata->getN())
            {
            h_force.data[idx_b].x -= fab[0] + fcb[0];
            h_force.data[idx_b].y -= fab[1] + fcb[1];
            h_force.data[idx_b].z -= fab[2] + fcb[2];
            h_force.data[idx_b].w += angle_eng;
            for (int k = 0; k < 6; k++)
                h_virial.data[k * virial_pitch + idx_b] += virial[k];
            }

        if (idx_c < m_pdata->getN())
            {
            h_force.data[idx_c].x += fcb[0] - fac*dac.x;
            h_force.data[idx_c].y += fcb[1] - fac*dac.y;
            h_force.data[idx_c].z += fcb[2] - fac*dac.z;
            h_force.data[idx_c].w += angle_eng;
            for (int k = 0; k < 6; k++)
                h_virial.data[k * virial_pitch + idx_c] += virial[k];
            }
        }
    if (m_prof)
        m_prof->pop();
    }

namespace detail
    {
void export_PCNDAngleForceCompute(pybind11::module& m)
    {
    pybind11::class_<PCNDAngleForceCompute,
	             ForceCompute,
	             std::shared_ptr<PCNDAngleForceCompute>>(m, "PCNDAngleForceCompute")
        .def(pybind11::init<std::shared_ptr<SystemDefinition>>())
        .def("setParams", &PCNDAngleForceCompute::setParamsPython)
	.def("getParams", &PCNDAngleForceCompute::getParams);
    }

    } // end namespace detail
    } // end namespace md
    } // end namespace hoomd
