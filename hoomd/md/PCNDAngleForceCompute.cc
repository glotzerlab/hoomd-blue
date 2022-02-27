// Copyright (c) 2009-2017 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "PCNDAngleForceCompute.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <math.h>
#include "hoomd/RNGIdentifiers.h"
#include "hoomd/RandomNumbers.h"

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
    : ForceCompute(sysdef), m_Xi(NULL), m_Tau(NULL)
    {
    m_exec_conf->msg->notice(5) << "Constructing PCNDAngleForceCompute" << endl;

    // access the angle data for later use
    m_pcnd_angle_data = m_sysdef->getAngleData();

    // check for some silly errors a user could make
    if (m_pcnd_angle_data->getNTypes() == 0)
        {
        throw runtime_error("No angle types in the system.");
        }

    uint64_t PCNDtimestep = 0;

    // allocate the parameters
    m_Xi = new Scalar[m_pcnd_angle_data->getNTypes()];
    m_Tau = new Scalar[m_pcnd_angle_data->getNTypes()];
    }

PCNDAngleForceCompute::~PCNDAngleForceCompute()
    {
    m_exec_conf->msg->notice(5) << "Destroying PCNDAngleForceCompute" << endl;

    delete[] m_Xi;
    delete[] m_Tau;
    m_Xi = NULL;
    m_Tau = NULL;
    }

/*! \param type Type of the angle to set parameters for
    \param Xi Root mean square magnitude of the PCND force
    \param Tau Correlation time

    Sets parameters for the potential of a particular angle type
*/
void PCNDAngleForceCompute::setParams(unsigned int type, Scalar Xi, Scalar Tau)
    {
    // make sure the type is valid
    if (type >= m_pcnd_angle_data->getNTypes())
        {
        throw runtime_error("Invalid angle type.");
        }

    m_Xi[type] = Xi;
    m_Tau[type] = Tau;

    // check for some silly errors a user could make
    if (Xi <= 0)
	m_exec_conf->msg->warning() << "angle.pcnd: specified Xi <= 0" << endl;
    if (Tau <= 0)
	m_exec_conf->msg->warning() << "angle.pcnd: specified Tau <= 0" << endl;
    }

void PCNDAngleForceCompute::setParamsPython(std::string type, pybind11::dict params)
    {                                                                           
    auto typ = m_pcnd_angle_data->getTypeByName(type);                               
    auto _params = angle_pcnd_params(params);                               
    setParams(typ, _params.Xi, _params.Tau);
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
    return params;                                                              
    }

/*! Actually perform the force computation
    \param timestep Current time step
 */
void PCNDAngleForceCompute::computeForces(uint64_t timestep, uint64_t PCNDtimestep)
    {
    if (m_prof)
	m_prof->push("PCNDAngle");

    assert(m_pdata);
    
    // access the particle data arrays
    ArrayHandle<Scalar4> h_pos(m_pdata->getPositions(), access_location::host, access_mode::read);
    ArrayHandle<unsigned int> h_rtag(m_pdata->getRTags(), access_location::host, access_mode::read);

    ArrayHandle<Scalar4> h_force(m_force,access_location::host, access_mode::overwrite);
    ArrayHandle<Scalar> h_virial(m_virial,access_location::host, access_mode::overwrite);
    size_t virial_pitch = m_virial.getPitch();

    // there are enough other checks on the input data: but it doesn't hurt to be safe
    assert(h_force.data);
    assert(h_virial.data);
    assert(h_pos.data);
    assert(h_rtag.data);

    // Zero data for force calculation.
    memset((void*)h_force.data,0,sizeof(Scalar4) * m_force.getNumElements());
    memset((void*)h_virial.data,0,sizeof(Scalar) * m_virial.getNumElements());

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getGlobalBox();
   
    //uint64_t PCNDtimestep = 0;
    //uint16_t seed = m_sysdef->getSeed();

    // for each of the angles
    const unsigned int size = (unsigned int)m_pcnd_angle_data->getN();
    for (unsigned int i = 0; i < size; i++)
        {
        // lookup the tag of each of the particles participating in the angle
	const AngleData::members_t& angle = m_pcnd_angle_data->getMembersByIndex(i);
	assert(angle.tag[0] <= m_pdata->getMaximumTag());
	assert(angle.tag[1] <= m_pdata->getMaximumTag());
	assert(angle.tag[2] <= m_pdata->getMaximumTag());
        
        // transform a, b, and c into indicies into the particle data arrays
	// MEM TRANSFER: 6 ints
	unsigned int idx_a = h_rtag.data[angle.tag[0]];	
	unsigned int idx_b = h_rtag.data[angle.tag[1]];	
	unsigned int idx_c = h_rtag.data[angle.tag[2]];

        // throw an error if this angle is incomplete
	if (idx_a == NOT_LOCAL || idx_b == NOT_LOCAL || idx_c == NOT_LOCAL)
	    {
	    this->m_exec_conf->msg->error()
	        << "angle.pcnd: angle " << angle.tag[0] << " " << angle.tag[1] << " "
		<< angle.tag[2] << " incomplete." << endl
		<< endl;
            throw std::runtime_error("Error in PCND calculation");
	    }

	assert(idx_a < m_pdata->getN() + m_pdata->getNGhosts());
	assert(idx_b < m_pdata->getN() + m_pdata->getNGhosts());
	assert(idx_c < m_pdata->getN() + m_pdata->getNGhosts());
        
	//unsigned int j = m_group->getMemberIndex(i);
	//unsigned int ptag = h_tag.data[j];
	uint16_t seed = 1;
	unsigned int angle_type = m_pcnd_angle_data->getTypeByIndex(i);

	// Initialize the Random Number Generator and generate the 6 random numbers
	RandomGenerator rng(hoomd::Seed(RNGIdentifier::PCNDAngleForceCompute, timestep, seed),
			    hoomd::Counter(0));
	UniformDistribution<Scalar> uniform(Scalar(0), Scalar(1));

	Scalar a_x = uniform(rng);
	Scalar b_x = uniform(rng);
	Scalar a_y = uniform(rng);
	Scalar b_y = uniform(rng);
	Scalar a_z = uniform(rng);
	Scalar b_z = uniform(rng);
        	
	if (idx_b < m_pdata->getN() && PCNDtimestep == 0)
	   {
	   h_force.data[idx_b].x = m_Xi[angle_type] * sqrt(-2 * log(a_x)) * cos(2 * 3.1415926535897 * b_x);
	   h_force.data[idx_b].y = m_Xi[angle_type] * sqrt(-2 * log(a_y)) * cos(2 * 3.1415926535897 * b_y);
	   h_force.data[idx_b].z = m_Xi[angle_type] * sqrt(-2 * log(a_z)) * cos(2 * 3.1415926535897 * b_z);
	
	   h_force.data[idx_b].w = sqrt(h_force.data[idx_b].x * h_force.data[idx_b].x + h_force.data[idx_b].y * h_force.data[idx_b].y + h_force.data[idx_b].z * h_force.data[idx_b].z);
	   }

	else if (idx_b < m_pdata->getN() && PCNDtimestep != 0)
           {
	   Scalar magx = h_force.data[idx_b].x;
	   Scalar magy = h_force.data[idx_b].y;
	   Scalar magz = h_force.data[idx_b].z;

	   Scalar E = exp(-1 / m_Tau[angle_type]);
	   Scalar hx = m_Xi[angle_type] * sqrt(-2 * (1 - E * E) * log(a_x)) * cos(2 * 3.1415926535897 * b_x);
	   Scalar hy = m_Xi[angle_type] * sqrt(-2 * (1 - E * E) * log(a_y)) * cos(2 * 3.1415926535897 * b_y);
	   Scalar hz = m_Xi[angle_type] * sqrt(-2 * (1 - E * E) * log(a_z)) * cos(2 * 3.1415926535897 * b_z);
	
	   if (hx > m_Xi[angle_type] * sqrt(-2 * log(0.001)))
	      {
	      hx = m_Xi[angle_type] * sqrt(-2 * log(0.001));
	      }
	   else if (hx <- m_Xi[angle_type] * sqrt(-2 * log(0.001)))
	      {		
	      hx = -m_Xi[angle_type] * sqrt(-2 * log(0.001));
	      }
	   if (hy > m_Xi[angle_type] * sqrt(-2 * log(0.001)))
	      {
	      hy = m_Xi[angle_type] * sqrt(-2 * log(0.001));		
	      }
	   else if (hy <- m_Xi[angle_type] * sqrt(-2 * log(0.001)))
	      {
	      hy = -m_Xi[angle_type] * sqrt(-2 * log(0.001));
	      }
	   if (hz > m_Xi[angle_type] * sqrt(-2 * log(0.001)))
	      {
	      hz = m_Xi[angle_type] * sqrt(-2 * log(0.001));
	      }
	   else if (hz <- m_Xi[angle_type] * sqrt(-2 * log(0.001)))
	      {
	      hz= -m_Xi[angle_type] * sqrt(-2 * log(0.001));
	      }

	   h_force.data[idx_b].x = E * magx + hx;
	   h_force.data[idx_b].y = E * magy + hy;
	   h_force.data[idx_b].z = E * magz + hz;
	   h_force.data[idx_b].w = sqrt(h_force.data[idx_b].x * h_force.data[idx_b].x + h_force.data[idx_b].y * h_force.data[idx_b].y + h_force.data[idx_b].z * h_force.data[idx_b].z);
	   }
        }
    if (m_prof)
        m_prof->pop();
    
    PCNDtimestep = PCNDtimestep + 1;
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
