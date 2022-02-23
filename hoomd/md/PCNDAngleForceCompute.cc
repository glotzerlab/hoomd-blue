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
    : ForceCompute(sysdef), m_Xi(NULL), m_Tau(NULL)
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

    assert(m_Xi);
    assert(m_Tau);

    memset((void*)m_Xi,0,sizeof(Scalar)*m_pcnd_angle_data->getNTypes());
    memset((void*)m_Tau,0,sizeof(Scalar)*m_pcnd_angle_data->getNTypes());
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
void PCNDAngleForceCompute::computeForces(uint64_t timestep)
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
    memset((void*)h_force.data,0,sizeof(Scalar4)*m_force.getNumElements());
    memset((void*)h_virial.data,0,sizeof(Scalar)*m_virial.getNumElements());

    // get a local copy of the simulation box too
    const BoxDim& box = m_pdata->getGlobalBox();
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
